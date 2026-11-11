#include "spotter/structure.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <opencv2/flann.hpp>

#include "spotter/features_orb.hpp"
#include "spotter/stats.hpp"

namespace spotter
{

const char* SurfaceName(Surface surface)
{
    switch (surface)
    {
    case Surface::Floor: return "floor";
    case Surface::Ceiling: return "ceiling";
    case Surface::Wall: return "wall";
    default: return "other";
    }
}

static const float Radians = (float)CV_PI / 180;

//cv::Point3f Unit(const cv::Point3f& v){
//
//@param1 - v: a vector
//
//@brief - the vector scaled to length 1 (a zero vector stays near zero instead of dividing by 0). O(1).
static cv::Point3f Unit(const cv::Point3f& v) { return v * (1.0f / std::max((float)cv::norm(v), 1e-9f)); }

//Surface Classify(const StructurePlane& p, const cv::Point3f& up, const StructureOptions& options){
//
//@param1 - p: a plane with its points
//@param2 - up: the room's up, unit
//@param3 - options: levelDegrees and minWallHeightMeters
//
//@brief - floor if it faces up (within levelDegrees), ceiling if down, other if it's slanted; an upright plane is a wall only
//         if its points span minWallHeightMeters of height (2nd to 98th percentile), else other (a cupboard side). O(k log k).
static Surface Classify(const StructurePlane& p, const cv::Point3f& up, const StructureOptions& options)
{
    float upness = p.plane.normal.dot(up);
    if (upness > std::cos(options.levelDegrees * Radians)) return Surface::Floor;   // faces up, towards a camera above it
    if (upness < -std::cos(options.levelDegrees * Radians)) return Surface::Ceiling;
    if (std::abs(upness) > std::sin(options.levelDegrees * Radians)) return Surface::Other;
    std::vector<float> heights;
    for (const cv::Point3f& q : p.points) heights.push_back(q.dot(up));
    return Percentile(heights, 0.98) - Percentile(heights, 0.02) >= options.minWallHeightMeters ? Surface::Wall : Surface::Other;
}

ScanStructure ExtractStructure(const std::vector<cv::Point3f>& cloud, const PinholeCamera& camera, const cv::Point3f& up,
                               const StructureOptions& options, std::mt19937& random)
{
    ScanStructure s;
    PointGrid grid = ThinCloud(cloud, camera.width, options.step);
    for (const cv::Point3f& p : grid.points) s.readings += !std::isnan(p.z);
    for (const PlanePiece& piece : FindPlanes(grid, options.planes, random))
    {
        StructurePlane p;
        p.plane = piece.plane;
        p.cells = (int)piece.cells.size();
        for (int cell : piece.cells)
        {
            const cv::Point3f& q = grid.points[cell];
            p.points.push_back(q);
            // The cell's patch: step x step pixels at its distance, stretched by how slanted the plane is to the view.
            float facing = std::max(std::abs(p.plane.normal.dot(Unit(q))), 0.2f);
            p.areaMeters2 += (options.step * q.z / camera.fx) * (options.step * q.z / camera.fy) / facing;
        }
        p.pointsPerMeter2 = p.areaMeters2 > 0 ? p.cells * options.step * options.step / p.areaMeters2 : 0;
        s.planes.push_back(std::move(p));
    }
    if (cv::norm(up) > 0.5) { s.up = Unit(up); s.upFrom = "given"; }
    else
    {
        s.upFrom = "camera";
        int biggest = 0;
        for (const StructurePlane& p : s.planes)
        {
            float upness = p.plane.normal.dot(cv::Point3f(0, -1, 0));
            if (std::abs(upness) < std::cos(options.maxTiltDegrees * Radians) || p.cells <= biggest) continue;
            biggest = p.cells;
            s.up = upness > 0 ? p.plane.normal : -p.plane.normal;
            s.upFrom = upness > 0 ? "floor" : "ceiling";
        }
        // No floor or ceiling: up from the walls (assumed vertical). Two walls at a clear angle give it
        // exactly (their cross product); one wall gives the camera's up laid into the wall (assumes no roll).
        const cv::Point3f cameraUp(0, -1, 0);
        std::vector<const StructurePlane*> upright;
        for (const StructurePlane& p : s.planes)
            if (std::abs(p.plane.normal.dot(cameraUp)) < std::sin(options.maxTiltDegrees * Radians)) upright.push_back(&p);
        for (size_t i = 0; i < upright.size() && s.upFrom == "camera"; i++)
            for (size_t j = i + 1; j < upright.size() && s.upFrom == "camera"; j++)
            {
                cv::Point3f across = upright[i]->plane.normal.cross(upright[j]->plane.normal);
                if (cv::norm(across) < std::sin(60 * Radians)) continue;
                s.up = Unit(across.dot(cameraUp) > 0 ? across : -across);
                s.upFrom = "two walls";
            }
        if (s.upFrom == "camera" && !upright.empty())
        {
            const cv::Point3f& n = upright[0]->plane.normal;   // the biggest (FindPlanes finds the biggest first)
            s.up = Unit(cameraUp - n * n.dot(cameraUp));
            s.upFrom = "one wall";
        }
    }
    for (StructurePlane& p : s.planes) p.surface = Classify(p, s.up, options);
    return s;
}

ScanStructure MoveStructure(const ScanStructure& structure, const Pose& pose)
{
    ScanStructure moved = structure;
    moved.up = pose.turn * structure.up;
    for (StructurePlane& p : moved.planes)
    {
        p.plane.normal = pose.turn * p.plane.normal;
        p.plane.distance -= p.plane.normal.dot(pose.shift);
        for (cv::Point3f& q : p.points) q = pose(q);
    }
    return moved;
}

ScanStructure MergeStructures(const std::vector<ScanStructure>& structures, float mergeDegrees, float mergeMeters)
{
    ScanStructure merged;
    if (structures.empty()) return merged;
    merged.up = structures[0].up;
    merged.upFrom = structures[0].upFrom;
    for (const ScanStructure& s : structures)
    {
        merged.readings += s.readings;
        for (const StructurePlane& p : s.planes)
        {
            cv::Point3f centre(0, 0, 0);
            for (const cv::Point3f& q : p.points) centre += q;
            centre *= 1.0f / std::max<size_t>(p.points.size(), 1);
            bool joined = false;
            for (StructurePlane& m : merged.planes)
            {
                if (m.surface != p.surface || m.plane.normal.dot(p.plane.normal) < std::cos(mergeDegrees * Radians) ||
                    std::abs(m.plane.DistanceTo(centre)) > mergeMeters) continue;
                m.points.insert(m.points.end(), p.points.begin(), p.points.end());
                m.cells += p.cells;
                m.areaMeters2 += p.areaMeters2;   // ponytail: overlapping views count twice; it's "area seen", summed
                cv::Point3f keep = m.plane.normal;
                m.plane = FitPlane(m.points);
                if (m.plane.normal.dot(keep) < 0) { m.plane.normal = -m.plane.normal; m.plane.distance = -m.plane.distance; }
                joined = true;
                break;
            }
            if (!joined) merged.planes.push_back(p);
        }
    }
    return merged;
}

std::string Edge::Kind(const ScanStructure& s) const
{
    std::string a = SurfaceName(s.planes[planes[0]].surface), b = SurfaceName(s.planes[planes[1]].surface);
    if (b < a) std::swap(a, b);
    return a + "-" + b;
}

std::vector<Edge> FindEdges(const ScanStructure& s, float minDegrees, float nearMeters)
{
    std::vector<Edge> edges;
    for (size_t i = 0; i < s.planes.size(); i++)
        for (size_t j = i + 1; j < s.planes.size(); j++)
        {
            const Plane &a = s.planes[i].plane, &b = s.planes[j].plane;
            if (std::abs(a.normal.dot(b.normal)) > std::cos(minDegrees * Radians)) continue;
            Edge e;
            e.planes[0] = (int)i;
            e.planes[1] = (int)j;
            e.direction = Unit(a.normal.cross(b.normal));
            cv::Matx33f A(a.normal.x, a.normal.y, a.normal.z, b.normal.x, b.normal.y, b.normal.z, e.direction.x, e.direction.y, e.direction.z);
            cv::Vec3f p = A.inv() * cv::Vec3f(-a.distance, -b.distance, 0);
            e.point = cv::Point3f(p[0], p[1], p[2]);
            std::vector<float> along;
            for (size_t k : { i, j })
                for (const cv::Point3f& q : s.planes[k].points)
                {
                    cv::Point3f v = q - e.point;
                    float t = v.dot(e.direction);
                    if (cv::norm(v - e.direction * t) < nearMeters) along.push_back(t);
                }
            e.support = (int)along.size();
            if (e.support < 10) continue;   // the planes never come near each other in this view
            e.ends[0] = e.point + e.direction * Percentile(along, 0.02);
            e.ends[1] = e.point + e.direction * Percentile(along, 0.98);
            edges.push_back(e);
        }
    return edges;
}

void EdgeOffset(const Edge& a, const Edge& b, float& degrees, float& meters)
{
    degrees = std::acos(std::min(1.0f, std::abs(a.direction.dot(b.direction)))) / Radians;
    cv::Point3f middle = (a.ends[0] + a.ends[1]) * 0.5f, v = middle - b.point;
    meters = (float)cv::norm(v - b.direction * v.dot(b.direction));
}

RoomMap MakeRoomMap(const std::vector<const Scan*>& scans, const PinholeCamera& camera, const StructureOptions& options, std::mt19937& random,
                    const cv::Point3f& roomUp)
{
    RoomMap map;
    std::vector<ScanStructure> structures;
    std::unordered_set<long long> taken;
    for (const Scan* scan : scans)
    {
        structures.push_back(MoveStructure(ExtractStructure(scan->cloud, camera, scan->pose.turn.t() * roomUp, options, random), scan->pose));
        for (size_t i = 0; i < scan->surface.points.size(); i++)
        {
            cv::Point3f p = scan->pose(scan->surface.points[i]);
            if (!taken.insert(CubeKey(p, 0.01f)).second) continue;
            map.surface.points.push_back(p);
            map.surface.normals.push_back(scan->pose.turn * scan->surface.normals[i]);
        }
    }
    map.structure = MergeStructures(structures);
    map.structure.up = Unit(roomUp);
    return map;
}

//int Biggest(const ScanStructure& s, Surface surface){
//
//@param1 - s: a structure
//@param2 - surface: the kind to look for
//
//@brief - the index of the plane of that kind with the most cells, or -1 if there's none. O(planes).
static int Biggest(const ScanStructure& s, Surface surface)
{
    int best = -1;
    for (int i = 0; i < (int)s.planes.size(); i++)
        if (s.planes[i].surface == surface && (best < 0 || s.planes[i].cells > s.planes[best].cells)) best = i;
    return best;
}

//cv::Matx33f YawOnto(const cv::Point3f& a, const cv::Point3f& b, const cv::Point3f& up){
//
//@param1 - a: a direction
//@param2 - b: the direction to turn it onto
//@param3 - up: the axis to turn about, unit
//
//@brief - the turn about `up` that takes the horizontal part of a onto the horizontal part of b (their signed angle seen
//         from above). O(1).
static cv::Matx33f YawOnto(const cv::Point3f& a, const cv::Point3f& b, const cv::Point3f& up)
{
    cv::Point3f ha = Unit(a - up * up.dot(a)), hb = Unit(b - up * up.dot(b));
    float angle = std::atan2(up.dot(ha.cross(hb)), ha.dot(hb));
    return TurnAbout(cv::Vec3d(up.x, up.y, up.z) * angle);
}

//bool Distinct(const Pose& a, const Pose& b, float meters, float degrees){
//
//@param1 - a: one pose
//@param2 - b: another pose
//@param3 - meters: how far apart their shifts must be...
//@param4 - degrees: ...or how far apart their turns
//
//@brief - true if the poses differ by more than meters or more than degrees, i.e. they are two different fits and not one. O(1).
static bool Distinct(const Pose& a, const Pose& b, float meters, float degrees)
{
    return cv::norm(a.shift - b.shift) > meters || Pose{ a.turn.t() * b.turn, cv::Point3f() }.Degrees() > degrees;
}

float ViewAgreement(const Scan& scan, const Pose& pose, const RoomMap& map, const PinholeCamera& camera)
{
    const int step = 4, columns = camera.width / step, rows = camera.height / step;
    std::vector<float> nearest(columns * rows, 1e9f);
    const cv::Matx33f back = pose.turn.t();
    for (const cv::Point3f& p : map.surface.points)
    {
        cv::Point3f c = back * (p - pose.shift);
        if (c.z < 0.3f) continue;
        int column = (int)std::lround((c.x * camera.fx / c.z + camera.cx - 0.5f) / step), row = (int)std::lround((c.y * camera.fy / c.z + camera.cy - 0.5f) / step);
        if (column < 0 || row < 0 || column >= columns || row >= rows) continue;
        nearest[row * columns + column] = std::min(nearest[row * columns + column], c.z);
    }
    int agree = 0, seeThrough = 0, both = 0;
    for (int row = 0; row < rows; row++)
        for (int column = 0; column < columns; column++)
        {
            float mapZ = nearest[row * columns + column], scanZ = scan.cloud[(row * step) * camera.width + column * step].z;
            if (mapZ > 1e8f || std::isnan(scanZ)) continue;
            both++;
            if (std::abs(scanZ - mapZ) < 0.05f) agree++;
            else if (scanZ > mapZ + 0.10f) seeThrough++;
        }
    return both > 0 ? (float)(agree - seeThrough) / both : -1;
}

RelocateResult Relocate(const Scan& scan, const ScanStructure& s, const RoomMap& map, const PinholeCamera& camera, const RelocateOptions& options)
{
    RelocateResult result;
    const ScanStructure& m = map.structure;
    const cv::Point3f up = m.up;
    // 1. Up: the scan's up onto the room's (2 of the 3 turns).
    cv::Point3f axis = s.up.cross(up);
    float angle = std::atan2((float)cv::norm(axis), s.up.dot(up));
    const cv::Matx33f level = cv::norm(axis) > 1e-6 ? TurnAbout(cv::Vec3d(axis.x, axis.y, axis.z) * (angle / cv::norm(axis))) : cv::Matx33f::eye();
    // 2. Height: the camera's height above the floor (or below the ceiling) is the same in the scan and the room.
    float height = 0;
    bool heightKnown = false;
    for (Surface level2 : { Surface::Floor, Surface::Ceiling })
    {
        int a = Biggest(s, level2), b = Biggest(m, level2);
        if (heightKnown || a < 0 || b < 0) continue;
        height = (s.planes[a].plane.distance - m.planes[b].plane.distance) / m.planes[b].plane.normal.dot(up);   // up . c
        heightKnown = true;
    }
    if (!heightKnown) result.note = "no floor or ceiling in both the scan and the map: height left to ICP; ";
    // 3. Walls: each scan wall matched to each map wall gives the yaw and one horizontal line;
    //    a second, non-parallel pair pins the position.
    std::vector<int> sw, mw;
    for (int i = 0; i < (int)s.planes.size(); i++) if (s.planes[i].surface == Surface::Wall) sw.push_back(i);
    for (int i = 0; i < (int)m.planes.size(); i++) if (m.planes[i].surface == Surface::Wall) mw.push_back(i);
    if (sw.empty() || mw.empty()) { result.note += "no wall in the scan or the map"; return result; }
    std::vector<RelocateCandidate> candidates;
    auto Solve = [&](const cv::Point3f& n1, float k1, const cv::Point3f& n2, float k2, cv::Point3f& c)
    {
        cv::Matx33f A(n1.x, n1.y, n1.z, n2.x, n2.y, n2.z, up.x, up.y, up.z);
        if (std::abs(cv::determinant(A)) < 1e-3) return false;
        cv::Vec3f x = A.inv() * cv::Vec3f(k1, k2, height);
        c = cv::Point3f(x[0], x[1], x[2]);
        return true;
    };
    bool anyPair = false;
    for (int a : sw)
        for (int A : mw)
        {
            const cv::Matx33f turn = YawOnto(level * s.planes[a].plane.normal, m.planes[A].plane.normal, up) * level;
            const float kA = s.planes[a].plane.distance - m.planes[A].plane.distance;   // n_A . c
            for (int b : sw)
            {
                if (b == a || std::abs(s.planes[a].plane.normal.dot(s.planes[b].plane.normal)) > std::cos(30 * Radians)) continue;
                for (int B : mw)
                {
                    if (B == A || (turn * s.planes[b].plane.normal).dot(m.planes[B].plane.normal) < std::cos(options.wallMatchDegrees * Radians)) continue;
                    RelocateCandidate c;
                    c.pose.turn = turn;
                    if (!Solve(m.planes[A].plane.normal, kA, m.planes[B].plane.normal, s.planes[b].plane.distance - m.planes[B].plane.distance, c.pose.shift)) continue;
                    candidates.push_back(c);
                    anyPair = true;
                }
            }
        }
    if (!anyPair)   // one wall direction only: the position along it is free; slide along the map wall
        for (int a : sw)
            for (int A : mw)
            {
                const cv::Point3f nA = m.planes[A].plane.normal, along = Unit(up.cross(nA));
                const cv::Matx33f turn = YawOnto(level * s.planes[a].plane.normal, nA, up) * level;
                std::vector<float> ts;
                for (const cv::Point3f& q : m.planes[A].points) ts.push_back(q.dot(along));
                for (float t = Percentile(ts, 0.02); t <= Percentile(ts, 0.98); t += options.alongStepMeters)
                {
                    RelocateCandidate c;
                    c.pose.turn = turn;
                    c.underConstrained = true;
                    if (Solve(nA, s.planes[a].plane.distance - m.planes[A].plane.distance, along, t, c.pose.shift)) candidates.push_back(c);
                }
            }
    result.generated = (int)candidates.size();
    if (!anyPair) result.note += "one wall direction only: slid along it (under-constrained)";
    if (candidates.empty()) return result;

    // 4. A quick score (every 8th scan point within 5 cm of the map), then ICP on the best distinct few.
    cv::Mat data((int)map.surface.points.size(), 3, CV_32F, (void*)map.surface.points.data());
    cv::flann::Index tree(data, cv::flann::KDTreeIndexParams(4));
    std::vector<cv::Point3f> few;
    for (size_t i = 0; i < scan.surface.points.size(); i += 8) few.push_back(scan.surface.points[i]);
    for (RelocateCandidate& c : candidates)
    {
        cv::Mat query((int)few.size(), 3, CV_32F), index, squared;
        for (int i = 0; i < query.rows; i++) { cv::Point3f p = c.pose(few[i]); query.at<float>(i, 0) = p.x; query.at<float>(i, 1) = p.y; query.at<float>(i, 2) = p.z; }
        tree.knnSearch(query, index, squared, 1, cv::flann::SearchParams(16));
        int near = 0;
        for (int i = 0; i < query.rows; i++) near += squared.at<float>(i) < 0.05f * 0.05f;
        c.share = (float)near / std::max(query.rows, 1);
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.share > b.share; });
    // ICP on every 4th scan point against the map thinned to 2 cm: plenty to polish a pose, and fast.
    std::vector<cv::Point3f> scanPoints, mapPoints, mapNormals;
    for (size_t i = 0; i < scan.surface.points.size(); i += 4) scanPoints.push_back(scan.surface.points[i]);
    std::unordered_set<long long> taken;
    for (size_t i = 0; i < map.surface.points.size(); i++)
        if (taken.insert(CubeKey(map.surface.points[i], 0.02f)).second) { mapPoints.push_back(map.surface.points[i]); mapNormals.push_back(map.surface.normals[i]); }
    std::vector<RelocateCandidate> refined;
    for (const RelocateCandidate& c : candidates)
    {
        if (c.share < 0.7f * candidates[0].share || (int)refined.size() >= options.refineAtMost) break;
        bool seen = false;
        for (const RelocateCandidate& r : refined) seen = seen || !Distinct(c.pose, r.pose, 0.1f, 3);
        if (seen) continue;
        IcpResult fit = Icp(scanPoints, mapPoints, mapNormals, c.pose, options.refine);
        RelocateCandidate r = c;
        r.pose = fit.pose;
        r.share = fit.pairedShare;
        r.rmsMeters = fit.rmsMeters;
        r.agreement = ViewAgreement(scan, r.pose, map, camera);
        refined.push_back(r);
    }
    std::sort(refined.begin(), refined.end(), [](const auto& a, const auto& b) { return a.agreement > b.agreement; });
    for (const RelocateCandidate& r : refined)   // distinct fits only, best first
    {
        bool seen = false;
        for (const RelocateCandidate& k : result.candidates) seen = seen || !Distinct(r.pose, k.pose, 0.1f, 3);
        if (!seen) result.candidates.push_back(r);
    }
    for (size_t i = 1; i < result.candidates.size(); i++)
        if (Distinct(result.candidates[0].pose, result.candidates[i].pose, options.distinctMeters, options.distinctDegrees))
        {
            result.ambiguous = result.candidates[0].agreement - result.candidates[i].agreement < options.ambiguousScore;
            break;
        }
    // Ambiguous: keep every rival (ties are in no useful order, and the truth must reach the tie-breaker).
    if (!result.ambiguous && result.candidates.size() > 4) result.candidates.resize(4);
    return result;
}

//cv::Point3f CloudAt(const Scan& scan, const cv::Point2f& pixel){
//
//@param1 - scan: a scan with its cloud and picture
//@param2 - pixel: a position on the depth grid, e.g. a keypoint
//
//@brief - the scan's 3D point at that pixel (rounded, kept inside the picture); NaN where depth read nothing. O(1).
static cv::Point3f CloudAt(const Scan& scan, const cv::Point2f& pixel)
{
    int column = std::clamp((int)std::lround(pixel.x), 0, scan.picture.cols - 1), row = std::clamp((int)std::lround(pixel.y), 0, scan.picture.rows - 1);
    return scan.cloud[row * scan.picture.cols + column];
}

int ColourVotes(const Scan& scan, const Pose& pose, const std::vector<const Scan*>& mapScans, float ratio)
{
    int votes = 0;
    for (const Scan* m : mapScans)
        for (auto [i, j] : MatchFeatures(scan.near.descriptors, m->near.descriptors, ratio))
        {
            cv::Point3f p = CloudAt(scan, scan.near.keypoints[i].pt), q = CloudAt(*m, m->near.keypoints[j].pt);
            if (!std::isnan(p.z) && !std::isnan(q.z) && cv::norm(pose(p) - m->pose(q)) < 0.10) votes++;
        }
    return votes;
}

float OffPlaneShare(const Scan& scan, const ScanStructure& structure, const Pose& pose, const RoomMap& map)
{
    std::vector<cv::Point3f> off;
    for (const cv::Point3f& p : scan.surface.points)
    {
        bool onStructure = false;
        for (const StructurePlane& sp : structure.planes)
            onStructure = onStructure || (sp.surface != Surface::Other && std::abs(sp.plane.DistanceTo(p)) < 0.05f);
        if (!onStructure) off.push_back(pose(p));
    }
    if (off.empty() || map.surface.points.empty()) return 0;
    cv::Mat data((int)map.surface.points.size(), 3, CV_32F, (void*)map.surface.points.data());
    cv::flann::Index tree(data, cv::flann::KDTreeIndexParams(4));
    cv::Mat query((int)off.size(), 3, CV_32F, off.data()), index, squared;
    tree.knnSearch(query, index, squared, 1, cv::flann::SearchParams(16));
    int near = 0;
    for (int i = 0; i < query.rows; i++) near += squared.at<float>(i) < 0.03f * 0.03f;
    return (float)near / query.rows;
}

IcpResult PlaceAfter(const Scan& scan, const Scan& previous, const Pose& guess, const IcpOptions& options)
{
    std::vector<cv::Point3f> points, normals;
    for (size_t i = 0; i < previous.surface.points.size(); i++)
    {
        points.push_back(previous.pose(previous.surface.points[i]));
        normals.push_back(previous.pose.turn * previous.surface.normals[i]);
    }
    return Icp(scan.surface.points, points, normals, guess, options);
}

}
