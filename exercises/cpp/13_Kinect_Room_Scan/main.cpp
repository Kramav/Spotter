// =====================================================================
//  13 · Kinect Room Scan
//
//  The whole room, from as many spots as you like, in one model. Each
//  example before this did one part; this one puts them together:
//
//    1. SPACE      scan here: average 30 frames (07), a point per pixel
//                  with colour (08), ORB features (11b).
//    2. Place it   the first scan is the room's origin. Every later scan
//                  is matched by features against every earlier one (11b)
//                  for a rough pose, then polished by ICP against ALL
//                  earlier scans at once (11, measured "point to plane":
//                  along each surface's normal, see Icp below).
//    3. Repeat     carry the Kinect to the next spot, scan again. The
//                  viewer shows the room growing.
//    4. E          merge and save. Every scan goes into one TSDF grid
//                  (new): for each little cube of space, "how far in front
//                  of a surface is it", averaged over every scan that saw
//                  it. A surface net (new) turns the places where that
//                  crosses zero into one clean mesh: no double walls, no
//                  overlaps. Then 09c's clean model (walls, floor,
//                  ceiling, furniture boxes), from all scans together.
//
//  Window 1 (viewer): every scan placed in the room. C colours each scan
//            differently, to check they line up. Circles = where the
//            camera stood, numbered.
//  Window 2 (camera): what the Kinect sees now, with what to do next.
//  Keys:     SPACE scan here    U undo the last scan    E export (merge + save)
//            C colour by scan   mouse drag / W A S D turn   wheel / Z X zoom
//            R reset view       Q or Esc quit
//
//  How to scan: see "Scan a room, step by step" in the main README.
//  In short: set the Kinect down still for each scan; between scans move
//  it about 0.5-1 m or turn it about 30 deg, so each new view shares at
//  least half of what an earlier one saw.
//
//  Privacy: the scans stay in memory and are gone when you quit. Only
//  E writes a file: room_scan_<time>.obj, a 3D capture of the room in
//  colour. *.obj is gitignored: never commit one. Scan with nobody in view.
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/registration.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// --- Scanning (07, 07b) ---
const int AverageFrames = 30;
const float MaxDepthMeters = 6.0f;

// --- Placing each scan: features (11b) and ICP (11) ---
const int FeatureCount = 1000;
const float MatchRatio = 0.8f;
const int RansacTries = 1000;
const float FeatureInlierMeters = 0.05f;
const int MinFeatureMatches = 15;
const int IcpRounds = 30;

// A scan needs at least this many features to be matched later. In the dark the colour picture
// is black: 0 features (depth still works, it brings its own infrared light).
const int MinScanFeatures = 100;
const float IcpStartPairMeters = 0.05f;   // 11 started at 0.20, but the features are already within a few cm
const float IcpFinalPairMeters = 0.02f;
const int IcpStep = 4;

// A placed scan must overlap the earlier ones: at least this share of its points need a partner
// within IcpFinalPairMeters after ICP, or it's refused (most likely placed wrong).
const float MinOverlapShare = 0.25f;

// --- Merging (new) ---
// The TSDF grid's cube size. 3 cm: walls look flat and a room needs a few million cubes (about
// 20 bytes each). Smaller = finer and slower; it grows by itself if the room would need more
// than MaxVoxels.
const float VoxelMeters = 0.03f;
const int MaxVoxels = 30000000;

// How far in front of and behind a surface a scan updates the grid, in cubes. Behind that, the
// scan can't know what's there (it might be the far side of a wall), so it leaves it alone.
const int TruncationVoxels = 3;

// --- Clean model (09c) ---
const int MaxPlanes = 8;
const float InlierThicknessMeters = 0.03f;
const int SampleStep = 4;
const int NeighbourPixels = 40;
const double MinPlaneShare = 0.03;
const float MaxTiltDegrees = 25;
const float MinWallHeightMeters = 0.8f;
const float MinCeilingHeightMeters = 1.8f;
const float CornerSnapMeters = 0.5f;
const int MinBlockCells = 30;

// New: the same plane seen from two spots is merged into one when it's this close in angle and
// distance. Boxes from different scans that touch or overlap become one box.
const float MergeDegrees = 10;
const float MergeMeters = 0.05f;

// --- Viewer (08) ---
const int ViewWidth = 960, ViewHeight = 720;
const float ViewFocalPixels = 700;
const float PivotMeters = 2.5f;
const int PointSizePixels = 2;
const float TurnStepDegrees = 5;
const float ZoomStepMeters = 0.25f;
const int ShowStep = 2;   // the viewer draws every 2nd pixel of each scan, to stay quick

const int Width = 512, Height = 424, ColorWidth = 1920, ColorHeight = 1080;

// ---------------------------------------------------------------------
//  The viewer (copied from 08)
// ---------------------------------------------------------------------

float yawDegrees = 0, pitchDegrees = 0;
float viewDistance = PivotMeters;
bool dragging = false;
int lastMouseX = 0, lastMouseY = 0;

void OnMouse(int event, int x, int y, int flags, void*)
{
    if (event == cv::EVENT_LBUTTONDOWN) { dragging = true; lastMouseX = x; lastMouseY = y; }
    if (event == cv::EVENT_LBUTTONUP) dragging = false;
    if (event == cv::EVENT_MOUSEMOVE && dragging)
    {
        yawDegrees += (x - lastMouseX) * 0.3f;
        pitchDegrees -= (y - lastMouseY) * 0.3f;
        lastMouseX = x;
        lastMouseY = y;
    }
    if (event == cv::EVENT_MOUSEWHEEL)
        viewDistance -= cv::getMouseWheelDelta(flags) / 120.0f * ZoomStepMeters;
}

bool Project(const cv::Point3f& point, cv::Point& pixel, float& depth)
{
    const float yaw = yawDegrees * (float)CV_PI / 180, pitch = pitchDegrees * (float)CV_PI / 180;
    float x = point.x, y = point.y, z = point.z - PivotMeters;
    float x1 = std::cos(yaw) * x + std::sin(yaw) * z;
    float z1 = -std::sin(yaw) * x + std::cos(yaw) * z;
    float y2 = std::cos(pitch) * y - std::sin(pitch) * z1;
    float z2 = std::sin(pitch) * y + std::cos(pitch) * z1;
    depth = z2 + viewDistance;
    if (depth < 0.05f) return false;
    pixel = cv::Point((int)(ViewFocalPixels * x1 / depth + ViewWidth / 2),
                      (int)(ViewFocalPixels * y2 / depth + ViewHeight / 2));
    return pixel.x >= 0 && pixel.y >= 0 && pixel.x < ViewWidth && pixel.y < ViewHeight;
}

cv::Mat Render(const std::vector<cv::Point3f>& points, const std::vector<cv::Vec3b>& colours)
{
    cv::Mat image(ViewHeight, ViewWidth, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::Mat nearest(ViewHeight, ViewWidth, CV_32F, cv::Scalar(1e9));
    for (size_t i = 0; i < points.size(); i++)
    {
        cv::Point pixel;
        float depth;
        if (!Project(points[i], pixel, depth)) continue;
        for (int dy = 0; dy < PointSizePixels; dy++)
        {
            for (int dx = 0; dx < PointSizePixels; dx++)
            {
                int u = pixel.x + dx, v = pixel.y + dy;
                if (u >= ViewWidth || v >= ViewHeight || depth >= nearest.at<float>(v, u)) continue;
                nearest.at<float>(v, u) = depth;
                image.at<cv::Vec3b>(v, u) = colours[i];
            }
        }
    }
    return image;
}

cv::Vec3b ColorForId(int id)   // as in 04
{
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar((id * 47) % 180, 255, 255)), bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr.at<cv::Vec3b>(0, 0);
}

// ---------------------------------------------------------------------
//  Poses and ICP (as in 11)
// ---------------------------------------------------------------------

struct Pose
{
    cv::Matx33f turn = cv::Matx33f::eye();
    cv::Point3f shift = cv::Point3f(0, 0, 0);
    cv::Point3f operator()(const cv::Point3f& p) const { return turn * p + shift; }
    Pose Then(const Pose& next) const { return { next.turn * turn, next.turn * shift + next.shift }; }
    float Degrees() const
    {
        float cosine = (turn(0, 0) + turn(1, 1) + turn(2, 2) - 1) / 2;
        return std::acos(std::clamp(cosine, -1.0f, 1.0f)) * 180 / (float)CV_PI;
    }
};

Pose BestFit(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to)
{
    cv::Point3f fromCentre(0, 0, 0), toCentre(0, 0, 0);
    for (size_t i = 0; i < from.size(); i++)
    {
        fromCentre += from[i];
        toCentre += to[i];
    }
    fromCentre *= 1.0f / from.size();
    toCentre *= 1.0f / to.size();
    cv::Matx33d together = cv::Matx33d::zeros();
    for (size_t i = 0; i < from.size(); i++)
    {
        cv::Point3f p = from[i] - fromCentre, q = to[i] - toCentre;
        const double a[3] = { p.x, p.y, p.z }, b[3] = { q.x, q.y, q.z };
        for (int row = 0; row < 3; row++)
            for (int column = 0; column < 3; column++) together(row, column) += a[row] * b[column];
    }
    cv::Mat s, u, vt;
    cv::SVD::compute(cv::Mat(together), s, u, vt);
    cv::Mat turn = vt.t() * u.t();
    if (cv::determinant(turn) < 0)
    {
        vt.row(2) *= -1;
        turn = vt.t() * u.t();
    }
    Pose pose;
    pose.turn = cv::Matx33f(cv::Matx33d(turn));
    pose.shift = toCentre - pose.turn * fromCentre;
    return pose;
}

// A turn of |w| radians about the axis w points along (Rodrigues' formula).
cv::Matx33f TurnAbout(const cv::Vec3d& w)
{
    double angle = cv::norm(w);
    if (angle < 1e-12) return cv::Matx33f::eye();
    cv::Vec3d k = w / angle;
    double c = std::cos(angle), s = std::sin(angle);
    cv::Matx33d turn;
    for (int row = 0; row < 3; row++)
        for (int column = 0; column < 3; column++) turn(row, column) = (row == column ? c : 0) + (1 - c) * k[row] * k[column];
    turn(0, 1) -= s * k[2]; turn(0, 2) += s * k[1];
    turn(1, 0) += s * k[2]; turn(1, 2) -= s * k[0];
    turn(2, 0) -= s * k[1]; turn(2, 1) += s * k[0];
    return cv::Matx33f(turn);
}

// 11's ICP, changed in one way: each pair's gap is measured along the target's surface normal
// ("point to plane"), not straight from point to point. Why: the target points are a sparse
// sample (every 4th pixel: about 3 cm apart at 3 m), so even a perfect fit leaves each point up to
// half a sample from its nearest neighbour, and 11's point-to-point fit slid 3.4 cm along a wall
// in the synthetic test. Along the normal, a point anywhere on the right wall counts as on it.
// The fit is then no longer Kabsch: for small turns it's a 6x6 set of linear equations.
// One more change: a view of only flat walls can't say where along them it is (11's experiment
// 3). ICP then slides on noise: 21 cm in the synthetic test, where the features had been right to
// 1 mm. So the equations are split into directions (eigenvectors), and any direction the shape
// barely pins is left alone: the starting pose (the features' guess) stands there.
// Also says how good the final fit is: the share of "from" points paired, their gap (rms), and
// how many of the 6 directions (3 turns, 3 shifts) the shape couldn't pin.
Pose Icp(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to, const std::vector<cv::Point3f>& toNormals,
         Pose pose, float& pairedShare, float& rmsMeters, int& weakDirections)
{
    pairedShare = 0;
    rmsMeters = 0;
    weakDirections = 0;
    if (from.size() < 10 || to.size() < 10) return pose;
    cv::Mat toData((int)to.size(), 3, CV_32F, (void*)to.data());
    cv::flann::Index tree(toData, cv::flann::KDTreeIndexParams(4));
    cv::Mat query((int)from.size(), 3, CV_32F), nearestIndex, squaredDistance;
    for (int round = 0; round < IcpRounds; round++)
    {
        for (int i = 0; i < (int)from.size(); i++)
        {
            cv::Point3f p = pose(from[i]);
            query.at<float>(i, 0) = p.x;
            query.at<float>(i, 1) = p.y;
            query.at<float>(i, 2) = p.z;
        }
        tree.knnSearch(query, nearestIndex, squaredDistance, 1, cv::flann::SearchParams(32));
        float maxPair = std::max(IcpFinalPairMeters, IcpStartPairMeters * std::pow(0.7f, (float)round));
        float maxNeighbour = std::max(maxPair, 0.06f);   // the nearest sample may be a sample spacing away
        // The pairs: B's point, the target's point and normal, and the gap along that normal.
        std::vector<cv::Point3f> ps, normals;
        std::vector<float> gaps;
        cv::Point3f centre(0, 0, 0);
        for (int i = 0; i < (int)from.size(); i++)
        {
            if (squaredDistance.at<float>(i) > maxNeighbour * maxNeighbour) continue;
            cv::Point3f p(query.at<float>(i, 0), query.at<float>(i, 1), query.at<float>(i, 2));
            int j = nearestIndex.at<int>(i);
            float gap = (p - to[j]).dot(toNormals[j]);   // how far p is in front of the target's surface
            if (std::abs(gap) > maxPair) continue;
            ps.push_back(p);
            normals.push_back(toNormals[j]);
            gaps.push_back(gap);
            centre += p;
        }
        const int pairs = (int)ps.size();
        if (pairs < 10) break;
        centre *= 1.0f / pairs;
        // Each pair adds one equation: gap + turn . ((p - centre) x n) + shift . n = 0, for a small
        // turn (axis * angle) about the pairs' centre, and a shift. Turning about the centre (not
        // the room's origin, metres away) keeps turns and shifts apart. Least squares: A x = b.
        cv::Matx66d A = cv::Matx66d::zeros();
        cv::Matx61d b = cv::Matx61d::zeros();
        double sum = 0;
        for (int i = 0; i < pairs; i++)
        {
            const cv::Point3f& n = normals[i];
            cv::Point3f c = (ps[i] - centre).cross(n);
            const double row[6] = { c.x, c.y, c.z, n.x, n.y, n.z };
            for (int r = 0; r < 6; r++)
            {
                for (int k = 0; k < 6; k++) A(r, k) += row[r] * row[k];
                b(r) -= row[r] * gaps[i];
            }
            sum += gaps[i] * gaps[i];
        }
        // Solve direction by direction. How strongly the pairs pin a direction = its eigenvalue:
        // about (share of pairs whose surface faces that way) x pairs. Under 2%: not pinned.
        cv::Mat values, directions;
        cv::eigen(cv::Mat(A), values, directions);
        cv::Matx61d x = cv::Matx61d::zeros();
        int weak = 0;
        for (int k = 0; k < 6; k++)
        {
            double strength = values.at<double>(k);
            if (strength < 0.02 * pairs) { weak++; continue; }
            double along = 0;
            for (int r = 0; r < 6; r++) along += directions.at<double>(k, r) * b(r);
            for (int r = 0; r < 6; r++) x(r) += along / strength * directions.at<double>(k, r);
        }
        if (maxPair == IcpFinalPairMeters)
        {
            pairedShare = (float)pairs / from.size();
            rmsMeters = (float)std::sqrt(sum / pairs);
            weakDirections = weak;
        }
        // The step: turn about the centre, then shift: p -> turn (p - centre) + centre + shift.
        cv::Matx33f turn = TurnAbout(cv::Vec3d(x(0), x(1), x(2)));
        cv::Point3f shift((float)x(3), (float)x(4), (float)x(5));
        Pose step{ turn, centre - turn * centre + shift };
        pose = pose.Then(step);
        if (maxPair == IcpFinalPairMeters && cv::norm(shift) < 1e-4 && step.Degrees() < 0.01f) break;
    }
    return pose;
}

// ---------------------------------------------------------------------
//  A scan, its features, and placing it (11b, now against many scans)
// ---------------------------------------------------------------------

struct Scan
{
    std::vector<cv::Point3f> cloud;   // a 3D point per depth pixel (NaN = none), in this scan's camera coordinates
    cv::Mat picture;                  // the colour picture on the depth grid (black = no colour)
    std::vector<cv::Point3f> sparse;  // every IcpStep-th pixel's point, for ICP...
    std::vector<cv::Point3f> normals; // ...and which way the surface faces there (towards the camera)
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    Pose pose;                        // this scan's camera -> the room (= the first scan's camera)
};

cv::Point3f CloudAt(const Scan& scan, const cv::Point2f& pixel)
{
    int column = std::clamp((int)std::lround(pixel.x), 0, Width - 1), row = std::clamp((int)std::lround(pixel.y), 0, Height - 1);
    return scan.cloud[row * Width + column];
}

// Step 1: a scan from its points and colour picture, with its sparse points, their normals and
// its features (11b).
Scan MakeScan(const std::vector<cv::Point3f>& cloud, const cv::Mat& picture)
{
    Scan scan;
    scan.cloud = cloud;
    scan.picture = picture.clone();
    cv::Mat usable(Height, Width, CV_8U);
    for (int i = 0; i < Width * Height; i++)
        usable.data[i] = !std::isnan(cloud[i].z) && picture.at<cv::Vec3b>(i / Width, i % Width) != cv::Vec3b(0, 0, 0) ? 255 : 0;
    // A normal from the neighbours 2 pixels left/right and up/down: across x down = the facing.
    const int k = 2;
    for (int row = k; row + k < Height; row += IcpStep)
        for (int column = k; column + k < Width; column += IcpStep)
        {
            const cv::Point3f& p = cloud[row * Width + column];
            const cv::Point3f &left = cloud[row * Width + column - k], &right = cloud[row * Width + column + k];
            const cv::Point3f &up = cloud[(row - k) * Width + column], &down = cloud[(row + k) * Width + column];
            if (std::isnan(p.z) || std::isnan(left.z) || std::isnan(right.z) || std::isnan(up.z) || std::isnan(down.z)) continue;
            if (std::abs(right.z - left.z) > 0.05f * p.z || std::abs(down.z - up.z) > 0.05f * p.z) continue;   // an edge
            cv::Point3f n = (right - left).cross(down - up);
            n *= 1.0f / (float)cv::norm(n);
            if (n.dot(p) > 0) n = -n;   // face the camera
            scan.sparse.push_back(p);
            scan.normals.push_back(n);
        }
    cv::erode(usable, usable, cv::Mat(), cv::Point(-1, -1), 3);
    cv::Mat grey;
    cv::cvtColor(picture, grey, cv::COLOR_BGR2GRAY);
    cv::ORB::create(FeatureCount)->detectAndCompute(grey, usable, scan.keypoints, scan.descriptors);
    return scan;
}

// As in 11b: the pose moving scan b's camera coordinates onto scan a's, from matched features.
// Returns how many matches agree with it.
int FeaturePose(const Scan& b, const Scan& a, std::mt19937& random, Pose& pose)
{
    if (b.descriptors.empty() || a.descriptors.empty()) return 0;
    std::vector<std::vector<cv::DMatch>> candidates;
    cv::BFMatcher(cv::NORM_HAMMING).knnMatch(b.descriptors, a.descriptors, candidates, 2);
    std::vector<cv::Point3f> from, to;
    for (const std::vector<cv::DMatch>& best : candidates)
    {
        if (best.size() < 2 || best[0].distance > MatchRatio * best[1].distance) continue;
        cv::Point3f p = CloudAt(b, b.keypoints[best[0].queryIdx].pt), q = CloudAt(a, a.keypoints[best[0].trainIdx].pt);
        if (std::isnan(p.z) || std::isnan(q.z)) continue;
        from.push_back(p);
        to.push_back(q);
    }
    if (from.size() < 3) return 0;
    std::uniform_int_distribution<size_t> pick(0, from.size() - 1);
    std::vector<size_t> bestAgreeing;
    for (int t = 0; t < RansacTries; t++)
    {
        size_t i = pick(random), j = pick(random), k = pick(random);
        if (cv::norm((from[j] - from[i]).cross(from[k] - from[i])) < 0.01) continue;
        Pose guess = BestFit({ from[i], from[j], from[k] }, { to[i], to[j], to[k] });
        std::vector<size_t> agree;
        for (size_t n = 0; n < from.size(); n++)
            if (cv::norm(guess(from[n]) - to[n]) < FeatureInlierMeters) agree.push_back(n);
        if (agree.size() > bestAgreeing.size()) bestAgreeing = agree;
    }
    if (bestAgreeing.size() < 3) return 0;
    std::vector<cv::Point3f> fromAgreeing, toAgreeing;
    for (size_t n : bestAgreeing)
    {
        fromAgreeing.push_back(from[n]);
        toAgreeing.push_back(to[n]);
    }
    pose = BestFit(fromAgreeing, toAgreeing);
    return (int)bestAgreeing.size();
}

// Step 2: place a new scan in the room. Features against every earlier scan (the one sharing the
// most wins) give the rough pose; ICP against all earlier scans together polishes it, which also
// stops small errors piling up from scan to scan. False = refused, with the reason in "report".
bool PlaceScan(Scan& scan, const std::vector<Scan>& earlier, std::mt19937& random, std::string& report)
{
    int bestCount = 0;
    size_t bestScan = 0;
    Pose bestRelative;
    for (size_t s = 0; s < earlier.size(); s++)
    {
        Pose relative;
        int count = FeaturePose(scan, earlier[s], random, relative);
        if (count > bestCount) { bestCount = count; bestScan = s; bestRelative = relative; }
    }
    if (bestCount < MinFeatureMatches)
    {
        report = cv::format("only %d feature matches agree with any earlier scan (need %d)", bestCount, MinFeatureMatches);
        return false;
    }
    Pose rough = bestRelative.Then(earlier[bestScan].pose);
    std::vector<cv::Point3f> target, targetNormals;
    for (const Scan& e : earlier)
        for (size_t i = 0; i < e.sparse.size(); i++)
        {
            target.push_back(e.pose(e.sparse[i]));
            targetNormals.push_back(e.pose.turn * e.normals[i]);
        }
    float pairedShare, rms;
    int weak;
    scan.pose = Icp(scan.sparse, target, targetNormals, rough, pairedShare, rms, weak);
    float icpMoved = (float)cv::norm(scan.pose.shift - rough.shift);
    Pose turnDifference{ rough.turn.t() * scan.pose.turn, cv::Point3f(0, 0, 0) };
    report = cv::format("matched scan %zu (%d features agree); ICP: %.0f%% of points overlap, %.1f mm apart, "
                        "moved %.1f cm and %.1f deg from the features' guess",
                        bestScan + 1, bestCount, pairedShare * 100, rms * 1000, icpMoved * 100, turnDifference.Degrees());
    if (weak > 0) report += cv::format(" (%d direction%s not pinned by the shape - bare walls? - the features' guess kept there)",
                                       weak, weak > 1 ? "s" : "");
    if (pairedShare < MinOverlapShare)
    {
        report += cv::format(": too little overlap (need %.0f%%)", MinOverlapShare * 100);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------
//  Merging: a TSDF grid and a surface net (new)
// ---------------------------------------------------------------------

// A triangle mesh with named groups (09c).
struct Mesh
{
    std::vector<cv::Vec3f> points;
    std::vector<cv::Vec3b> colours;
    std::map<std::string, std::vector<cv::Vec3i>> groups;
};

// The depth camera's lens, to find which pixel sees a point (as in 10).
struct Lens { float fx, fy, cx, cy; };

// Step 4a: every scan into one grid of little cubes ("voxels"), then one mesh where the surface is.
// Each voxel keeps a signed distance to the nearest surface along the camera's view: + in front
// (empty air), - just behind, cut off ("truncated") at TruncationVoxels cubes, and averaged over
// every scan that saw it. Noise averages out; two scans of the same wall become one wall.
Mesh FuseScans(const std::vector<Scan>& scans, const Lens& lens, float& voxel)
{
    // The box to fill: around the scans' points, 1-99% so stray points don't blow it up.
    std::vector<float> xs, ys, zs;
    for (const Scan& scan : scans)
        for (const cv::Point3f& p : scan.sparse)
        {
            cv::Point3f q = scan.pose(p);
            xs.push_back(q.x); ys.push_back(q.y); zs.push_back(q.z);
        }
    auto percentile = [](std::vector<float>& v, double share)
    {
        size_t i = std::min(v.size() - 1, (size_t)(share * v.size()));
        std::nth_element(v.begin(), v.begin() + i, v.end());
        return v[i];
    };
    const float margin = 0.3f;
    cv::Point3f lo(percentile(xs, 0.01) - margin, percentile(ys, 0.01) - margin, percentile(zs, 0.01) - margin);
    cv::Point3f hi(percentile(xs, 0.99) + margin, percentile(ys, 0.99) + margin, percentile(zs, 0.99) + margin);
    voxel = VoxelMeters;
    int nx, ny, nz;
    while (true)
    {
        nx = (int)std::ceil((hi.x - lo.x) / voxel);
        ny = (int)std::ceil((hi.y - lo.y) / voxel);
        nz = (int)std::ceil((hi.z - lo.z) / voxel);
        if ((double)nx * ny * nz <= MaxVoxels) break;
        voxel *= 1.25f;   // too big a room for this cube size: coarser cubes
    }
    const size_t count = (size_t)nx * ny * nz;
    std::vector<float> distance(count, 0), weight(count, 0), colourWeight(count, 0);
    std::vector<cv::Vec3f> colourSum(count, cv::Vec3f(0, 0, 0));
    const float truncation = TruncationVoxels * voxel;
    auto index = [&](int x, int y, int z) { return ((size_t)z * ny + y) * nx + x; };

    for (const Scan& scan : scans)
    {
        cv::Matx33f back = scan.pose.turn.t();   // room -> this scan's camera
        cv::parallel_for_(cv::Range(0, nz), [&](const cv::Range& slices)
        {
            for (int z = slices.start; z < slices.end; z++)
                for (int y = 0; y < ny; y++)
                    for (int x = 0; x < nx; x++)
                    {
                        cv::Point3f centre(lo.x + (x + 0.5f) * voxel, lo.y + (y + 0.5f) * voxel, lo.z + (z + 0.5f) * voxel);
                        cv::Point3f p = back * (centre - scan.pose.shift);
                        if (p.z < 0.3f) continue;   // behind or right at the camera
                        int column = (int)std::lround(p.x * lens.fx / p.z + lens.cx - 0.5f);
                        int row = (int)std::lround(p.y * lens.fy / p.z + lens.cy - 0.5f);
                        if (column < 0 || row < 0 || column >= Width || row >= Height) continue;
                        float seen = scan.cloud[row * Width + column].z;
                        if (std::isnan(seen)) continue;
                        float signedDistance = seen - p.z;          // + = in front of what the camera saw
                        if (signedDistance < -truncation) continue;   // hidden behind it: unknown
                        size_t i = index(x, y, z);
                        float value = std::min(1.0f, signedDistance / truncation);
                        distance[i] = (distance[i] * weight[i] + value) / (weight[i] + 1);
                        weight[i] += 1;
                        if (signedDistance < truncation)
                        {
                            cv::Vec3b c = scan.picture.at<cv::Vec3b>(row, column);
                            if (c != cv::Vec3b(0, 0, 0)) { colourSum[i] += cv::Vec3f(c[0], c[1], c[2]); colourWeight[i] += 1; }
                        }
                    }
        });
    }

    // Step 4b: a surface net. Each cube of 8 neighbouring voxels where the sign changes gets one
    // point, at the average of where the surface crosses its edges. Then every voxel edge that
    // crosses the surface joins the 4 cubes around it into a square (two triangles).
    Mesh mesh;
    std::vector<int> pointOf(count, -1);
    auto corner = [](int c) { return cv::Vec3f((float)(c & 1), (float)((c >> 1) & 1), (float)((c >> 2) & 1)); };
    for (int z = 0; z + 1 < nz; z++)
        for (int y = 0; y + 1 < ny; y++)
            for (int x = 0; x + 1 < nx; x++)
            {
                float value[8];
                int inside = 0;
                bool seenAll = true;
                cv::Vec3f colour(0, 0, 0);
                float colourCount = 0;
                for (int c = 0; c < 8 && seenAll; c++)
                {
                    size_t i = index(x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1));
                    seenAll = weight[i] > 0;
                    value[c] = distance[i];
                    if (value[c] < 0) inside++;
                    colour += colourSum[i];
                    colourCount += colourWeight[i];
                }
                if (!seenAll || inside == 0 || inside == 8) continue;   // never seen, or no surface here
                cv::Vec3f sum(0, 0, 0);
                int crossings = 0;
                for (int bit : { 1, 2, 4 })
                    for (int c = 0; c < 8; c++)
                    {
                        if (c & bit) continue;
                        float a = value[c], b = value[c | bit];
                        if ((a < 0) == (b < 0)) continue;
                        sum += corner(c) + (corner(c | bit) - corner(c)) * (a / (a - b));   // where it crosses zero
                        crossings++;
                    }
                cv::Vec3f inCube = sum / (float)crossings;
                pointOf[index(x, y, z)] = (int)mesh.points.size();
                mesh.points.emplace_back(lo.x + (x + 0.5f + inCube[0]) * voxel, lo.y + (y + 0.5f + inCube[1]) * voxel,
                                         lo.z + (z + 0.5f + inCube[2]) * voxel);
                mesh.colours.push_back(colourCount > 0 ? cv::Vec3b(colour / colourCount) : cv::Vec3b(128, 128, 128));
            }
    std::vector<cv::Vec3i>& faces = mesh.groups["seen_room"];
    const int size[3] = { nx, ny, nz };
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
            {
                size_t i = index(x, y, z);
                if (weight[i] == 0) continue;
                for (int axis = 0; axis < 3; axis++)
                {
                    int at[3] = { x, y, z };
                    if (at[axis] + 1 >= size[axis]) continue;
                    at[axis]++;
                    size_t j = index(at[0], at[1], at[2]);
                    if (weight[j] == 0 || (distance[i] < 0) == (distance[j] < 0)) continue;
                    // The 4 cubes around this edge: one step back along the other two axes, or not.
                    int u = (axis + 1) % 3, w = (axis + 2) % 3, quad[4];
                    const int du[4] = { -1, 0, 0, -1 }, dw[4] = { -1, -1, 0, 0 };
                    bool complete = true;
                    for (int k = 0; k < 4 && complete; k++)
                    {
                        int c[3] = { x, y, z };
                        c[u] += du[k];
                        c[w] += dw[k];
                        complete = c[u] >= 0 && c[w] >= 0 && c[u] + 1 < size[u] && c[w] + 1 < size[w];
                        if (complete) { quad[k] = pointOf[index(c[0], c[1], c[2])]; complete = quad[k] >= 0; }
                    }
                    if (!complete) continue;
                    // Face the empty side (+): towards voxel i if it's the + one.
                    cv::Vec3f facing(0, 0, 0);
                    facing[axis] = distance[i] > 0 ? -1.0f : 1.0f;
                    cv::Vec3f normal = (mesh.points[quad[1]] - mesh.points[quad[0]]).cross(mesh.points[quad[2]] - mesh.points[quad[0]]);
                    if (normal.dot(facing) < 0) std::swap(quad[1], quad[3]);
                    faces.emplace_back(quad[0], quad[1], quad[2]);
                    faces.emplace_back(quad[0], quad[2], quad[3]);
                }
            }
    return mesh;
}

// ---------------------------------------------------------------------
//  The clean model, from all scans (09c, plus merging planes and boxes)
// ---------------------------------------------------------------------

struct Plane
{
    cv::Vec3f normal;
    float distance;
    float DistanceTo(const cv::Vec3f& point) const { return normal.dot(point) + distance; }
};

struct Piece { Plane plane; std::vector<int> cells; };   // one scan's plane and its grid cells (09c)

bool PlaneThrough(const cv::Vec3f& a, const cv::Vec3f& b, const cv::Vec3f& c, Plane& plane)
{
    cv::Vec3f normal = (b - a).cross(c - a);
    float length = (float)cv::norm(normal);
    if (length < 1e-6f) return false;
    plane.normal = normal / length;
    plane.distance = -plane.normal.dot(a);
    if (plane.distance < 0) { plane.normal = -plane.normal; plane.distance = -plane.distance; }
    return true;
}

Plane FitPlane(const std::vector<cv::Vec3f>& points)
{
    cv::Mat data((int)points.size(), 3, CV_32F, (void*)points.data());
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Vec3f mean(pca.mean.at<float>(0), pca.mean.at<float>(1), pca.mean.at<float>(2));
    Plane plane{ cv::Vec3f(pca.eigenvectors.row(2)), 0 };
    plane.distance = -plane.normal.dot(mean);
    if (plane.distance < 0) { plane.normal = -plane.normal; plane.distance = -plane.distance; }
    return plane;
}

float Percentile(std::vector<float> values, double share)
{
    size_t i = std::min(values.size() - 1, (size_t)(share * values.size()));
    std::nth_element(values.begin(), values.begin() + i, values.end());
    return values[i];
}

// As in 09c.
std::vector<Piece> FindPlanes(const std::vector<cv::Vec3f>& grid, int gridColumns, std::mt19937& random)
{
    const int gridRows = (int)grid.size() / gridColumns, reach = NeighbourPixels / SampleStep;
    std::uniform_int_distribution<int> pickOffset(-reach, reach);
    std::vector<bool> free(grid.size());
    int readings = 0;
    for (size_t i = 0; i < grid.size(); i++)
    {
        free[i] = !std::isnan(grid[i][2]);
        if (free[i]) readings++;
    }
    std::vector<Piece> pieces;
    std::vector<int> freeCells;
    for (int search = 0; search < MaxPlanes; search++)
    {
        freeCells.clear();
        for (int i = 0; i < (int)grid.size(); i++) if (free[i]) freeCells.push_back(i);
        if (freeCells.size() < 3) break;
        std::uniform_int_distribution<size_t> pickCell(0, freeCells.size() - 1);
        Plane best{};
        int bestCount = 0;
        for (int t = 0; t < 200; t++)
        {
            int a = freeCells[pickCell(random)];
            int row = a / gridColumns, column = a % gridColumns;
            int bRow = row + pickOffset(random), bColumn = column + pickOffset(random);
            int cRow = row + pickOffset(random), cColumn = column + pickOffset(random);
            if (std::min({ bRow, bColumn, cRow, cColumn }) < 0 || std::max(bRow, cRow) >= gridRows ||
                std::max(bColumn, cColumn) >= gridColumns) continue;
            int b = bRow * gridColumns + bColumn, c = cRow * gridColumns + cColumn;
            if (!free[b] || !free[c]) continue;
            Plane plane;
            if (!PlaneThrough(grid[a], grid[b], grid[c], plane)) continue;
            int count = 0;
            for (int i : freeCells)
                if (std::abs(plane.DistanceTo(grid[i])) < InlierThicknessMeters) count++;
            if (count > bestCount) { bestCount = count; best = plane; }
        }
        if (bestCount < MinPlaneShare * readings) break;
        cv::Mat onPlane(gridRows, gridColumns, CV_8U, cv::Scalar(0)), labels, stats, centroids;
        for (int i : freeCells)
            if (std::abs(best.DistanceTo(grid[i])) < InlierThicknessMeters) onPlane.data[i] = 255;
        int count = cv::connectedComponentsWithStats(onPlane, labels, stats, centroids);
        int biggest = 1;
        for (int piece = 2; piece < count; piece++)
            if (stats.at<int>(piece, cv::CC_STAT_AREA) > stats.at<int>(biggest, cv::CC_STAT_AREA)) biggest = piece;
        if (count < 2 || stats.at<int>(biggest, cv::CC_STAT_AREA) < MinPlaneShare * readings) continue;
        Piece piece;
        std::vector<cv::Vec3f> inliers;
        for (int i : freeCells)
        {
            if (labels.ptr<int>()[i] != biggest) continue;
            piece.cells.push_back(i);
            inliers.push_back(grid[i]);
            free[i] = false;
        }
        piece.plane = FitPlane(inliers);
        pieces.push_back(piece);
    }
    return pieces;
}

void AddQuad(Mesh& mesh, const std::string& group, const cv::Vec3f& a, const cv::Vec3f& b, const cv::Vec3f& c,
             const cv::Vec3f& d, const cv::Vec3b& colour, const cv::Vec3f& facing)
{
    int first = (int)mesh.points.size();
    for (const cv::Vec3f& p : { a, b, c, d })
    {
        mesh.points.push_back(p);
        mesh.colours.push_back(colour);
    }
    bool flip = (b - a).cross(c - a).dot(facing) < 0;
    std::vector<cv::Vec3i>& faces = mesh.groups[group];
    faces.push_back(flip ? cv::Vec3i(first, first + 2, first + 1) : cv::Vec3i(first, first + 1, first + 2));
    faces.push_back(flip ? cv::Vec3i(first, first + 3, first + 2) : cv::Vec3i(first, first + 2, first + 3));
}

void AddBox(Mesh& mesh, const std::string& group, const cv::Vec3f& lo, const cv::Vec3f& hi, const cv::Vec3b& colour)
{
    cv::Vec3f corner[8];
    for (int i = 0; i < 8; i++) corner[i] = cv::Vec3f(i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]);
    const int sides[6][4] = { { 0, 2, 6, 4 }, { 1, 3, 7, 5 }, { 0, 1, 5, 4 }, { 2, 3, 7, 6 }, { 0, 1, 3, 2 }, { 4, 5, 7, 6 } };
    cv::Vec3f centre = (lo + hi) / 2;
    for (const auto& s : sides)
    {
        cv::Vec3f middle = (corner[s[0]] + corner[s[1]] + corner[s[2]] + corner[s[3]]) / 4;
        AddQuad(mesh, group, corner[s[0]], corner[s[1]], corner[s[2]], corner[s[3]], colour, middle - centre);
    }
}

struct Room
{
    Mesh mesh;                 // room coordinates: X right, Y up, Z towards the first camera; first camera at 0,0,0
    std::string upFrom;
    float floorY = 0, ceilingY = 0;
    bool floorSeen = false, ceilingSeen = false;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> walls;
    std::vector<std::pair<cv::Vec3f, cv::Vec3f>> blocks;
    cv::Matx33f firstCamera = cv::Matx33f::eye();   // scan 1's camera (X right, Y down, Z ahead) -> room
};

// Step 4c: 09c's clean model from every scan. Each scan's planes are found as in 09c, moved into
// the room, and merged with the same plane seen from other spots; then sorted, levelled, and
// built as in 09c. The merged mesh from step 4b is levelled with it.
Room BuildRoom(const std::vector<Scan>& scans, const Mesh& seen, std::mt19937& random)
{
    Room room;
    room.mesh = seen;
    const int gridColumns = Width / SampleStep, gridRows = Height / SampleStep, cells = gridColumns * gridRows;
    struct RoomPiece { Plane plane; std::vector<cv::Vec3f> points; cv::Vec3d colourSum; std::vector<std::pair<int, int>> cells; };
    std::vector<RoomPiece> pieces;
    std::vector<std::vector<cv::Vec3f>> roomGrid(scans.size());   // each scan's sampled points, in the room (first camera)
    std::vector<std::vector<cv::Vec3b>> gridColour(scans.size());
    std::vector<std::vector<float>> gridDepth(scans.size());        // ...and their distance from their own camera

    for (size_t s = 0; s < scans.size(); s++)
    {
        std::vector<cv::Vec3f> grid(cells);
        roomGrid[s].resize(cells);
        gridColour[s].resize(cells);
        gridDepth[s].resize(cells);
        for (int cell = 0; cell < cells; cell++)
        {
            int pixel = (cell / gridColumns) * SampleStep * Width + (cell % gridColumns) * SampleStep;
            cv::Point3f p = scans[s].cloud[pixel];
            grid[cell] = cv::Vec3f(p.x, p.y, p.z);
            roomGrid[s][cell] = std::isnan(p.z) ? cv::Vec3f(NAN, NAN, NAN) : cv::Vec3f(scans[s].pose(p));
            gridColour[s][cell] = scans[s].picture.at<cv::Vec3b>(pixel / Width, pixel % Width);
            gridDepth[s][cell] = p.z;
        }
        for (const Piece& found : FindPlanes(grid, gridColumns, random))
        {
            RoomPiece piece{ {}, {}, cv::Vec3d(0, 0, 0), {} };
            cv::Vec3f centre(0, 0, 0);
            for (int i : found.cells)
            {
                piece.points.push_back(roomGrid[s][i]);
                piece.colourSum += cv::Vec3d(gridColour[s][i]);
                piece.cells.emplace_back((int)s, i);
                centre += roomGrid[s][i];
            }
            centre /= (float)found.cells.size();
            piece.plane = FitPlane(piece.points);
            // New: the same plane seen from an earlier spot? Then it's one piece.
            bool merged = false;
            for (RoomPiece& other : pieces)
            {
                if (other.plane.normal.dot(piece.plane.normal) < std::cos(MergeDegrees * (float)CV_PI / 180) ||
                    std::abs(other.plane.DistanceTo(centre)) > MergeMeters) continue;
                other.points.insert(other.points.end(), piece.points.begin(), piece.points.end());
                other.cells.insert(other.cells.end(), piece.cells.begin(), piece.cells.end());
                other.colourSum += piece.colourSum;
                other.plane = FitPlane(other.points);
                merged = true;
                break;
            }
            if (!merged) pieces.push_back(piece);
        }
    }
    auto averageColour = [](const cv::Vec3d& sum, size_t count) { return cv::Vec3b(sum / (double)std::max<size_t>(1, count)); };

    // --- Sort the planes (09c step 5), by the first camera's up ---
    const cv::Vec3f cameraUp(0, -1, 0);
    const float levelCos = std::cos(MaxTiltDegrees * (float)CV_PI / 180), uprightSin = std::sin(MaxTiltDegrees * (float)CV_PI / 180);
    std::vector<float> heights;
    for (const auto& grid : roomGrid)
        for (const cv::Vec3f& p : grid) if (!std::isnan(p[2])) heights.push_back(p.dot(cameraUp));
    if (heights.empty()) return room;
    float lowest = Percentile(heights, 0.02);
    int floorPiece = -1, ceilingPiece = -1;
    std::vector<int> wallPieces;
    for (int k = 0; k < (int)pieces.size(); k++)
    {
        const Plane& plane = pieces[k].plane;
        float upness = plane.normal.dot(cameraUp);
        if (upness > levelCos)
        {
            if (floorPiece < 0 || plane.distance > pieces[floorPiece].plane.distance) floorPiece = k;
        }
        else if (-upness > levelCos)
        {
            bool highEnough = plane.distance - lowest >= MinCeilingHeightMeters;
            if (highEnough && (ceilingPiece < 0 || plane.distance > pieces[ceilingPiece].plane.distance)) ceilingPiece = k;
        }
        else if (std::abs(upness) < uprightSin)
        {
            std::vector<float> wallHeights;
            for (const cv::Vec3f& p : pieces[k].points) wallHeights.push_back(p.dot(cameraUp));
            if (Percentile(wallHeights, 0.98) - Percentile(wallHeights, 0.02) >= MinWallHeightMeters) wallPieces.push_back(k);
        }
    }

    // --- Up and ahead (09c step 6) ---
    cv::Vec3f up = cameraUp;
    room.upFrom = "the first camera (nothing level found, under 2 walls)";
    if (floorPiece >= 0) { up = pieces[floorPiece].plane.normal; room.upFrom = "the floor"; }
    else if (ceilingPiece >= 0) { up = -pieces[ceilingPiece].plane.normal; room.upFrom = "the ceiling"; }
    else
    {
        float bestAcross = 0.5f;
        for (size_t i = 0; i < wallPieces.size(); i++)
            for (size_t j = i + 1; j < wallPieces.size(); j++)
            {
                cv::Vec3f across = pieces[wallPieces[i]].plane.normal.cross(pieces[wallPieces[j]].plane.normal);
                if (cv::norm(across) <= bestAcross) continue;
                bestAcross = (float)cv::norm(across);
                up = across / bestAcross;
                if (up.dot(cameraUp) < 0) up = -up;
                room.upFrom = "two walls";
            }
    }
    std::sort(wallPieces.begin(), wallPieces.end(), [&](int a, int b) { return pieces[a].points.size() > pieces[b].points.size(); });
    auto Flatten = [&](cv::Vec3f v) { v -= up * up.dot(v); return v / (float)cv::norm(v); };
    cv::Vec3f towards = wallPieces.empty() ? Flatten(cv::Vec3f(0, 0, -1)) : Flatten(pieces[wallPieces[0]].plane.normal);
    cv::Vec3f right = up.cross(towards);
    auto ToRoom = [&](const cv::Vec3f& p) { return cv::Vec3f(right.dot(p), up.dot(p), towards.dot(p)); };
    for (cv::Vec3f& p : room.mesh.points) p = ToRoom(p);
    room.firstCamera = cv::Matx33f(right[0], right[1], right[2], up[0], up[1], up[2], towards[0], towards[1], towards[2]);

    std::vector<float> roomHeights, roomX, roomZ;
    for (auto& grid : roomGrid)
        for (cv::Vec3f& p : grid)
        {
            if (std::isnan(p[2])) continue;
            p = ToRoom(p);
            roomHeights.push_back(p[1]);
            roomX.push_back(p[0]);
            roomZ.push_back(p[2]);
        }
    for (RoomPiece& piece : pieces) for (cv::Vec3f& p : piece.points) p = ToRoom(p);
    room.floorSeen = floorPiece >= 0;
    room.ceilingSeen = ceilingPiece >= 0;
    room.floorY = room.floorSeen ? -pieces[floorPiece].plane.distance : Percentile(roomHeights, 0.02);
    room.ceilingY = room.ceilingSeen ? pieces[ceilingPiece].plane.distance : Percentile(roomHeights, 0.98);
    std::vector<std::vector<char>> taken(scans.size(), std::vector<char>(cells, 0));   // cells used by floor, ceiling, walls
    for (int k : { floorPiece, ceilingPiece })
        if (k >= 0) for (const auto& [s, i] : pieces[k].cells) taken[s][i] = 1;

    // --- Walls (09c step 7) ---
    struct Line { cv::Point2f towards, ends[2]; float offset; cv::Vec3b colour; };
    std::vector<Line> lines;
    for (int k : wallPieces)
    {
        cv::Vec3f n = ToRoom(Flatten(pieces[k].plane.normal));
        Line line;
        line.towards = cv::Point2f(n[0], n[2]) / std::hypot(n[0], n[2]);
        cv::Point2f along(-line.towards.y, line.towards.x);
        std::vector<float> offsets, alongs;
        for (const cv::Vec3f& p : pieces[k].points)
        {
            cv::Point2f q(p[0], p[2]);
            offsets.push_back(line.towards.dot(q));
            alongs.push_back(along.dot(q));
        }
        for (const auto& [s, i] : pieces[k].cells) taken[s][i] = 1;
        line.offset = Percentile(offsets, 0.5);
        line.ends[0] = line.towards * line.offset + along * Percentile(alongs, 0.02);
        line.ends[1] = line.towards * line.offset + along * Percentile(alongs, 0.98);
        line.colour = averageColour(pieces[k].colourSum, pieces[k].points.size());
        lines.push_back(line);
    }
    for (size_t i = 0; i < lines.size(); i++)
        for (size_t j = i + 1; j < lines.size(); j++)
        {
            cv::Point2f a = lines[i].towards, b = lines[j].towards;
            float determinant = a.x * b.y - a.y * b.x;
            if (std::abs(determinant) < 0.5f) continue;
            cv::Point2f crossing((lines[i].offset * b.y - lines[j].offset * a.y) / determinant,
                                 (a.x * lines[j].offset - b.x * lines[i].offset) / determinant);
            auto nearestEnd = [&](Line& line) -> cv::Point2f&
            { return cv::norm(line.ends[0] - crossing) < cv::norm(line.ends[1] - crossing) ? line.ends[0] : line.ends[1]; };
            cv::Point2f& endI = nearestEnd(lines[i]);
            cv::Point2f& endJ = nearestEnd(lines[j]);
            if (cv::norm(endI - crossing) < CornerSnapMeters && cv::norm(endJ - crossing) < CornerSnapMeters)
                endI = endJ = crossing;
        }
    for (const Line& line : lines)
    {
        cv::Point2f e0 = line.ends[0], e1 = line.ends[1];
        AddQuad(room.mesh, "clean_walls", { e0.x, room.floorY, e0.y }, { e1.x, room.floorY, e1.y },
                { e1.x, room.ceilingY, e1.y }, { e0.x, room.ceilingY, e0.y }, line.colour,
                { line.towards.x, 0, line.towards.y });
        room.walls.emplace_back(e0, e1);
    }

    // Floor and ceiling: rectangles over the walls, most of the points, and every camera.
    float minX = Percentile(roomX, 0.02), maxX = Percentile(roomX, 0.98);
    float minZ = Percentile(roomZ, 0.02), maxZ = Percentile(roomZ, 0.98);
    std::vector<cv::Point2f> outline;
    for (const auto& wall : room.walls) { outline.push_back(wall.first); outline.push_back(wall.second); }
    for (const Scan& scan : scans) { cv::Vec3f c = ToRoom(cv::Vec3f(scan.pose.shift)); outline.emplace_back(c[0], c[2]); }
    for (const cv::Point2f& e : outline)
    {
        minX = std::min(minX, e.x); maxX = std::max(maxX, e.x);
        minZ = std::min(minZ, e.y); maxZ = std::max(maxZ, e.y);
    }
    const cv::Vec3b grey(128, 128, 128);
    for (bool isFloor : { true, false })
    {
        float y = isFloor ? room.floorY : room.ceilingY;
        int k = isFloor ? floorPiece : ceilingPiece;
        AddQuad(room.mesh, isFloor ? "clean_floor" : "clean_ceiling", { minX, y, minZ }, { maxX, y, minZ }, { maxX, y, maxZ },
                { minX, y, maxZ }, k >= 0 ? averageColour(pieces[k].colourSum, pieces[k].points.size()) : grey,
                { 0, isFloor ? 1.0f : -1.0f, 0 });
    }

    // --- Furniture (09c step 8), per scan, then boxes merged across scans (new) ---
    struct Box { cv::Vec3f lo, hi; cv::Vec3d colourSum; size_t count; };
    std::vector<Box> boxes;
    const float margin = 0.05f;
    for (size_t s = 0; s < scans.size(); s++)
    {
        cv::Mat left(gridRows, gridColumns, CV_8U, cv::Scalar(0));
        for (int i = 0; i < cells; i++)
        {
            const cv::Vec3f& p = roomGrid[s][i];
            if (std::isnan(p[2]) || taken[s][i] || p[1] < room.floorY + margin || p[1] > room.ceilingY - margin) continue;
            // New: a point on a wall's plane is wall, even if this scan saw too little of it to find
            // the plane itself (a sliver at the edge of the view).
            bool onWall = false;
            for (const Line& line : lines)
                onWall = onWall || std::abs(line.towards.dot(cv::Point2f(p[0], p[2])) - line.offset) < InlierThicknessMeters;
            if (!onWall) left.data[i] = 255;
        }
        for (int row = 0; row + 1 < gridRows; row++)
            for (int column = 0; column + 1 < gridColumns; column++)
            {
                int i = row * gridColumns + column;
                for (int neighbour : { i + 1, i + gridColumns })
                {
                    float z = gridDepth[s][i], zn = gridDepth[s][neighbour];
                    if (!left.data[i] || std::isnan(zn)) continue;
                    if (std::abs(z - zn) > 0.03f * std::min(z, zn)) left.data[z < zn ? i : neighbour] = 0;
                }
            }
        cv::Mat labels, stats, centroids;
        int count = cv::connectedComponentsWithStats(left, labels, stats, centroids);
        std::vector<std::vector<int>> blobCells(count);
        for (int i = 0; i < cells; i++) if (labels.ptr<int>()[i] > 0) blobCells[labels.ptr<int>()[i]].push_back(i);
        for (int label = 1; label < count; label++)
        {
            if ((int)blobCells[label].size() < MinBlockCells) continue;
            std::vector<float> xs, ys, zs;
            Box box{ {}, {}, cv::Vec3d(0, 0, 0), blobCells[label].size() };
            for (int i : blobCells[label])
            {
                xs.push_back(roomGrid[s][i][0]);
                ys.push_back(roomGrid[s][i][1]);
                zs.push_back(roomGrid[s][i][2]);
                box.colourSum += cv::Vec3d(gridColour[s][i]);
            }
            box.lo = cv::Vec3f(Percentile(xs, 0.02), room.floorY, Percentile(zs, 0.02));
            box.hi = cv::Vec3f(Percentile(xs, 0.98), Percentile(ys, 0.98), Percentile(zs, 0.98));
            boxes.push_back(box);
        }
    }
    // The same couch seen from two spots gives two overlapping boxes: join any that touch.
    for (bool joined = true; joined; )
    {
        joined = false;
        for (size_t i = 0; i < boxes.size() && !joined; i++)
            for (size_t j = i + 1; j < boxes.size() && !joined; j++)
            {
                bool touch = true;
                for (int a = 0; a < 3; a++)
                    touch = touch && boxes[i].lo[a] <= boxes[j].hi[a] + 0.02f && boxes[j].lo[a] <= boxes[i].hi[a] + 0.02f;
                if (!touch) continue;
                for (int a = 0; a < 3; a++)
                {
                    boxes[i].lo[a] = std::min(boxes[i].lo[a], boxes[j].lo[a]);
                    boxes[i].hi[a] = std::max(boxes[i].hi[a], boxes[j].hi[a]);
                }
                boxes[i].colourSum += boxes[j].colourSum;
                boxes[i].count += boxes[j].count;
                boxes.erase(boxes.begin() + j);
                joined = true;
            }
    }
    for (const Box& box : boxes)
    {
        AddBox(room.mesh, "furniture_blocks", box.lo, box.hi, averageColour(box.colourSum, box.count));
        room.blocks.emplace_back(box.lo, box.hi);
    }
    return room;
}

// As in 09c.
bool SaveObj(const std::string& path, const Mesh& mesh)
{
    std::ofstream file(path);
    file << "# Spotter 13: seen_room = every scan merged, clean_* and furniture_blocks = the tidy (partly guessed) model\n";
    for (size_t i = 0; i < mesh.points.size(); i++)
    {
        const cv::Vec3f& p = mesh.points[i];
        const cv::Vec3b& c = mesh.colours[i];
        file << cv::format("v %.4f %.4f %.4f %.3f %.3f %.3f\n", p[0], p[1], p[2], c[2] / 255.0, c[1] / 255.0, c[0] / 255.0);
    }
    for (const auto& [name, faces] : mesh.groups)
    {
        file << "g " << name << "\n";
        for (const cv::Vec3i& f : faces) file << cv::format("f %d %d %d\n", f[0] + 1, f[1] + 1, f[2] + 1);
    }
    return (bool)file;
}

// Step 4: merge every scan, build the clean model, save. Prints what it made.
void Export(const std::vector<Scan>& scans, const Lens& lens, std::mt19937& random)
{
    if (scans.empty()) { std::printf("Nothing to export yet: SPACE scans.\n"); return; }
    std::printf("Merging %zu scans...\n", scans.size());
    auto start = std::chrono::steady_clock::now();
    float voxel;
    Mesh seen = FuseScans(scans, lens, voxel);
    Room room = BuildRoom(scans, seen, random);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::string path = (std::filesystem::current_path() / cv::format("room_scan_%lld.obj", (long long)std::time(nullptr))).string();
    if (!SaveObj(path, room.mesh)) { std::printf("Could not write %s\n", path.c_str()); return; }
    // How scan 1's camera sat in the room, for presentation/Demo4_Room_Twin (an OBJ comment, ignored elsewhere).
    const cv::Matx33f& t = room.firstCamera;
    std::ofstream(path, std::ios::app) << cv::format("# first camera turn: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                                                     t(0, 0), t(0, 1), t(0, 2), t(1, 0), t(1, 1), t(1, 2), t(2, 0), t(2, 1), t(2, 2));
    std::printf("Saved %s (%.1f MB, %.1f s). A capture of the room: never commit it.\n", path.c_str(),
                std::filesystem::file_size(path) / 1e6, seconds);
    std::printf("  merged mesh: %zu triangles, %.1f cm cubes\n", room.mesh.groups["seen_room"].size(), voxel * 100);
    std::printf("  up from %s; floor %.2f m below the first camera (%s), ceiling %.2f m above it (%s): %.2f m high\n",
                room.upFrom.c_str(), -room.floorY, room.floorSeen ? "seen" : "guessed",
                room.ceilingY, room.ceilingSeen ? "seen" : "guessed", room.ceilingY - room.floorY);
    for (const auto& [a, b] : room.walls)
        std::printf("  wall  %.2f m long, from (%+.2f, %.2f) to (%+.2f, %.2f) m\n", cv::norm(b - a), a.x, -a.y, b.x, -b.y);
    for (const auto& [lo, hi] : room.blocks)
        std::printf("  block %.2f wide x %.2f deep x %.2f high, at (%+.2f, %.2f) m\n", hi[0] - lo[0], hi[2] - lo[2],
                    hi[1] - lo[1], (lo[0] + hi[0]) / 2, -(lo[2] + hi[2]) / 2);
    std::printf("  (positions: X right, Z ahead, along the biggest wall, from the first camera)\n");
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // -----------------------------------------------------------------
    //  Setup: depth and colour, registered (as in 11b)
    // -----------------------------------------------------------------

    libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Warning));
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device* device = freenect2.enumerateDevices() > 0 ? freenect2.openDefaultDevice() : nullptr;
    if (device == nullptr)
    {
        std::printf("No Kinect v2 found, or it is in use. See the main README's troubleshooting table.\n");
        return 1;
    }
    libfreenect2::Freenect2Device::Config config;
    config.MaxDepth = MaxDepthMeters;
    device->setConfiguration(config);
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Depth);
    device->setColorFrameListener(&listener);
    device->setIrAndDepthFrameListener(&listener);
    device->startStreams(true, true);
    libfreenect2::Freenect2Device::IrCameraParams ir = device->getIrCameraParams();
    const Lens lens{ ir.fx, ir.fy, ir.cx, ir.cy };
    libfreenect2::Registration registration(ir, device->getColorCameraParams());
    libfreenect2::Frame undistorted(Width, Height, 4), registered(Width, Height, 4), averaged(Width, Height, 4);

    std::vector<Scan> scans;
    std::vector<std::vector<cv::Point3f>> shownPoints;   // what the viewer draws of each scan, in the room
    std::vector<std::vector<cv::Vec3b>> shownColours;
    int framesCaptured = -1;   // -1 = not scanning
    std::vector<double> sumMm(Width * Height);
    std::vector<int> readings(Width * Height);
    std::string message = "Set the Kinect down where you'll start, then press SPACE (in a window) to scan.";
    bool colourByScan = false;
    std::mt19937 random(1);

    const std::string viewer = "13 room scan - the room so far  (drag/WASD turn, wheel zoom, C colour by scan)";
    const std::string camera = "13 room scan - camera  (SPACE scan, U undo, E export, Q quit)";
    cv::namedWindow(viewer, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(viewer, OnMouse);
    cv::namedWindow(camera, cv::WINDOW_AUTOSIZE);
    std::printf("%s\n", message.c_str());

    // -----------------------------------------------------------------
    //  Main operation
    // -----------------------------------------------------------------

    libfreenect2::FrameMap frames;
    cv::Mat frame;
    while (true)
    {
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == 'a') yawDegrees -= TurnStepDegrees;
        if (key == 'd') yawDegrees += TurnStepDegrees;
        if (key == 'w') pitchDegrees += TurnStepDegrees;
        if (key == 's') pitchDegrees -= TurnStepDegrees;
        if (key == 'z') viewDistance -= ZoomStepMeters;
        if (key == 'x') viewDistance += ZoomStepMeters;
        if (key == 'r') { yawDegrees = 0; pitchDegrees = 0; viewDistance = PivotMeters; }
        if (key == 'c') colourByScan = !colourByScan;
        pitchDegrees = std::clamp(pitchDegrees, -89.0f, 89.0f);
        viewDistance = std::max(viewDistance, 0.1f);

        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }
        libfreenect2::Frame* colorFrame = frames[libfreenect2::Frame::Color];
        cv::cvtColor(cv::Mat(ColorHeight, ColorWidth, CV_8UC4, colorFrame->data), frame, cv::COLOR_BGRA2BGR);
        registration.apply(colorFrame, frames[libfreenect2::Frame::Depth], &undistorted, &registered);
        listener.release(frames);

        // --- U: undo the last scan ---
        if (key == 'u' && framesCaptured < 0 && !scans.empty())
        {
            scans.pop_back();
            shownPoints.pop_back();
            shownColours.pop_back();
            message = cv::format("Removed scan %zu. %zu left.", scans.size() + 1, scans.size());
            std::printf("%s\n", message.c_str());
        }

        // --- E: merge and save ---
        if (key == 'e' && framesCaptured < 0)
        {
            Export(scans, lens, random);
            message = scans.empty() ? "Nothing to export yet." : "Exported (see the console). Scan more and export again, or Q.";
        }

        // --- Step 1: SPACE starts a scan; average AverageFrames frames (07) ---
        if (key == ' ' && framesCaptured < 0)
        {
            framesCaptured = 0;
            std::fill(sumMm.begin(), sumMm.end(), 0.0);
            std::fill(readings.begin(), readings.end(), 0);
        }
        if (framesCaptured >= 0)
        {
            const float* depthMm = (const float*)undistorted.data;
            for (int i = 0; i < Width * Height; i++)
            {
                if (!(depthMm[i] > 0) || std::isinf(depthMm[i])) continue;
                sumMm[i] += depthMm[i];
                readings[i]++;
            }
            framesCaptured++;
        }
        if (framesCaptured == AverageFrames)
        {
            framesCaptured = -1;
            float* averageMm = (float*)averaged.data;
            for (int i = 0; i < Width * Height; i++)
                averageMm[i] = readings[i] >= AverageFrames / 2 ? (float)(sumMm[i] / readings[i]) : 0;
            std::vector<cv::Point3f> cloud(Width * Height);
            for (int row = 0; row < Height; row++)
                for (int column = 0; column < Width; column++)
                {
                    float x, y, z;
                    registration.getPointXYZ(&averaged, row, column, x, y, z);
                    cloud[row * Width + column] = cv::Point3f(x, y, z);
                }
            cv::Mat picture;
            cv::cvtColor(cv::Mat(Height, Width, CV_8UC4, registered.data), picture, cv::COLOR_BGRA2BGR);
            Scan scan = MakeScan(cloud, picture);

            // --- Step 2: place it. Without features no later scan could be matched to it. ---
            std::string report;
            bool placed = false;
            if ((int)scan.keypoints.size() < MinScanFeatures)
                report = cv::format("only %zu features (need %d): the colour picture is too dark or too plain. "
                                    "Turn the lights on, or aim at something with more pattern", scan.keypoints.size(), MinScanFeatures);
            else if (scans.empty())
            {
                placed = true;
                report = "the room's starting point";
            }
            else
                placed = PlaceScan(scan, scans, random, report);
            if (placed)
            {
                scans.push_back(scan);
                std::vector<cv::Point3f> points;
                std::vector<cv::Vec3b> colours;
                for (int row = 0; row < Height; row += ShowStep)
                    for (int column = 0; column < Width; column += ShowStep)
                    {
                        const cv::Point3f& p = scan.cloud[row * Width + column];
                        if (std::isnan(p.z)) continue;
                        points.push_back(scan.pose(p));
                        colours.push_back(scan.picture.at<cv::Vec3b>(row, column));
                    }
                shownPoints.push_back(points);
                shownColours.push_back(colours);
                message = cv::format("Scan %zu placed. Move 0.5-1 m or turn about 30 deg, set it down, SPACE again. E when done.", scans.size());
                std::printf("Scan %zu: %zu features; %s.\n", scans.size(), scan.keypoints.size(), report.c_str());
            }
            else
            {
                message = "Couldn't place that scan (see the console). Move back closer to an earlier spot and SPACE again.";
                std::printf("Scan refused: %s.\n", report.c_str());
            }
            std::printf("%s\n", message.c_str());
        }

        // -----------------------------------------------------------------
        //  Output: the room so far, and the camera with instructions
        // -----------------------------------------------------------------

        std::vector<cv::Point3f> points;
        std::vector<cv::Vec3b> colours;
        for (size_t s = 0; s < shownPoints.size(); s++)
        {
            points.insert(points.end(), shownPoints[s].begin(), shownPoints[s].end());
            if (colourByScan) colours.insert(colours.end(), shownPoints[s].size(), ColorForId((int)s + 1));
            else colours.insert(colours.end(), shownColours[s].begin(), shownColours[s].end());
        }
        cv::Mat image = Render(points, colours);
        for (size_t s = 0; s < scans.size(); s++)
        {
            cv::Point pixel;
            float depth;
            if (!Project(scans[s].pose.shift, pixel, depth)) continue;
            cv::circle(image, pixel, 6, cv::Scalar(0, 255, 255), 2);
            cv::putText(image, std::to_string(s + 1), pixel + cv::Point(8, -8), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        }
        cv::putText(image, cv::format("%zu scans", scans.size()), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 1);
        cv::imshow(viewer, image);

        cv::Mat display;
        cv::resize(frame, display, cv::Size(ColorWidth / 2, ColorHeight / 2));
        std::string top = framesCaptured >= 0 ? cv::format("HOLD STILL  %d / %d", framesCaptured, AverageFrames)
                                              : cv::format("%zu scans   SPACE scan here   U undo   E export   Q quit", scans.size());
        cv::putText(display, top, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        cv::putText(display, message, cv::Point(10, display.rows - 15), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 1);
        cv::imshow(camera, display);
    }

    device->stop();
    device->close();
    cv::destroyAllWindows();
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Scan one spot, export, and compare with 09c's file: same room, but the merged mesh has no
//     torn edges at depth jumps, only gaps where nothing was seen.
//  2. Scan the same spot twice and export: the TSDF averages them, so the mesh gets smoother.
//  3. Press C after a few scans. Each scan's colour should sit on the others' surfaces; a scan
//     that shows as a separate, shifted copy was placed wrong: U removes it.
//  4. VoxelMeters = 0.015 versus 0.06: detail, file size and merge time.
//  5. Walk the Kinect all the way round the room, back to the start. Does the last scan meet the
//     first, or is there a step where small errors added up ("drift")? Real scanners fix that
//     with "loop closure"; here, ICP against all earlier scans keeps it small.
//  6. Scan a bare wall from 2 m, then slide 1 m along it. Refused (no features) or placed wrong
//     (ICP slides along a flat wall, 11)? Put a poster up and try again.
//  7. Hide groups in Blender: seen_room (what was scanned) against the clean_* groups and
//     furniture_blocks (the tidy model). Which boxes are real furniture?
