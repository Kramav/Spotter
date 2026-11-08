// Self-check and measurements for structure.hpp, with no Kinect: rooms made from a floor plan (wall
// segments, a floor 1.2 m below and a ceiling 1.3 m above the first camera), a patterned surface,
// raycast from known camera spots.
//   1. Relocate in an L-shaped room: one answer, within 2 cm and 1 deg, from several known poses.
//   2. Symmetry: a 4 x 3 m rectangle (2-fold) and a 3.5 m square (4-fold) must be flagged
//      ambiguous with the truth among the rivals; colour votes must pick the truth.
//   3. A rectangle with a couch and a doorway: does the furniture / doorway break the tie?
//   4. Views: floor + one wall (must say under-constrained), a corner, walls + ceiling without floor.
//   5. Edge stability between two nearby scans, and coverage per surface.
//   6. Continuous: the capture range of PlaceAfter from the previous pose (a grid of moves).
// Run: build\check_structure.exe. Exit code 0 = passed.

#include "spotter/stats.hpp"
#include "spotter/structure.hpp"

#include <chrono>
#include <cstdio>

int failures = 0;
void Check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAILED: %s\n", what); failures++; }
}

const spotter::PinholeCamera Camera{ 365.5f, 365.5f, 256.0f, 212.0f, 512, 424 };
const float FloorY = 1.2f, CeilingY = -1.3f;   // camera coordinates: Y down

struct Box { cv::Point3f low, high; };
struct Room
{
    std::vector<std::pair<cv::Point2f, cv::Point2f>> walls;   // floor-plan segments (x, z)
    std::vector<Box> boxes;
};

Room Polygon(const std::vector<cv::Point2f>& corners)
{
    Room room;
    for (size_t i = 0; i < corners.size(); i++) room.walls.emplace_back(corners[i], corners[(i + 1) % corners.size()]);
    return room;
}

// The nearest surface along a ray, and the colour of the 10 cm block it hits there.
float Raycast(const Room& room, const cv::Point3f& o, const cv::Point3f& d, cv::Vec3b& colour)
{
    float best = 1e9f;
    int face = 0;
    if (d.y > 0) { best = (FloorY - o.y) / d.y; face = 1; }
    if (d.y < 0) { best = (CeilingY - o.y) / d.y; face = 2; }
    for (size_t w = 0; w < room.walls.size(); w++)
    {
        cv::Point2f a = room.walls[w].first, b = room.walls[w].second, e = b - a, r(d.x, d.z), q(o.x - a.x, o.z - a.y);
        float denom = r.x * e.y - r.y * e.x;
        if (std::abs(denom) < 1e-9f) continue;
        float t = (e.x * q.y - e.y * q.x) / denom, s = (r.x * q.y - r.y * q.x) / denom;   // along the ray, along the wall
        if (t > 0 && t < best && s >= 0 && s <= 1) { best = t; face = 3 + (int)w; }
    }
    for (size_t k = 0; k < room.boxes.size(); k++)
    {
        const float lo[3] = { room.boxes[k].low.x, room.boxes[k].low.y, room.boxes[k].low.z }, hi[3] = { room.boxes[k].high.x, room.boxes[k].high.y, room.boxes[k].high.z };
        const float oo[3] = { o.x, o.y, o.z }, dd[3] = { d.x, d.y, d.z };
        float enter = 0, leave = 1e9f;
        for (int a = 0; a < 3; a++)
        {
            float t1 = (lo[a] - oo[a]) / dd[a], t2 = (hi[a] - oo[a]) / dd[a];
            enter = std::max(enter, std::min(t1, t2));
            leave = std::min(leave, std::max(t1, t2));
        }
        if (enter < leave && enter > 0 && enter < best) { best = enter; face = 100 + (int)k; }
    }
    cv::Point3f p = o + d * best;
    int i = (int)std::floor(p.x / 0.1f), j = (int)std::floor(p.y / 0.1f), k = (int)std::floor(p.z / 0.1f);
    unsigned h = (unsigned)(face * 73856093) ^ (unsigned)(i * 19349663) ^ (unsigned)(j * 83492791) ^ (unsigned)(k * 2654435761u);
    h ^= h >> 13; h *= 0x5bd1e995; h ^= h >> 15;
    colour = cv::Vec3b((uchar)(h & 255), (uchar)((h >> 8) & 255), (uchar)((h >> 16) & 255));
    return best;
}

spotter::Scan Photograph(const Room& room, const spotter::Pose& pose, int seed)
{
    cv::RNG noise(seed);
    std::vector<cv::Point3f> cloud(512 * 424);
    cv::Mat picture(424, 512, CV_8UC3);
    for (int row = 0; row < 424; row++)
        for (int column = 0; column < 512; column++)
        {
            cv::Point3f ray = Camera.Unproject(cv::Point2f((float)column, (float)row), 1.0f);
            cv::Vec3b colour;
            float t = Raycast(room, pose.shift, pose.turn * ray, colour);
            cloud[row * 512 + column] = t > 6 ? cv::Point3f(NAN, NAN, NAN) : ray * (t * (1 + (float)noise.gaussian(0.001)));   // 6 m range
            picture.at<cv::Vec3b>(row, column) = colour;
        }
    spotter::Scan scan = spotter::MakeScan(cloud, picture, spotter::ScanOptions());
    scan.pose = pose;
    return scan;
}

// right / forward in metres, yaw = turned left, tilt = looking up (degrees).
spotter::Pose Spot(float x, float z, float leftDegrees, float upDegrees = 0)
{
    return { spotter::TurnAbout(cv::Vec3d(0, -leftDegrees * CV_PI / 180, 0)) * spotter::TurnAbout(cv::Vec3d(upDegrees * CV_PI / 180, 0, 0)), cv::Point3f(x, 0, z) };
}

void Off(const spotter::Pose& a, const spotter::Pose& b, double& cm, double& degrees)
{
    cm = cv::norm(a.shift - b.shift) * 100;
    degrees = spotter::Pose{ a.turn.t() * b.turn, cv::Point3f() }.Degrees();
}

struct Scene
{
    Room room;
    std::vector<spotter::Scan> scans;
    std::vector<const spotter::Scan*> pointers;
    spotter::RoomMap map;
};

// The map: four scans from the middle, looking four ways, tilted 15 deg down (like a 13e session's first spots).
Scene Build(const Room& room, cv::Point2f middle)
{
    Scene scene{ room };
    for (int k = 0; k < 4; k++) scene.scans.push_back(Photograph(room, Spot(middle.x, middle.y, 90.0f * k, -15), 10 + k));
    for (const spotter::Scan& s : scene.scans) scene.pointers.push_back(&s);
    std::mt19937 random(1);
    scene.map = spotter::MakeRoomMap(scene.pointers, Camera, spotter::StructureOptions(), random, cv::Point3f(0, -1, 0));
    return scene;
}

spotter::RelocateResult Locate(const Scene& scene, const spotter::Scan& scan, bool print, const char* name, const spotter::Pose& truth)
{
    std::mt19937 random(2);
    spotter::ScanStructure s = spotter::ExtractStructure(scan.cloud, Camera, cv::Point3f(), spotter::StructureOptions(), random);
    auto start = std::chrono::steady_clock::now();
    spotter::RelocateResult r = spotter::Relocate(scan, s, scene.map, Camera, spotter::RelocateOptions());
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    for (spotter::RelocateCandidate& c : r.candidates)
    {
        c.colourVotes = spotter::ColourVotes(scan, c.pose, scene.pointers);
        c.offPlaneShare = spotter::OffPlaneShare(scan, s, c.pose, scene.map);
    }
    if (print)
    {
        int walls = 0;
        for (const spotter::StructurePlane& p : s.planes) walls += p.surface == spotter::Surface::Wall;
        std::printf("  %-34s %d walls seen, %3d candidates, %4.0f ms%s%s %s\n", name, walls, r.generated, ms, r.ambiguous ? ", AMBIGUOUS" : "",
                    r.note.empty() ? "" : ";", r.note.c_str());
        for (size_t i = 0; i < r.candidates.size() && i < 4; i++)
        {
            const spotter::RelocateCandidate& c = r.candidates[i];
            double cm, degrees;
            Off(c.pose, truth, cm, degrees);
            std::printf("     agreement %+.3f  share %.3f  rms %.1f mm  colour votes %4d  off-plane fit %.2f  -> %6.1f cm, %5.1f deg from the truth\n",
                        c.agreement, c.share, c.rmsMeters * 1000, c.colourVotes, c.offPlaneShare, cm, degrees);
        }
    }
    return r;
}

bool Found(const spotter::RelocateResult& r, const spotter::Pose& truth, size_t among)
{
    for (size_t i = 0; i < r.candidates.size() && i < among; i++)
    {
        double cm, degrees;
        Off(r.candidates[i].pose, truth, cm, degrees);
        if (cm < 2 && degrees < 1) return true;
    }
    return false;
}

size_t ByColour(const spotter::RelocateResult& r)
{
    size_t best = 0;
    for (size_t i = 1; i < r.candidates.size(); i++) if (r.candidates[i].colourVotes > r.candidates[best].colourVotes) best = i;
    return best;
}

void Coverage(const Scene& scene, const char* name)
{
    std::mt19937 random(3);
    double area[4] = { 0, 0, 0, 0 }, density[4] = { 0, 0, 0, 0 };
    int seen[4] = { 0, 0, 0, 0 };
    for (const spotter::Scan& scan : scene.scans)
    {
        spotter::ScanStructure s = spotter::ExtractStructure(scan.cloud, Camera, scan.pose.turn.t() * cv::Point3f(0, -1, 0), spotter::StructureOptions(), random);
        bool any[4] = { false, false, false, false };
        for (const spotter::StructurePlane& p : s.planes)
        {
            int k = (int)p.surface;
            area[k] += p.areaMeters2;
            density[k] += p.pointsPerMeter2 * p.areaMeters2;
            any[k] = true;
        }
        for (int k = 0; k < 4; k++) seen[k] += any[k];
    }
    std::printf("  %s: ", name);
    for (int k = 0; k < 3; k++)
        std::printf("%s in %d/%zu scans, %.1f m2 per scan, %.0f points/m2;  ", spotter::SurfaceName((spotter::Surface)k), seen[k], scene.scans.size(),
                    area[k] / scene.scans.size(), area[k] > 0 ? density[k] / area[k] : 0.0);
    std::printf("\n");
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const Room lShape = Polygon({ { -2, -1.5f }, { 2, -1.5f }, { 2, 0.5f }, { 0, 0.5f }, { 0, 1.5f }, { -2, 1.5f } });
    const Room rectangle = Polygon({ { -2, -1.5f }, { 2, -1.5f }, { 2, 1.5f }, { -2, 1.5f } });
    const Room square = Polygon({ { -1.75f, -1.75f }, { 1.75f, -1.75f }, { 1.75f, 1.75f }, { -1.75f, 1.75f } });
    Room furnished = Polygon({ { -2, -1.5f }, { 2, -1.5f }, { 2, 1.5f }, { 1.4f, 1.5f } });   // the far wall has a doorway at x 0.5 to 1.4
    furnished.walls.push_back({ { 0.5f, 1.5f }, { -2, 1.5f } });
    furnished.walls.push_back({ { -2, 1.5f }, { -2, -1.5f } });
    furnished.walls.push_back({ { 0.5f, 1.5f }, { 0.5f, 3.0f } });   // a corridor behind the doorway
    furnished.walls.push_back({ { 1.4f, 1.5f }, { 1.4f, 3.0f } });
    furnished.walls.push_back({ { 0.5f, 3.0f }, { 1.4f, 3.0f } });
    furnished.boxes.push_back({ { -1.9f, 0.6f, -1.4f }, { -1.0f, 1.2f, -0.6f } });   // a couch in one corner

    // 1. The L-shaped room: one answer, from several known poses.
    std::printf("1. L-shaped room (no symmetry): Relocate from known poses\n");
    Scene l = Build(lShape, { -1, -0.5f });
    // The safety property: never a wrong answer presented as the only one. Otherwise: right and unique;
    // ambiguous with the truth among the rivals and colour picking it; or ambiguous without it (say so,
    // ask the user).
    int unique = 0, ambiguous = 0, unresolved = 0, wrong = 0;
    for (const spotter::Pose& truth : { Spot(-1.2f, -0.6f, 30, -15), Spot(-0.5f, 0, -60, -10), Spot(1, -0.8f, 150, -15), Spot(-1.5f, 1, -120, 5) })
    {
        spotter::RelocateResult r = Locate(l, Photograph(lShape, truth, 50), true, "pose", truth);
        if (!r.ambiguous) (Found(r, truth, 1) ? unique : wrong)++;
        else if (Found(r, truth, 99) && Found({ { r.candidates[ByColour(r)] } }, truth, 1)) ambiguous++;
        else unresolved++;
    }
    std::printf("  -> %d right and unique, %d ambiguous (colour picks the truth), %d ambiguous without the truth (ask the user), %d WRONG and unique\n",
                unique, ambiguous, unresolved, wrong);
    Check(wrong == 0, "L-shape: never a wrong answer presented as the only one");
    Check(unique + ambiguous >= 3, "L-shape: at least 3 of the 4 poses found (directly or via colour)");

    // 2. Symmetric rooms.
    std::printf("2. Symmetric rooms\n");
    Scene rect = Build(rectangle, { -0.3f, -0.2f });
    spotter::Pose truth = Spot(0.6f, 0.4f, 70, -15);
    spotter::RelocateResult r = Locate(rect, Photograph(rectangle, truth, 60), true, "4 x 3 m rectangle", truth);
    Check(r.ambiguous, "rectangle: flagged ambiguous");
    Check(Found(r, truth, 99), "rectangle: the truth is among the rivals");
    Check(!r.candidates.empty() && Found({ { r.candidates[ByColour(r)] } }, truth, 1), "rectangle: colour votes pick the truth");
    Scene sq = Build(square, { 0.2f, -0.3f });
    truth = Spot(-0.4f, 0.5f, -35, -15);
    r = Locate(sq, Photograph(square, truth, 61), true, "3.5 m square", truth);
    Check(r.ambiguous, "square: flagged ambiguous");
    Check(!r.candidates.empty() && Found({ { r.candidates[ByColour(r)] } }, truth, 1), "square: colour votes pick the truth");

    // 3. Furniture and a doorway.
    std::printf("3. Rectangle with a couch and a doorway\n");
    Scene furn = Build(furnished, { 0.2f, -0.2f });
    truth = Spot(0.8f, 0.3f, 200, -15);   // facing the couch corner
    r = Locate(furn, Photograph(furnished, truth, 62), true, "facing the couch", truth);
    std::printf("     -> %s\n", r.ambiguous ? "still ambiguous by structure; see the colour / off-plane columns" : "the couch and doorway broke the tie");
    Check(Found(r, truth, 99), "furnished: the truth is among the candidates");

    // 4. What a view can pin.
    std::printf("4. Views\n");
    truth = Spot(1.4f, -1.0f, -90, -15);   // 0.6 m from the right wall of the L, facing it: floor + one wall
    r = Locate(l, Photograph(lShape, truth, 70), true, "floor + one wall (close)", truth);
    Check(r.note.find("under-constrained") != std::string::npos, "floor + one wall is reported as under-constrained");
    truth = Spot(-1.0f, -0.5f, 135, -15);
    r = Locate(l, Photograph(lShape, truth, 71), true, "a corner (floor + 2 walls)", truth);
    Check(!r.ambiguous ? Found(r, truth, 1) : Found(r, truth, 99) && Found({ { r.candidates[ByColour(r)] } }, truth, 1),
          "a corner view: found, or ambiguous among the room's corners with colour picking the truth");
    truth = Spot(-1.0f, -0.5f, 60, 35);   // tilted up 35 deg
    r = Locate(l, Photograph(lShape, truth, 72), true, "tilted up: walls + ceiling, little floor", truth);

    // 5. Edges and coverage.
    std::printf("5. Edges between two nearby scans of the L (10 cm, 5 deg apart), and coverage\n");
    {
        std::mt19937 random(4);
        spotter::Scan a = Photograph(lShape, Spot(-1, -0.5f, 120), 80), b = Photograph(lShape, Spot(-0.9f, -0.5f, 125), 81);
        spotter::ScanStructure sa = spotter::MoveStructure(spotter::ExtractStructure(a.cloud, Camera, a.pose.turn.t() * cv::Point3f(0, -1, 0), {}, random), a.pose);
        spotter::ScanStructure sb = spotter::MoveStructure(spotter::ExtractStructure(b.cloud, Camera, b.pose.turn.t() * cv::Point3f(0, -1, 0), {}, random), b.pose);
        for (const spotter::Edge& ea : spotter::FindEdges(sa))
        {
            float bestDegrees = 90, bestMeters = 1e9f;
            for (const spotter::Edge& eb : spotter::FindEdges(sb))
            {
                if (eb.Kind(sb) != ea.Kind(sa)) continue;
                float degrees, meters;
                spotter::EdgeOffset(ea, eb, degrees, meters);
                if (meters < bestMeters) { bestMeters = meters; bestDegrees = degrees; }
            }
            std::printf("  %-13s %4.2f m long: matched %s\n", ea.Kind(sa).c_str(), cv::norm(ea.ends[1] - ea.ends[0]),
                        bestMeters < 1e8f ? cv::format("%.2f deg, %.1f mm apart", bestDegrees, bestMeters * 1000).c_str() : "not seen in the other scan");
        }
    }
    Coverage(l, "L map, 4 level scans");
    Coverage(rect, "rectangle map");

    // 6. Continuous: from the previous pose, how big a move does ICP recover? Two views: one that sees a
    //    corner (every direction pinned), one that sees a single wall (sliding along it is free).
    bool smallOk = true;
    for (int view = 0; view < 2; view++)
    {
        const float heading = view == 0 ? 135.0f : 100.0f;   // 135: the corner at (-2, -1.5); 100: mostly the left wall
        std::printf("6. Continuous (PlaceAfter from the previous pose), %s: recovered within 1 cm and 0.5 deg?\n       ",
                    view == 0 ? "a corner view" : "a one-wall view");
        const float moves[] = { 0.05f, 0.10f, 0.20f, 0.30f, 0.45f, 0.60f }, turns[] = { 0, 5, 10, 20, 30 };
        for (float t : turns) std::printf("%5.0f deg", t);
        std::printf("\n");
        spotter::Scan previous = Photograph(lShape, Spot(-1, -0.5f, heading, -15), 90);
        for (float m : moves)
        {
            std::printf("  %4.0f cm", m * 100);
            for (float t : turns)
            {
                spotter::Pose next = Spot(-1 + m * 0.8f, -0.5f + m * 0.6f, heading + t, -15);
                spotter::IcpResult fit = spotter::PlaceAfter(Photograph(lShape, next, 91), previous, previous.pose);
                double cm, degrees;
                Off(fit.pose, next, cm, degrees);
                bool ok = cm < 1 && degrees < 0.5;
                std::printf("%9s", ok ? "yes" : cv::format("%.0fcm%s", cm, fit.weakDirections ? "w" : "").c_str());
                if (view == 0 && m <= 0.20f && t <= 10) smallOk = smallOk && ok;
            }
            std::printf("\n");
        }
    }
    std::printf("  (w = ICP reported directions the view can't pin)\n");
    Check(smallOk, "Continuous recovers moves up to 20 cm and 10 deg from a corner view");

    std::printf(failures ? "check_structure: %d FAILED\n" : "check_structure: all passed\n", failures);
    return failures ? 1 : 0;
}
