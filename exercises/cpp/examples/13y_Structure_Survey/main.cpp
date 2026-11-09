// =====================================================================
//  13y · Structure Survey                              (no Kinect needed)
//
//  Measures, on saved rooms, what the structure-localisation study needs
//  (docs/structure-localization-findings.md). Numbers only: it reads each
//  scan's cloud.bin and pose.txt and the room's room.txt, never the colour
//  pictures or features.
//
//    1. Coverage   per surface (floor, ceiling, walls): in how many scans,
//                  how much area, how dense; who sees 2 walls, a corner.
//    2. Edges      floor-wall / ceiling-wall / wall-wall edges of each pair
//                  of consecutive scans, in the room: how well they agree.
//    3. Relocate   each scan against a map of all the OTHER scans, from
//                  structure alone (spotter::Relocate), compared with its
//                  stored pose; and two multi-start ICP baselines (legacy
//                  Demo 4's search around scan 1's spot, and a wide one
//                  from every other scan's spot, all round).
//    4. Continuous each scan placed against the previous one only, from the
//                  previous pose and from the "next shot" pose
//                  (spotter::PlaceAfter), compared with its stored pose.
//
//  The stored poses are 13e's own estimates (features + ICP), not ground
//  truth: agreement means "consistent", not "accurate".
//
//  Run:  13y_Structure_Survey [folder]   (default: room_scans where it runs)
//        13y_Structure_Survey room_scans view 12   one scan's Relocate answers in 3D: green = its
//            stored pose, red = Relocate's (N steps through the candidates), grey = the other scans
//        13y_Structure_Survey room_scans 11        sections 4 then 3, from scan 11 (resumes a long run)
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_set>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/room_folder.hpp"
#include "spotter/stats.hpp"
#include "spotter/structure.hpp"
#include "spotter/view.hpp"

// ---------------------------------------------------------------------
//  Settings
// ---------------------------------------------------------------------

const float AgreeMeters = 0.05f, AgreeDegrees = 2;   // a pose "agrees" with the stored one within this
const spotter::StructureOptions Structure;

// ---------------------------------------------------------------------
//  Loading (numbers only)
// ---------------------------------------------------------------------

struct Room
{
    spotter::RoomInfo info;
    std::vector<spotter::Scan> scans;                 // included scans, in order
    std::vector<int> numbers;
    cv::Point3f up;                                   // the room's up in scan 1's coordinates
    std::vector<spotter::ScanStructure> inRoom;       // each scan's structure, moved into the room
};

bool Load(const std::filesystem::path& folder, Room& room)
{
    std::string problem;
    if (!spotter::ReadRoomInfo(folder, room.info, problem)) { std::printf("  %s\n", problem.c_str()); return false; }
    for (const std::filesystem::path& scanFolder : spotter::ScanFolders(folder))
    {
        spotter::Scan scan;
        int number;
        if (!spotter::LoadScan(scanFolder, room.info, scan, number, problem, false)) { std::printf("  scan %d skipped: %s\n", number, problem.c_str()); continue; }
        if (scan.excluded) continue;
        room.scans.push_back(std::move(scan));
        room.numbers.push_back(number);
    }
    const cv::Matx33f& t = room.info.firstCamera;   // scan 1's camera -> levelled room (Y up): up is its second row
    room.up = room.info.finished ? cv::Point3f(t(1, 0), t(1, 1), t(1, 2)) : cv::Point3f(0, -1, 0);
    std::mt19937 random(1);
    for (const spotter::Scan& s : room.scans)
        room.inRoom.push_back(spotter::MoveStructure(spotter::ExtractStructure(s.cloud, room.info.depthCamera, s.pose.turn.t() * room.up, Structure, random), s.pose));
    std::printf("  %zu included scans; up %s\n", room.scans.size(), room.info.finished ? "from room.txt (levelled by BuildRoom)" : "assumed (camera -Y)");
    return !room.scans.empty();
}

// ---------------------------------------------------------------------
//  1. Coverage
// ---------------------------------------------------------------------

int NonParallelWalls(const spotter::ScanStructure& s)
{
    std::vector<cv::Point3f> directions;
    for (const spotter::StructurePlane& p : s.planes)
    {
        if (p.surface != spotter::Surface::Wall) continue;
        bool same = false;
        for (const cv::Point3f& d : directions) same = same || std::abs(d.dot(p.plane.normal)) > std::cos(30 * CV_PI / 180);
        if (!same) directions.push_back(p.plane.normal);
    }
    return (int)directions.size();
}

void Coverage(const Room& room)
{
    std::printf("\n1. Coverage (per scan: m2 of floor / ceiling / walls seen; wall directions; points per m2 on the ceiling)\n");
    double area[3] = {}, readings[3] = {};
    int seen[3] = {}, twoWalls = 0, corner = 0, ceilingEdge = 0;
    for (size_t k = 0; k < room.inRoom.size(); k++)
    {
        const spotter::ScanStructure& s = room.inRoom[k];
        double a[3] = {}, ceilingDensity = 0;
        for (const spotter::StructurePlane& p : s.planes)
        {
            int i = (int)p.surface;
            if (i > 2) continue;
            a[i] += p.areaMeters2;
            readings[i] += p.pointsPerMeter2 * p.areaMeters2;
            if (p.surface == spotter::Surface::Ceiling) ceilingDensity = p.pointsPerMeter2;
        }
        int directions = NonParallelWalls(s);
        bool edgeToCeiling = false;
        for (const spotter::Edge& e : spotter::FindEdges(s)) edgeToCeiling = edgeToCeiling || e.Kind(s) == "ceiling-wall";
        for (int i = 0; i < 3; i++) { area[i] += a[i]; seen[i] += a[i] > 0; }
        twoWalls += directions >= 2;
        corner += directions >= 2 && a[0] > 0;
        ceilingEdge += edgeToCeiling;
        std::printf("  scan %3d: floor %5.2f  ceiling %5.2f  walls %5.2f m2   %d wall directions   ceiling %6.0f pts/m2%s\n", room.numbers[k], a[0], a[1], a[2],
                    directions, ceilingDensity, edgeToCeiling ? "   ceiling-wall edge" : "");
    }
    const int n = (int)room.inRoom.size();
    const char* names[3] = { "floor", "ceiling", "walls" };
    for (int i = 0; i < 3; i++)
        std::printf("  %-8s seen in %2d of %d scans (%3.0f%%), %5.2f m2 per scan on average, %6.0f readings per m2\n", names[i], seen[i], n, 100.0 * seen[i] / n,
                    area[i] / n, area[i] > 0 ? readings[i] / area[i] : 0.0);
    std::printf("  2+ wall directions in %d of %d scans (%.0f%%); a corner with floor (all 6 directions pinned) in %d (%.0f%%); a ceiling-wall edge in %d (%.0f%%)\n",
                twoWalls, n, 100.0 * twoWalls / n, corner, 100.0 * corner / n, ceilingEdge, 100.0 * ceilingEdge / n);
}

// ---------------------------------------------------------------------
//  2. Edges between consecutive scans
// ---------------------------------------------------------------------

void EdgeStability(const Room& room)
{
    std::printf("\n2. Edges seen by consecutive scans, in the room (matched by kind, nearest within 30 cm and 10 deg)\n");
    std::map<std::string, std::vector<float>> degrees, meters;
    std::map<std::string, int> found, matched;
    for (size_t k = 1; k < room.inRoom.size(); k++)
    {
        std::vector<spotter::Edge> a = spotter::FindEdges(room.inRoom[k - 1]), b = spotter::FindEdges(room.inRoom[k]);
        for (const spotter::Edge& ea : a)
        {
            std::string kind = ea.Kind(room.inRoom[k - 1]);
            found[kind]++;
            float bestD = 0, bestM = 1e9f;
            for (const spotter::Edge& eb : b)
            {
                if (eb.Kind(room.inRoom[k]) != kind) continue;
                float d, m;
                spotter::EdgeOffset(ea, eb, d, m);
                if (d < 10 && m < bestM) { bestM = m; bestD = d; }
            }
            if (bestM > 0.3f) continue;
            matched[kind]++;
            degrees[kind].push_back(bestD);
            meters[kind].push_back(bestM);
        }
    }
    for (const auto& [kind, count] : found)
        std::printf("  %-13s %3d found, %3d seen again in the next scan; offset median %5.1f mm (95%%: %5.1f mm), angle median %.2f deg\n", kind.c_str(), count,
                    matched[kind], spotter::Median(meters[kind]) * 1000, spotter::Percentile(meters[kind], 0.95) * 1000, spotter::Median(degrees[kind]));
}

// ---------------------------------------------------------------------
//  3. Relocate (leave one out) and the baselines
// ---------------------------------------------------------------------

spotter::RoomMap MapWithout(const Room& room, size_t skip)
{
    spotter::RoomMap map;
    std::vector<spotter::ScanStructure> structures;
    std::unordered_set<long long> taken;
    for (size_t k = 0; k < room.scans.size(); k++)
    {
        if (k == skip) continue;
        structures.push_back(room.inRoom[k]);
        const spotter::Scan& s = room.scans[k];
        for (size_t i = 0; i < s.surface.points.size(); i++)
        {
            cv::Point3f p = s.pose(s.surface.points[i]);
            if (!taken.insert(spotter::CubeKey(p, 0.01f)).second) continue;
            map.surface.points.push_back(p);
            map.surface.normals.push_back(s.pose.turn * s.surface.normals[i]);
        }
    }
    map.structure = spotter::MergeStructures(structures);
    map.structure.up = room.up;
    return map;
}

bool Agrees(const spotter::Pose& a, const spotter::Pose& b)
{
    return cv::norm(a.shift - b.shift) < AgreeMeters && spotter::Pose{ a.turn.t() * b.turn, cv::Point3f() }.Degrees() < AgreeDegrees;
}

// Multi-start ICP (legacy Demo 4's Locate): ICP from each start on part of the view, stop at 95%, refine the best.
spotter::Pose MultiStart(const spotter::Scan& scan, const spotter::RoomMap& map, const std::vector<spotter::Pose>& starts, int& tried)
{
    std::vector<cv::Point3f> part, half;
    for (size_t i = 0; i < scan.surface.points.size(); i += 8) part.push_back(scan.surface.points[i]);
    for (size_t i = 0; i < scan.surface.points.size(); i += 2) half.push_back(scan.surface.points[i]);
    const spotter::IcpOptions wide{ 30, 0.5f, 0.02f, 0.05f };
    std::vector<cv::Point3f> coarse, coarseNormals;   // the search runs on the map thinned to 3 cm (each ICP rebuilds its tree)
    std::unordered_set<long long> taken;
    for (size_t i = 0; i < map.surface.points.size(); i++)
        if (taken.insert(spotter::CubeKey(map.surface.points[i], 0.03f)).second) { coarse.push_back(map.surface.points[i]); coarseNormals.push_back(map.surface.normals[i]); }
    spotter::IcpResult best;
    tried = 0;
    for (const spotter::Pose& start : starts)
    {
        spotter::IcpResult fit = spotter::Icp(part, coarse, coarseNormals, start, wide);
        tried++;
        if (fit.pairedShare > best.pairedShare) best = fit;
        if (best.pairedShare >= 0.95f) break;
    }
    return spotter::Icp(half, map.surface.points, map.surface.normals, best.pose, wide).pose;
}

void RelocateAll(const Room& room, int firstNumber)
{
    std::printf("\n3. Relocate each scan in a map of the others (agrees = within %.0f cm and %.0f deg of its stored pose)\n", AgreeMeters * 100, AgreeDegrees);
    const cv::Point3f up = room.up, across = cv::norm(up.cross(cv::Point3f(0, 0, 1))) > 0.1 ? up.cross(cv::Point3f(0, 0, 1)) : up.cross(cv::Point3f(1, 0, 0));
    const cv::Point3f side = across * (1.0f / (float)cv::norm(across)), ahead = up.cross(side);
    auto TurnUp = [&](float degrees) { return spotter::TurnAbout(cv::Vec3d(up.x, up.y, up.z) * (degrees * CV_PI / 180)); };
    int counts[3][4] = {};   // [method][right & unique, ambiguous with truth, ambiguous without, wrong]
    double seconds[3] = {};
    for (size_t k = 0; k < room.scans.size(); k++)
    {
        if (room.numbers[k] < firstNumber) continue;   // resuming a long run: the totals below count only these
        spotter::RoomMap map = MapWithout(room, k);
        const spotter::Scan& scan = room.scans[k];
        std::mt19937 random(2);
        auto start = std::chrono::steady_clock::now();
        spotter::ScanStructure s = spotter::ExtractStructure(scan.cloud, room.info.depthCamera, cv::Point3f(), Structure, random);
        spotter::RelocateResult r = spotter::Relocate(scan, s, map, room.info.depthCamera, spotter::RelocateOptions());
        seconds[0] += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        bool among = false;
        for (const spotter::RelocateCandidate& c : r.candidates) among = among || Agrees(c.pose, scan.pose);
        int outcome = r.candidates.empty() ? 2 : !r.ambiguous ? (Agrees(r.candidates[0].pose, scan.pose) ? 0 : 3) : among ? 1 : 2;
        counts[0][outcome]++;
        // Baselines: Demo 4's search around scan 1's spot, and a wide one from every other scan's spot.
        std::vector<spotter::Pose> demo4, wide;
        const cv::Point2i shifts[] = { { 0, 0 }, { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 }, { 1, 1 }, { -1, 1 }, { 1, -1 }, { -1, -1 } };
        for (float degrees : { 0.0f, -10.0f, 10.0f, -20.0f, 20.0f, -30.0f, 30.0f })
            for (const cv::Point2i& sh : shifts) demo4.push_back({ TurnUp(degrees), side * (0.4f * sh.x) + ahead * (0.4f * sh.y) });
        for (size_t j = 0; j < room.scans.size(); j++)
            if (j != k) for (int degrees = 0; degrees < 360; degrees += 30) wide.push_back({ TurnUp((float)degrees), room.scans[j].pose.shift });
        int tried[2];
        start = std::chrono::steady_clock::now();
        bool demoOk = Agrees(MultiStart(scan, map, demo4, tried[0]), scan.pose);
        seconds[1] += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        start = std::chrono::steady_clock::now();
        bool wideOk = Agrees(MultiStart(scan, map, wide, tried[1]), scan.pose);
        seconds[2] += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        counts[1][demoOk ? 0 : 3]++;
        counts[2][wideOk ? 0 : 3]++;
        int walls = 0;
        for (const spotter::StructurePlane& p : s.planes) walls += p.surface == spotter::Surface::Wall;
        std::printf("  scan %3d: up from %-9s %d walls, %3d candidates -> %-26s | Demo 4 (%3d starts): %-5s | wide (%3d starts): %s\n", room.numbers[k],
                    s.upFrom.c_str(), walls, r.generated,
                    outcome == 0 ? "agrees, unique" : outcome == 1 ? "AMBIGUOUS, truth among rivals" : outcome == 2 ? "ambiguous / none, no truth" : "WRONG, unique",
                    tried[0], demoOk ? "agree" : "off", tried[1], wideOk ? "agree" : "off");
    }
    const int n = counts[0][0] + counts[0][1] + counts[0][2] + counts[0][3];   // the scans run (fewer when resumed)
    const char* names[3] = { "structure Relocate", "Demo 4 multi-start", "wide multi-start" };
    for (int m = 0; m < 3; m++)
        std::printf("  %-19s agrees+unique %2d, ambiguous with truth %2d, ambiguous without %2d, wrong %2d  (of %d; %.1f s per scan)\n", names[m], counts[m][0],
                    counts[m][1], counts[m][2], counts[m][3], n, seconds[m] / n);
}

// ---------------------------------------------------------------------
//  4. Continuous
// ---------------------------------------------------------------------

void ContinuousAll(const Room& room)
{
    std::printf("\n4. Continuous: each scan against the previous one, from the previous pose / the recommended next shot\n");
    int agree[2] = {}, n = 0;
    std::vector<spotter::Scan> sofar;
    sofar.push_back(room.scans[0]);
    for (size_t k = 1; k < room.scans.size(); k++)
    {
        const spotter::Scan &previous = room.scans[k - 1], &scan = room.scans[k];
        spotter::Recommendation next = spotter::RecommendNext(sofar, room.info.depthCamera, spotter::ScanOptions());
        spotter::IcpResult fromLast = spotter::PlaceAfter(scan, previous, previous.pose);
        spotter::IcpResult fromNext = spotter::PlaceAfter(scan, previous, next.found ? next.pose : previous.pose);
        double moved = cv::norm(scan.pose.shift - previous.pose.shift) * 100, turned = spotter::Pose{ previous.pose.turn.t() * scan.pose.turn, cv::Point3f() }.Degrees();
        double fromAdvice = next.found ? cv::norm(scan.pose.shift - next.pose.shift) * 100 : -1;
        double adviceTurn = next.found ? spotter::Pose{ next.pose.turn.t() * scan.pose.turn, cv::Point3f() }.Degrees() : -1;
        bool a = Agrees(fromLast.pose, scan.pose), b = Agrees(fromNext.pose, scan.pose);
        agree[0] += a;
        agree[1] += b;
        n++;
        std::printf("  scan %3d: moved %5.1f cm, turned %5.1f deg; was %5.1f cm / %5.1f deg from the advice | from last: %-5s (%d weak) | from advice: %s\n",
                    room.numbers[k], moved, turned, fromAdvice, adviceTurn, a ? "agree" : "off", fromLast.weakDirections, b ? "agree" : "off");
        sofar.push_back(scan);
    }
    std::printf("  from the previous pose: %d of %d agree; from the recommended shot: %d of %d\n", agree[0], n, agree[1], n);
}

// ---------------------------------------------------------------------
//  View: one scan's Relocate answers, in 3D, against where 13e put it
// ---------------------------------------------------------------------

// The map of the other scans (grey), the scan at its stored pose (green), and at one Relocate
// candidate (red; N steps through them). No colour pictures: points and cameras only.
void ViewScan(const Room& room, int number)
{
    size_t k = std::find(room.numbers.begin(), room.numbers.end(), number) - room.numbers.begin();
    if (k == room.numbers.size()) { std::printf("  no included scan %d\n", number); return; }
    const spotter::Scan& scan = room.scans[k];
    spotter::RoomMap map = MapWithout(room, k);
    std::mt19937 random(2);
    spotter::ScanStructure s = spotter::ExtractStructure(scan.cloud, room.info.depthCamera, cv::Point3f(), Structure, random);
    spotter::RelocateResult r = spotter::Relocate(scan, s, map, room.info.depthCamera, spotter::RelocateOptions());
    std::printf("  scan %d: %zu candidates kept of %d%s. %s\n", number, r.candidates.size(), r.generated, r.ambiguous ? ", ambiguous" : "", r.note.c_str());
    if (r.candidates.empty()) return;

    std::vector<cv::Point3f> grey, stored, placed;
    std::unordered_set<long long> taken;
    cv::Point3f middle(0, 0, 0);
    for (const cv::Point3f& p : map.surface.points)
        if (taken.insert(spotter::CubeKey(p, 0.03f)).second) { grey.push_back(p); middle += p; }
    middle *= 1.0f / grey.size();
    for (const cv::Point3f& p : scan.surface.points) stored.push_back(scan.pose(p));
    const std::string window = "13y view: scan " + std::to_string(number);
    const std::string keys = "N  next candidate   drag / W A S D  turn   wheel / Z X  zoom   R  reset   Q or Esc  quit";
    spotter::OrbitView view(spotter::OrbitView::Options(), middle, 5);
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    view.Attach(window);
    spotter::PointRenderer renderer;
    size_t shown = 0;
    for (int key = 0; !spotter::IsQuitKey(key); key = cv::waitKey(30))
    {
        if (key == 'n') shown = (shown + 1) % r.candidates.size();
        view.HandleKey(key);
        const spotter::RelocateCandidate& c = r.candidates[shown];
        placed.clear();
        for (const cv::Point3f& p : scan.surface.points) placed.push_back(c.pose(p));
        cv::Mat& image = renderer.Begin(view);
        renderer.Draw(grey, cv::Vec3b(110, 110, 110), 2);
        renderer.Draw(stored, cv::Vec3b(0, 200, 0), 2, 2);
        renderer.Draw(placed, cv::Vec3b(0, 0, 255), 2, 2);
        spotter::DrawCamera(image, view, scan.pose, room.info.depthCamera, cv::Scalar(0, 200, 0), 2, "13e");
        spotter::DrawCamera(image, view, c.pose, room.info.depthCamera, cv::Scalar(0, 0, 255), 2, "Relocate #" + std::to_string(shown + 1));
        double off = cv::norm(c.pose.shift - scan.pose.shift), turned = spotter::Pose{ c.pose.turn.t() * scan.pose.turn, cv::Point3f() }.Degrees();
        cv::putText(image, "green = where 13e placed it, red = Relocate's answer, grey = the other scans", cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
        cv::putText(image, cv::format("#%zu of %zu: agreement %.2f, paired %.0f%%, %.2f m / %.0f deg from green%s%s", shown + 1, r.candidates.size(), c.agreement,
                                      c.share * 100, off, turned, c.underConstrained ? ", under-constrained" : "", r.ambiguous ? " (ambiguous)" : ""),
                    cv::Point(10, 44), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        cv::imshow(window, spotter::WithKeys(image, keys));
    }
    cv::destroyWindow(window);
}

// ---------------------------------------------------------------------
//  Main operation
// ---------------------------------------------------------------------

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::filesystem::path where = std::filesystem::absolute(argc > 1 ? argv[1] : "room_scans");
    std::vector<spotter::SavedRoom> rooms = spotter::FindRooms(where);
    std::printf("%zu saved rooms in %s\n", rooms.size(), where.string().c_str());
    for (const spotter::SavedRoom& saved : rooms)
    {
        std::printf("\n=== %s (format %d, %s, saved %s) ===\n", saved.folder.filename().string().c_str(), saved.format, saved.program.c_str(), saved.saved.c_str());
        Room room;
        if (!Load(saved.folder, room)) continue;
        if (argc > 3 && std::string(argv[2]) == "view") { ViewScan(room, std::atoi(argv[3])); continue; }
        // A second argument resumes section 3 from that scan number, after section 4 (Relocate takes minutes per scan).
        if (argc > 2) { ContinuousAll(room); RelocateAll(room, std::atoi(argv[2])); continue; }
        Coverage(room);
        EdgeStability(room);
        RelocateAll(room, 1);
        ContinuousAll(room);
    }
    return 0;
}
