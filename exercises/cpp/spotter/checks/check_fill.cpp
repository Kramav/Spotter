// Self-check for 13x's (and Demo 6's) hole filling (examples/13x_Room_Fill/fill_holes.hpp), with no Kinect:
// check_room's patterned box room with its couch, 3 scans written to a room folder and loaded as 13x does
// (LoadTwinRoom, FindTwinFloor, the points above the under-floor noise), then FillHoles up to 1.6 m
// (review 2026-11-11, initiative 6: this replaces the temporary harnesses behind D40 and 13x).
//   1. The floor under the couch, which no scan saw, is filled.
//   2. The filled points lie on the room's true surfaces (a wall, the floor, a couch face): 98% within 3 cm, none
//      past 11 cm. The rest are the fill's own edges: the couch top runs on about 10 cm past its back, a side face
//      about 6 cm (the 3 cm grid, and a piece filled out to its rectangle; legacy's, unchanged).
//   3. What was seen isn't filled again: few filled points sit in a 3 cm cube a real point is in.
//   4. The walls are found (traced round the room as one path), the floor is called seen, the marking has both bands.
// Run: build\check_fill.exe. Exit code 0 = passed. Writes only in the system's temp folder.

#include "../../examples/13x_Room_Fill/fill_holes.hpp"
#include "spotter/room_folder.hpp"
#include "spotter/room_twin.hpp"

#include <chrono>
#include <cstdio>
#include <unordered_set>

namespace fs = std::filesystem;

int failures = 0;
void Check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAILED: %s\n", what); failures++; }
}

const float Low[3] = { -1.5f, -1.3f, -1.0f }, High[3] = { 2.0f, 1.2f, 3.0f };            // the room (Y down), as check_room
const float CouchLow[3] = { 0.2f, 0.6f, 2.0f }, CouchHigh[3] = { 1.0f, 1.2f, 2.5f };
const spotter::PinholeCamera Camera{ 365.5f, 365.5f, 256.0f, 212.0f, 512, 424 };

float Raycast(const cv::Point3f& from, const cv::Point3f& ray, cv::Vec3b& colour)
{
    const float o[3] = { from.x, from.y, from.z }, d[3] = { ray.x, ray.y, ray.z };
    float best = 1e9f;
    int face = 0;
    for (int a = 0; a < 3; a++)
        if (d[a] != 0)
        {
            float t = ((d[a] > 0 ? High[a] : Low[a]) - o[a]) / d[a];
            if (t < best) { best = t; face = 2 * a + (d[a] > 0); }
        }
    float enter = 0, leave = 1e9f;
    int enterFace = 0;
    for (int a = 0; a < 3; a++)
    {
        float t1 = (CouchLow[a] - o[a]) / d[a], t2 = (CouchHigh[a] - o[a]) / d[a];
        if (std::min(t1, t2) > enter) { enter = std::min(t1, t2); enterFace = 6 + a; }
        leave = std::min(leave, std::max(t1, t2));
    }
    if (enter < leave && enter > 0 && enter < best) { best = enter; face = enterFace; }
    cv::Point3f p = from + ray * best;
    int i = (int)std::floor(p.x / 0.1f), j = (int)std::floor(p.y / 0.1f), k = (int)std::floor(p.z / 0.1f);
    unsigned h = (unsigned)(face * 73856093) ^ (unsigned)(i * 19349663) ^ (unsigned)(j * 83492791) ^ (unsigned)(k * 2654435761u);
    h ^= h >> 13; h *= 0x5bd1e995; h ^= h >> 15;
    colour = cv::Vec3b((uchar)(h & 255), (uchar)((h >> 8) & 255), (uchar)((h >> 16) & 255));
    return best;
}

spotter::Scan Photograph(const spotter::Pose& pose, int seed)
{
    cv::RNG noise(seed);
    std::vector<cv::Point3f> cloud(512 * 424);
    cv::Mat picture(424, 512, CV_8UC3);
    for (int row = 0; row < 424; row++)
        for (int column = 0; column < 512; column++)
        {
            cv::Point3f ray = Camera.Unproject(cv::Point2f((float)column, (float)row), 1.0f);
            cv::Vec3b colour;
            float t = Raycast(pose.shift, pose.turn * ray, colour);
            cloud[row * 512 + column] = ray * (t * (1 + (float)noise.gaussian(0.0005)));
            picture.at<cv::Vec3b>(row, column) = colour;
        }
    spotter::Scan scan = spotter::MakeScan(cloud, picture, spotter::ScanOptions());
    scan.colourIndex.assign(512 * 424, -1);
    scan.pose = pose;
    return scan;
}

spotter::Pose Spot(float right, float forward, float leftDegrees, float downDegrees)
{
    return { spotter::TurnAbout(cv::Vec3d(0, -leftDegrees * CV_PI / 180, 0)) * spotter::TurnAbout(cv::Vec3d(-downDegrees * CV_PI / 180, 0, 0)),
             cv::Point3f(right, 0, forward) };
}

// How far a point (in the raycast room's coordinates) is from the nearest true surface: a wall, the floor, the
// ceiling, or a face of the couch.
float OffSurfaces(const cv::Point3f& p)
{
    const float q[3] = { p.x, p.y, p.z };
    float d = 1e9f;
    for (int a = 0; a < 3; a++) d = std::min({ d, std::abs(q[a] - Low[a]), std::abs(q[a] - High[a]) });
    for (int a = 0; a < 3; a++)
    {
        float outside = 0;   // the distance to the couch's face across axis a, if p is within the face's extent
        bool within = true;
        for (int b = 0; b < 3; b++)
            if (b != a) { within = within && q[b] > CouchLow[b] - 0.02f && q[b] < CouchHigh[b] + 0.02f; }
        if (within) outside = std::min(std::abs(q[a] - CouchLow[a]), std::abs(q[a] - CouchHigh[a])), d = std::min(d, outside);
    }
    return d;
}

int main()
{
    // The room folder: 3 scans at their true poses, tilted 15 deg down to see the floor (check_room_twin's).
    const fs::path folder = fs::temp_directory_path() / "spotter_check_fill";
    std::error_code error;
    fs::remove_all(folder, error);
    spotter::RoomInfo info;
    info.program = "check_fill";
    info.depthCamera = Camera;
    std::vector<spotter::Scan> scans = { Photograph(Spot(0, 0, 0, 15), 1), Photograph(Spot(0.3f, 0.2f, 40, 15), 2), Photograph(Spot(-0.3f, 0.3f, -50, 15), 3) };
    size_t written;
    bool ok = spotter::StartRoomFolder(folder, info);
    for (size_t k = 0; k < scans.size(); k++) ok = ok && spotter::WriteScan(folder, k, scans[k]);
    ok = ok && spotter::FinishRoomFolder(folder, info, scans, spotter::RoomOptions(), written);
    spotter::TwinRoom twin;
    std::string problem;
    ok = ok && spotter::LoadTwinRoom(folder, 0.03f, twin, problem);
    Check(ok, "the room folder is written and loads");
    fs::remove_all(folder, error);
    if (!ok) return 1;

    // Loaded as 13x does.
    spotter::TwinFloor floor = spotter::FindTwinFloor(twin);
    std::vector<cv::Point3f> points, normals;
    std::vector<cv::Vec3b> colours;
    for (size_t i = 0; i < twin.points.size(); i++)
        if (twin.points[i].y >= floor.y - 0.10f)
        {
            points.push_back(twin.points[i]);
            normals.push_back(twin.normals[i]);
            colours.push_back(twin.colours[i]);
        }
    auto start = std::chrono::steady_clock::now();
    Filled filled = FillHoles(points, normals, colours, floor.y, floor.seen, floor.y + 1.6f, twin.firstCamera);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("Filled %.2f m^2 in %.2f s: floor %.2f, %d walls %.2f, %d other surfaces %.2f, %d legs, %d shadows; floor at %.3f\n",
                filled.floorArea + filled.wallArea + filled.otherArea, seconds, filled.floorArea, filled.walls, filled.wallArea,
                filled.others, filled.otherArea, filled.legs, filled.shadows, floor.y);
    for (const std::string& line : filled.report) std::printf("%s\n", line.c_str());

    // The twin room is the raycast room levelled; back into the raycast room's coordinates (Y down) with
    // the first camera's turn, as check_room does with firstCamera.
    auto toRoom = [&](const cv::Point3f& p) { return twin.firstCamera.t() * p; };

    // 1. The floor under the couch (5 cm in from its edges), in 3 cm cells.
    std::unordered_set<int> under;
    for (const cv::Point3f& f : filled.points)
    {
        cv::Point3f p = toRoom(f);
        if (std::abs(p.y - High[1]) < 0.03f && p.x > CouchLow[0] + 0.05f && p.x < CouchHigh[0] - 0.05f && p.z > CouchLow[2] + 0.05f && p.z < CouchHigh[2] - 0.05f)
            under.insert((int)std::floor(p.x / 0.03f) * 1000 + (int)std::floor(p.z / 0.03f));
    }
    const int cells = (int)((0.7f / 0.03f) * (0.4f / 0.03f));
    std::printf("1. under the couch: %zu of about %d cells filled\n", under.size(), cells);
    Check(under.size() > 0.7 * cells, "the floor under the couch is filled");

    // 2. On a true surface.
    std::vector<float> off;
    for (const cv::Point3f& f : filled.points) off.push_back(OffSurfaces(toRoom(f)));
    const float most = off.empty() ? 1e9f : spotter::Percentile(off, 0.98), worst = off.empty() ? 1e9f : spotter::Percentile(off, 1.0);
    std::printf("2. %zu filled points: %.1f mm median, %.1f mm 98%%, %.1f mm worst off the true surfaces\n", off.size(),
                spotter::Median(off) * 1000, most * 1000, worst * 1000);
    Check(!off.empty() && most < 0.03f && worst < 0.11f, "the filled points lie on the true surfaces (98% within 3 cm, none past 11 cm)");

    // 3. Not where it was seen.
    std::unordered_set<long long> seen;
    auto key = [](const cv::Point3f& p) {
        return ((long long)std::floor(p.x / 0.03f) + 1000) * 4000000 + ((long long)std::floor(p.y / 0.03f) + 1000) * 2000 + ((long long)std::floor(p.z / 0.03f) + 1000);
    };
    for (const cv::Point3f& p : points) seen.insert(key(p));
    size_t again = 0;
    for (const cv::Point3f& f : filled.points) again += seen.count(key(f));
    const double share = filled.points.empty() ? 1 : (double)again / filled.points.size();
    std::printf("3. %.1f%% of the filled points share a 3 cm cube with a real point\n", share * 100);
    Check(share < 0.2, "what was seen isn't filled again");

    // 4.
    bool bands[2] = { false, false };
    for (char s : filled.stripe) bands[s != 0] = true;
    Check(filled.walls >= 1 && filled.wallArea > 0 && filled.floorSeen, "the walls are found and filled, and the floor called seen");
    Check(bands[0] && bands[1], "the marking has both bands");

    std::printf(failures ? "check_fill: %d FAILED\n" : "check_fill: all passed\n", failures);
    return failures ? 1 : 0;
}
