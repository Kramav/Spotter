// Self-check for room_twin.hpp, with no Kinect: check_room's patterned box room (floor 1.2 m below the
// first camera), 3 scans written to a room folder in the temp folder, then:
//   1. LoadTwinRoom: levelled (the floor is level at -1.2 m).
//   2. FindTwinFloor: the floor at -1.2 m, also with a 0.36 m^2 level "reflection" 0.4 m under it
//      (the review's bug 7: the legacy rule took any big enough patch below).
//      And with the walls' lower 0.6 m hidden and a couch seat: still the floor (D43).
//   3. LocateKinect: views from 0.3 m / 20 deg and 0.64 m / 30 deg off scan 1's spot, found within 2 cm and 1 deg.
//   4. OneViewRoom: one view, levelled, the floor found.
// Run: build\check_room_twin.exe. Exit code 0 = passed.

#include "spotter/room_folder.hpp"
#include "spotter/room_twin.hpp"

#include <chrono>
#include <cstdio>

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
    unsigned h = (unsigned)(face * 73856093) ^ (unsigned)((int)std::floor(p.x / 0.1f) * 19349663) ^ (unsigned)((int)std::floor(p.y / 0.1f) * 83492791) ^
                 (unsigned)((int)std::floor(p.z / 0.1f) * 2654435761u);
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
    scan.fullPicture = picture;
    scan.pose = pose;
    return scan;
}

// Turned left about the camera's up, then down, as check_room's Spot.
spotter::Pose Spot(float right, float forward, float leftDegrees, float downDegrees)
{
    return { spotter::TurnAbout(cv::Vec3d(0, -leftDegrees * CV_PI / 180, 0)) * spotter::TurnAbout(cv::Vec3d(-downDegrees * CV_PI / 180, 0, 0)),
             cv::Point3f(right, 0, forward) };
}

int main()
{
    // The room folder: 3 scans at their true poses, tilted 15 deg down to see the floor.
    const fs::path folder = fs::temp_directory_path() / "spotter_check_room_twin";
    fs::remove_all(folder);
    spotter::RoomInfo info;
    info.program = "check_room_twin";
    info.depthCamera = Camera;
    std::vector<spotter::Scan> scans = { Photograph(Spot(0, 0, 0, 15), 1), Photograph(Spot(0.3f, 0.2f, 40, 15), 2), Photograph(Spot(-0.3f, 0.3f, -50, 15), 3) };
    size_t points;
    bool written = spotter::StartRoomFolder(folder, info);
    for (size_t k = 0; k < scans.size(); k++) written = written && spotter::WriteScan(folder, k, scans[k]);
    written = written && spotter::FinishRoomFolder(folder, info, scans, spotter::RoomOptions(), points);
    Check(written, "the room folder is written");

    // 1. Loaded and levelled.
    spotter::TwinRoom room;
    std::string problem;
    auto start = std::chrono::steady_clock::now();
    bool loaded = spotter::LoadTwinRoom(folder, 0.03f, room, problem);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("1. loaded %s: %zu points in %.1f s %s\n", room.from.c_str(), room.points.size(), seconds, problem.c_str());
    Check(loaded && room.points.size() > 10000, "LoadTwinRoom loads the folder");

    // 2. The floor, then with a reflection patch 0.4 m under it.
    spotter::TwinFloor floor = spotter::FindTwinFloor(room);
    for (const std::string& line : floor.report) std::printf("2. %s\n", line.c_str());
    Check(floor.seen && std::abs(floor.y + 1.2f) < 0.03f, "the floor is seen at -1.2 m");
    spotter::TwinRoom reflected = room;
    for (float x = 0; x < 0.6f; x += 0.02f)
        for (float z = -1.5f; z < -0.9f; z += 0.02f)
        {
            reflected.points.emplace_back(x, -1.6f, z);
            reflected.normals.emplace_back(0.0f, 1.0f, 0.0f);
            reflected.colours.push_back(cv::Vec3b());
        }
    spotter::TwinFloor under = spotter::FindTwinFloor(reflected);
    for (const std::string& line : under.report) std::printf("   %s\n", line.c_str());
    Check(under.seen && std::abs(under.y + 1.2f) < 0.03f, "a level patch under the floor isn't taken for it (bug 7)");

    // ...and with the walls' lower 0.6 m hidden (behind furniture, or seen by few scans) and a 0.5 m
    // high couch seat: the walls' foot comes out too high, but the floor is still the floor (D43).
    spotter::TwinRoom hidden;
    for (size_t i = 0; i < room.points.size(); i++)
    {
        const bool upright = std::abs(room.normals[i].y) < 0.34f;
        if (upright && room.points[i].y < -0.6f) continue;
        hidden.points.push_back(room.points[i]);
        hidden.normals.push_back(room.normals[i]);
        hidden.colours.push_back(room.colours[i]);
    }
    for (float x = 0.2f; x < 1.0f; x += 0.02f)
        for (float z = -1.4f; z < -0.9f; z += 0.02f)
        {
            hidden.points.emplace_back(x, -0.7f, z);
            hidden.normals.emplace_back(0.0f, 1.0f, 0.0f);
            hidden.colours.push_back(cv::Vec3b());
        }
    spotter::TwinFloor low = spotter::FindTwinFloor(hidden);
    for (const std::string& line : low.report) std::printf("   %s\n", line.c_str());
    Check(low.seen && std::abs(low.y + 1.2f) < 0.03f, "the walls' lower part hidden: the floor is still at -1.2 m, not the seat (D43)");

    // 3. Locate the Kinect put back off scan 1's spot: near (one start does it), and far (needs the other starts).
    for (const spotter::Pose& truth : { Spot(0.2f, 0.2f, 20, 15), Spot(-0.5f, 0.4f, -30, 15) })
    {
        spotter::Scan now = Photograph(truth, 9);
        std::vector<cv::Point3f> view;
        for (int row = 0; row < 424; row += 8)   // every 8th pixel each way, as the demos
            for (int column = 0; column < 512; column += 8)
                if (!std::isnan(now.cloud[row * 512 + column].z)) view.push_back(now.cloud[row * 512 + column]);
        start = std::chrono::steady_clock::now();
        spotter::Located located = spotter::LocateKinect(view, room);
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        spotter::Pose inRoom{ room.firstCamera * truth.turn, room.firstCamera * truth.shift };
        double cm = cv::norm(located.pose.shift - inRoom.shift) * 100, degrees = spotter::Pose{ located.pose.turn.t() * inRoom.turn, cv::Point3f() }.Degrees();
        std::printf("3. put back %.2f m / %.0f deg off: located from %d starts in %.1f s, %.0f%% matched, %.1f cm / %.2f deg off%s\n%s",
                    cv::norm(truth.shift), truth.Degrees(), located.tried, seconds, located.share * 100, cm, degrees, located.twoAlike ? " (two alike)" : "",
                    located.report.c_str());
        Check(cm < 2 && degrees < 1, "LocateKinect finds the Kinect within 2 cm and 1 deg");
    }

    // 4. One view as the room: level when told the tilt (15 deg down = -15 up).
    spotter::TwinRoom one = spotter::OneViewRoom(scans[0].cloud, scans[0].picture, 512, -15);
    spotter::TwinFloor oneFloor = spotter::FindTwinFloor(one);
    std::printf("4. one view: %zu points; %s\n", one.points.size(), oneFloor.report[0].c_str());
    Check(oneFloor.seen && std::abs(oneFloor.y + 1.2f) < 0.03f, "OneViewRoom stands level: the floor at -1.2 m");

    fs::remove_all(folder);
    std::printf(failures ? "check_room_twin: %d FAILED\n" : "check_room_twin: all passed\n", failures);
    return failures ? 1 : 0;
}
