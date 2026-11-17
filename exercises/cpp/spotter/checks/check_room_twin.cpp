// Self-check for room_twin.hpp, with no Kinect: check_room's patterned box room (floor 1.2 m below the
// first camera), 3 scans written to a room folder in the temp folder, then:
//   1. LoadTwinRoom: levelled (the floor is level at -1.2 m).
//   2. FindTwinFloor: the floor at -1.2 m, also with a 0.36 m^2 level "reflection" 0.4 m under it
//      (the review's bug 7: the legacy rule took any big enough patch below).
//      And with the walls' lower 0.6 m hidden and a couch seat: still the floor (D43).
//   3. LocateKinect: views from 0.3 m / 20 deg and 0.64 m / 30 deg off scan 1's spot, found within 2 cm and 1 deg.
//   4. OneViewRoom: one view, levelled, the floor found.
//   5. The ceiling (D58): none in the folder above (every scan looks down); with a scan looking up, FindTwinCeiling
//      finds it 2.5 m above the floor, and CeilingPoints hides its points but keeps every wall point up to 5 cm
//      under it (the walls at full height).
// Run: build\check_room_twin.exe. Exit code 0 = passed.

#include "spotter/room_folder.hpp"
#include "spotter/room_twin.hpp"
#include "test_room.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace fs = std::filesystem;


// Photographed, with its pose (the truth) and the full picture, as a room folder holds them.
spotter::Scan Taken(const spotter::Pose& pose, int seed)
{
    spotter::Scan scan = Photograph(pose, seed);
    scan.fullPicture = scan.picture.clone();
    scan.pose = pose;
    return scan;
}

int main()
{
    // The room folder: 3 scans at their true poses, tilted 15 deg down to see the floor.
    const fs::path folder = fs::temp_directory_path() / "spotter_check_room_twin";
    { std::error_code ignored; fs::remove_all(folder, ignored); }
    spotter::RoomInfo info;
    info.program = "check_room_twin";
    info.depthCamera = Camera;
    std::vector<spotter::Scan> scans = { Taken(Spot(0, 0, 0, 0, 15), 1), Taken(Spot(0.3f, 0, 0.2f, 40, 15), 2), Taken(Spot(-0.3f, 0, 0.3f, -50, 15), 3) };
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
    for (const spotter::Pose& truth : { Spot(0.2f, 0, 0.2f, 20, 15), Spot(-0.5f, 0, 0.4f, -30, 15) })
    {
        spotter::Scan now = Taken(truth, 9);
        std::vector<cv::Point3f> view;
        for (int row = 0; row < 424; row += 8)   // every 8th pixel each way, as the demos
            for (int column = 0; column < 512; column += 8)
                if (!std::isnan(now.cloud[row * 512 + column].z)) view.push_back(now.cloud[row * 512 + column]);
        start = std::chrono::steady_clock::now();
        spotter::Located located = spotter::LocateKinect(view, room);
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        spotter::Pose inRoom{ room.firstCamera * truth.turn, room.firstCamera * truth.shift };
        double cm = cv::norm(located.pose.shift - inRoom.shift) * 100, degrees = located.pose.Inverse().Then(inRoom).Degrees();
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

    // 5. The ceiling. The folder above has none: every scan looks 15 deg down.
    spotter::TwinCeiling none = spotter::FindTwinCeiling(room, floor.y);
    std::printf("5. %s\n", none.report[0].c_str());
    const std::vector<char> noneHidden = spotter::CeilingPoints(room, none);
    Check(!none.seen && std::count(noneHidden.begin(), noneHidden.end(), 1) == 0, "no scan looked up: no ceiling, nothing hidden");

    // ...and a room folder with a scan looking 30 deg up (and one looking back), as check_room_shell's shots.
    const fs::path upFolder = fs::temp_directory_path() / "spotter_check_room_twin_up";
    { std::error_code ignored; fs::remove_all(upFolder, ignored); }
    std::vector<spotter::Scan> upScans = scans;
    upScans.push_back(Taken(Spot(0, 0, 0.5f, 0, -30), 4));
    upScans.push_back(Taken(Spot(0.3f, 0, 1.0f, 180, 10), 5));
    written = spotter::StartRoomFolder(upFolder, info);
    for (size_t k = 0; k < upScans.size(); k++) written = written && spotter::WriteScan(upFolder, k, upScans[k]);
    written = written && spotter::FinishRoomFolder(upFolder, info, upScans, spotter::RoomOptions(), points);
    spotter::TwinRoom upRoom;
    Check(written && spotter::LoadTwinRoom(upFolder, 0.03f, upRoom, problem), "the room folder with a scan looking up is written and loads");
    spotter::TwinFloor upFloor = spotter::FindTwinFloor(upRoom);
    spotter::TwinCeiling ceiling = spotter::FindTwinCeiling(upRoom, upFloor.y);
    std::printf("   %s\n", ceiling.report[0].c_str());
    Check(ceiling.seen && std::abs(ceiling.y - 1.3f) < 0.02f && std::abs(ceiling.y - upFloor.y - 2.5f) < 0.03f,
          "FindTwinCeiling finds the ceiling 1.3 m above the first camera, 2.5 m above the floor");
    const std::vector<char> isCeiling = spotter::CeilingPoints(upRoom, ceiling);
    size_t onCeiling = 0, ceilingHidden = 0, onWalls = 0, wallsHidden = 0;
    float highestWall = -1e9f;
    for (size_t i = 0; i < upRoom.points.size(); i++)
    {
        const cv::Point3f& p = upRoom.points[i];
        if (std::abs(p.y - 1.3f) < 0.01f && std::abs(upRoom.normals[i].y) > 0.94f) { onCeiling++; ceilingHidden += isCeiling[i]; }
        if (std::abs(upRoom.normals[i].y) < 0.34f && p.y < 1.25f)   // upright (within 20 deg), up to 5 cm under the ceiling
        {
            onWalls++;
            wallsHidden += isCeiling[i];
            if (!isCeiling[i]) highestWall = std::max(highestWall, p.y);
        }
    }
    std::printf("   ceiling points hidden: %zu of %zu; upright points up to 5 cm under it hidden: %zu of %zu; the highest kept: %.3f m\n",
                ceilingHidden, onCeiling, wallsHidden, onWalls, highestWall);
    Check(onCeiling > 1000 && ceilingHidden == onCeiling, "CeilingPoints hides every point on the ceiling");
    Check(onWalls > 1000 && wallsHidden == 0 && highestWall > 1.2f, "...and keeps every wall point up to 5 cm under it: the walls at full height");
    { std::error_code ignored; fs::remove_all(upFolder, ignored); }

    { std::error_code ignored; fs::remove_all(folder, ignored); }
    std::printf(failures ? "check_room_twin: %d FAILED\n" : "check_room_twin: all passed\n", failures);
    return failures ? 1 : 0;
}
