// Self-check for room_scan.hpp, room.hpp and room_folder.hpp, with no Kinect: a made-up room with a
// patterned surface (blocks of colour), raycast from 5 camera spots.
//   1. MakeScan + PlaceScan place scans 2-5 among the earlier ones, close to their true poses.
//   2. FuseScans + BuildRoom: the floor 1.2 m below, the ceiling 1.3 m above the first camera, the
//      walls, the couch as one box; the merged mesh lies on the true surfaces.
//   3. The room folder: written and read back unchanged; SetExcluded; a zeroed cloud.bin refused
//      (bug 4); folders sorted by number (bug 5); a format-1 (mirrored) scan comes back unmirrored;
//      FinishRoomFolder writes points.ply and finished 1.
// Run: build\check_room.exe. Exit code 0 = passed. Writes only in the system's temp folder.

#include "spotter/room.hpp"
#include "spotter/room_folder.hpp"
#include "spotter/stats.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

int failures = 0;
void Check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAILED: %s\n", what); failures++; }
}

const float Low[3] = { -1.5f, -1.3f, -1.0f }, High[3] = { 2.0f, 1.2f, 3.0f };            // the room (Y down)
const float CouchLow[3] = { 0.2f, 0.6f, 2.0f }, CouchHigh[3] = { 1.0f, 1.2f, 2.5f };     // 0.6 m tall, on the floor
const spotter::PinholeCamera Camera{ 365.5f, 365.5f, 256.0f, 212.0f, 512, 424 };

// The nearest surface along a ray: distance, and the colour of the 10 cm block it hits.
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

// What a depth camera at `pose` (its coordinates -> the room's) sees: cloud and colour picture.
spotter::Scan Photograph(const spotter::Pose& pose, int seed, const spotter::ScanOptions& options)
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
            cloud[row * 512 + column] = ray * (t * (1 + (float)noise.gaussian(0.0005)));   // about 1 mm at 2 m (07's average)
            picture.at<cv::Vec3b>(row, column) = colour;
        }
    spotter::Scan scan = spotter::MakeScan(cloud, picture, options);
    scan.colourIndex.assign(512 * 424, -1);
    return scan;
}

spotter::Pose Spot(float right, float down, float forward, float leftDegrees, float downDegrees)
{
    return { spotter::TurnAbout(cv::Vec3d(0, -leftDegrees * CV_PI / 180, 0)) * spotter::TurnAbout(cv::Vec3d(-downDegrees * CV_PI / 180, 0, 0)),
             cv::Point3f(right, down, forward) };
}

bool SameCloud(const std::vector<cv::Point3f>& a, const std::vector<cv::Point3f>& b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (std::isnan(a[i].z) != std::isnan(b[i].z) || (!std::isnan(a[i].z) && cv::norm(a[i] - b[i]) > 1e-6)) return false;
    return true;
}

int main()
{
    // 1. Scans placed one by one.
    const spotter::ScanOptions options;
    const std::vector<spotter::Pose> truth = { Spot(0, 0, 0, 0, 0), Spot(0.25f, 0, 0.1f, 15, 0), Spot(0.5f, 0, 0.2f, 30, 0),
                                               Spot(0.3f, -0.05f, -0.2f, -15, 5), Spot(0, 0, 0.3f, -40, 0) };
    std::vector<spotter::Scan> scans;
    std::mt19937 random(1);
    for (size_t k = 0; k < truth.size(); k++)
    {
        spotter::Scan scan = Photograph(truth[k], (int)k + 1, options);
        if (k > 0)
        {
            auto start = std::chrono::steady_clock::now();
            spotter::Placement placed = spotter::PlaceScan(scan, scans, Camera, random, -1, {}, options);
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            double mm = cv::norm(placed.pose.shift - truth[k].shift) * 1000;
            double degrees = spotter::Pose{ placed.pose.turn.t() * truth[k].turn, cv::Point3f() }.Degrees();
            std::printf("1. scan %zu: %zu features; %s, %.1f mm / %.3f deg off, %.2f s%s\n", k + 1, scan.near.keypoints.size(),
                        placed.placed ? "placed" : "NOT placed", mm, degrees, seconds, placed.suspicious ? " (suspicious)" : "");
            if (!placed.placed) std::printf("   %s\n", placed.report.c_str());
            Check(placed.placed && mm < 5 && degrees < 0.2, "PlaceScan puts each scan within 5 mm and 0.2 deg");
            Check(!placed.suspicious, "and doesn't call a good placement suspicious");
            scan.pose = placed.pose;
        }
        scans.push_back(std::move(scan));
    }
    spotter::Recommendation next = spotter::RecommendNext(scans, Camera, options);
    std::printf("   %s\n", next.text.c_str());
    bool atASpot = false;
    for (const spotter::Scan& s : scans) atASpot = atASpot || cv::norm(next.pose.shift - s.pose.shift) < 1e-6;
    Check(next.found && next.text.rfind("Next: at spot ", 0) == 0 && atASpot, "RecommendNext suggests a new view from a scan's spot");
    Check(!spotter::RecommendNext({}, Camera, options).found, "and nothing before the first scan");

    // 2. The merged mesh and the clean model.
    std::vector<const spotter::Scan*> all;
    for (const spotter::Scan& s : scans) all.push_back(&s);
    spotter::RoomOptions roomOptions;
    float voxel;
    auto start = std::chrono::steady_clock::now();
    spotter::Mesh seen = spotter::FuseScans(all, Camera, roomOptions, voxel);
    spotter::Room room = spotter::BuildRoom(all, seen, random, roomOptions);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::vector<float> off;   // each merged point's distance to the nearest true wall, floor or ceiling (couch left out)
    for (size_t i = 0; i < seen.points.size(); i++)
    {
        cv::Point3f p = room.firstCamera.t() * room.mesh.points[i];   // back into scan 1's camera
        const float q[3] = { p.x, p.y, p.z };
        bool nearCouch = true;
        for (int a = 0; a < 3; a++) nearCouch = nearCouch && q[a] > CouchLow[a] - 0.05f && q[a] < CouchHigh[a] + 0.05f;
        if (nearCouch) continue;
        float d = 1e9f;
        for (int a = 0; a < 3; a++) d = std::min({ d, std::abs(q[a] - Low[a]), std::abs(q[a] - High[a]) });
        off.push_back(d);
    }
    std::printf("2. merged: %zu triangles, %.0f cm cubes, %.1f mm median / %.1f mm 95%% off the true surfaces; %.1f s\n",
                room.mesh.groups["seen_room"].size(), voxel * 100, spotter::Median(off) * 1000, spotter::Percentile(off, 0.95) * 1000, seconds);
    std::printf("   up from %s; floor %.4f, ceiling %.4f; %zu walls, %zu blocks\n", room.upFrom.c_str(), room.floorY, room.ceilingY,
                room.walls.size(), room.blocks.size());
    Check(spotter::Median(off) < 0.003f, "the merged mesh lies within 3 mm (median) of the true surfaces");
    Check(room.floorSeen && std::abs(room.floorY + 1.2f) < 0.01f, "the floor 1.2 m below the first camera");
    Check(room.ceilingSeen && std::abs(room.ceilingY - 1.3f) < 0.01f, "the ceiling 1.3 m above it");
    Check(room.walls.size() >= 3, "at least the left, back and right walls");
    bool couch = false;
    for (const auto& [lo, hi] : room.blocks)
    {
        float wide = hi.x - lo.x, deep = hi.z - lo.z;
        std::printf("   block %.2f x %.2f x %.2f m\n", wide, deep, hi.y - lo.y);
        couch = couch || (std::abs(hi.y - lo.y - 0.6f) < 0.05f && std::abs(std::max(wide, deep) - 0.8f) < 0.08f && std::abs(std::min(wide, deep) - 0.5f) < 0.08f);
    }
    Check(couch, "the couch as one 0.8 x 0.5 x 0.6 m box");

    // 3. The room folder.
    const fs::path folder = fs::temp_directory_path() / "spotter_check_room";
    std::error_code error;
    fs::remove_all(folder, error);
    spotter::RoomInfo info;
    info.program = "check_room";
    info.depthCamera = Camera;
    info.calibration = "ir_params 365.5 365.5 256 212 0 0 0 0 0\n";
    bool ok = spotter::StartRoomFolder(folder, info);
    for (size_t k = 0; k < scans.size(); k++) ok = spotter::WriteScan(folder, k, scans[k]) && ok;
    info.scans = (int)scans.size();
    ok = spotter::WriteRoomInfo(folder, info) && spotter::WriteLandmarks(folder, {}) && ok;
    Check(ok, "the folder is written");
    std::vector<spotter::SavedRoom> found = spotter::FindRooms(folder.parent_path() / "spotter_check_room");
    Check(found.size() == 1 && found[0].scans == 5 && found[0].format == 2, "FindRooms lists it: format 2, 5 scans");
    spotter::RoomInfo read;
    std::string problem;
    Check(spotter::ReadRoomInfo(folder, read, problem) && read.format == 2 && read.depthCamera.cx == Camera.cx, "room.txt reads back");
    std::vector<fs::path> folders = spotter::ScanFolders(folder);
    bool same = folders.size() == 5;
    for (size_t k = 0; k < folders.size() && same; k++)
    {
        spotter::Scan back;
        int number;
        same = spotter::LoadScan(folders[k], read, back, number, problem) && number == (int)k + 1 && SameCloud(back.cloud, scans[k].cloud) &&
               cv::norm(back.pose.shift - scans[k].pose.shift) < 1e-5 && cv::norm(cv::Mat(back.pose.turn), cv::Mat(scans[k].pose.turn)) < 1e-5 &&
               back.surface.points.size() == scans[k].surface.points.size();
    }
    Check(same, "every scan loads back unchanged (cloud, pose, the same ICP points)");
    spotter::Scan back;
    int number;
    Check(spotter::SetExcluded(folders[1], true) && spotter::LoadScan(folders[1], read, back, number, problem) && back.excluded,
          "SetExcluded is saved");
    {
        std::vector<char> zeros(512 * 424 * sizeof(cv::Point3f), 0);
        std::ofstream(folders[2] / "cloud.bin", std::ios::binary).write(zeros.data(), zeros.size());
    }
    Check(!spotter::LoadScan(folders[2], read, back, number, problem), "a zeroed cloud.bin is refused (bug 4)");
    std::printf("3. zeroed cloud.bin: \"%s\"\n", problem.c_str());
    fs::create_directories(folder / "scan_1000", error);
    fs::create_directories(folder / "scan_999", error);
    folders = spotter::ScanFolders(folder);
    Check(folders.size() == 7 && folders[5].filename() == "scan_999" && folders[6].filename() == "scan_1000", "folders sort by number (bug 5)");
    fs::remove_all(folder / "scan_1000", error);
    fs::remove_all(folder / "scan_999", error);

    // A second save of a scan is swapped in whole; one cut off part-way leaves a whole scan to read.
    const fs::path four = spotter::ScanFolder(folder, 3), fresh = fs::path(four) += ".new", older = fs::path(four) += ".old";
    auto readsAs = [&](const fs::path& expected, const spotter::Scan& scan) {
        std::vector<fs::path> now = spotter::ScanFolders(folder);
        spotter::Scan got;
        int n;
        return now.size() == 5 && now[3] == expected && spotter::LoadScan(now[3], read, got, n, problem) && n == 4 && SameCloud(got.cloud, scan.cloud);
    };
    Check(spotter::WriteScan(folder, 3, scans[3]) && readsAs(four, scans[3]) && !fs::exists(fresh) && !fs::exists(older),
          "saving a scan again swaps it in whole, leaving no .new or .old");
    fs::copy(four, fresh, fs::copy_options::recursive, error);
    fs::remove(fresh / "pose.txt", error);
    Check(readsAs(four, scans[3]), "cut off while writing the new copy: the old save is read, the unfinished .new ignored");
    fs::copy(fresh, folder / "scan_006.new", fs::copy_options::recursive, error);
    Check(readsAs(four, scans[3]), "a new scan's first save cut off (an unfinished .new, nothing else): not a scan");
    fs::remove_all(folder / "scan_006.new", error);
    fs::remove_all(fresh, error);
    fs::copy(four, fresh, fs::copy_options::recursive, error);
    fs::rename(four, older, error);
    Check(readsAs(fresh, scans[3]), "cut off between the swap's renames: the finished .new is read");
    fs::remove_all(fresh, error);
    Check(readsAs(older, scans[3]), "only the .old left: it is read");
    Check(spotter::WriteScan(folder, 3, scans[3]) && readsAs(four, scans[3]) && !fs::exists(older), "the next save tidies up");
    fs::create_directories(folder / "scan_005.old", error);
    fs::create_directories(folder / "scan_007x", error);
    Check(spotter::RemoveScansFrom(folder, 4) && spotter::ScanFolders(folder).size() == 4 && !fs::exists(folder / "scan_005.old") &&
          fs::exists(folder / "scan_007x"), "RemoveScansFrom deletes scan 5 and its leftovers, and nothing that isn't a scan");
    fs::remove_all(folder / "scan_007x", error);

    // Damaged files are refused with a reason, never a crash.
    const fs::path broken = fs::temp_directory_path() / "spotter_check_room_broken";
    fs::remove_all(broken, error);
    fs::copy(folder, broken, fs::copy_options::recursive, error);
    spotter::RoomInfo bad;
    std::ofstream(broken / "room.txt") << "format 2\ndepth_size -512 424\ndepth_camera 365.5 365.5 256 212\n";
    Check(!spotter::ReadRoomInfo(broken, bad, problem), "an impossible depth_size is refused");
    std::printf("3. damaged room.txt: \"%s\"\n", problem.c_str());
    std::ofstream(broken / "scan_001" / "cloud.bin", std::ios::binary | std::ios::app) << "extra";
    Check(!spotter::LoadScan(broken / "scan_001", read, back, number, problem), "a cloud.bin that's too long is refused");
    std::ofstream(broken / "landmarks.txt") << "0.1 0.2 3.0 2 zz\n0.4 0.5 6.0 3 0aff\n";
    std::vector<spotter::Landmark> marks = spotter::LoadLandmarks(broken, read);
    Check(marks.size() == 2 && marks[0].descriptor.empty() && marks[1].descriptor.cols == 2 && marks[1].descriptor.at<uchar>(0, 1) == 0xff,
          "a landmark line with a broken descriptor loads without one, and doesn't throw");
    fs::remove_all(broken, error);

    // A format-1 scan: what legacy 13e wrote for the same view (mirrored), must load as scan 1 is.
    const fs::path old = fs::temp_directory_path() / "spotter_check_room_format1";
    fs::remove_all(old, error);
    fs::create_directories(old / "scan_001", error);
    std::ofstream(old / "room.txt") << "format 1\nprogram 13e_Kinect_Room_Scan_Autosave\ndepth_size 512 424\nir_params 365.5 365.5 256 212 0 0 0 0 0\n";
    const spotter::Scan& one = scans[1];
    std::vector<cv::Point3f> mirrored(one.cloud.size());
    for (int row = 0; row < 424; row++)
        for (int column = 0; column < 512; column++)
        {
            cv::Point3f p = one.cloud[row * 512 + column];
            mirrored[row * 512 + 511 - column] = std::isnan(p.z) ? p : cv::Point3f(-p.x, p.y, p.z);
        }
    std::ofstream(old / "scan_001" / "cloud.bin", std::ios::binary).write((const char*)mirrored.data(), mirrored.size() * sizeof(cv::Point3f));
    cv::Mat flipped;
    cv::flip(one.picture, flipped, 1);
    cv::imwrite((old / "scan_001" / "colour.bmp").string(), flipped);
    const cv::Matx33f S(-1, 0, 0, 0, 1, 0, 0, 0, 1), r = S * one.pose.turn * S;
    std::ofstream(old / "scan_001" / "pose.txt") << cv::format("turn %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f\nshift %.6f %.6f %.6f\nexcluded 0\n",
                                                               r(0, 0), r(0, 1), r(0, 2), r(1, 0), r(1, 1), r(1, 2), r(2, 0), r(2, 1), r(2, 2),
                                                               -one.pose.shift.x, one.pose.shift.y, one.pose.shift.z);
    spotter::RoomInfo oldInfo;
    bool unmirrored = spotter::ReadRoomInfo(old, oldInfo, problem) && oldInfo.format == 1 && oldInfo.depthCamera.cx == 512 - 256.0f &&
                      spotter::LoadScan(old / "scan_001", oldInfo, back, number, problem);
    unmirrored = unmirrored && SameCloud(back.cloud, one.cloud) && cv::norm(back.pose.shift - one.pose.shift) < 1e-5 &&
                 cv::norm(cv::Mat(back.pose.turn), cv::Mat(one.pose.turn)) < 1e-5 && cv::norm(back.picture, one.picture, cv::NORM_INF) == 0;
    Check(unmirrored, "a format-1 scan loads unmirrored: cloud, picture and pose as the format-2 one");
    fs::remove_all(old, error);

    size_t points = 0;
    scans.erase(scans.begin() + 2);   // the zeroed one
    Check(spotter::FinishRoomFolder(folder, info, scans, roomOptions, points) && fs::exists(folder / "points.ply") &&
          spotter::ReadRoomInfo(folder, read, problem) && read.finished, "FinishRoomFolder writes points.ply and finished 1");
    std::printf("   points.ply: %zu points\n", points);
    fs::remove_all(folder, error);

    std::printf(failures ? "check_room: %d FAILED\n" : "check_room: all passed\n", failures);
    return failures ? 1 : 0;
}
