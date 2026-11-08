// =====================================================================
//  00h · Tour: Scan, the room folder, FuseScans and BuildRoom   (no Kinect needed)
//
//  What they're for: 13e scans a room from many spots and saves each scan
//  the moment it's placed; 13z reads the folder back and rebuilds the room.
//    Scan              one spot: a 3D point per depth pixel, colour, features, pose
//    WriteScan         one scan's files, pose.txt last
//    FindRooms,        the reading side: every saved room under a folder,
//    LoadScan          one scan back (a scan cut off by a crash is refused)
//    FuseScans         every scan into one mesh (a grid of signed distances)
//    BuildRoom         floor, ceiling, walls and furniture boxes, levelled
//
//  How it works: a scan's pose says where its camera stood in the room
//  (scan 1's camera = the room's origin). pose.txt is written last, after
//  everything else is on the disk, and holds the cloud's point count: a scan
//  without it, or whose cloud.bin doesn't match, is skipped. Old rooms
//  (format 1, legacy 13c-13e) were mirrored left-right; they're read back
//  the right way round.
//
//  This program makes 3 scans of a made-up room with known poses (no
//  placing: tour 00f and check_room show that), writes them to a folder in
//  the system's temp folder, reads them back, and deletes it at the end.
// =====================================================================

#include <cstdio>
#include <fstream>

#include "spotter/room.hpp"
#include "spotter/room_folder.hpp"

namespace fs = std::filesystem;

const spotter::PinholeCamera Camera{ 365.5f, 365.5f, 256.0f, 212.0f, 512, 424 };

// A box room: x -1.5 to 2, ceiling 1.3 m above the first camera, floor 1.2 m below, z -1 to 3.
// Each wall has its own colour.
spotter::Scan Photograph(const spotter::Pose& pose)
{
    const float low[3] = { -1.5f, -1.3f, -1.0f }, high[3] = { 2.0f, 1.2f, 3.0f };
    const cv::Vec3b colours[6] = { { 60, 60, 200 }, { 60, 200, 60 }, { 230, 230, 230 }, { 90, 90, 90 }, { 200, 60, 60 }, { 60, 200, 200 } };
    spotter::Scan scan;
    scan.cloud.resize(512 * 424);
    scan.picture.create(424, 512, CV_8UC3);
    for (int row = 0; row < 424; row++)
        for (int column = 0; column < 512; column++)
        {
            cv::Point3f ray = Camera.Unproject(cv::Point2f((float)column, (float)row), 1.0f), d = pose.turn * ray;
            const float o[3] = { pose.shift.x, pose.shift.y, pose.shift.z }, dir[3] = { d.x, d.y, d.z };
            float best = 1e9f;
            int face = 0;
            for (int a = 0; a < 3; a++)
            {
                float t = dir[a] == 0 ? 1e9f : ((dir[a] > 0 ? high[a] : low[a]) - o[a]) / dir[a];
                if (t < best) { best = t; face = 2 * a + (dir[a] > 0); }
            }
            scan.cloud[row * 512 + column] = ray * best;
            scan.picture.at<cv::Vec3b>(row, column) = colours[face];
        }
    scan.surface = spotter::SampleSurface(scan.cloud, 512, 2);
    scan.pose = pose;
    return scan;
}

void WriteRoom(const fs::path& folder, const std::vector<spotter::Scan>& scans)
{
    std::printf("1. Writing %zu scans to %s\n", scans.size(), folder.string().c_str());
    spotter::RoomInfo info;
    info.program = "00h_Room_Folder";
    info.depthCamera = Camera;
    spotter::StartRoomFolder(folder, info);
    for (size_t k = 0; k < scans.size(); k++) spotter::WriteScan(folder, k, scans[k]);
    info.scans = (int)scans.size();
    spotter::WriteRoomInfo(folder, info);
    for (const auto& entry : fs::recursive_directory_iterator(folder))
        if (entry.is_regular_file() && entry.path().parent_path().filename() != "scan_002" && entry.path().parent_path().filename() != "scan_003")
            std::printf("   %-28s %9ju bytes\n", fs::relative(entry.path(), folder).string().c_str(), entry.file_size());
    std::printf("   (scan_002 and scan_003 hold the same files)\n");
}

std::vector<spotter::Scan> ReadRoom(const fs::path& folder, spotter::RoomInfo& info)
{
    std::printf("2. Reading it back:\n");
    for (const spotter::SavedRoom& room : spotter::FindRooms(folder))
        std::printf("   FindRooms: %s, format %d, %d scans, %.0f MB\n", room.folder.filename().string().c_str(), room.format, room.scans, room.megabytes);
    std::string problem;
    spotter::ReadRoomInfo(folder, info, problem);
    std::vector<spotter::Scan> scans;
    for (const fs::path& scanFolder : spotter::ScanFolders(folder))
    {
        spotter::Scan scan;
        int number;
        bool ok = spotter::LoadScan(scanFolder, info, scan, number, problem);
        std::printf("   scan %d: %s", number, ok ? "loaded" : ("refused: " + problem).c_str());
        if (ok) std::printf(", camera at (%+.2f, %+.2f, %+.2f), turned %.0f deg, %zu ICP points", scan.pose.shift.x, scan.pose.shift.y,
                            scan.pose.shift.z, scan.pose.Degrees(), scan.surface.points.size());
        std::printf("\n");
        if (ok) scans.push_back(std::move(scan));
    }
    return scans;
}

void Damage(const fs::path& folder, const spotter::RoomInfo& info)
{
    std::printf("3. What a blue screen can leave: scan 3's cloud.bin the right size, but all zeros:\n");
    std::vector<char> zeros(512 * 424 * sizeof(cv::Point3f), 0);
    std::ofstream(spotter::ScanFolder(folder, 2) / "cloud.bin", std::ios::binary).write(zeros.data(), zeros.size());
    spotter::Scan scan;
    int number;
    std::string problem;
    bool ok = spotter::LoadScan(spotter::ScanFolder(folder, 2), info, scan, number, problem);
    std::printf("   LoadScan: %s\n   (legacy 13z loaded it: 217,088 points at the camera.)\n", ok ? "loaded (WRONG)" : ("refused: " + problem).c_str());
    spotter::SetExcluded(spotter::ScanFolder(folder, 1), true);
    ok = spotter::LoadScan(spotter::ScanFolder(folder, 1), info, scan, number, problem);
    std::printf("   SetExcluded(scan 2): it now loads with excluded = %d (pose.txt rewritten in one go)\n", ok && scan.excluded);
}

void Rebuild(const std::vector<spotter::Scan>& scans)
{
    std::printf("4. FuseScans + BuildRoom on the scans that loaded:\n");
    std::vector<const spotter::Scan*> all;
    for (const spotter::Scan& s : scans) all.push_back(&s);
    std::mt19937 random(1);
    float voxel;
    spotter::RoomOptions options;
    spotter::Room room = spotter::BuildRoom(all, spotter::FuseScans(all, Camera, options, voxel), random, options);
    std::printf("   %zu triangles at %.0f cm; up from %s; floor %.3f m, ceiling %.3f m (true -1.2 and 1.3); %zu walls\n",
                room.mesh.groups["seen_room"].size(), voxel * 100, room.upFrom.c_str(), room.floorY, room.ceilingY, room.walls.size());
    for (const auto& [a, b] : room.walls) std::printf("   wall %.2f m long\n", cv::norm(b - a));
}

int main()
{
    const fs::path folder = fs::temp_directory_path() / "spotter_tour_00h";
    std::error_code error;
    fs::remove_all(folder, error);
    std::vector<spotter::Scan> scans;
    for (float left : { 0.0f, 35.0f, -35.0f })
        scans.push_back(Photograph({ spotter::TurnAbout(cv::Vec3d(0, -left * CV_PI / 180, 0)), cv::Point3f(0.2f * (scans.size() > 0), 0, 0.1f * scans.size()) }));
    WriteRoom(folder, scans);
    spotter::RoomInfo info;
    std::vector<spotter::Scan> back = ReadRoom(folder, info);
    Damage(folder, info);
    Rebuild(back);
    fs::remove_all(folder, error);
    std::printf("\nDeleted %s again. Press Enter to close.\n", folder.string().c_str());
    std::getchar();
    return 0;
}
