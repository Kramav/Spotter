// spotter/room_folder.hpp: a scanned room on disk (13e writes it as it scans, 13z reads it back).
//
//   .gitignore         "*": git never takes anything in this folder
//   room.txt           format, program, cameras, scan count; after finishing: first_camera_turn, finished 1
//   scan_001\ ...      cloud.bin        x, y, z floats per depth pixel (NaN = none), the scan's camera
//                      colour.bmp       colour on the depth grid
//                      colour_full.bmp  the full colour picture
//                      colour_index.bin per depth pixel, the full picture's pixel it landed on (int, -1 = none)
//                      features.yml     near and far features, far rays and origin
//                      pose.txt         turn, shift, excluded, suspicious, stats, points (written LAST)
//   landmarks.txt      far landmarks: x y z (scan 1's camera), times seen, descriptor (hex)
//   points.ply         every included scan's points, levelled (written when the room is finished)
//
// Format 2 (this library): left and right as in the room. room.txt's depth_camera line is the
// camera the clouds were made with. Format 1 (legacy 13c-13e) was mirrored; LoadScan and
// LoadLandmarks mirror it back (x -> -x, a turn R -> S R S with S = diag(-1, 1, 1)).
// Fixed from 13e/13z (the review's bugs 3-5): every file is flushed to disk before pose.txt, which
// holds the cloud's point count (a zeroed cloud.bin after a blue screen is refused); pose.txt is
// rewritten atomically; folders sort by number (scan_1000 after scan_999).
#pragma once

#include "spotter/frame.hpp"
#include "spotter/room.hpp"
#include "spotter/room_scan.hpp"

#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace spotter
{

struct RoomInfo
{
    int format = 2;
    std::string program, saved;
    int scans = 0;
    PinholeCamera depthCamera;        // as the clouds were made (format 1: unmirrored on reading)
    cv::Size colourSize{ 1920, 1080 };
    std::string calibration;          // the device's factory numbers as text lines (KinectSource::CalibrationText)
    bool finished = false;            // points.ply and first_camera_turn are up to date
    cv::Matx33f firstCamera = cv::Matx33f::eye();
    std::string upFrom;
};

// --- Writing (13e) ---
std::filesystem::path ScanFolder(const std::filesystem::path& room, size_t index);   // index 0 -> scan_001
bool StartRoomFolder(const std::filesystem::path& room, const RoomInfo& info);       // the folder, .gitignore, room.txt
bool WriteRoomInfo(const std::filesystem::path& room, const RoomInfo& info);
bool WriteScan(const std::filesystem::path& room, size_t index, const Scan& scan);   // every file, pose.txt last
bool WritePose(const std::filesystem::path& room, size_t index, const Scan& scan);   // after X or P
bool RemoveScan(const std::filesystem::path& room, size_t index);                    // after U
bool WriteLandmarks(const std::filesystem::path& room, const std::vector<Landmark>& landmarks);
// points.ply (included scans, levelled by BuildRoom's planes) and room.txt with finished 1.
// `points` = how many went into points.ply.
bool FinishRoomFolder(const std::filesystem::path& room, RoomInfo& info, const std::vector<Scan>& scans, const RoomOptions& options,
                      size_t& points);

// --- Reading (13z) ---
struct SavedRoom
{
    std::filesystem::path folder;
    std::string program, saved;
    int format = 0, scans = 0, excluded = 0;
    double megabytes = 0;
};
// Every folder holding a room.txt, from `root` down to maxDepth folders (a room's own scan folders aren't searched).
std::vector<SavedRoom> FindRooms(const std::filesystem::path& root, int maxDepth = 3);
std::vector<std::filesystem::path> ScanFolders(const std::filesystem::path& room);   // sorted by number
bool ReadRoomInfo(const std::filesystem::path& room, RoomInfo& info, std::string& problem);
// One scan's cloud, colour picture, pose and flags (not its features or full picture), with its
// ICP points made by SampleSurface (the same rule as when it was scanned). number = 5 for scan_005.
// picture = false: colour.bmp isn't opened at all (numbers only; Scan::picture stays empty).
bool LoadScan(const std::filesystem::path& folder, const RoomInfo& info, Scan& scan, int& number, std::string& problem, bool picture = true);
bool SetExcluded(const std::filesystem::path& scanFolder, bool excluded);
std::vector<Landmark> LoadLandmarks(const std::filesystem::path& room, const RoomInfo& info);

// --- The console ---
// A line typed in the console, without spaces or quotes round it ("Copy as path" adds quotes). "cancel" at end of input.
std::string Ask(const std::string& question);
// Where a new room goes: a folder (Enter = defaultFolder) and a name (Enter = room_<date>_<time>).
// A saved room of that name is overwritten only after a yes; any other folder is never touched.
// Empty if the answer was cancel.
std::filesystem::path AskRoomFolder(const std::string& defaultFolder);

}
