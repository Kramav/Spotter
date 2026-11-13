// spotter/room_folder.hpp: a scanned room on disk (13e writes it on K, Q or the rescue; 13z reads it back).
// Programs keep rooms in the repo's room_scans/ (gitignored), outside build/, so a rebuild never deletes them.
// CheckRoomFolder refuses a folder that can't take a room (inside build/, relative, unwritable, no space: D48).
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

#include "spotter/file_dialog.hpp"   // Ask, AskSaveAs, Confirm, Tell
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

//std::filesystem::path ScanFolder(const std::filesystem::path& room, size_t index){
//
//@param1 - room: the room folder
//@param2 - index: the scan's index, from 0
//
//@brief - that scan's folder: index 0 -> room/scan_001. O(1).
std::filesystem::path ScanFolder(const std::filesystem::path& room, size_t index);

//bool StartRoomFolder(const std::filesystem::path& room, const RoomInfo& info){
//
//@param1 - room: the room folder to create
//@param2 - info: what goes in room.txt
//
//@brief - creates the folder, its .gitignore ("*") and room.txt (0 scans, not finished). False if it
//         fails. O(1).
bool StartRoomFolder(const std::filesystem::path& room, const RoomInfo& info);

//bool WriteRoomInfo(const std::filesystem::path& room, const RoomInfo& info){
//
//@param1 - room: the room folder
//@param2 - info: format, program, date, scan count, cameras, calibration, and once finished the first
//          camera turn
//
//@brief - (re)writes room.txt atomically, its "saved" date set to now. False if it fails. O(1).
bool WriteRoomInfo(const std::filesystem::path& room, const RoomInfo& info);

//bool WriteScan(const std::filesystem::path& room, size_t index, const Scan& scan){
//
//@param1 - room: the room folder
//@param2 - index: the scan's index, from 0
//@param3 - scan: the scan to write
//
//@brief - every file of the scan's folder (cloud.bin, colour.bmp, colour_full.bmp, colour_index.bin,
//         features.yml), each flushed to disk, then pose.txt last with the cloud's point count. Written as
//         scan_NNN.new and then swapped in, so an older save of this scan stays whole until the new one is
//         complete; ScanFolders reads a swap that was cut off. False if anything fails. O(pixels).
bool WriteScan(const std::filesystem::path& room, size_t index, const Scan& scan);

//bool WritePose(const std::filesystem::path& room, size_t index, const Scan& scan){
//
//@param1 - room: the room folder
//@param2 - index: the scan's index, from 0
//@param3 - scan: the scan whose pose and flags changed (after X or P)
//
//@brief - rewrites only that scan's pose.txt, atomically. False if it fails. O(1).
bool WritePose(const std::filesystem::path& room, size_t index, const Scan& scan);

//bool RemoveScansFrom(const std::filesystem::path& room, size_t index){
//
//@param1 - room: the room folder
//@param2 - index: the first scan index to delete, from 0
//
//@brief - deletes the folders of that scan and every later one (scans U took back), with any scan_NNN.new
//         or .old a cut-off save left. False if it fails. O(files in them).
bool RemoveScansFrom(const std::filesystem::path& room, size_t index);

//bool WriteLandmarks(const std::filesystem::path& room, const std::vector<Landmark>& landmarks){
//
//@param1 - room: the room folder
//@param2 - landmarks: the far landmarks
//
//@brief - writes landmarks.txt: x y z (scan 1's camera), times seen, descriptor as hex, one per line.
//         False if it fails. O(landmarks).
bool WriteLandmarks(const std::filesystem::path& room, const std::vector<Landmark>& landmarks);

//bool FinishRoomFolder(const std::filesystem::path& room, RoomInfo& info, const std::vector<Scan>& scans, const RoomOptions& options, size_t& points){
//
//@param1 - room: the room folder
//@param2 - info: updated with the first camera turn, "up" and finished = true, then written
//@param3 - scans: every scan (excluded ones are left out)
//@param4 - options: BuildRoom's settings, to level the room
//@param5 - points: set to how many points went into points.ply
//
//@brief - levels the included scans by BuildRoom's planes alone (no merged mesh), writes points.ply
//         (RoomPoints) through a .new file flushed and renamed, then room.txt with finished 1, last.
//         False if anything fails. BuildRoom's plane searches plus O(total pixels).
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

//std::vector<SavedRoom> FindRooms(const std::filesystem::path& root, int maxDepth){
//
//@param1 - root: the folder to search from
//@param2 - maxDepth: how many folders down to look (default 3)
//
//@brief - every folder holding a room.txt, with its program, date, format, scan and excluded counts and
//         size on disk; a room's own scan folders aren't searched. Sorted by folder. O(folders and files walked).
std::vector<SavedRoom> FindRooms(const std::filesystem::path& root, int maxDepth = 3);

//std::vector<std::filesystem::path> ScanFolders(const std::filesystem::path& room){
//
//@param1 - room: the room folder
//
//@brief - its scan_NNN folders, sorted by number (scan_1000 after scan_999). Where a save was cut off
//         mid-swap (WriteScan), the finished scan_NNN.new stands in, else scan_NNN.old. O(k log k) for k folders.
std::vector<std::filesystem::path> ScanFolders(const std::filesystem::path& room);

//bool ReadRoomInfo(const std::filesystem::path& room, RoomInfo& info, std::string& problem){
//
//@param1 - room: the room folder
//@param2 - info: filled from room.txt
//@param3 - problem: why it failed, for the console
//
//@brief - reads room.txt (format 1 cameras are unmirrored on reading). False if it's missing or broken, or its depth_size or
//         camera numbers are impossible (a damaged file). O(1).
bool ReadRoomInfo(const std::filesystem::path& room, RoomInfo& info, std::string& problem);

//bool LoadScan(const std::filesystem::path& folder, const RoomInfo& info, Scan& scan, int& number, std::string& problem, bool picture){
//
//@param1 - folder: a scan folder (scan_005)
//@param2 - info: the room's room.txt, for its format and camera
//@param3 - scan: filled with the cloud, colour picture, pose and flags
//@param4 - number: set to the scan's number (5 for scan_005)
//@param5 - problem: why it failed, for the console
//@param6 - picture: false = colour.bmp isn't opened at all (numbers only; Scan::picture stays empty) (default true)
//
//@brief - one scan's cloud, colour picture, pose and flags (not its features or full picture), with its
//         ICP points made by SampleSurface (the same rule as when it was scanned). Refuses a scan without
//         pose.txt, whose cloud.bin is the wrong size, or whose cloud doesn't hold pose.txt's point count.
//         Format 1 is mirrored back. O(pixels).
bool LoadScan(const std::filesystem::path& folder, const RoomInfo& info, Scan& scan, int& number, std::string& problem, bool picture = true);

//bool SetExcluded(const std::filesystem::path& scanFolder, bool excluded){
//
//@param1 - scanFolder: a scan folder
//@param2 - excluded: the new flag
//
//@brief - rewrites the "excluded" line of its pose.txt, atomically (bug 5). False if it fails. O(lines).
bool SetExcluded(const std::filesystem::path& scanFolder, bool excluded);

//std::vector<Landmark> LoadLandmarks(const std::filesystem::path& room, const RoomInfo& info){
//
//@param1 - room: the room folder
//@param2 - info: its room.txt, for the format
//
//@brief - reads landmarks.txt (format 1 mirrored back); empty if there's none. A broken line is skipped, and a
//         broken descriptor left out; it never throws. O(landmarks).
std::vector<Landmark> LoadLandmarks(const std::filesystem::path& room, const RoomInfo& info);

// --- Where a room goes (D48) ---

// What one scan takes on disk, with its share of points.ply and room for the .new copies a save makes. Measured
// on the 2026-11-13 room: 11.5 MB of scan files + 2.8 MB of points.ply per scan.
const uintmax_t RoomBytesPerScan = 16ull * 1024 * 1024;

struct FolderCheck
{
    bool ok = false;
    std::string problem;     // why not, in plain words, when !ok
    std::string warning;     // a yes-but (little free space), when ok
    uintmax_t freeBytes = 0;
    int scansThatFit = 0;    // freeBytes / RoomBytesPerScan
};

//FolderCheck CheckRoomFolder(const std::filesystem::path& room, int scansToWrite, bool quick){
//
//@param1 - room: the room folder a program is about to write
//@param2 - scansToWrite: how many scans the next save writes (default 1)
//@param3 - quick: a small test file instead of one scan's size, for checking often (default false)
//
//@brief - can a room be saved there? Refuses a relative path (it would land wherever the program runs), a folder
//         inside build/ (a clean rebuild deletes it), one inside another room folder (FindRooms wouldn't list it),
//         a file in the way, a path too long for Windows, a folder that can't be created, a test file that can't
//         be written, synced and read back (the system's reason: access denied, read-only, ...), and too little
//         free space (under scansToWrite + 5 scans). Warns under 60 scans' room. Folders it created are removed
//         again, so nothing is left on disk. O(1) plus writing the test file.
FolderCheck CheckRoomFolder(const std::filesystem::path& room, int scansToWrite = 1, bool quick = false);

//std::vector<std::filesystem::path> RescueFolders(){
//
//@brief - where a session's scans go, in this order, when its own folder can't be written at the end: the repo's
//         room_scans/rescue_<date>_<time>, then <home>/Documents/Spotter rescue/rescue_<date>_<time>, then the
//         system's temp folder. All absolute. O(1).
std::vector<std::filesystem::path> RescueFolders();

//std::filesystem::path AskRoomFolder(const std::string& defaultFolder, const std::filesystem::path& current, int scansToWrite){
//
//@param1 - defaultFolder: where the Save As window opens for a new room
//@param2 - current: the room folder already in use (13e's K), or empty at start-up (default empty)
//@param3 - scansToWrite: for CheckRoomFolder's free-space test (default 1)
//
//@brief - where a room goes, chosen in a Save As window (AskSaveAs; the name is the folder's, room_<date>_<time>
//         suggested). Then: a file in the way is refused; a saved room of that name is deleted and replaced only
//         after a yes; any other folder that isn't empty is left alone (choose again); `current` itself is never
//         a clash. CheckRoomFolder last; a refusal is shown (Tell) and the window opens again, and little free
//         space asks first. Empty if cancelled. O(1) plus the waits.
std::filesystem::path AskRoomFolder(const std::string& defaultFolder, const std::filesystem::path& current = {}, int scansToWrite = 1);

}
