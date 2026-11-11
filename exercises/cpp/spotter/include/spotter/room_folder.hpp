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
//         features.yml), each flushed to disk, then pose.txt last with the cloud's point count, so a
//         scan cut off half-way is skipped on reading. False if anything fails. O(pixels).
bool WriteScan(const std::filesystem::path& room, size_t index, const Scan& scan);

//bool WritePose(const std::filesystem::path& room, size_t index, const Scan& scan){
//
//@param1 - room: the room folder
//@param2 - index: the scan's index, from 0
//@param3 - scan: the scan whose pose and flags changed (after X or P)
//
//@brief - rewrites only that scan's pose.txt, atomically. False if it fails. O(1).
bool WritePose(const std::filesystem::path& room, size_t index, const Scan& scan);

//bool RemoveScan(const std::filesystem::path& room, size_t index){
//
//@param1 - room: the room folder
//@param2 - index: the scan's index, from 0
//
//@brief - deletes that scan's folder (after U). Later scans keep their numbers. False if it fails.
//         O(files in it).
bool RemoveScan(const std::filesystem::path& room, size_t index);

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
//@brief - its scan_NNN folders, sorted by number (scan_1000 after scan_999). O(k log k) for k folders.
std::vector<std::filesystem::path> ScanFolders(const std::filesystem::path& room);

//bool ReadRoomInfo(const std::filesystem::path& room, RoomInfo& info, std::string& problem){
//
//@param1 - room: the room folder
//@param2 - info: filled from room.txt
//@param3 - problem: why it failed, for the console
//
//@brief - reads room.txt (format 1 cameras are unmirrored on reading). False if it's missing or broken. O(1).
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
//         pose.txt or whose cloud doesn't hold pose.txt's point count. Format 1 is mirrored back. O(pixels).
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
//@brief - reads landmarks.txt (format 1 mirrored back); empty if there's none. O(landmarks).
std::vector<Landmark> LoadLandmarks(const std::filesystem::path& room, const RoomInfo& info);

// --- The console ---

//std::string Ask(const std::string& question){
//
//@param1 - question: printed before waiting
//
//@brief - a line typed in the console, without spaces or quotes round it ("Copy as path" adds quotes).
//         "cancel" at the end of input. O(answer length).
std::string Ask(const std::string& question);

//std::filesystem::path AskRoomFolder(const std::string& defaultFolder){
//
//@param1 - defaultFolder: the folder used when Enter is pressed
//
//@brief - where a new room goes: asks for a folder (Enter = defaultFolder) and a name (Enter =
//         room_<date>_<time>). A saved room of that name is overwritten only after a yes; any other
//         existing folder is never touched. Empty if the answer was cancel.
std::filesystem::path AskRoomFolder(const std::string& defaultFolder);

}
