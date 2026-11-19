// spotter/room_twin.hpp: the scanned room for the presentation's room twin (Demo 7, 13v): loaded
// from a room folder (13e's), its floor and ceiling, and where the Kinect stands in it now.
// Coordinates: X right, Y up, Z towards the first camera (scan 1's camera at 0, 0, 0), as BuildRoom's.
#pragma once

#include "spotter/frame.hpp"
#include "spotter/geometry.hpp"
#include "spotter/room_folder.hpp"   // RoomInfo
#include "spotter/room_scan.hpp"     // Scan

#include <filesystem>
#include <string>
#include <vector>

namespace spotter
{

struct TwinRoom
{
    std::vector<cv::Point3f> points, normals;   // the merged surface's corners (FuseScans), and which way it faces there
    std::vector<cv::Vec3b> colours;
    cv::Matx33f firstCamera = cv::Matx33f::eye();   // scan 1's camera (X right, Y down, Z ahead) -> the room
    std::string from;                               // what it was made from, for the console
};

//std::filesystem::path NewestRoomFolder(const std::filesystem::path& root){
//
//@param1 - root: the folder to search (FindRooms, 3 folders deep)
//
//@brief - the room folder with the newest "saved" date in its room.txt, or empty if there's none.
//         O(rooms) after FindRooms' walk.
std::filesystem::path NewestRoomFolder(const std::filesystem::path& root);

//std::filesystem::path ChooseRoom(int argc, char** argv, const std::string& roomsFolder, const std::string& program){
//
//@param1 - argc: how many command-line words there are
//@param2 - argv: the words: an optional room folder
//@param3 - roomsFolder: where the Open window starts (the repo's room_scans/)
//@param4 - program: the program's name, for the window's title
//
//@brief - which room to open (D48): the folder given on the command line, else one chosen in a Select Folder window
//         (AskOpenFolder) that starts in roomsFolder with the newest room selected. A scan_NNN folder picked
//         stands for its room. A folder that isn't a room is refused with a message box (one holding rooms says
//         which) and the window opens again; so is a room.txt that can't be read. Empty if cancelled, with a line
//         saying so. O(FindRooms) plus the waits.
std::filesystem::path ChooseRoom(int argc, char** argv, const std::string& roomsFolder, const std::string& program);

//bool LoadTwinRoom(const std::filesystem::path& folder, float cubeMeters, TwinRoom& room, std::string& problem){
//
//@param1 - folder: a room folder (13e's)
//@param2 - cubeMeters: the merge's cube size (13 exported 3 cm)
//@param3 - room: filled with the merged, levelled surface
//@param4 - problem: why it failed, for the console
//
//@brief - every included scan merged into one surface (FuseScans, 13z's "seen_room"), each corner facing
//         the sum of its triangles' facings, turned by room.txt's first camera turn (BuildRoom's when the
//         folder isn't finished). False if nothing could be loaded. FuseScans dominates: O(scans x voxels).
bool LoadTwinRoom(const std::filesystem::path& folder, float cubeMeters, TwinRoom& room, std::string& problem);

// LoadTwinRoom in two halves, for a program that changes the scans in between (Demo 8 straightens them).
//bool LoadTwinScans(const std::filesystem::path& folder, RoomInfo& info, std::vector<Scan>& scans, std::string& problem){
//
//@param1 - folder: a room folder (13e's)
//@param2 - info: set to its room.txt
//@param3 - scans: set to every included scan, in order (no full colour pictures); a damaged one is skipped with a line
//@param4 - problem: why it failed, for the console
//
//@brief - LoadTwinRoom's first half. False if room.txt can't be read or no scan loads. O(scans x pixels).
bool LoadTwinScans(const std::filesystem::path& folder, RoomInfo& info, std::vector<Scan>& scans, std::string& problem);

//TwinRoom MergeTwinRoom(const std::vector<Scan>& scans, const RoomInfo& info, float cubeMeters, const std::string& name){
//
//@param1 - scans: the placed scans
//@param2 - info: their room.txt (its first camera turn, if finished; else BuildRoom's)
//@param3 - cubeMeters: the merge's cube size
//@param4 - name: the room's name, for `from`
//
//@brief - LoadTwinRoom's second half: FuseScans, each corner's facing, turned into the room. O(scans x voxels).
TwinRoom MergeTwinRoom(const std::vector<Scan>& scans, const RoomInfo& info, float cubeMeters, const std::string& name);

//TwinRoom OneViewRoom(const std::vector<cv::Point3f>& cloud, const cv::Mat& colours, int width, float levelDegrees){
//
//@param1 - cloud: an averaged depth cloud from the Kinect, one point per pixel (NaN = none)
//@param2 - colours: the registered colour, one per cloud point
//@param3 - width: points per row
//@param4 - levelDegrees: the Kinect's tilt, upwards positive, to stand the view level
//
//@brief - no room folder: the Kinect's own view as the room, good for rehearsing. Each pixel with clear
//         neighbours (no depth jump over 3%) becomes a point facing their cross product, levelled by the
//         tilt and flipped to Y up. O(pixels).
TwinRoom OneViewRoom(const std::vector<cv::Point3f>& cloud, const cv::Mat& colours, int width, float levelDegrees);

struct TwinFloor
{
    float y = 0;
    bool seen = false;
    std::vector<std::string> report;
};

//TwinFloor FindTwinFloor(const TwinRoom& room, float minAreaMeters2, float aboveFootMeters){
//
//@param1 - room: the levelled room; give it evenly spread points (e.g. one per 3 cm cube): the walls'
//          foot counts points per height
//@param2 - minAreaMeters2: a floor needs this much level area in one piece, seen from above in 5 cm cells
//          (default 0.25)
//@param3 - aboveFootMeters: how far above or below the walls' foot the floor may be (default 0.25)
//
//@brief - the floor (D33): the walls' foot is the lowest 2 cm step where upright points crowd as densely
//         as up a wall. Level points go into 2 cm height bins; each peak's biggest connected footprint is
//         measured. The floor is the lowest piece of minAreaMeters2 within aboveFootMeters of the foot,
//         above or below (bug 7: a reflection patch under the floor was taken), unless it's the biggest
//         level piece that isn't too high (D43: then the foot was measured too high, the walls' lower part
//         hidden). A floor nobody saw is guessed at the walls' foot. `report`: how it chose.
//         O(n + bins x n) for n points (one footprint per peak).
TwinFloor FindTwinFloor(const TwinRoom& room, float minAreaMeters2 = 0.25f, float aboveFootMeters = 0.25f);

// The ceiling (D58): found, so it can be shown or hidden without cutting the walls' tops off.
struct CeilingOptions
{
    float levelDegrees = 20;                 // a level reading faces within this of straight up or down
    float minAboveFloorMeters = 1.8f;
    float stepMeters = 0.02f;                // the height histogram's step
    float minAreaMeters2 = 0.5f;             // the ceiling: the HIGHEST level readings covering this much of the plan
    float areaCellMeters = 0.05f;            // area counted in cells this big, seen from below
    float guessAboveFloorMeters = 2.4f;      // TwinCeiling::y when no ceiling was seen
};

//bool FindCeiling(const std::vector<cv::Point3f>& points, const std::vector<cv::Point3f>& normals, float floorY, float& ceilingY, const CeilingOptions& options){
//
//@param1 - points: the room's points, levelled (Y up)
//@param2 - normals: which way each faces
//@param3 - floorY: the floor's height
//@param4 - ceilingY: set to the ceiling's height
//@param5 - options: the limits (defaults: CeilingOptions())
//
//@brief - the ceiling (13w's rule, D58 moved it here): the highest height (stepMeters steps, at least
//         minAboveFloorMeters above the floor) where level readings cover minAreaMeters2 of the plan (areaCellMeters
//         cells). A fan, a lamp or a beam hangs lower and covers less, so it never wins. False if none qualifies.
//         O(points).
bool FindCeiling(const std::vector<cv::Point3f>& points, const std::vector<cv::Point3f>& normals, float floorY, float& ceilingY,
                 const CeilingOptions& options = CeilingOptions());

struct TwinCeiling
{
    float y = 0;          // seen: its height; not seen: floor + guessAboveFloorMeters (for filling walls up to)
    bool seen = false;
    std::vector<std::string> report;
};

//TwinCeiling FindTwinCeiling(const TwinRoom& room, float floorY, const CeilingOptions& options){
//
//@param1 - room: the levelled room (evenly spread points, as FindTwinFloor's)
//@param2 - floorY: FindTwinFloor's floor
//@param3 - options: the limits (defaults: CeilingOptions())
//
//@brief - FindCeiling on the room, with a report line for the console. O(points).
TwinCeiling FindTwinCeiling(const TwinRoom& room, float floorY, const CeilingOptions& options = CeilingOptions());

//std::vector<char> CeilingPoints(const TwinRoom& room, const TwinCeiling& ceiling, float bandMeters, float levelDegrees){
//
//@param1 - room: the levelled room
//@param2 - ceiling: FindTwinCeiling's
//@param3 - bandMeters: how far under the ceiling a level reading still belongs to it (default 0.10)
//@param4 - levelDegrees: level = facing within this of straight down or up (default 20)
//
//@brief - 1 for each point that is the ceiling: level and within bandMeters under it (or anywhere above it), or
//         more than 5 cm above it whatever its facing (recesses, stray readings). Walls keep every upright point up
//         to the ceiling, and a fan hanging lower stays. All 0 if the ceiling wasn't seen. O(points).
std::vector<char> CeilingPoints(const TwinRoom& room, const TwinCeiling& ceiling, float bandMeters = 0.10f, float levelDegrees = 20);

// Where the Kinect is (D33): ICP from several starts around scan 1's spot.
struct LocateOptions
{
    std::vector<float> turnsDegrees{ 0, -10, 10, -20, 20, -30, 30 };
    float shiftMeters = 0.4f;
    float enoughShare = 0.95f;       // wrong fits reached 87% in the legacy test
    float startPairMeters = 0.5f;    // the Kinect may be put back 50 cm off
};
struct Located
{
    Pose pose;                // the Kinect -> the room
    float share = 0;
    int tried = 0;
    bool twoAlike = false;    // a distinct fit (0.3 m or 10 deg away) within 5% of the best: the room looks alike from both
    std::string report;
};

//Located LocateKinect(const std::vector<cv::Point3f>& view, const TwinRoom& room, const LocateOptions& options){
//
//@param1 - view: the Kinect's points now, in its own coordinates
//@param2 - room: the room to find it in
//@param3 - options: the starting turns and shifts, and when a start is good enough (defaults: LocateOptions())
//
//@brief - ICP on a quarter of the view from starts around scan 1's spot, each turned about up by
//         turnsDegrees and shifted by -1, 0, +1 times shiftMeters sideways and ahead (up to 63); a start
//         pairing enoughShare is taken at once, else the best. Then refined on the whole view. twoAlike
//         flags a distinct fit within 5% of the best; `report` lists the top 3. `share`: of the view,
//         paired within 2 cm. O(starts x rounds x n log m) (each Icp builds its own k-d tree).
Located LocateKinect(const std::vector<cv::Point3f>& view, const TwinRoom& room, const LocateOptions& options = LocateOptions());

}
