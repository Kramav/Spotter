// spotter/room_twin.hpp: the scanned room for the presentation's room twin (Demos 4 and 6): loaded
// from a room folder (13e's), its floor, and where the Kinect stands in it now.
// Coordinates: X right, Y up, Z towards the first camera (scan 1's camera at 0, 0, 0), as BuildRoom's.
#pragma once

#include "spotter/frame.hpp"
#include "spotter/geometry.hpp"

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

// The newest room under `root` (FindRooms; the newest "saved" date), or empty.
std::filesystem::path NewestRoomFolder(const std::filesystem::path& root);
// Every included scan of a room folder merged into one surface at cubeMeters (13z's "seen_room";
// 13 exported 3 cm), turned by room.txt's first camera turn (BuildRoom's when the folder isn't finished).
bool LoadTwinRoom(const std::filesystem::path& folder, float cubeMeters, TwinRoom& room, std::string& problem);
// No room folder: the Kinect's own view (an averaged depth cloud and its colours) as the room, levelled
// by the Kinect's tilt (upwards positive). One view only: good for rehearsing.
TwinRoom OneViewRoom(const std::vector<cv::Point3f>& cloud, const cv::Mat& colours, int width, float levelDegrees);

// The floor (D33): the lowest level surface with minAreaMeters2 in one piece (seen from above, 5 cm
// cells), within aboveFootMeters of the walls' foot (the lowest height where upright points crowd as
// densely as up a wall), above or below (the review's bug 7: a reflection patch under the floor was
// taken), unless it's the biggest level piece that isn't too high (D43: then the foot was measured too
// high, the walls' lower part hidden). A floor nobody saw is guessed at the walls' foot. Give it evenly
// spread points (e.g. one per 3 cm cube): the foot counts points per height. `report`: how it chose.
struct TwinFloor
{
    float y = 0;
    bool seen = false;
    std::vector<std::string> report;
};
TwinFloor FindTwinFloor(const TwinRoom& room, float minAreaMeters2 = 0.25f, float aboveFootMeters = 0.25f);

// Where the Kinect is (D33): ICP from several starts around scan 1's spot, each turned about up by
// turnsDegrees and shifted by -1, 0, +1 times shiftMeters sideways and ahead, on a quarter of the
// view; a start matching enoughShare is taken at once, else the best; then refined on all of it.
// `view`: points in the Kinect's coordinates. `share`: of the view, within 2 cm of the room.
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
Located LocateKinect(const std::vector<cv::Point3f>& view, const TwinRoom& room, const LocateOptions& options = LocateOptions());

}
