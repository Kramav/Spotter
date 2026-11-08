// spotter/room.hpp: a room from many placed scans (examples 13e, 13z). FuseScans merges them into one
// mesh (a TSDF grid and a surface net); BuildRoom finds the floor, ceiling, walls and furniture
// boxes (09c across every scan) and stands the room up straight.
#pragma once

#include "spotter/frame.hpp"
#include "spotter/geometry.hpp"
#include "spotter/mesh.hpp"
#include "spotter/room_scan.hpp"

#include <random>
#include <string>
#include <utility>
#include <vector>

namespace spotter
{

// 13e's settings, as its defaults.
struct RoomOptions
{
    float voxelMeters = 0.02f;       // the merge's cube size; grows by itself past maxVoxels
    long long maxVoxels = 80000000;  // about 24 bytes each: up to 1.9 GB
    int truncationVoxels = 3;        // how far in front of / behind a surface a scan updates the grid
    int sampleStep = 4;              // planes are searched on every 4th pixel of each scan
    PlaneSearch planes;              // facing any way (09c)
    float maxTiltDegrees = 25;
    float levelDegrees = 10;         // once "up" is known, floor/ceiling/walls must be this close to level/upright
    float minWallHeightMeters = 0.8f;
    float minCeilingHeightMeters = 1.8f;
    float cornerSnapMeters = 0.5f;
    int minBlockCells = 30;
    float maxBlockMeters = 3.0f;     // furniture boxes wider than this are wall pieces, and joins stop there
    float mergeDegrees = 10;         // the same plane seen from two spots: this close in angle...
    float mergeMeters = 0.05f;       // ...and distance
};

// Every scan into one grid of little cubes holding a signed distance to the surface (+ = air in
// front of it), averaged over the scans that saw each cube; then a surface net where the sign
// changes. One group, "seen_room". `voxel` = the cube size it really used.
Mesh FuseScans(const std::vector<const Scan*>& scans, const PinholeCamera& depthCamera, const RoomOptions& options, float& voxel);

// The tidy model. Coordinates: X right, Y up, Z towards the first camera (first camera at 0, 0, 0).
struct Room
{
    Mesh mesh;                 // `seen` (levelled), plus groups clean_walls, clean_floor, clean_ceiling, furniture_blocks
    std::string upFrom;        // what "up" came from: the floor, the ceiling, two walls, or the first camera
    float floorY = 0, ceilingY = 0;
    bool floorSeen = false, ceilingSeen = false;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> walls;          // ends, as (x, z)
    std::vector<std::pair<cv::Point3f, cv::Point3f>> blocks;         // lowest and highest corners
    cv::Matx33f firstCamera = cv::Matx33f::eye();                    // scan 1's camera (X right, Y down, Z ahead) -> room
};
Room BuildRoom(const std::vector<const Scan*>& scans, const Mesh& seen, std::mt19937& random, const RoomOptions& options);

// The room as an .obj: far landmarks (scan 1's camera coordinates) added as 10 cm magenta boxes in
// group far_landmarks; a "# spotter format 2" line (not mirrored) and the first camera's turn
// (for presentation Demo 4) as comments.
bool SaveRoomObj(const std::string& path, Room room, const std::vector<cv::Point3f>& landmarks, const std::string& comment);

// Every scan's own points and colours, turned by `turn` (e.g. Room::firstCamera), for a .ply.
void RoomPoints(const std::vector<const Scan*>& scans, const cv::Matx33f& turn, std::vector<cv::Point3f>& points, std::vector<cv::Vec3b>& colours);

}
