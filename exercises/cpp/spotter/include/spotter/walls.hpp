// spotter/walls.hpp: walls found in depth alone (example 06b), as lines on the floor seen from
// above. Assumes the camera is roughly level: "upright" means upright for the CAMERA.
#pragma once

#include "spotter/geometry.hpp"

namespace spotter
{

// A wall seen from above: a line on the floor, X (right) and Z (ahead) in metres.
struct Wall
{
    cv::Point2d ends[2];
    double meters;         // the camera's distance from the wall's plane (at right angles to it)
    double degrees;        // 0 = facing the camera squarely, + = its right end is farther away
    double heightMeters;   // how much of its height is in view
};

struct WallSearch
{
    PlaneSearch planes = Upright();
    float minHeightMeters = 0.8f;   // a piece shorter than this is a couch back or a cupboard, not a wall

    static PlaneSearch Upright()
    {
        PlaneSearch search;
        search.facing = PlaneSearch::Facing::Upright;
        search.maxTiltDegrees = 20;
        search.maxPlanes = 4;          // the camera rarely sees more than 3 walls
        search.minShare = 0.05;
        return search;
    }
};

// The biggest upright planes, each turned into a line from its leftmost to its rightmost point
// (2nd and 98th percentiles, so stray points don't stretch it).
std::vector<Wall> FindWalls(const PointGrid& grid, const WallSearch& search, std::mt19937& random);

}
