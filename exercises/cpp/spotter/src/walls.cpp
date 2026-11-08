#include "spotter/walls.hpp"
#include "spotter/stats.hpp"

#include <cmath>

namespace spotter
{

std::vector<Wall> FindWalls(const PointGrid& grid, const WallSearch& search, std::mt19937& random)
{
    std::vector<Wall> walls;
    for (const PlanePiece& piece : FindPlanes(grid, search.planes, random))
    {
        const Plane& plane = piece.plane;
        // Drop the normal's up/down part: what's left points from the wall to the camera, across
        // the floor. "along" runs along the wall; "foot" is the wall's nearest point to the camera.
        // ponytail: drawn on the camera's own level, so a camera tilted 10 deg draws the walls 1.5%
        // too far; a floor plane (09) would give the true level.
        double across = std::hypot(plane.normal.x, plane.normal.z);
        cv::Point2d normal(plane.normal.x / across, plane.normal.z / across), along(-normal.y, normal.x);
        cv::Point2d foot = -normal * (plane.distance / across);

        std::vector<float> alongs, heights;
        for (int i : piece.cells)
        {
            const cv::Point3f& p = grid.points[i];
            alongs.push_back((float)(along.x * p.x + along.y * p.z));
            heights.push_back(p.y);
        }
        float height = Percentile(heights, 0.98) - Percentile(heights, 0.02);
        if (height < search.minHeightMeters) continue;
        walls.push_back({ { foot + along * Percentile(alongs, 0.02), foot + along * Percentile(alongs, 0.98) },
                          plane.distance, std::atan2(normal.x, -normal.y) * 180 / CV_PI, height });
    }
    return walls;
}

}
