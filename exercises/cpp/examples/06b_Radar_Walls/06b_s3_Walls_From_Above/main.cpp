// =====================================================================
//  06b · step 3 of 3 · Walls seen from above
//
//  Step 2's pieces now come from the library (spotter::FindPlanes, upright).
//  Seen from above, an upright plane is a line on the floor:
//    - drop the normal's up/down part: what's left points from the wall to
//      the camera, across the floor;
//    - "along" is that direction turned 90 degrees;
//    - each point's position along the wall; the 2nd and 98th percentiles
//      are the ends (so stray points don't stretch it).
//  Pieces less than 0.8 m tall are furniture, not walls. Drawn on 06's radar
//  (spotter::RadarView), no objects yet.
//
//  New in this step: a plane -> a line on the floor plan; the height rule.
//  Keys:  Q or Esc  quit
//  Next:  06b_Radar_Walls (the full example adds the tracked objects)
// =====================================================================

#include <cmath>
#include <random>
#include <opencv2/highgui.hpp>

#include "spotter/geometry.hpp"
#include "spotter/radar.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/stats.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "Q or Esc  quit";

const float MinWallHeightMeters = 0.8f;

spotter::PlaneSearch UprightPlanes()
{
    spotter::PlaneSearch search;
    search.facing = spotter::PlaneSearch::Facing::Upright;
    search.maxTiltDegrees = 20;
    search.maxPlanes = 4;
    search.minShare = 0.05;
    return search;
}

// One piece -> a line on the floor, if it's tall enough to be a wall.
bool WallLine(const spotter::PlanePiece& piece, const spotter::PointGrid& grid, cv::Point2d& a, cv::Point2d& b)
{
    const cv::Point3f& n = piece.plane.normal;
    double across = std::hypot(n.x, n.z);
    cv::Point2d towards(n.x / across, n.z / across), along(-towards.y, towards.x);
    cv::Point2d foot = -towards * (piece.plane.distance / across);   // the wall's nearest point to the camera
    std::vector<float> alongs, heights;
    for (int i : piece.cells)
    {
        alongs.push_back((float)(along.x * grid.points[i].x + along.y * grid.points[i].z));
        heights.push_back(grid.points[i].y);
    }
    a = foot + along * spotter::Percentile(alongs, 0.02);
    b = foot + along * spotter::Percentile(alongs, 0.98);
    return spotter::Percentile(heights, 0.98) - spotter::Percentile(heights, 0.02) >= MinWallHeightMeters;
}

int main()
{
    spotter::KinectSource kinect(spotter::KinectSource::Options{});   // colour too: the radar shows its field of view
    if (!kinect.IsOpen()) return 1;

    spotter::RadarView radar;
    std::mt19937 random(1);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        spotter::PointGrid grid = spotter::ThinCloud(frame.depthCamera.Cloud(frame.depthMm), 512, 4);
        radar.Begin(frame.colourCamera);
        for (const spotter::PlanePiece& piece : spotter::FindPlanes(grid, UprightPlanes(), random))
        {
            cv::Point2d a, b;
            bool wall = WallLine(piece, grid, a, b);
            radar.DrawLine(a, b, wall ? cv::Scalar(190, 190, 190) : cv::Scalar(0, 80, 160), wall ? 3 : 1,
                           wall ? cv::format("wall %.2f m", piece.plane.distance) : "too short");
        }
        cv::imshow("06b step 3: walls from above", spotter::WithKeys(radar.Image(), Keys));
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: walk along a wall holding the camera level, then tilt it down 30 degrees: the wall facing
// it disappears, because "upright" is measured against the camera's own up.
