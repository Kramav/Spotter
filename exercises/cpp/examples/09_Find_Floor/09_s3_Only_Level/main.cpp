// =====================================================================
//  09 · step 3 of 3 · Only level planes: a floor finder
//
//  Step 2 found the biggest flat thing. A floor is the biggest flat thing
//  that is roughly LEVEL. The camera's "up" is -Y (its Y axis points down),
//  so a plane is level when its normal points nearly along (0, -1, 0):
//  normal . up > cos(45 deg). Skip every other plane. The floor's distance
//  is now the camera's height above it.
//
//  New in this step: a direction test on the plane's normal.
//  Keys:  Q or Esc  quit
//  Next:  09_Find_Floor (the full example refits the plane, adds tilt, roll and a crosshair)
// =====================================================================

#include <algorithm>
#include <cmath>
#include <random>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/geometry.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "Q or Esc  quit";

const int Tries = 200, Reach = 10;
const float ThicknessMeters = 0.02f, MaxSlopeDegrees = 45;

//spotter::Plane BestLevelPlane(const spotter::PointGrid& grid, std::mt19937& random, int& bestCount){
//
//@param1 - grid: a thinned depth cloud
//@param2 - random: the random number generator
//@param3 - bestCount: set to how many points lie on the best plane
//
//@brief - RANSAC as in step 2 with one change: planes tilted more than MaxSlopeDegrees from level (their normal
//         must point nearly up) are skipped, so a wall can't win. O(Tries x cells).
spotter::Plane BestLevelPlane(const spotter::PointGrid& grid, std::mt19937& random, int& bestCount)
{
    const cv::Point3f up(0, -1, 0);
    const float minUpness = std::cos(MaxSlopeDegrees * (float)CV_PI / 180);
    std::uniform_int_distribution<int> anyCell(0, (int)grid.points.size() - 1), offset(-Reach, Reach);
    spotter::Plane best;
    bestCount = 0;
    for (int t = 0; t < Tries; t++)
    {
        int a = anyCell(random), row = a / grid.columns, column = a % grid.columns;
        int b = std::clamp(row + offset(random), 0, grid.rows - 1) * grid.columns + std::clamp(column + offset(random), 0, grid.columns - 1);
        int c = std::clamp(row + offset(random), 0, grid.rows - 1) * grid.columns + std::clamp(column + offset(random), 0, grid.columns - 1);
        spotter::Plane plane;
        if (std::isnan(grid.points[a].z) || std::isnan(grid.points[b].z) || std::isnan(grid.points[c].z) ||
            !spotter::PlaneThrough(grid.points[a], grid.points[b], grid.points[c], plane)) continue;
        if (plane.normal.dot(up) < minUpness) continue;   // new: too steep, a wall
        int count = 0;
        for (const cv::Point3f& p : grid.points) count += !std::isnan(p.z) && std::abs(plane.DistanceTo(p)) < ThicknessMeters;
        if (count > bestCount) { bestCount = count; best = plane; }
    }
    return best;
}

//int main(){
//
//@brief - the program: 09 · step 3 of 3 · Only level planes: a floor finder. Step 2 found the biggest flat thing.
//         A floor is the biggest flat thing that is roughly LEVEL. The camera's "up" is -Y (its Y axis points
//         down), so a plane is level when its normal points nearly along (0, -1, 0): normal . up > cos(45 deg).
//         Setup, main loop and clean-up are laid out in the file's sections. Returns 0 on a normal exit, 1 if it
//         couldn't start.
int main()
{
    spotter::KinectSource::Options options;
    options.colour = false;
    spotter::KinectSource kinect(options);
    if (!kinect.IsOpen()) return 1;

    std::mt19937 random(1);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        std::vector<cv::Point3f> cloud = frame.depthCamera.Cloud(frame.depthMm);
        int count;
        spotter::Plane floor = BestLevelPlane(spotter::ThinCloud(cloud, 512, 4), random, count);
        cv::Mat picture = spotter::DepthPicture(frame.depthMm);
        for (int i = 0; i < (int)cloud.size(); i++)
            if (count > 0 && !std::isnan(cloud[i].z) && std::abs(floor.DistanceTo(cloud[i])) < ThicknessMeters) picture.at<cv::Vec3b>(i) = cv::Vec3b(0, 255, 0);
        cv::putText(picture, count > 0 ? cv::format("floor: camera %.2f m above it", floor.distance) : "no level plane",
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        cv::imshow("09 step 3: the biggest LEVEL plane", spotter::WithKeys(picture, Keys));
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: point it with no floor in view: it finds a table top (level, and below the camera). Why
// never the ceiling, which is level too? A plane's normal always points to the camera's side, so
// the ceiling's points DOWN, and fails the test.
