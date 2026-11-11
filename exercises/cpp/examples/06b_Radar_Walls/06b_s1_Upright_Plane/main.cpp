// =====================================================================
//  06b · step 1 of 3 · The biggest upright plane
//
//  09_s3 kept only LEVEL planes to find the floor. Turn the test round:
//  keep only planes whose normal is nearly horizontal (|normal . up| small)
//  and RANSAC finds the biggest UPRIGHT plane: usually a wall. It's
//  painted green.
//
//  New in this step: an "upright" test on the normal.
//  Keys:  Q or Esc  quit
//  Next:  06b_s2_Several_Planes (the 2nd, 3rd... biggest too)
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
const float ThicknessMeters = 0.03f, MaxTiltDegrees = 20;

//spotter::Plane BestUprightPlane(const spotter::PointGrid& grid, std::mt19937& random, int& bestCount){
//
//@param1 - grid: a thinned depth cloud
//@param2 - random: the random number generator
//@param3 - bestCount: set to how many points lie on the best plane
//
//@brief - RANSAC as in 09 with one change: planes that aren't upright (tilted more than MaxTiltDegrees from
//         vertical) are skipped. Tries Tries random planes through three neighbouring cells and keeps the one with
//         most points within ThicknessMeters. O(Tries x cells).
spotter::Plane BestUprightPlane(const spotter::PointGrid& grid, std::mt19937& random, int& bestCount)
{
    const float maxUpness = std::sin(MaxTiltDegrees * (float)CV_PI / 180);
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
        if (std::abs(plane.normal.dot(cv::Point3f(0, -1, 0))) > maxUpness) continue;   // new: not upright
        int count = 0;
        for (const cv::Point3f& p : grid.points) count += !std::isnan(p.z) && std::abs(plane.DistanceTo(p)) < ThicknessMeters;
        if (count > bestCount) { bestCount = count; best = plane; }
    }
    return best;
}

//int main(){
//
//@brief - the program: 06b · step 1 of 3 · The biggest upright plane. 09_s3 kept only LEVEL planes to find the
//         floor. Turn the test round: keep only planes whose normal is nearly horizontal (|normal . up| small) and
//         RANSAC finds the biggest UPRIGHT plane: usually a wall. It's painted green. Setup, main loop and clean-
//         up are laid out in the file's sections. Returns 0 on a normal exit, 1 if it couldn't start.
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
        spotter::Plane wall = BestUprightPlane(spotter::ThinCloud(cloud, 512, 4), random, count);
        cv::Mat picture = spotter::DepthPicture(frame.depthMm);
        for (int i = 0; i < (int)cloud.size(); i++)
            if (count > 0 && !std::isnan(cloud[i].z) && std::abs(wall.DistanceTo(cloud[i])) < ThicknessMeters) picture.at<cv::Vec3b>(i) = cv::Vec3b(0, 255, 0);
        cv::putText(picture, count > 0 ? cv::format("upright plane %.2f m from the camera", wall.distance) : "no upright plane",
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        cv::imshow("06b step 1: the biggest upright plane", spotter::WithKeys(picture, Keys));
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: look at the green: besides the wall, thin strips light up wherever the plane slices through
// the floor or furniture. A plane goes on for ever. Step 2 keeps only the wall itself.
