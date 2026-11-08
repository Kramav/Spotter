// =====================================================================
//  09 · step 2 of 3 · RANSAC: try many, keep the best
//
//  Step 1's random plane, 200 times per frame, scoring each by how many
//  points lie on it; the best one is painted green. That's RANSAC (RANdom
//  SAmple Consensus). To be quick it scores a thinned cloud: every 4th
//  pixel each way (spotter::ThinCloud). The winner is the BIGGEST flat
//  thing in view: often a wall, not the floor.
//
//  New in this step: RANSAC; a thinned grid of points.
//  Keys:  Q or Esc  quit
//  Next:  09_s3_Only_Level (make it a floor finder)
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

const int Tries = 200;
const float ThicknessMeters = 0.02f;
const int Reach = 10;   // the other two points come from within 10 grid cells (40 pixels)

spotter::Plane BestPlane(const spotter::PointGrid& grid, std::mt19937& random, int& bestCount)
{
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
        int count = 0;   // the score: points on it
        for (const cv::Point3f& p : grid.points) count += !std::isnan(p.z) && std::abs(plane.DistanceTo(p)) < ThicknessMeters;
        if (count > bestCount) { bestCount = count; best = plane; }
    }
    return best;
}

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
        spotter::Plane best = BestPlane(spotter::ThinCloud(cloud, 512, 4), random, count);
        cv::Mat picture = spotter::DepthPicture(frame.depthMm);
        for (int i = 0; i < (int)cloud.size(); i++)
            if (count > 0 && !std::isnan(cloud[i].z) && std::abs(best.DistanceTo(cloud[i])) < ThicknessMeters) picture.at<cv::Vec3b>(i) = cv::Vec3b(0, 255, 0);
        cv::putText(picture, cv::format("best of %d: %d sampled points on it", Tries, count), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        cv::imshow("09 step 2: RANSAC, the biggest plane", picture);
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: Tries = 5: it flickers between surfaces. 2000: steady, but slower. Where's the balance?
