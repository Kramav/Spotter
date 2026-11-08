// =====================================================================
//  06b · step 2 of 3 · Several planes, each as one piece
//
//  Two additions to step 1:
//    - of the points on the plane, keep the biggest CONNECTED piece in the
//      picture (connectedComponentsWithStats): the wall, without the strips
//      the plane slices off other things;
//    - take that piece's points away and search again, up to 4 times, so
//      the 2nd, 3rd... biggest upright planes come out too.
//  Each piece gets its own colour.
//
//  New in this step: the biggest connected piece; repeated searches.
//  Keys:  Q or Esc  quit
//  Next:  06b_s3_Walls_From_Above (each piece as a line on a map)
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

const int Tries = 200, Reach = 10, MaxPlanes = 4, MinCells = 300;
const float ThicknessMeters = 0.03f, MaxTiltDegrees = 20;

// Step 1's RANSAC, over the cells still free.
bool BestUprightPlane(const spotter::PointGrid& grid, const std::vector<bool>& free, std::mt19937& random, spotter::Plane& best)
{
    const float maxUpness = std::sin(MaxTiltDegrees * (float)CV_PI / 180);
    std::uniform_int_distribution<int> anyCell(0, (int)grid.points.size() - 1), offset(-Reach, Reach);
    int bestCount = 0;
    for (int t = 0; t < Tries; t++)
    {
        int a = anyCell(random), row = a / grid.columns, column = a % grid.columns;
        int b = std::clamp(row + offset(random), 0, grid.rows - 1) * grid.columns + std::clamp(column + offset(random), 0, grid.columns - 1);
        int c = std::clamp(row + offset(random), 0, grid.rows - 1) * grid.columns + std::clamp(column + offset(random), 0, grid.columns - 1);
        spotter::Plane plane;
        if (!free[a] || !free[b] || !free[c] || !spotter::PlaneThrough(grid.points[a], grid.points[b], grid.points[c], plane)) continue;
        if (std::abs(plane.normal.dot(cv::Point3f(0, -1, 0))) > maxUpness) continue;
        int count = 0;
        for (size_t i = 0; i < grid.points.size(); i++) count += free[i] && std::abs(plane.DistanceTo(grid.points[i])) < ThicknessMeters;
        if (count > bestCount) { bestCount = count; best = plane; }
    }
    return bestCount >= MinCells;
}

// New: up to MaxPlanes searches; each keeps its plane's biggest connected piece and takes it away.
// Returns, per grid cell, which piece it belongs to (0 = none).
std::vector<int> Pieces(const spotter::PointGrid& grid, std::mt19937& random)
{
    std::vector<bool> free(grid.points.size());
    for (size_t i = 0; i < free.size(); i++) free[i] = !std::isnan(grid.points[i].z);
    std::vector<int> pieceOf(grid.points.size(), 0);
    for (int search = 1; search <= MaxPlanes; search++)
    {
        spotter::Plane plane;
        if (!BestUprightPlane(grid, free, random, plane)) break;
        cv::Mat onPlane(grid.rows, grid.columns, CV_8U, cv::Scalar(0)), labels, stats, centroids;
        for (size_t i = 0; i < free.size(); i++) if (free[i] && std::abs(plane.DistanceTo(grid.points[i])) < ThicknessMeters) onPlane.data[i] = 255;
        int count = cv::connectedComponentsWithStats(onPlane, labels, stats, centroids);
        int biggest = 1;
        for (int label = 2; label < count; label++)
            if (stats.at<int>(label, cv::CC_STAT_AREA) > stats.at<int>(biggest, cv::CC_STAT_AREA)) biggest = label;
        for (size_t i = 0; i < free.size(); i++)
        {
            if (!onPlane.data[i]) continue;
            free[i] = false;   // the whole plane leaves the search, so it isn't found again
            if (labels.ptr<int>()[i] == biggest) pieceOf[i] = search;
        }
    }
    return pieceOf;
}

int main()
{
    spotter::KinectSource::Options options;
    options.colour = false;
    spotter::KinectSource kinect(options);
    if (!kinect.IsOpen()) return 1;

    const cv::Scalar colours[] = { { 0, 0, 0 }, { 0, 255, 0 }, { 255, 128, 0 }, { 0, 128, 255 }, { 255, 0, 255 } };
    std::mt19937 random(1);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        spotter::PointGrid grid = spotter::ThinCloud(frame.depthCamera.Cloud(frame.depthMm), 512, 4);
        std::vector<int> pieceOf = Pieces(grid, random);
        cv::Mat picture = spotter::DepthPicture(frame.depthMm);
        for (int i = 0; i < (int)pieceOf.size(); i++)   // each grid cell stands for a 4 x 4 square of pixels
            if (pieceOf[i]) picture(cv::Rect((i % grid.columns) * 4, (i / grid.columns) * 4, 4, 4)).setTo(colours[pieceOf[i]]);
        cv::imshow("06b step 2: upright planes, one colour each", spotter::WithKeys(picture, Keys));
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: a couch back or a desk front shows up as a piece too. Step 3 sorts walls from furniture by
// how tall the piece is.
