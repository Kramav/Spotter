// =====================================================================
//  09 · Find the Floor
//
//  Example 08's points are in the CAMERA's coordinates: "down" is
//  wherever the camera's bottom edge points. Tilt the Kinect and the
//  whole room tilts with it. To make a map, we need the room's own
//  "down": the floor.
//
//  The floor is a big flat plane. RANSAC finds it in a messy cloud:
//    1. pick 3 points; they define a plane
//    2. count how many points lie within a few cm of it ("inliers")
//    3. repeat a few hundred times; keep the plane with the most inliers
//  Step 1 picks 3 points NEAR EACH OTHER in the picture. Three points from
//  anywhere are almost never all on the floor (floor = 7% of the view ->
//  1 try in 3,000), but neighbours usually share a surface. Walls are flat
//  too, so step 1 also skips any plane that isn't roughly level. Finally,
//  fit the plane to all its inliers (more accurate than any 3 points).
//
//  From the plane: the camera's HEIGHT above the floor, its TILT and ROLL,
//  and the height above the floor of any point.
//
//  Window: depth (as in 01), the floor painted green.
//  Keys:   W A S D  move the crosshair (shows height above the floor)
//          Q or Esc  quit
//  Some floor must be in view: tilt the Kinect down a little.
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/geometry.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "W A S D  move the crosshair (shows height above the floor)   Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// RANSAC: how many random planes to try, and how close (metres) a point must be to count as on it.
// More tries = more likely to find the best plane. Thicker = tolerates noise, but swallows rugs.
const int RansacTries = 200;
const float InlierThicknessMeters = 0.02f;

// Only every Nth pixel in each direction goes into RANSAC (4 -> 1/16 of the points, 16x faster).
const int SampleStep = 4;

// How far apart (in pixels) RANSAC's 3 points may be. Small = more likely on one surface, but
// noise tips the plane more. Try 1000 (anywhere in the picture) to see why this matters.
const int NeighbourPixels = 40;

// A plane only counts as a floor if it's within this many degrees of level, as the camera sees it.
const float MaxFloorSlopeDegrees = 45;

// A floor needs at least this share of the sampled points, or we report "no floor".
const double MinFloorFraction = 0.05;

const int CrosshairStepPixels = 10;
const std::string Window = "09 find the floor  (WASD crosshair, Q quits)";

// ---------------------------------------------------------------------
//  The new idea: RANSAC for one level plane
// ---------------------------------------------------------------------

struct Floor
{
    bool found = false;
    spotter::Plane plane;   // the normal points up, so DistanceTo(p) = p's height above the floor
    double share = 0;       // of the sampled points
};

//spotter::Plane Ransac(const spotter::PointGrid& grid, const std::vector<int>& valid, std::mt19937& random, int& bestCount){
//
//@param1 - grid: a thinned depth cloud
//@param2 - valid: the indices of the cells that have a reading
//@param3 - random: the random number generator
//@param4 - bestCount: set to how many points lie on the best plane
//
//@brief - steps 1-3: the level plane (within MaxFloorSlopeDegrees) with the most points near it. Each try takes
//         one random reading and two more within NeighbourPixels of it, since three random points from anywhere
//         almost never all hit a small floor. O(RansacTries x valid cells).
spotter::Plane Ransac(const spotter::PointGrid& grid, const std::vector<int>& valid, std::mt19937& random, int& bestCount)
{
    const cv::Point3f cameraUp(0, -1, 0);   // the camera's Y axis points DOWN
    const float minUpness = std::cos(MaxFloorSlopeDegrees * (float)CV_PI / 180);
    const int reach = NeighbourPixels / SampleStep;
    std::uniform_int_distribution<size_t> pickCell(0, valid.size() - 1);
    std::uniform_int_distribution<int> pickOffset(-reach, reach);
    spotter::Plane best;
    bestCount = 0;
    for (int t = 0; t < RansacTries; t++)
    {
        // One random point, and two more within `reach` grid cells of it.
        int a = valid[pickCell(random)], row = a / grid.columns, column = a % grid.columns;
        int bRow = row + pickOffset(random), bColumn = column + pickOffset(random);
        int cRow = row + pickOffset(random), cColumn = column + pickOffset(random);
        if (std::min({ bRow, bColumn, cRow, cColumn }) < 0 || std::max(bRow, cRow) >= grid.rows || std::max(bColumn, cColumn) >= grid.columns) continue;
        const cv::Point3f& b = grid.points[bRow * grid.columns + bColumn];
        const cv::Point3f& c = grid.points[cRow * grid.columns + cColumn];
        if (std::isnan(b.z) || std::isnan(c.z)) continue;
        spotter::Plane plane;
        if (!spotter::PlaneThrough(grid.points[a], b, c, plane)) continue;
        if (plane.normal.dot(cameraUp) < minUpness) continue;   // too steep: a wall, not a floor
        int count = 0;
        for (int i : valid) count += std::abs(plane.DistanceTo(grid.points[i])) < InlierThicknessMeters;
        if (count > bestCount) { bestCount = count; best = plane; }
    }
    return best;
}

//Floor FindFloor(const std::vector<cv::Point3f>& cloud, std::mt19937& random){
//
//@param1 - cloud: a depth cloud, 512 wide
//@param2 - random: the random number generator
//
//@brief - thins the cloud, runs Ransac, and calls it a floor if the plane holds at least MinFloorFraction of the
//         readings; step 4 refits the plane to ALL its inliers. Returns the floor, its share and whether it was
//         found. O(RansacTries x cells).
Floor FindFloor(const std::vector<cv::Point3f>& cloud, std::mt19937& random)
{
    spotter::PointGrid grid = spotter::ThinCloud(cloud, 512, SampleStep);
    std::vector<int> valid;
    for (int i = 0; i < (int)grid.points.size(); i++) if (!std::isnan(grid.points[i].z)) valid.push_back(i);
    Floor floor;
    if (valid.size() < 3) return floor;
    int count;
    spotter::Plane best = Ransac(grid, valid, random, count);
    floor.found = count >= 3 && count >= MinFloorFraction * valid.size();
    floor.share = (double)count / valid.size();
    if (!floor.found) return floor;

    // Step 4: refine. Fit the plane to ALL its inliers.
    std::vector<cv::Point3f> inliers;
    for (int i : valid) if (std::abs(best.DistanceTo(grid.points[i])) < InlierThicknessMeters) inliers.push_back(grid.points[i]);
    floor.plane = spotter::FitPlane(inliers);
    return floor;
}

//std::string CameraPose(const Floor& floor){
//
//@param1 - floor: a found floor
//
//@brief - what the plane says about the camera (which sits at 0, 0, 0): height = the plane's distance; tilt = how
//         far the forward axis (0, 0, 1) points below level; roll = how far its right-hand side dips (+ =
//         clockwise, seen from behind). Returned as text. O(1).
std::string CameraPose(const Floor& floor)
{
    const cv::Point3f& n = floor.plane.normal;
    return cv::format("camera %.2f m up, tilt %.1f deg down, roll %.1f deg", floor.plane.distance,
                      std::asin(-n.z) * 180 / CV_PI, std::atan2(-n.x, -n.y) * 180 / CV_PI);
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

//void Show(const cv::Mat& depthMm, const std::vector<cv::Point3f>& cloud, const Floor& floor, cv::Point crosshair){
//
//@param1 - depthMm: the depth picture, mm
//@param2 - cloud: its 3D points
//@param3 - floor: the floor found (or not)
//@param4 - crosshair: the pixel to read
//
//@brief - the depth picture with the floor's pixels green, the camera's pose text (or "no floor found"), and a
//         cross at the crosshair with its height above the floor (or its distance when there is no floor); shown
//         with the key strip. O(pixels).
void Show(const cv::Mat& depthMm, const std::vector<cv::Point3f>& cloud, const Floor& floor, cv::Point crosshair)
{
    cv::Mat picture = spotter::DepthPicture(depthMm);
    for (int i = 0; i < (int)cloud.size(); i++)
        if (floor.found && !std::isnan(cloud[i].z) && std::abs(floor.plane.DistanceTo(cloud[i])) < InlierThicknessMeters)
            picture.at<cv::Vec3b>(i) = cv::Vec3b(0, picture.at<cv::Vec3b>(i)[1], 0);   // green, as bright as the grey was
    const cv::Scalar yellow(0, 255, 255);
    cv::putText(picture, floor.found ? CameraPose(floor) : "no floor found", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                floor.found ? yellow : cv::Scalar(0, 0, 255), 1);
    const cv::Point3f& aimed = cloud[crosshair.y * 512 + crosshair.x];
    std::string text = std::isnan(aimed.z) ? "no reading" : floor.found ? cv::format("%.2f m above floor", floor.plane.DistanceTo(aimed))
                                                                       : cv::format("%.2f m away", aimed.z);
    cv::drawMarker(picture, crosshair, yellow, cv::MARKER_CROSS, 20, 1);
    cv::putText(picture, text, cv::Point(std::min(crosshair.x + 8, 362), std::max(crosshair.y - 8, 40)), cv::FONT_HERSHEY_SIMPLEX, 0.4, yellow, 1);
    cv::imshow(Window, spotter::WithKeys(picture, Keys));
}

//void MoveCrosshair(int key, cv::Point& crosshair){
//
//@param1 - key: what cv::waitKey returned
//@param2 - crosshair: the crosshair's pixel; moved in place
//
//@brief - W A S D move the crosshair CrosshairStepPixels, kept inside the 512 x 424 picture. O(1).
void MoveCrosshair(int key, cv::Point& crosshair)
{
    if (key == 'w') crosshair.y -= CrosshairStepPixels;
    if (key == 's') crosshair.y += CrosshairStepPixels;
    if (key == 'a') crosshair.x -= CrosshairStepPixels;
    if (key == 'd') crosshair.x += CrosshairStepPixels;
    crosshair.x = std::clamp(crosshair.x, 0, 511);
    crosshair.y = std::clamp(crosshair.y, 0, 423);
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

//int main(){
//
//@brief - the program: 09 · Find the Floor. Example 08's points are in the CAMERA's coordinates: "down" is
//         wherever the camera's bottom edge points. Tilt the Kinect and the whole room tilts with it. To make a
//         map, we need the room's own "down": the floor. Setup, main loop and clean-up are laid out in the file's
//         sections. Returns 0 on a normal exit, 1 if it couldn't start.
int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    spotter::KinectSource::Options kinectOptions;
    kinectOptions.colour = false;
    spotter::KinectSource kinect(kinectOptions);
    if (!kinect.IsOpen()) return 1;

    cv::namedWindow(Window, cv::WINDOW_NORMAL);
    cv::resizeWindow(Window, 1024, 848);
    std::mt19937 random(1);
    cv::Point crosshair(256, 212);
    spotter::Every report(1000);
    spotter::RgbdFrame frame;

    while (kinect.Next(frame))
    {
        std::vector<cv::Point3f> cloud = frame.depthCamera.Cloud(frame.depthMm);   // as in 08
        Floor floor = FindFloor(cloud, random);
        Show(frame.depthMm, cloud, floor, crosshair);
        if (report.Due())
            std::printf(floor.found ? "floor: %s (%.0f%% of points)\n" : "no floor found: tilt the Kinect down so some floor is in view\n",
                        CameraPose(floor).c_str(), 100 * floor.share);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        MoveCrosshair(key, crosshair);
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Measure the Kinect's height (to the middle of its lens) with something of known size. How
//     close is the printed height? Then put the crosshair on a table top: is its height right?
//  2. Set NeighbourPixels = 1000: the 3 points come from anywhere, and the floor flickers.
//  3. Watch the numbers jitter. Set RansacTries = 20, then 2000. Which settles them more: more
//     tries, or the refit in step 4? (Skip the refit to see.)
//  4. Tilt the Kinect slowly. Does the reported tilt follow? Does the green stay on the floor?
//  5. Set MaxFloorSlopeDegrees = 90 and point it at a big wall with a little floor in view.
//     RANSAC now picks the wall: the "level" check is what makes it a FLOOR finder.
//  6. A big table can win over a small patch of floor. How would you prefer the LOWEST level
//     plane instead of the biggest?
//  7. Set InlierThicknessMeters = 0.005, then 0.05. At 4 m the depth noise is about +-1 cm (07):
//     what happens to the green far away with the thin setting?
