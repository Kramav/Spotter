// =====================================================================
//  09 · Kinect Find the Floor
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
//  Step 1 picks 3 points NEAR EACH OTHER in the image. Three points from
//  anywhere are almost never all on the floor (floor = 7% of the view ->
//  1 try in 3,000), but neighbours usually share a surface. Walls are
//  flat too, so step 1 also skips any plane that isn't roughly level. Finally, fit the plane to all its inliers (more accurate than
//  any 3 points).
//
//  From the plane we get the camera's HEIGHT above the floor, its TILT
//  (pitch) and ROLL, and the height above the floor of any point.
//
//  Window: depth (as in 01), the floor painted green.
//  Keys:   W A S D  move the crosshair (shows height above the floor)
//          Q or Esc  quit
//
//  Some floor must be in view: tilt the Kinect down a little.
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/registration.h>
#include <opencv2/opencv.hpp>

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
// noise tips the plane more. Try 1000 (anywhere in the image) to see why this matters.
const int NeighbourPixels = 40;

// A plane only counts as a floor if it's within this many degrees of level, as the camera sees it.
// So the camera itself may be tilted or rolled up to about this much.
const float MaxFloorSlopeDegrees = 45;

// A floor needs at least this share of the sampled points, or we report "no floor".
const double MinFloorFraction = 0.05;

const float MaxDepthMeters = 6.0f;   // 07b
const int NearMm = 500, FarMm = 4500, CrosshairStepPixels = 10;
const int Width = 512, Height = 424;

// A plane: normal . point + distance = 0. The normal points UP (towards the camera's side), so
// normal . point + distance = the point's height above the floor.
struct Plane
{
    cv::Vec3f normal;
    float distance;
    float HeightOf(const cv::Vec3f& point) const { return normal.dot(point) + distance; }
};

// The plane through three points; false if they're (nearly) in a line.
bool PlaneThrough(const cv::Vec3f& a, const cv::Vec3f& b, const cv::Vec3f& c, Plane& plane)
{
    cv::Vec3f normal = (b - a).cross(c - a);   // at right angles to both edges
    float length = (float)cv::norm(normal);
    if (length < 1e-6f) return false;
    plane.normal = normal / length;
    plane.distance = -plane.normal.dot(a);
    // Point the normal at the camera (at 0,0,0), which is above the floor.
    if (plane.distance < 0) { plane.normal = -plane.normal; plane.distance = -plane.distance; }
    return true;
}

// Least-squares plane through many points: the direction they spread least in is the normal.
Plane FitPlane(const std::vector<cv::Vec3f>& points)
{
    cv::Mat data((int)points.size(), 3, CV_32F, (void*)points.data());
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Vec3f mean(pca.mean.at<float>(0), pca.mean.at<float>(1), pca.mean.at<float>(2));
    Plane plane{ cv::Vec3f(pca.eigenvectors.row(2)), 0 };   // eigenvectors come largest first
    plane.distance = -plane.normal.dot(mean);
    if (plane.distance < 0) { plane.normal = -plane.normal; plane.distance = -plane.distance; }
    return plane;
}

int main()
{
    // -----------------------------------------------------------------
    //  Setup: depth only, plus registration for undistortion and XYZ
    // -----------------------------------------------------------------

    std::setvbuf(stdout, nullptr, _IONBF, 0);
    libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Warning));
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device* device = freenect2.enumerateDevices() > 0 ? freenect2.openDefaultDevice() : nullptr;
    if (device == nullptr)
    {
        std::printf("No Kinect v2 found, or it is in use. See the main README's troubleshooting table.\n");
        return 1;
    }
    libfreenect2::Freenect2Device::Config config;
    config.MaxDepth = MaxDepthMeters;
    device->setConfiguration(config);
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Depth);
    device->setIrAndDepthFrameListener(&listener);
    device->startStreams(false, true);
    libfreenect2::Registration registration(device->getIrCameraParams(), device->getColorCameraParams());
    libfreenect2::Frame undistorted(Width, Height, 4);

    std::mt19937 random(1);
    const int gridColumns = Width / SampleStep, gridRows = Height / SampleStep;
    std::vector<cv::Vec3f> cloud(Width * Height), grid(gridColumns * gridRows), samples, inliers;
    std::vector<int> validCells;   // grid cells with a reading
    const float minUpness = std::cos(MaxFloorSlopeDegrees * (float)CV_PI / 180);
    const cv::Vec3f cameraUp(0, -1, 0);   // the camera's Y axis points DOWN, so its "up" is -Y
    int crosshairX = Width / 2, crosshairY = Height * 3 / 4, frameCount = 0;

    const std::string window = "09 find the floor  (WASD crosshair, Q quits)";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, Width * 2, Height * 2);

    // -----------------------------------------------------------------
    //  Main operation
    // -----------------------------------------------------------------

    libfreenect2::FrameMap frames;
    while (true)
    {
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == 'w') crosshairY -= CrosshairStepPixels;
        if (key == 's') crosshairY += CrosshairStepPixels;
        if (key == 'a') crosshairX -= CrosshairStepPixels;
        if (key == 'd') crosshairX += CrosshairStepPixels;
        crosshairX = std::clamp(crosshairX, 0, Width - 1);
        crosshairY = std::clamp(crosshairY, 0, Height - 1);

        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }
        registration.undistortDepth(frames[libfreenect2::Frame::Depth], &undistorted);
        listener.release(frames);

        // --- Step 1: the point cloud (as in 08), and a sparse sample of it ---------
        samples.clear();
        validCells.clear();
        for (int row = 0; row < Height; row++)
        {
            for (int column = 0; column < Width; column++)
            {
                float x, y, z;
                registration.getPointXYZ(&undistorted, row, column, x, y, z);   // NaN = no reading
                cloud[row * Width + column] = cv::Vec3f(x, y, z);
                if (row % SampleStep != 0 || column % SampleStep != 0) continue;
                int cell = (row / SampleStep) * gridColumns + column / SampleStep;
                if (cell >= (int)grid.size()) continue;
                grid[cell] = cv::Vec3f(x, y, z);
                if (std::isnan(z)) continue;
                samples.emplace_back(x, y, z);
                validCells.push_back(cell);
            }
        }

        // --- Step 2: RANSAC - the level plane with the most points near it ----------
        Plane best{};
        int bestCount = 0;
        if (samples.size() >= 3)
        {
            std::uniform_int_distribution<size_t> pickCell(0, validCells.size() - 1);
            const int reach = NeighbourPixels / SampleStep;
            std::uniform_int_distribution<int> pickOffset(-reach, reach);
            for (int t = 0; t < RansacTries; t++)
            {
                // One random point, and two more within `reach` grid cells of it.
                int a = validCells[pickCell(random)];
                int row = a / gridColumns, column = a % gridColumns;
                int bRow = row + pickOffset(random), bColumn = column + pickOffset(random);
                int cRow = row + pickOffset(random), cColumn = column + pickOffset(random);
                if (std::min({ bRow, bColumn, cRow, cColumn }) < 0 || std::max(bRow, cRow) >= gridRows ||
                    std::max(bColumn, cColumn) >= gridColumns) continue;
                const cv::Vec3f& b = grid[bRow * gridColumns + bColumn];
                const cv::Vec3f& c = grid[cRow * gridColumns + cColumn];
                if (std::isnan(b[2]) || std::isnan(c[2])) continue;
                Plane plane;
                if (!PlaneThrough(grid[a], b, c, plane)) continue;
                if (plane.normal.dot(cameraUp) < minUpness) continue;   // too steep: a wall, not a floor
                int count = 0;
                for (const cv::Vec3f& p : samples)
                    if (std::abs(plane.HeightOf(p)) < InlierThicknessMeters) count++;
                if (count > bestCount) { bestCount = count; best = plane; }
            }
        }
        bool found = bestCount >= MinFloorFraction * samples.size() && bestCount >= 3;

        // --- Step 3: refine - fit the plane to ALL its inliers ---------------------
        if (found)
        {
            inliers.clear();
            for (const cv::Vec3f& p : samples)
                if (std::abs(best.HeightOf(p)) < InlierThicknessMeters) inliers.push_back(p);
            best = FitPlane(inliers);
        }

        // --- Step 4: what the plane says about the camera ---------------------------
        //  height: the camera is at (0,0,0), so its height is just the plane's distance.
        //  tilt:   how far the camera's forward axis (0,0,1) points below level.
        //  roll:   how far its right-hand side dips (+ = clockwise, seen from behind).
        float heightMeters = best.distance;
        float tiltDegrees = std::asin(-best.normal[2]) * 180 / (float)CV_PI;
        float rollDegrees = std::atan2(-best.normal[0], -best.normal[1]) * 180 / (float)CV_PI;
        if (++frameCount % 30 == 0)
        {
            if (found)
                std::printf("floor: camera %.3f m above it, tilted %.1f deg down, rolled %.1f deg  (%.0f%% of points)\n",
                            heightMeters, tiltDegrees, rollDegrees, 100.0 * bestCount / samples.size());
            else
                std::printf("no floor found: tilt the Kinect down so some floor is in view\n");
        }

        // -----------------------------------------------------------------
        //  Output: depth picture with the floor in green
        // -----------------------------------------------------------------

        cv::Mat picture(Height, Width, CV_8UC3);
        for (int i = 0; i < Width * Height; i++)
        {
            cv::Vec3b& pixel = picture.at<cv::Vec3b>(i / Width, i % Width);
            const cv::Vec3f& p = cloud[i];
            if (std::isnan(p[2])) { pixel = cv::Vec3b(0, 0, 255); continue; }
            uchar grey = (uchar)(255 * (1.0 - std::clamp((p[2] * 1000 - NearMm) / (double)(FarMm - NearMm), 0.0, 1.0)));
            pixel = found && std::abs(best.HeightOf(p)) < InlierThicknessMeters ? cv::Vec3b(0, grey, 0) : cv::Vec3b(grey, grey, grey);
        }
        cv::Scalar yellow(0, 255, 255);
        if (found)
            cv::putText(picture, cv::format("camera %.2f m up, tilt %.1f, roll %.1f", heightMeters, tiltDegrees, rollDegrees),
                        cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, yellow, 1);
        else
            cv::putText(picture, "no floor found", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

        const cv::Vec3f& aimed = cloud[crosshairY * Width + crosshairX];
        std::string text = std::isnan(aimed[2]) ? "no reading"
                         : found ? cv::format("%.2f m above floor", best.HeightOf(aimed))
                                 : cv::format("%.2f m away", aimed[2]);
        cv::drawMarker(picture, cv::Point(crosshairX, crosshairY), yellow, cv::MARKER_CROSS, 20, 1);
        cv::putText(picture, text, cv::Point(std::min(crosshairX + 8, Width - 150), std::max(crosshairY - 8, 40)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, yellow, 1);
        cv::imshow(window, picture);
    }

    device->stop();
    device->close();
    cv::destroyAllWindows();
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Measure the Kinect's height with a tape (to the middle of its lens). How close is the
//     printed height? Then put the crosshair on a table top: is its height right?
//  2. Set NeighbourPixels = 1000: the 3 points come from anywhere, and the floor flickers
//     between planes. (Built that way first, it found the floor in only some frames.)
//  3. Watch the numbers jitter from frame to frame. Set RansacTries = 20, then 2000. Which
//     settles them more: more tries, or the refit in step 3? (Comment the refit out to see.)
//  4. Tilt the Kinect slowly. Does the reported tilt follow? Does the green stay on the floor?
//  5. Set MaxFloorSlopeDegrees = 90 and point it at a big wall with a little floor in view.
//     RANSAC now picks the wall: the "level" check is what makes it a FLOOR finder.
//  6. A big table can win over a small patch of floor. How would you prefer the LOWEST level
//     plane instead of the biggest?
//  7. Set InlierThicknessMeters = 0.005, then 0.05. At 4 m the depth noise is about ±1 cm
//     (07b): what happens to the green far away with the thin setting?
