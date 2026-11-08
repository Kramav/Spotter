// =====================================================================
//  06b · Radar + Walls
//
//  Example 06 puts the objects on a map, but the map is empty space: no
//  room around them. This one adds the WALLS, found in the depth alone
//  (no neural network). A wall is a big, flat, UPRIGHT surface:
//
//    1-7.  Detect, track, deproject, radar      as in 06 (library: Tracker, RadarView)
//    8.    Points     a thinned grid of 3D points from the depth
//    9.    RANSAC     find the plane with the most points near it, but
//                     skip planes that AREN'T upright (09 skips the ones
//                     that aren't level).
//    10.   Repeat     keep the plane's biggest connected piece (a plane also
//                     slices strips off the floor and ceiling), refit it,
//                     take its points away and search again, so the 2nd,
//                     3rd... biggest upright planes come out too. A piece
//                     that isn't tall enough is a couch back or a cupboard
//                     front, not a wall: skipped.
//    11.   Draw       seen from above, a wall is a line: from its leftmost
//                     to its rightmost point.
//
//  Assumes the camera is roughly level: "upright" means upright for the
//  CAMERA. Tilt it down 30 degrees and the wall facing it stops counting.
//  (spotter::FindWalls is this same search, for the later examples.)
//
//  Windows: the camera view, and the radar with the walls as grey lines.
//  Console: each object's position, and each wall's distance, length and angle.
//  Keys:   C  clear the trails      Q or Esc  quit
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
#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/geometry.hpp"
#include "spotter/radar.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/stats.hpp"
#include "spotter/track.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "C  clear the trails      Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

const int MaxWalls = 4;                     // the camera rarely sees more than 3 walls
const int RansacTries = 200;                // random planes per search (09)
const float InlierThicknessMeters = 0.03f;  // how close a point must be to count as on the plane
const int SampleStep = 4;                   // every 4th depth pixel each way (1/16 of them)
const int NeighbourPixels = 40;             // RANSAC's three points lie this close together (09)
const float MaxWallTiltDegrees = 20;        // "upright": at most this far from vertical, as the camera sees it
const double MinWallShare = 0.05;           // a wall needs at least this share of the points
const float MinWallHeightMeters = 0.8f;     // shorter = furniture

const int ReportEveryMs = 1000;
const std::string CameraWindow = "06b camera  (C clears trails, Q quits)", RadarWindow = "06b radar";

spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = 0.15f;
    return options;
}

// ---------------------------------------------------------------------
//  The new idea: walls from depth alone
// ---------------------------------------------------------------------

struct Wall
{
    cv::Point2d ends[2];   // on the floor, X right and Z ahead, metres
    double meters;         // the camera's distance from the wall (at right angles to it)
    double degrees;        // 0 = facing the camera squarely, + = its right end is farther away
    double heightMeters;   // how much of its height is in view
};

// Step 9: RANSAC as in 09, over the cells still free, for upright planes only.
bool BestUprightPlane(const spotter::PointGrid& grid, const std::vector<bool>& free, const std::vector<int>& freeCells,
                      std::mt19937& random, spotter::Plane& best, int& bestCount)
{
    const int reach = NeighbourPixels / SampleStep;
    const float maxUpness = std::sin(MaxWallTiltDegrees * (float)CV_PI / 180);
    std::uniform_int_distribution<size_t> pickCell(0, freeCells.size() - 1);
    std::uniform_int_distribution<int> pickOffset(-reach, reach);
    bestCount = 0;
    for (int t = 0; t < RansacTries; t++)
    {
        int a = freeCells[pickCell(random)], row = a / grid.columns, column = a % grid.columns;
        int bRow = row + pickOffset(random), bColumn = column + pickOffset(random);
        int cRow = row + pickOffset(random), cColumn = column + pickOffset(random);
        if (std::min({ bRow, bColumn, cRow, cColumn }) < 0 || std::max(bRow, cRow) >= grid.rows || std::max(bColumn, cColumn) >= grid.columns) continue;
        int b = bRow * grid.columns + bColumn, c = cRow * grid.columns + cColumn;
        spotter::Plane plane;
        if (!free[b] || !free[c] || !spotter::PlaneThrough(grid.points[a], grid.points[b], grid.points[c], plane)) continue;
        if (std::abs(plane.normal.dot(cv::Point3f(0, -1, 0))) > maxUpness) continue;   // floor, table top or ceiling
        int count = 0;
        for (int i : freeCells) count += std::abs(plane.DistanceTo(grid.points[i])) < InlierThicknessMeters;
        if (count > bestCount) { bestCount = count; best = plane; }
    }
    return bestCount > 0;
}

// Step 11: the wall as a line on the floor. Drop the normal's up/down part: what's left points
// from the wall to the camera, across the floor. "foot" is the wall's nearest point to the camera.
// ponytail: drawn on the camera's own level, so a camera tilted 10 deg draws walls 1.5% too far.
bool WallFrom(const spotter::Plane& plane, const std::vector<cv::Point3f>& inliers, Wall& wall)
{
    double across = std::hypot(plane.normal.x, plane.normal.z);
    cv::Point2d normal(plane.normal.x / across, plane.normal.z / across), along(-normal.y, normal.x);
    cv::Point2d foot = -normal * (plane.distance / across);
    std::vector<float> alongs, heights;
    for (const cv::Point3f& p : inliers)
    {
        alongs.push_back((float)(along.x * p.x + along.y * p.z));
        heights.push_back(p.y);
    }
    wall.heightMeters = spotter::Percentile(heights, 0.98) - spotter::Percentile(heights, 0.02);   // percentiles ignore stray points
    wall.ends[0] = foot + along * spotter::Percentile(alongs, 0.02);
    wall.ends[1] = foot + along * spotter::Percentile(alongs, 0.98);
    wall.meters = plane.distance;
    wall.degrees = std::atan2(normal.x, -normal.y) * 180 / CV_PI;
    return wall.heightMeters >= MinWallHeightMeters;   // too short: furniture, not a wall
}

// Steps 8-11.
std::vector<Wall> FindWalls(const std::vector<cv::Point3f>& cloud, std::mt19937& random)
{
    spotter::PointGrid grid = spotter::ThinCloud(cloud, 512, SampleStep);
    std::vector<bool> free(grid.points.size());   // has a reading, and no plane has taken it yet
    int readings = 0;
    for (size_t i = 0; i < free.size(); i++) readings += (free[i] = !std::isnan(grid.points[i].z));

    std::vector<Wall> walls;
    for (int search = 0; search < MaxWalls; search++)
    {
        std::vector<int> freeCells;
        for (int i = 0; i < (int)free.size(); i++) if (free[i]) freeCells.push_back(i);
        spotter::Plane best;
        int count;
        if (freeCells.size() < 3 || !BestUprightPlane(grid, free, freeCells, random, best, count) || count < MinWallShare * readings) break;

        // Step 10: the plane's biggest connected piece; take it away, refit.
        cv::Mat onPlane(grid.rows, grid.columns, CV_8U, cv::Scalar(0)), labels, stats, centroids;
        for (int i : freeCells) if (std::abs(best.DistanceTo(grid.points[i])) < InlierThicknessMeters) onPlane.data[i] = 255;
        int pieces = cv::connectedComponentsWithStats(onPlane, labels, stats, centroids);
        int biggest = 1;
        for (int piece = 2; piece < pieces; piece++)
            if (stats.at<int>(piece, cv::CC_STAT_AREA) > stats.at<int>(biggest, cv::CC_STAT_AREA)) biggest = piece;
        if (pieces < 2 || stats.at<int>(biggest, cv::CC_STAT_AREA) < MinWallShare * readings)
        {
            for (int i : freeCells) if (onPlane.data[i]) free[i] = false;   // too scattered: take it away anyway, or it's found again
            continue;
        }
        std::vector<cv::Point3f> inliers;
        for (int i : freeCells) if (labels.ptr<int>()[i] == biggest) { inliers.push_back(grid.points[i]); free[i] = false; }
        Wall wall;
        if (WallFrom(spotter::FitPlane(inliers), inliers, wall)) walls.push_back(wall);
    }
    return walls;
}

// Steps 1-7, as in 06: YOLO on its own thread, the tracker, and each object's trail on the floor.
void TrackObjects(const spotter::RgbdFrame& frame, double dt, spotter::AsyncDetector& detector, spotter::Tracker& tracker,
                  spotter::BoxDepthSampler& boxDepth, spotter::TrackTrails& trails)
{
    boxDepth.Build(frame);
    tracker.Predict(dt);
    if (auto batch = detector.TryTake()) tracker.Match(*batch, boxDepth);
    detector.Offer(frame.colour);
    tracker.Remeasure(boxDepth);
    trails.Update(tracker, frame.colourCamera);
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

void ShowCamera(cv::Mat& picture, const spotter::Tracker& tracker, const spotter::ClassList& classes, const spotter::FpsMeter& fps)
{
    for (const spotter::Track& track : tracker.Tracks())
    {
        cv::Scalar colour = spotter::ColourForId(track.id);
        cv::rectangle(picture, track.Box(), colour, track.Coasting() ? 1 : 3);
        cv::putText(picture, "#" + std::to_string(track.id) + " " + classes.Name(track.classId), track.Box().tl() + cv::Point2d(4, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, colour, 2);
    }
    fps.Draw(picture);
    cv::imshow(CameraWindow, spotter::WithKeys(picture, Keys));
}

void ShowRadar(spotter::RadarView& radar, const std::vector<Wall>& walls, const spotter::Tracker& tracker, const spotter::TrackTrails& trails,
               const spotter::ClassList& classes, const spotter::PinholeCamera& colourCamera)
{
    radar.Begin(colourCamera);
    // ponytail: walls come from the depth camera, objects from the colour camera, about 5 cm apart.
    for (const Wall& wall : walls) radar.DrawLine(wall.ends[0], wall.ends[1], cv::Scalar(160, 160, 160), 3);
    for (const spotter::Track& track : tracker.Tracks())
        radar.DrawTrail(trails.Of(track.id), spotter::ColourForId(track.id), track.Coasting(), "#" + std::to_string(track.id) + " " + classes.Name(track.classId));
    cv::imshow(RadarWindow, spotter::WithKeys(radar.Image(), Keys));
}

void Report(const std::vector<Wall>& walls, const spotter::Tracker& tracker, const spotter::TrackTrails& trails, const spotter::ClassList& classes)
{
    std::printf("%zu objects, %zu walls\n", tracker.Tracks().size(), walls.size());
    for (const Wall& wall : walls)
        std::printf("  wall %.2f m away, %.2f m long, turned %+.0f deg, %.1f m of it in view\n", wall.meters, cv::norm(wall.ends[1] - wall.ends[0]), wall.degrees, wall.heightMeters);
    for (const spotter::Track& track : tracker.Tracks())
        if (!trails.Of(track.id).empty())
            std::printf("  #%-3d %-12s X %+.2f m, Z %.2f m\n", track.id, classes.Name(track.classId).c_str(), trails.Of(track.id).back().x, trails.Of(track.id).back().y);
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    spotter::AsyncDetector detector(yolo);
    spotter::Tracker tracker(spotter::Tracker::Options{});
    spotter::BoxDepthSampler boxDepth;
    spotter::TrackTrails trails;
    spotter::RadarView radar;
    std::mt19937 random(1);
    cv::namedWindow(CameraWindow, cv::WINDOW_NORMAL);
    cv::resizeWindow(CameraWindow, 960, 540);
    spotter::FpsMeter fps;
    spotter::Every report(ReportEveryMs);
    spotter::FrameTimer timer;
    spotter::RgbdFrame frame;

    while (kinect.Next(frame))
    {
        TrackObjects(frame, timer.Step(frame), detector, tracker, boxDepth, trails);   // steps 1-7
        std::vector<Wall> walls = FindWalls(frame.depthCamera.Cloud(frame.depthMm), random);   // steps 8-11

        fps.Tick();
        ShowCamera(frame.colour, tracker, yolo.Classes(), fps);
        ShowRadar(radar, walls, tracker, trails, yolo.Classes(), frame.colourCamera);
        if (report.Due()) Report(walls, tracker, trails, yolo.Classes());

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 'c') trails.Clear();
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Stand still near a wall: your dot sits just in front of its line. Walk along it: does your
//     trail stay parallel to it?
//  2. Watch a wall's distance in the console for a minute: it wobbles a few mm (RANSAC is random,
//     depth is noisy). 06c filters it.
//  3. Tilt the camera down 30 degrees: the wall facing it stops counting ("upright" is the
//     camera's). Raise MaxWallTiltDegrees.
//  4. A couch back or a desk front: short, so it's skipped. Set MinWallHeightMeters = 0.3.
//  5. Set MaxWalls = 1: only the biggest wall. Which one wins, and why?
