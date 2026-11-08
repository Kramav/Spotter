// =====================================================================
//  09c · step 2 of 3 · Floor, ceiling, walls
//
//  Step 1's planes, sorted by which way they face (the camera's up is -Y):
//    level, normal pointing up       -> floor (the lowest such plane)
//    level, normal pointing down     -> ceiling (if it's high enough)
//    upright and at least 0.8 m tall -> wall
//    anything else                   -> left for the furniture
//  Dotted green (floor), yellow (ceiling), blue (walls).
//
//  New in this step: classifying planes by their normal and height.
//  Keys:  S  scan     Q or Esc  quit
//  Next:  09c_s3_Level_It (stand the room up straight)
// =====================================================================

#include <cmath>
#include <cstdio>
#include <random>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/geometry.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/stats.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "S  scan     Q or Esc  quit";

const int Step = 4;
const float MaxTiltDegrees = 25, MinWallHeightMeters = 0.8f, MinCeilingHeightMeters = 1.8f;

// 'f' floor, 'c' ceiling, 'w' wall, 0 none, for one plane piece.
char Kind(const spotter::PlanePiece& piece, const spotter::PointGrid& grid, float lowest)
{
    const cv::Point3f up(0, -1, 0);
    const float levelCos = std::cos(MaxTiltDegrees * (float)CV_PI / 180), uprightSin = std::sin(MaxTiltDegrees * (float)CV_PI / 180);
    float upness = piece.plane.normal.dot(up);   // the normal points at the camera: + = the plane is below it
    if (upness > levelCos) return 'f';
    if (-upness > levelCos) return piece.plane.distance - lowest >= MinCeilingHeightMeters ? 'c' : 0;
    if (std::abs(upness) > uprightSin) return 0;   // slanted
    std::vector<float> heights;
    for (int i : piece.cells) heights.push_back(grid.points[i].dot(up));
    return spotter::Percentile(heights, 0.98) - spotter::Percentile(heights, 0.02) >= MinWallHeightMeters ? 'w' : 0;
}

// Every plane, sorted, its points dotted in its kind's colour.
std::vector<std::pair<cv::Point, cv::Scalar>> SortedDots(const spotter::PointGrid& grid, const spotter::RgbdFrame& frame, std::mt19937& random)
{
    std::vector<float> heights;   // every point's height (camera = 0), for the ceiling rule
    for (const cv::Point3f& p : grid.points) if (!std::isnan(p.z)) heights.push_back(-p.y);
    float lowest = spotter::Percentile(heights, 0.02);
    std::vector<std::pair<cv::Point, cv::Scalar>> dots;
    for (const spotter::PlanePiece& piece : spotter::FindPlanes(grid, spotter::PlaneSearch(), random))
    {
        char kind = Kind(piece, grid, lowest);
        std::printf("%s, %.2f m from the camera\n", kind == 'f' ? "floor" : kind == 'c' ? "ceiling" : kind == 'w' ? "wall" : "other", piece.plane.distance);
        if (!kind) continue;
        cv::Scalar colour = kind == 'f' ? cv::Scalar(0, 200, 0) : kind == 'c' ? cv::Scalar(0, 255, 255) : cv::Scalar(255, 80, 0);
        for (int cell : piece.cells)
        {
            int c = frame.colourIndex[(cell / grid.columns) * Step * 512 + (cell % grid.columns) * Step];   // the cell's depth pixel -> colour pixel
            if (c >= 0) dots.emplace_back(cv::Point(c % 1920, c / 1920), colour);
        }
    }
    return dots;
}

int main()
{
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!kinect.IsOpen()) return 1;

    spotter::DepthAverager averager(30);
    std::vector<std::pair<cv::Point, cv::Scalar>> dots;
    std::mt19937 random(1);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        if (averager.Running() && averager.Add(frame.depthMm))
            dots = SortedDots(spotter::ThinCloud(frame.depthCamera.Cloud(averager.Mean()), 512, Step), frame, random);
        cv::Mat display;
        cv::resize(frame.colour, display, frame.colour.size() / 2);
        for (const auto& [p, colour] : dots) cv::circle(display, p / 2, 2, colour, cv::FILLED);
        cv::imshow("09c step 2: floor green, ceiling yellow, walls blue (S scans)", spotter::WithKeys(display, Keys));
        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 's' && !averager.Running()) { averager.Start(); std::printf("Scanning: hold still...\n"); }
    }
    return 0;
}

// Try: scan a table from just above its height: its top is level and below the camera, so it may
// be called the floor. The full example takes the LOWEST level plane as the floor.
