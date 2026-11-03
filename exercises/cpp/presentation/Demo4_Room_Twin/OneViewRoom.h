// =====================================================================
//  Demo 4's fallback: a room from the Kinect's own view
//
//  Demo4_Room_Twin calls this when it finds no room file from 13. The
//  empty view in front of the Kinect becomes the room instead:
//
//    1. Capture   average AverageFrames depth frames per pixel (07), with
//                 the colour of the last one (08). Nobody in view.
//    2. Normals   each point's surface direction, from its neighbours to
//                 the right and below (Demo 4's ICP measures along it).
//    3. Level     turn the view back by the Kinect's tilt (as in Demo 1),
//                 so the doll's house stands upright, then flip it to the
//                 room's way round (X right, Y up, Z towards the camera).
//
//  It's one view, so the "room" is only the front of what the Kinect sees
//  (08's shell): good for rehearsing, but 13's scan from many spots is the
//  real thing. Nothing is saved: the view stays in memory.
// =====================================================================

#pragma once

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <opencv2/opencv.hpp>

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

namespace OneView
{
const int AverageFrames = 15;          // 07: 15 frames take the noise from about ±1.2 to ±0.3 mm
const int SecondsToClear = 3;          // time to step out of view first
const int NormalStepPixels = 2;        // neighbours this far apart give each point's normal
const float MaxJumpShare = 0.03f;      // 09b: neighbours further apart than 3% of the distance = an edge, no normal

// The Kinect's tilt, upwards positive, as Demo 1's LevelDegrees: 14 suits it sitting on the floor
// aimed up, 0 standing level on a tripod.
const float LevelDegrees = 14;
}

// Steps 2-3, with no Kinect needed (so it can be tested): a grid of Kinect points (512 x 424, X
// right, Y down, Z ahead; NaN z = no reading) and their colours -> the room's points, colours and
// normals, and the turn from the Kinect into the room.
inline void MakeOneViewRoom(const std::vector<cv::Point3f>& grid, const std::vector<cv::Vec3b>& gridColours, int width, int height,
                            std::vector<cv::Point3f>& points, std::vector<cv::Vec3b>& colours,
                            std::vector<cv::Point3f>& normals, cv::Matx33f& firstCamera)
{
    // Level about the Kinect's left-right axis, then flip Y and Z (09b's flip).
    float c = std::cos(OneView::LevelDegrees * (float)CV_PI / 180), s = std::sin(OneView::LevelDegrees * (float)CV_PI / 180);
    firstCamera = cv::Matx33f(1, 0, 0, 0, -1, 0, 0, 0, -1) * cv::Matx33f(1, 0, 0, 0, c, -s, 0, s, c);

    const int step = OneView::NormalStepPixels;
    for (int row = 0; row + step < height; row++)
    {
        for (int column = 0; column + step < width; column++)
        {
            const cv::Point3f& p = grid[row * width + column];
            const cv::Point3f& right = grid[row * width + column + step];
            const cv::Point3f& below = grid[(row + step) * width + column];
            if (std::isnan(p.z) || std::isnan(right.z) || std::isnan(below.z)) continue;
            float jump = OneView::MaxJumpShare * p.z;
            if (std::abs(right.z - p.z) > jump || std::abs(below.z - p.z) > jump) continue;   // an edge
            cv::Point3f n = (right - p).cross(below - p);
            float length = (float)cv::norm(n);
            if (length == 0) continue;
            points.push_back(firstCamera * p);
            colours.push_back(gridColours[row * width + column]);
            normals.push_back(firstCamera * (n / length));
        }
    }
}

// Steps 1-3 with the Kinect (already started, with colour and depth). False if nothing was seen.
inline bool OneViewRoom(libfreenect2::SyncMultiFrameListener& listener, libfreenect2::Registration& registration,
                        std::vector<cv::Point3f>& points, std::vector<cv::Vec3b>& colours,
                        std::vector<cv::Point3f>& normals, cv::Matx33f& firstCamera)
{
    const int width = 512, height = 424;
    libfreenect2::Frame undistorted(width, height, 4), registered(width, height, 4);
    libfreenect2::FrameMap frames;
    std::printf("No room file, so the Kinect's own view will be the room (one view: rehearsal only).\n"
                "Clear the view: capturing in %d s.\n", OneView::SecondsToClear);
    auto start = std::chrono::steady_clock::now();
    std::vector<cv::Point3f> sum(width * height, cv::Point3f(0, 0, 0));
    std::vector<int> count(width * height, 0);
    std::vector<cv::Vec3b> gridColours(width * height);
    int captured = 0;
    while (captured < OneView::AverageFrames)
    {
        if (!listener.waitForNewFrame(frames, 10 * 1000)) return false;
        bool waiting = std::chrono::steady_clock::now() - start < std::chrono::seconds(OneView::SecondsToClear);
        if (!waiting)
        {
            registration.apply(frames[libfreenect2::Frame::Color], frames[libfreenect2::Frame::Depth], &undistorted, &registered);
            for (int row = 0; row < height; row++)
            {
                for (int column = 0; column < width; column++)
                {
                    float x, y, z;
                    registration.getPointXYZ(&undistorted, row, column, x, y, z);
                    if (std::isnan(z)) continue;
                    sum[row * width + column] += cv::Point3f(x, y, z);
                    count[row * width + column]++;
                    const unsigned char* bgrx = registered.data + 4 * (row * width + column);
                    gridColours[row * width + column] = cv::Vec3b(bgrx[0], bgrx[1], bgrx[2]);
                }
            }
            captured++;
        }
        listener.release(frames);
    }
    // A pixel counts if it read in at least half the frames: edges that flicker in and out don't.
    std::vector<cv::Point3f> grid(width * height, cv::Point3f(0, 0, NAN));
    for (int i = 0; i < width * height; i++)
        if (count[i] * 2 >= OneView::AverageFrames) grid[i] = sum[i] * (1.0f / count[i]);
    MakeOneViewRoom(grid, gridColours, width, height, points, colours, normals, firstCamera);
    std::printf("Captured the view: %zu points (levelled by %.0f deg: OneView::LevelDegrees).\n", points.size(), OneView::LevelDegrees);
    return !points.empty();
}
