// =====================================================================
//  06c · step 1 of 2 · Walls wobble
//
//  06b's walls (now spotter::FindWalls) found afresh every frame. Point
//  the camera at a wall and keep it still: the wall doesn't move, but its
//  measured distance does, by a few mm, because RANSAC is random and the
//  depth is noisy. This step prints the nearest wall's distance and how
//  much it wobbles (standard deviation over the last second).
//
//  New in this step: measuring the wobble a filter should remove.
//  Keys:  Q or Esc  quit
//  Next:  06c_s2_EKF_One_Wall (a filter for one wall, without the Kinect)
// =====================================================================

#include <cmath>
#include <cstdio>
#include <deque>
#include <random>
#include <opencv2/highgui.hpp>

#include "spotter/radar.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"
#include "spotter/walls.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "Q or Esc  quit";

int main()
{
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!kinect.IsOpen()) return 1;

    spotter::RadarView radar;
    std::mt19937 random(1);
    std::deque<double> lastSecond;   // the nearest wall's distance, last 30 frames
    spotter::Every report(1000);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        std::vector<spotter::Wall> walls = spotter::FindWalls(spotter::ThinCloud(frame.depthCamera.Cloud(frame.depthMm), 512, 4), spotter::WallSearch(), random);
        radar.Begin(frame.colourCamera);
        double nearest = 1e9;
        for (const spotter::Wall& wall : walls)
        {
            radar.DrawLine(wall.ends[0], wall.ends[1], cv::Scalar(190, 190, 190), 2);
            nearest = std::min(nearest, wall.meters);
        }
        if (nearest < 1e9) lastSecond.push_back(nearest);
        if (lastSecond.size() > 30) lastSecond.pop_front();
        if (report.Due() && lastSecond.size() > 1)
        {
            double mean = 0, squares = 0;
            for (double m : lastSecond) mean += m / lastSecond.size();
            for (double m : lastSecond) squares += (m - mean) * (m - mean) / (lastSecond.size() - 1);
            std::printf("nearest wall %.3f m, wobbles +-%.1f mm from frame to frame\n", mean, 1000 * std::sqrt(squares));
        }
        cv::imshow("06c step 1: raw walls, found afresh each frame", spotter::WithKeys(radar.Image(), Keys));
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: watch the line's ends too: they jump far more than the distance, especially when someone
// walks in front of the wall and hides part of it.
