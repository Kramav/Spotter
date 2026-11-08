// =====================================================================
//  01 · step 2 of 3 · Read one pixel
//
//  The picture is just a grid of numbers. Read the one in the middle and
//  print it about once a second. depthMm.at<float>(row, column) is that
//  pixel's distance in mm; 0 means the sensor got no reading there.
//
//  New in this step: reading a pixel, spotter::Every (a timer).
//  Keys:  Q or Esc  quit
//  Next:  01_s3_Crosshair (move the pixel you read, see it on the picture)
// =====================================================================

#include <cstdio>
#include <opencv2/highgui.hpp>

#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

void PrintCentre(const cv::Mat& depthMm)
{
    int row = depthMm.rows / 2, column = depthMm.cols / 2;   // 212, 256
    float mm = depthMm.at<float>(row, column);
    if (mm > 0) std::printf("centre pixel (%d, %d): %.0f mm\n", column, row, mm);
    else std::printf("centre pixel (%d, %d): no reading\n", column, row);
}

int main()
{
    spotter::KinectSource::Options options;
    options.colour = false;
    spotter::KinectSource kinect(options);
    if (!kinect.IsOpen()) return 1;

    spotter::Every second(1000);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        if (second.Due()) PrintCentre(frame.depthMm);
        cv::imshow("01 step 2: depth", frame.depthMm / 4500);
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: point the middle at a wall and watch the number. It wobbles by a few mm from frame to
// frame: that's the sensor's noise. Example 07 measures it properly.
