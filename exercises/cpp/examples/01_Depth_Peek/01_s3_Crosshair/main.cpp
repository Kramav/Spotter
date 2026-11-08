// =====================================================================
//  01 · step 3 of 3 · A crosshair you can move
//
//  Draw the depth picture the way the examples do (spotter::DepthPicture:
//  near = WHITE, far = black, no reading = RED), put a crosshair on it,
//  and write the distance under it. W A S D move the crosshair.
//
//  New in this step: DepthPicture, drawing on the picture, keys.
//  Keys:  W A S D  move the crosshair     Q or Esc  quit
//  Next:  01_Depth_Peek (the full example adds statistics and a frame rate)
// =====================================================================

#include <algorithm>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

const int StepPixels = 10;   // how far one key press moves the crosshair

void MoveCrosshair(int key, cv::Point& crosshair)
{
    if (key == 'w') crosshair.y -= StepPixels;
    if (key == 's') crosshair.y += StepPixels;
    if (key == 'a') crosshair.x -= StepPixels;
    if (key == 'd') crosshair.x += StepPixels;
    crosshair.x = std::clamp(crosshair.x, 0, 511);   // stay inside the 512 x 424 picture
    crosshair.y = std::clamp(crosshair.y, 0, 423);
}

void Show(const cv::Mat& depthMm, cv::Point crosshair)
{
    cv::Mat picture = spotter::DepthPicture(depthMm);   // 0.5 m = white .. 4.5 m = black, red = no reading
    float mm = depthMm.at<float>(crosshair);
    std::string text = mm > 0 ? std::to_string((int)mm) + " mm" : "no reading";
    cv::drawMarker(picture, crosshair, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 20, 1);
    cv::putText(picture, text, crosshair + cv::Point(8, -8), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    cv::imshow("01 step 3: crosshair (WASD moves)", picture);
}

int main()
{
    spotter::KinectSource::Options options;
    options.colour = false;
    spotter::KinectSource kinect(options);
    if (!kinect.IsOpen()) return 1;

    cv::Point crosshair(256, 212);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        Show(frame.depthMm, crosshair);
        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        MoveCrosshair(key, crosshair);
    }
    return 0;
}

// Try: aim the crosshair at a black monitor, a window or a mirror: red, no reading.
// Turn the lights off: depth keeps working, the Kinect lights the room with its own infrared.
