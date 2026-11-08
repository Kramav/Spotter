// =====================================================================
//  01 · step 1 of 3 · Show the depth picture
//
//  The smallest Kinect program: open the depth camera and show what it
//  sends. Each pixel is a distance in millimetres. Dividing by 4500 turns
//  0..4500 mm into 0..1, which OpenCV shows as black..white: far is
//  bright, near is dark, and "no reading" (0) is black.
//
//  New in this step: KinectSource, RgbdFrame::depthMm, the main loop.
//  Keys:  Q or Esc  quit
//  Next:  01_s2_Read_A_Pixel (the numbers behind the picture)
// =====================================================================

#include <opencv2/highgui.hpp>

#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

int main()
{
    spotter::KinectSource::Options options;
    options.colour = false;   // depth only
    spotter::KinectSource kinect(options);
    if (!kinect.IsOpen()) return 1;

    spotter::RgbdFrame frame;
    while (kinect.Next(frame))   // waits for the next frame: 30 a second
    {
        cv::imshow("01 step 1: depth (far = bright, no reading = black)", frame.depthMm / 4500);
        if (spotter::IsQuitKey(cv::waitKey(1))) break;   // waitKey also lets the window redraw
    }
    return 0;
}

// Try: put your hand close to the sensor. Under ~0.5 m it turns BLACK, not darker:
// too close gives no reading at all, not a small number.
