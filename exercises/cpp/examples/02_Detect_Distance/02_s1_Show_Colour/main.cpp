// =====================================================================
//  02 · step 1 of 4 · Show the colour camera
//
//  The Kinect has a second camera: colour, 1920 x 1080. Ask KinectSource
//  for colour too (the default) and show it. The window is shrunk to half
//  size so it fits on the screen.
//
//  New in this step: RgbdFrame::colour.
//  Keys:  Q or Esc  quit
//  Next:  02_s2_Detect (find objects in it)
// =====================================================================

#include <opencv2/highgui.hpp>

#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

int main()
{
    spotter::KinectSource kinect(spotter::KinectSource::Options{});   // colour + depth
    if (!kinect.IsOpen()) return 1;

    const char* window = "02 step 1: colour";
    cv::namedWindow(window, cv::WINDOW_NORMAL);   // NORMAL = can be resized
    cv::resizeWindow(window, 960, 540);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        cv::imshow(window, frame.colour);
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: raise your right hand. It shows on the right: KinectSource un-mirrors the picture
// (the Kinect itself sends a mirror image, like a selfie camera).
