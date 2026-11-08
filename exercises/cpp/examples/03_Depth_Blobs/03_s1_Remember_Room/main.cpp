// =====================================================================
//  03 · step 1 of 3 · Remember the room
//
//  Press B with nobody in view: the program keeps a copy of that depth
//  picture (the "background"). From then on it shows how much NEARER each
//  pixel is than the remembered room: the room itself goes black, and
//  anything new (you) lights up. No AI, and it works in the dark.
//
//  New in this step: keeping a frame, subtracting two depth pictures.
//  Keys:  B  remember the room now (step out of view first)
//         Q or Esc  quit
//  Next:  03_s2_Foreground (decide which pixels are "new")
// =====================================================================

#include <cstdio>
#include <opencv2/highgui.hpp>

#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

int main()
{
    spotter::KinectSource::Options options;
    options.colour = false;
    spotter::KinectSource kinect(options);
    if (!kinect.IsOpen()) return 1;

    cv::Mat backgroundMm;   // empty until B
    spotter::RgbdFrame frame;
    std::printf("Step out of view and press B.\n");
    while (kinect.Next(frame))
    {
        if (backgroundMm.empty()) cv::imshow("03 step 1: press B to remember the room", frame.depthMm / 6000);
        else cv::imshow("03 step 1: press B to remember the room", (backgroundMm - frame.depthMm) / 1000);   // 1 m nearer = white

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 'b') { backgroundMm = frame.depthMm.clone(); std::printf("Room remembered. Walk in!\n"); }   // clone: a copy, not a view
    }
    return 0;
}

// Try: the room isn't perfectly black: faint speckles flicker. That's the sensor's noise, a few
// mm from frame to frame. Step 2 needs a threshold so the noise doesn't count as "new".
// Also: where the room had no reading (0), background - live is negative, so it stays black.
