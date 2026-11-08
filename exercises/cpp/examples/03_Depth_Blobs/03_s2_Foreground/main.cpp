// =====================================================================
//  03 · step 2 of 3 · Which pixels are new?
//
//  A pixel is "foreground" when it is clearly nearer than the remembered
//  room (by more than the noise) AND has a reading. The answer is a mask:
//  255 = foreground, 0 = not. It is drawn green on the grey depth picture.
//
//  New in this step: a threshold, a mask, painting with a mask.
//  Keys:  B  remember the room now     Q or Esc  quit
//  Next:  03_s3_Blobs (group the green pixels into objects)
// =====================================================================

#include <cstdio>
#include <opencv2/highgui.hpp>

#include "spotter/depth.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// Too small: the noise shows up green. Too big: a hand flat on a table disappears.
const float NearerMm = 100;

cv::Mat Foreground(const cv::Mat& liveMm, const cv::Mat& backgroundMm)
{
    if (backgroundMm.empty()) return cv::Mat::zeros(liveMm.size(), CV_8U);
    return (backgroundMm - liveMm > NearerMm) & (liveMm > 0);   // each comparison gives a 255/0 mask
}

int main()
{
    spotter::KinectSource::Options options;
    options.colour = false;
    spotter::KinectSource kinect(options);
    if (!kinect.IsOpen()) return 1;

    cv::Mat backgroundMm;
    spotter::RgbdFrame frame;
    std::printf("Step out of view and press B.\n");
    while (kinect.Next(frame))
    {
        cv::Mat picture = spotter::DepthPicture(frame.depthMm);
        picture.setTo(cv::Scalar(0, 255, 0), Foreground(frame.depthMm, backgroundMm));   // green where the mask is 255
        cv::imshow("03 step 2: foreground (B remembers the room)", picture);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 'b') { backgroundMm = frame.depthMm.clone(); std::printf("Room remembered. Walk in!\n"); }
    }
    return 0;
}

// Try: set NearerMm = 20. The floor and walls start to sparkle green: the noise. Find the smallest
// value that stays clean at your distances. Also look at the edges of your outline: green specks.
