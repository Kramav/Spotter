// =====================================================================
//  09b · step 1 of 2 · A still, averaged scan
//
//  Press S and hold still: 30 depth frames are averaged per pixel (07's
//  idea, now spotter::DepthAverager). The averaged depth picture opens in
//  its own window next to the live one: fewer red holes, steadier greys.
//  This average is what the mesh is built from.
//
//  New in this step: DepthAverager (Start, Add, Mean).
//  Keys:  S  scan     Q or Esc  quit
//  Next:  09b_s2_Grid_Triangles (turn the scan into triangles)
// =====================================================================

#include <cstdio>
#include <opencv2/highgui.hpp>

#include "spotter/depth.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "S  scan     Q or Esc  quit";

int main()
{
    spotter::KinectSource::Options options;
    options.colour = false;
    spotter::KinectSource kinect(options);
    if (!kinect.IsOpen()) return 1;

    spotter::DepthAverager averager(30);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        if (averager.Running() && averager.Add(frame.depthMm))
        {
            cv::Mat averageMm = averager.Mean();   // 0 where a pixel read in under half the frames
            std::printf("Averaged %d frames: %d of %d pixels have depth.\n", averager.Frames(), cv::countNonZero(averageMm), (int)averageMm.total());
            cv::imshow("09b step 1: the averaged scan", spotter::WithKeys(spotter::DepthPicture(averageMm), Keys));
        }
        cv::imshow("09b step 1: live (S scans)", spotter::WithKeys(spotter::DepthPicture(frame.depthMm), Keys));
        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 's' && !averager.Running()) { averager.Start(); std::printf("Scanning: hold still...\n"); }
    }
    return 0;
}

// Try: wave a hand during the 30 frames: a ghostly smear in the average. Averaging only works for
// what stays still.
