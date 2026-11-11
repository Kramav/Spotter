// =====================================================================
//  11b · step 1 of 3 · Features
//
//  A feature is a small patch of the picture distinctive enough to find
//  again from somewhere else: a poster's corner, a plug socket, the edge
//  of a shelf. ORB finds corners (FAST: a ring of pixels around the spot
//  is clearly brighter or darker than the middle), keeps the strongest,
//  and describes each with 256 yes/no answers ("is this pixel brighter than
//  that one?"), 32 bytes, so the same corner can be matched later.
//
//  Here: ORB on the colour picture laid onto the depth grid (each feature
//  then has a depth under it, for step 3), drawn as circles, live.
//
//  New in this step: cv::ORB and keypoints.
//  Keys:  Q or Esc  quit
//  Next:  11b_s2_Match_Features (find the same features in another view)
// =====================================================================

#include <cstdio>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "Q or Esc  quit";

const int FeatureCount = 1000;   // at most this many, the strongest first

//int main(){
//
//@brief - the program: 11b · step 1 of 3 · Features. A feature is a small patch of the picture distinctive enough
//         to find again from somewhere else: a poster's corner, a plug socket, the edge of a shelf. Setup, main
//         loop and clean-up are laid out in the file's sections. Returns 0 on a normal exit, 1 if it couldn't
//         start.
int main()
{
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!kinect.IsOpen()) return 1;

    cv::Ptr<cv::ORB> orb = cv::ORB::create(FeatureCount);   // built once, used every frame
    spotter::Every report(1000);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        cv::Mat grey, display;
        cv::cvtColor(frame.registered, grey, cv::COLOR_BGR2GRAY);   // ORB looks at brightness only
        std::vector<cv::KeyPoint> keypoints;
        orb->detect(grey, keypoints);
        cv::drawKeypoints(frame.registered, keypoints, display, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::resize(display, display, display.size() * 2);
        cv::imshow("11b step 1: ORB features (circle = patch size, line = its direction)", spotter::WithKeys(display, Keys));
        if (report.Due()) std::printf("%zu features\n", keypoints.size());
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: aim at a bookshelf, then at a bare wall. Then dim the lights: the count falls fast, and
// that's what made 13b add contrast evening (CLAHE) before ORB.
