// =====================================================================
//  03 · step 3 of 3 · Group the pixels into blobs
//
//  Two OpenCV calls turn green pixels into objects:
//    1. morphologyEx OPEN: shrink every green region a little, then grow it
//       back. Specks smaller than the shape vanish and never come back.
//    2. connectedComponentsWithStats: label each group of touching pixels
//       1, 2, 3 ... with its box and pixel count (label 0 = the rest).
//  Groups big enough to be something get a yellow box.
//
//  New in this step: cleaning a mask, connected components.
//  Keys:  B  remember the room now     Q or Esc  quit
//  Next:  03_Depth_Blobs (the full example measures each blob's distance and real size)
// =====================================================================

#include <cstdio>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "B  remember the room now     Q or Esc  quit";

const float NearerMm = 100;
const int CleanupPixels = 5;     // the OPEN shape's size; 1 = no cleanup
const int MinBlobPixels = 400;   // a person at 3 m covers a few thousand

cv::Mat Foreground(const cv::Mat& liveMm, const cv::Mat& backgroundMm)
{
    if (backgroundMm.empty()) return cv::Mat::zeros(liveMm.size(), CV_8U);
    return (backgroundMm - liveMm > NearerMm) & (liveMm > 0);
}

std::vector<cv::Rect> Blobs(const cv::Mat& foreground)
{
    cv::Mat cleaned, labels, stats, centroids;
    cv::morphologyEx(foreground, cleaned, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(CleanupPixels, CleanupPixels)));
    int count = cv::connectedComponentsWithStats(cleaned, labels, stats, centroids, 8);
    std::vector<cv::Rect> boxes;
    for (int label = 1; label < count; label++)
        if (stats.at<int>(label, cv::CC_STAT_AREA) >= MinBlobPixels)
            boxes.emplace_back(stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP),
                               stats.at<int>(label, cv::CC_STAT_WIDTH), stats.at<int>(label, cv::CC_STAT_HEIGHT));
    return boxes;
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
        cv::Mat foreground = Foreground(frame.depthMm, backgroundMm);
        cv::Mat picture = spotter::DepthPicture(frame.depthMm);
        picture.setTo(cv::Scalar(0, 255, 0), foreground);
        for (const cv::Rect& box : Blobs(foreground)) cv::rectangle(picture, box, cv::Scalar(0, 255, 255), 1);
        cv::imshow("03 step 3: blobs (B remembers the room)", spotter::WithKeys(picture, Keys));

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 'b') { backgroundMm = frame.depthMm.clone(); std::printf("Room remembered. Walk in!\n"); }
    }
    return 0;
}

// Try: CleanupPixels = 1, then 11. With 1, every speck at your outline's edge gets its own tiny
// group (too small to box, but count them). With 11, thin things like an arm start to vanish.
