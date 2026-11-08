// =====================================================================
//  08b · step 2 of 3 · Cut the object out of the depth
//
//  Click a box. Hold still while 30 depth frames are averaged (07's idea,
//  spotter::DepthAverager), then keep the depth pixels that:
//    - land inside the box (colourIndex, 02_s3), and
//    - lie within 0.3 m of the box's middle depth (02_s4's median),
//  so the wall behind the object is left out. Those pixels are tinted
//  green on the picture.
//
//  New in this step: averaging for a capture; a depth band around the object.
//  Keys:  click a box  capture it     Q or Esc  quit
//  Next:  08b_s3_Mesh_Front (join the pixels into triangles and save them)
// =====================================================================

#include <cmath>
#include <cstdio>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/stats.hpp"
#include "spotter/view.hpp"

const float DepthBandMeters = 0.3f;

void OnMouse(int event, int x, int y, int, void* click)
{
    if (event == cv::EVENT_LBUTTONDOWN) *(cv::Point2d*)click = cv::Point2d(x * 2, y * 2);
}

spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = 0.4f;
    return options;
}

// The depth pixels in the box and near its middle depth. Returns their colour pixels, for the tint.
std::vector<cv::Point> CutOut(const cv::Rect2d& box, const cv::Mat& depthMm, const spotter::RgbdFrame& frame)
{
    cv::Rect2d centre(box.x + box.width / 4, box.y + box.height / 4, box.width / 2, box.height / 2);
    std::vector<int> inBox;
    std::vector<float> centreMm;
    for (int i = 0; i < (int)frame.colourIndex.size(); i++)
    {
        int c = frame.colourIndex[i];
        if (c < 0 || depthMm.at<float>(i) <= 0) continue;
        cv::Point2d pixel(c % frame.colour.cols, c / frame.colour.cols);
        if (!box.contains(pixel)) continue;
        inBox.push_back(i);
        if (centre.contains(pixel)) centreMm.push_back(depthMm.at<float>(i));
    }
    float middle = spotter::Median(centreMm);
    std::vector<cv::Point> kept;
    for (int i : inBox)
        if (std::abs(depthMm.at<float>(i) - middle) < DepthBandMeters * 1000)
            kept.emplace_back(frame.colourIndex[i] % frame.colour.cols, frame.colourIndex[i] / frame.colour.cols);
    std::printf("%zu depth pixels in the box, %zu of them within %.1f m of its middle (%.2f m)\n", inBox.size(), kept.size(), DepthBandMeters, middle / 1000);
    return kept;
}

int main()
{
    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    const char* window = "08b step 2: click a box, hold still";
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    cv::Point2d click(-1, -1);
    cv::setMouseCallback(window, OnMouse, &click);
    spotter::DepthAverager averager(30);
    cv::Rect2d chosen;
    std::vector<cv::Point> tint;
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        if (averager.Running() && averager.Add(frame.depthMm)) tint = CutOut(chosen, averager.Mean(), frame);   // the last frame is in
        else if (!averager.Running())
            for (const spotter::Detection& d : yolo.Detect(frame.colour))
                if (d.box.contains(click)) { chosen = d.box; averager.Start(); }
        click = cv::Point2d(-1, -1);

        cv::Mat display;
        cv::resize(frame.colour, display, frame.colour.size() / 2);
        for (const cv::Point& p : tint) display.at<cv::Vec3b>(p.y / 2, p.x / 2) = cv::Vec3b(0, 255, 0);
        if (averager.Running()) cv::putText(display, cv::format("HOLD STILL %d / 30", averager.Added()), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        cv::imshow(window, display);
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: capture a cup on a table: the green covers the table top inside the box too (it's at the
// same depth). The full example finds that surface with RANSAC and drops it.
