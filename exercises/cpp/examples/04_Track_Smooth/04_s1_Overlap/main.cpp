// =====================================================================
//  04 · step 1 of 3 · How much do two boxes overlap?
//
//  To follow an object from one frame to the next, compare its new box
//  with last frame's boxes. The usual measure is "intersection over
//  union" (IoU): the shared area divided by the area both cover together.
//  1 = the same box, 0 = not touching.
//
//  Grey = last frame's boxes. Green = this frame's, labelled with the best
//  overlap with a last-frame box of the same class.
//
//  New in this step: Overlap (IoU), remembering last frame's boxes.
//  Keys:  Q or Esc  quit
//  Next:  04_s2_Keep_An_ID (high overlap = the same object)
// =====================================================================

#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = 0.4f;
    return options;
}

double Overlap(const cv::Rect2d& a, const cv::Rect2d& b)
{
    double intersection = (a & b).area();                   // & = the boxes' shared part
    double together = a.area() + b.area() - intersection;   // counted once
    return together > 0 ? intersection / together : 0;
}

double BestOverlap(const spotter::Detection& now, const std::vector<spotter::Detection>& before)
{
    double best = 0;
    for (const spotter::Detection& old : before)
        if (old.classId == now.classId) best = std::max(best, Overlap(old.box, now.box));
    return best;
}

int main()
{
    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    const char* window = "04 step 1: overlap with last frame";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, 960, 540);
    std::vector<spotter::Detection> lastFrame;
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        std::vector<spotter::Detection> detections = yolo.Detect(frame.colour);
        for (const spotter::Detection& old : lastFrame) cv::rectangle(frame.colour, old.box, cv::Scalar(150, 150, 150), 2);
        for (const spotter::Detection& d : detections)
        {
            cv::rectangle(frame.colour, d.box, cv::Scalar(0, 255, 0), 3);
            cv::putText(frame.colour, cv::format("%s overlap %.2f", yolo.Classes().Name(d.classId).c_str(), BestOverlap(d, lastFrame)),
                        d.box.tl() + cv::Point2d(4, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        }
        lastFrame = detections;
        cv::imshow(window, frame.colour);
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: stand still: overlap ~0.9. Walk across the view: it drops, because YOLO runs only ~6 times a
// second and you move far between frames. Below ~0.3 it's hard to tell "moved" from "someone else".
