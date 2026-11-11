// =====================================================================
//  05b · step 1 of 2 · Sure and unsure detections
//
//  Ask YOLO for everything down to 0.15 confidence and draw the sure ones
//  (0.40 and up) green, the unsure ones yellow. Watch a still object: its
//  confidence flickers, and when it dips under 0.40 a 0.40 cut-off loses
//  it, yet the yellow box is still right there on it.
//
//  New in this step: looking at confidence, two thresholds.
//  Keys:  Q or Esc  quit
//  Next:  05b_s2_Fade (a coasting object should slow down)
// =====================================================================

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "Q or Esc  quit";

const float StartConfidence = 0.40f;   // sure: may start a new ID (05b)
const float KeepConfidence = 0.15f;    // unsure: may only continue one

//spotter::YoloDetector::Options YoloSettings(){
//
//@brief - the detector's options: yolo26n.onnx and its names file from the repo; minConfidence is the weak
//         threshold, so YOLO hands back everything down to it. O(1).
spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = KeepConfidence;
    return options;
}

//int main(){
//
//@brief - the program: 05b · step 1 of 2 · Sure and unsure detections. Ask YOLO for everything down to 0.15
//         confidence and draw the sure ones (0.40 and up) green, the unsure ones yellow. Setup, main loop and
//         clean-up are laid out in the file's sections. Returns 0 on a normal exit, 1 if it couldn't start.
int main()
{
    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    const char* window = "05b step 1: green = sure (0.40+), yellow = unsure (0.15+)";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, 960, 540);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        for (const spotter::Detection& d : yolo.Detect(frame.colour))
        {
            bool sure = d.confidence >= StartConfidence;
            cv::Scalar colour = sure ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
            cv::rectangle(frame.colour, d.box, colour, sure ? 3 : 2);
            cv::putText(frame.colour, cv::format("%s %.2f", yolo.Classes().Name(d.classId).c_str(), d.confidence), d.box.tl() + cv::Point2d(4, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, colour, 2);
        }
        cv::imshow(window, spotter::WithKeys(frame.colour, Keys));
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: count how often a still object turns yellow in a minute. Also look for yellow boxes on
// nothing in particular: that's why weak detections may not START an object, only keep one alive.
