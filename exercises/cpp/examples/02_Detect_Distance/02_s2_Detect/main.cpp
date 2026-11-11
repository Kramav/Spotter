// =====================================================================
//  02 · step 2 of 4 · Find objects with YOLO
//
//  Run the detector on each colour picture and draw what it finds: a box,
//  the class name, and how sure it is (0..1). The detector never sees
//  depth; it only looks at colour.
//
//  New in this step: YoloDetector, Detection, ClassList.
//  Keys:  Q or Esc  quit
//  Needs: yolo26n.onnx and yolo26n.names in the repo root.
//  Next:  02_s3_Where_Depth_Lands (how depth and colour line up)
// =====================================================================

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "Q or Esc  quit";

//spotter::YoloDetector::Options YoloSettings(){
//
//@brief - the detector's options: yolo26n.onnx and its names file from the repo, minimum confidence 0.4 (lower =
//         more boxes, more mistakes). O(1).
spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";   // SPOTTER_REPO_ROOT comes from CMake
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = 0.4f;                            // lower = more boxes, more mistakes
    return options;
}

//void Draw(cv::Mat& picture, const std::vector<spotter::Detection>& detections, const spotter::ClassList& classes){
//
//@param1 - picture: the colour picture; drawn on
//@param2 - detections: what the detector found
//@param3 - classes: the class names
//
//@brief - a green box on each detection with its class name and confidence. O(detections).
void Draw(cv::Mat& picture, const std::vector<spotter::Detection>& detections, const spotter::ClassList& classes)
{
    for (const spotter::Detection& d : detections)
    {
        cv::rectangle(picture, d.box, cv::Scalar(0, 255, 0), 3);
        cv::putText(picture, cv::format("%s %.2f", classes.Name(d.classId).c_str(), d.confidence),
                    d.box.tl() + cv::Point2d(4, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }
}

//int main(){
//
//@brief - the program: 02 · step 2 of 4 · Find objects with YOLO. Run the detector on each colour picture and draw
//         what it finds: a box, the class name, and how sure it is (0..1). The detector never sees depth; it only
//         looks at colour. Setup, main loop and clean-up are laid out in the file's sections. Returns 0 on a
//         normal exit, 1 if it couldn't start.
int main()
{
    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    const char* window = "02 step 2: detect";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, 960, 540);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        std::vector<spotter::Detection> detections = yolo.Detect(frame.colour);
        Draw(frame.colour, detections, yolo.Classes());
        cv::imshow(window, spotter::WithKeys(frame.colour, Keys));
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: the frame rate drops to about 6 per second: YOLO takes ~170 ms per picture on the CPU.
// Example 05c moves it onto its own thread. Hold up a cup, a phone, a book: what does it call them?
