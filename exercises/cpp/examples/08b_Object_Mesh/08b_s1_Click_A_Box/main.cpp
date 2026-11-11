// =====================================================================
//  08b · step 1 of 3 · Click a box
//
//  YOLO's boxes on a half-size colour picture. Click inside one: it turns
//  yellow and its class is printed. The mouse arrives through a callback
//  that OpenCV calls with the click's position; the window is half size,
//  so the position is doubled to get colour-picture pixels.
//
//  New in this step: a mouse callback; which box holds a point.
//  Keys:  click a box  choose it     Q or Esc  quit
//  Next:  08b_s2_Cut_Out (the depth pixels of the chosen object)
// =====================================================================

#include <cstdio>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "click a box  choose it     Q or Esc  quit";

//void OnMouse(int event, int x, int y, int flags, void* click){
//
//@param1 - event: OpenCV's mouse event
//@param2 - x: the mouse's column in the half-size window
//@param3 - y: its row
//@param4 - flags: OpenCV's flags (unused)
//@param5 - click: a cv::Point2d the left-click is written to, in colour pixels
//
//@brief - OpenCV's mouse callback: a left click stores its spot doubled (the window is half size), i.e. in colour
//         pixels. O(1).
void OnMouse(int event, int x, int y, int, void* click)
{
    if (event == cv::EVENT_LBUTTONDOWN) *(cv::Point2d*)click = cv::Point2d(x * 2, y * 2);   // half-size window -> colour pixels
}

//spotter::YoloDetector::Options YoloSettings(){
//
//@brief - the detector's options: yolo26n.onnx and its names file from the repo, minimum confidence 0.4. O(1).
spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = 0.4f;
    return options;
}

//int main(){
//
//@brief - the program: 08b · step 1 of 3 · Click a box. YOLO's boxes on a half-size colour picture. Click inside
//         one: it turns yellow and its class is printed. Setup, main loop and clean-up are laid out in the file's
//         sections. Returns 0 on a normal exit, 1 if it couldn't start.
int main()
{
    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    const char* window = "08b step 1: click a box";
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    cv::Point2d click(-1, -1);
    cv::setMouseCallback(window, OnMouse, &click);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        cv::Mat display;
        cv::resize(frame.colour, display, frame.colour.size() / 2);
        for (const spotter::Detection& d : yolo.Detect(frame.colour))
        {
            bool chosen = d.box.contains(click);
            if (chosen) std::printf("clicked: %s (%.2f)\n", yolo.Classes().Name(d.classId).c_str(), d.confidence);
            cv::Rect half((int)d.box.x / 2, (int)d.box.y / 2, (int)d.box.width / 2, (int)d.box.height / 2);
            cv::rectangle(display, half, chosen ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0), chosen ? 4 : 2);
        }
        click = cv::Point2d(-1, -1);   // a click counts once
        cv::imshow(window, spotter::WithKeys(display, Keys));
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: click where two boxes overlap: both light up. The full example picks the smaller one
// (usually the thing in front).
