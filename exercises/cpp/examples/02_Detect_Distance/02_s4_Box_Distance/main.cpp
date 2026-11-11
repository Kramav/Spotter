// =====================================================================
//  02 · step 4 of 4 · How far away is each box?
//
//  Steps 2 and 3 together: for each YOLO box, collect the depth pixels
//  that land in the MIDDLE half of the box (step 3's colourIndex), and
//  take their median. Why the middle: a loose box around a person is
//  mostly the wall behind them. Why the median: the few wall and edge
//  pixels left over don't move it.
//
//  New in this step: the depth of a box ("late fusion").
//  Keys:  Q or Esc  quit
//  Next:  02_Detect_Distance (the full example adds X and Y in metres and timings)
// =====================================================================

#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/stats.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "Q or Esc  quit";

const int MinDepthPixels = 20;   // fewer than this in the box's middle: "?" instead of a guess

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

//double BoxMeters(const cv::Rect2d& box, const spotter::RgbdFrame& frame){
//
//@param1 - box: a box on the colour picture
//@param2 - frame: a frame with depth and colourIndex
//
//@brief - the median distance in metres of the depth pixels landing in the middle half of the box; 0 if fewer than
//         MinDepthPixels land there. Visits every depth pixel: O(depth pixels).
double BoxMeters(const cv::Rect2d& box, const spotter::RgbdFrame& frame)
{
    cv::Rect2d middle(box.x + box.width / 4, box.y + box.height / 4, box.width / 2, box.height / 2);
    std::vector<float> depthsMm;
    for (int i = 0; i < (int)frame.colourIndex.size(); i++)
    {
        int index = frame.colourIndex[i];
        float mm = frame.depthMm.at<float>(i);
        if (index >= 0 && mm > 0 && middle.contains(cv::Point2d(index % frame.colour.cols, index / frame.colour.cols)))
            depthsMm.push_back(mm);
    }
    return (int)depthsMm.size() >= MinDepthPixels ? spotter::Median(depthsMm) / 1000 : 0;
}

//void Draw(cv::Mat& picture, const spotter::Detection& d, const std::string& name, double meters){
//
//@param1 - picture: the colour picture; drawn on
//@param2 - d: the detection
//@param3 - name: its class name
//@param4 - meters: its distance, or 0 for unknown
//
//@brief - a green box with "name 1.23 m" (or "name ? m"). O(1).
void Draw(cv::Mat& picture, const spotter::Detection& d, const std::string& name, double meters)
{
    std::string label = meters > 0 ? cv::format("%s %.2f m", name.c_str(), meters) : name + " ? m";
    cv::rectangle(picture, d.box, cv::Scalar(0, 255, 0), 3);
    cv::putText(picture, label, d.box.tl() + cv::Point2d(4, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
}

//int main(){
//
//@brief - the program: 02 · step 4 of 4 · How far away is each box?. Steps 2 and 3 together: for each YOLO box,
//         collect the depth pixels that land in the MIDDLE half of the box (step 3's colourIndex), and take their
//         median. Why the middle: a loose box around a person is mostly the wall behind them. Setup, main loop and
//         clean-up are laid out in the file's sections. Returns 0 on a normal exit, 1 if it couldn't start.
int main()
{
    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    const char* window = "02 step 4: box distance";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, 960, 540);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        for (const spotter::Detection& d : yolo.Detect(frame.colour))
            Draw(frame.colour, d, yolo.Classes().Name(d.classId), BoxMeters(d.box, frame));
        cv::imshow(window, spotter::WithKeys(frame.colour, Keys));
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: stand 1 m in front of a far wall, then change "middle" to the whole box. Your distance
// jumps toward the wall's: most of a loose box is background.
