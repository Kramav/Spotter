// =====================================================================
//  07d · step 1 of 3 · Which depth pixels are inside a person box?
//
//  YOLO's boxes live in the 1920 x 1080 colour picture; the depth pixels
//  are 512 x 424. colourIndex (02_s3) says where each depth pixel lands in
//  colour, so test that spot against each person box. Shown on the
//  "registered" picture: the colour seen at each depth pixel, so colour
//  and depth line up pixel for pixel. Person pixels are tinted red.
//
//  New in this step: frame.registered; boxes -> a per-depth-pixel mask;
//  finding the person class by name.
//  Keys:  Q or Esc  quit
//  Next:  07d_s2_Colour_Memory (remember what the room looks like)
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
//@brief - the detector's options: yolo26n.onnx and its names file from the repo, minimum confidence 0.4. O(1).
spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = 0.4f;
    return options;
}

//cv::Mat InsideBoxes(const std::vector<spotter::Detection>& detections, int classId, const spotter::RgbdFrame& frame){
//
//@param1 - detections: what the detector found
//@param2 - classId: the class to look for (e.g. person)
//@param3 - frame: a frame with depth and colourIndex
//
//@brief - a mask over the depth picture: 1 where the depth pixel lands inside a box of that class (via
//         colourIndex). O(depth pixels x detections).
cv::Mat InsideBoxes(const std::vector<spotter::Detection>& detections, int classId, const spotter::RgbdFrame& frame)
{
    cv::Mat mask = cv::Mat::zeros(frame.depthMm.size(), CV_8U);
    for (size_t i = 0; i < frame.colourIndex.size(); i++)
    {
        int c = frame.colourIndex[i];
        if (c < 0) continue;
        cv::Point2d spot(c % frame.colour.cols, c / frame.colour.cols);
        for (const spotter::Detection& d : detections)
            if (d.classId == classId && d.box.contains(spot)) mask.data[i] = 1;
    }
    return mask;
}

//int main(){
//
//@brief - the program: 07d · step 1 of 3 · Which depth pixels are inside a person box?. YOLO's boxes live in the
//         1920 x 1080 colour picture; the depth pixels are 512 x 424. colourIndex (02_s3) says where each depth
//         pixel lands in colour, so test that spot against each person box. Setup, main loop and clean-up are laid
//         out in the file's sections. Returns 0 on a normal exit, 1 if it couldn't start.
int main()
{
    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;
    const int person = yolo.Classes().Find("person");   // by name, so another model's numbering works too

    const char* window = "07d step 1: depth pixels in a person box (red)";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, 1024, 848);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        cv::Mat inPerson = InsideBoxes(yolo.Detect(frame.colour), person, frame);
        cv::Mat picture = frame.registered.clone();
        picture.setTo(cv::Scalar(0, 0, 255), inPerson);
        cv::addWeighted(picture, 0.5, frame.registered, 0.5, 0, picture);   // a see-through tint
        cv::imshow(window, spotter::WithKeys(picture, Keys));
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: the red covers the whole box, wall behind you included. Step 2 adds depth, so only what is
// actually in front of the room counts.
