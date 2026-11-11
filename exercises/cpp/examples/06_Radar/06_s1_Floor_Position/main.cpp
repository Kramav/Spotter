// =====================================================================
//  06 · step 1 of 2 · Where on the floor is each object?
//
//  Each box already has a distance (02_s4). The box centre's offset from
//  the middle of the picture gives the sideways position: the pinhole
//  model on the colour camera, X = (u + 0.5 - cx) * Z / fx. So each object
//  gets a spot on the floor: X metres to the side, Z metres ahead.
//
//  New in this step: colourCamera.Unproject, floor positions.
//  Keys:  Q or Esc  quit
//  Next:  06_s2_Radar_Map (draw those spots from above)
// =====================================================================

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
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

//int main(){
//
//@brief - the program: 06 · step 1 of 2 · Where on the floor is each object?. Each box already has a distance
//         (02_s4). The box centre's offset from the middle of the picture gives the sideways position: the pinhole
//         model on the colour camera, X = (u + 0.5 - cx) * Z / fx. So each object gets a spot on the floor. Setup,
//         main loop and clean-up are laid out in the file's sections. Returns 0 on a normal exit, 1 if it couldn't
//         start.
int main()
{
    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    const char* window = "06 step 1: floor position (x = sideways, z = ahead)";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, 960, 540);
    spotter::BoxDepthSampler boxDepth;
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        boxDepth.Build(frame);
        for (const spotter::Detection& d : yolo.Detect(frame.colour))
        {
            std::optional<double> meters = boxDepth.MedianMeters(d.box);
            if (!meters) continue;
            cv::Point2f centre((float)(d.box.x + d.box.width / 2), (float)(d.box.y + d.box.height / 2));
            cv::Point3f p = frame.colourCamera.Unproject(centre, (float)*meters);   // new
            cv::rectangle(frame.colour, d.box, cv::Scalar(0, 255, 0), 3);
            cv::putText(frame.colour, cv::format("%s x %+.2f z %.2f m", yolo.Classes().Name(d.classId).c_str(), p.x, p.z),
                        d.box.tl() + cv::Point2d(4, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        }
        cv::imshow(window, spotter::WithKeys(frame.colour, Keys));
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: stand straight in front of the camera: x near 0. Step one big step left: x about -0.7.
// Then step back: z grows, x stays.
