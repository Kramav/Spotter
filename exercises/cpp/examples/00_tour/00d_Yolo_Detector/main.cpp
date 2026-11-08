// =====================================================================
//  00d · Tour: YoloDetector and ClassList     (needs the Kinect, or a picture)
//
//  What it's for: finding objects in a colour picture. Detect(picture)
//  returns spotter::Detection {classId, confidence, box}; ClassList turns
//  the classId into a name ("person", "cup").
//
//  How it works (spotter/src/detector_yolo_opencvdnn.cpp, the only file that
//  uses OpenCV's DNN module):
//    1. Letterbox: shrink the picture into the model's square (640 x 640),
//       keeping its shape, and pad with grey.
//    2. blobFromImage: pixel values 0..1, RGB order, 1 x 3 x 640 x 640.
//    3. forward(): the network runs.
//    4. DecodeYolo: read the output table and undo the letterbox, so boxes
//       are in the original picture's pixels.
//
//  Swapping the model is settings only (YoloDetector::Options):
//    modelPath + inputSize   another export, e.g. yolo26n_320.onnx at 320
//    namesPath               its class names, one per line (yolo26n.names)
//    layout                  EndToEnd (YOLO26, nms=False: 1 x 300 x 6) or
//                            Classic (older YOLOs: 1 x (4 + classes) x 8400)
//  Another runtime (NCNN on the Pi) = a sibling file, detector_yolo_ncnn.
//
//  Run:  00d_Yolo_Detector             the Kinect, live (nothing is saved)
//        00d_Yolo_Detector photo.bmp   one picture of your own instead (BMP: this OpenCV
//                                      build has no JPEG or PNG codecs, D29)
//  Keys: Q or Esc  quit
// =====================================================================

#include <chrono>
#include <cstdio>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "Q or Esc  quit";

const std::string Window = "00d YoloDetector  (Q quits)";

spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.inputSize = 640;
    options.layout = spotter::YoloLayout::EndToEnd;
    options.minConfidence = 0.4f;
    return options;
}

// Detect, time it, draw and list what was found.
void DetectAndShow(spotter::YoloDetector& yolo, cv::Mat& picture, bool print)
{
    auto start = std::chrono::steady_clock::now();
    std::vector<spotter::Detection> detections = yolo.Detect(picture);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (print) std::printf("%zu detections in %.0f ms:\n", detections.size(), ms);
    for (const spotter::Detection& d : detections)
    {
        std::string label = cv::format("%s %.2f", yolo.Classes().Name(d.classId).c_str(), d.confidence);
        if (print) std::printf("  class %2d  %-20s box (%.0f, %.0f) %.0f x %.0f\n", d.classId, label.c_str(), d.box.x, d.box.y, d.box.width, d.box.height);
        cv::rectangle(picture, d.box, cv::Scalar(0, 255, 0), 3);
        cv::putText(picture, label, d.box.tl() + cv::Point2d(4, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow(Window, spotter::WithKeys(picture, Keys));
}

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    spotter::YoloDetector yolo(YoloSettings());
    if (!yolo.IsReady()) return 1;
    std::printf("%zu class names; \"person\" is class %d.\n", yolo.Classes().Size(), yolo.Classes().Find("person"));
    cv::namedWindow(Window, cv::WINDOW_NORMAL);
    cv::resizeWindow(Window, 960, 540);

    if (argc > 1)   // one picture from disk
    {
        cv::Mat picture = cv::imread(argv[1]);
        if (picture.empty()) { std::printf("Could not read %s\n", argv[1]); return 1; }
        DetectAndShow(yolo, picture, true);
        cv::waitKey(0);
        return 0;
    }

    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!kinect.IsOpen()) return 1;
    spotter::Every report(1000);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        DetectAndShow(yolo, frame.colour, report.Due());
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}
