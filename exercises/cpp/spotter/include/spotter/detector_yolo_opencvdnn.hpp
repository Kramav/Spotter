// spotter/detector_yolo_opencvdnn.hpp: a YOLO model run by OpenCV's DNN module (adapter, D34).
// The only place that uses cv::dnn. Another runtime (NCNN on the Pi) = a sibling file,
// detector_yolo_ncnn, with the same Options and Detect().
//
// A different model, or one you trained yourself, needs no code change, only Options:
//   modelPath   the .onnx file (OpenCV 4.14 can't load dynamic-size exports: one file per inputSize)
//   inputSize   the square size it was exported with (imgsz)
//   namesPath   its class names, one per line, in training order
//   layout      EndToEnd for YOLO26 exported with nms=False (1 x 300 x 6),
//               Classic for older exports (1 x (4 + classes) x 8400, duplicates removed here)
#pragma once

#include "spotter/detection.hpp"

#include <opencv2/dnn.hpp>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace spotter
{

enum class YoloLayout { EndToEnd, Classic };

// How a picture was fitted into the model's square: shrunk by `scale`, then padded.
struct Letterbox
{
    double scale = 1;
    int padLeft = 0, padTop = 0;

    static Letterbox For(cv::Size picture, int inputSize);
    cv::Rect2d ToPicture(double x1, double y1, double x2, double y2) const;   // model pixels -> picture pixels
};

// The model's raw output -> detections in picture pixels, at least minConfidence.
// Separate from YoloDetector so a check can feed it a made-up tensor.
std::vector<Detection> DecodeYolo(const cv::Mat& output, YoloLayout layout, const Letterbox& letterbox,
                                  float minConfidence, float maxOverlap = 0.45f);

class YoloDetector
{
public:
    struct Options
    {
        std::string modelPath;
        std::string namesPath;
        int inputSize = 640;
        YoloLayout layout = YoloLayout::EndToEnd;
        float minConfidence = 0.25f;   // boxes below this are dropped
        float maxOverlap = 0.45f;      // Classic only: of two same-class boxes overlapping more, keep the better
    };

    // Loads the model. Prints what's wrong (with the export command) if it can't; then IsReady() is false.
    explicit YoloDetector(const Options& options);

    bool IsReady() const { return ready; }
    const ClassList& Classes() const { return classes; }

    // One BGR picture of any size -> its detections, boxes in that picture's pixels.
    std::vector<Detection> Detect(const cv::Mat& bgr);

private:
    Options options;
    ClassList classes;
    cv::dnn::Net net;
    bool ready = false;
};

// Runs a YoloDetector on its own thread (example 05c), so the caller's loop keeps the camera's
// 30 fps while YOLO runs a few times a second. Each frame: TryTake() a finished batch if there is
// one, then Offer() the newest picture (taken only when the detector is idle).
// A batch describes a picture ~one detector run old.
class AsyncDetector
{
public:
    explicit AsyncDetector(YoloDetector& detector);   // the detector must outlive this
    ~AsyncDetector();                                 // stops and joins the thread
    AsyncDetector(const AsyncDetector&) = delete;
    AsyncDetector& operator=(const AsyncDetector&) = delete;

    void Offer(const cv::Mat& bgr);                       // copies it, if the detector is idle
    std::optional<std::vector<Detection>> TryTake();      // a finished batch, once
    double LastMs() const;                                // how long the last batch took

private:
    void Run();
    YoloDetector& detector;
    mutable std::mutex shared;              // everything below is touched only under this lock
    std::condition_variable handedOver;
    cv::Mat picture;
    bool pictureWaiting = false, busy = false, quit = false;
    std::optional<std::vector<Detection>> result;
    double lastMs = 0;
    std::thread thread;                     // last: it starts after everything above exists
};

}
