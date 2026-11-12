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
#include <exception>
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

    //static Letterbox For(cv::Size picture, int inputSize){
    //
    //@param1 - picture: the size of the picture to fit
    //@param2 - inputSize: the model's square input size (e.g. 640)
    //
    //@brief - the scale that fits the whole picture into the square (the smaller of the two ratios) and the
    //         padding that centres it. O(1).
    static Letterbox For(cv::Size picture, int inputSize);

    //cv::Rect2d ToPicture(double x1, double y1, double x2, double y2){
    //
    //@param1 - x1: the box's left edge, model pixels
    //@param2 - y1: the box's top edge, model pixels
    //@param3 - x2: the box's right edge, model pixels
    //@param4 - y2: the box's bottom edge, model pixels
    //
    //@brief - a box in the model's square -> the same box in the original picture's pixels (padding taken
    //         off, then divided by the scale). O(1).
    cv::Rect2d ToPicture(double x1, double y1, double x2, double y2) const;
};

//std::vector<Detection> DecodeYolo(const cv::Mat& output, YoloLayout layout, const Letterbox& letterbox, float minConfidence, float maxOverlap){
//
//@param1 - output: the network's raw output tensor, 1 x A x B floats
//@param2 - layout: EndToEnd (1 x 300 x 6: x1 y1 x2 y2 confidence class per row) or Classic (1 x (4 + classes) x 8400)
//@param3 - letterbox: how the picture was fitted into the model's square, to map boxes back
//@param4 - minConfidence: boxes below this are dropped
//@param5 - maxOverlap: Classic only: of two same-class boxes overlapping more than this (IoU), keep the better (default 0.45)
//
//@brief - the model's raw output -> detections in picture pixels. EndToEnd: one pass over the rows, O(300).
//         Classic: each candidate's best class (O(8400 x classes)), then non-maximum suppression per class
//         (cv::dnn::NMSBoxesBatched, O(k^2) for k boxes above minConfidence). Separate from YoloDetector so
//         a check can feed it a made-up tensor.
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

    //YoloDetector(const Options& options){
    //
    //@param1 - options: the model file, names file, input size, layout and thresholds
    //
    //@brief - loads the model and its class names. Prints what's wrong (with the export command) if it
    //         can't; then IsReady() is false.
    explicit YoloDetector(const Options& options);

    //bool IsReady(){
    //
    //@brief - true if the model loaded. O(1).
    bool IsReady() const { return ready; }

    //const ClassList& Classes(){
    //
    //@brief - the model's class names. O(1).
    const ClassList& Classes() const { return classes; }

    //std::vector<Detection> Detect(const cv::Mat& bgr){
    //
    //@param1 - bgr: a BGR picture of any size
    //
    //@brief - letterboxes the picture into the model's square (grey 114 padding), scales it to 0..1 RGB,
    //         runs the network and decodes it (DecodeYolo). Boxes come back in the picture's pixels.
    //         The network's cost dominates: about 170 ms at 640 on the Ryzen.
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
    //AsyncDetector(YoloDetector& detector){
    //
    //@param1 - detector: the detector to run; it must outlive this
    //
    //@brief - starts the background thread, which waits for a picture. O(1).
    explicit AsyncDetector(YoloDetector& detector);

    //~AsyncDetector(){
    //
    //@brief - tells the thread to quit and joins it (waits for a run in progress to finish).
    ~AsyncDetector();
    AsyncDetector(const AsyncDetector&) = delete;
    AsyncDetector& operator=(const AsyncDetector&) = delete;

    //void Offer(const cv::Mat& bgr){
    //
    //@param1 - bgr: the newest colour picture
    //
    //@brief - if the detector is idle, copies the picture and wakes the thread; if it's busy, does nothing.
    //         O(pixels) for the copy when taken, O(1) otherwise.
    void Offer(const cv::Mat& bgr);

    //std::optional<std::vector<Detection>> TryTake(){
    //
    //@brief - a finished batch of detections, handed over once; empty if none finished since the last call.
    //         If the detector threw on its thread (an OpenCV error in the model), the error is thrown here,
    //         on the caller's thread, where the program's own error handling can see it. O(1).
    std::optional<std::vector<Detection>> TryTake();

    //double LastMs(){
    //
    //@brief - how long the last batch took, in milliseconds. O(1).
    double LastMs() const;

private:
    //void Run(){
    //
    //@brief - the thread's loop: wait for a picture (or quit), detect, store the batch (or the error) and its time, repeat.
    void Run();
    YoloDetector& detector;
    mutable std::mutex shared;              // everything below is touched only under this lock
    std::condition_variable handedOver;
    cv::Mat picture;
    bool pictureWaiting = false, busy = false, quit = false;
    std::optional<std::vector<Detection>> result;
    std::exception_ptr failure;             // an error from the thread, thrown again by TryTake
    double lastMs = 0;
    std::thread thread;                     // last: it starts after everything above exists
};

}
