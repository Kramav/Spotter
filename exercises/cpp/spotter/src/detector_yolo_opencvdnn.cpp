#include "spotter/detector_yolo_opencvdnn.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <opencv2/imgproc.hpp>

namespace spotter
{

Letterbox Letterbox::For(cv::Size picture, int inputSize)
{
    Letterbox box;
    box.scale = std::min((double)inputSize / picture.width, (double)inputSize / picture.height);
    box.padLeft = (inputSize - (int)std::round(picture.width * box.scale)) / 2;
    box.padTop = (inputSize - (int)std::round(picture.height * box.scale)) / 2;
    return box;
}

cv::Rect2d Letterbox::ToPicture(double x1, double y1, double x2, double y2) const
{
    cv::Point2d topLeft((x1 - padLeft) / scale, (y1 - padTop) / scale);
    cv::Point2d bottomRight((x2 - padLeft) / scale, (y2 - padTop) / scale);
    return cv::Rect2d(topLeft, bottomRight);
}

std::vector<Detection> DecodeYolo(const cv::Mat& output, YoloLayout layout, const Letterbox& letterbox,
                                  float minConfidence, float maxOverlap)
{
    std::vector<Detection> detections;
    // The output is 1 x A x B: view it as A rows of B numbers.
    cv::Mat table(output.size[1], output.size[2], CV_32F, (void*)output.ptr<float>());

    if (layout == YoloLayout::EndToEnd)
    {
        // One row per final box: x1, y1, x2, y2, confidence, classId. The network already removed duplicates.
        for (int row = 0; row < table.rows; row++)
        {
            const float* r = table.ptr<float>(row);
            if (r[4] < minConfidence) continue;
            detections.push_back({ (int)r[5], r[4], letterbox.ToPicture(r[0], r[1], r[2], r[3]) });
        }
        return detections;
    }

    // Classic: one COLUMN per candidate (8400 of them): centre x, centre y, width, height, then
    // one score per class. Keep each candidate's best class, then remove duplicates per class.
    cv::Mat candidates = table.t();
    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    for (int row = 0; row < candidates.rows; row++)
    {
        const float* r = candidates.ptr<float>(row);
        cv::Mat classScores(1, candidates.cols - 4, CV_32F, (void*)(r + 4));
        cv::Point best;
        double score;
        cv::minMaxLoc(classScores, nullptr, &score, nullptr, &best);
        if (score < minConfidence) continue;
        boxes.push_back(letterbox.ToPicture(r[0] - r[2] / 2, r[1] - r[3] / 2, r[0] + r[2] / 2, r[1] + r[3] / 2));
        scores.push_back((float)score);
        classIds.push_back(best.x);
    }
    std::vector<int> kept;
    cv::dnn::NMSBoxesBatched(boxes, scores, classIds, minConfidence, maxOverlap, kept);
    for (int i : kept) detections.push_back({ classIds[i], scores[i], boxes[i] });
    return detections;
}

YoloDetector::YoloDetector(const Options& options) : options(options), classes(options.namesPath)
{
    if (!std::filesystem::exists(options.modelPath))
    {
        std::printf("Model not found: %s\n", std::filesystem::absolute(options.modelPath).string().c_str());
        std::printf("Export it once from the repo root:\n");
        std::printf("  yolo export model=yolo26n.pt format=onnx nms=False imgsz=%d\n", options.inputSize);
        return;
    }
    net = cv::dnn::readNetFromONNX(options.modelPath);
    ready = !net.empty();
    std::printf("Loaded %s (%d x %d input, %zu classes)\n", options.modelPath.c_str(), options.inputSize,
                options.inputSize, classes.Size());
}

std::vector<Detection> YoloDetector::Detect(const cv::Mat& bgr)
{
    if (!ready || bgr.empty()) return {};
    // Letterbox: shrink the picture to fit the square keeping its shape, and fill the rest with
    // grey (114), as Ultralytics does in Python.
    Letterbox letterbox = Letterbox::For(bgr.size(), options.inputSize);
    cv::Mat scaled, square;
    cv::resize(bgr, scaled, cv::Size((int)std::round(bgr.cols * letterbox.scale), (int)std::round(bgr.rows * letterbox.scale)));
    cv::copyMakeBorder(scaled, square, letterbox.padTop, options.inputSize - scaled.rows - letterbox.padTop,
                       letterbox.padLeft, options.inputSize - scaled.cols - letterbox.padLeft,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    // Pixel values 0..1, RGB order, laid out 1 x 3 x size x size.
    net.setInput(cv::dnn::blobFromImage(square, 1.0 / 255.0, square.size(), cv::Scalar(), true, false));
    return DecodeYolo(net.forward(), options.layout, letterbox, options.minConfidence, options.maxOverlap);
}

AsyncDetector::AsyncDetector(YoloDetector& detector) : detector(detector), thread(&AsyncDetector::Run, this) {}

AsyncDetector::~AsyncDetector()
{
    {
        std::lock_guard<std::mutex> lock(shared);
        quit = true;
    }
    handedOver.notify_one();
    thread.join();
}

void AsyncDetector::Offer(const cv::Mat& bgr)
{
    std::lock_guard<std::mutex> lock(shared);
    if (busy) return;
    picture = bgr.clone();   // a copy: the caller keeps drawing on its own
    pictureWaiting = busy = true;
    handedOver.notify_one();
}

std::optional<std::vector<Detection>> AsyncDetector::TryTake()
{
    std::lock_guard<std::mutex> lock(shared);
    std::optional<std::vector<Detection>> taken;
    taken.swap(result);
    return taken;
}

double AsyncDetector::LastMs() const
{
    std::lock_guard<std::mutex> lock(shared);
    return lastMs;
}

void AsyncDetector::Run()
{
    while (true)
    {
        cv::Mat image;
        {
            std::unique_lock<std::mutex> lock(shared);
            handedOver.wait(lock, [this] { return pictureWaiting || quit; });   // sleep until there's work
            if (quit) return;
            image = picture;
            pictureWaiting = false;
        }
        auto start = std::chrono::steady_clock::now();
        std::vector<Detection> found = detector.Detect(image);   // the slow part, without the lock
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::lock_guard<std::mutex> lock(shared);
        result = std::move(found);
        lastMs = ms;
        busy = false;
    }
}

}
