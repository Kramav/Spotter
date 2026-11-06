// =====================================================================
//  05c · Kinect Background Detect
//
//  Examples 02-05b run YOLO on every frame and wait for it: ~150 ms, so
//  everything (picture, tracking, distances) runs at ~6 fps, while the
//  Kinect delivers 30. YOLO is ~95% of each frame.
//
//  One idea: stop waiting for YOLO. Run it on a second thread.
//
//    Detector thread   takes the newest frame, runs YOLO (~6 times a second),
//                      hands back boxes, takes the next frame, repeat.
//    Main loop         runs at the camera's 30 fps. Every frame it:
//                        - predicts every box forward (the Kalman filter, 05)
//                        - re-measures each object's distance from FRESH depth
//                        - folds in YOLO's boxes whenever a new batch arrives
//                          (two-pass matching, as in 05b)
//
//  So the picture, the boxes and the distances move at 30 fps, and YOLO
//  only has to "recognise" things a few times a second.
//
//  Two settings to compare speed:
//    DetectInBackground = false   -> waits for YOLO every frame, like 05b
//    ModelInputSize     = 320     -> a smaller YOLO, ~4x faster, misses small things
//
//  Window: thick box = seen in YOLO's latest batch, thin = predicted only.
//          Bottom-left: display fps and YOLO batches per second.
//
//  Keys:   Q or Esc  quit
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/registration.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video/tracking.hpp>   // cv::KalmanFilter

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// --- Speed (new) ---
// true = YOLO on its own thread, main loop at camera speed. false = wait for YOLO every frame.
const bool DetectInBackground = true;

// YOLO's input size: 640 or 320. 320 is ~4x faster but misses small and far-away objects.
// Each size is its own model file (an ONNX model's input size is fixed when it's exported).
const int ModelInputSize = 640;
const std::string ModelPath = std::string(SPOTTER_REPO_ROOT) + (ModelInputSize == 320 ? "/yolo26n_320.onnx" : "/yolo26n.onnx");

// In a dim room the colour camera's automatic exposure needs more than one frame's time, and the
// Kinect drops to 15 fps. true = cap the exposure so it keeps 30 fps: a darker, grainier picture.
const bool CapColorExposure = true;

// --- Memory (changed from 05b: seconds, because frames and YOLO batches now differ) ---
const double MaxUnseenSeconds = 5.0;      // forget an object YOLO hasn't seen for this long
const double CoastSlowdownSeconds = 0.5;  // as in 05b

// --- Unchanged from example 05b ---
const float StartConfidence = 0.40f;
const float KeepConfidence = 0.15f;
const double BoxShrink = 0.5;
const int MinDepthSamples = 20;
const double MinOverlap = 0.3;
const double MeasurementNoisePixels = 15.0;
const double MeasurementNoiseMeters = 0.03;
const double AccelerationPixels = 500.0;
const double AccelerationMeters = 1.0;
const double FpsSmoothing = 0.9;
const int ReportEveryMs = 1000;

// The 80 COCO classes yolo26n.pt was trained on, in classId order.
const char* ClassNames[] =
{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
};

// What YOLO hands back: no depth yet, the main loop measures that.
struct Detection
{
    int classId;
    float confidence;
    cv::Rect2d box;
};

struct Track
{
    int id;
    int classId;
    cv::KalmanFilter kalman;
    cv::Size2d boxSize;
    int missedBatches = 0;         // YOLO batches in a row that didn't include it
    double secondsUnseen = 0;      // time since YOLO last saw it

    double CentreX() const { return kalman.statePost.at<float>(0); }
    double CentreY() const { return kalman.statePost.at<float>(1); }
    double Meters() const { return kalman.statePost.at<float>(2); }
    double MetersPerSecond() const { return kalman.statePost.at<float>(5); }
    cv::Rect2d Box() const
    {
        return cv::Rect2d(CentreX() - boxSize.width / 2, CentreY() - boxSize.height / 2, boxSize.width, boxSize.height);
    }
};

float Median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) return values[middle];
    return (values[middle - 1] + values[middle]) / 2;
}

double Overlap(const cv::Rect2d& a, const cv::Rect2d& b)
{
    double intersection = (a & b).area();
    return intersection / (a.area() + b.area() - intersection);
}

cv::Scalar ColorForId(int id)
{
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar((id * 47) % 180, 255, 255)), bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(c[0], c[1], c[2]);
}

// YOLO on one colour frame (letterbox, forward, undo letterbox: as in example 02).
// Runs on the detector thread, so it only touches its own arguments.
std::vector<Detection> Detect(cv::dnn::Net& detector, const cv::Mat& frame)
{
    double scale = std::min((double)ModelInputSize / frame.cols, (double)ModelInputSize / frame.rows);
    int scaledWidth = (int)std::round(frame.cols * scale), scaledHeight = (int)std::round(frame.rows * scale);
    int padLeft = (ModelInputSize - scaledWidth) / 2, padTop = (ModelInputSize - scaledHeight) / 2;
    cv::Mat scaled, letterboxed;
    cv::resize(frame, scaled, cv::Size(scaledWidth, scaledHeight));
    cv::copyMakeBorder(scaled, letterboxed, padTop, ModelInputSize - scaledHeight - padTop,
                       padLeft, ModelInputSize - scaledWidth - padLeft, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    detector.setInput(cv::dnn::blobFromImage(letterboxed, 1.0 / 255.0, cv::Size(ModelInputSize, ModelInputSize),
                                             cv::Scalar(0, 0, 0), true, false));
    cv::Mat output = detector.forward();
    cv::Mat rows(output.size[1], output.size[2], CV_32F, output.ptr<float>());

    std::vector<Detection> detections;
    for (int row = 0; row < rows.rows; row++)
    {
        float confidence = rows.at<float>(row, 4);
        if (confidence < KeepConfidence) continue;
        double left = (rows.at<float>(row, 0) - padLeft) / scale, top = (rows.at<float>(row, 1) - padTop) / scale;
        double right = (rows.at<float>(row, 2) - padLeft) / scale, bottom = (rows.at<float>(row, 3) - padTop) / scale;
        detections.push_back({ (int)rows.at<float>(row, 5), confidence, cv::Rect2d(left, top, right - left, bottom - top) });
    }
    return detections;
}

// Median depth of the centre part of a box, in metres (as in example 02). 0 = not enough depth.
// ponytail: scans every depth pixel per box (~3 ms); with many objects, invert the map once per frame.
double SampleMeters(const cv::Rect2d& box, const float* depthMm, const std::vector<int>& colorIndexOfDepthPixel, int colorWidth)
{
    cv::Rect2d sample(box.x + box.width * (1 - BoxShrink) / 2, box.y + box.height * (1 - BoxShrink) / 2,
                      box.width * BoxShrink, box.height * BoxShrink);
    std::vector<float> samplesMm;
    for (size_t i = 0; i < colorIndexOfDepthPixel.size(); i++)
    {
        int colorIndex = colorIndexOfDepthPixel[i];
        if (colorIndex < 0 || !(depthMm[i] > 0) || std::isinf(depthMm[i])) continue;
        if (sample.contains(cv::Point2d(colorIndex % colorWidth, colorIndex / colorWidth))) samplesMm.push_back(depthMm[i]);
    }
    return (int)samplesMm.size() >= MinDepthSamples ? Median(samplesMm) / 1000.0 : 0;
}

// As in example 05.
cv::KalmanFilter NewKalman(double centreX, double centreY, double meters)
{
    cv::KalmanFilter kalman(6, 3, 0, CV_32F);
    kalman.measurementMatrix = cv::Mat::eye(3, 6, CV_32F);
    kalman.measurementNoiseCov = (cv::Mat_<float>(3, 3) <<
        MeasurementNoisePixels * MeasurementNoisePixels, 0, 0,
        0, MeasurementNoisePixels * MeasurementNoisePixels, 0,
        0, 0, MeasurementNoiseMeters * MeasurementNoiseMeters);
    kalman.statePost = (cv::Mat_<float>(6, 1) << centreX, centreY, meters, 0, 0, 0);
    double metersVariance = meters > 0 ? MeasurementNoiseMeters * MeasurementNoiseMeters : 100.0;
    kalman.errorCovPost = cv::Mat::diag((cv::Mat_<float>(6, 1) <<
        MeasurementNoisePixels * MeasurementNoisePixels, MeasurementNoisePixels * MeasurementNoisePixels,
        metersVariance, 1000 * 1000, 1000 * 1000, 2 * 2));
    return kalman;
}

// As in example 05.
void SetTimeStep(cv::KalmanFilter& kalman, double dt)
{
    cv::setIdentity(kalman.transitionMatrix);
    for (int i = 0; i < 3; i++) kalman.transitionMatrix.at<float>(i, i + 3) = (float)dt;
    double accelerations[3] = { AccelerationPixels, AccelerationPixels, AccelerationMeters };
    kalman.processNoiseCov = cv::Mat::zeros(6, 6, CV_32F);
    for (int i = 0; i < 3; i++)
    {
        double a = accelerations[i];
        kalman.processNoiseCov.at<float>(i, i) = (float)std::pow(a * dt * dt / 2, 2);
        kalman.processNoiseCov.at<float>(i + 3, i + 3) = (float)std::pow(a * dt, 2);
    }
}

// Correct a track's filter. Any measurement given as "unknown" gets a huge noise, so the filter
// keeps its own guess for it (the trick from example 05, now also used for X and Y).
void Correct(Track& track, bool haveBox, double centreX, double centreY, double meters)
{
    cv::Mat& noise = track.kalman.measurementNoiseCov;
    if (!haveBox) { centreX = track.CentreX(); centreY = track.CentreY(); noise.at<float>(0, 0) = noise.at<float>(1, 1) = 1e6f; }
    if (meters <= 0) { meters = track.Meters(); noise.at<float>(2, 2) = 1e6f; }
    track.kalman.correct((cv::Mat_<float>(3, 1) << centreX, centreY, meters));
    noise.at<float>(0, 0) = noise.at<float>(1, 1) = (float)(MeasurementNoisePixels * MeasurementNoisePixels);
    noise.at<float>(2, 2) = (float)(MeasurementNoiseMeters * MeasurementNoiseMeters);
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // print immediately, even when piped (see example 01)

    // -----------------------------------------------------------------
    //  Setup: detector and Kinect (as in example 05b)
    // -----------------------------------------------------------------

    if (!std::filesystem::exists(ModelPath))
    {
        std::printf("Model not found: %s\n", ModelPath.c_str());
        return 1;
    }
    cv::dnn::Net detector = cv::dnn::readNetFromONNX(ModelPath);
    std::printf("Loaded %s (input %d x %d), detecting %s\n", ModelPath.c_str(), ModelInputSize, ModelInputSize,
                DetectInBackground ? "in the background" : "on every frame (waiting for it)");

    libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Warning));
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device* device = freenect2.enumerateDevices() > 0 ? freenect2.openDefaultDevice() : nullptr;
    if (device == nullptr)
    {
        std::printf("No Kinect v2 found, or it is in use. See the main README's troubleshooting table.\n");
        return 1;
    }
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Depth);
    device->setColorFrameListener(&listener);
    device->setIrAndDepthFrameListener(&listener);
    device->startStreams(true, true);
    if (CapColorExposure) device->setColorSemiAutoExposure(16.0f);   // at most ~16 ms per picture (see the setting)

    const int colorWidth = 1920, colorHeight = 1080, depthWidth = 512, depthHeight = 424;
    libfreenect2::Registration registration(device->getIrCameraParams(), device->getColorCameraParams());
    libfreenect2::Frame undistorted(depthWidth, depthHeight, 4), registered(depthWidth, depthHeight, 4);
    std::vector<int> colorIndexOfDepthPixel(depthWidth * depthHeight);

    // -----------------------------------------------------------------
    //  Setup: the detector thread (new)
    // -----------------------------------------------------------------
    // The two threads share only the variables below, and touch them only while holding the
    // mutex (a lock: one thread at a time). The condition variable lets the detector thread
    // sleep until the main loop hands it a frame.
    std::mutex shared;
    std::condition_variable frameHandedOver;
    cv::Mat frameForDetector;                 // main -> detector
    bool frameWaiting = false;
    std::vector<Detection> detectorResult;    // detector -> main
    bool resultWaiting = false;
    double lastDetectMs = 0;
    bool quit = false;

    std::thread detectorThread;
    if (DetectInBackground)
    {
        detectorThread = std::thread([&]
        {
            while (true)
            {
                cv::Mat image;
                {
                    std::unique_lock<std::mutex> lock(shared);
                    frameHandedOver.wait(lock, [&] { return frameWaiting || quit; });   // sleep until there's work
                    if (quit) return;
                    image = frameForDetector;
                    frameWaiting = false;
                }
                auto start = std::chrono::steady_clock::now();
                std::vector<Detection> result = Detect(detector, image);   // the slow part, done without the lock
                double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                {
                    std::lock_guard<std::mutex> lock(shared);
                    detectorResult = std::move(result);
                    resultWaiting = true;
                    lastDetectMs = ms;
                }
            }
        });
    }
    bool detectorBusy = false;   // main loop only: has it handed over a frame and not got the answer yet?

    std::vector<Track> tracks;
    int nextId = 1;

    const std::string windowName = "05c Kinect background detect  (Q quits)";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, colorWidth / 2, colorHeight / 2);

    auto reportStart = std::chrono::steady_clock::now();
    auto lastFrameTime = std::chrono::steady_clock::now();
    double liveFps = 0;
    int batchesSinceReport = 0;
    double workMsSum = 0, sampleMsSum = 0, showMsSum = 0;   // where the main loop's time goes
    int framesSinceReport = 0;
    cv::Mat frame;

    // -----------------------------------------------------------------
    //  Main operation: runs at the camera's speed
    // -----------------------------------------------------------------

    libfreenect2::FrameMap frames;
    while (true)
    {
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;   // 27 = Esc

        // --- Capture + register (as in example 05b) ---------------------------
        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }
        libfreenect2::Frame* colorFrame = frames[libfreenect2::Frame::Color];
        cv::cvtColor(cv::Mat(colorHeight, colorWidth, CV_8UC4, colorFrame->data), frame, cv::COLOR_BGRA2BGR);
        registration.apply(colorFrame, frames[libfreenect2::Frame::Depth], &undistorted, &registered,
                           false, nullptr, colorIndexOfDepthPixel.data());
        const float* depthMm = (const float*)undistorted.data;
        listener.release(frames);

        auto frameTime = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(frameTime - lastFrameTime).count();
        lastFrameTime = frameTime;
        auto workStart = frameTime;

        // --- Predict every object forward (as in 05b, unseen ones slow down) ----
        for (Track& track : tracks)
        {
            SetTimeStep(track.kalman, dt);
            track.kalman.predict();
            track.secondsUnseen += dt;
            if (track.missedBatches > 0)
            {
                float fade = (float)std::exp(-dt / CoastSlowdownSeconds);
                for (int i = 3; i < 6; i++)
                {
                    track.kalman.statePost.at<float>(i) *= fade;
                    track.kalman.statePre.at<float>(i) *= fade;
                }
            }
        }

        // --- YOLO: hand over this frame / collect a finished batch (new) ----------
        bool haveBatch = false;
        std::vector<Detection> detections;
        if (DetectInBackground)
        {
            std::lock_guard<std::mutex> lock(shared);
            if (resultWaiting)
            {
                detections = std::move(detectorResult);
                resultWaiting = false;
                detectorBusy = false;
                haveBatch = true;
            }
            if (!detectorBusy)   // idle: give it the newest frame (a copy: the main loop keeps drawing on `frame`)
            {
                frameForDetector = frame.clone();
                frameWaiting = true;
                detectorBusy = true;
                frameHandedOver.notify_one();
            }
        }
        else
        {
            auto start = std::chrono::steady_clock::now();
            detections = Detect(detector, frame);   // wait for it, like examples 02-05b
            lastDetectMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            haveBatch = true;
        }
        // ponytail: a background batch describes a frame that's ~1 YOLO-run old (~150 ms at 640).
        // Treated as current here; the fix would be to correct the filter at that older time.

        // --- A new batch: match it, as in example 05b ------------------------------
        std::vector<bool> correctedThisFrame(tracks.size(), false);
        if (haveBatch)
        {
            batchesSinceReport++;
            std::vector<bool> detectionMatched(detections.size(), false);
            for (int pass = 0; pass < 2; pass++)   // pass 0: strong detections, pass 1: weak ones
            {
                float minConfidence = pass == 0 ? StartConfidence : KeepConfidence;
                float maxConfidence = pass == 0 ? 2.0f : StartConfidence;
                while (true)
                {
                    double bestOverlap = MinOverlap;
                    int bestTrack = -1, bestDetection = -1;
                    for (size_t t = 0; t < tracks.size(); t++)
                    {
                        for (size_t d = 0; d < detections.size(); d++)
                        {
                            const Detection& detection = detections[d];
                            if (detection.confidence < minConfidence || detection.confidence >= maxConfidence) continue;
                            if (correctedThisFrame[t] || detectionMatched[d] || tracks[t].classId != detection.classId) continue;
                            double overlap = Overlap(tracks[t].Box(), detection.box);
                            if (overlap >= bestOverlap) { bestOverlap = overlap; bestTrack = (int)t; bestDetection = (int)d; }
                        }
                    }
                    if (bestTrack < 0) break;
                    correctedThisFrame[bestTrack] = detectionMatched[bestDetection] = true;

                    Track& track = tracks[bestTrack];
                    const Detection& detection = detections[bestDetection];
                    Correct(track, true, detection.box.x + detection.box.width / 2, detection.box.y + detection.box.height / 2,
                            SampleMeters(detection.box, depthMm, colorIndexOfDepthPixel, colorWidth));
                    track.boxSize = detection.box.size();
                    track.missedBatches = 0;
                    track.secondsUnseen = 0;
                }
            }

            for (size_t t = 0; t < correctedThisFrame.size(); t++)
            {
                if (!correctedThisFrame[t]) tracks[t].missedBatches++;
            }

            // Strong unmatched detections start new objects (as in 05b).
            for (size_t d = 0; d < detections.size(); d++)
            {
                const Detection& detection = detections[d];
                if (detectionMatched[d] || detection.confidence < StartConfidence) continue;
                Track track;
                track.id = nextId++;
                track.classId = detection.classId;
                track.boxSize = detection.box.size();
                track.kalman = NewKalman(detection.box.x + detection.box.width / 2, detection.box.y + detection.box.height / 2,
                                         SampleMeters(detection.box, depthMm, colorIndexOfDepthPixel, colorWidth));
                tracks.push_back(std::move(track));
                std::printf("  new ID #%d %s\n", tracks.back().id, ClassNames[detection.classId]);
            }
        }

        auto sampleStart = std::chrono::steady_clock::now();

        // --- Between batches: re-measure distance from this frame's depth (new) ----
        // Objects YOLO saw in its latest batch get their distance updated from fresh depth every
        // frame, at their predicted box. (Not unseen ones: their box may be on the wall by now.)
        for (size_t t = 0; t < correctedThisFrame.size(); t++)
        {
            Track& track = tracks[t];
            if (correctedThisFrame[t] || track.missedBatches > 0) continue;
            Correct(track, false, 0, 0, SampleMeters(track.Box(), depthMm, colorIndexOfDepthPixel, colorWidth));
        }

        sampleMsSum += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sampleStart).count();

        // Forget objects YOLO hasn't seen for MaxUnseenSeconds.
        for (const Track& track : tracks)
        {
            if (track.secondsUnseen > MaxUnseenSeconds)
                std::printf("  dropped #%d %s: unseen for %.1f s\n", track.id, ClassNames[track.classId], track.secondsUnseen);
        }
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                    [](const Track& t) { return t.secondsUnseen > MaxUnseenSeconds; }),
                     tracks.end());

        // -----------------------------------------------------------------
        //  Output
        // -----------------------------------------------------------------

        for (const Track& track : tracks)
        {
            bool coasting = track.missedBatches > 0;
            cv::Scalar color = ColorForId(track.id);
            std::string label = "#" + std::to_string(track.id) + " " + ClassNames[track.classId];
            if (track.Meters() > 0)
            {
                double speed = track.MetersPerSecond();
                label += cv::format(" %.2fm %s %.1fm/s", track.Meters(), speed < 0 ? "approaching" : "away", std::abs(speed));
            }
            if (coasting) label += " (predicted)";
            cv::rectangle(frame, track.Box(), color, coasting ? 1 : 3);
            cv::putText(frame, label, cv::Point((int)track.Box().x, std::max((int)track.Box().y - 10, 30)),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, color, coasting ? 1 : 2);
        }

        liveFps = (liveFps == 0) ? 1.0 / dt : FpsSmoothing * liveFps + (1 - FpsSmoothing) * (1.0 / dt);
        double reportSeconds = std::chrono::duration<double>(frameTime - reportStart).count();
        cv::putText(frame, cv::format("%.1f fps | YOLO %d ms", liveFps, (int)lastDetectMs), cv::Point(10, frame.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        auto showStart = std::chrono::steady_clock::now();
        cv::imshow(windowName, frame);
        auto workEnd = std::chrono::steady_clock::now();
        showMsSum += std::chrono::duration<double, std::milli>(workEnd - showStart).count();
        workMsSum += std::chrono::duration<double, std::milli>(workEnd - workStart).count();
        framesSinceReport++;

        if (reportSeconds * 1000 >= ReportEveryMs)
        {
            std::printf("%.1f fps display | YOLO: %.1f batches/s, %.0f ms each (input %d) | %d objects\n",
                        liveFps, batchesSinceReport / reportSeconds, lastDetectMs, ModelInputSize, (int)tracks.size());
            for (const Track& track : tracks)
            {
                std::printf("  #%-3d %-12s %s%s\n", track.id, ClassNames[track.classId],
                            track.Meters() > 0 ? cv::format("%.2f m", track.Meters()).c_str() : "?",
                            track.missedBatches > 0 ? cv::format("  (unseen %.1f s)", track.secondsUnseen).c_str() : "");
            }
            // The main loop must finish its work within one camera frame (33 ms) to keep up with 30 fps.
            std::printf("  main loop: %.0f ms per frame (camera frame = 33 ms), of which depth sampling %.0f ms, imshow %.0f ms\n",
                        workMsSum / framesSinceReport, sampleMsSum / framesSinceReport, showMsSum / framesSinceReport);
            reportStart = frameTime;
            batchesSinceReport = 0;
            workMsSum = sampleMsSum = showMsSum = 0;
            framesSinceReport = 0;
        }
    }

    // Stop the detector thread cleanly before the program ends.
    if (detectorThread.joinable())
    {
        {
            std::lock_guard<std::mutex> lock(shared);
            quit = true;
        }
        frameHandedOver.notify_one();
        detectorThread.join();
    }
    device->stop();
    device->close();
    cv::destroyAllWindows();
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Note the fps, then set DetectInBackground = false. Same YOLO speed, but now everything
//     waits for it: the fps drops to YOLO's rate.
//  2. Set ModelInputSize = 320. YOLO gets ~4x faster: more batches per second, so boxes snap to
//     objects more often. What does it stop finding? Try something small, or far away.
//  3. Walk across the view. Between YOLO batches your box keeps moving with the prediction;
//     each batch nudges it back. Is the box behind you? That's the ~150 ms age of each batch.
//  4. Watch the distance while you sway toward and away from the camera: it updates every
//     frame from fresh depth, not just when YOLO runs.
//  5. With 320, compare the display fps on the PC with what the Pi manages (Pi bring-up).
//  6. In a dim room, set CapColorExposure = false: the display drops to 15 fps, because the
//     camera itself slows down to collect more light. Does YOLO find more with the brighter
//     picture? That's the trade.
