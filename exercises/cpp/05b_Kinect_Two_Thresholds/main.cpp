// =====================================================================
//  05b · Kinect Two Thresholds
//
//  Example 05's logs showed why IDs reset: YOLO's confidence that you're a
//  "person" wobbles around the 0.40 cut-off (half in view, very close,
//  turned away). Below 0.40 the detection is thrown away, so after ~2 s of
//  that your ID is dropped, and you come back as someone new.
//
//  One idea fixes it (it's the core of the "ByteTrack" tracker):
//  use TWO confidence levels.
//
//    strong  (>= StartConfidence)   may start a new ID, as before
//    weak    (>= KeepConfidence)    may only CONTINUE an existing ID
//
//  Matching runs twice: strong detections first, then the weak ones for
//  whatever is still unmatched. A half-visible you at 0.25 keeps your ID;
//  a random 0.25 blip on the wall never becomes a new object.
//
//  Plus a small companion change, so the longer memory works: an unseen
//  object's predicted speed fades out, so its box slows to a stop instead
//  of drifting away while it waits to be found again.
//
//  Everything else is example 05. Run both and compare the "new ID" lines.
//
//  Window: thick box = strong match, medium = weak match ("weak 0.23"),
//          thin = predicted only.
//  Console: per object, how many times this second a weak detection kept it.
//
//  Keys:   Q or Esc  quit
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
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

// --- Two thresholds (new; example 05 had one MinConfidence = 0.40) ---
// Only detections this confident can start a new ID.
const float StartConfidence = 0.40f;
// Detections this confident can keep an existing ID going. Set equal to StartConfidence to get
// example 05's behaviour back.
const float KeepConfidence = 0.15f;

// --- Memory (changed) ---
// Example 05: 12 frames (~2 s). Now ~5 s, which only works because of CoastSlowdownSeconds.
const int MaxMissedFrames = 30;
// While an object is unseen, its predicted speed fades: after this many seconds it's down to
// about a third (1/e). Small = stops quickly. Very large = example 05 (keeps drifting).
const double CoastSlowdownSeconds = 0.5;

// --- Unchanged from example 05 ---
const std::string ModelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
const int ModelInputSize = 640;
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

struct Detection
{
    int classId;
    float confidence;   // new: kept, because strong and weak detections are treated differently
    cv::Rect2d box;
    double rawMeters;   // 0 = no depth
};

// As in example 05, plus how it was matched this frame and a weak-match counter.
struct Track
{
    int id;
    int classId;
    cv::KalmanFilter kalman;
    cv::Size2d boxSize;
    int missedFrames = 0;
    float lastConfidence = 0;   // of the detection it matched this frame
    int weakMatches = 0;        // since the last console report

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

// As in example 05's correct step: blend one detection into a track's filter.
void Correct(Track& track, const Detection& detection)
{
    double measuredMeters = detection.rawMeters;
    if (measuredMeters <= 0)
    {
        measuredMeters = track.Meters();   // no depth: feed back the guess with a huge noise
        track.kalman.measurementNoiseCov.at<float>(2, 2) = 1e6f;
    }
    track.kalman.correct((cv::Mat_<float>(3, 1) << detection.box.x + detection.box.width / 2,
                          detection.box.y + detection.box.height / 2, measuredMeters));
    track.kalman.measurementNoiseCov.at<float>(2, 2) = (float)(MeasurementNoiseMeters * MeasurementNoiseMeters);
    track.boxSize = detection.box.size();
    track.missedFrames = 0;
    track.lastConfidence = detection.confidence;
}

// One matching pass, as in example 05: greedily pair the best-overlapping (track, detection) of
// the same class, but only detections with confidence in [minConfidence, maxConfidence).
void MatchPass(std::vector<Track>& tracks, const std::vector<Detection>& detections,
               std::vector<bool>& trackMatched, std::vector<bool>& detectionMatched,
               float minConfidence, float maxConfidence)
{
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
                if (trackMatched[t] || detectionMatched[d] || tracks[t].classId != detection.classId) continue;
                double overlap = Overlap(tracks[t].Box(), detection.box);
                if (overlap >= bestOverlap)
                {
                    bestOverlap = overlap;
                    bestTrack = (int)t;
                    bestDetection = (int)d;
                }
            }
        }
        if (bestTrack < 0) return;
        trackMatched[bestTrack] = detectionMatched[bestDetection] = true;
        Correct(tracks[bestTrack], detections[bestDetection]);
    }
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // print immediately, even when piped (see example 01)

    // -----------------------------------------------------------------
    //  Setup: detector and Kinect (as in example 05)
    // -----------------------------------------------------------------

    if (!std::filesystem::exists(ModelPath))
    {
        std::printf("Model not found: %s\n", ModelPath.c_str());
        return 1;
    }
    cv::dnn::Net detector = cv::dnn::readNetFromONNX(ModelPath);

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

    const int colorWidth = 1920, colorHeight = 1080, depthWidth = 512, depthHeight = 424;
    libfreenect2::Registration registration(device->getIrCameraParams(), device->getColorCameraParams());
    libfreenect2::Frame undistorted(depthWidth, depthHeight, 4), registered(depthWidth, depthHeight, 4);
    std::vector<int> colorIndexOfDepthPixel(depthWidth * depthHeight);

    double modelScale = std::min((double)ModelInputSize / colorWidth, (double)ModelInputSize / colorHeight);
    int scaledWidth = (int)std::round(colorWidth * modelScale);
    int scaledHeight = (int)std::round(colorHeight * modelScale);
    int padLeft = (ModelInputSize - scaledWidth) / 2;
    int padTop = (ModelInputSize - scaledHeight) / 2;

    std::vector<Track> tracks;
    int nextId = 1;

    const std::string windowName = "05b Kinect two thresholds  (Q quits)";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, colorWidth / 2, colorHeight / 2);

    auto reportStart = std::chrono::steady_clock::now();
    auto lastFrameTime = std::chrono::steady_clock::now();
    double liveFps = 0;
    cv::Mat frame;

    // -----------------------------------------------------------------
    //  Main operation
    // -----------------------------------------------------------------

    libfreenect2::FrameMap frames;
    while (true)
    {
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;   // 27 = Esc

        // --- Capture, register, detect (as in example 05) ---------------------
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

        cv::Mat scaled, letterboxed;
        cv::resize(frame, scaled, cv::Size(scaledWidth, scaledHeight));
        cv::copyMakeBorder(scaled, letterboxed, padTop, ModelInputSize - scaledHeight - padTop,
                           padLeft, ModelInputSize - scaledWidth - padLeft, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        detector.setInput(cv::dnn::blobFromImage(letterboxed, 1.0 / 255.0, cv::Size(ModelInputSize, ModelInputSize),
                                                 cv::Scalar(0, 0, 0), true, false));
        cv::Mat output = detector.forward();
        cv::Mat rows(output.size[1], output.size[2], CV_32F, output.ptr<float>());

        // --- Sample (as in example 05, but now keeping weak detections too) ----
        std::vector<Detection> detections;
        for (int row = 0; row < rows.rows; row++)
        {
            float confidence = rows.at<float>(row, 4);
            if (confidence < KeepConfidence) continue;   // example 05 cut at 0.40 here
            double left = (rows.at<float>(row, 0) - padLeft) / modelScale;
            double top = (rows.at<float>(row, 1) - padTop) / modelScale;
            double right = (rows.at<float>(row, 2) - padLeft) / modelScale;
            double bottom = (rows.at<float>(row, 3) - padTop) / modelScale;
            cv::Rect2d box(left, top, right - left, bottom - top);

            cv::Rect2d sample(left + box.width * (1 - BoxShrink) / 2, top + box.height * (1 - BoxShrink) / 2,
                              box.width * BoxShrink, box.height * BoxShrink);
            std::vector<float> samplesMm;
            for (int i = 0; i < depthWidth * depthHeight; i++)
            {
                int colorIndex = colorIndexOfDepthPixel[i];
                if (colorIndex < 0 || !(depthMm[i] > 0) || std::isinf(depthMm[i])) continue;
                if (sample.contains(cv::Point2d(colorIndex % colorWidth, colorIndex / colorWidth)))
                    samplesMm.push_back(depthMm[i]);
            }
            double rawMeters = (int)samplesMm.size() >= MinDepthSamples ? Median(samplesMm) / 1000.0 : 0;
            detections.push_back({ (int)rows.at<float>(row, 5), confidence, box, rawMeters });
        }

        // --- Predict (as in example 05), plus: unseen objects slow down (new) ----
        for (Track& track : tracks)
        {
            SetTimeStep(track.kalman, dt);
            track.kalman.predict();
            if (track.missedFrames > 0)
            {
                // Unseen last frame: let its speed fade, so its box comes to a stop and waits
                // where it was last heading, instead of drifting off for MaxMissedFrames.
                // (Both copies: statePost is the current guess, statePre is what correct() starts from.)
                float fade = (float)std::exp(-dt / CoastSlowdownSeconds);
                for (int i = 3; i < 6; i++)
                {
                    track.kalman.statePost.at<float>(i) *= fade;
                    track.kalman.statePre.at<float>(i) *= fade;
                }
            }
        }

        // --- Match in two passes (new) ----------------------------------------
        std::vector<bool> trackMatched(tracks.size(), false), detectionMatched(detections.size(), false);

        // Pass 1: strong detections, exactly like example 05.
        MatchPass(tracks, detections, trackMatched, detectionMatched, StartConfidence, 2.0f);

        // Pass 2: weak detections, but only for objects pass 1 left unmatched.
        std::vector<bool> matchedBefore = trackMatched;
        MatchPass(tracks, detections, trackMatched, detectionMatched, KeepConfidence, StartConfidence);
        for (size_t t = 0; t < tracks.size(); t++)
        {
            if (trackMatched[t] && !matchedBefore[t]) tracks[t].weakMatches++;
        }

        // Only STRONG unmatched detections start new IDs. Weak ones are dropped here: on their
        // own they're not trustworthy enough to be a new object.
        for (size_t d = 0; d < detections.size(); d++)
        {
            const Detection& detection = detections[d];
            if (detectionMatched[d] || detection.confidence < StartConfidence) continue;

            // Why a new ID? (as in example 05)
            int closestId = -1, closestMissed = 0;
            double closestOverlap = 0;
            for (size_t t = 0; t < trackMatched.size(); t++)
            {
                if (trackMatched[t] || tracks[t].classId != detection.classId) continue;
                double overlap = Overlap(tracks[t].Box(), detection.box);
                if (closestId < 0 || overlap > closestOverlap)
                {
                    closestId = tracks[t].id;
                    closestOverlap = overlap;
                    closestMissed = tracks[t].missedFrames + 1;
                }
            }
            if (closestId >= 0)
                std::printf("  new ID #%d %s | unmatched %s #%d: overlap %.2f with its PREDICTED box (needs %.2f), unseen for %d frame(s)\n",
                            nextId, ClassNames[detection.classId], ClassNames[detection.classId],
                            closestId, closestOverlap, MinOverlap, closestMissed);
            else
                std::printf("  new ID #%d %s | no %s was being tracked\n",
                            nextId, ClassNames[detection.classId], ClassNames[detection.classId]);

            Track track;
            track.id = nextId++;
            track.classId = detection.classId;
            track.boxSize = detection.box.size();
            track.lastConfidence = detection.confidence;
            track.kalman = NewKalman(detection.box.x + detection.box.width / 2, detection.box.y + detection.box.height / 2,
                                     detection.rawMeters);
            tracks.push_back(std::move(track));
        }

        // Unmatched objects coast. Forget them after MaxMissedFrames (as in example 05).
        for (size_t t = 0; t < trackMatched.size(); t++)
        {
            if (!trackMatched[t]) tracks[t].missedFrames++;
            if (tracks[t].missedFrames > MaxMissedFrames)
                std::printf("  dropped #%d %s: unseen for %d frames in a row (MaxMissedFrames = %d)\n",
                            tracks[t].id, ClassNames[tracks[t].classId], tracks[t].missedFrames, MaxMissedFrames);
        }
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                    [](const Track& t) { return t.missedFrames > MaxMissedFrames; }),
                     tracks.end());

        // -----------------------------------------------------------------
        //  Output
        // -----------------------------------------------------------------

        for (const Track& track : tracks)
        {
            bool coasting = track.missedFrames > 0;
            bool weak = !coasting && track.lastConfidence < StartConfidence;
            cv::Scalar color = ColorForId(track.id);
            std::string label = "#" + std::to_string(track.id) + " " + ClassNames[track.classId];
            if (track.Meters() > 0)
            {
                double speed = track.MetersPerSecond();
                label += cv::format(" %.2fm %s %.1fm/s", track.Meters(), speed < 0 ? "approaching" : "away", std::abs(speed));
            }
            if (weak) label += cv::format(" (weak %.2f)", track.lastConfidence);
            if (coasting) label += " (predicted)";
            int thickness = coasting ? 1 : (weak ? 2 : 3);
            cv::rectangle(frame, track.Box(), color, thickness);
            cv::putText(frame, label, cv::Point((int)track.Box().x, std::max((int)track.Box().y - 10, 30)),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, color, std::min(thickness, 2));
        }

        liveFps = (liveFps == 0) ? 1.0 / dt : FpsSmoothing * liveFps + (1 - FpsSmoothing) * (1.0 / dt);
        cv::putText(frame, cv::format("%.1f fps", liveFps), cv::Point(10, frame.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        cv::imshow(windowName, frame);

        if (frameTime - reportStart >= std::chrono::milliseconds(ReportEveryMs))
        {
            std::printf("%.1f fps | %d objects tracked, next new ID will be #%d\n", liveFps, (int)tracks.size(), nextId);
            for (Track& track : tracks)
            {
                std::printf("  #%-3d %-12s %s | kept by a weak detection %d time(s) this second%s\n",
                            track.id, ClassNames[track.classId],
                            track.Meters() > 0 ? cv::format("%.2f m", track.Meters()).c_str() : "?    ",
                            track.weakMatches, track.missedFrames > 0 ? "  (predicted)" : "");
                track.weakMatches = 0;
            }
            reportStart = frameTime;
        }
    }

    device->stop();
    device->close();
    cv::destroyAllWindows();
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Do whatever made your ID reset in example 05 (sit close, half out of view, turn away).
//     Count the "new ID ... person" lines here vs there. Watch the label switch to "weak".
//  2. Set KeepConfidence = StartConfidence: that's example 05's tracking again. IDs reset.
//  3. Set KeepConfidence = 0.05. Now almost anything keeps an ID alive, even when you've left:
//     your box may "stick" to a chair or a coat. Where's the sweet spot?
//  4. Leave the view for 3 seconds and come back near where you left. With MaxMissedFrames = 30
//     you keep your ID. Then set CoastSlowdownSeconds = 1000 (no slowdown): the predicted box
//     drifts away while you're gone, and you come back as someone new.
//  5. Why can't weak detections start new IDs? Set the new-ID check to KeepConfidence and
//     watch the ID numbers climb from blips on walls and shelves.
