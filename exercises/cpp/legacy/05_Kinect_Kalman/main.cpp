// =====================================================================
//  05 · Kinect Kalman
//
//  Example 04 matched each box to where the object WAS last frame, and
//  smoothed its distance with a running average. Two problems:
//    - at ~6 frames per second, a walking person moves so far between frames
//      that the old and new boxes barely overlap -> new ID
//    - averaging lags behind anything that moves
//
//  One new idea fixes both: a Kalman filter per object. It keeps a guess of
//  where the object is AND how fast it's moving, plus how sure it is, and
//  repeats two steps every frame:
//
//    Predict   move the guess forward: position += speed * time. Being a
//              guess, it becomes less sure.
//    Correct   blend in this frame's measurement. Whichever is more certain,
//              the measurement or the prediction, gets more weight.
//
//  Matching now uses the PREDICTED box, so fast movers keep their ID. When
//  YOLO misses an object, the prediction keeps it moving ("coasting"), and
//  its speed comes for free: how fast it approaches or moves away.
//
//  Window: thick box = measured this frame, thin box = predicted only.
//          Label: #id, filtered distance, speed toward/away from the camera.
//  Console: raw vs filtered jitter (compare with example 04), ID changes.
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

// --- Detection and depth sampling: same meaning as in example 02 ---
const std::string ModelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
const int ModelInputSize = 640;
const float MinConfidence = 0.40f;
const double BoxShrink = 0.5;
const int MinDepthSamples = 20;

// --- Tracking: same meaning as in example 04, so the two compare fairly ---
const double MinOverlap = 0.3;
const int MaxMissedFrames = 12;

// --- Kalman filter (new) ---
// How much the MEASUREMENTS wobble: YOLO's box centre (pixels) and the sampled depth (metres).
// Bigger = trust measurements less = smoother but slower to react.
const double MeasurementNoisePixels = 15.0;
const double MeasurementNoiseMeters = 0.03;

// How suddenly objects can change speed (acceleration). Bigger = trust the prediction less =
// follows sudden moves faster but jitters more. A walking person: about 1 m/s^2, which is
// roughly 500 px/s^2 in the 1920-wide colour image at 2 m.
const double AccelerationPixels = 500.0;
const double AccelerationMeters = 1.0;

// --- Output ---
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

// One detection in this frame (as in example 04).
struct Detection
{
    int classId;
    cv::Rect2d box;
    double rawMeters;   // 0 = no depth
};

// One object being followed. The Kalman filter's state has 6 numbers:
//     [ centreX, centreY, metres,  speedX, speedY, speedMetres ]
//       (pixels) (pixels) (depth)  (px/s)  (px/s)  (m/s)
// It measures 3 of them each frame: centreX, centreY, metres.
struct Track
{
    int id;
    int classId;
    cv::KalmanFilter kalman;
    cv::Size2d boxSize;       // the filter tracks the centre; the size is copied from the last detection
    int missedFrames = 0;
    double rawMeters = 0;

    // Jitter statistics since the last console report (as in example 04).
    double lastFilteredMeters = 0, rawChangeSum = 0, filteredChangeSum = 0;
    int changeCount = 0;

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

// A new Kalman filter for an object first seen at this centre and distance, standing still.
cv::KalmanFilter NewKalman(double centreX, double centreY, double meters)
{
    cv::KalmanFilter kalman(6, 3, 0, CV_32F);   // 6 state numbers, 3 measured

    // Measurement matrix H: the measurement is simply the first 3 state numbers.
    kalman.measurementMatrix = cv::Mat::eye(3, 6, CV_32F);

    // R: how noisy each measurement is (a variance = noise squared).
    kalman.measurementNoiseCov = (cv::Mat_<float>(3, 3) <<
        MeasurementNoisePixels * MeasurementNoisePixels, 0, 0,
        0, MeasurementNoisePixels * MeasurementNoisePixels, 0,
        0, 0, MeasurementNoiseMeters * MeasurementNoiseMeters);

    // Starting guess: where we saw it, not moving.
    kalman.statePost = (cv::Mat_<float>(6, 1) << centreX, centreY, meters, 0, 0, 0);

    // P: how unsure the starting guess is. Position: about one measurement's noise.
    // Speed: we have no idea yet, so very unsure (up to ~1000 px/s, ~2 m/s).
    // No depth yet (meters = 0)? Then the distance is unknown too: very unsure, so the first
    // real depth measurement replaces the 0 at once instead of being averaged with it.
    double metersVariance = meters > 0 ? MeasurementNoiseMeters * MeasurementNoiseMeters : 100.0;
    kalman.errorCovPost = cv::Mat::diag((cv::Mat_<float>(6, 1) <<
        MeasurementNoisePixels * MeasurementNoisePixels, MeasurementNoisePixels * MeasurementNoisePixels,
        metersVariance, 1000 * 1000, 1000 * 1000, 2 * 2));
    return kalman;
}

// The time between frames changes, so the "how things move" matrices are rebuilt every frame.
void SetTimeStep(cv::KalmanFilter& kalman, double dt)
{
    // Transition matrix F: position += speed * dt, speed stays the same (constant velocity).
    cv::setIdentity(kalman.transitionMatrix);
    for (int i = 0; i < 3; i++) kalman.transitionMatrix.at<float>(i, i + 3) = (float)dt;

    // Q: how much the real motion can drift from "constant speed" during dt, caused by an
    // acceleration a: position by about a*dt^2/2, speed by about a*dt.
    // ponytail: diagonal approximation of the textbook Q; the full one adds position-speed correlation terms.
    double accelerations[3] = { AccelerationPixels, AccelerationPixels, AccelerationMeters };
    kalman.processNoiseCov = cv::Mat::zeros(6, 6, CV_32F);
    for (int i = 0; i < 3; i++)
    {
        double a = accelerations[i];
        kalman.processNoiseCov.at<float>(i, i) = (float)std::pow(a * dt * dt / 2, 2);
        kalman.processNoiseCov.at<float>(i + 3, i + 3) = (float)std::pow(a * dt, 2);
    }
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // print immediately, even when piped (see example 01)

    // -----------------------------------------------------------------
    //  Setup: detector and Kinect (as in examples 02 and 04)
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

    const std::string windowName = "05 Kinect Kalman  (Q quits)";
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

        // --- Capture + register, detect, sample (as in example 04) -----------
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
        double dt = std::chrono::duration<double>(frameTime - lastFrameTime).count();   // seconds since last frame
        lastFrameTime = frameTime;

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
            if (rows.at<float>(row, 4) < MinConfidence) continue;
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
            detections.push_back({ (int)rows.at<float>(row, 5), box, rawMeters });
        }

        // --- Step 4: Predict -----------------------------------------------------
        // Move every object forward in time. After predict(), the filter's best guess (statePost)
        // IS the prediction, so an object YOLO misses this frame simply keeps moving ("coasting").
        for (Track& track : tracks)
        {
            SetTimeStep(track.kalman, dt);
            track.kalman.predict();
        }

        // --- Step 5: Match, against the PREDICTED boxes ----------------------------
        // Same greedy overlap matching as example 04. The only change: Box() is where the object
        // should be NOW, not where it was last frame.
        std::vector<bool> trackMatched(tracks.size(), false), detectionMatched(detections.size(), false);
        while (true)
        {
            double bestOverlap = MinOverlap;
            int bestTrack = -1, bestDetection = -1;
            for (size_t t = 0; t < tracks.size(); t++)
            {
                for (size_t d = 0; d < detections.size(); d++)
                {
                    if (trackMatched[t] || detectionMatched[d] || tracks[t].classId != detections[d].classId) continue;
                    double overlap = Overlap(tracks[t].Box(), detections[d].box);
                    if (overlap >= bestOverlap)
                    {
                        bestOverlap = overlap;
                        bestTrack = (int)t;
                        bestDetection = (int)d;
                    }
                }
            }
            if (bestTrack < 0) break;
            trackMatched[bestTrack] = detectionMatched[bestDetection] = true;

            // --- Step 6: Correct with this frame's measurement ------------------------
            Track& track = tracks[bestTrack];
            const Detection& detection = detections[bestDetection];
            double measuredX = detection.box.x + detection.box.width / 2;
            double measuredY = detection.box.y + detection.box.height / 2;
            double measuredMeters = detection.rawMeters;

            // No depth this frame? Feed in the prediction's own distance with a huge noise, so the
            // filter ignores it and keeps its guess. (Centre X and Y still correct normally.)
            bool haveDepth = measuredMeters > 0;
            if (!haveDepth)
            {
                measuredMeters = track.Meters();
                track.kalman.measurementNoiseCov.at<float>(2, 2) = 1e6f;
            }
            track.kalman.correct((cv::Mat_<float>(3, 1) << measuredX, measuredY, measuredMeters));
            track.kalman.measurementNoiseCov.at<float>(2, 2) = (float)(MeasurementNoiseMeters * MeasurementNoiseMeters);

            track.boxSize = detection.box.size();
            track.missedFrames = 0;
            if (haveDepth)
            {
                if (track.rawMeters > 0)
                {
                    track.rawChangeSum += std::abs(detection.rawMeters - track.rawMeters);
                    track.filteredChangeSum += std::abs(track.Meters() - track.lastFilteredMeters);
                    track.changeCount++;
                }
                track.rawMeters = detection.rawMeters;
                track.lastFilteredMeters = track.Meters();
            }
        }

        // Unmatched detections: new objects, each with a new filter. Print why, as in example 04.
        for (size_t d = 0; d < detections.size(); d++)
        {
            if (detectionMatched[d]) continue;

            int closestId = -1, closestMissed = 0;
            double closestOverlap = 0;
            for (size_t t = 0; t < trackMatched.size(); t++)
            {
                if (trackMatched[t] || tracks[t].classId != detections[d].classId) continue;
                double overlap = Overlap(tracks[t].Box(), detections[d].box);
                if (closestId < 0 || overlap > closestOverlap)
                {
                    closestId = tracks[t].id;
                    closestOverlap = overlap;
                    closestMissed = tracks[t].missedFrames + 1;
                }
            }
            if (closestId >= 0)
            {
                std::printf("  new ID #%d %s | unmatched %s #%d: overlap %.2f with its PREDICTED box (needs %.2f), unseen for %d frame(s)\n",
                            nextId, ClassNames[detections[d].classId], ClassNames[detections[d].classId],
                            closestId, closestOverlap, MinOverlap, closestMissed);
            }
            else
            {
                // Nothing of this class is being tracked: either it's really new, or its old ID was
                // already dropped (see the "dropped" lines above).
                std::printf("  new ID #%d %s | no %s was being tracked\n",
                            nextId, ClassNames[detections[d].classId], ClassNames[detections[d].classId]);
            }

            const Detection& detection = detections[d];
            Track track;
            track.id = nextId++;
            track.classId = detection.classId;
            track.boxSize = detection.box.size();
            track.kalman = NewKalman(detection.box.x + detection.box.width / 2, detection.box.y + detection.box.height / 2,
                                     detection.rawMeters);
            track.rawMeters = track.lastFilteredMeters = detection.rawMeters;
            tracks.push_back(std::move(track));
        }

        // Unmatched tracks coast on their prediction. Forget them after MaxMissedFrames.
        for (size_t t = 0; t < trackMatched.size(); t++)
        {
            if (!trackMatched[t]) tracks[t].missedFrames++;
            if (tracks[t].missedFrames > MaxMissedFrames)
            {
                std::printf("  dropped #%d %s: unseen for %d frames in a row (MaxMissedFrames = %d)\n",
                            tracks[t].id, ClassNames[tracks[t].classId], tracks[t].missedFrames, MaxMissedFrames);
            }
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
        cv::putText(frame, cv::format("%.1f fps", liveFps), cv::Point(10, frame.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        cv::imshow(windowName, frame);

        if (frameTime - reportStart >= std::chrono::milliseconds(ReportEveryMs))
        {
            std::printf("%.1f fps | %d objects tracked, next new ID will be #%d\n", liveFps, (int)tracks.size(), nextId);
            for (Track& track : tracks)
            {
                if (track.changeCount == 0) continue;
                std::printf("  #%-3d %-12s raw %.3f m, filtered %.3f m, speed %+.2f m/s | change per frame: raw %5.1f mm, filtered %5.1f mm\n",
                            track.id, ClassNames[track.classId], track.rawMeters, track.Meters(), track.MetersPerSecond(),
                            1000 * track.rawChangeSum / track.changeCount, 1000 * track.filteredChangeSum / track.changeCount);
                track.rawChangeSum = track.filteredChangeSum = 0;
                track.changeCount = 0;
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
//  1. Run example 04, walk across the view, and count the ID changes. Then do the same here.
//     The console's "new ID" lines show the overlap with the predicted box: usually much higher.
//  2. Walk toward the camera and back. Watch "approaching" / "away" and the speed. Normal walking
//     is about 1-1.5 m/s. Then compare the lag: 04's smoothed distance trails behind, this one
//     keeps up, because it predicts instead of only averaging.
//  3. Step behind something for a second. Your box keeps moving thin ("predicted") in the
//     direction you were going, then snaps back to thick when you reappear.
//  4. Set MeasurementNoiseMeters = 0.3: the filter trusts the depth less, so the distance gets
//     very smooth but slow. Set it to 0.003: it follows every wobble, like raw.
//  5. Set AccelerationMeters = 0.1: the filter believes objects barely change speed. Stop
//     suddenly and watch the predicted distance overshoot before it corrects.
//  6. Stand still and read the speed. It should hover near 0.00 m/s. How far from zero does it
//     wander? That's the noise left in the speed estimate.
