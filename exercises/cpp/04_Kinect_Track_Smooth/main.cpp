// =====================================================================
//  04 · Kinect Track + Smooth
//
//  Example 02 measures every box from scratch each frame, so the distances
//  jitter and nothing connects "the person" in one frame to the next.
//  This example adds one idea on top: follow each object over time.
//
//    1-3. Detect, register, sample   exactly as in example 02 (shortened here)
//    4.   Track    match this frame's boxes to last frame's by how much they
//                  overlap, so each object keeps an ID number:  #3 person
//    5.   Smooth   a running average of each object's own distance. Because
//                  it follows the OBJECT, not a pixel, it still works when
//                  the object or the camera moves (docs/decisions.md D12).
//
//  Window: one colour per ID. Label = smoothed distance, raw value in brackets.
//  Console: per object, how much the raw and smoothed distance change from
//           frame to frame. That's the jitter, and the smoothing should cut it.
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

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// --- Detection and depth sampling: same meaning as in example 02 ---
const std::string ModelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
const int ModelInputSize = 640;
const float MinConfidence = 0.40f;
const double BoxShrink = 0.5;
const int MinDepthSamples = 20;

// --- Tracking (new) ---
// Two boxes of the same class count as the same object when they overlap at least this much.
// Overlap = IoU, "intersection over union": 1 = identical boxes, 0 = not touching.
// Too high: a moving object keeps getting a new ID. Too low: two neighbours swap IDs.
const double MinOverlap = 0.3;

// How many frames an object may go unseen (hidden, or missed by YOLO) before its ID is dropped.
// At about 6 frames per second, 12 frames is about 2 seconds.
const int MaxMissedFrames = 12;

// --- Smoothing (new) ---
// Each frame, the smoothed distance moves only (1 - DistanceSmoothing) of the way toward the
// new raw measurement. 0 = no smoothing (raw), 0.8 = calm, 0.95 = very calm but slow to follow.
const double DistanceSmoothing = 0.8;

// --- Output ---
const double FpsSmoothing = 0.9;   // as in examples 01-03
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

// One detection in this frame: what, where in the colour image, and how far (0 = no depth).
struct Detection
{
    int classId;
    cv::Rect2d box;
    double rawMeters;
};

// One object being followed over time.
struct Track
{
    int id;
    int classId;
    cv::Rect2d box;            // where it was last seen
    int missedFrames = 0;      // frames in a row without a matching detection
    double rawMeters = 0;      // latest measurement (0 = none yet)
    double smoothedMeters = 0; // running average (0 = none yet)

    // Jitter statistics since the last console report: average frame-to-frame change.
    double rawChangeSum = 0, smoothedChangeSum = 0;
    int changeCount = 0;
};

// Middle value of a list (as in example 02).
float Median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) return values[middle];
    return (values[middle - 1] + values[middle]) / 2;
}

// Intersection over union: the overlapping area divided by the area the two boxes cover together.
double Overlap(const cv::Rect2d& a, const cv::Rect2d& b)
{
    double intersection = (a & b).area();
    return intersection / (a.area() + b.area() - intersection);
}

// A fixed, distinct colour for each ID, so you can see when an object's ID changes.
cv::Scalar ColorForId(int id)
{
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar((id * 47) % 180, 255, 255)), bgr;   // hue steps around the colour wheel
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(c[0], c[1], c[2]);
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // print immediately, even when piped (see example 01)

    // -----------------------------------------------------------------
    //  Setup: detector and Kinect (as in example 02)
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

    std::vector<Track> tracks;   // every object currently being followed
    int nextId = 1;

    const std::string windowName = "04 Kinect track + smooth  (Q quits)";
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

        // --- Capture + Step 2: Register (as in example 02) -------------------
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

        // --- Step 1: Detect (as in example 02) -------------------------------
        cv::Mat scaled, letterboxed;
        cv::resize(frame, scaled, cv::Size(scaledWidth, scaledHeight));
        cv::copyMakeBorder(scaled, letterboxed, padTop, ModelInputSize - scaledHeight - padTop,
                           padLeft, ModelInputSize - scaledWidth - padLeft, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        detector.setInput(cv::dnn::blobFromImage(letterboxed, 1.0 / 255.0, cv::Size(ModelInputSize, ModelInputSize),
                                                 cv::Scalar(0, 0, 0), true, false));
        cv::Mat output = detector.forward();
        cv::Mat rows(output.size[1], output.size[2], CV_32F, output.ptr<float>());

        // --- Step 3: Sample each box's centre for its raw distance (as in 02) --
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

        // --- Step 4: Track -----------------------------------------------------
        // Greedy matching: of all (track, detection) pairs with the same class, take the pair
        // that overlaps most, pair them up, and repeat until no pair overlaps at least MinOverlap.
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
                    double overlap = Overlap(tracks[t].box, detections[d].box);
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

            // --- Step 5: Smooth this object's distance ---------------------
            Track& track = tracks[bestTrack];
            const Detection& detection = detections[bestDetection];
            track.box = detection.box;
            track.missedFrames = 0;
            if (detection.rawMeters > 0)
            {
                if (track.smoothedMeters == 0)
                {
                    track.smoothedMeters = detection.rawMeters;   // first measurement: nothing to average yet
                }
                else
                {
                    double newSmoothed = DistanceSmoothing * track.smoothedMeters + (1 - DistanceSmoothing) * detection.rawMeters;
                    track.rawChangeSum += std::abs(detection.rawMeters - track.rawMeters);
                    track.smoothedChangeSum += std::abs(newSmoothed - track.smoothedMeters);
                    track.changeCount++;
                    track.smoothedMeters = newSmoothed;
                }
                track.rawMeters = detection.rawMeters;
            }
        }

        // Detections nobody matched are new objects: give each a new ID.
        for (size_t d = 0; d < detections.size(); d++)
        {
            if (detectionMatched[d]) continue;

            // Why a new ID? If an object of the same class went unmatched this frame, it was
            // probably the same thing. Print how much they overlapped: below MinOverlap means it
            // moved too far between frames, or was last seen too long ago.
            int closestId = -1, closestMissed = 0;
            double closestOverlap = 0;
            for (size_t t = 0; t < trackMatched.size(); t++)
            {
                if (trackMatched[t] || tracks[t].classId != detections[d].classId) continue;
                double overlap = Overlap(tracks[t].box, detections[d].box);
                if (closestId < 0 || overlap > closestOverlap)
                {
                    closestId = tracks[t].id;
                    closestOverlap = overlap;
                    closestMissed = tracks[t].missedFrames + 1;
                }
            }
            if (closestId >= 0)
            {
                std::printf("  new ID #%d %s | unmatched %s #%d: overlap %.2f (needs %.2f), unseen for %d frame(s)\n",
                            nextId, ClassNames[detections[d].classId], ClassNames[detections[d].classId],
                            closestId, closestOverlap, MinOverlap, closestMissed);
            }

            Track track;
            track.id = nextId++;
            track.classId = detections[d].classId;
            track.box = detections[d].box;
            track.rawMeters = track.smoothedMeters = detections[d].rawMeters;
            tracks.push_back(track);
        }

        // Tracks nobody matched were not seen this frame. Forget them after MaxMissedFrames.
        // (New tracks were added after trackMatched was sized, so they are skipped here.)
        for (size_t t = 0; t < trackMatched.size(); t++)
        {
            if (!trackMatched[t]) tracks[t].missedFrames++;
        }
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                    [](const Track& t) { return t.missedFrames > MaxMissedFrames; }),
                     tracks.end());

        // -----------------------------------------------------------------
        //  Output
        // -----------------------------------------------------------------

        for (const Track& track : tracks)
        {
            if (track.missedFrames > 0) continue;   // only draw what is visible right now
            cv::Scalar color = ColorForId(track.id);
            std::string label = "#" + std::to_string(track.id) + " " + ClassNames[track.classId] + " "
                              + (track.smoothedMeters > 0 ? cv::format("%.2fm (raw %.2f)", track.smoothedMeters, track.rawMeters) : "?m");
            cv::rectangle(frame, track.box, color, 3);
            cv::putText(frame, label, cv::Point((int)track.box.x, std::max((int)track.box.y - 10, 30)),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
        }

        auto frameTime = std::chrono::steady_clock::now();
        double newestFps = 1.0 / std::chrono::duration<double>(frameTime - lastFrameTime).count();
        lastFrameTime = frameTime;
        liveFps = (liveFps == 0) ? newestFps : FpsSmoothing * liveFps + (1 - FpsSmoothing) * newestFps;
        cv::putText(frame, cv::format("%.1f fps", liveFps), cv::Point(10, frame.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        cv::imshow(windowName, frame);

        if (frameTime - reportStart >= std::chrono::milliseconds(ReportEveryMs))
        {
            std::printf("%.1f fps | %d objects tracked, next new ID will be #%d\n", liveFps, (int)tracks.size(), nextId);
            for (Track& track : tracks)
            {
                if (track.changeCount == 0) continue;
                // Average frame-to-frame change, in millimetres: how much the number jumps around.
                std::printf("  #%-3d %-12s raw %.3f m, smoothed %.3f m | change per frame: raw %5.1f mm, smoothed %5.1f mm\n",
                            track.id, ClassNames[track.classId], track.rawMeters, track.smoothedMeters,
                            1000 * track.rawChangeSum / track.changeCount, 1000 * track.smoothedChangeSum / track.changeCount);
                track.rawChangeSum = track.smoothedChangeSum = 0;
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
//  1. Sit still. Watch the console: the smoothed "change per frame" should be a fraction of the
//     raw one. Try DistanceSmoothing = 0, 0.5, 0.8, 0.95 and compare.
//  2. Walk slowly toward the camera with DistanceSmoothing = 0.95. The smoothed value lags behind
//     the raw one: smoothing trades jitter for delay. A Kalman filter (example 05, planned)
//     predicts the motion instead of just averaging, so it lags much less.
//  3. Walk quickly across the view. If your box jumps further than it overlaps between frames
//     (about 6 per second here), you get a new ID. Lower MinOverlap and try again.
//  4. Walk behind a chair or out of view for a second and come back. Within MaxMissedFrames you
//     keep your ID; after that you're a new object. Set MaxMissedFrames = 0 to see the difference.
//  5. Two people cross paths: do they keep their IDs, or swap? Matching only by overlap can't
//     tell them apart. That's what appearance features (or a Kalman prediction) fix.
//  6. Add a speed readout: (smoothed distance now - smoothed distance 1 s ago). Try it with
//     DistanceSmoothing = 0 first and see why the smoothing matters for speed.
