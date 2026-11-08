// =====================================================================
//  05 · step 3 of 3 · A Kalman filter per box
//
//  Steps 1-2 on real detections: every track (04_s3) gets step 2's filter
//  on its box centre. Each frame all filters predict; matching then uses
//  the PREDICTED box (where the object should be now), not last frame's.
//  A track YOLO misses coasts on its prediction (thin box).
//
//  New in this step: match against predicted boxes; coasting boxes.
//  Keys:  Q or Esc  quit
//  Next:  05_Kalman (the full example adds distance and speed to the filter)
// =====================================================================

#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/track.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "Q or Esc  quit";

const double MinOverlap = 0.3, NoisePixels = 15, AccelerationPixels = 500, Dt = 1.0 / 6;   // YOLO: ~6 frames a second
const int MaxMissedFrames = 12;

struct Track
{
    int id, classId;
    cv::KalmanFilter kalman;   // x, y, speed x, speed y of the box centre
    cv::Size2d size;
    int missedFrames = 0;
    cv::Rect2d Box() const { return cv::Rect2d(kalman.statePost.at<float>(0) - size.width / 2, kalman.statePost.at<float>(1) - size.height / 2, size.width, size.height); }
};

cv::KalmanFilter NewFilter(cv::Point2d start)
{
    cv::KalmanFilter kalman(4, 2, 0, CV_32F);
    kalman.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, Dt, 0,   0, 1, 0, Dt,   0, 0, 1, 0,   0, 0, 0, 1);
    kalman.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0,   0, 1, 0, 0);
    cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar(NoisePixels * NoisePixels));
    double p = AccelerationPixels * Dt * Dt / 2, v = AccelerationPixels * Dt;
    kalman.processNoiseCov = cv::Mat::diag((cv::Mat_<float>(4, 1) << p * p, p * p, v * v, v * v));
    kalman.statePost = (cv::Mat_<float>(4, 1) << start.x, start.y, 0, 0);
    kalman.errorCovPost = cv::Mat::diag((cv::Mat_<float>(4, 1) << NoisePixels * NoisePixels, NoisePixels * NoisePixels, 1e6, 1e6));
    return kalman;
}

cv::Point2d Centre(const cv::Rect2d& box) { return { box.x + box.width / 2, box.y + box.height / 2 }; }

spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = 0.4f;
    return options;
}

void Update(std::vector<Track>& tracks, const std::vector<spotter::Detection>& detections, int& nextId)
{
    for (Track& t : tracks) t.kalman.predict();   // new: everyone moves on first
    std::vector<bool> trackUsed(tracks.size(), false), detectionUsed(detections.size(), false);
    while (true)
    {
        double best = MinOverlap;
        int bestTrack = -1, bestDetection = -1;
        for (size_t t = 0; t < tracks.size(); t++)
            for (size_t d = 0; d < detections.size(); d++)
            {
                if (trackUsed[t] || detectionUsed[d] || tracks[t].classId != detections[d].classId) continue;
                double overlap = spotter::Overlap(tracks[t].Box(), detections[d].box);   // new: the PREDICTED box
                if (overlap >= best) { best = overlap; bestTrack = (int)t; bestDetection = (int)d; }
            }
        if (bestTrack < 0) break;
        trackUsed[bestTrack] = detectionUsed[bestDetection] = true;
        Track& track = tracks[bestTrack];
        cv::Point2d c = Centre(detections[bestDetection].box);
        track.kalman.correct((cv::Mat_<float>(2, 1) << c.x, c.y));   // new: blend in the measurement
        track.size = detections[bestDetection].box.size();
        track.missedFrames = 0;
    }
    for (size_t t = 0; t < trackUsed.size(); t++) if (!trackUsed[t]) tracks[t].missedFrames++;
    for (size_t d = 0; d < detections.size(); d++)
        if (!detectionUsed[d]) tracks.push_back({ nextId++, detections[d].classId, NewFilter(Centre(detections[d].box)), detections[d].box.size() });
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [](const Track& t) { return t.missedFrames > MaxMissedFrames; }), tracks.end());
}

int main()
{
    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    const char* window = "05 step 3: a Kalman filter per box (thin = predicted)";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, 960, 540);
    std::vector<Track> tracks;
    int nextId = 1;
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        Update(tracks, yolo.Detect(frame.colour), nextId);
        for (const Track& t : tracks)
        {
            cv::Scalar colour = spotter::ColourForId(t.id);
            bool coasting = t.missedFrames > 0;
            cv::rectangle(frame.colour, t.Box(), colour, coasting ? 1 : 3);   // new: draw coasting boxes too, thin
            cv::putText(frame.colour, "#" + std::to_string(t.id) + " " + yolo.Classes().Name(t.classId), t.Box().tl() + cv::Point2d(4, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, colour, coasting ? 1 : 2);
        }
        cv::imshow(window, spotter::WithKeys(frame.colour, Keys));
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: walk across the view at the speed that gave you new IDs in 04_s3. Now you keep yours:
// your predicted box is already where you are. Duck behind a chair: the thin box keeps walking.
// (Dt is fixed at YOLO's ~6 frames a second here; the full example measures each frame's real time.)
