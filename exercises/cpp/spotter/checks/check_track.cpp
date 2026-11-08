// Self-check for spotter::Tracker (track.hpp), with no Kinect: example 05c's own tracking code
// (copied below, unchanged apart from taking its settings and depth as arguments) and the
// library's Tracker run side by side on a made-up sequence (objects moving, missed by the
// detector, weak detections, one leaving, one arriving). Every frame they must agree: same IDs,
// same boxes, same distances, same coasting.
// Run: build\check_track.exe. Exit code 0 = passed.

#include "spotter/track.hpp"

#include <cmath>
#include <cstdio>
#include <random>

const float StartConfidence = 0.40f, KeepConfidence = 0.15f;
const double MinOverlap = 0.3, MaxUnseenSeconds = 1.0, CoastSlowdownSeconds = 0.5;
const double MeasurementNoisePixels = 15.0, MeasurementNoiseMeters = 0.03, AccelerationPixels = 500.0, AccelerationMeters = 1.0;

// --- The reference: example 05c's Track, NewKalman, SetTimeStep, Correct and main-loop steps ---
struct Track
{
    int id;
    int classId;
    cv::KalmanFilter kalman;
    cv::Size2d boxSize;
    int missedBatches = 0;
    double secondsUnseen = 0;
    double CentreX() const { return kalman.statePost.at<float>(0); }
    double CentreY() const { return kalman.statePost.at<float>(1); }
    double Meters() const { return kalman.statePost.at<float>(2); }
    cv::Rect2d Box() const { return cv::Rect2d(CentreX() - boxSize.width / 2, CentreY() - boxSize.height / 2, boxSize.width, boxSize.height); }
};

double Overlap(const cv::Rect2d& a, const cv::Rect2d& b)
{
    double intersection = (a & b).area();
    return intersection / (a.area() + b.area() - intersection);
}

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

void Correct(Track& track, bool haveBox, double centreX, double centreY, double meters)
{
    cv::Mat& noise = track.kalman.measurementNoiseCov;
    if (!haveBox) { centreX = track.CentreX(); centreY = track.CentreY(); noise.at<float>(0, 0) = noise.at<float>(1, 1) = 1e6f; }
    if (meters <= 0) { meters = track.Meters(); noise.at<float>(2, 2) = 1e6f; }
    track.kalman.correct((cv::Mat_<float>(3, 1) << centreX, centreY, meters));
    noise.at<float>(0, 0) = noise.at<float>(1, 1) = (float)(MeasurementNoisePixels * MeasurementNoisePixels);
    noise.at<float>(2, 2) = (float)(MeasurementNoiseMeters * MeasurementNoiseMeters);
}

struct Reference
{
    std::vector<Track> tracks;
    int nextId = 1;

    void Step(double dt, bool haveBatch, const std::vector<spotter::Detection>& detections, const spotter::BoxDepthSampler& depth)
    {
        auto SampleMeters = [&](const cv::Rect2d& box) { return depth.MedianMeters(box).value_or(0); };
        for (Track& track : tracks)
        {
            SetTimeStep(track.kalman, dt);
            track.kalman.predict();
            track.secondsUnseen += dt;
            if (track.missedBatches > 0)
            {
                float fade = (float)std::exp(-dt / CoastSlowdownSeconds);
                for (int i = 3; i < 6; i++) { track.kalman.statePost.at<float>(i) *= fade; track.kalman.statePre.at<float>(i) *= fade; }
            }
        }
        std::vector<bool> correctedThisFrame(tracks.size(), false);
        if (haveBatch)
        {
            std::vector<bool> detectionMatched(detections.size(), false);
            for (int pass = 0; pass < 2; pass++)
            {
                float minConfidence = pass == 0 ? StartConfidence : KeepConfidence;
                float maxConfidence = pass == 0 ? 2.0f : StartConfidence;
                while (true)
                {
                    double bestOverlap = MinOverlap;
                    int bestTrack = -1, bestDetection = -1;
                    for (size_t t = 0; t < tracks.size(); t++)
                        for (size_t d = 0; d < detections.size(); d++)
                        {
                            const spotter::Detection& detection = detections[d];
                            if (detection.confidence < minConfidence || detection.confidence >= maxConfidence) continue;
                            if (correctedThisFrame[t] || detectionMatched[d] || tracks[t].classId != detection.classId) continue;
                            double overlap = Overlap(tracks[t].Box(), detection.box);
                            if (overlap >= bestOverlap) { bestOverlap = overlap; bestTrack = (int)t; bestDetection = (int)d; }
                        }
                    if (bestTrack < 0) break;
                    correctedThisFrame[bestTrack] = detectionMatched[bestDetection] = true;
                    Track& track = tracks[bestTrack];
                    const spotter::Detection& detection = detections[bestDetection];
                    Correct(track, true, detection.box.x + detection.box.width / 2, detection.box.y + detection.box.height / 2, SampleMeters(detection.box));
                    track.boxSize = detection.box.size();
                    track.missedBatches = 0;
                    track.secondsUnseen = 0;
                }
            }
            for (size_t t = 0; t < correctedThisFrame.size(); t++) if (!correctedThisFrame[t]) tracks[t].missedBatches++;
            for (size_t d = 0; d < detections.size(); d++)
            {
                const spotter::Detection& detection = detections[d];
                if (detectionMatched[d] || detection.confidence < StartConfidence) continue;
                Track track;
                track.id = nextId++;
                track.classId = detection.classId;
                track.boxSize = detection.box.size();
                track.kalman = NewKalman(detection.box.x + detection.box.width / 2, detection.box.y + detection.box.height / 2, SampleMeters(detection.box));
                tracks.push_back(std::move(track));
            }
        }
        for (size_t t = 0; t < correctedThisFrame.size(); t++)
        {
            Track& track = tracks[t];
            if (correctedThisFrame[t] || track.missedBatches > 0) continue;
            Correct(track, false, 0, 0, SampleMeters(track.Box()));
        }
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [](const Track& t) { return t.secondsUnseen > MaxUnseenSeconds; }), tracks.end());
    }
};

int failures = 0;
void Check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAILED: %s\n", what); failures++; }
}

int main()
{
    // Depth that changes across the picture (1 m on the left .. 3.5 m on the right), mapped simply into colour.
    cv::Mat depthMm(424, 512, CV_32F);
    std::vector<int> colourIndex(512 * 424);
    for (int row = 0; row < 424; row++)
        for (int column = 0; column < 512; column++)
        {
            depthMm.at<float>(row, column) = 1000 + column * 5.0f;
            colourIndex[row * 512 + column] = (int)(row * 2.55) * 1920 + (int)(column * 3.75);
        }
    spotter::BoxDepthSampler depth;
    depth.Build(depthMm.ptr<float>(), colourIndex, 1920, 1080);

    spotter::Tracker::Options options;
    options.maxUnseenSeconds = MaxUnseenSeconds;
    spotter::Tracker tracker(options);
    Reference reference;

    std::mt19937 random(3);
    std::normal_distribution<double> jitter(0, 4);
    const double dt = 1.0 / 30;
    int compared = 0, maxTracks = 0, started = 0, dropped = 0;
    for (int frame = 0; frame < 240; frame++)
    {
        bool haveBatch = frame % 5 == 0;   // a detector batch every 5th frame, as on a background thread
        std::vector<spotter::Detection> detections;
        if (haveBatch)
        {
            int batch = frame / 5;
            // A: a person walking right, sometimes only weakly seen.
            detections.push_back({ 0, batch % 4 == 3 ? 0.2f : 0.7f, cv::Rect2d(200 + frame * 10 + jitter(random), 300 + jitter(random), 200, 500) });
            // B: a still TV, missed by the detector for batches 10-13, 0.8 s (it coasts, and keeps its ID).
            if (batch < 10 || batch > 13) detections.push_back({ 62, 0.55f, cv::Rect2d(1400 + jitter(random), 200 + jitter(random), 300, 180) });
            // C: a second person, from frame 100 to 150, then gone (dropped after a second).
            if (frame >= 100 && frame < 150) detections.push_back({ 0, 0.5f, cv::Rect2d(1000 - (frame - 100) * 6 + jitter(random), 350, 180, 450) });
            // A weak false alarm that must never start an ID.
            detections.push_back({ 41, 0.2f, cv::Rect2d(100, 100, 50, 50) });
        }

        reference.Step(dt, haveBatch, detections, depth);
        tracker.Predict(dt);
        if (haveBatch) tracker.Match(detections, depth);
        tracker.Remeasure(depth);
        for (const spotter::TrackEvent& e : tracker.TakeEvents()) (e.started ? started : dropped)++;

        const std::vector<spotter::Track>& tracks = tracker.Tracks();
        Check(tracks.size() == reference.tracks.size(), "same number of objects");
        if (tracks.size() != reference.tracks.size()) break;
        maxTracks = std::max(maxTracks, (int)tracks.size());
        for (size_t t = 0; t < tracks.size(); t++)
        {
            const Track& r = reference.tracks[t];
            Check(tracks[t].id == r.id && tracks[t].classId == r.classId, "same IDs");
            Check(cv::norm(tracks[t].filter.Centre() - cv::Point2d(r.CentreX(), r.CentreY())) < 1e-3, "same box centre");
            Check(std::abs(tracks[t].filter.Meters() - r.Meters()) < 1e-6, "same distance");
            Check(tracks[t].missedBatches == r.missedBatches, "same coasting");
            compared++;
        }
    }
    std::printf("%d object-frames compared with 05c's tracking; up to %d objects at once, %d IDs started, %d dropped\n",
                compared, maxTracks, started, dropped);
    Check(started == 3 && dropped == 1, "3 IDs (A, B, C; never the weak false alarm), C dropped after it left");

    // Overlap no longer divides by zero.
    Check(spotter::Overlap(cv::Rect2d(), cv::Rect2d()) == 0, "two empty boxes overlap 0, not NaN");

    std::printf(failures == 0 ? "PASSED\n" : "%d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
