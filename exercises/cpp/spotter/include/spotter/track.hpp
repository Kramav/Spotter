// spotter/track.hpp: following detected objects over time (examples 04-05c).
// Each object keeps an ID; a Kalman filter per object predicts where its box will be, so fast
// movers keep their ID and objects YOLO misses for a moment keep moving ("coasting").
#pragma once

#include "spotter/depth.hpp"
#include "spotter/detection.hpp"

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

namespace spotter
{

// Intersection over union: the boxes' shared area divided by the area they cover together.
// 1 = identical, 0 = not touching (also for empty boxes).
double Overlap(const cv::Rect2d& a, const cv::Rect2d& b);

// A fixed, distinct colour per ID (hue steps around the wheel), so an ID change shows.
cv::Scalar ColourForId(int id);

// A Kalman filter for one box (example 05). State: centre x, centre y (colour pixels), distance
// (metres) and their speeds. It measures the first three.
class BoxFilter
{
public:
    struct Noise
    {
        double measurementPixels = 15;   // how much YOLO's box centre wobbles
        double measurementMeters = 0.03; // how much the sampled depth wobbles
        double accelerationPixels = 500; // how suddenly things change speed: ~1 m/s^2 at 2 m in 1920 px
        double accelerationMeters = 1.0;
    };

    BoxFilter() = default;
    // Starts where it was seen, standing still. meters <= 0 = distance unknown yet.
    BoxFilter(cv::Point2d centre, double meters, const Noise& noise);

    void Predict(double dt);                        // position += speed * dt; less sure
    // Blend in a measurement. A missing part (no box, or meters <= 0) keeps the filter's own guess.
    void Correct(const cv::Point2d* centre, double meters);
    void Fade(double factor);                       // multiply the speeds (coasting slows down)

    cv::Point2d Centre() const;
    double Meters() const;                          // 0 until a distance has been measured
    double MetersPerSecond() const;                 // + = moving away

private:
    cv::KalmanFilter kalman;
    Noise noise;
};

struct Track
{
    int id = 0;
    int classId = -1;
    BoxFilter filter;
    cv::Size2d boxSize;          // the filter follows the centre; the size is the last detection's
    int missedBatches = 0;       // detector batches in a row that didn't include it
    double secondsUnseen = 0;    // time since a detection last matched it
    float lastConfidence = 0;

    cv::Rect2d Box() const;      // where it should be now
    bool Coasting() const { return missedBatches > 0; }
};

// Something a log might want to say: an ID started, or one was dropped.
struct TrackEvent
{
    bool started;                // false = dropped
    int id, classId;
    double secondsUnseen;
};

// Many objects (example 05c): predict every frame, fold in each detector batch with two
// confidence thresholds (05b), re-measure distances from fresh depth between batches.
class Tracker
{
public:
    struct Options
    {
        float startConfidence = 0.40f;    // only detections this sure start a new ID
        float keepConfidence = 0.15f;     // weaker ones (down to this) may only continue an ID
        double minOverlap = 0.3;          // with the PREDICTED box, same class
        double maxUnseenSeconds = 5.0;    // forget an object unseen this long
        double coastSlowdownSeconds = 0.5;// an unseen object's speed fades with this time constant
        BoxFilter::Noise noise;
    };

    explicit Tracker(const Options& options) : options(options) {}

    void Predict(double dt);   // every frame, first
    // A detector batch: match (strong detections first, then weak), correct, start new IDs.
    // Distances come from the box's median depth.
    void Match(const std::vector<Detection>& detections, const BoxDepthSampler& depth);
    // Every frame, after Match: objects seen in the latest batch get their distance from this
    // frame's depth at their predicted box. Then objects unseen too long are dropped.
    void Remeasure(const BoxDepthSampler& depth);

    const std::vector<Track>& Tracks() const { return tracks; }
    std::vector<TrackEvent> TakeEvents();   // since the last call
    void Clear() { tracks.clear(); }

private:
    Options options;
    std::vector<Track> tracks;
    std::vector<bool> correctedThisFrame;
    std::vector<TrackEvent> events;
    int nextId = 1;
};

}
