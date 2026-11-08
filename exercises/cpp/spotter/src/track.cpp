#include "spotter/track.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace spotter
{

double Overlap(const cv::Rect2d& a, const cv::Rect2d& b)
{
    double intersection = (a & b).area();
    double together = a.area() + b.area() - intersection;
    return together > 0 ? intersection / together : 0;
}

cv::Scalar ColourForId(int id)
{
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar((id * 47) % 180, 255, 255)), bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(c[0], c[1], c[2]);
}

BoxFilter::BoxFilter(cv::Point2d centre, double meters, const Noise& noise) : kalman(6, 3, 0, CV_32F), noise(noise)
{
    double pixels2 = noise.measurementPixels * noise.measurementPixels, meters2 = noise.measurementMeters * noise.measurementMeters;
    kalman.measurementMatrix = cv::Mat::eye(3, 6, CV_32F);   // H: the measurement is the first 3 state numbers
    kalman.measurementNoiseCov = (cv::Mat_<float>(3, 3) << pixels2, 0, 0, 0, pixels2, 0, 0, 0, meters2);   // R
    kalman.statePost = (cv::Mat_<float>(6, 1) << centre.x, centre.y, std::max(meters, 0.0), 0, 0, 0);
    // P: position about one measurement's noise; speed unknown (~1000 px/s, ~2 m/s). No distance
    // yet: very unsure, so the first real one replaces the 0 instead of being averaged with it.
    kalman.errorCovPost = cv::Mat::diag((cv::Mat_<float>(6, 1) << pixels2, pixels2, meters > 0 ? meters2 : 100.0,
                                         1000 * 1000, 1000 * 1000, 2 * 2));
}

void BoxFilter::Predict(double dt)
{
    // F: position += speed * dt. Q: an acceleration a moves position ~a dt^2/2 and speed ~a dt.
    // ponytail: diagonal Q; the textbook one adds position-speed correlation terms.
    cv::setIdentity(kalman.transitionMatrix);
    for (int i = 0; i < 3; i++) kalman.transitionMatrix.at<float>(i, i + 3) = (float)dt;
    const double accelerations[3] = { noise.accelerationPixels, noise.accelerationPixels, noise.accelerationMeters };
    kalman.processNoiseCov = cv::Mat::zeros(6, 6, CV_32F);
    for (int i = 0; i < 3; i++)
    {
        kalman.processNoiseCov.at<float>(i, i) = (float)std::pow(accelerations[i] * dt * dt / 2, 2);
        kalman.processNoiseCov.at<float>(i + 3, i + 3) = (float)std::pow(accelerations[i] * dt, 2);
    }
    kalman.predict();   // after this, statePost is the prediction, so an unseen object keeps moving
}

void BoxFilter::Correct(const cv::Point2d* centre, double meters)
{
    // A missing measurement is fed in as the filter's own guess with a huge noise: it changes nothing.
    cv::Mat& r = kalman.measurementNoiseCov;
    cv::Point2d c = centre ? *centre : Centre();
    if (!centre) r.at<float>(0, 0) = r.at<float>(1, 1) = 1e6f;
    if (meters <= 0) { meters = Meters(); r.at<float>(2, 2) = 1e6f; }
    kalman.correct((cv::Mat_<float>(3, 1) << c.x, c.y, meters));
    r.at<float>(0, 0) = r.at<float>(1, 1) = (float)(noise.measurementPixels * noise.measurementPixels);
    r.at<float>(2, 2) = (float)(noise.measurementMeters * noise.measurementMeters);
}

void BoxFilter::Fade(double factor)
{
    for (int i = 3; i < 6; i++)
    {
        kalman.statePost.at<float>(i) *= (float)factor;
        kalman.statePre.at<float>(i) *= (float)factor;
    }
}

cv::Point2d BoxFilter::Centre() const { return { kalman.statePost.at<float>(0), kalman.statePost.at<float>(1) }; }
double BoxFilter::Meters() const { return kalman.statePost.at<float>(2); }
double BoxFilter::MetersPerSecond() const { return kalman.statePost.at<float>(5); }

cv::Rect2d Track::Box() const
{
    cv::Point2d c = filter.Centre();
    return cv::Rect2d(c.x - boxSize.width / 2, c.y - boxSize.height / 2, boxSize.width, boxSize.height);
}

namespace
{
cv::Point2d CentreOf(const cv::Rect2d& box) { return { box.x + box.width / 2, box.y + box.height / 2 }; }
}

void Tracker::Predict(double dt)
{
    for (Track& track : tracks)
    {
        track.filter.Predict(dt);
        track.secondsUnseen += dt;
        if (track.Coasting()) track.filter.Fade(std::exp(-dt / options.coastSlowdownSeconds));
    }
    correctedThisFrame.assign(tracks.size(), false);
}

void Tracker::Match(const std::vector<Detection>& detections, const BoxDepthSampler& depth)
{
    correctedThisFrame.resize(tracks.size(), false);
    std::vector<bool> detectionMatched(detections.size(), false);

    // Greedy: of all same-class pairs, take the one overlapping most, pair them, repeat. Pass 0
    // uses only strong detections, pass 1 the weak ones, so a weak box never steals a strong match.
    for (int pass = 0; pass < 2; pass++)
    {
        float minConfidence = pass == 0 ? options.startConfidence : options.keepConfidence;
        float maxConfidence = pass == 0 ? 2.0f : options.startConfidence;
        while (true)
        {
            double bestOverlap = options.minOverlap;
            int bestTrack = -1, bestDetection = -1;
            for (size_t t = 0; t < tracks.size(); t++)
                for (size_t d = 0; d < detections.size(); d++)
                {
                    const Detection& detection = detections[d];
                    if (detection.confidence < minConfidence || detection.confidence >= maxConfidence) continue;
                    if (correctedThisFrame[t] || detectionMatched[d] || tracks[t].classId != detection.classId) continue;
                    double overlap = Overlap(tracks[t].Box(), detection.box);
                    if (overlap >= bestOverlap) { bestOverlap = overlap; bestTrack = (int)t; bestDetection = (int)d; }
                }
            if (bestTrack < 0) break;
            correctedThisFrame[bestTrack] = detectionMatched[bestDetection] = true;

            Track& track = tracks[bestTrack];
            const Detection& detection = detections[bestDetection];
            cv::Point2d centre = CentreOf(detection.box);
            track.filter.Correct(&centre, depth.MedianMeters(detection.box).value_or(0));
            track.boxSize = detection.box.size();
            track.missedBatches = 0;
            track.secondsUnseen = 0;
            track.lastConfidence = detection.confidence;
        }
    }
    for (size_t t = 0; t < tracks.size(); t++)
        if (!correctedThisFrame[t]) tracks[t].missedBatches++;

    // Strong detections nobody matched are new objects.
    for (size_t d = 0; d < detections.size(); d++)
    {
        const Detection& detection = detections[d];
        if (detectionMatched[d] || detection.confidence < options.startConfidence) continue;
        Track track;
        track.id = nextId++;
        track.classId = detection.classId;
        track.boxSize = detection.box.size();
        track.lastConfidence = detection.confidence;
        track.filter = BoxFilter(CentreOf(detection.box), depth.MedianMeters(detection.box).value_or(0), options.noise);
        tracks.push_back(track);
        correctedThisFrame.push_back(true);
        events.push_back({ true, track.id, track.classId, 0 });
    }
}

void Tracker::Remeasure(const BoxDepthSampler& depth)
{
    correctedThisFrame.resize(tracks.size(), false);
    // Not the unseen ones: their predicted box may be on the wall by now.
    for (size_t t = 0; t < tracks.size(); t++)
        if (!correctedThisFrame[t] && !tracks[t].Coasting()) tracks[t].filter.Correct(nullptr, depth.MedianMeters(tracks[t].Box()).value_or(0));

    for (const Track& track : tracks)
        if (track.secondsUnseen > options.maxUnseenSeconds) events.push_back({ false, track.id, track.classId, track.secondsUnseen });
    double limit = options.maxUnseenSeconds;
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [limit](const Track& t) { return t.secondsUnseen > limit; }), tracks.end());
    correctedThisFrame.clear();
}

std::vector<TrackEvent> Tracker::TakeEvents()
{
    std::vector<TrackEvent> taken;
    taken.swap(events);
    return taken;
}

}
