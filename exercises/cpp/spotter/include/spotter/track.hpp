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

//double Overlap(const cv::Rect2d& a, const cv::Rect2d& b){
//
//@param1 - a: one box
//@param2 - b: the other box
//
//@brief - intersection over union: the boxes' shared area divided by the area they cover together.
//         1 = identical, 0 = not touching (also for empty boxes). O(1).
double Overlap(const cv::Rect2d& a, const cv::Rect2d& b);

//cv::Scalar ColourForId(int id){
//
//@param1 - id: a track's ID
//
//@brief - a fixed, distinct BGR colour per ID (the hue steps 47 of 180 per ID), so an ID change shows. O(1).
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

    //BoxFilter(cv::Point2d centre, double meters, const Noise& noise){
    //
    //@param1 - centre: where the box's centre was first seen, colour pixels
    //@param2 - meters: its first distance; <= 0 = not known yet
    //@param3 - noise: how much the measurements wobble and how suddenly speeds change
    //
    //@brief - a filter that starts where the object was seen, standing still. O(1).
    BoxFilter(cv::Point2d centre, double meters, const Noise& noise);

    //void Predict(double dt){
    //
    //@param1 - dt: seconds since the last prediction
    //
    //@brief - moves the state on: position += speed x dt, and grows the uncertainty by the acceleration
    //         noise (constant-velocity model). O(1) (fixed 6 x 6 matrices).
    void Predict(double dt);

    //void Correct(const cv::Point2d* centre, double meters){
    //
    //@param1 - centre: the measured box centre, or nullptr for no box this time
    //@param2 - meters: the measured distance, or <= 0 for none
    //
    //@brief - blends a measurement into the state, weighted by how sure each side is (the Kalman gain).
    //         A missing part keeps the filter's own guess. O(1).
    void Correct(const cv::Point2d* centre, double meters);

    //void Fade(double factor){
    //
    //@param1 - factor: what to multiply the speeds by, 0..1
    //
    //@brief - slows the object down (a coasting object shouldn't fly off for ever). O(1).
    void Fade(double factor);

    //cv::Point2d Centre(){
    //
    //@brief - the filtered box centre, colour pixels. O(1).
    cv::Point2d Centre() const;

    //double Meters(){
    //
    //@brief - the filtered distance in metres; 0 until a distance has been measured. O(1).
    double Meters() const;

    //double MetersPerSecond(){
    //
    //@brief - the filtered speed along the distance: + = moving away. O(1).
    double MetersPerSecond() const;

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

    //cv::Rect2d Box(){
    //
    //@brief - where the box should be now: the last detection's size round the filtered centre. O(1).
    cv::Rect2d Box() const;

    //bool Coasting(){
    //
    //@brief - true when the latest detector batch didn't include it (it moves on its prediction). O(1).
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

    //Tracker(const Options& options){
    //
    //@param1 - options: the thresholds, time limits and filter noise above
    //
    //@brief - a tracker with no objects yet; IDs start at 1. O(1).
    explicit Tracker(const Options& options) : options(options) {}

    //void Predict(double dt){
    //
    //@param1 - dt: seconds since the last frame
    //
    //@brief - every frame, first: moves every track's filter on, adds dt to its time unseen, and fades a
    //         coasting track's speed by exp(-dt / coastSlowdownSeconds). O(tracks).
    void Predict(double dt);

    //void Match(const std::vector<Detection>& detections, const BoxDepthSampler& depth){
    //
    //@param1 - detections: one detector batch
    //@param2 - depth: this frame's depth lookup, for each matched box's median distance
    //
    //@brief - greedy matching against the PREDICTED boxes, same class only: repeatedly take the pair that
    //         overlaps most (at least minOverlap), strong detections first, then weak ones, so a weak box
    //         never steals a strong match. Matched tracks are corrected; unmatched strong detections start
    //         new IDs. O(T x D x min(T, D)) for T tracks and D detections (each greedy round scans every
    //         pair), plus one MedianMeters per match.
    void Match(const std::vector<Detection>& detections, const BoxDepthSampler& depth);

    //void Remeasure(const BoxDepthSampler& depth){
    //
    //@param1 - depth: this frame's depth lookup
    //
    //@brief - every frame, after Match: tracks seen in the latest batch but not corrected this frame get
    //         their distance from this frame's depth at their predicted box (coasting ones don't: their box
    //         may be on the wall by now). Then tracks unseen over maxUnseenSeconds are dropped, with an
    //         event each. O(tracks), plus one MedianMeters per track remeasured.
    void Remeasure(const BoxDepthSampler& depth);

    //const std::vector<Track>& Tracks(){
    //
    //@brief - the objects being followed now. O(1).
    const std::vector<Track>& Tracks() const { return tracks; }

    //std::vector<TrackEvent> TakeEvents(){
    //
    //@brief - the IDs started and dropped since the last call (the list is emptied). O(events).
    std::vector<TrackEvent> TakeEvents();

    //void Clear(){
    //
    //@brief - forgets every object (IDs keep counting up). O(tracks).
    void Clear() { tracks.clear(); }

private:
    Options options;
    std::vector<Track> tracks;
    std::vector<bool> correctedThisFrame;
    std::vector<TrackEvent> events;
    int nextId = 1;
};

}
