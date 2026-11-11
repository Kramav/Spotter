// spotter/depth.hpp: helpers for the Kinect's depth picture (D34).
#pragma once

#include "spotter/frame.hpp"

#include <opencv2/core.hpp>
#include <algorithm>
#include <optional>
#include <vector>

namespace spotter
{

//cv::Mat DepthPicture(const cv::Mat& depthMm, float nearMm, float farMm){
//
//@param1 - depthMm: a depth picture, CV_32F millimetres, 0 = no reading
//@param2 - nearMm: this distance and nearer shows white (default 500)
//@param3 - farMm: this distance and farther shows black (default 6000)
//
//@brief - the depth picture as grey for showing (example 01's picture): near = white, far = black, and
//         red where there's no reading (too close, too far, black, shiny or glass). O(pixels).
cv::Mat DepthPicture(const cv::Mat& depthMm, float nearMm = 500, float farMm = 6000);

// Averages depth over many frames, per pixel (example 07): random noise falls with the square root
// of the frame count, and a pixel that only sometimes reads gets filled from the frames where it did.
class DepthAverager
{
public:
    //DepthAverager(int frames){
    //
    //@param1 - frames: how many frames one capture averages
    //
    //@brief - an averager that isn't running yet: Start() begins a capture. O(1).
    explicit DepthAverager(int frames) : frames(frames) {}

    //void Start(){
    //
    //@brief - forgets any capture in progress and begins a new one. O(1).
    void Start();

    //bool Running(){
    //
    //@brief - true while a capture has started and still needs frames. O(1).
    bool Running() const { return added >= 0 && added < frames; }

    //int Added(){
    //
    //@brief - how many frames the current capture has so far (0 before Start). O(1).
    int Added() const { return std::max(added, 0); }

    //int Frames(){
    //
    //@brief - how many frames a capture needs (the constructor's number). O(1).
    int Frames() const { return frames; }

    //bool Add(const cv::Mat& depthMm){
    //
    //@param1 - depthMm: the next depth picture, CV_32F millimetres, 0 = no reading
    //
    //@brief - while running: adds each pixel's reading to its sum and counts it. True when this was the
    //         last frame needed. O(pixels).
    bool Add(const cv::Mat& depthMm);

    //cv::Mat Mean(double minShare){
    //
    //@param1 - minShare: the share of the frames a pixel must have read in (default 0.5)
    //
    //@brief - the average depth in mm per pixel; 0 where a pixel read in fewer than minShare of the frames
    //         (too often a hole to trust). O(pixels).
    cv::Mat Mean(double minShare = 0.5) const;

private:
    int frames, added = -1;
    cv::Mat sumMm, readings;               // CV_64F, CV_32S
};

// Remembers the room's depth per pixel and keeps itself up to date (example 07c): the same
// distance blends in, clearly farther for leaveFrames = something left, clearly nearer for
// absorbFrames = something new stayed. Update() returns what is in front of it (255 = foreground).
class AdaptiveBackground
{
public:
    struct Options
    {
        float nearerMm = 100;     // foreground = at least this much nearer than remembered
        int absorbFrames = 300;   // something new that stays this long joins the room (~10 s)
        int leaveFrames = 10;     // farther this long = it left (one frame isn't enough: ceiling fans)
        float blendRate = 0.05f;  // per frame, how far the memory moves toward what it sees
    };

    //AdaptiveBackground(const Options& options){
    //
    //@param1 - options: the thresholds and blend rate above
    //
    //@brief - an empty memory: the first Update() becomes the background. O(1).
    explicit AdaptiveBackground(const Options& options) : options(options) {}

    //cv::Mat Update(const cv::Mat& liveMm, const cv::Mat& neverAbsorb){
    //
    //@param1 - liveMm: this frame's depth in mm, 0 = no reading
    //@param2 - neverAbsorb: optional CV_8U mask of the same size: pixels that stay foreground however long
    //          they stay (07d: inside a person box); empty = none
    //
    //@brief - per pixel: the first reading is taken as the room; the same distance blends into the memory by
    //         blendRate; nearer by nearerMm counts frames and joins the memory after absorbFrames; farther
    //         counts frames and replaces the memory after leaveFrames. Returns the foreground mask (255 = in
    //         front of the room). O(pixels).
    cv::Mat Update(const cv::Mat& liveMm, const cv::Mat& neverAbsorb = cv::Mat());

    //void Forget(){
    //
    //@brief - drops the memory, so the next frame becomes the background. O(1).
    void Forget();

    //const cv::Mat& DepthMm(){
    //
    //@brief - the remembered room depth, mm per pixel. O(1).
    const cv::Mat& DepthMm() const { return rememberedMm; }

private:
    Options options;
    cv::Mat rememberedMm, nearerFrames, fartherFrames;   // CV_32F, CV_32S, CV_32S
};

// The empty view's depth, recorded once after a countdown to step out of view (example 03's idea;
// Demos 4 and 5 use this copy). Foreground = clearly nearer than it, cleaned and only big blobs kept.
class StillBackground
{
public:
    //StillBackground(double countdownSeconds){
    //
    //@param1 - countdownSeconds: how long to wait before recording, to step out of view
    //
    //@brief - starts the countdown at once (as Restart). O(1).
    explicit StillBackground(double countdownSeconds);

    //void Restart(){
    //
    //@brief - forgets the recording, counts down again, then records; prints what to do. O(1).
    void Restart();

    //bool RecordWhenDue(const cv::Mat& liveMm){
    //
    //@param1 - liveMm: this frame's depth in mm
    //
    //@brief - once the countdown is over and nothing is recorded yet, keeps a copy of this frame as the
    //         background. True on the frame it records. O(pixels) on that frame, O(1) otherwise.
    bool RecordWhenDue(const cv::Mat& liveMm);

    //bool Recorded(){
    //
    //@brief - true once a background has been recorded. O(1).
    bool Recorded() const { return !depthMm.empty(); }

    //double SecondsLeft(){
    //
    //@brief - the seconds left on the countdown (negative once it's over). O(1).
    double SecondsLeft() const;

    //const cv::Mat& DepthMm(){
    //
    //@brief - the recorded background depth in mm (empty before it's recorded). O(1).
    const cv::Mat& DepthMm() const { return depthMm; }

    //cv::Mat Foreground(const cv::Mat& liveMm, float nearerMm, int cleanupPixels, int minPixels, std::vector<cv::Rect>& boxes){
    //
    //@param1 - liveMm: this frame's depth in mm
    //@param2 - nearerMm: foreground = at least this much nearer than the background
    //@param3 - cleanupPixels: the size of the round opening that removes specks and thin edges
    //@param4 - minPixels: blobs smaller than this are dropped
    //@param5 - boxes: filled with the boxes of the blobs kept
    //
    //@brief - 255 where liveMm reads at least nearerMm nearer, after a morphological opening, in blobs of
    //         minPixels or more (KeepBigBlobs). All zero before it is recorded.
    //         O(pixels x cleanupPixels^2) at worst for the opening, O(pixels) for the rest.
    cv::Mat Foreground(const cv::Mat& liveMm, float nearerMm, int cleanupPixels, int minPixels, std::vector<cv::Rect>& boxes) const;

private:
    double countdownSeconds, startSeconds = 0;
    cv::Mat depthMm;
};

//std::vector<cv::Rect> KeepBigBlobs(cv::Mat& mask, int minPixels){
//
//@param1 - mask: a CV_8U mask (255 = set); changed in place
//@param2 - minPixels: blobs (groups of touching 255 pixels) smaller than this are cleared
//
//@brief - labels the blobs (cv::connectedComponentsWithStats), clears the small ones and returns the
//         boxes of the ones left. O(pixels).
std::vector<cv::Rect> KeepBigBlobs(cv::Mat& mask, int minPixels);

// "How far away is the thing in this box?", for boxes drawn on the colour picture (YOLO's).
//
// The depth and colour cameras don't line up, so finding the depth pixels inside a colour box
// means looking up where each depth pixel lands in colour. Doing that per box scans all 217k
// depth pixels every time. Build() does the lookup once per frame and files the readings in
// cells of 32 colour columns on each colour row; a box then visits only the cells it covers.
// Every reading inside the box is used, none are skipped.
//
// Only for boxes. Room scans and point clouds still use every depth pixel directly.
class BoxDepthSampler
{
public:
    //void Build(const float* depthMm, const std::vector<int>& colourIndex, int colourWidth, int colourHeight){
    //
    //@param1 - depthMm: the undistorted depth picture (width x height floats, mm; 0, NaN or inf = no reading)
    //@param2 - colourIndex: registration.apply's last output: for each depth pixel, the colour pixel it lands
    //          on (row * colourWidth + column), or -1 for none. Same length as depthMm.
    //@param3 - colourWidth: the colour picture's width in pixels
    //@param4 - colourHeight: the colour picture's height in pixels
    //
    //@brief - once per frame, after registration.apply: files every depth reading in its cell (colour row,
    //         then 32-column block) with a counting sort, two passes. O(depth pixels + cells).
    void Build(const float* depthMm, const std::vector<int>& colourIndex, int colourWidth, int colourHeight);

    //void Build(const RgbdFrame& frame){
    //
    //@param1 - frame: a frame with depthMm, colourIndex and colour filled
    //
    //@brief - the same, taking the pictures and sizes from the frame. O(depth pixels + cells).
    void Build(const RgbdFrame& frame) { Build(frame.depthMm.ptr<float>(), frame.colourIndex, frame.colour.cols, frame.colour.rows); }

    //std::optional<double> MedianMeters(const cv::Rect2d& box, double shrink, int minSamples){
    //
    //@param1 - box: a box on the colour picture, pixels
    //@param2 - shrink: the share of the box used each way, round its middle (default 0.5 = the middle half):
    //          the whole box mostly measures the wall behind the object (example 02)
    //@param3 - minSamples: fewer readings than this gives nothing (default 20)
    //
    //@brief - the median distance in metres of the readings in the middle part of the box. Per box row it
    //         visits only the cells the box covers. Empty before Build() or with too few readings (black,
    //         shiny or too-close objects). O(k) for k readings in those cells (Median is O(k)).
    std::optional<double> MedianMeters(const cv::Rect2d& box, double shrink = 0.5, int minSamples = 20) const;

private:
    static const int CellColumns = 32;   // cell width; any size works, 32 keeps the box's end cells small

    struct Reading
    {
        int column;   // in the colour picture
        float mm;
    };
    int colourWidth = 0, colourHeight = 0;
    std::vector<Reading> readings;   // filed by cell: colour row, then 32-column block
    std::vector<int> cellStart;      // cell c's readings are [cellStart[c], cellStart[c + 1])
    std::vector<int> cellOf;         // Build()'s scratch: each depth pixel's cell, kept to save allocating
};

}
