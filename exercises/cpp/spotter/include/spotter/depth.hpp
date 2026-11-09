// spotter/depth.hpp: helpers for the Kinect's depth picture (D34).
#pragma once

#include "spotter/frame.hpp"

#include <opencv2/core.hpp>
#include <algorithm>
#include <optional>
#include <vector>

namespace spotter
{

// "How far away is the thing in this box?", for boxes drawn on the colour picture (YOLO's).
//
// The depth and colour cameras don't line up, so finding the depth pixels inside a colour box
// means looking up where each depth pixel lands in colour. Doing that per box scans all 217k
// depth pixels every time. Build() does the lookup once per frame and files the readings in
// cells of 32 colour columns on each colour row; a box then visits only the cells it covers.
// Every reading inside the box is used, none are skipped.
//
// Only for boxes. Room scans and point clouds still use every depth pixel directly.
// The depth picture as grey: near (nearMm) = white, far (farMm and beyond) = black, and red
// where there's no reading (too close, too far, black, shiny or glass). Example 01's picture.
cv::Mat DepthPicture(const cv::Mat& depthMm, float nearMm = 500, float farMm = 6000);

// Averages depth over many frames, per pixel (example 07): random noise falls with the square root
// of the frame count, and a pixel that only sometimes reads gets filled from the frames where it did.
class DepthAverager
{
public:
    explicit DepthAverager(int frames) : frames(frames) {}
    void Start();                          // forget any capture in progress and begin a new one
    bool Running() const { return added >= 0 && added < frames; }
    int Added() const { return std::max(added, 0); }
    int Frames() const { return frames; }
    bool Add(const cv::Mat& depthMm);      // while running; true when this was the last frame needed
    // The average in mm; 0 where a pixel read in fewer than minShare of the frames (too often a hole).
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

    explicit AdaptiveBackground(const Options& options) : options(options) {}

    // liveMm: depth in mm, 0 = no reading. neverAbsorb (optional, CV_8U, same size): pixels
    // that must stay foreground however long they stay (07d: inside a person box).
    cv::Mat Update(const cv::Mat& liveMm, const cv::Mat& neverAbsorb = cv::Mat());
    void Forget();   // the next frame becomes the background
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
    explicit StillBackground(double countdownSeconds);
    void Restart();                                   // count down again, then record (prints what to do)
    bool RecordWhenDue(const cv::Mat& liveMm);        // true on the frame it records
    bool Recorded() const { return !depthMm.empty(); }
    double SecondsLeft() const;
    const cv::Mat& DepthMm() const { return depthMm; }
    // 255 where liveMm reads at least nearerMm nearer, after an opening of cleanupPixels, in
    // blobs of minPixels or more (their boxes in `boxes`). All zero before it is recorded.
    cv::Mat Foreground(const cv::Mat& liveMm, float nearerMm, int cleanupPixels, int minPixels, std::vector<cv::Rect>& boxes) const;

private:
    double countdownSeconds, startSeconds = 0;
    cv::Mat depthMm;
};

// Drops blobs (groups of touching 255 pixels) smaller than minPixels from the mask.
// Returns the boxes of the ones left.
std::vector<cv::Rect> KeepBigBlobs(cv::Mat& mask, int minPixels);

class BoxDepthSampler
{
public:
    // Once per frame, after registration.apply.
    // depthMm: the undistorted depth picture (width x height floats, mm; 0, NaN or inf = no reading).
    // colourIndex: registration.apply's last output: for each depth pixel, the colour pixel it
    //   lands on (row * colourWidth + column), or -1 for none. Same length as depthMm.
    void Build(const float* depthMm, const std::vector<int>& colourIndex, int colourWidth, int colourHeight);
    void Build(const RgbdFrame& frame) { Build(frame.depthMm.ptr<float>(), frame.colourIndex, frame.colour.cols, frame.colour.rows); }

    // The median distance in metres of the middle part of the box. shrink 0.5 = the middle half
    // each way: the whole box mostly measures the wall behind the object (example 02).
    // Empty when fewer than minSamples readings land there (black, shiny or too-close objects).
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
