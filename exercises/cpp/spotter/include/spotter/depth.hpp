// spotter/depth.hpp: helpers for the Kinect's depth picture (D34).
#pragma once

#include <opencv2/core.hpp>
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
class BoxDepthSampler
{
public:
    // Once per frame, after registration.apply.
    // depthMm: the undistorted depth picture (width x height floats, mm; 0, NaN or inf = no reading).
    // colourIndex: registration.apply's last output: for each depth pixel, the colour pixel it
    //   lands on (row * colourWidth + column), or -1 for none. Same length as depthMm.
    void Build(const float* depthMm, const std::vector<int>& colourIndex, int colourWidth, int colourHeight);

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
