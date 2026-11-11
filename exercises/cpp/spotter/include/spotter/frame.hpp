// spotter/frame.hpp: one moment from an RGB-D camera, and the pinhole maths that turns a
// depth pixel into a 3D point and back. No camera driver in here: a source (source_freenect2)
// fills an RgbdFrame, and everything else reads it.
#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <vector>

namespace spotter
{

// A perfect pinhole camera: focal lengths and principal point, in pixels.
//
// 3D coordinates are in metres, from the camera: X to the right, Y down, Z straight out.
// A pixel's centre is at +0.5 (pixel 0 covers 0..1), as in libfreenect2's getPointXYZ.
struct PinholeCamera
{
    float fx = 0, fy = 0, cx = 0, cy = 0;
    int width = 0, height = 0;

    //cv::Point3f Unproject(cv::Point2f pixel, float meters){
    //
    //@param1 - pixel: the pixel, x = column and y = row; fractions allowed
    //@param2 - meters: how far along Z (straight out of the camera) it was seen
    //
    //@brief - a pixel seen at a distance -> its 3D point, by the pinhole formula:
    //         X = (column + 0.5 - cx) * Z / fx,  Y = (row + 0.5 - cy) * Z / fy,  Z = meters. O(1).
    cv::Point3f Unproject(cv::Point2f pixel, float meters) const;

    //std::optional<cv::Point2f> Project(const cv::Point3f& point){
    //
    //@param1 - point: a 3D point in this camera's coordinates (metres)
    //
    //@brief - the other way: where the point lands in the picture as (column, row); the pixel holding
    //         it is (int)std::lround of each. Empty when the point is behind the camera (Z <= 0 or NaN).
    //         It isn't checked against the picture's edges: the caller does that. O(1).
    std::optional<cv::Point2f> Project(const cv::Point3f& point) const;

    //std::vector<cv::Point3f> Cloud(const cv::Mat& depthMm){
    //
    //@param1 - depthMm: a depth picture of this camera's size, CV_32F millimetres, 0 = no reading
    //
    //@brief - every pixel as a 3D point (Unproject), row by row. Pixels without a reading are NaN, so
    //         index i is still pixel (row i / width, column i % width). O(width x height).
    std::vector<cv::Point3f> Cloud(const cv::Mat& depthMm) const;
};

// One frame from an RGB-D camera, already the right way round (see Unmirror).
// Pictures a source didn't ask for are empty.
struct RgbdFrame
{
    cv::Mat colour;       // the colour picture, BGR, 1920 x 1080 for the Kinect
    cv::Mat depthMm;      // depth in mm, CV_32F, 512 x 424; lens distortion removed, 0 = no reading
    cv::Mat registered;   // colour per depth pixel, BGR, the depth picture's size; black = not seen
    cv::Mat bigdepth;     // depth per colour pixel (mm, CV_32F, the colour picture's size); 0 = none
    std::vector<int> colourIndex;   // per depth pixel: the colour pixel it lands on (row * colour width
                                    // + column), or -1 for none. Same order as depthMm.
    PinholeCamera depthCamera, colourCamera;
    double seconds = 0;   // when it arrived (steady clock), for time steps between frames
};

// The time between two frames, for filters that move things on by speed x time (example 05).
class FrameTimer
{
public:
    //double Step(const RgbdFrame& frame){
    //
    //@param1 - frame: the newest frame (its `seconds` is read)
    //
    //@brief - the seconds since the previous frame passed in, and remembers this one. The first call
    //         gives 1/30 (one Kinect frame). O(1).
    double Step(const RgbdFrame& frame)
    {
        double dt = last > 0 ? frame.seconds - last : 1.0 / 30;
        last = frame.seconds;
        return dt;
    }

private:
    double last = 0;
};

//void Unmirror(RgbdFrame& frame){
//
//@param1 - frame: a frame as the Kinect sent it (mirror images); changed in place
//
//@brief - the Kinect sends mirror images, like a selfie camera: your right hand shows on the left.
//         This flips every picture left-right, moves colourIndex to match, and moves both cameras'
//         principal points (cx -> width - cx), so 3D points come out the right way round too.
//         Sources call it; it's here so the checks can test it without a device. O(pixels).
void Unmirror(RgbdFrame& frame);

}
