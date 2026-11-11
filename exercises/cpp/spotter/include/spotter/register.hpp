// spotter/register.hpp: lining two views of the room up (examples 11, 11b, 13e). Points are in
// the camera's coordinates: metres, X right, Y down, Z ahead.
#pragma once

#include "spotter/geometry.hpp"

#include <opencv2/core.hpp>
#include <vector>

namespace spotter
{

// A depth cloud thinned for ICP: points on clear surfaces, and which way each surface faces.
struct SurfacePoints
{
    std::vector<cv::Point3f> points, normals;
};

//SurfacePoints SampleSurface(const std::vector<cv::Point3f>& cloud, int width, int step, int reach, float maxJumpShare){
//
//@param1 - cloud: a depth cloud, one point per pixel row by row (NaN = none)
//@param2 - width: points per row
//@param3 - step: use every step-th pixel each way
//@param4 - reach: the normal comes from the neighbours this many pixels left/right and up/down (default 2)
//@param5 - maxJumpShare: a pixel whose neighbours' depths jump by more than this share of its distance is at
//          an edge and left out (default 0.05)
//
//@brief - every step-th pixel that sits on a clear surface, with its normal (left-to-right cross
//         up-to-down, turned towards the camera). Pixels with a missing neighbour, at an edge, or whose
//         neighbours lie in a line (bug 2: no facing) are left out. One rule for every scan, live or
//         loaded (the review's bug 3). O(n / step^2).
SurfacePoints SampleSurface(const std::vector<cv::Point3f>& cloud, int width, int step, int reach = 2, float maxJumpShare = 0.05f);

struct IcpOptions
{
    int rounds = 50;                  // pair up and fit, at most this many times
    float startPairMeters = 0.05f;    // pairs farther apart than this are ignored; shrinks 30% a round...
    float finalPairMeters = 0.02f;    // ...down to this (about the sensor's noise plus a bit)
    float weakShare = 0.05f;          // a direction fewer than this share of the pairs pin is left alone
};

struct IcpResult
{
    Pose pose;
    float pairedShare = 0;     // of the `from` points, the share paired in the last round
    float rmsMeters = 0;       // their gap along the surface normal (root mean square)
    int weakDirections = 0;    // of the 6 (3 turns, 3 shifts), how many the shape couldn't pin
};

//IcpResult Icp(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to, const std::vector<cv::Point3f>& toNormals, const Pose& start, const IcpOptions& options){
//
//@param1 - from: the points to move (e.g. a new scan's SampleSurface points)
//@param2 - to: the points to move them onto (the room so far)
//@param3 - toNormals: the facing at each `to` point, same order
//@param4 - start: the first guess of the pose moving `from` onto `to`
//@param5 - options: the rounds, the pairing gates and the weak-direction share (defaults: IcpOptions())
//
//@brief - point-to-plane ICP (example 13e): move `from` by the pose, pair each point with the nearest `to`
//         point (a FLANN k-d tree), measure the gap along that point's normal, solve the linearised 6 x 6
//         least squares for the small turn + shift that closes the gaps, repeat until it settles. A
//         direction the shape barely pins (sliding along a bare wall) keeps the starting pose's value
//         instead of drifting on noise. Changes from 13e: a NaN normal can't poison the solve (bug 2), and
//         the turn is measured in metres at the pairs' typical radius, so a small or near patch isn't
//         called weak (bug 13). Fewer than 10 points either side returns `start`.
//         O(m log m) to build the tree, then O(rounds x n log m) for n from points and m to points.
IcpResult Icp(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to, const std::vector<cv::Point3f>& toNormals,
              const Pose& start, const IcpOptions& options = IcpOptions());

}
