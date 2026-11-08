// spotter/register.hpp: lining two views of the room up (examples 11, 11b, 13e). Points are in
// the camera's coordinates: metres, X right, Y down, Z ahead.
#pragma once

#include "spotter/geometry.hpp"

#include <opencv2/core.hpp>
#include <vector>

namespace spotter
{

// A depth cloud thinned for ICP: every step-th pixel that sits on a clear surface, and which way
// that surface faces (towards the camera). The normal comes from the neighbours `reach` pixels
// left/right and up/down; pixels at an edge (their depth jumps by more than maxJumpShare of the
// distance) are left out. One rule for every scan, live or loaded (the review's bug 3).
struct SurfacePoints
{
    std::vector<cv::Point3f> points, normals;
};
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

// Point-to-plane ICP (example 13e): move `from` by the pose, pair each point with the nearest `to`
// point, measure the gap along that point's normal, solve for the small turn + shift that closes
// the gaps, repeat. A direction the shape barely pins (sliding along a bare wall) keeps the
// starting pose's value instead of drifting on noise.
// Changes from 13e: a NaN normal can't poison the solve (bug 2), and the turn is measured in
// metres at the pairs' typical radius, so a small or near patch isn't called weak (bug 13).
IcpResult Icp(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to, const std::vector<cv::Point3f>& toNormals,
              const Pose& start, const IcpOptions& options = IcpOptions());

}
