// test_room.hpp: what the room checks share (check_room, check_room_twin, check_fill, check_room_shell,
// check_scan_session): Check(), and the made-up room they photograph. One copy, so a fix to the raycaster
// reaches every check (it was copied into each before the 2026-11-14 audit, D54).
//
// The room, in the first camera's coordinates (Y down, as the depth camera): a box from x -1.5 to 2.0, y -1.3
// (ceiling) to 1.2 (floor), z -1.0 to 3.0, with a couch 0.8 x 0.6 x 0.5 m on the floor 2 m ahead. Every
// surface is painted in 10 cm blocks of random colour, so ORB finds features everywhere. Closed, square to the
// axes, no holes: a check that needs holes, plain walls or slanted walls needs a scene of its own.
#pragma once

#include "spotter/frame.hpp"
#include "spotter/room_scan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

inline int failures = 0;
inline void Check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAILED: %s\n", what); failures++; }
}

const float Low[3] = { -1.5f, -1.3f, -1.0f }, High[3] = { 2.0f, 1.2f, 3.0f };            // the room (Y down)
const float CouchLow[3] = { 0.2f, 0.6f, 2.0f }, CouchHigh[3] = { 1.0f, 1.2f, 2.5f };     // 0.6 m tall, on the floor
const spotter::PinholeCamera Camera{ 365.5f, 365.5f, 256.0f, 212.0f, 512, 424 };

//float Raycast(const cv::Point3f& from, const cv::Point3f& ray, cv::Vec3b& colour, float frontMeters){
//
//@brief - the nearest surface along a ray: its distance (in ray lengths), and the colour of the 10 cm block it
//         hits. frontMeters > 0 takes the block from that far in front of the surface: at the hit point itself,
//         y = 1.2 falls in block 11 or 12 by rounding, so two rays to one spot can disagree (check_room_shell
//         compares two rays, and uses 1 mm). O(1).
inline float Raycast(const cv::Point3f& from, const cv::Point3f& ray, cv::Vec3b& colour, float frontMeters = 0)
{
    const float o[3] = { from.x, from.y, from.z }, d[3] = { ray.x, ray.y, ray.z };
    float best = 1e9f;
    int face = 0;
    for (int a = 0; a < 3; a++)
        if (d[a] != 0)
        {
            float t = ((d[a] > 0 ? High[a] : Low[a]) - o[a]) / d[a];
            if (t < best) { best = t; face = 2 * a + (d[a] > 0); }
        }
    float enter = 0, leave = 1e9f;
    int enterFace = 0;
    for (int a = 0; a < 3; a++)
    {
        float t1 = (CouchLow[a] - o[a]) / d[a], t2 = (CouchHigh[a] - o[a]) / d[a];
        if (std::min(t1, t2) > enter) { enter = std::min(t1, t2); enterFace = 6 + a; }
        leave = std::min(leave, std::max(t1, t2));
    }
    if (enter < leave && enter > 0 && enter < best) { best = enter; face = enterFace; }
    cv::Point3f p = frontMeters > 0 ? from + ray * (best - frontMeters / (float)cv::norm(ray)) : from + ray * best;
    int i = (int)std::floor(p.x / 0.1f), j = (int)std::floor(p.y / 0.1f), k = (int)std::floor(p.z / 0.1f);
    unsigned h = (unsigned)(face * 73856093) ^ (unsigned)(i * 19349663) ^ (unsigned)(j * 83492791) ^ (unsigned)(k * 2654435761u);
    h ^= h >> 13; h *= 0x5bd1e995; h ^= h >> 15;
    colour = cv::Vec3b((uchar)(h & 255), (uchar)((h >> 8) & 255), (uchar)((h >> 16) & 255));
    return best;
}

//spotter::Scan Photograph(const spotter::Pose& pose, int seed, const spotter::ScanOptions& options, float frontMeters){
//
//@brief - what a depth camera at `pose` (its coordinates -> the room's) sees: MakeScan of its cloud (about 1 mm
//         noise at 2 m, 07's average) and colour picture; colourIndex all -1, pose not set. O(pixels).
inline spotter::Scan Photograph(const spotter::Pose& pose, int seed, const spotter::ScanOptions& options = spotter::ScanOptions(), float frontMeters = 0)
{
    cv::RNG noise(seed);
    std::vector<cv::Point3f> cloud(512 * 424);
    cv::Mat picture(424, 512, CV_8UC3);
    for (int row = 0; row < 424; row++)
        for (int column = 0; column < 512; column++)
        {
            cv::Point3f ray = Camera.Unproject(cv::Point2f((float)column, (float)row), 1.0f);
            cv::Vec3b colour;
            float t = Raycast(pose.shift, pose.turn * ray, colour, frontMeters);
            cloud[row * 512 + column] = ray * (t * (1 + (float)noise.gaussian(0.0005)));
            picture.at<cv::Vec3b>(row, column) = colour;
        }
    spotter::Scan scan = spotter::MakeScan(cloud, picture, options);
    scan.colourIndex.assign(512 * 424, -1);
    return scan;
}

//spotter::Pose Spot(float right, float down, float forward, float leftDegrees, float downDegrees){
//
//@brief - a camera spot: turned left about the camera's up, then tipped down, at (right, down, forward). O(1).
inline spotter::Pose Spot(float right, float down, float forward, float leftDegrees, float downDegrees)
{
    return { spotter::TurnAbout(cv::Vec3d(0, -leftDegrees * CV_PI / 180, 0)) * spotter::TurnAbout(cv::Vec3d(-downDegrees * CV_PI / 180, 0, 0)),
             cv::Point3f(right, down, forward) };
}
