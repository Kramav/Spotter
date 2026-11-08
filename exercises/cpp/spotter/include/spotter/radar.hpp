// spotter/radar.hpp: example 06's bird's-eye map. The camera sits at the bottom centre looking up
// the screen; X (right) and Z (ahead) are metres on the floor.
#pragma once

#include "spotter/frame.hpp"
#include "spotter/track.hpp"

#include <deque>
#include <map>
#include <string>

namespace spotter
{

class RadarView
{
public:
    explicit RadarView(double rangeMeters = 6.0, int sizePixels = 700);

    // A fresh map: a ring every metre, the colour camera's field of view, the 0.5 m minimum range.
    cv::Mat& Begin(const PinholeCamera& colourCamera);
    cv::Point ToPixel(cv::Point2d floor) const;
    void DrawLine(cv::Point2d a, cv::Point2d b, cv::Scalar colour, int thickness, const std::string& label = "");
    // A trail (oldest first) and a dot at its end: filled, or hollow for "predicted only".
    void DrawTrail(const std::deque<cv::Point2d>& trail, cv::Scalar colour, bool hollow, const std::string& label);
    cv::Mat& Image() { return image; }

private:
    double rangeMeters, pixelsPerMeter;
    int sizePixels;
    cv::Point camera;
    cv::Mat image;
};

// Where each tracked object has been on the floor: its filtered box centre and distance, through
// the colour camera's pinhole model (06). Forgets the IDs the tracker forgot.
class TrackTrails
{
public:
    explicit TrackTrails(int length = 150) : length(length) {}
    void Update(const Tracker& tracker, const PinholeCamera& colourCamera);
    const std::deque<cv::Point2d>& Of(int id) const;   // empty until the object has a distance
    void Clear();

private:
    int length;
    std::map<int, std::deque<cv::Point2d>> trails;
};

}
