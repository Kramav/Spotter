#include "spotter/radar.hpp"

#include <cmath>
#include <opencv2/imgproc.hpp>

namespace spotter
{

RadarView::RadarView(double rangeMeters, int sizePixels)
    : rangeMeters(rangeMeters), pixelsPerMeter((sizePixels - 40) / rangeMeters), sizePixels(sizePixels), camera(sizePixels / 2, sizePixels - 20)
{
}

cv::Point RadarView::ToPixel(cv::Point2d floor) const
{
    return { (int)std::lround(camera.x + floor.x * pixelsPerMeter), (int)std::lround(camera.y - floor.y * pixelsPerMeter) };
}

cv::Mat& RadarView::Begin(const PinholeCamera& colourCamera)
{
    image.create(sizePixels, sizePixels, CV_8UC3);
    image.setTo(cv::Scalar(25, 25, 25));
    const cv::Scalar grey(80, 80, 80);
    for (int meters = 1; meters <= (int)rangeMeters; meters++)
    {
        int radius = (int)(meters * pixelsPerMeter);
        cv::ellipse(image, camera, cv::Size(radius, radius), 0, 180, 360, grey, 1);
        cv::putText(image, std::to_string(meters) + " m", camera + cv::Point(4, -radius - 4), cv::FONT_HERSHEY_SIMPLEX, 0.4, grey, 1);
    }
    // The picture's edge is width/2 pixels from its centre, at a focal length of fx pixels.
    double halfFov = std::atan((colourCamera.width / 2.0) / colourCamera.fx);
    for (int side : { -1, 1 })
        cv::line(image, camera, ToPixel({ side * rangeMeters * std::sin(halfFov), rangeMeters * std::cos(halfFov) }), cv::Scalar(0, 120, 160), 1);
    int half = (int)(0.5 * pixelsPerMeter);
    cv::ellipse(image, camera, cv::Size(half, half), 0, 180, 360, cv::Scalar(0, 0, 140), 1);
    cv::drawMarker(image, camera, cv::Scalar(0, 255, 255), cv::MARKER_TRIANGLE_UP, 14, 2);
    return image;
}

void RadarView::DrawLine(cv::Point2d a, cv::Point2d b, cv::Scalar colour, int thickness, const std::string& label)
{
    cv::Point pa = ToPixel(a), pb = ToPixel(b);
    cv::line(image, pa, pb, colour, thickness);
    if (!label.empty()) cv::putText(image, label, (pa + pb) / 2 + cv::Point(6, -6), cv::FONT_HERSHEY_SIMPLEX, 0.45, colour, 1);
}

void RadarView::DrawTrail(const std::deque<cv::Point2d>& trail, cv::Scalar colour, bool hollow, const std::string& label)
{
    if (trail.empty()) return;
    for (size_t i = 1; i < trail.size(); i++) cv::line(image, ToPixel(trail[i - 1]), ToPixel(trail[i]), colour, 1);
    cv::Point now = ToPixel(trail.back());
    cv::circle(image, now, 7, colour, hollow ? 1 : cv::FILLED);
    cv::putText(image, label, now + cv::Point(10, 4), cv::FONT_HERSHEY_SIMPLEX, 0.45, colour, 1);
}

void TrackTrails::Update(const Tracker& tracker, const PinholeCamera& colourCamera)
{
    std::map<int, std::deque<cv::Point2d>> kept;   // only the IDs still tracked
    for (const Track& track : tracker.Tracks())
    {
        std::deque<cv::Point2d>& trail = kept[track.id] = std::move(trails[track.id]);
        if (track.filter.Meters() <= 0) continue;
        // ponytail: the colour camera is the origin (a few cm beside the depth camera); small next to box jitter.
        cv::Point3f p = colourCamera.Unproject(cv::Point2f(track.filter.Centre()), (float)track.filter.Meters());
        trail.push_back({ p.x, p.z });
        if ((int)trail.size() > length) trail.pop_front();
    }
    trails.swap(kept);
}

const std::deque<cv::Point2d>& TrackTrails::Of(int id) const
{
    static const std::deque<cv::Point2d> none;
    auto found = trails.find(id);
    return found == trails.end() ? none : found->second;
}

void TrackTrails::Clear()
{
    for (auto& entry : trails) entry.second.clear();
}

}
