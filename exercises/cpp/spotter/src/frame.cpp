#include "spotter/frame.hpp"

#include <cmath>
#include <limits>

namespace spotter
{

cv::Point3f PinholeCamera::Unproject(cv::Point2f pixel, float meters) const
{
    return { (pixel.x + 0.5f - cx) * meters / fx, (pixel.y + 0.5f - cy) * meters / fy, meters };
}

std::optional<cv::Point2f> PinholeCamera::Project(const cv::Point3f& point) const
{
    if (!(point.z > 0)) return std::nullopt;
    return cv::Point2f(fx * point.x / point.z + cx - 0.5f, fy * point.y / point.z + cy - 0.5f);
}

std::vector<cv::Point3f> PinholeCamera::Cloud(const cv::Mat& depthMm) const
{
    const float none = std::numeric_limits<float>::quiet_NaN();
    std::vector<cv::Point3f> cloud(depthMm.total(), cv::Point3f(none, none, none));
    for (int row = 0; row < depthMm.rows; row++)
    {
        const float* mm = depthMm.ptr<float>(row);
        for (int column = 0; column < depthMm.cols; column++)
            if (mm[column] > 0) cloud[row * depthMm.cols + column] = Unproject(cv::Point2f((float)column, (float)row), mm[column] / 1000);
    }
    return cloud;
}

void Unmirror(RgbdFrame& frame)
{
    for (cv::Mat* picture : { &frame.colour, &frame.depthMm, &frame.registered, &frame.bigdepth })
        if (!picture->empty()) cv::flip(*picture, *picture, 1);

    // Depth pixel (row, column) is now what (row, width - 1 - column) was, and the colour pixel
    // it points at moved the same way.
    const int depthWidth = frame.depthMm.cols, colourWidth = frame.colourCamera.width;
    std::vector<int> flipped(frame.colourIndex.size(), -1);
    for (size_t i = 0; i < frame.colourIndex.size(); i++)
    {
        int row = (int)i / depthWidth, column = (int)i % depthWidth;
        int index = frame.colourIndex[row * depthWidth + depthWidth - 1 - column];
        if (index >= 0) flipped[i] = (index / colourWidth) * colourWidth + colourWidth - 1 - index % colourWidth;
    }
    frame.colourIndex.swap(flipped);

    // A flipped column c' is the old 511 - c, so (c + 0.5 - cx) = -(c' + 0.5 - (width - cx)).
    for (PinholeCamera* camera : { &frame.depthCamera, &frame.colourCamera })
        camera->cx = camera->width - camera->cx;
}

}
