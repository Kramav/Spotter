#include "spotter/view.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace spotter
{

namespace
{
double NowSeconds() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}

Every::Every(int milliseconds) : periodSeconds(milliseconds / 1000.0), nextSeconds(NowSeconds() + milliseconds / 1000.0) {}

bool Every::Due()
{
    double now = NowSeconds();
    if (now < nextSeconds) return false;
    nextSeconds = now + periodSeconds;
    return true;
}

double FpsMeter::Tick()
{
    double now = NowSeconds();
    if (lastSeconds >= 0 && now > lastSeconds)
    {
        double newest = 1.0 / (now - lastSeconds);
        fps = (fps == 0) ? newest : smoothing * fps + (1 - smoothing) * newest;
    }
    lastSeconds = now;
    return fps;
}

void FpsMeter::Draw(cv::Mat& image, const std::string& extra) const
{
    double size = std::clamp(image.cols / 1000.0, 0.45, 1.0);
    cv::putText(image, cv::format("%.1f fps", fps) + extra, cv::Point(10, image.rows - 10), cv::FONT_HERSHEY_SIMPLEX,
                size, cv::Scalar(0, 255, 255), size > 0.7 ? 2 : 1);
}

OrbitView::OrbitView(const Options& options, cv::Point3f pivot, float distance)
    : options(options), pivot(pivot), distance(distance), startDistance(distance)
{
}

void OrbitView::Attach(const std::string& window) { cv::setMouseCallback(window, OnMouse, this); }

void OrbitView::OnMouse(int event, int x, int y, int flags, void* userdata)
{
    OrbitView& view = *(OrbitView*)userdata;
    if (event == cv::EVENT_LBUTTONDOWN) { view.dragging = true; view.lastX = x; view.lastY = y; }
    if (event == cv::EVENT_LBUTTONUP) view.dragging = false;
    if (event == cv::EVENT_MOUSEMOVE && view.dragging)
    {
        view.yawDegrees += (x - view.lastX) * 0.3f;
        view.pitchDegrees -= (y - view.lastY) * 0.3f;
        view.lastX = x;
        view.lastY = y;
    }
    if (event == cv::EVENT_MOUSEWHEEL) view.distance -= cv::getMouseWheelDelta(flags) / 120.0f * view.options.zoomStepMeters;
}

bool OrbitView::HandleKey(int key)
{
    switch (key)
    {
    case 'a': yawDegrees -= options.turnStepDegrees; break;
    case 'd': yawDegrees += options.turnStepDegrees; break;
    case 'w': pitchDegrees += options.turnStepDegrees; break;
    case 's': pitchDegrees -= options.turnStepDegrees; break;
    case 'z': distance -= options.zoomStepMeters; break;
    case 'x': distance += options.zoomStepMeters; break;
    case 'r': Reset(); break;
    default: return false;
    }
    return true;
}

void OrbitView::Reset()
{
    yawDegrees = pitchDegrees = 0;
    distance = startDistance;
}

void OrbitView::Update()
{
    pitchDegrees = std::clamp(pitchDegrees, -89.0f, 89.0f);   // past straight down the view flips over
    distance = std::max(distance, 0.1f);
    const float toRadians = (float)CV_PI / 180;
    cosYaw = std::cos(yawDegrees * toRadians);
    sinYaw = std::sin(yawDegrees * toRadians);
    cosPitch = std::cos(pitchDegrees * toRadians);
    sinPitch = std::sin(pitchDegrees * toRadians);
}

bool OrbitView::Project(const cv::Point3f& point, cv::Point& pixel, float& depth) const
{
    // 1. Make the pivot the centre of the world. 2. Turn left/right around the vertical axis,
    // 3. then tilt around the sideways axis. 4. Step back from the pivot. 5. The pinhole camera.
    float x = point.x - pivot.x, y = point.y - pivot.y, z = point.z - pivot.z;
    float x1 = cosYaw * x + sinYaw * z;
    float z1 = -sinYaw * x + cosYaw * z;
    float y2 = cosPitch * y - sinPitch * z1;
    float z2 = sinPitch * y + cosPitch * z1;
    depth = z2 + distance;
    if (!(depth >= 0.05f)) return false;   // behind the viewer, or NaN
    pixel = cv::Point((int)(options.focalPixels * x1 / depth + options.width / 2),
                      (int)(options.focalPixels * y2 / depth + options.height / 2));
    return pixel.x >= 0 && pixel.y >= 0 && pixel.x < options.width && pixel.y < options.height;
}

cv::Mat& PointRenderer::Begin(OrbitView& orbitView, cv::Scalar background)
{
    view = &orbitView;
    view->Update();
    image.create(view->options.height, view->options.width, CV_8UC3);
    image.setTo(background);
    nearest.create(image.size(), CV_32F);
    nearest.setTo(1e9);
    return image;
}

void PointRenderer::Splat(const cv::Point3f& point, const cv::Vec3b& colour, int sizePixels)
{
    cv::Point pixel;
    float depth;
    if (!view->Project(point, pixel, depth)) return;
    for (int dy = 0; dy < sizePixels; dy++)
        for (int dx = 0; dx < sizePixels; dx++)
        {
            int u = pixel.x + dx, v = pixel.y + dy;
            if (u >= image.cols || v >= image.rows || depth >= nearest.at<float>(v, u)) continue;
            nearest.at<float>(v, u) = depth;
            image.at<cv::Vec3b>(v, u) = colour;
        }
}

void PointRenderer::Draw(const std::vector<cv::Point3f>& points, const std::vector<cv::Vec3b>& colours, int sizePixels)
{
    for (size_t i = 0; i < points.size() && i < colours.size(); i++) Splat(points[i], colours[i], sizePixels);
}

void PointRenderer::Draw(const std::vector<cv::Point3f>& points, cv::Vec3b colour, int sizePixels)
{
    for (const cv::Point3f& point : points) Splat(point, colour, sizePixels);
}

}
