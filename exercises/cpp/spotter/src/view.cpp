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
//double NowSeconds(){
//
//@brief - seconds on the steady clock, for Every and FpsMeter. O(1).
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
    if (event == cv::EVENT_LBUTTONDOWN) { view.dragging = true; view.lastX = x; view.lastY = y; view.pressedAt = cv::Point(x, y); }
    if (event == cv::EVENT_LBUTTONUP)
    {
        view.dragging = false;
        if (cv::norm(cv::Point(x, y) - view.pressedAt) <= 4) { view.clicked = true; view.clickedAt = cv::Point(x, y); }
    }
    if (event == cv::EVENT_MOUSEMOVE && view.dragging)
    {
        view.yawDegrees += (x - view.lastX) * 0.3f;
        view.pitchDegrees -= (y - view.lastY) * 0.3f;
        view.lastX = x;
        view.lastY = y;
    }
    if (event == cv::EVENT_MOUSEWHEEL) view.distance -= cv::getMouseWheelDelta(flags) / 120.0f * view.options.zoomStepMeters;
}

cv::Mat WithKeys(const cv::Mat& image, const std::string& keys)
{
    cv::Mat picture = image;
    // As imshow shows them: floats 0-1 -> 0-255, 16-bit / 256.
    if (picture.depth() == CV_32F || picture.depth() == CV_64F) picture.convertTo(picture, CV_8U, 255);
    else if (picture.depth() == CV_16U || picture.depth() == CV_16S) picture.convertTo(picture, CV_8U, 1.0 / 256);
    else if (picture.depth() != CV_8U) picture.convertTo(picture, CV_8U);
    if (picture.channels() == 1) cv::cvtColor(picture, picture, cv::COLOR_GRAY2BGR);
    else if (picture.channels() == 4) cv::cvtColor(picture, picture, cv::COLOR_BGRA2BGR);
    // The entries, split at runs of 3+ spaces, laid out in lines that fit the picture's width.
    std::vector<std::string> entries;
    for (size_t start = 0; start < keys.size(); )
    {
        size_t gap = keys.find("   ", start);
        std::string entry = keys.substr(start, gap == std::string::npos ? std::string::npos : gap - start);
        if (!entry.empty()) entries.push_back(entry);
        if (gap == std::string::npos) break;
        start = keys.find_first_not_of(' ', gap);
    }
    const double scale = 0.5;
    const int lineHeight = 20, margin = 8;
    std::vector<std::string> lines(1);
    for (const std::string& entry : entries)
    {
        std::string longer = lines.back().empty() ? entry : lines.back() + "    " + entry;
        if (!lines.back().empty() && cv::getTextSize(longer, cv::FONT_HERSHEY_SIMPLEX, scale, 1, nullptr).width > picture.cols - 2 * margin)
            lines.push_back(entry);
        else lines.back() = longer;
    }
    cv::Mat shown(picture.rows + (int)lines.size() * lineHeight + 6, picture.cols, CV_8UC3, cv::Scalar(40, 40, 40));
    picture.copyTo(shown(cv::Rect(0, 0, picture.cols, picture.rows)));
    for (size_t i = 0; i < lines.size(); i++)
        cv::putText(shown, lines[i], cv::Point(margin, picture.rows + (int)(i + 1) * lineHeight - 4), cv::FONT_HERSHEY_SIMPLEX, scale,
                    cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
    return shown;
}

void OpenExhibitWindow(const std::string& window, cv::Size size, bool fullscreen)
{
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, size.width, size.height);
    if (fullscreen) ToggleFullscreen(window);
}

void ToggleFullscreen(const std::string& window)
{
    bool full = cv::getWindowProperty(window, cv::WND_PROP_FULLSCREEN) == cv::WINDOW_FULLSCREEN;
    cv::setWindowProperty(window, cv::WND_PROP_FULLSCREEN, full ? cv::WINDOW_NORMAL : cv::WINDOW_FULLSCREEN);
}

bool OrbitView::TakeClick(cv::Point& where)
{
    if (!clicked) return false;
    clicked = false;
    where = clickedAt;
    return true;
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

void PointRenderer::Draw(const std::vector<cv::Point3f>& points, const std::vector<cv::Vec3b>& colours, int sizePixels, size_t every)
{
    for (size_t i = 0; i < points.size() && i < colours.size(); i += std::max<size_t>(every, 1)) Splat(points[i], colours[i], sizePixels);
}

void PointRenderer::Draw(const std::vector<cv::Point3f>& points, cv::Vec3b colour, int sizePixels, size_t every)
{
    for (size_t i = 0; i < points.size(); i += std::max<size_t>(every, 1)) Splat(points[i], colour, sizePixels);
}

void DrawCamera(cv::Mat& image, const OrbitView& view, const Pose& pose, const PinholeCamera& camera, const cv::Scalar& colour,
                int thickness, const std::string& label, float arrowMeters)
{
    cv::Point pixel, tip;
    float depth;
    if (!view.Project(pose.shift, pixel, depth)) return;
    cv::circle(image, pixel, 6, colour, thickness + 1);
    for (cv::Point2f corner : { cv::Point2f(0, 0), cv::Point2f((float)camera.width, 0), cv::Point2f(0, (float)camera.height),
                                cv::Point2f((float)camera.width, (float)camera.height) })
        if (view.Project(pose(camera.Unproject(corner - cv::Point2f(0.5f, 0.5f), arrowMeters)), tip, depth))
            cv::arrowedLine(image, pixel, tip, colour, thickness, cv::LINE_AA, 0, 0.15);
    cv::putText(image, label, pixel + cv::Point(8, -8), cv::FONT_HERSHEY_SIMPLEX, 0.5, colour, 1);
}

}
