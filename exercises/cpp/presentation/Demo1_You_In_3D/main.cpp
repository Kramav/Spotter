// =====================================================================
//  Demo 1 · You in 3D                 (presentation; built from 08)
//
//  Whoever stands in front of the Kinect sees themselves as a live 3D
//  point cloud, slowly swinging round so they see themselves from the
//  side. The surprise: from the side you're a thin shell. The camera only
//  sees your FRONT; your back is missing.
//
//  Same maths as 08 (a 3D point per depth pixel, our own little viewer),
//  plus: fullscreen, the view swings by itself, and C switches between
//  true colour and colour by distance (works in a dim hall too).
//  Nothing is saved: 08's P key is gone (privacy, visitors are in view).
//
//  Keys:   C  true colour / colour by distance    O  swing on/off
//          mouse drag / W A S D  turn   wheel / Z X  zoom   R reset
//          F  fullscreen on/off    Q or Esc  quit
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/registration.h>
#include <opencv2/opencv.hpp>

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// Where visitors stand, straight ahead of the Kinect. The view swings round this point, so a
// visitor standing here turns in place. Mark the spot on the floor.
const float PivotMeters = 2.0f;

// The swing: this many degrees either side, one full left-right-left every OrbitSeconds.
const float OrbitSwingDegrees = 70;
const float OrbitSeconds = 16;

// Everything farther than this is dropped. 4.5 m keeps the back wall as a backdrop; 3 m shows
// only the visitors (the shell effect is clearer with nothing behind them).
const float MaxDepthMeters = 4.5f;

// The viewer: 1280 x 720 scaled to the screen. 930 px across 1280 is about 69°, the depth
// camera's own field of view (08 used 700 across 960).
const int ViewWidth = 1280, ViewHeight = 720;
const float ViewFocalPixels = 930;
const int PointSizePixels = 2;
const float TurnStepDegrees = 5;
const float ZoomStepMeters = 0.25f;

const bool StartFullscreen = true;

// ---------------------------------------------------------------------
//  The viewer (as in 08)
// ---------------------------------------------------------------------

float yawDegrees = 0, pitchDegrees = 0, swingDegrees = 0;
float viewDistance = PivotMeters;
bool dragging = false;
int lastMouseX = 0, lastMouseY = 0;

void OnMouse(int event, int x, int y, int flags, void*)
{
    if (event == cv::EVENT_LBUTTONDOWN) { dragging = true; lastMouseX = x; lastMouseY = y; }
    if (event == cv::EVENT_LBUTTONUP) dragging = false;
    if (event == cv::EVENT_MOUSEMOVE && dragging)
    {
        yawDegrees += (x - lastMouseX) * 0.3f;
        pitchDegrees -= (y - lastMouseY) * 0.3f;
        lastMouseX = x;
        lastMouseY = y;
    }
    if (event == cv::EVENT_MOUSEWHEEL)
        viewDistance -= cv::getMouseWheelDelta(flags) / 120.0f * ZoomStepMeters;
}

bool Project(const cv::Point3f& point, cv::Point& pixel, float& depth)
{
    const float yaw = (yawDegrees + swingDegrees) * (float)CV_PI / 180, pitch = pitchDegrees * (float)CV_PI / 180;
    float x = point.x, y = point.y, z = point.z - PivotMeters;
    float x1 = std::cos(yaw) * x + std::sin(yaw) * z;
    float z1 = -std::sin(yaw) * x + std::cos(yaw) * z;
    float y2 = std::cos(pitch) * y - std::sin(pitch) * z1;
    float z2 = std::sin(pitch) * y + std::cos(pitch) * z1;
    depth = z2 + viewDistance;
    if (depth < 0.05f) return false;
    pixel = cv::Point((int)(ViewFocalPixels * x1 / depth + ViewWidth / 2),
                      (int)(ViewFocalPixels * y2 / depth + ViewHeight / 2));
    return pixel.x >= 0 && pixel.y >= 0 && pixel.x < ViewWidth && pixel.y < ViewHeight;
}

cv::Mat Render(const std::vector<cv::Point3f>& points, const std::vector<cv::Vec3b>& colours)
{
    cv::Mat image(ViewHeight, ViewWidth, CV_8UC3, cv::Scalar(20, 20, 20));
    cv::Mat nearest(ViewHeight, ViewWidth, CV_32F, cv::Scalar(1e9));
    for (size_t i = 0; i < points.size(); i++)
    {
        cv::Point pixel;
        float depth;
        if (!Project(points[i], pixel, depth)) continue;
        for (int dy = 0; dy < PointSizePixels; dy++)
        {
            for (int dx = 0; dx < PointSizePixels; dx++)
            {
                int u = pixel.x + dx, v = pixel.y + dy;
                if (u >= ViewWidth || v >= ViewHeight || depth >= nearest.at<float>(v, u)) continue;
                nearest.at<float>(v, u) = depth;
                image.at<cv::Vec3b>(v, u) = colours[i];
            }
        }
    }
    return image;
}

void ToggleFullscreen(const std::string& window)
{
    bool full = cv::getWindowProperty(window, cv::WND_PROP_FULLSCREEN) == cv::WINDOW_FULLSCREEN;
    cv::setWindowProperty(window, cv::WND_PROP_FULLSCREEN, full ? cv::WINDOW_NORMAL : cv::WINDOW_FULLSCREEN);
}

int main()
{
    // -----------------------------------------------------------------
    //  Setup: depth + colour + registration (as in 08)
    // -----------------------------------------------------------------

    std::setvbuf(stdout, nullptr, _IONBF, 0);
    libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Warning));
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device* device = freenect2.enumerateDevices() > 0 ? freenect2.openDefaultDevice() : nullptr;
    if (device == nullptr)
    {
        std::printf("No Kinect v2 found, or it is in use. See the main README's troubleshooting table.\n");
        return 1;
    }
    libfreenect2::Freenect2Device::Config config;
    config.MaxDepth = MaxDepthMeters;
    device->setConfiguration(config);
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Depth);
    device->setColorFrameListener(&listener);
    device->setIrAndDepthFrameListener(&listener);
    device->start();

    libfreenect2::Registration registration(device->getIrCameraParams(), device->getColorCameraParams());
    libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4);

    // Colour by distance (new): near = red, far = blue, from OpenCV's "turbo" colour map.
    cv::Mat steps(1, 256, CV_8U), distanceColours;
    for (int i = 0; i < 256; i++) steps.at<uchar>(i) = (uchar)(255 - i);
    cv::applyColorMap(steps, distanceColours, cv::COLORMAP_TURBO);

    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colours;
    bool byDistance = false, orbit = true;
    auto start = std::chrono::steady_clock::now();

    const std::string window = "Demo 1 - you in 3D";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, ViewWidth, ViewHeight);
    if (StartFullscreen) ToggleFullscreen(window);
    cv::setMouseCallback(window, OnMouse);

    // -----------------------------------------------------------------
    //  Main operation
    // -----------------------------------------------------------------

    libfreenect2::FrameMap frames;
    while (true)
    {
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == 'a') yawDegrees -= TurnStepDegrees;
        if (key == 'd') yawDegrees += TurnStepDegrees;
        if (key == 'w') pitchDegrees += TurnStepDegrees;
        if (key == 's') pitchDegrees -= TurnStepDegrees;
        if (key == 'z') viewDistance -= ZoomStepMeters;
        if (key == 'x') viewDistance += ZoomStepMeters;
        if (key == 'r') { yawDegrees = 0; pitchDegrees = 0; viewDistance = PivotMeters; }
        if (key == 'c') byDistance = !byDistance;
        if (key == 'o') orbit = !orbit;
        if (key == 'f') ToggleFullscreen(window);
        pitchDegrees = std::clamp(pitchDegrees, -89.0f, 89.0f);
        viewDistance = std::max(viewDistance, 0.1f);

        // The swing (new): a slow sine, so it pauses at each side, where the shell shows best.
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        swingDegrees = orbit ? OrbitSwingDegrees * (float)std::sin(2 * CV_PI * seconds / OrbitSeconds) : 0;

        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }

        // --- Every depth pixel -> a coloured 3D point (08) ----------------------
        registration.apply(frames[libfreenect2::Frame::Color], frames[libfreenect2::Frame::Depth], &undistorted, &registered);
        points.clear();
        colours.clear();
        for (int row = 0; row < 424; row++)
        {
            for (int column = 0; column < 512; column++)
            {
                float x, y, z;
                registration.getPointXYZ(&undistorted, row, column, x, y, z);
                if (std::isnan(z)) continue;
                points.emplace_back(x, y, z);
                if (byDistance)
                    colours.push_back(distanceColours.at<cv::Vec3b>(std::clamp((int)(255 * (z - 0.5f) / (MaxDepthMeters - 0.5f)), 0, 255)));
                else
                {
                    const unsigned char* bgrx = registered.data + 4 * (row * 512 + column);
                    colours.emplace_back(bgrx[0], bgrx[1], bgrx[2]);
                }
            }
        }
        listener.release(frames);

        // -----------------------------------------------------------------
        //  Output
        // -----------------------------------------------------------------

        cv::Mat image = Render(points, colours);
        cv::Point kinectPixel;
        float kinectDepth;
        if (Project(cv::Point3f(0, 0, 0), kinectPixel, kinectDepth))
        {
            cv::circle(image, kinectPixel, 8, cv::Scalar(0, 255, 255), 2);
            cv::putText(image, "camera", kinectPixel + cv::Point(12, 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
        }
        cv::putText(image, "This is you, in 3D. The camera only sees your front.", cv::Point(20, 45),
                    cv::FONT_HERSHEY_SIMPLEX, 1.1, cv::Scalar(255, 255, 255), 2);
        cv::putText(image, cv::format("%zu depth points, live   (live only: nothing is recorded)", points.size()),
                    cv::Point(20, ViewHeight - 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(160, 160, 160), 1);
        cv::imshow(window, image);
    }

    device->stop();
    device->close();
    cv::destroyAllWindows();
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas (and things to show visitors)
// ---------------------------------------------------------------------
//  1. Ask a visitor to hold out an arm towards the camera, then watch from the side: the arm is
//     a hollow trough, and there's an empty "shadow" on the wall behind it.
//  2. Set MaxDepthMeters = 3: only the visitors are left, floating in black.
//  3. Press C in a dim hall: colour by distance still works (depth brings its own infrared light).
//  4. Two visitors, one behind the other: the one behind has a hole the shape of the one in front.
//  5. Set OrbitSwingDegrees = 180 and OrbitSeconds = 30: the view goes right round behind them.
