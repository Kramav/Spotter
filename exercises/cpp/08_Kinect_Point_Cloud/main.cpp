// =====================================================================
//  08 · Kinect Point Cloud
//
//  Every depth pixel is a point in 3D. Example 03 already did the maths
//  for one pixel:  X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fy.
//  Do it for all 217,088 pixels, give each the colour seen at that spot,
//  and you have a POINT CLOUD: the room as 3D dots.
//
//  This example draws it with its own tiny 3D viewer (no extra library):
//  turn each point around a pivot, then project it onto a virtual camera,
//  exactly the pinhole maths in reverse. The view starts where the Kinect
//  is, so it looks like the camera image. Drag it and the room turns out
//  to be a thin shell: the camera only sees the FRONT of everything.
//
//  Window: the point cloud. Yellow circle = where the Kinect is.
//  Keys:   mouse drag / W A S D  turn      mouse wheel / Z X  zoom
//          R  reset view    SPACE  freeze/unfreeze    P  save as .ply
//          Q or Esc  quit
//
//  Privacy: a .ply file is a 3D capture of the room. Saved only when you
//  press P, into the folder you ran from. *.ply is gitignored: never
//  commit one.
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
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

// The depth cut-off. 07b: readings to about 6 m are fine, past that almost nothing is right.
const float MaxDepthMeters = 6.0f;

// The viewer window, and its virtual camera's focal length. 700 px across 960 px is about 69°,
// close to the depth camera's own 70°, so the starting view matches the Kinect's.
const int ViewWidth = 960, ViewHeight = 720;
const float ViewFocalPixels = 700;

// The view turns around this point, straight ahead of the Kinect. Put it near the middle of the room.
const float PivotMeters = 2.5f;

const int PointSizePixels = 2;      // 1 = sharp but gappy up close, 3 = solid but blocky
const float TurnStepDegrees = 5;    // per W/A/S/D press
const float ZoomStepMeters = 0.25f; // per Z/X press or wheel notch

// ---------------------------------------------------------------------
//  The viewer's state: changed by the mouse and keys
// ---------------------------------------------------------------------

float yawDegrees = 0, pitchDegrees = 0;   // turn left/right, tilt up/down
float viewDistance = PivotMeters;         // virtual camera's distance from the pivot
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

// Kinect coordinates (metres; X right, Y down, Z forward) -> a pixel in the viewer.
// Returns false if the point is behind the virtual camera or off screen.
bool Project(const cv::Point3f& point, cv::Point& pixel, float& depth)
{
    const float yaw = yawDegrees * (float)CV_PI / 180, pitch = pitchDegrees * (float)CV_PI / 180;

    // 1. Make the pivot the centre of the world.
    float x = point.x, y = point.y, z = point.z - PivotMeters;
    // 2. Turn left/right (around the vertical axis)...
    float x1 = std::cos(yaw) * x + std::sin(yaw) * z;
    float z1 = -std::sin(yaw) * x + std::cos(yaw) * z;
    // 3. ...then tilt (around the sideways axis).
    float y2 = std::cos(pitch) * y - std::sin(pitch) * z1;
    float z2 = std::sin(pitch) * y + std::cos(pitch) * z1;
    // 4. Step the virtual camera back from the pivot.
    depth = z2 + viewDistance;
    if (depth < 0.05f) return false;
    // 5. The pinhole camera, as in example 03 but backwards: 3D -> pixel.
    pixel = cv::Point((int)(ViewFocalPixels * x1 / depth + ViewWidth / 2),
                      (int)(ViewFocalPixels * y2 / depth + ViewHeight / 2));
    return pixel.x >= 0 && pixel.y >= 0 && pixel.x < ViewWidth && pixel.y < ViewHeight;
}

// Draw every point; where two land on the same pixel, the nearer one wins (a "z-buffer").
cv::Mat Render(const std::vector<cv::Point3f>& points, const std::vector<cv::Vec3b>& colours)
{
    cv::Mat image(ViewHeight, ViewWidth, CV_8UC3, cv::Scalar(30, 30, 30));
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

// PLY: the simplest common 3D file. A text header, then one "x y z red green blue" line per point.
// Opens in MeshLab or CloudCompare (both free). Saved Y up and Z towards the viewer (the usual
// convention in 3D programs), which is the Kinect's frame turned 180° about X.
bool SavePly(const std::string& path, const std::vector<cv::Point3f>& points, const std::vector<cv::Vec3b>& colours)
{
    std::ofstream file(path);
    file << "ply\nformat ascii 1.0\nelement vertex " << points.size() << "\n"
         << "property float x\nproperty float y\nproperty float z\n"
         << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (size_t i = 0; i < points.size(); i++)
    {
        const cv::Vec3b& c = colours[i];   // OpenCV stores blue, green, red
        file << cv::format("%.4f %.4f %.4f %d %d %d\n", points[i].x, -points[i].y, -points[i].z, c[2], c[1], c[0]);
    }
    return (bool)file;
}

int main()
{
    // -----------------------------------------------------------------
    //  Setup: depth + colour + registration (as in example 02)
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

    // Registration maps each depth pixel to the colour pixel that saw the same spot (example 02).
    libfreenect2::Registration registration(device->getIrCameraParams(), device->getColorCameraParams());
    libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4);

    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colours;
    bool frozen = false;

    const std::string window = "08 point cloud  (drag/WASD turn, wheel/ZX zoom, R reset, SPACE freeze, P save, Q quit)";
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
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
        if (key == ' ') frozen = !frozen;
        pitchDegrees = std::clamp(pitchDegrees, -89.0f, 89.0f);   // past straight down, it flips over
        viewDistance = std::max(viewDistance, 0.1f);

        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }

        // --- Step 1: every depth pixel -> a coloured 3D point ---------------
        if (!frozen)
        {
            registration.apply(frames[libfreenect2::Frame::Color], frames[libfreenect2::Frame::Depth],
                               &undistorted, &registered);
            points.clear();
            colours.clear();
            for (int row = 0; row < 424; row++)
            {
                for (int column = 0; column < 512; column++)
                {
                    float x, y, z;
                    registration.getPointXYZ(&undistorted, row, column, x, y, z);   // X = (u - cx) * Z / fx ...
                    if (std::isnan(z)) continue;                                    // no depth reading
                    // Black where the colour camera can't see (it covers less height than depth).
                    const unsigned char* bgrx = registered.data + 4 * (row * 512 + column);
                    points.emplace_back(x, y, z);
                    colours.emplace_back(bgrx[0], bgrx[1], bgrx[2]);
                }
            }
        }
        listener.release(frames);

        // --- Step 2: save on request ------------------------------------
        if (key == 'p')
        {
            std::string path = (std::filesystem::current_path() / ("cloud_" + std::to_string(std::time(nullptr)) + ".ply")).string();
            if (SavePly(path, points, colours))
                std::printf("Saved %zu points to %s (a capture of the room: never commit it)\n", points.size(), path.c_str());
            else
                std::printf("Could not write %s\n", path.c_str());
        }

        // -----------------------------------------------------------------
        //  Output: the viewer
        // -----------------------------------------------------------------

        cv::Mat image = Render(points, colours);
        cv::Point kinectPixel;
        float kinectDepth;
        if (Project(cv::Point3f(0, 0, 0), kinectPixel, kinectDepth))
            cv::circle(image, kinectPixel, 6, cv::Scalar(0, 255, 255), 2);
        cv::putText(image, cv::format("%zu points%s   turn %.0f, tilt %.0f deg", points.size(),
                    frozen ? "   FROZEN" : "", yawDegrees, pitchDegrees),
                    cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
        cv::imshow(window, image);
    }

    device->stop();
    device->close();
    cv::destroyAllWindows();
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Turn the view 90° (side on). Flat walls should be straight lines. Are they, or do they
//     bow or fan out? That's lens distortion or depth bias, and it matters when merging views.
//  2. Look behind a chair from the side: an empty "shadow" where the camera couldn't see.
//     That's why stage C needs several views.
//  3. Look at object edges side on: a trail of points between the object and the wall behind
//     ("flying pixels"). They come from pixels that saw half of each.
//  4. Set MaxDepthMeters = 8 and look at the far end. 07b's noise, now visible in 3D.
//  5. Freeze, then save with P and open the file in MeshLab or CloudCompare. Delete it after.
//  6. Combine with 07: build the cloud from a 30-frame average instead of one frame. How much
//     thinner does a flat wall get when you look at it edge on?
