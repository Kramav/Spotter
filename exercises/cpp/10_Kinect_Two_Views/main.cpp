// =====================================================================
//  10 · Kinect Two Views
//
//  One camera position sees only the front of things (08). To map a room
//  you capture it from several places and put the clouds together. That
//  needs each capture's POSE: where the camera stood and which way it
//  pointed, relative to the first.
//
//  Here the pose is measured by hand. Capture view A, move the Kinect by
//  a measured amount, and type that move into the settings. Each point of
//  view B is then turned and shifted into A's coordinates:
//      point in A = turn(point in B) + where B's camera stands
//  If the pose is right, the two clouds overlap. If it's a little off,
//  every wall shows double. Example 11 (ICP) will fix that automatically.
//
//  Window: orange = view A (captured), blue = view B (live until frozen).
//          Yellow circle = camera A, cyan circle = camera B.
//  Keys:   SPACE  capture A, then freeze B, then start over
//          mouse drag / W A S D  turn the view   mouse wheel / Z X  zoom   R  reset view
//          nudge pose B:  J L  right   T G  up   I K  forward   U O  turn left
//          Q or Esc  quit
//
//  Try: mark the Kinect's spot with tape, capture A, slide it 30 cm to the
//  right, keep it pointing the same way, and set MoveRightMeters = 0.30.
// =====================================================================

#include <algorithm>
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

// Where camera B stands relative to camera A, as you measured it with a tape.
// "Right", "up" and "forward" are camera A's directions. Turn = how far B is turned to the left.
const float MoveRightMeters = 0.30f;
const float MoveUpMeters = 0.0f;
const float MoveForwardMeters = 0.0f;
const float TurnLeftDegrees = 0.0f;

// The J/L, T/G, I/K and U/O keys nudge the pose by this much, to line the clouds up by hand.
const float NudgeMeters = 0.01f;
const float NudgeDegrees = 0.5f;

const float MaxDepthMeters = 6.0f;   // 07b: fine to about 6 m

// The viewer, as in 08.
const int ViewWidth = 960, ViewHeight = 720;
const float ViewFocalPixels = 700;
const float PivotMeters = 2.5f;
const int PointSizePixels = 2;
const float TurnStepDegrees = 5;
const float ZoomStepMeters = 0.25f;

const int Width = 512, Height = 424;
const cv::Vec3b Orange(0, 140, 255), Blue(255, 160, 0);   // OpenCV order: blue, green, red

// ---------------------------------------------------------------------
//  The viewer (copied from 08)
// ---------------------------------------------------------------------

float yawDegrees = 0, pitchDegrees = 0;
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
    const float yaw = yawDegrees * (float)CV_PI / 180, pitch = pitchDegrees * (float)CV_PI / 180;
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

// ---------------------------------------------------------------------
//  The new idea: a pose, and moving points by it
// ---------------------------------------------------------------------

struct Pose
{
    float right, up, forward, turnLeftDegrees;

    // A point seen by camera B -> the same point in camera A's coordinates (X right, Y down, Z forward).
    cv::Point3f ToA(const cv::Point3f& p) const
    {
        const float turn = turnLeftDegrees * (float)CV_PI / 180;
        // 1. Turn around the vertical axis: B's "straight ahead" becomes A's "ahead and to the left".
        float x = std::cos(turn) * p.x - std::sin(turn) * p.z;
        float z = std::sin(turn) * p.x + std::cos(turn) * p.z;
        // 2. Shift by where camera B stands. Y points down, so "up" is minus Y.
        return cv::Point3f(x + right, p.y - up, z + forward);
    }
};

int main()
{
    // -----------------------------------------------------------------
    //  Setup: depth only, as in 09
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

    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Depth);
    device->setIrAndDepthFrameListener(&listener);
    device->startStreams(false, true);
    libfreenect2::Registration registration(device->getIrCameraParams(), device->getColorCameraParams());
    libfreenect2::Frame undistorted(Width, Height, 4);

    std::vector<cv::Point3f> viewA, viewB;   // A in its own coordinates, B in B's own coordinates
    std::vector<float> depthA;               // A's distance per pixel (0 = none), to compare B against
    bool frozenB = false;
    Pose pose{MoveRightMeters, MoveUpMeters, MoveForwardMeters, TurnLeftDegrees};

    const std::string window = "10 two views  (SPACE capture, drag/WASD turn, JL TG IK UO nudge pose, Q quit)";
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(window, OnMouse);
    std::printf("Aim at the room and press SPACE to capture view A.\n");

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
        pitchDegrees = std::clamp(pitchDegrees, -89.0f, 89.0f);
        viewDistance = std::max(viewDistance, 0.1f);

        const std::string nudgeKeys = "ljtgikuo";
        if (key > 0 && nudgeKeys.find((char)key) != std::string::npos)
        {
            if (key == 'l') pose.right += NudgeMeters;
            if (key == 'j') pose.right -= NudgeMeters;
            if (key == 't') pose.up += NudgeMeters;
            if (key == 'g') pose.up -= NudgeMeters;
            if (key == 'i') pose.forward += NudgeMeters;
            if (key == 'k') pose.forward -= NudgeMeters;
            if (key == 'u') pose.turnLeftDegrees += NudgeDegrees;
            if (key == 'o') pose.turnLeftDegrees -= NudgeDegrees;
            std::printf("pose B: right %.2f m, up %.2f m, forward %.2f m, turned left %.1f deg\n",
                        pose.right, pose.up, pose.forward, pose.turnLeftDegrees);
        }

        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }

        // --- Step 1: the live frame -> points (as in 08) --------------------
        std::vector<cv::Point3f> live;
        std::vector<float> liveDepth(Width * Height, 0);
        registration.undistortDepth(frames[libfreenect2::Frame::Depth], &undistorted);
        listener.release(frames);
        for (int row = 0; row < Height; row++)
        {
            for (int column = 0; column < Width; column++)
            {
                float x, y, z;
                registration.getPointXYZ(&undistorted, row, column, x, y, z);
                if (std::isnan(z)) continue;
                live.emplace_back(x, y, z);
                liveDepth[row * Width + column] = z;
            }
        }

        // --- Step 2: SPACE moves through capture A -> freeze B -> start over --
        if (key == ' ')
        {
            if (viewA.empty())
            {
                viewA = live;
                depthA = liveDepth;
                std::printf("View A captured (%zu points). Move the Kinect by your measured pose; "
                            "the live view is now drawn as B. SPACE freezes B.\n", viewA.size());
            }
            else if (!frozenB)
            {
                frozenB = true;
                std::printf("View B frozen. Turn the view side on and nudge the pose until the walls meet.\n");
            }
            else
            {
                viewA.clear();
                frozenB = false;
                std::printf("Starting over. SPACE captures view A.\n");
            }
        }
        if (!frozenB) viewB = live;

        // --- Step 3: move B into A's coordinates, and measure how well they meet --
        // Project each moved B point back into camera A's image (the pinhole maths of 03) and
        // compare its distance with what A saw at that pixel. Aligned views: the gap is ~0.
        // ponytail: gap along A's line of sight only (blind to sliding along a wall); 11 measures properly.
        std::vector<cv::Point3f> points;
        std::vector<cv::Vec3b> colours;
        std::vector<float> gaps;
        const libfreenect2::Freenect2Device::IrCameraParams lens = device->getIrCameraParams();
        for (const cv::Point3f& p : viewA) { points.push_back(p); colours.push_back(Orange); }
        for (const cv::Point3f& b : viewB)
        {
            cv::Point3f p = viewA.empty() ? b : pose.ToA(b);   // before A exists, just show the live view
            points.push_back(p);
            colours.push_back(Blue);
            if (viewA.empty() || p.z <= 0) continue;
            int column = (int)std::lround(p.x * lens.fx / p.z + lens.cx - 0.5f);
            int row = (int)std::lround(p.y * lens.fy / p.z + lens.cy - 0.5f);
            if (column < 0 || row < 0 || column >= Width || row >= Height) continue;
            float seenByA = depthA[row * Width + column];
            if (seenByA > 0) gaps.push_back(std::abs(p.z - seenByA));
        }

        // -----------------------------------------------------------------
        //  Output: both clouds in one viewer
        // -----------------------------------------------------------------

        cv::Mat image = Render(points, colours);
        cv::Point pixel;
        float depth;
        if (Project(cv::Point3f(0, 0, 0), pixel, depth)) cv::circle(image, pixel, 6, cv::Scalar(0, 255, 255), 2);
        if (!viewA.empty() && Project(pose.ToA(cv::Point3f(0, 0, 0)), pixel, depth))
            cv::circle(image, pixel, 6, cv::Scalar(255, 255, 0), 2);

        std::string status = viewA.empty() ? "live - SPACE captures A" : frozenB ? "A + B (frozen)" : "A + B (live)";
        if (!gaps.empty())
        {
            size_t middle = gaps.size() / 2;
            std::nth_element(gaps.begin(), gaps.begin() + middle, gaps.end());
            status += cv::format("   median gap %.1f cm over %.0f%% of B", 100 * gaps[middle],
                                 100.0 * gaps.size() / viewB.size());
        }
        cv::putText(image, status, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
        cv::putText(image, cv::format("pose B: right %.2f  up %.2f  forward %.2f m  turned left %.1f deg",
                    pose.right, pose.up, pose.forward, pose.turnLeftDegrees),
                    cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
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
//  1. Capture A, then capture B WITHOUT moving the Kinect, with all the Move settings at 0.
//     The gap is just the sensor noise (a few mm). That's the best any pose can do.
//  2. Same, but nudge the pose 1 cm at a time (J/L). How far off before you can see a double wall
//     side on? The gap number barely moves (first test: 0.5 cm at 0, only 1.2 cm at 5 cm off),
//     because it is measured along A's line of sight, and sliding along a flat wall facing the
//     camera changes nothing on that line. Now nudge forward (I/K): the gap jumps. Only corners and
//     furniture pin a sideways shift, and ICP (11) has the same blind spot.
//  3. Nudge the turn by 1° (U/O, two presses). A 1° error moves things about 5 cm at 3 m:
//     far walls split, near ones barely. Turn errors hurt more than shift errors.
//  4. Slide the Kinect 30 cm right, as measured. Does your tape-measured pose give the smallest
//     gap, or does nudging find a better one? The difference is your measuring error.
//  5. Turn the Kinect about 20° left instead. A turn is hard to measure by hand: how many
//     degrees off was your guess? This is why 11 (ICP) and 12 (ArUco) exist.
//  6. Move it further, until little of B overlaps A (the % in the window). The gap number gets
//     unreliable as overlap shrinks: aligning needs shared surfaces.
