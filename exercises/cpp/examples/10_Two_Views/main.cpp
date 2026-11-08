// =====================================================================
//  10 · Two Views
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

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/source_freenect2.hpp"
#include "spotter/stats.hpp"
#include "spotter/view.hpp"

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

const float MaxDepthMeters = 6.0f;   // 07: fine to about 6 m
const float PivotMeters = 2.5f;      // the viewer turns around this point, straight ahead

const cv::Vec3b Orange(0, 140, 255), Blue(255, 160, 0);   // OpenCV order: blue, green, red
const std::string Window = "10 two views  (SPACE capture, drag/WASD turn, JL TG IK UO nudge pose, Q quit)";

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

void Nudge(Pose& pose, int key)
{
    const std::string keys = "ljtgikuo";
    if (key <= 0 || keys.find((char)key) == std::string::npos) return;
    if (key == 'l') pose.right += NudgeMeters;
    if (key == 'j') pose.right -= NudgeMeters;
    if (key == 't') pose.up += NudgeMeters;
    if (key == 'g') pose.up -= NudgeMeters;
    if (key == 'i') pose.forward += NudgeMeters;
    if (key == 'k') pose.forward -= NudgeMeters;
    if (key == 'u') pose.turnLeftDegrees += NudgeDegrees;
    if (key == 'o') pose.turnLeftDegrees -= NudgeDegrees;
    std::printf("pose B: right %.2f m, up %.2f m, forward %.2f m, turned left %.1f deg\n", pose.right, pose.up, pose.forward, pose.turnLeftDegrees);
}

// ---------------------------------------------------------------------
//  Capturing: SPACE moves through capture A -> freeze B -> start over
// ---------------------------------------------------------------------

struct Views
{
    std::vector<cv::Point3f> a, b;   // A in its own coordinates, B in B's (NaN = no reading)
    cv::Mat depthA;                  // A's depth picture (mm), to compare B against
    bool frozenB = false;
};

void OnSpace(Views& views, const spotter::RgbdFrame& frame)
{
    if (views.a.empty())
    {
        views.a = frame.depthCamera.Cloud(frame.depthMm);
        views.depthA = frame.depthMm.clone();
        std::printf("View A captured. Move the Kinect by your measured pose; the live view is now B. SPACE freezes B.\n");
    }
    else if (!views.frozenB)
    {
        views.frozenB = true;
        std::printf("View B frozen. Turn the view side on and nudge the pose until the walls meet.\n");
    }
    else
    {
        views = Views();
        std::printf("Starting over. SPACE captures view A.\n");
    }
}

// How well they meet: each moved B point is projected into camera A's picture (the pinhole maths of
// 03) and its distance compared with what A saw there. Aligned views: the gap is about 0.
// ponytail: gap along A's line of sight only (blind to sliding along a wall); 11 measures properly.
std::string GapText(const Views& views, const Pose& pose, const spotter::PinholeCamera& camera)
{
    std::vector<float> gaps;
    int seen = 0;
    for (const cv::Point3f& b : views.b)
    {
        if (std::isnan(b.z)) continue;
        seen++;
        cv::Point3f p = pose.ToA(b);
        std::optional<cv::Point2f> pixel = camera.Project(p);
        if (!pixel) continue;
        int column = (int)std::lround(pixel->x), row = (int)std::lround(pixel->y);
        if (column < 0 || row < 0 || column >= views.depthA.cols || row >= views.depthA.rows) continue;
        float seenByA = views.depthA.at<float>(row, column) / 1000;
        if (seenByA > 0) gaps.push_back(std::abs(p.z - seenByA));
    }
    if (gaps.empty()) return "";
    return cv::format("   median gap %.1f cm over %.0f%% of B", 100 * spotter::Median(gaps), 100.0 * gaps.size() / seen);
}

// ---------------------------------------------------------------------
//  Output: both clouds in one viewer
// ---------------------------------------------------------------------

void Show(const Views& views, const Pose& pose, const spotter::PinholeCamera& camera, spotter::OrbitView& view, spotter::PointRenderer& renderer)
{
    const bool haveA = !views.a.empty();
    std::vector<cv::Point3f> b = views.b;
    if (haveA) for (cv::Point3f& p : b) p = pose.ToA(p);   // before A exists, just show the live view
    cv::Mat& image = renderer.Begin(view);
    renderer.Draw(views.a, Orange);
    renderer.Draw(b, Blue);
    cv::Point pixel;
    float depth;
    if (view.Project(cv::Point3f(0, 0, 0), pixel, depth)) cv::circle(image, pixel, 6, cv::Scalar(0, 255, 255), 2);
    if (haveA && view.Project(pose.ToA(cv::Point3f(0, 0, 0)), pixel, depth)) cv::circle(image, pixel, 6, cv::Scalar(255, 255, 0), 2);
    std::string status = !haveA ? "live - SPACE captures A" : views.frozenB ? "A + B (frozen)" : "A + B (live)";
    if (haveA) status += GapText(views, pose, camera);
    cv::putText(image, status, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
    cv::putText(image, cv::format("pose B: right %.2f  up %.2f  forward %.2f m  turned left %.1f deg", pose.right, pose.up, pose.forward,
                pose.turnLeftDegrees), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
    cv::imshow(Window, image);
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    spotter::KinectSource::Options kinectOptions;
    kinectOptions.colour = false;   // depth only
    kinectOptions.maxDepthMeters = MaxDepthMeters;
    spotter::KinectSource kinect(kinectOptions);
    if (!kinect.IsOpen()) return 1;

    cv::namedWindow(Window, cv::WINDOW_AUTOSIZE);
    spotter::OrbitView view(spotter::OrbitView::Options(), cv::Point3f(0, 0, PivotMeters), PivotMeters);
    view.Attach(Window);
    spotter::PointRenderer renderer;
    Pose pose{ MoveRightMeters, MoveUpMeters, MoveForwardMeters, TurnLeftDegrees };
    Views views;
    spotter::RgbdFrame frame;
    std::printf("Aim at the room and press SPACE to capture view A.\n");

    while (kinect.Next(frame))
    {
        if (!views.frozenB) views.b = frame.depthCamera.Cloud(frame.depthMm);
        Show(views, pose, frame.depthCamera, view, renderer);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == ' ') OnSpace(views, frame);
        Nudge(pose, key);
        view.HandleKey(key);
    }
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
