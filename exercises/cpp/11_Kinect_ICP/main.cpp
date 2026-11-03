// =====================================================================
//  11 · Kinect ICP
//
//  In example 10 you lined two views up by hand, nudging the pose until
//  the walls stopped showing double. ICP (Iterative Closest Point) does
//  the nudging itself:
//
//    1. Move B's points by the current pose (10's settings to start).
//    2. Pair each one with the NEAREST point of A (a k-d tree finds it
//       fast). Pairs farther apart than a few cm are ignored: they're
//       parts of the room only one view saw.
//    3. Find the one turn + shift that brings the pairs closest together.
//       That has an exact answer: centre both sets, and the SVD of how
//       they vary together gives the turn (the "Kabsch" method).
//    4. Apply it and repeat. The pairs get better each round, the pose
//       too, until nothing moves any more.
//
//  ICP only finds the NEAREST good fit: it needs a starting pose roughly
//  right (within about 10-20 cm and 10-15 degrees). Example 11b finds that
//  starting pose by itself.
//
//  Window: orange = view A (captured), blue = view B (live until frozen).
//          Yellow circle = camera A, cyan circle = camera B.
//  Keys:   SPACE  capture A, then freeze B, then start over
//          P  run ICP (after B is frozen)
//          mouse drag / W A S D  turn the view   mouse wheel / Z X  zoom   R  reset view
//          nudge pose B:  J L  right   T G  up   I K  forward   U O  turn left
//          Q or Esc  quit
//
//  Try: capture A, move the Kinect about 20 cm and turn it a little,
//  freeze B, press P.
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
#include <opencv2/flann.hpp>   // the k-d tree

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// Where camera B stands relative to camera A, as in 10: ICP starts from this guess.
const float MoveRightMeters = 0.0f;
const float MoveUpMeters = 0.0f;
const float MoveForwardMeters = 0.0f;
const float TurnLeftDegrees = 0.0f;

const float NudgeMeters = 0.01f;
const float NudgeDegrees = 0.5f;

// --- ICP (new) ---
// Rounds of pair-up-and-fit. It usually settles in 10-20.
const int IcpRounds = 30;

// Pairs farther apart than this are ignored. It starts generous (the first guess may be well off)
// and shrinks by 30% each round down to the final value (about the sensor's noise plus a bit).
const float IcpStartPairMeters = 0.20f;
const float IcpFinalPairMeters = 0.02f;

// Only every Nth depth pixel each way goes into ICP (4 -> about 13,000 points per view).
const int IcpStep = 4;

const float MaxDepthMeters = 6.0f;   // 07b

// The viewer, as in 08.
const int ViewWidth = 960, ViewHeight = 720;
const float ViewFocalPixels = 700;
const float PivotMeters = 2.5f;
const int PointSizePixels = 2;
const float TurnStepDegrees = 5;
const float ZoomStepMeters = 0.25f;

const int Width = 512, Height = 424;
const cv::Vec3b Orange(0, 140, 255), Blue(255, 160, 0);

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
//  The pose, now a full turn (any axis) + shift: ICP may tilt it too
// ---------------------------------------------------------------------

struct Pose
{
    cv::Matx33f turn = cv::Matx33f::eye();
    cv::Point3f shift = cv::Point3f(0, 0, 0);

    // A point in B's coordinates -> the same point in A's (X right, Y down, Z forward).
    cv::Point3f operator()(const cv::Point3f& p) const { return turn * p + shift; }

    // This pose first, then "next" on top of it.
    Pose Then(const Pose& next) const { return { next.turn * turn, next.turn * shift + next.shift }; }

    // How far it turns, whatever the axis (the angle of a rotation matrix, from its trace).
    float Degrees() const
    {
        float cosine = (turn(0, 0) + turn(1, 1) + turn(2, 2) - 1) / 2;
        return std::acos(std::clamp(cosine, -1.0f, 1.0f)) * 180 / (float)CV_PI;
    }
};

// A turn to the left by some degrees, about the vertical axis (as 10's pose).
cv::Matx33f TurnLeft(float degrees)
{
    float a = degrees * (float)CV_PI / 180;
    return cv::Matx33f(std::cos(a), 0, -std::sin(a),
                       0,           1, 0,
                       std::sin(a), 0, std::cos(a));
}

// ---------------------------------------------------------------------
//  The new idea: ICP
// ---------------------------------------------------------------------

// Step 3: the turn + shift that best moves each "from" point onto its partner in "to" (Kabsch).
Pose BestFit(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to)
{
    // Centre both sets: what's left is the turn.
    cv::Point3f fromCentre(0, 0, 0), toCentre(0, 0, 0);
    for (size_t i = 0; i < from.size(); i++)
    {
        fromCentre += from[i];
        toCentre += to[i];
    }
    fromCentre *= 1.0f / from.size();
    toCentre *= 1.0f / to.size();

    // How the two sets vary together, axis by axis (a 3x3 "cross-covariance").
    cv::Matx33d together = cv::Matx33d::zeros();
    for (size_t i = 0; i < from.size(); i++)
    {
        cv::Point3f p = from[i] - fromCentre, q = to[i] - toCentre;
        const double a[3] = { p.x, p.y, p.z }, b[3] = { q.x, q.y, q.z };
        for (int row = 0; row < 3; row++)
            for (int column = 0; column < 3; column++) together(row, column) += a[row] * b[column];
    }
    // Its SVD (together = U * S * Vt) gives the turn: V * Ut.
    cv::Mat s, u, vt;
    cv::SVD::compute(cv::Mat(together), s, u, vt);
    cv::Mat turn = vt.t() * u.t();
    if (cv::determinant(turn) < 0)   // that's a mirror image, not a turn: flip the weakest axis
    {
        vt.row(2) *= -1;
        turn = vt.t() * u.t();
    }
    Pose pose;
    pose.turn = cv::Matx33f(cv::Matx33d(turn));
    pose.shift = toCentre - pose.turn * fromCentre;
    return pose;
}

// Steps 1-4: move "from" by the pose, pair with the nearest "to" points, fit, repeat.
// Prints each round when asked. Returns the improved pose.
Pose Icp(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to, Pose pose, bool print)
{
    if (from.size() < 10 || to.size() < 10) return pose;
    cv::Mat toData((int)to.size(), 3, CV_32F, (void*)to.data());
    cv::flann::Index tree(toData, cv::flann::KDTreeIndexParams(4));
    cv::Mat query((int)from.size(), 3, CV_32F), nearestIndex, squaredDistance;
    for (int round = 0; round < IcpRounds; round++)
    {
        // Step 1: move B by the current pose.
        for (int i = 0; i < (int)from.size(); i++)
        {
            cv::Point3f p = pose(from[i]);
            query.at<float>(i, 0) = p.x;
            query.at<float>(i, 1) = p.y;
            query.at<float>(i, 2) = p.z;
        }
        // Step 2: nearest A point to each (squared distances), keeping only close pairs.
        tree.knnSearch(query, nearestIndex, squaredDistance, 1, cv::flann::SearchParams(32));
        float maxPair = std::max(IcpFinalPairMeters, IcpStartPairMeters * std::pow(0.7f, (float)round));
        std::vector<cv::Point3f> moved, partners;
        double sum = 0;
        for (int i = 0; i < (int)from.size(); i++)
        {
            float squared = squaredDistance.at<float>(i);
            if (squared > maxPair * maxPair) continue;
            moved.emplace_back(query.at<float>(i, 0), query.at<float>(i, 1), query.at<float>(i, 2));
            partners.push_back(to[nearestIndex.at<int>(i)]);
            sum += squared;
        }
        if (moved.size() < 10) break;   // nothing in common: give up, keep the pose

        // Steps 3-4: the best fit for these pairs, on top of the pose so far.
        Pose step = BestFit(moved, partners);
        pose = pose.Then(step);
        if (print)
            std::printf("  ICP round %2d: %5zu pairs within %4.1f cm, %5.1f mm apart (rms); moved %.1f mm, %.2f deg\n",
                        round + 1, moved.size(), maxPair * 100, std::sqrt(sum / moved.size()) * 1000,
                        cv::norm(step.shift) * 1000, step.Degrees());
        if (maxPair == IcpFinalPairMeters && cv::norm(step.shift) < 1e-4 && step.Degrees() < 0.01f) break;   // settled
    }
    return pose;
}

int main()
{
    // -----------------------------------------------------------------
    //  Setup: depth only, as in 10
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

    std::vector<cv::Point3f> viewA, viewB, sparseA, sparseB;   // all points, and every IcpStep-th pixel's
    std::vector<float> depthA;
    bool frozenB = false;
    Pose start;
    start.turn = TurnLeft(TurnLeftDegrees);
    start.shift = cv::Point3f(MoveRightMeters, -MoveUpMeters, MoveForwardMeters);
    Pose pose = start;

    const std::string window = "11 ICP  (SPACE capture, P run ICP, drag/WASD turn, JL TG IK UO nudge, Q quit)";
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

        // Nudges, as in 10, now on top of whatever pose ICP left.
        if (key == 'l') pose.shift.x += NudgeMeters;
        if (key == 'j') pose.shift.x -= NudgeMeters;
        if (key == 't') pose.shift.y -= NudgeMeters;   // Y points down
        if (key == 'g') pose.shift.y += NudgeMeters;
        if (key == 'i') pose.shift.z += NudgeMeters;
        if (key == 'k') pose.shift.z -= NudgeMeters;
        if (key == 'u') pose.turn = TurnLeft(NudgeDegrees) * pose.turn;
        if (key == 'o') pose.turn = TurnLeft(-NudgeDegrees) * pose.turn;

        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }

        // --- The live frame -> points (as in 10), plus a sparse copy for ICP ---
        std::vector<cv::Point3f> live, liveSparse;
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
                if (row % IcpStep == 0 && column % IcpStep == 0) liveSparse.emplace_back(x, y, z);
            }
        }

        // --- SPACE: capture A -> freeze B -> start over (as in 10) ---
        if (key == ' ')
        {
            if (viewA.empty())
            {
                viewA = live;
                sparseA = liveSparse;
                depthA = liveDepth;
                std::printf("View A captured (%zu points). Move the Kinect a little; SPACE freezes B, then P runs ICP.\n", viewA.size());
            }
            else if (!frozenB)
            {
                frozenB = true;
                std::printf("View B frozen. Press P to run ICP from the current pose.\n");
            }
            else
            {
                viewA.clear();
                frozenB = false;
                pose = start;
                std::printf("Starting over. SPACE captures view A.\n");
            }
        }
        if (!frozenB) { viewB = live; sparseB = liveSparse; }

        // --- P: ICP from the current pose (new) ---
        if (key == 'p' && frozenB)
        {
            int64 tick = cv::getTickCount();
            pose = Icp(sparseB, sparseA, pose, true);
            double ms = (cv::getTickCount() - tick) * 1000.0 / cv::getTickFrequency();
            std::printf("ICP done in %.0f ms. Pose B: right %+.3f m, up %+.3f m, forward %+.3f m, turned %.2f deg in all\n",
                        ms, pose.shift.x, -pose.shift.y, pose.shift.z, pose.Degrees());
        }

        // --- Move B into A's coordinates, and measure the gap (as in 10) ---
        std::vector<cv::Point3f> points;
        std::vector<cv::Vec3b> colours;
        std::vector<float> gaps;
        const libfreenect2::Freenect2Device::IrCameraParams lens = device->getIrCameraParams();
        for (const cv::Point3f& p : viewA) { points.push_back(p); colours.push_back(Orange); }
        for (const cv::Point3f& b : viewB)
        {
            cv::Point3f p = viewA.empty() ? b : pose(b);
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
        //  Output: both clouds in one viewer (as in 10)
        // -----------------------------------------------------------------

        cv::Mat image = Render(points, colours);
        cv::Point pixel;
        float depth;
        if (Project(cv::Point3f(0, 0, 0), pixel, depth)) cv::circle(image, pixel, 6, cv::Scalar(0, 255, 255), 2);
        if (!viewA.empty() && Project(pose(cv::Point3f(0, 0, 0)), pixel, depth))
            cv::circle(image, pixel, 6, cv::Scalar(255, 255, 0), 2);

        std::string status = viewA.empty() ? "live - SPACE captures A" : frozenB ? "A + B (frozen) - P runs ICP" : "A + B (live)";
        if (!gaps.empty())
        {
            size_t middle = gaps.size() / 2;
            std::nth_element(gaps.begin(), gaps.begin() + middle, gaps.end());
            status += cv::format("   median gap %.1f cm over %.0f%% of B", 100 * gaps[middle],
                                 100.0 * gaps.size() / viewB.size());
        }
        cv::putText(image, status, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
        cv::putText(image, cv::format("pose B: right %+.2f  up %+.2f  forward %+.2f m  turned %.1f deg",
                    pose.shift.x, -pose.shift.y, pose.shift.z, pose.Degrees()),
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
//  1. Capture A and B without moving the Kinect, then nudge the pose 10 cm off (J/L) and press P.
//     Does ICP come back to 0? Watch the "mm apart" fall round by round.
//  2. Move the Kinect 20 cm and turn it 10 deg, freeze B, press P with the pose at 0. Then try
//     50 cm and 30 deg: ICP settles on a WRONG fit (the pairs it starts with are wrong). That's
//     why 11b finds a starting pose first.
//  3. Point the Kinect at one flat, bare wall and slide it sideways. ICP can't tell where along
//     the wall it is: any slide fits equally well. Add a chair to the view and try again.
//  4. IcpFinalPairMeters = 0.005 versus 0.10: too strict leaves too few pairs to trust; too loose
//     pairs up things that only one view saw, and pulls the fit off.
//  5. IcpStep = 1 versus 8: how much slower, and is the answer any better?
//  6. Compare 10's "median gap" before and after P. Then look side on: are the double walls gone?
