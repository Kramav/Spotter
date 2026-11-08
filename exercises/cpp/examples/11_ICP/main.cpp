// =====================================================================
//  11 · ICP
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
//  starting pose by itself. (The library's spotter::Icp is 13e's better
//  version of this: tour 00f.)
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

#include <opencv2/flann.hpp>   // the k-d tree
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/source_freenect2.hpp"
#include "spotter/stats.hpp"
#include "spotter/view.hpp"

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

const float MaxDepthMeters = 6.0f;   // 07
const float PivotMeters = 2.5f;

const cv::Vec3b Orange(0, 140, 255), Blue(255, 160, 0);
const std::string Window = "11 ICP  (SPACE capture, P run ICP, drag/WASD turn, JL TG IK UO nudge, Q quit)";

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

// Nudges, as in 10, now on top of whatever pose ICP left.
void Nudge(Pose& pose, int key)
{
    if (key == 'l') pose.shift.x += NudgeMeters;
    if (key == 'j') pose.shift.x -= NudgeMeters;
    if (key == 't') pose.shift.y -= NudgeMeters;   // Y points down
    if (key == 'g') pose.shift.y += NudgeMeters;
    if (key == 'i') pose.shift.z += NudgeMeters;
    if (key == 'k') pose.shift.z -= NudgeMeters;
    if (key == 'u') pose.turn = TurnLeft(NudgeDegrees) * pose.turn;
    if (key == 'o') pose.turn = TurnLeft(-NudgeDegrees) * pose.turn;
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

// Steps 1-4: move "from" by the pose, pair with the nearest "to" points, fit, repeat. Prints each round.
Pose Icp(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to, Pose pose)
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
        std::printf("  ICP round %2d: %5zu pairs within %4.1f cm, %5.1f mm apart (rms); moved %.1f mm, %.2f deg\n",
                    round + 1, moved.size(), maxPair * 100, std::sqrt(sum / moved.size()) * 1000,
                    cv::norm(step.shift) * 1000, step.Degrees());
        if (maxPair == IcpFinalPairMeters && cv::norm(step.shift) < 1e-4 && step.Degrees() < 0.01f) break;   // settled
    }
    return pose;
}

// ---------------------------------------------------------------------
//  Capturing (as in 10), plus a sparse copy of each view for ICP
// ---------------------------------------------------------------------

struct Views
{
    std::vector<cv::Point3f> a, b, sparseA, sparseB;   // all points (NaN = none), and every IcpStep-th pixel's
    cv::Mat depthA;
    bool frozenB = false;
};

std::vector<cv::Point3f> Sparse(const std::vector<cv::Point3f>& cloud, int width)
{
    std::vector<cv::Point3f> sparse;
    for (size_t i = 0; i < cloud.size(); i++)
        if ((i / width) % IcpStep == 0 && (i % width) % IcpStep == 0 && !std::isnan(cloud[i].z)) sparse.push_back(cloud[i]);
    return sparse;
}

void OnSpace(Views& views, Pose& pose, const Pose& start, const spotter::RgbdFrame& frame)
{
    if (views.a.empty())
    {
        views.a = views.b;
        views.sparseA = views.sparseB;
        views.depthA = frame.depthMm.clone();
        std::printf("View A captured. Move the Kinect a little; SPACE freezes B, then P runs ICP.\n");
    }
    else if (!views.frozenB)
    {
        views.frozenB = true;
        std::printf("View B frozen. Press P to run ICP from the current pose.\n");
    }
    else
    {
        views = Views();
        pose = start;
        std::printf("Starting over. SPACE captures view A.\n");
    }
}

void RunIcp(const Views& views, Pose& pose)
{
    int64 tick = cv::getTickCount();
    pose = Icp(views.sparseB, views.sparseA, pose);
    std::printf("ICP done in %.0f ms. Pose B: right %+.3f m, up %+.3f m, forward %+.3f m, turned %.2f deg in all\n",
                (cv::getTickCount() - tick) * 1000.0 / cv::getTickFrequency(), pose.shift.x, -pose.shift.y, pose.shift.z, pose.Degrees());
}

// 10's gap: B's points projected into A's picture, compared along A's line of sight.
std::string GapText(const Views& views, const Pose& pose, const spotter::PinholeCamera& camera)
{
    std::vector<float> gaps;
    int seen = 0;
    for (const cv::Point3f& b : views.b)
    {
        if (std::isnan(b.z)) continue;
        seen++;
        cv::Point3f p = pose(b);
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
//  Output: both clouds in one viewer (as in 10)
// ---------------------------------------------------------------------

void Show(const Views& views, const Pose& pose, const spotter::PinholeCamera& camera, spotter::OrbitView& view, spotter::PointRenderer& renderer)
{
    const bool haveA = !views.a.empty();
    std::vector<cv::Point3f> b = views.b;
    if (haveA) for (cv::Point3f& p : b) p = pose(p);
    cv::Mat& image = renderer.Begin(view);
    renderer.Draw(views.a, Orange);
    renderer.Draw(b, Blue);
    cv::Point pixel;
    float depth;
    if (view.Project(cv::Point3f(0, 0, 0), pixel, depth)) cv::circle(image, pixel, 6, cv::Scalar(0, 255, 255), 2);
    if (haveA && view.Project(pose(cv::Point3f(0, 0, 0)), pixel, depth)) cv::circle(image, pixel, 6, cv::Scalar(255, 255, 0), 2);
    std::string status = !haveA ? "live - SPACE captures A" : views.frozenB ? "A + B (frozen) - P runs ICP" : "A + B (live)";
    if (haveA) status += GapText(views, pose, camera);
    cv::putText(image, status, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
    cv::putText(image, cv::format("pose B: right %+.2f  up %+.2f  forward %+.2f m  turned %.1f deg", pose.shift.x, -pose.shift.y,
                pose.shift.z, pose.Degrees()), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
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
    const Pose start{ TurnLeft(TurnLeftDegrees), cv::Point3f(MoveRightMeters, -MoveUpMeters, MoveForwardMeters) };
    Pose pose = start;
    Views views;
    spotter::RgbdFrame frame;
    std::printf("Aim at the room and press SPACE to capture view A.\n");

    while (kinect.Next(frame))
    {
        if (!views.frozenB) { views.b = frame.depthCamera.Cloud(frame.depthMm); views.sparseB = Sparse(views.b, frame.depthMm.cols); }
        Show(views, pose, frame.depthCamera, view, renderer);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == ' ') OnSpace(views, pose, start, frame);
        if (key == 'p' && views.frozenB) RunIcp(views, pose);
        Nudge(pose, key);
        view.HandleKey(key);
    }
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
//     (Tour 00f shows the same on made-up data, and how 13e's ICP refuses to slide.)
//  4. IcpFinalPairMeters = 0.005 versus 0.10: too strict leaves too few pairs to trust; too loose
//     pairs up things that only one view saw, and pulls the fit off.
//  5. IcpStep = 1 versus 8: how much slower, and is the answer any better?
//  6. Compare 10's "median gap" before and after P. Then look side on: are the double walls gone?
