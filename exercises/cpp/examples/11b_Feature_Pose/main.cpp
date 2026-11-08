// =====================================================================
//  11b · Feature Pose
//
//  ICP (11) only works from a starting pose that's already roughly right,
//  and in 11 you had to give it one. This example finds the rough pose by
//  itself, the way photogrammetry tools like Meshroom do: from FEATURES,
//  small distinctive patches of the colour picture (a poster's corner, a
//  plug socket, the edge of a shelf) that can be found again in the other
//  view.
//
//    1. Features   ORB finds about 1,000 corners in each view's colour
//                  picture and describes each with 256 yes/no answers
//                  ("is this pixel brighter than that one?").
//    2. Match      each of B's features to the A feature with the most
//                  similar description, kept only if clearly better than
//                  the second best (the "ratio test").
//    3. Lift to 3D the depth under each feature makes it a 3D point, so
//                  every match is a pair of 3D points. (Meshroom has no
//                  depth: it has to work the 3D out from many photos.)
//    4. RANSAC     some matches are wrong. Try a pose from 3 random pairs
//                  (11's Kabsch fit, now spotter::Kabsch), count how many
//                  other pairs agree within 5 cm, keep the best, refit to
//                  all that agree.
//    5. ICP        polishes that rough pose with every point (spotter::Icp:
//                  13e's point-to-plane version of 11's ICP, tour 00f).
//
//  Window 1: orange = view A, blue = view B (C shows true colours).
//  Window 2: after B is captured, the two colour pictures side by side,
//            joined by the matches RANSAC kept.
//  Keys:     SPACE  capture A, then capture B (it lines up by itself),
//                   then start over
//            C  true colours on/off
//            mouse drag / W A S D  turn the view   mouse wheel / Z X  zoom
//            R  reset view     Q or Esc  quit
//
//  Try: capture A, carry the Kinect 50 cm to the side and turn it 20 deg
//  (something ICP alone gets wrong), capture B.
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <opencv2/features2d.hpp>   // ORB
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/geometry.hpp"
#include "spotter/register.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// --- Features (new) ---
// How many ORB features to look for in each colour picture.
const int FeatureCount = 1000;

// A match counts only if its best partner is clearly better than the second best: its
// difference must be under this share of the second's (Lowe's ratio test).
const float MatchRatio = 0.8f;

// RANSAC: random 3-pair guesses to try, and how close (metres) another pair must land to agree.
const int RansacTries = 1000;
const float FeatureInlierMeters = 0.05f;

// Fewer agreeing pairs than this = the two views don't share enough to trust the pose.
const int MinFeatureMatches = 15;

// --- ICP ---
const int IcpStep = 4;                    // every 4th pixel each way
const float IcpStartPairMeters = 0.05f;   // 13e's: the features already put B within a few cm

const float MaxDepthMeters = 6.0f;   // 07
const float PivotMeters = 2.5f;

const cv::Vec3b Orange(0, 140, 255), Blue(255, 160, 0);
const std::string Window = "11b feature pose  (SPACE capture, C colours, drag/WASD turn, Q quit)";
const std::string MatchWindow = "11b matches  (B left, A right)";

// ---------------------------------------------------------------------
//  The new idea: a rough pose from colour features
// ---------------------------------------------------------------------

// One captured view.
struct View
{
    std::vector<cv::Point3f> cloud;       // a 3D point per depth pixel (NaN = no reading), camera coordinates
    std::vector<cv::Vec3b> colours;       // each depth pixel's colour
    cv::Mat picture;                      // the colour picture on the depth grid (512 x 424; black = no colour)
    spotter::SurfacePoints surface;       // every IcpStep-th pixel's point and facing, for ICP
    std::vector<cv::KeyPoint> keypoints;  // step 1's features...
    cv::Mat descriptors;                  // ...and their descriptions (32 bytes = 256 yes/no answers each)
};

cv::Point3f CloudAt(const View& view, const cv::Point2f& pixel)
{
    int column = std::clamp((int)std::lround(pixel.x), 0, view.picture.cols - 1), row = std::clamp((int)std::lround(pixel.y), 0, view.picture.rows - 1);
    return view.cloud[row * view.picture.cols + column];
}

// Step 1: ORB features, only where there's both depth and colour (and not right at a hole's
// edge, where the depth under a corner is unreliable).
void FindFeatures(View& view)
{
    cv::Mat usable(view.picture.size(), CV_8U);
    for (int i = 0; i < (int)view.cloud.size(); i++)
        usable.data[i] = !std::isnan(view.cloud[i].z) && view.colours[i] != cv::Vec3b(0, 0, 0) ? 255 : 0;
    cv::erode(usable, usable, cv::Mat(), cv::Point(-1, -1), 3);
    cv::Mat grey;
    cv::cvtColor(view.picture, grey, cv::COLOR_BGR2GRAY);
    cv::ORB::create(FeatureCount)->detectAndCompute(grey, usable, view.keypoints, view.descriptors);
}

View Capture(const spotter::RgbdFrame& frame)
{
    View view;
    view.cloud = frame.depthCamera.Cloud(frame.depthMm);
    view.picture = frame.registered.clone();
    view.colours.assign(view.picture.begin<cv::Vec3b>(), view.picture.end<cv::Vec3b>());
    view.surface = spotter::SampleSurface(view.cloud, view.picture.cols, IcpStep);
    FindFeatures(view);
    return view;
}

// Steps 2-4: match B's features to A's, lift them to 3D, RANSAC a pose that moves B onto A.
// Returns how many matches agree with it (fewer than MinFeatureMatches = don't trust it).
int FeaturePose(const View& b, const View& a, std::mt19937& random, spotter::Pose& pose, std::vector<cv::DMatch>& agreeing)
{
    agreeing.clear();
    if (b.descriptors.empty() || a.descriptors.empty()) return 0;

    // Step 2: the two most similar A features for each B feature (Hamming distance: how many of
    // the 256 answers differ), kept if the best is clearly better than the second.
    std::vector<std::vector<cv::DMatch>> candidates;
    cv::BFMatcher(cv::NORM_HAMMING).knnMatch(b.descriptors, a.descriptors, candidates, 2);
    std::vector<cv::Point3f> from, to;
    std::vector<cv::DMatch> matches;
    for (const std::vector<cv::DMatch>& best : candidates)
    {
        if (best.size() < 2 || best[0].distance > MatchRatio * best[1].distance) continue;
        // Step 3: the depth under each feature makes the match a pair of 3D points.
        cv::Point3f p = CloudAt(b, b.keypoints[best[0].queryIdx].pt), q = CloudAt(a, a.keypoints[best[0].trainIdx].pt);
        if (std::isnan(p.z) || std::isnan(q.z)) continue;
        from.push_back(p);
        to.push_back(q);
        matches.push_back(best[0]);
    }
    if (from.size() < 3) return 0;

    // Step 4: RANSAC. A guess from 3 random pairs; the guess most other pairs agree with wins.
    std::uniform_int_distribution<size_t> pick(0, from.size() - 1);
    std::vector<size_t> bestAgreeing;
    for (int t = 0; t < RansacTries; t++)
    {
        size_t i = pick(random), j = pick(random), k = pick(random);
        if (cv::norm((from[j] - from[i]).cross(from[k] - from[i])) < 0.01) continue;   // in a line, or too close
        spotter::Pose guess = spotter::Kabsch({ from[i], from[j], from[k] }, { to[i], to[j], to[k] });
        std::vector<size_t> agree;
        for (size_t n = 0; n < from.size(); n++)
            if (cv::norm(guess(from[n]) - to[n]) < FeatureInlierMeters) agree.push_back(n);
        if (agree.size() > bestAgreeing.size()) bestAgreeing = agree;
    }
    if (bestAgreeing.size() < 3) return 0;
    std::vector<cv::Point3f> fromAgreeing, toAgreeing;
    for (size_t n : bestAgreeing)
    {
        fromAgreeing.push_back(from[n]);
        toAgreeing.push_back(to[n]);
        agreeing.push_back(matches[n]);
    }
    pose = spotter::Kabsch(fromAgreeing, toAgreeing);   // refit to every pair that agreed
    std::printf("  features: %zu in B, %zu in A, %zu matched, %zu agree with the best pose\n",
                b.keypoints.size(), a.keypoints.size(), from.size(), bestAgreeing.size());
    return (int)bestAgreeing.size();
}

void PrintPose(const char* what, const spotter::Pose& pose)
{
    std::printf("%s: right %+.3f m, up %+.3f m, forward %+.3f m, turned %.2f deg\n", what, pose.shift.x, -pose.shift.y, pose.shift.z, pose.Degrees());
}

// Steps 1-5 for B, then the match picture.
spotter::Pose LineUp(const View& b, const View& a, std::mt19937& random)
{
    spotter::Pose pose;
    std::vector<cv::DMatch> agreeing;
    int64 tick = cv::getTickCount();
    int count = FeaturePose(b, a, random, pose, agreeing);
    if (count < MinFeatureMatches)
    {
        std::printf("Only %d matches agree: not enough in common. Move less between A and B, or aim at\n"
                    "something with more pattern than a bare wall. SPACE starts over.\n", count);
        pose = spotter::Pose();
    }
    else
    {
        PrintPose("Rough pose from features", pose);
        spotter::IcpOptions options;
        options.startPairMeters = IcpStartPairMeters;
        spotter::IcpResult fit = spotter::Icp(b.surface.points, a.surface.points, a.surface.normals, pose, options);   // step 5
        pose = fit.pose;
        PrintPose(cv::format("After ICP (%.0f ms in all; %.0f%% paired, %.1f mm apart, %d weak directions)",
                             (cv::getTickCount() - tick) * 1000.0 / cv::getTickFrequency(), fit.pairedShare * 100,
                             fit.rmsMeters * 1000, fit.weakDirections).c_str(), pose);
    }
    cv::Mat matchPicture;
    cv::drawMatches(b.picture, b.keypoints, a.picture, a.keypoints, agreeing, matchPicture, cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255));
    cv::imshow(MatchWindow, matchPicture);
    return pose;
}

// ---------------------------------------------------------------------
//  Capturing: SPACE moves through capture A -> capture B and line it up -> start over
// ---------------------------------------------------------------------

struct Session
{
    View a, b;
    bool haveA = false, haveB = false, trueColours = false;
    spotter::Pose pose;
    std::mt19937 random{ 1 };
};

void OnSpace(Session& session, const spotter::RgbdFrame& frame)
{
    if (!session.haveA)
    {
        session.a = Capture(frame);
        session.haveA = true;
        std::printf("View A captured: %zu features. Move the Kinect, then SPACE captures B.\n", session.a.keypoints.size());
    }
    else if (!session.haveB)
    {
        session.b = Capture(frame);
        session.haveB = true;
        session.pose = LineUp(session.b, session.a, session.random);
    }
    else
    {
        session.haveA = session.haveB = false;
        session.pose = spotter::Pose();
        cv::destroyWindow(MatchWindow);
        std::printf("Starting over. SPACE captures view A.\n");
    }
}

// ---------------------------------------------------------------------
//  Output: A, and B (live until captured) moved by the pose
// ---------------------------------------------------------------------

void Show(const Session& session, const spotter::RgbdFrame& frame, spotter::OrbitView& view, spotter::PointRenderer& renderer)
{
    std::vector<cv::Point3f> b = session.haveB ? session.b.cloud : frame.depthCamera.Cloud(frame.depthMm);
    std::vector<cv::Vec3b> bColours = session.haveB ? session.b.colours : std::vector<cv::Vec3b>(frame.registered.begin<cv::Vec3b>(), frame.registered.end<cv::Vec3b>());
    if (session.haveA) for (cv::Point3f& p : b) p = session.pose(p);
    cv::Mat& image = renderer.Begin(view);
    if (session.haveA && session.trueColours) renderer.Draw(session.a.cloud, session.a.colours);
    else if (session.haveA) renderer.Draw(session.a.cloud, Orange);
    if (session.trueColours || !session.haveA) renderer.Draw(b, bColours);
    else renderer.Draw(b, Blue);
    cv::Point pixel;
    float depth;
    if (view.Project(cv::Point3f(0, 0, 0), pixel, depth)) cv::circle(image, pixel, 6, cv::Scalar(0, 255, 255), 2);
    if (session.haveA && view.Project(session.pose(cv::Point3f(0, 0, 0)), pixel, depth)) cv::circle(image, pixel, 6, cv::Scalar(255, 255, 0), 2);
    std::string status = !session.haveA ? "live - SPACE captures A" : !session.haveB ? "A + live - SPACE captures B" : "A + B lined up - SPACE starts over";
    cv::putText(image, status, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
    if (session.haveB)
        cv::putText(image, cv::format("pose B: right %+.2f  up %+.2f  forward %+.2f m  turned %.1f deg", session.pose.shift.x,
                    -session.pose.shift.y, session.pose.shift.z, session.pose.Degrees()), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
    cv::imshow(Window, image);
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    spotter::KinectSource::Options kinectOptions;   // depth AND colour now, registered onto the depth grid
    kinectOptions.maxDepthMeters = MaxDepthMeters;
    spotter::KinectSource kinect(kinectOptions);
    if (!kinect.IsOpen()) return 1;

    cv::namedWindow(Window, cv::WINDOW_AUTOSIZE);
    spotter::OrbitView view(spotter::OrbitView::Options(), cv::Point3f(0, 0, PivotMeters), PivotMeters);
    view.Attach(Window);
    spotter::PointRenderer renderer;
    Session session;
    spotter::RgbdFrame frame;
    std::printf("Aim at the room and press SPACE to capture view A.\n");

    while (kinect.Next(frame))
    {
        Show(session, frame, view, renderer);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == ' ') OnSpace(session, frame);
        if (key == 'c') session.trueColours = !session.trueColours;
        view.HandleKey(key);
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Look at the match window. Are the green lines joining the same thing in both pictures?
//     Where do features gather (posters, books, edges), and where are there none (bare walls)?
//  2. Move further each time: 20 cm, 50 cm, 1 m, then also turn 45 deg. When does "agree" drop
//     under MinFeatureMatches? That's how far apart two scan spots may be (13e needs it).
//  3. Turn the lights down. Features need a picture; depth doesn't. Which fails first?
//  4. MatchRatio = 0.95 versus 0.6: more matches but more wrong ones, or fewer but surer.
//  5. FeatureInlierMeters = 0.01 versus 0.2: what does RANSAC's "agree" count do?
//  6. Compare the rough pose with the pose after ICP. How far off were the features (often a
//     few cm)? Then try ICP alone from zero, as in 11, on the same big move.
//  7. Aim at a bare wall: "weak directions" goes up. ICP keeps the features' value there
//     instead of sliding on noise (13e's change to 11's ICP).
