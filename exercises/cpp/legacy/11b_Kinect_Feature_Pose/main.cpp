// =====================================================================
//  11b · Kinect Feature Pose
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
//                  (11's Kabsch fit), count how many other pairs agree
//                  within 5 cm, keep the best, refit to all that agree.
//    5. ICP        11's ICP polishes that rough pose with every point.
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

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/registration.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>   // ORB
#include <opencv2/flann.hpp>        // the k-d tree (11)

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

// --- ICP (as in 11) ---
const int IcpRounds = 30;
const float IcpStartPairMeters = 0.20f;
const float IcpFinalPairMeters = 0.02f;
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
//  The pose: a full turn (any axis) + shift (as in 11)
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


// ---------------------------------------------------------------------
//  ICP (as in 11)
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

// ---------------------------------------------------------------------
//  The new idea: a rough pose from colour features
// ---------------------------------------------------------------------

// One captured view.
struct View
{
    std::vector<cv::Point3f> cloud;    // a 3D point per depth pixel (NaN = no reading), camera coordinates
    cv::Mat picture;                   // the colour picture on the depth grid (512 x 424; black = no colour)
    std::vector<cv::Point3f> points, sparse;   // the readings, and every IcpStep-th pixel's (for ICP)
    std::vector<cv::Vec3b> colours;            // each reading's colour
    std::vector<cv::KeyPoint> keypoints;       // step 1's features...
    cv::Mat descriptors;                       // ...and their descriptions (32 bytes = 256 yes/no answers each)
};

cv::Point3f CloudAt(const View& view, const cv::Point2f& pixel)
{
    int column = std::clamp((int)std::lround(pixel.x), 0, Width - 1), row = std::clamp((int)std::lround(pixel.y), 0, Height - 1);
    return view.cloud[row * Width + column];
}

// Step 1: ORB features, only where there's both depth and colour (and not right at a hole's
// edge, where the depth under a corner is unreliable).
void FindFeatures(View& view)
{
    cv::Mat usable(Height, Width, CV_8U);
    for (int i = 0; i < Width * Height; i++)
    {
        cv::Vec3b colour = view.picture.at<cv::Vec3b>(i / Width, i % Width);
        usable.data[i] = !std::isnan(view.cloud[i].z) && colour != cv::Vec3b(0, 0, 0) ? 255 : 0;
    }
    cv::erode(usable, usable, cv::Mat(), cv::Point(-1, -1), 3);
    cv::Mat grey;
    cv::cvtColor(view.picture, grey, cv::COLOR_BGR2GRAY);
    cv::ORB::create(FeatureCount)->detectAndCompute(grey, usable, view.keypoints, view.descriptors);
}

// Steps 2-4: match B's features to A's, lift them to 3D, RANSAC a pose that moves B onto A.
// Returns how many matches agree with it (fewer than MinFeatureMatches = don't trust it).
int FeaturePose(const View& b, const View& a, std::mt19937& random, Pose& pose, std::vector<cv::DMatch>& agreeing)
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
        Pose guess = BestFit({ from[i], from[j], from[k] }, { to[i], to[j], to[k] });
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
    pose = BestFit(fromAgreeing, toAgreeing);   // refit to every pair that agreed
    std::printf("  features: %zu in B, %zu in A, %zu matched, %zu agree with the best pose\n",
                b.keypoints.size(), a.keypoints.size(), from.size(), bestAgreeing.size());
    return (int)bestAgreeing.size();
}

int main()
{
    // -----------------------------------------------------------------
    //  Setup: depth AND colour now (as in 08), registered onto the depth grid
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
    device->startStreams(true, true);
    libfreenect2::Registration registration(device->getIrCameraParams(), device->getColorCameraParams());
    libfreenect2::Frame undistorted(Width, Height, 4), registered(Width, Height, 4);

    View viewA, viewB, live;
    bool haveA = false, haveB = false, trueColours = false;
    Pose pose;
    std::mt19937 random(1);

    const std::string window = "11b feature pose  (SPACE capture, C colours, drag/WASD turn, Q quit)";
    const std::string matchWindow = "11b matches  (B left, A right)";
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
        if (key == 'c') trueColours = !trueColours;
        pitchDegrees = std::clamp(pitchDegrees, -89.0f, 89.0f);
        viewDistance = std::max(viewDistance, 0.1f);

        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }

        // --- The live frame -> a View (points as in 11, plus colour) ---
        registration.apply(frames[libfreenect2::Frame::Color], frames[libfreenect2::Frame::Depth], &undistorted, &registered);
        listener.release(frames);
        cv::cvtColor(cv::Mat(Height, Width, CV_8UC4, registered.data), live.picture, cv::COLOR_BGRA2BGR);
        live.cloud.assign(Width * Height, cv::Point3f(NAN, NAN, NAN));
        live.points.clear();
        live.sparse.clear();
        live.colours.clear();
        for (int row = 0; row < Height; row++)
        {
            for (int column = 0; column < Width; column++)
            {
                float x, y, z;
                registration.getPointXYZ(&undistorted, row, column, x, y, z);
                if (std::isnan(z)) continue;
                live.cloud[row * Width + column] = cv::Point3f(x, y, z);
                live.points.emplace_back(x, y, z);
                live.colours.push_back(live.picture.at<cv::Vec3b>(row, column));
                if (row % IcpStep == 0 && column % IcpStep == 0) live.sparse.emplace_back(x, y, z);
            }
        }

        // --- SPACE: capture A -> capture B and line it up -> start over ---
        if (key == ' ')
        {
            if (!haveA)
            {
                viewA = live;
                viewA.picture = live.picture.clone();
                FindFeatures(viewA);
                haveA = true;
                std::printf("View A captured: %zu points, %zu features. Move the Kinect, then SPACE captures B.\n",
                            viewA.points.size(), viewA.keypoints.size());
            }
            else if (!haveB)
            {
                viewB = live;
                viewB.picture = live.picture.clone();
                FindFeatures(viewB);
                haveB = true;
                std::vector<cv::DMatch> agreeing;
                int64 tick = cv::getTickCount();
                int count = FeaturePose(viewB, viewA, random, pose, agreeing);
                if (count < MinFeatureMatches)
                {
                    std::printf("Only %d matches agree: not enough in common. Move less between A and B, or aim at\n"
                                "something with more pattern than a bare wall. SPACE starts over.\n", count);
                    pose = Pose();
                }
                else
                {
                    std::printf("Rough pose from features: right %+.3f m, up %+.3f m, forward %+.3f m, turned %.1f deg\n",
                                pose.shift.x, -pose.shift.y, pose.shift.z, pose.Degrees());
                    pose = Icp(viewB.sparse, viewA.sparse, pose, true);   // step 5
                    std::printf("After ICP (%.0f ms in all): right %+.3f m, up %+.3f m, forward %+.3f m, turned %.2f deg\n",
                                (cv::getTickCount() - tick) * 1000.0 / cv::getTickFrequency(),
                                pose.shift.x, -pose.shift.y, pose.shift.z, pose.Degrees());
                }
                cv::Mat matchPicture;
                cv::drawMatches(viewB.picture, viewB.keypoints, viewA.picture, viewA.keypoints, agreeing, matchPicture,
                                cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255));
                cv::imshow(matchWindow, matchPicture);
            }
            else
            {
                haveA = haveB = false;
                pose = Pose();
                cv::destroyWindow(matchWindow);
                std::printf("Starting over. SPACE captures view A.\n");
            }
        }

        // -----------------------------------------------------------------
        //  Output: A, and B (live until captured) moved by the pose
        // -----------------------------------------------------------------

        const View& b = haveB ? viewB : live;
        std::vector<cv::Point3f> points;
        std::vector<cv::Vec3b> colours;
        if (haveA)
        {
            for (size_t i = 0; i < viewA.points.size(); i++)
            {
                points.push_back(viewA.points[i]);
                colours.push_back(trueColours ? viewA.colours[i] : Orange);
            }
        }
        for (size_t i = 0; i < b.points.size(); i++)
        {
            points.push_back(haveA ? pose(b.points[i]) : b.points[i]);
            colours.push_back(trueColours || !haveA ? b.colours[i] : Blue);
        }
        cv::Mat image = Render(points, colours);
        cv::Point pixel;
        float depth;
        if (Project(cv::Point3f(0, 0, 0), pixel, depth)) cv::circle(image, pixel, 6, cv::Scalar(0, 255, 255), 2);
        if (haveA && Project(pose(cv::Point3f(0, 0, 0)), pixel, depth)) cv::circle(image, pixel, 6, cv::Scalar(255, 255, 0), 2);
        std::string status = !haveA ? "live - SPACE captures A" : !haveB ? "A + live - SPACE captures B" : "A + B lined up - SPACE starts over";
        cv::putText(image, status, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
        if (haveB)
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
//  1. Look at the match window. Are the green lines joining the same thing in both pictures?
//     Where do features gather (posters, books, edges), and where are there none (bare walls)?
//  2. Move further each time: 20 cm, 50 cm, 1 m, then also turn 45 deg. When does "agree" drop
//     under MinFeatureMatches? That's how far apart two scan spots may be (13 needs it).
//  3. Turn the lights down. Features need a picture; depth doesn't. Which fails first?
//  4. MatchRatio = 0.95 versus 0.6: more matches but more wrong ones, or fewer but surer.
//  5. FeatureInlierMeters = 0.01 versus 0.2: what does RANSAC's "agree" count do?
//  6. Compare the rough pose with the pose after ICP. How far off were the features (often a
//     few cm)? Then try ICP alone from zero, as in 11, on the same big move.
