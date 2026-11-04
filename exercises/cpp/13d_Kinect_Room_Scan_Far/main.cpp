// =====================================================================
//  13d · Kinect Room Scan, far features
//
//  13c, plus what's too far for depth. A long hallway kept failing in 13b
//  and 13c: its walls are plain, and the doorway at the far end has no
//  depth reading (past about 6 m), so its features were never used. But a
//  detail without depth still gives a DIRECTION (a ray). New:
//
//    1. Far features  ORB on the FULL colour picture where depth reads
//                     nothing, each as a ray. The colour camera sits about
//                     5 cm from the depth camera; that offset, and how its
//                     pixels map onto the depth camera's view, is measured
//                     from every scan's own pixels that do have depth.
//    2. They vote     a far match agrees with a pose if its two rays meet
//                     far away. Near and far matches count together, so a
//                     plain hallway with a detailed doorway can now pass.
//    3. Turn by rays  with under 3 near matches, the turn comes from the far
//                     rays alone (a far doorway is a good compass). Where
//                     the camera is along the hallway can't come from rays
//                     alone (they carry no distance), only from:
//    4. Landmarks     after each placement, far matches seen from two spots
//                     at a clear angle are triangulated into points in the
//                     room. Later scans' rays to them pin the position too.
//                     Magenta in the viewer, and saved in the .obj
//                     (far_landmarks).
//
//  Everything else is 13c: live anchors (far ones magenta), next shot,
//  red = suspicious, click + X / P, SPACE, U, E (room_scan_<time>.obj).
//  A scan placed by far rays with no landmarks to pin its distance is
//  flagged red: its position along the hallway was kept from the scan it
//  matched. Move sideways a little between scans to make landmarks.
//
//  Added later, as in 13c: K keeps the scans. It asks (in the console) for
//  a folder and a name, and writes every scan's points, pictures, far
//  features and placement there, the landmarks, and points.ply: every
//  scan's own points and colours, what the viewer shows (E's .obj averages
//  them). 13z rebuilds rooms from such folders later, without the Kinect.
//
//  Window 1 (viewer): every scan placed in the room. C colours each scan
//            differently. Circles + 4 arrows = where each camera stood:
//            yellow = fine, red = suspicious, grey = excluded, white =
//            selected, green = the next shot. Magenta dots = far landmarks.
//  Window 2 (camera): what the Kinect sees now, the live anchors (green =
//            near, magenta = far), what to do next.
//  Keys:     SPACE scan here   U undo the last scan   E export (merge + save)
//            K keep the scans (a folder, for 13z)
//            click a camera circle to select it: X exclude / include   P place again
//            Esc deselect      C colour by scan
//            mouse drag / W A S D turn    wheel / + - zoom    R reset view    Q quit
//
//  Privacy: the scans stay in memory unless you press K. E writes
//  room_scan_<time>.obj and K a folder of pictures and points: both are
//  captures of the room in colour. *.obj, *.ply and room_scans/ are
//  gitignored, and K puts a .gitignore in every folder it writes: never
//  commit them. Scan with nobody in view.
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/registration.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// --- Scanning (07, 07b) ---
const int AverageFrames = 60;   // 13: 30. Twice the frames: noise down by about 1.4x (07: noise falls as 1 / sqrt(frames))
const float MaxDepthMeters = 6.0f;

// --- Recognition (new) ---
// CLAHE: the picture is cut into ClaheTiles x ClaheTiles tiles and each tile's contrast stretched
// to use the full range, so a dim corner gets the same chance as a bright one. ClaheClipLimit
// caps the stretch (higher = more contrast, but also more amplified sensor noise).
const bool EqualiseContrast = true;
const double ClaheClipLimit = 2.0;
const int ClaheTiles = 8;
// ORB finds corners with the FAST test: a pixel is a corner if a ring of pixels around it is this
// much brighter or darker (0-255). ORB's default is 20; 7 still finds corners in a dim picture.
const int FastThreshold = 7;

// --- Placing each scan: features (11b) and ICP (11) ---
const int FeatureCount = 3000;      // 13: 1000
const float MatchRatio = 0.8f;
const int RansacTries = 5000;       // 13: 1000. More tries: surer to find the best pose among many matches
const float FeatureInlierMeters = 0.05f;
const int MinFeatureMatches = 20;   // 13: 15. With far more features, ask more to agree: fewer wrong placements
const int IcpRounds = 50;           // 13: 30

// A scan needs at least this many features to be matched later. In the dark the colour picture
// is black: 0 features (depth still works, it brings its own infrared light).
const int MinScanFeatures = 100;
const float IcpStartPairMeters = 0.05f;   // 11 started at 0.20, but the features are already within a few cm
const float IcpFinalPairMeters = 0.02f;
const int IcpStep = 2;   // 13: 4. ICP uses every 2nd pixel: 4x the points, a closer fit

// New: ICP leaves a direction alone (the features' guess stands there) when under this share of
// the pairs pins it. 13 and 13b used 2%. The synthetic 30-scan walk showed why that's too low: a
// view of a wall, floor and ceiling pins sliding along the wall through only 2.4-2.7% of its pairs
// (the bent normals where wall meets floor). Each ICP round then slid the scan another 2.3 mm the
// same way: 2 mm off from the features, 119 mm off after 50 rounds, and 18 s instead of 1.5 s.
// At 5% that direction keeps the features' value.
const float WeakShare = 0.05f;

// A placed scan must overlap the earlier ones: at least this share of its points need a partner
// within IcpFinalPairMeters after ICP, or it's refused (most likely placed wrong).
const float MinOverlapShare = 0.25f;

// --- Merging (new) ---
// The TSDF grid's cube size. 3 cm: walls look flat and a room needs a few million cubes (about
// 20 bytes each). Smaller = finer and slower; it grows by itself if the room would need more
// than MaxVoxels.
const float VoxelMeters = 0.02f;     // 13: 0.03
const int MaxVoxels = 80000000;      // 13: 30M. About 24 bytes each: up to 1.9 GB (fine on the PC, too much for a Pi)

// How far in front of and behind a surface a scan updates the grid, in cubes. Behind that, the
// scan can't know what's there (it might be the far side of a wall), so it leaves it alone.
const int TruncationVoxels = 3;

// --- Clean model (09c) ---
const int MaxPlanes = 8;
const float InlierThicknessMeters = 0.03f;
const int SampleStep = 4;
const int NeighbourPixels = 40;
const double MinPlaneShare = 0.03;
const float MaxTiltDegrees = 25;
const float MinWallHeightMeters = 0.8f;
const float MinCeilingHeightMeters = 1.8f;
const float CornerSnapMeters = 0.5f;
const int MinBlockCells = 30;
const float LevelDegrees = 10;       // New: once "up" is known, floor/ceiling/walls must be this close to level/upright
const float MaxBlockMeters = 3.0f;   // New: furniture boxes that touch join only up to this size (else one box swallows the flat)

// New: the same plane seen from two spots is merged into one when it's this close in angle and
// distance. Boxes from different scans that touch or overlap become one box.
const float MergeDegrees = 10;
const float MergeMeters = 0.05f;

// --- Viewer (08) ---
const int ViewWidth = 960, ViewHeight = 720;
const float ViewFocalPixels = 700;
const float PivotMeters = 2.5f;
const int PointSizePixels = 2;
const float TurnStepDegrees = 5;
const float ZoomStepMeters = 0.25f;
const int ShowStep = 2;   // the viewer draws every 2nd pixel of each scan, to stay quick
const float ScanArrowMeters = 0.6f;   // the 4 arrows from each camera circle: its field of view, this far ahead
const int MaxShownPoints = 600000;    // new: the viewer draws at most this many (every k-th point of each scan)

// --- Many scans (new) ---
const int PlaceCandidates = 4;        // the full feature match + RANSAC runs only against the 4 earlier scans...
const int ShortlistHamming = 50;      // ...that win a quick vote: a feature votes for the scan holding its nearest twin, if within this many bits of 256
const float ThinMeters = 0.01f;       // ICP's target: one point per 1 cm cube...
const float CropMarginShare = 0.1f;   // ...of what the rough pose can see, plus 10% round the edges

// --- Live anchors (new) ---
const int LiveRansacTries = 500;      // a quick check, about twice a second
const int LiveEveryMs = 500;

// --- Next shot (new) ---
const float RecommendMinOverlap = 0.5f;      // a suggested view must share at least half of what it sees (to be placed)
const float RecommendDistancePenalty = 0.05f; // per metre from the last spot: prefer nearby spots
const float CoverageCellMeters = 0.05f;      // the map used for this: one point per 5 cm cube
const float SeenFromMeters = 0.5f;           // a direction counts as looked at if a scan within this distance looked that way
const float RecommendMinNew = 0.1f;          // under 10% new: say the spots have seen it all

// --- Suspicious placements (new): any one of these flags a scan red ---
const float SuspiciousMoveMeters = 0.10f;    // ICP moved it this far from the features' guess...
const float SuspiciousDegrees = 5;           // ...or turned it this much
const float SuspiciousRmsMeters = 0.008f;    // the points sit this far apart on average
const float SuspiciousOverlap = 0.4f;        // under 40% of its points found a partner
// The second opinion (see PlaceScan): ICP against the room WITHOUT the scan the features matched,
// starting with pairs up to 30 cm apart (a misplaced earlier scan is often 10-30 cm off). It only
// counts when at least 20% of the points overlap the rest of the room.
const float CheckStartPairMeters = 0.3f;
const float CheckMinOverlap = 0.2f;

// --- Far features (new in 13d) ---
const int FarFeatureCount = 1500;        // ORB features where depth reads nothing, in the 1920 x 1080 colour picture
const float FarAgreeDegrees = 0.5f;      // a far match agrees if its rays miss each other by less than this angle...
const float FarMinMeters = 4.0f;         // ...and meet at least this far away (nearer = it wasn't far after all)
const float FarMaxMeters = 30.0f;        // landmarks further than this are too uncertain to keep
const float MinParallaxDegrees = 2.0f;   // two rays must cross at least this steeply to make a landmark
const int LandmarkMatchHamming = 40;     // the same landmark seen again: descriptors this close (of 256 bits)...
const float LandmarkMergeMeters = 1.0f;  // ...and within this distance: it's refined, not added again
const float BearingMinSpread = 0.001f;   // landmarks must be spread across the view this much to pin a position (about 2 degrees on average)
const float FarIcpStartPairMeters = 0.5f; // ICP's first gate when far rays placed the scan (their guess can be decimetres off)

// --- Keeping the scans (added: K, as in 13c) ---
// Where K saves when you just press Enter at its question. Relative = inside the folder you ran from.
const std::string SaveFolder = "room_scans";

const int Width = 512, Height = 424, ColorWidth = 1920, ColorHeight = 1080;

// ---------------------------------------------------------------------
//  The viewer (copied from 08)
// ---------------------------------------------------------------------

float yawDegrees = 0, pitchDegrees = 0;
float viewDistance = PivotMeters;
bool dragging = false;
int lastMouseX = 0, lastMouseY = 0;
int pressX = 0, pressY = 0;          // new: where the button went down...
bool clicked = false;                // ...and a click = released within 4 px of it (not a drag)
int clickX = 0, clickY = 0;
float viewCosYaw = 1, viewSinYaw = 0, viewCosPitch = 1, viewSinPitch = 0;   // new: once per frame, not per point

void UpdateView()
{
    const float yaw = yawDegrees * (float)CV_PI / 180, pitch = pitchDegrees * (float)CV_PI / 180;
    viewCosYaw = std::cos(yaw);
    viewSinYaw = std::sin(yaw);
    viewCosPitch = std::cos(pitch);
    viewSinPitch = std::sin(pitch);
}

void OnMouse(int event, int x, int y, int flags, void*)
{
    if (event == cv::EVENT_LBUTTONDOWN) { dragging = true; lastMouseX = pressX = x; lastMouseY = pressY = y; }
    if (event == cv::EVENT_LBUTTONUP)
    {
        dragging = false;
        if (std::abs(x - pressX) + std::abs(y - pressY) < 4) { clicked = true; clickX = x; clickY = y; }
    }
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
    float x = point.x, y = point.y, z = point.z - PivotMeters;
    float x1 = viewCosYaw * x + viewSinYaw * z;
    float z1 = -viewSinYaw * x + viewCosYaw * z;
    float y2 = viewCosPitch * y - viewSinPitch * z1;
    float z2 = viewSinPitch * y + viewCosPitch * z1;
    depth = z2 + viewDistance;
    if (depth < 0.05f) return false;
    pixel = cv::Point((int)(ViewFocalPixels * x1 / depth + ViewWidth / 2),
                      (int)(ViewFocalPixels * y2 / depth + ViewHeight / 2));
    return pixel.x >= 0 && pixel.y >= 0 && pixel.x < ViewWidth && pixel.y < ViewHeight;
}

cv::Vec3b ColorForId(int id)   // as in 04
{
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar((id * 47) % 180, 255, 255)), bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr.at<cv::Vec3b>(0, 0);
}

// As in 08, but drawn straight from each scan's own points (13b copied them all into one list every
// frame), and only every stride-th point once there are more than MaxShownPoints. Excluded scans
// are drawn dim grey.
// ponytail: the stride drops detail evenly everywhere; a merged, thinned display cloud would keep it.
cv::Mat Render(const std::vector<std::vector<cv::Point3f>>& points, const std::vector<std::vector<cv::Vec3b>>& colours,
               const std::vector<char>& excluded, bool byScan)
{
    cv::Mat image(ViewHeight, ViewWidth, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::Mat nearest(ViewHeight, ViewWidth, CV_32F, cv::Scalar(1e9));
    size_t total = 0;
    for (const auto& p : points) total += p.size();
    const size_t stride = std::max<size_t>(1, (total + MaxShownPoints - 1) / MaxShownPoints);
    for (size_t s = 0; s < points.size(); s++)
    {
        const cv::Vec3b scanColour = excluded[s] ? cv::Vec3b(70, 70, 70) : ColorForId((int)s + 1);
        for (size_t i = 0; i < points[s].size(); i += stride)
        {
            cv::Point pixel;
            float depth;
            if (!Project(points[s][i], pixel, depth)) continue;
            const cv::Vec3b& colour = excluded[s] || byScan ? scanColour : colours[s][i];
            for (int dy = 0; dy < PointSizePixels; dy++)
            {
                for (int dx = 0; dx < PointSizePixels; dx++)
                {
                    int u = pixel.x + dx, v = pixel.y + dy;
                    if (u >= ViewWidth || v >= ViewHeight || depth >= nearest.at<float>(v, u)) continue;
                    nearest.at<float>(v, u) = depth;
                    image.at<cv::Vec3b>(v, u) = colour;
                }
            }
        }
    }
    return image;
}

// ---------------------------------------------------------------------
//  Poses and ICP (as in 11)
// ---------------------------------------------------------------------

struct Pose
{
    cv::Matx33f turn = cv::Matx33f::eye();
    cv::Point3f shift = cv::Point3f(0, 0, 0);
    cv::Point3f operator()(const cv::Point3f& p) const { return turn * p + shift; }
    Pose Then(const Pose& next) const { return { next.turn * turn, next.turn * shift + next.shift }; }
    float Degrees() const
    {
        float cosine = (turn(0, 0) + turn(1, 1) + turn(2, 2) - 1) / 2;
        return std::acos(std::clamp(cosine, -1.0f, 1.0f)) * 180 / (float)CV_PI;
    }
};

Pose BestFit(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to)
{
    cv::Point3f fromCentre(0, 0, 0), toCentre(0, 0, 0);
    for (size_t i = 0; i < from.size(); i++)
    {
        fromCentre += from[i];
        toCentre += to[i];
    }
    fromCentre *= 1.0f / from.size();
    toCentre *= 1.0f / to.size();
    cv::Matx33d together = cv::Matx33d::zeros();
    for (size_t i = 0; i < from.size(); i++)
    {
        cv::Point3f p = from[i] - fromCentre, q = to[i] - toCentre;
        const double a[3] = { p.x, p.y, p.z }, b[3] = { q.x, q.y, q.z };
        for (int row = 0; row < 3; row++)
            for (int column = 0; column < 3; column++) together(row, column) += a[row] * b[column];
    }
    cv::Mat s, u, vt;
    cv::SVD::compute(cv::Mat(together), s, u, vt);
    cv::Mat turn = vt.t() * u.t();
    if (cv::determinant(turn) < 0)
    {
        vt.row(2) *= -1;
        turn = vt.t() * u.t();
    }
    Pose pose;
    pose.turn = cv::Matx33f(cv::Matx33d(turn));
    pose.shift = toCentre - pose.turn * fromCentre;
    return pose;
}

// A turn of |w| radians about the axis w points along (Rodrigues' formula).
cv::Matx33f TurnAbout(const cv::Vec3d& w)
{
    double angle = cv::norm(w);
    if (angle < 1e-12) return cv::Matx33f::eye();
    cv::Vec3d k = w / angle;
    double c = std::cos(angle), s = std::sin(angle);
    cv::Matx33d turn;
    for (int row = 0; row < 3; row++)
        for (int column = 0; column < 3; column++) turn(row, column) = (row == column ? c : 0) + (1 - c) * k[row] * k[column];
    turn(0, 1) -= s * k[2]; turn(0, 2) += s * k[1];
    turn(1, 0) += s * k[2]; turn(1, 2) -= s * k[0];
    turn(2, 0) -= s * k[1]; turn(2, 1) += s * k[0];
    return cv::Matx33f(turn);
}

// 11's ICP, changed in one way: each pair's gap is measured along the target's surface normal
// ("point to plane"), not straight from point to point. Why: the target points are a sparse
// sample (every 4th pixel: about 3 cm apart at 3 m), so even a perfect fit leaves each point up to
// half a sample from its nearest neighbour, and 11's point-to-point fit slid 3.4 cm along a wall
// in the synthetic test. Along the normal, a point anywhere on the right wall counts as on it.
// The fit is then no longer Kabsch: for small turns it's a 6x6 set of linear equations.
// One more change: a view of only flat walls can't say where along them it is (11's experiment
// 3). ICP then slides on noise: 21 cm in the synthetic test, where the features had been right to
// 1 mm. So the equations are split into directions (eigenvectors), and any direction the shape
// barely pins is left alone: the starting pose (the features' guess) stands there.
// Also says how good the final fit is: the share of "from" points paired, their gap (rms), and
// how many of the 6 directions (3 turns, 3 shifts) the shape couldn't pin.
Pose Icp(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to, const std::vector<cv::Point3f>& toNormals,
         Pose pose, float& pairedShare, float& rmsMeters, int& weakDirections, float startPairMeters = IcpStartPairMeters)
{
    pairedShare = 0;
    rmsMeters = 0;
    weakDirections = 0;
    if (from.size() < 10 || to.size() < 10) return pose;
    cv::Mat toData((int)to.size(), 3, CV_32F, (void*)to.data());
    cv::flann::Index tree(toData, cv::flann::KDTreeIndexParams(4));
    cv::Mat query((int)from.size(), 3, CV_32F), nearestIndex, squaredDistance;
    for (int round = 0; round < IcpRounds; round++)
    {
        for (int i = 0; i < (int)from.size(); i++)
        {
            cv::Point3f p = pose(from[i]);
            query.at<float>(i, 0) = p.x;
            query.at<float>(i, 1) = p.y;
            query.at<float>(i, 2) = p.z;
        }
        tree.knnSearch(query, nearestIndex, squaredDistance, 1, cv::flann::SearchParams(32));
        float maxPair = std::max(IcpFinalPairMeters, startPairMeters * std::pow(0.7f, (float)round));
        float maxNeighbour = std::max(maxPair, 0.06f);   // the nearest sample may be a sample spacing away
        // The pairs: B's point, the target's point and normal, and the gap along that normal.
        std::vector<cv::Point3f> ps, normals;
        std::vector<float> gaps;
        cv::Point3f centre(0, 0, 0);
        for (int i = 0; i < (int)from.size(); i++)
        {
            if (squaredDistance.at<float>(i) > maxNeighbour * maxNeighbour) continue;
            cv::Point3f p(query.at<float>(i, 0), query.at<float>(i, 1), query.at<float>(i, 2));
            int j = nearestIndex.at<int>(i);
            float gap = (p - to[j]).dot(toNormals[j]);   // how far p is in front of the target's surface
            if (std::abs(gap) > maxPair) continue;
            ps.push_back(p);
            normals.push_back(toNormals[j]);
            gaps.push_back(gap);
            centre += p;
        }
        const int pairs = (int)ps.size();
        if (pairs < 10) break;
        centre *= 1.0f / pairs;
        // Each pair adds one equation: gap + turn . ((p - centre) x n) + shift . n = 0, for a small
        // turn (axis * angle) about the pairs' centre, and a shift. Turning about the centre (not
        // the room's origin, metres away) keeps turns and shifts apart. Least squares: A x = b.
        cv::Matx66d A = cv::Matx66d::zeros();
        cv::Matx61d b = cv::Matx61d::zeros();
        double sum = 0;
        for (int i = 0; i < pairs; i++)
        {
            const cv::Point3f& n = normals[i];
            cv::Point3f c = (ps[i] - centre).cross(n);
            const double row[6] = { c.x, c.y, c.z, n.x, n.y, n.z };
            for (int r = 0; r < 6; r++)
            {
                for (int k = 0; k < 6; k++) A(r, k) += row[r] * row[k];
                b(r) -= row[r] * gaps[i];
            }
            sum += gaps[i] * gaps[i];
        }
        // Solve direction by direction. How strongly the pairs pin a direction = its eigenvalue:
        // about (share of pairs whose surface faces that way) x pairs. Under WeakShare: not pinned.
        cv::Mat values, directions;
        cv::eigen(cv::Mat(A), values, directions);
        cv::Matx61d x = cv::Matx61d::zeros();
        int weak = 0;
        for (int k = 0; k < 6; k++)
        {
            double strength = values.at<double>(k);
            if (strength < WeakShare * pairs) { weak++; continue; }
            double along = 0;
            for (int r = 0; r < 6; r++) along += directions.at<double>(k, r) * b(r);
            for (int r = 0; r < 6; r++) x(r) += along / strength * directions.at<double>(k, r);
        }
        if (maxPair == IcpFinalPairMeters)
        {
            pairedShare = (float)pairs / from.size();
            rmsMeters = (float)std::sqrt(sum / pairs);
            weakDirections = weak;
        }
        // The step: turn about the centre, then shift: p -> turn (p - centre) + centre + shift.
        cv::Matx33f turn = TurnAbout(cv::Vec3d(x(0), x(1), x(2)));
        cv::Point3f shift((float)x(3), (float)x(4), (float)x(5));
        Pose step{ turn, centre - turn * centre + shift };
        pose = pose.Then(step);
        if (maxPair == IcpFinalPairMeters && cv::norm(shift) < 1e-4 && step.Degrees() < 0.01f) break;
    }
    return pose;
}

// ---------------------------------------------------------------------
//  A scan, its features, and placing it (11b, now against many scans)
// ---------------------------------------------------------------------

// The depth camera's lens, to find which pixel sees a point (as in 10).
struct Lens { float fx, fy, cx, cy; };

struct Scan
{
    std::vector<cv::Point3f> cloud;   // a 3D point per depth pixel (NaN = none), in this scan's camera coordinates
    cv::Mat picture;                  // the colour picture on the depth grid (black = no colour)
    std::vector<cv::Point3f> sparse;  // every IcpStep-th pixel's point, for ICP...
    std::vector<cv::Point3f> normals; // ...and which way the surface faces there (towards the camera)
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    Pose pose;                        // this scan's camera -> the room (= the first scan's camera)
    std::vector<cv::KeyPoint> farKeypoints;   // new in 13d: features where depth reads nothing (colour pixels)...
    cv::Mat farDescriptors;
    std::vector<cv::Point3f> farRays;         // ...each as a direction from farOrigin, in this camera's coordinates
    cv::Point3f farOrigin = cv::Point3f(0, 0, 0);   // the colour camera, in the depth camera's coordinates
    bool excluded = false;            // new: X takes it out of placing, the advice and E (and puts it back)
    bool suspicious = false;          // new: its placement looked wrong (see the Suspicious settings)
    std::string stats;                // new: how its placement went, shown when it's selected
    cv::Mat fullPicture;              // added (K): the full 1920 x 1080 colour picture...
    std::vector<int> colourIndex;     // ...and which of its pixels each depth pixel landed on (-1 = none)
};

// New: a number for the little cube of space a point is in (21 bits per axis), to keep one point per cube.
long long CubeKey(const cv::Point3f& p, float size)
{
    auto cell = [&](float v) { return (long long)std::floor(v / size) & 0x1FFFFF; };
    return (cell(p.x) << 42) | (cell(p.y) << 21) | cell(p.z);
}

cv::Point3f CloudAt(const Scan& scan, const cv::Point2f& pixel)
{
    int column = std::clamp((int)std::lround(pixel.x), 0, Width - 1), row = std::clamp((int)std::lround(pixel.y), 0, Height - 1);
    return scan.cloud[row * Width + column];
}

// Step 1: a scan from its points and colour picture, with its sparse points, their normals and
// its features (11b).
Scan MakeScan(const std::vector<cv::Point3f>& cloud, const cv::Mat& picture)
{
    Scan scan;
    scan.cloud = cloud;
    scan.picture = picture.clone();
    cv::Mat usable(Height, Width, CV_8U);
    for (int i = 0; i < Width * Height; i++)
        usable.data[i] = !std::isnan(cloud[i].z) && picture.at<cv::Vec3b>(i / Width, i % Width) != cv::Vec3b(0, 0, 0) ? 255 : 0;
    // A normal from the neighbours 2 pixels left/right and up/down: across x down = the facing.
    const int k = 2;
    for (int row = k; row + k < Height; row += IcpStep)
        for (int column = k; column + k < Width; column += IcpStep)
        {
            const cv::Point3f& p = cloud[row * Width + column];
            const cv::Point3f &left = cloud[row * Width + column - k], &right = cloud[row * Width + column + k];
            const cv::Point3f &up = cloud[(row - k) * Width + column], &down = cloud[(row + k) * Width + column];
            if (std::isnan(p.z) || std::isnan(left.z) || std::isnan(right.z) || std::isnan(up.z) || std::isnan(down.z)) continue;
            if (std::abs(right.z - left.z) > 0.05f * p.z || std::abs(down.z - up.z) > 0.05f * p.z) continue;   // an edge
            cv::Point3f n = (right - left).cross(down - up);
            n *= 1.0f / (float)cv::norm(n);
            if (n.dot(p) > 0) n = -n;   // face the camera
            scan.sparse.push_back(p);
            scan.normals.push_back(n);
        }
    cv::erode(usable, usable, cv::Mat(), cv::Point(-1, -1), 3);
    cv::Mat grey;
    cv::cvtColor(picture, grey, cv::COLOR_BGR2GRAY);
    // New: even out the contrast, tile by tile, then look for corners with the gentler FAST test.
    // Only the features see this picture; the colours saved in the mesh stay as they were.
    if (EqualiseContrast) cv::createCLAHE(ClaheClipLimit, cv::Size(ClaheTiles, ClaheTiles))->apply(grey, grey);
    cv::ORB::create(FeatureCount, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, FastThreshold)
        ->detectAndCompute(grey, usable, scan.keypoints, scan.descriptors);
    return scan;
}

// As in 11b: the pose moving scan b's camera coordinates onto scan a's, from matched features.
// Returns how many matches agree with it.
// ---------------------------------------------------------------------
//  Far features (new in 13d): details too far for depth, kept as directions
// ---------------------------------------------------------------------

// How a depth-camera point lands in the colour picture: column = a x/z + e/z + b, row = c y/z + f/z + d.
// The registration knows this, but only hands it out for pixels with depth, so it's measured from
// those, every scan (least squares). e and f come from the colour camera sitting about 5 cm away:
// its centre is at (-e/a, -f/c, 0), and every colour pixel is a ray from there.
struct ColourModel { double a = 1, b = 0, c = 1, d = 0, e = 0, f = 0; bool ok = false; };

ColourModel FitColourModel(const std::vector<cv::Point3f>& cloud, const std::vector<int>& colourIndex)
{
    std::vector<double> across, down, columns, rows;
    for (int i = 0; i < Width * Height; i += 7)
    {
        const cv::Point3f& p = cloud[i];
        if (std::isnan(p.z) || p.z <= 0 || colourIndex[i] < 0) continue;
        across.insert(across.end(), { p.x / p.z, 1 / p.z, 1 });
        down.insert(down.end(), { p.y / p.z, 1 / p.z, 1 });
        columns.push_back(colourIndex[i] % ColorWidth);
        rows.push_back(colourIndex[i] / ColorWidth);
    }
    ColourModel model;
    if (columns.size() < 200) return model;
    cv::Mat x, y;
    cv::solve(cv::Mat((int)columns.size(), 3, CV_64F, across.data()), cv::Mat(columns), x, cv::DECOMP_SVD);
    cv::solve(cv::Mat((int)rows.size(), 3, CV_64F, down.data()), cv::Mat(rows), y, cv::DECOMP_SVD);
    model = { x.at<double>(0), x.at<double>(2), y.at<double>(0), y.at<double>(2), x.at<double>(1), y.at<double>(1), true };
    return model;
}

// Step 1b (new): features where the colour camera sees detail but depth reads nothing (too far, or
// too dark for infrared), from the full colour picture, each as a ray. Only inside the depth
// camera's field of view: past its edges "no depth" just means "not looked at".
void AddFarFeatures(Scan& scan, const cv::Mat& colourGrey, const cv::Mat& noDepth, const ColourModel& model, const Lens& lens)
{
    if (!model.ok || colourGrey.empty() || noDepth.empty()) return;
    cv::Mat grey, mask;
    if (EqualiseContrast) cv::createCLAHE(ClaheClipLimit, cv::Size(ClaheTiles, ClaheTiles))->apply(colourGrey, grey);
    else grey = colourGrey;
    cv::erode(noDepth, mask, cv::Mat(), cv::Point(-1, -1), 8);   // keep clear of where depth starts
    int left = std::max(0, (int)(model.b + model.a * (0 - lens.cx) / lens.fx)), right = std::min(ColorWidth, (int)(model.b + model.a * (Width - lens.cx) / lens.fx));
    int top = std::max(0, (int)(model.d + model.c * (0 - lens.cy) / lens.fy)), bottom = std::min(ColorHeight, (int)(model.d + model.c * (Height - lens.cy) / lens.fy));
    cv::Mat inView = cv::Mat::zeros(mask.size(), CV_8U);
    if (right > left && bottom > top) inView(cv::Rect(left, top, right - left, bottom - top)).setTo(255);
    cv::bitwise_and(mask, inView, mask);
    cv::ORB::create(FarFeatureCount, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, FastThreshold)
        ->detectAndCompute(grey, mask, scan.farKeypoints, scan.farDescriptors);
    scan.farOrigin = cv::Point3f((float)(-model.e / model.a), (float)(-model.f / model.c), 0);
    scan.farRays.clear();
    for (const cv::KeyPoint& k : scan.farKeypoints)
    {
        cv::Point3f d((float)((k.pt.x - model.b) / model.a), (float)((k.pt.y - model.d) / model.c), 1);
        scan.farRays.push_back(d * (1.0f / (float)cv::norm(d)));
    }
}

// Both kinds of feature, for the shortlist vote.
cv::Mat AllDescriptors(const Scan& scan)
{
    cv::Mat all = scan.descriptors.clone();
    if (!scan.farDescriptors.empty()) all.push_back(scan.farDescriptors);
    return all;
}

// A far detail placed in the room by two scans' rays to it (new).
struct Landmark
{
    cv::Point3f point;
    cv::Mat descriptor;
    int seen = 1;
};

// The turn that best maps directions "from" onto "to": Kabsch (BestFit) without the centring.
cv::Matx33f RotationFromRays(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to)
{
    cv::Matx33d together = cv::Matx33d::zeros();
    for (size_t i = 0; i < from.size(); i++)
    {
        const double a[3] = { from[i].x, from[i].y, from[i].z }, b[3] = { to[i].x, to[i].y, to[i].z };
        for (int row = 0; row < 3; row++)
            for (int column = 0; column < 3; column++) together(row, column) += a[row] * b[column];
    }
    cv::Mat s, u, vt;
    cv::SVD::compute(cv::Mat(together), s, u, vt);
    cv::Mat turn = vt.t() * u.t();
    if (cv::determinant(turn) < 0)
    {
        vt.row(2) *= -1;
        turn = vt.t() * u.t();
    }
    return cv::Matx33f(cv::Matx33d(turn));
}

// Where two rays (unit directions) pass closest: how far along each. False if they're parallel.
bool Triangulate(const cv::Point3f& oa, const cv::Point3f& da, const cv::Point3f& ob, const cv::Point3f& db, float& sa, float& sb)
{
    cv::Point3f w = oa - ob;
    float b = da.dot(db), d = da.dot(w), e = db.dot(w), denom = 1 - b * b;
    if (denom < 1e-7f) return false;
    sa = (b * e - d) / denom;
    sb = (e - b * d) / denom;
    return true;
}

// Does a far match agree with a pose h (b's camera -> a's)? The two rays must meet, far away. From
// (almost) the same spot they can't be triangulated, so then only their directions are compared.
bool FarAgrees(const Pose& h, const cv::Point3f& ob, const cv::Point3f& rb, const cv::Point3f& oa, const cv::Point3f& ra)
{
    const float agree = FarAgreeDegrees * (float)CV_PI / 180;
    cv::Point3f o = h(ob), d = h.turn * rb;
    float sa, sb;
    if (cv::norm(o - oa) < 0.2f || !Triangulate(oa, ra, o, d, sa, sb)) return d.dot(ra) > std::cos(agree);
    if (sa < FarMinMeters || sb < FarMinMeters) return false;
    return cv::norm((oa + ra * sa) - (o + d * sb)) < agree * sa;
}

// As in 11b, now with far features (13d). Near matches are 3D to 3D (as before); far matches are
// ray to ray (FarAgrees); landmark matches are a ray to a triangulated point. They all vote for
// RANSAC's poses. The poses come from 3 near matches, or with fewer than 3, from 2 far rays (the
// turn only: the colour cameras assumed at the same spot). Pose = b's camera -> a's.
struct FeatureMatch
{
    int count = 0;                                   // matches that agree: near + far + landmarks
    int near = 0;                                    // ...of which near
    Pose pose;
    std::vector<int> nearKeypoints, farKeypoints;    // b's keypoints that agree (the live dots)
    std::vector<std::pair<int, int>> farPairs;       // agreeing far matches (b's, a's): new landmarks
    std::vector<std::pair<int, int>> landmarkPairs;  // agreeing (b's far feature, landmark)
};

FeatureMatch FeaturePose(const Scan& b, const Scan& a, std::mt19937& random, int tries = RansacTries,
                         const std::vector<Landmark>* landmarks = nullptr, const cv::Mat* landmarkDescriptors = nullptr)
{
    FeatureMatch result;
    auto Matches = [](const cv::Mat& from, const cv::Mat& to)
    {
        std::vector<std::pair<int, int>> pairs;
        if (from.empty() || to.rows < 2) return pairs;
        std::vector<std::vector<cv::DMatch>> candidates;
        cv::BFMatcher(cv::NORM_HAMMING).knnMatch(from, to, candidates, 2);
        for (const std::vector<cv::DMatch>& best : candidates)
            if (best.size() == 2 && best[0].distance <= MatchRatio * best[1].distance) pairs.emplace_back(best[0].queryIdx, best[0].trainIdx);
        return pairs;
    };
    std::vector<cv::Point3f> from, to;
    std::vector<int> fromKeypoint;
    for (auto [i, j] : Matches(b.descriptors, a.descriptors))
    {
        cv::Point3f p = CloudAt(b, b.keypoints[i].pt), q = CloudAt(a, a.keypoints[j].pt);
        if (std::isnan(p.z) || std::isnan(q.z)) continue;
        from.push_back(p);
        to.push_back(q);
        fromKeypoint.push_back(i);
    }
    std::vector<std::pair<int, int>> far = Matches(b.farDescriptors, a.farDescriptors);
    std::vector<std::pair<int, int>> marks;
    std::vector<cv::Point3f> markInA;   // the matched landmarks, in a's camera coordinates
    if (landmarks && landmarkDescriptors && !landmarkDescriptors->empty())
    {
        marks = Matches(b.farDescriptors, *landmarkDescriptors);
        const cv::Matx33f back = a.pose.turn.t();
        for (const auto& m : marks) markInA.push_back(back * ((*landmarks)[m.second].point - a.pose.shift));
    }
    const float agreeCos = std::cos(FarAgreeDegrees * (float)CV_PI / 180);
    auto Count = [&](const Pose& h, std::vector<size_t>* nearIn, std::vector<size_t>* farIn, std::vector<size_t>* markIn)
    {
        int n = 0;
        for (size_t k = 0; k < from.size(); k++)
            if (cv::norm(h(from[k]) - to[k]) < FeatureInlierMeters) { n++; if (nearIn) nearIn->push_back(k); }
        for (size_t k = 0; k < far.size(); k++)
            if (FarAgrees(h, b.farOrigin, b.farRays[far[k].first], a.farOrigin, a.farRays[far[k].second])) { n++; if (farIn) farIn->push_back(k); }
        for (size_t k = 0; k < marks.size(); k++)
        {
            cv::Point3f o = h(b.farOrigin), d = h.turn * b.farRays[marks[k].first], v = markInA[k] - o;
            float length = (float)cv::norm(v);
            if (length > 1 && d.dot(v) > length * agreeCos) { n++; if (markIn) markIn->push_back(k); }
        }
        return n;
    };
    const bool nearMode = from.size() >= 3;
    if (!nearMode && far.size() < 2) return result;
    std::uniform_int_distribution<size_t> pickNear(0, std::max<size_t>(from.size(), 1) - 1), pickFar(0, std::max<size_t>(far.size(), 1) - 1);
    Pose best;
    int bestCount = 0;
    for (int t = 0; t < tries; t++)
    {
        Pose guess;
        if (nearMode)
        {
            size_t i = pickNear(random), j = pickNear(random), k = pickNear(random);
            if (cv::norm((from[j] - from[i]).cross(from[k] - from[i])) < 0.01) continue;
            guess = BestFit({ from[i], from[j], from[k] }, { to[i], to[j], to[k] });
        }
        else
        {
            size_t i = pickFar(random), j = pickFar(random);
            const cv::Point3f &bi = b.farRays[far[i].first], &bj = b.farRays[far[j].first];
            if (cv::norm(bi.cross(bj)) < 0.05) continue;   // two directions too close together pin no turn
            guess.turn = RotationFromRays({ bi, bj }, { a.farRays[far[i].second], a.farRays[far[j].second] });
            guess.shift = a.farOrigin - guess.turn * b.farOrigin;
        }
        int n = Count(guess, nullptr, nullptr, nullptr);
        if (n > bestCount) { bestCount = n; best = guess; }
    }
    if (bestCount < 3) return result;
    std::vector<size_t> nearIn, farIn, markIn;
    Count(best, &nearIn, &farIn, &markIn);
    if (nearIn.size() >= 3)
    {
        std::vector<cv::Point3f> fromAgreeing, toAgreeing;
        for (size_t k : nearIn) { fromAgreeing.push_back(from[k]); toAgreeing.push_back(to[k]); }
        result.pose = BestFit(fromAgreeing, toAgreeing);
    }
    else
    {
        std::vector<cv::Point3f> fromRays, toRays;
        for (size_t k : farIn) { fromRays.push_back(b.farRays[far[k].first]); toRays.push_back(a.farRays[far[k].second]); }
        result.pose = best;
        if (fromRays.size() >= 2)
        {
            result.pose.turn = RotationFromRays(fromRays, toRays);
            result.pose.shift = a.farOrigin - result.pose.turn * b.farOrigin;
        }
    }
    result.count = (int)(nearIn.size() + farIn.size() + markIn.size());
    result.near = (int)nearIn.size();
    for (size_t k : nearIn) result.nearKeypoints.push_back(fromKeypoint[k]);
    for (size_t k : farIn) { result.farKeypoints.push_back(far[k].first); result.farPairs.push_back(far[k]); }
    for (size_t k : markIn) { result.farKeypoints.push_back(marks[k].first); result.landmarkPairs.push_back(marks[k]); }
    return result;
}

// New: which earlier scans share most features with this one, cheaply. Every feature looks up its
// nearest twin among ALL earlier scans' features at once (one brute-force match against them
// stacked) and votes for the scan that twin belongs to. The k scans with most votes come back,
// best first. No ratio test here: overlapping scans hold the same feature twice, so the runner-up
// is often a true twin as well. Excluded scans, and "skip", get no votes.
std::vector<size_t> Shortlist(const cv::Mat& descriptors, const std::vector<Scan>& scans, int k, int skip = -1)
{
    cv::Mat all;
    std::vector<size_t> owner;
    for (size_t s = 0; s < scans.size(); s++)
    {
        if (scans[s].excluded || (int)s == skip) continue;
        cv::Mat both = AllDescriptors(scans[s]);   // 13d: near and far features
        if (both.empty()) continue;
        all.push_back(both);
        owner.insert(owner.end(), both.rows, s);
    }
    std::vector<int> votes(scans.size(), 0);
    if (!all.empty() && !descriptors.empty())
    {
        std::vector<cv::DMatch> matches;
        cv::BFMatcher(cv::NORM_HAMMING).match(descriptors, all, matches);
        for (const cv::DMatch& m : matches)
            if (m.distance <= ShortlistHamming) votes[owner[m.trainIdx]]++;
    }
    std::vector<size_t> order(scans.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return votes[a] > votes[b]; });
    while (!order.empty() && (order.size() > (size_t)k || votes[order.back()] == 0)) order.pop_back();
    return order;
}

// Step 2: features against the shortlisted earlier scans (the best one gives the rough pose), then
// ICP against everything the rough pose can see of the earlier scans. "skip" = a scan to leave out
// (P places a scan again without itself). Sets scan.suspicious and scan.stats (new).
// 13d: also "landmarks" (matched as far anchors), and back out: which scan it matched, and the far
// matches that agreed with it (the main loop turns those into new landmarks).
bool PlaceScan(Scan& scan, const std::vector<Scan>& earlier, const Lens& lens, std::mt19937& random, std::string& report,
               int skip = -1, const std::vector<Landmark>* landmarks = nullptr, size_t* matched = nullptr,
               std::vector<std::pair<int, int>>* farPairs = nullptr)
{
    cv::Mat landmarkDescriptors;
    if (landmarks) for (const Landmark& l : *landmarks) landmarkDescriptors.push_back(l.descriptor);
    FeatureMatch best;
    size_t bestScan = 0;
    for (size_t s : Shortlist(AllDescriptors(scan), earlier, PlaceCandidates, skip))
    {
        FeatureMatch m = FeaturePose(scan, earlier[s], random, RansacTries, landmarks, &landmarkDescriptors);
        if (m.count > best.count) { best = m; bestScan = s; }
    }
    const int bestCount = best.count;
    if (bestCount < MinFeatureMatches)
    {
        report = cv::format("only %d feature matches agree with any earlier scan (need %d; near and far count)", bestCount, MinFeatureMatches);
        return false;
    }
    Pose rough = best.pose.Then(earlier[bestScan].pose);

    // New in 13d: with under 3 near matches, the far rays gave a rough turn, with the camera at the
    // matched scan's spot. Rough, because a far point isn't infinitely far: stepping 0.6 m sideways
    // turns a doorway 12 m away by 2.9 degrees, which rays alone can't tell from a 2.9 degree turn.
    // Rays to landmarks pin both: the camera must sit on every landmark's line of sight (position =
    // where those lines pass closest, a 3 x 3 least squares), and the rays must point at the landmarks
    // from there (turn = RotationFromRays). Each fixes the other: a few rounds. Only when the landmarks
    // are spread across the view (BearingMinSpread); otherwise the lines are nearly parallel.
    int bearings = 0;
    if (best.near < 3 && best.landmarkPairs.size() >= 3)
    {
        Pose solved = rough;
        bool spread = true;
        for (int round = 0; round < 5 && spread; round++)
        {
            cv::Matx33d A = cv::Matx33d::zeros();
            cv::Vec3d B(0, 0, 0);
            const cv::Point3f offset = solved.turn * scan.farOrigin;   // the colour camera, from the depth camera
            for (const auto& [i, l] : best.landmarkPairs)
            {
                cv::Point3f u = solved.turn * scan.farRays[i], L = (*landmarks)[l].point - offset;
                cv::Matx33d across = cv::Matx33d::eye() - cv::Matx33d(u.x * u.x, u.x * u.y, u.x * u.z, u.y * u.x, u.y * u.y, u.y * u.z,
                                                                       u.z * u.x, u.z * u.y, u.z * u.z);
                A += across;
                B += across * cv::Vec3d(L.x, L.y, L.z);
            }
            cv::Mat values;
            cv::eigen(cv::Mat(A), values);
            spread = values.at<double>(2) > BearingMinSpread * best.landmarkPairs.size();
            if (!spread) break;
            cv::Vec3d p = A.inv() * B;
            solved.shift = cv::Point3f((float)p[0], (float)p[1], (float)p[2]);
            std::vector<cv::Point3f> rays, toLandmarks;   // in the camera, and from the camera to the landmark, in the room
            for (const auto& [i, l] : best.landmarkPairs)
            {
                cv::Point3f v = (*landmarks)[l].point - solved(scan.farOrigin);
                rays.push_back(scan.farRays[i]);
                toLandmarks.push_back(v * (1.0f / (float)cv::norm(v)));
            }
            solved.turn = RotationFromRays(rays, toLandmarks);
        }
        if (spread && cv::norm(solved.shift - rough.shift) < 3) { rough = solved; bearings = (int)best.landmarkPairs.size(); }
    }
    // Placed by far rays, the rough pose can be decimetres off: ICP starts with a wider gate then
    // (FarIcpStartPairMeters). In a hallway the walls, floor and ceiling pin the sideways position, the
    // height and every turn; only the distance along it is left to the landmarks.
    const float icpStart = best.near < 3 ? FarIcpStartPairMeters : IcpStartPairMeters;

    // New: ICP's target is still every earlier scan, but only what the rough pose can see of them
    // (13b took all of every scan: about 35k points each), and one point per ThinMeters cube, so
    // ten scans of the same wall give one wall's worth of points. It grows with the room, not the scans.
    std::vector<cv::Point3f> target, targetNormals;
    std::vector<cv::Point3f> others, otherNormals;   // the same, without the scan the features matched (the second opinion, below)
    std::unordered_set<long long> taken, takenByOthers;
    const cv::Matx33f back = rough.turn.t();
    const float marginU = CropMarginShare * Width, marginV = CropMarginShare * Height;
    for (size_t e = 0; e < earlier.size(); e++)
    {
        if (earlier[e].excluded || (int)e == skip) continue;
        for (size_t i = 0; i < earlier[e].sparse.size(); i++)
        {
            cv::Point3f p = earlier[e].pose(earlier[e].sparse[i]);
            cv::Point3f c = back * (p - rough.shift);   // as the new camera would see it
            if (c.z < 0.3f || c.z > MaxDepthMeters + 0.5f) continue;
            float u = lens.fx * c.x / c.z + lens.cx, v = lens.fy * c.y / c.z + lens.cy;
            if (u < -marginU || u > Width + marginU || v < -marginV || v > Height + marginV) continue;
            cv::Point3f n = earlier[e].pose.turn * earlier[e].normals[i];
            if (e != bestScan && takenByOthers.insert(CubeKey(p, ThinMeters)).second)
            {
                others.push_back(p);
                otherNormals.push_back(n);
            }
            if (!taken.insert(CubeKey(p, ThinMeters)).second) continue;
            target.push_back(p);
            targetNormals.push_back(n);
        }
    }
    float pairedShare, rms;
    int weak;
    scan.pose = Icp(scan.sparse, target, targetNormals, rough, pairedShare, rms, weak, icpStart);
    float icpMoved = (float)cv::norm(scan.pose.shift - rough.shift);
    float icpTurned = Pose{ rough.turn.t() * scan.pose.turn, cv::Point3f(0, 0, 0) }.Degrees();
    report = cv::format("matched scan %zu (%d features agree: %d near, %d far, %zu landmarks); ICP: %.0f%% of points overlap, "
                        "%.1f mm apart, moved %.1f cm and %.1f deg from the features' guess",
                        bestScan + 1, bestCount, best.near, (int)best.farPairs.size(), best.landmarkPairs.size(),
                        pairedShare * 100, rms * 1000, icpMoved * 100, icpTurned);
    // New in 13d: placed by far rays, the turn is known; the position only where something pins it.
    bool distanceUnknown = best.near < 3 && bearings == 0 && weak > 0;
    if (best.near < 3)
        report += bearings > 0 ? cv::format("; turned by far rays, positioned by %d far landmarks", bearings)
                               : cv::format("; turned by far rays%s", distanceUnknown ? ", but the position along the free direction is "
                                                                                       "unknown (kept from the matched scan's spot)" : "");
    if (weak > 0) report += cv::format(" (%d direction%s not pinned by the shape - bare walls? - the features' guess kept there)",
                                       weak, weak > 1 ? "s" : "");
    if (pairedShare < MinOverlapShare)
    {
        report += cv::format(": too little overlap (need %.0f%%)", MinOverlapShare * 100);
        return false;
    }
    // New: placed, but does it look right? ICP should only polish the features' guess.
    // (After far rays, ICP is meant to move it a long way: then only its fit counts.)
    bool icpMovedFar = best.near >= 3 && (icpMoved > SuspiciousMoveMeters || icpTurned > SuspiciousDegrees);
    scan.suspicious = icpMovedFar || rms > SuspiciousRmsMeters || pairedShare < SuspiciousOverlap || distanceUnknown;

    // New: a second opinion from the rest of the room. If the scan the features matched was itself
    // misplaced, this scan inherits its error, and ICP agrees (that scan's points are right there).
    // So: ICP again, from the result, against every other scan, with a wider gate (CheckStartPairMeters).
    // If they overlap enough and pull it far away, one of the two is wrong.
    std::vector<cv::Point3f> few;
    for (size_t i = 0; i < scan.sparse.size(); i += 4) few.push_back(scan.sparse[i]);   // a quick check: every 4th point
    float checkShare = 0, checkRms;
    int checkWeak;
    Pose check = others.empty() ? scan.pose : Icp(few, others, otherNormals, scan.pose, checkShare, checkRms, checkWeak, CheckStartPairMeters);
    float checkMoved = (float)cv::norm(check.shift - scan.pose.shift);
    float checkTurned = Pose{ scan.pose.turn.t() * check.turn, cv::Point3f(0, 0, 0) }.Degrees();
    bool roomDisagrees = checkShare >= CheckMinOverlap && (checkMoved > SuspiciousMoveMeters || checkTurned > SuspiciousDegrees);
    if (roomDisagrees)
    {
        scan.suspicious = true;
        report += cv::format("; the rest of the room disagrees by %.1f cm / %.1f deg: scan %zu (its feature match) may itself be misplaced",
                             checkMoved * 100, checkTurned, bestScan + 1);
    }
    scan.stats = cv::format("%d features agree with scan %zu; ICP %.0f%% overlap, %.1f mm apart, moved %.1f cm / %.1f deg%s%s",
                            bestCount, bestScan + 1, pairedShare * 100, rms * 1000, icpMoved * 100, icpTurned,
                            roomDisagrees ? cv::format("; the rest of the room disagrees by %.1f cm", checkMoved * 100).c_str() : "",
                            scan.suspicious ? "  SUSPICIOUS" : "");
    if (scan.suspicious) report += "; SUSPICIOUS: check it with C (colour by scan); click it and press X to exclude it";
    if (matched) *matched = bestScan;
    if (farPairs) *farPairs = best.farPairs;
    return true;
}

// New in 13d: after a scan is placed, its far matches with the scan it matched become landmarks,
// where the two rays cross far away (FarMinMeters to FarMaxMeters) and at a clear angle
// (MinParallaxDegrees: from the same spot, rays don't say how far). The same detail seen again
// (a close descriptor, within LandmarkMergeMeters) refines its landmark instead. Returns how many are new.
// ponytail: landmarks stay when U or X removes one of their two scans; rebuild them if that matters.
int AddLandmarks(const Scan& b, const Scan& a, const std::vector<std::pair<int, int>>& pairs, std::vector<Landmark>& landmarks)
{
    int added = 0;
    const float minCos = std::cos(MinParallaxDegrees * (float)CV_PI / 180);
    for (const auto& [i, j] : pairs)
    {
        cv::Point3f ob = b.pose(b.farOrigin), db = b.pose.turn * b.farRays[i];
        cv::Point3f oa = a.pose(a.farOrigin), da = a.pose.turn * a.farRays[j];
        float sa, sb;
        if (da.dot(db) > minCos || !Triangulate(oa, da, ob, db, sa, sb)) continue;
        if (sa < FarMinMeters || sb < FarMinMeters || sa > FarMaxMeters || sb > FarMaxMeters) continue;
        cv::Point3f point = ((oa + da * sa) + (ob + db * sb)) * 0.5f;
        cv::Mat descriptor = b.farDescriptors.row(i);
        bool merged = false;
        for (Landmark& l : landmarks)
        {
            if (cv::norm(l.point - point) > LandmarkMergeMeters || cv::norm(l.descriptor, descriptor, cv::NORM_HAMMING) > LandmarkMatchHamming) continue;
            l.point = (l.point * (float)l.seen + point) * (1.0f / (l.seen + 1));
            l.seen++;
            merged = true;
            break;
        }
        if (!merged) { landmarks.push_back({ point, descriptor.clone(), 1 }); added++; }
    }
    return added;
}

// ---------------------------------------------------------------------
//  Live anchors and the next shot (new)
// ---------------------------------------------------------------------

// The live check: this frame's features against the 2 best-voted scans. The ones that agree with
// the best pose are "anchors": what the next scan would be placed by. Pixels are in the colour picture.
struct LiveCheck
{
    int anchors = 0, farAnchors = 0;   // 13d: farAnchors = the far ones (and landmarks) among them
    int scan = -1;
    std::vector<cv::Point2f> anchorPixels, farAnchorPixels, otherPixels;
};

LiveCheck CheckLive(std::vector<cv::Point3f> cloud, cv::Mat picture, std::vector<int> colourIndex, cv::Mat colourGrey,
                    cv::Mat noDepth, ColourModel model, Lens lens, const std::vector<Scan>& scans, const std::vector<Landmark>& landmarks)
{
    LiveCheck result;
    Scan live = MakeScan(cloud, picture);
    AddFarFeatures(live, colourGrey, noDepth, model, lens);
    cv::Mat landmarkDescriptors;
    for (const Landmark& l : landmarks) landmarkDescriptors.push_back(l.descriptor);
    std::mt19937 random(7);
    FeatureMatch best;
    for (size_t s : Shortlist(AllDescriptors(live), scans, 2))
    {
        FeatureMatch m = FeaturePose(live, scans[s], random, LiveRansacTries, &landmarks, &landmarkDescriptors);
        if (m.count > best.count) { best = m; result.scan = (int)s; }
    }
    result.anchors = best.count;
    result.farAnchors = best.count - best.near;
    std::vector<char> isAnchor(live.keypoints.size(), 0), isFarAnchor(live.farKeypoints.size(), 0);
    for (int k : best.nearKeypoints) isAnchor[k] = 1;
    for (int k : best.farKeypoints) isFarAnchor[k] = 1;
    for (size_t k = 0; k < live.keypoints.size(); k++)
    {
        const cv::Point2f& pt = live.keypoints[k].pt;
        int column = std::clamp((int)std::lround(pt.x), 0, Width - 1), row = std::clamp((int)std::lround(pt.y), 0, Height - 1);
        int c = colourIndex[row * Width + column];
        if (c < 0) continue;
        (isAnchor[k] ? result.anchorPixels : result.otherPixels).emplace_back((float)(c % ColorWidth), (float)(c / ColorWidth));
    }
    for (size_t k = 0; k < live.farKeypoints.size(); k++)   // far features are colour pixels already
        (isFarAnchor[k] ? result.farAnchorPixels : result.otherPixels).push_back(live.farKeypoints[k].pt);
    return result;
}

// The next shot: from which existing spot to turn which way to see most that's new, while still
// sharing at least RecommendMinOverlap of the view (or it couldn't be placed).
struct Recommendation
{
    bool found = false;
    Pose pose;
    std::string text = "Next: (after the first scan)";
};

Recommendation RecommendNext(const std::vector<Scan>& scans, const Lens& lens)
{
    Recommendation best;
    std::vector<size_t> spots;
    for (size_t s = 0; s < scans.size(); s++) if (!scans[s].excluded) spots.push_back(s);
    if (spots.empty()) return best;

    // What's been seen: every included scan's points, one per CoverageCellMeters cube.
    std::vector<cv::Point3f> map;
    std::unordered_set<long long> taken;
    for (size_t s : spots)
        for (const cv::Point3f& p : scans[s].sparse)
        {
            cv::Point3f q = scans[s].pose(p);
            if (taken.insert(CubeKey(q, CoverageCellMeters)).second) map.push_back(q);
        }
    // Up: the scans' camera "up" (-Y), averaged, so the suggested turns are about the room's vertical.
    cv::Point3f up(0, 0, 0);
    for (size_t s : spots) up += scans[s].pose.turn * cv::Point3f(0, -1, 0);
    up *= 1.0f / (float)cv::norm(up);

    const int columns = 16, rows = 12;
    const cv::Point3f last = scans[spots.back()].pose.shift;
    float bestScore = -1e9f, bestNew = 0;
    for (size_t s : spots)
    {
        const Pose& spot = scans[s].pose;
        for (float yaw : { -90.0f, -60.0f, -30.0f, 30.0f, 60.0f, 90.0f, 180.0f })
        {
            for (float pitch : { -20.0f, 0.0f, 20.0f })
            {
                Pose candidate;
                cv::Vec3d axis(up.x, up.y, up.z);
                candidate.turn = TurnAbout(axis * (yaw * CV_PI / 180)) * spot.turn * TurnAbout(cv::Vec3d(pitch * CV_PI / 180, 0, 0));
                candidate.shift = spot.shift;
                // A cell of the candidate's view is known if a seen point lands in it...
                std::vector<char> known(columns * rows, 0);
                const cv::Matx33f back = candidate.turn.t();
                for (const cv::Point3f& p : map)
                {
                    cv::Point3f c = back * (p - candidate.shift);
                    if (c.z < 0.5f || c.z > 4.5f) continue;
                    float u = lens.fx * c.x / c.z + lens.cx, v = lens.fy * c.y / c.z + lens.cy;
                    if (u < 0 || v < 0 || u >= Width || v >= Height) continue;
                    known[(int)(v * rows / Height) * columns + (int)(u * columns / Width)] = 1;
                }
                // ...or if a scan from (nearly) this spot already looked that way: a window or an
                // open door reads nothing, and would otherwise be recommended for ever.
                for (int cell = 0; cell < columns * rows; cell++)
                {
                    if (known[cell]) continue;
                    float u = (cell % columns + 0.5f) * Width / columns, v = (cell / columns + 0.5f) * Height / rows;
                    cv::Point3f ray = candidate.turn * cv::Point3f((u - lens.cx) / lens.fx, (v - lens.cy) / lens.fy, 1);
                    for (size_t e : spots)
                    {
                        if (cv::norm(scans[e].pose.shift - candidate.shift) > SeenFromMeters) continue;
                        cv::Point3f d = scans[e].pose.turn.t() * ray;
                        if (d.z <= 0) continue;
                        float ue = lens.fx * d.x / d.z + lens.cx, ve = lens.fy * d.y / d.z + lens.cy;
                        if (ue >= 0 && ve >= 0 && ue < Width && ve < Height) { known[cell] = 1; break; }
                    }
                }
                float knownShare = (float)std::count(known.begin(), known.end(), 1) / (columns * rows);
                if (knownShare < RecommendMinOverlap) continue;
                float score = (1 - knownShare) - RecommendDistancePenalty * (float)cv::norm(spot.shift - last);
                if (score <= bestScore) continue;
                bestScore = score;
                bestNew = 1 - knownShare;
                best.pose = candidate;
                bool right = (candidate.turn * cv::Point3f(0, 0, 1)).dot(spot.turn * cv::Point3f(1, 0, 0)) > 0;
                std::string turnText = yaw == 180 ? "turn round" : cv::format("turn %.0f deg %s", std::abs(yaw), right ? "right" : "left");
                std::string tiltText = pitch == 0 ? "" : cv::format(", tilt %.0f deg %s", std::abs(pitch), pitch > 0 ? "up" : "down");
                best.text = cv::format("Next: at spot %zu, %s%s (about %.0f%% new)", s + 1, turnText.c_str(), tiltText.c_str(),
                                       bestNew * 100);
            }
        }
    }
    best.found = bestScore > -1e9f && bestNew >= RecommendMinNew;
    if (!best.found)
        best.text = "Next: these spots have seen it all - move somewhere new, looking back at scanned parts, or press E";
    // ponytail: only existing spots are suggested (a new position could be inside furniture);
    // free-space checks would allow suggesting new places.
    return best;
}

// ---------------------------------------------------------------------
//  Merging: a TSDF grid and a surface net (new)
// ---------------------------------------------------------------------

// A triangle mesh with named groups (09c).
struct Mesh
{
    std::vector<cv::Vec3f> points;
    std::vector<cv::Vec3b> colours;
    std::map<std::string, std::vector<cv::Vec3i>> groups;
};


// Step 4a: every scan into one grid of little cubes ("voxels"), then one mesh where the surface is.
// Each voxel keeps a signed distance to the nearest surface along the camera's view: + in front
// (empty air), - just behind, cut off ("truncated") at TruncationVoxels cubes, and averaged over
// every scan that saw it. Noise averages out; two scans of the same wall become one wall.
Mesh FuseScans(const std::vector<Scan>& scans, const Lens& lens, float& voxel)
{
    // The box to fill: around the scans' points, 1-99% so stray points don't blow it up.
    std::vector<float> xs, ys, zs;
    for (const Scan& scan : scans)
        for (const cv::Point3f& p : scan.sparse)
        {
            cv::Point3f q = scan.pose(p);
            xs.push_back(q.x); ys.push_back(q.y); zs.push_back(q.z);
        }
    auto percentile = [](std::vector<float>& v, double share)
    {
        size_t i = std::min(v.size() - 1, (size_t)(share * v.size()));
        std::nth_element(v.begin(), v.begin() + i, v.end());
        return v[i];
    };
    const float margin = 0.3f;
    cv::Point3f lo(percentile(xs, 0.01) - margin, percentile(ys, 0.01) - margin, percentile(zs, 0.01) - margin);
    cv::Point3f hi(percentile(xs, 0.99) + margin, percentile(ys, 0.99) + margin, percentile(zs, 0.99) + margin);
    voxel = VoxelMeters;
    int nx, ny, nz;
    while (true)
    {
        nx = (int)std::ceil((hi.x - lo.x) / voxel);
        ny = (int)std::ceil((hi.y - lo.y) / voxel);
        nz = (int)std::ceil((hi.z - lo.z) / voxel);
        if ((double)nx * ny * nz <= MaxVoxels) break;
        voxel *= 1.25f;   // too big a room for this cube size: coarser cubes
    }
    const size_t count = (size_t)nx * ny * nz;
    std::vector<float> distance(count, 0), weight(count, 0), colourWeight(count, 0);
    std::vector<cv::Vec3f> colourSum(count, cv::Vec3f(0, 0, 0));
    const float truncation = TruncationVoxels * voxel;
    auto index = [&](int x, int y, int z) { return ((size_t)z * ny + y) * nx + x; };

    for (const Scan& scan : scans)
    {
        cv::Matx33f back = scan.pose.turn.t();   // room -> this scan's camera
        cv::parallel_for_(cv::Range(0, nz), [&](const cv::Range& slices)
        {
            for (int z = slices.start; z < slices.end; z++)
                for (int y = 0; y < ny; y++)
                    for (int x = 0; x < nx; x++)
                    {
                        cv::Point3f centre(lo.x + (x + 0.5f) * voxel, lo.y + (y + 0.5f) * voxel, lo.z + (z + 0.5f) * voxel);
                        cv::Point3f p = back * (centre - scan.pose.shift);
                        if (p.z < 0.3f) continue;   // behind or right at the camera
                        int column = (int)std::lround(p.x * lens.fx / p.z + lens.cx - 0.5f);
                        int row = (int)std::lround(p.y * lens.fy / p.z + lens.cy - 0.5f);
                        if (column < 0 || row < 0 || column >= Width || row >= Height) continue;
                        float seen = scan.cloud[row * Width + column].z;
                        if (std::isnan(seen)) continue;
                        float signedDistance = seen - p.z;          // + = in front of what the camera saw
                        if (signedDistance < -truncation) continue;   // hidden behind it: unknown
                        size_t i = index(x, y, z);
                        float value = std::min(1.0f, signedDistance / truncation);
                        distance[i] = (distance[i] * weight[i] + value) / (weight[i] + 1);
                        weight[i] += 1;
                        if (signedDistance < truncation)
                        {
                            cv::Vec3b c = scan.picture.at<cv::Vec3b>(row, column);
                            if (c != cv::Vec3b(0, 0, 0)) { colourSum[i] += cv::Vec3f(c[0], c[1], c[2]); colourWeight[i] += 1; }
                        }
                    }
        });
    }

    // Step 4b: a surface net. Each cube of 8 neighbouring voxels where the sign changes gets one
    // point, at the average of where the surface crosses its edges. Then every voxel edge that
    // crosses the surface joins the 4 cubes around it into a square (two triangles).
    Mesh mesh;
    std::vector<int> pointOf(count, -1);
    auto corner = [](int c) { return cv::Vec3f((float)(c & 1), (float)((c >> 1) & 1), (float)((c >> 2) & 1)); };
    for (int z = 0; z + 1 < nz; z++)
        for (int y = 0; y + 1 < ny; y++)
            for (int x = 0; x + 1 < nx; x++)
            {
                float value[8];
                int inside = 0;
                bool seenAll = true;
                cv::Vec3f colour(0, 0, 0);
                float colourCount = 0;
                for (int c = 0; c < 8 && seenAll; c++)
                {
                    size_t i = index(x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1));
                    seenAll = weight[i] > 0;
                    value[c] = distance[i];
                    if (value[c] < 0) inside++;
                    colour += colourSum[i];
                    colourCount += colourWeight[i];
                }
                if (!seenAll || inside == 0 || inside == 8) continue;   // never seen, or no surface here
                cv::Vec3f sum(0, 0, 0);
                int crossings = 0;
                for (int bit : { 1, 2, 4 })
                    for (int c = 0; c < 8; c++)
                    {
                        if (c & bit) continue;
                        float a = value[c], b = value[c | bit];
                        if ((a < 0) == (b < 0)) continue;
                        sum += corner(c) + (corner(c | bit) - corner(c)) * (a / (a - b));   // where it crosses zero
                        crossings++;
                    }
                cv::Vec3f inCube = sum / (float)crossings;
                pointOf[index(x, y, z)] = (int)mesh.points.size();
                mesh.points.emplace_back(lo.x + (x + 0.5f + inCube[0]) * voxel, lo.y + (y + 0.5f + inCube[1]) * voxel,
                                         lo.z + (z + 0.5f + inCube[2]) * voxel);
                mesh.colours.push_back(colourCount > 0 ? cv::Vec3b(colour / colourCount) : cv::Vec3b(128, 128, 128));
            }
    std::vector<cv::Vec3i>& faces = mesh.groups["seen_room"];
    const int size[3] = { nx, ny, nz };
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
            {
                size_t i = index(x, y, z);
                if (weight[i] == 0) continue;
                for (int axis = 0; axis < 3; axis++)
                {
                    int at[3] = { x, y, z };
                    if (at[axis] + 1 >= size[axis]) continue;
                    at[axis]++;
                    size_t j = index(at[0], at[1], at[2]);
                    if (weight[j] == 0 || (distance[i] < 0) == (distance[j] < 0)) continue;
                    // The 4 cubes around this edge: one step back along the other two axes, or not.
                    int u = (axis + 1) % 3, w = (axis + 2) % 3, quad[4];
                    const int du[4] = { -1, 0, 0, -1 }, dw[4] = { -1, -1, 0, 0 };
                    bool complete = true;
                    for (int k = 0; k < 4 && complete; k++)
                    {
                        int c[3] = { x, y, z };
                        c[u] += du[k];
                        c[w] += dw[k];
                        complete = c[u] >= 0 && c[w] >= 0 && c[u] + 1 < size[u] && c[w] + 1 < size[w];
                        if (complete) { quad[k] = pointOf[index(c[0], c[1], c[2])]; complete = quad[k] >= 0; }
                    }
                    if (!complete) continue;
                    // Face the empty side (+): towards voxel i if it's the + one.
                    cv::Vec3f facing(0, 0, 0);
                    facing[axis] = distance[i] > 0 ? -1.0f : 1.0f;
                    cv::Vec3f normal = (mesh.points[quad[1]] - mesh.points[quad[0]]).cross(mesh.points[quad[2]] - mesh.points[quad[0]]);
                    if (normal.dot(facing) < 0) std::swap(quad[1], quad[3]);
                    faces.emplace_back(quad[0], quad[1], quad[2]);
                    faces.emplace_back(quad[0], quad[2], quad[3]);
                }
            }
    return mesh;
}

// ---------------------------------------------------------------------
//  The clean model, from all scans (09c, plus merging planes and boxes)
// ---------------------------------------------------------------------

struct Plane
{
    cv::Vec3f normal;
    float distance;
    float DistanceTo(const cv::Vec3f& point) const { return normal.dot(point) + distance; }
};

struct Piece { Plane plane; std::vector<int> cells; };   // one scan's plane and its grid cells (09c)

bool PlaneThrough(const cv::Vec3f& a, const cv::Vec3f& b, const cv::Vec3f& c, Plane& plane)
{
    cv::Vec3f normal = (b - a).cross(c - a);
    float length = (float)cv::norm(normal);
    if (length < 1e-6f) return false;
    plane.normal = normal / length;
    plane.distance = -plane.normal.dot(a);
    if (plane.distance < 0) { plane.normal = -plane.normal; plane.distance = -plane.distance; }
    return true;
}

Plane FitPlane(const std::vector<cv::Vec3f>& points)
{
    cv::Mat data((int)points.size(), 3, CV_32F, (void*)points.data());
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Vec3f mean(pca.mean.at<float>(0), pca.mean.at<float>(1), pca.mean.at<float>(2));
    Plane plane{ cv::Vec3f(pca.eigenvectors.row(2)), 0 };
    plane.distance = -plane.normal.dot(mean);
    if (plane.distance < 0) { plane.normal = -plane.normal; plane.distance = -plane.distance; }
    return plane;
}

float Percentile(std::vector<float> values, double share)
{
    size_t i = std::min(values.size() - 1, (size_t)(share * values.size()));
    std::nth_element(values.begin(), values.begin() + i, values.end());
    return values[i];
}

// As in 09c.
std::vector<Piece> FindPlanes(const std::vector<cv::Vec3f>& grid, int gridColumns, std::mt19937& random)
{
    const int gridRows = (int)grid.size() / gridColumns, reach = NeighbourPixels / SampleStep;
    std::uniform_int_distribution<int> pickOffset(-reach, reach);
    std::vector<bool> free(grid.size());
    int readings = 0;
    for (size_t i = 0; i < grid.size(); i++)
    {
        free[i] = !std::isnan(grid[i][2]);
        if (free[i]) readings++;
    }
    std::vector<Piece> pieces;
    std::vector<int> freeCells;
    for (int search = 0; search < MaxPlanes; search++)
    {
        freeCells.clear();
        for (int i = 0; i < (int)grid.size(); i++) if (free[i]) freeCells.push_back(i);
        if (freeCells.size() < 3) break;
        std::uniform_int_distribution<size_t> pickCell(0, freeCells.size() - 1);
        Plane best{};
        int bestCount = 0;
        for (int t = 0; t < 200; t++)
        {
            int a = freeCells[pickCell(random)];
            int row = a / gridColumns, column = a % gridColumns;
            int bRow = row + pickOffset(random), bColumn = column + pickOffset(random);
            int cRow = row + pickOffset(random), cColumn = column + pickOffset(random);
            if (std::min({ bRow, bColumn, cRow, cColumn }) < 0 || std::max(bRow, cRow) >= gridRows ||
                std::max(bColumn, cColumn) >= gridColumns) continue;
            int b = bRow * gridColumns + bColumn, c = cRow * gridColumns + cColumn;
            if (!free[b] || !free[c]) continue;
            Plane plane;
            if (!PlaneThrough(grid[a], grid[b], grid[c], plane)) continue;
            int count = 0;
            for (int i : freeCells)
                if (std::abs(plane.DistanceTo(grid[i])) < InlierThicknessMeters) count++;
            if (count > bestCount) { bestCount = count; best = plane; }
        }
        if (bestCount < MinPlaneShare * readings) break;
        cv::Mat onPlane(gridRows, gridColumns, CV_8U, cv::Scalar(0)), labels, stats, centroids;
        for (int i : freeCells)
            if (std::abs(best.DistanceTo(grid[i])) < InlierThicknessMeters) onPlane.data[i] = 255;
        int count = cv::connectedComponentsWithStats(onPlane, labels, stats, centroids);
        int biggest = 1;
        for (int piece = 2; piece < count; piece++)
            if (stats.at<int>(piece, cv::CC_STAT_AREA) > stats.at<int>(biggest, cv::CC_STAT_AREA)) biggest = piece;
        if (count < 2 || stats.at<int>(biggest, cv::CC_STAT_AREA) < MinPlaneShare * readings) continue;
        Piece piece;
        std::vector<cv::Vec3f> inliers;
        for (int i : freeCells)
        {
            if (labels.ptr<int>()[i] != biggest) continue;
            piece.cells.push_back(i);
            inliers.push_back(grid[i]);
            free[i] = false;
        }
        piece.plane = FitPlane(inliers);
        pieces.push_back(piece);
    }
    return pieces;
}

void AddQuad(Mesh& mesh, const std::string& group, const cv::Vec3f& a, const cv::Vec3f& b, const cv::Vec3f& c,
             const cv::Vec3f& d, const cv::Vec3b& colour, const cv::Vec3f& facing)
{
    int first = (int)mesh.points.size();
    for (const cv::Vec3f& p : { a, b, c, d })
    {
        mesh.points.push_back(p);
        mesh.colours.push_back(colour);
    }
    bool flip = (b - a).cross(c - a).dot(facing) < 0;
    std::vector<cv::Vec3i>& faces = mesh.groups[group];
    faces.push_back(flip ? cv::Vec3i(first, first + 2, first + 1) : cv::Vec3i(first, first + 1, first + 2));
    faces.push_back(flip ? cv::Vec3i(first, first + 3, first + 2) : cv::Vec3i(first, first + 2, first + 3));
}

void AddBox(Mesh& mesh, const std::string& group, const cv::Vec3f& lo, const cv::Vec3f& hi, const cv::Vec3b& colour)
{
    cv::Vec3f corner[8];
    for (int i = 0; i < 8; i++) corner[i] = cv::Vec3f(i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]);
    const int sides[6][4] = { { 0, 2, 6, 4 }, { 1, 3, 7, 5 }, { 0, 1, 5, 4 }, { 2, 3, 7, 6 }, { 0, 1, 3, 2 }, { 4, 5, 7, 6 } };
    cv::Vec3f centre = (lo + hi) / 2;
    for (const auto& s : sides)
    {
        cv::Vec3f middle = (corner[s[0]] + corner[s[1]] + corner[s[2]] + corner[s[3]]) / 4;
        AddQuad(mesh, group, corner[s[0]], corner[s[1]], corner[s[2]], corner[s[3]], colour, middle - centre);
    }
}

struct Room
{
    Mesh mesh;                 // room coordinates: X right, Y up, Z towards the first camera; first camera at 0,0,0
    std::string upFrom;
    float floorY = 0, ceilingY = 0;
    bool floorSeen = false, ceilingSeen = false;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> walls;
    std::vector<std::pair<cv::Vec3f, cv::Vec3f>> blocks;
    cv::Matx33f firstCamera = cv::Matx33f::eye();   // scan 1's camera (X right, Y down, Z ahead) -> room
};

// Step 4c: 09c's clean model from every scan. Each scan's planes are found as in 09c, moved into
// the room, and merged with the same plane seen from other spots; then sorted, levelled, and
// built as in 09c. The merged mesh from step 4b is levelled with it.
Room BuildRoom(const std::vector<Scan>& scans, const Mesh& seen, std::mt19937& random)
{
    Room room;
    room.mesh = seen;
    const int gridColumns = Width / SampleStep, gridRows = Height / SampleStep, cells = gridColumns * gridRows;
    struct RoomPiece { Plane plane; std::vector<cv::Vec3f> points; cv::Vec3d colourSum; std::vector<std::pair<int, int>> cells; };
    std::vector<RoomPiece> pieces;
    std::vector<std::vector<cv::Vec3f>> roomGrid(scans.size());   // each scan's sampled points, in the room (first camera)
    std::vector<std::vector<cv::Vec3b>> gridColour(scans.size());
    std::vector<std::vector<float>> gridDepth(scans.size());        // ...and their distance from their own camera

    for (size_t s = 0; s < scans.size(); s++)
    {
        std::vector<cv::Vec3f> grid(cells);
        roomGrid[s].resize(cells);
        gridColour[s].resize(cells);
        gridDepth[s].resize(cells);
        for (int cell = 0; cell < cells; cell++)
        {
            int pixel = (cell / gridColumns) * SampleStep * Width + (cell % gridColumns) * SampleStep;
            cv::Point3f p = scans[s].cloud[pixel];
            grid[cell] = cv::Vec3f(p.x, p.y, p.z);
            roomGrid[s][cell] = std::isnan(p.z) ? cv::Vec3f(NAN, NAN, NAN) : cv::Vec3f(scans[s].pose(p));
            gridColour[s][cell] = scans[s].picture.at<cv::Vec3b>(pixel / Width, pixel % Width);
            gridDepth[s][cell] = p.z;
        }
        for (const Piece& found : FindPlanes(grid, gridColumns, random))
        {
            RoomPiece piece{ {}, {}, cv::Vec3d(0, 0, 0), {} };
            cv::Vec3f centre(0, 0, 0);
            for (int i : found.cells)
            {
                piece.points.push_back(roomGrid[s][i]);
                piece.colourSum += cv::Vec3d(gridColour[s][i]);
                piece.cells.emplace_back((int)s, i);
                centre += roomGrid[s][i];
            }
            centre /= (float)found.cells.size();
            piece.plane = FitPlane(piece.points);
            // New: the same plane seen from an earlier spot? Then it's one piece.
            bool merged = false;
            for (RoomPiece& other : pieces)
            {
                if (other.plane.normal.dot(piece.plane.normal) < std::cos(MergeDegrees * (float)CV_PI / 180) ||
                    std::abs(other.plane.DistanceTo(centre)) > MergeMeters) continue;
                other.points.insert(other.points.end(), piece.points.begin(), piece.points.end());
                other.cells.insert(other.cells.end(), piece.cells.begin(), piece.cells.end());
                other.colourSum += piece.colourSum;
                other.plane = FitPlane(other.points);
                merged = true;
                break;
            }
            if (!merged) pieces.push_back(piece);
        }
    }
    auto averageColour = [](const cv::Vec3d& sum, size_t count) { return cv::Vec3b(sum / (double)std::max<size_t>(1, count)); };

    // --- Sort the planes (09c step 5): first by the first camera's up, then again by the room's ---
    const cv::Vec3f cameraUp(0, -1, 0);
    int floorPiece = -1, ceilingPiece = -1;
    std::vector<int> wallPieces;
    auto Sort = [&](const cv::Vec3f& by, float tiltDegrees)
    {
        const float levelCos = std::cos(tiltDegrees * (float)CV_PI / 180), uprightSin = std::sin(tiltDegrees * (float)CV_PI / 180);
        std::vector<float> heights;
        for (const auto& grid : roomGrid)
            for (const cv::Vec3f& p : grid) if (!std::isnan(p[2])) heights.push_back(p.dot(by));
        float lowest = Percentile(heights, 0.02);
        floorPiece = ceilingPiece = -1;
        wallPieces.clear();
        for (int k = 0; k < (int)pieces.size(); k++)
        {
            const Plane& plane = pieces[k].plane;
            float upness = plane.normal.dot(by);
            if (upness > levelCos)
            {
                if (floorPiece < 0 || plane.distance > pieces[floorPiece].plane.distance) floorPiece = k;
            }
            else if (-upness > levelCos)
            {
                bool highEnough = plane.distance - lowest >= MinCeilingHeightMeters;
                if (highEnough && (ceilingPiece < 0 || plane.distance > pieces[ceilingPiece].plane.distance)) ceilingPiece = k;
            }
            else if (std::abs(upness) < uprightSin)
            {
                std::vector<float> wallHeights;
                for (const cv::Vec3f& p : pieces[k].points) wallHeights.push_back(p.dot(by));
                if (Percentile(wallHeights, 0.98) - Percentile(wallHeights, 0.02) >= MinWallHeightMeters) wallPieces.push_back(k);
            }
        }
    };
    bool anyPoints = false;
    for (const auto& grid : roomGrid) for (const cv::Vec3f& p : grid) anyPoints = anyPoints || !std::isnan(p[2]);
    if (!anyPoints) return room;

    // --- Up and ahead (09c step 6) ---
    // New: up comes from the BIGGEST level plane, floor or ceiling, not the farthest floor-ish one.
    // The farthest could be a ramp or a couch back (a real 34-scan export took a slanted piece as
    // the floor and came out tilted 28 degrees, with its ceiling ignored).
    cv::Vec3f up = cameraUp;
    room.upFrom = "the first camera (nothing level found, under 2 walls)";
    const float tiltCos = std::cos(MaxTiltDegrees * (float)CV_PI / 180);
    size_t biggest = 0;
    for (const RoomPiece& piece : pieces)
    {
        float upness = piece.plane.normal.dot(cameraUp);
        if (std::abs(upness) <= tiltCos || piece.points.size() <= biggest) continue;
        biggest = piece.points.size();
        up = upness > 0 ? piece.plane.normal : -piece.plane.normal;
        room.upFrom = upness > 0 ? "the floor" : "the ceiling";
    }
    Sort(cameraUp, MaxTiltDegrees);
    if (biggest == 0)
    {
        float bestAcross = 0.5f;
        for (size_t i = 0; i < wallPieces.size(); i++)
            for (size_t j = i + 1; j < wallPieces.size(); j++)
            {
                cv::Vec3f across = pieces[wallPieces[i]].plane.normal.cross(pieces[wallPieces[j]].plane.normal);
                if (cv::norm(across) <= bestAcross) continue;
                bestAcross = (float)cv::norm(across);
                up = across / bestAcross;
                if (up.dot(cameraUp) < 0) up = -up;
                room.upFrom = "two walls";
            }
    }
    Sort(up, LevelDegrees);   // now by the room's own up, tightly: a ramp is neither floor nor wall
    std::sort(wallPieces.begin(), wallPieces.end(), [&](int a, int b) { return pieces[a].points.size() > pieces[b].points.size(); });
    auto Flatten = [&](cv::Vec3f v) { v -= up * up.dot(v); return v / (float)cv::norm(v); };
    cv::Vec3f towards = wallPieces.empty() ? Flatten(cv::Vec3f(0, 0, -1)) : Flatten(pieces[wallPieces[0]].plane.normal);
    cv::Vec3f right = up.cross(towards);
    auto ToRoom = [&](const cv::Vec3f& p) { return cv::Vec3f(right.dot(p), up.dot(p), towards.dot(p)); };
    for (cv::Vec3f& p : room.mesh.points) p = ToRoom(p);
    room.firstCamera = cv::Matx33f(right[0], right[1], right[2], up[0], up[1], up[2], towards[0], towards[1], towards[2]);

    std::vector<float> roomHeights, roomX, roomZ;
    for (auto& grid : roomGrid)
        for (cv::Vec3f& p : grid)
        {
            if (std::isnan(p[2])) continue;
            p = ToRoom(p);
            roomHeights.push_back(p[1]);
            roomX.push_back(p[0]);
            roomZ.push_back(p[2]);
        }
    for (RoomPiece& piece : pieces) for (cv::Vec3f& p : piece.points) p = ToRoom(p);
    room.floorSeen = floorPiece >= 0;
    room.ceilingSeen = ceilingPiece >= 0;
    room.floorY = room.floorSeen ? -pieces[floorPiece].plane.distance : Percentile(roomHeights, 0.02);
    room.ceilingY = room.ceilingSeen ? pieces[ceilingPiece].plane.distance : Percentile(roomHeights, 0.98);
    std::vector<std::vector<char>> taken(scans.size(), std::vector<char>(cells, 0));   // cells used by floor, ceiling, walls
    for (int k : { floorPiece, ceilingPiece })
        if (k >= 0) for (const auto& [s, i] : pieces[k].cells) taken[s][i] = 1;

    // --- Walls (09c step 7) ---
    struct Line { cv::Point2f towards, ends[2]; float offset; cv::Vec3b colour; };
    std::vector<Line> lines;
    for (int k : wallPieces)
    {
        cv::Vec3f n = ToRoom(Flatten(pieces[k].plane.normal));
        Line line;
        line.towards = cv::Point2f(n[0], n[2]) / std::hypot(n[0], n[2]);
        cv::Point2f along(-line.towards.y, line.towards.x);
        std::vector<float> offsets, alongs;
        for (const cv::Vec3f& p : pieces[k].points)
        {
            cv::Point2f q(p[0], p[2]);
            offsets.push_back(line.towards.dot(q));
            alongs.push_back(along.dot(q));
        }
        for (const auto& [s, i] : pieces[k].cells) taken[s][i] = 1;
        line.offset = Percentile(offsets, 0.5);
        line.ends[0] = line.towards * line.offset + along * Percentile(alongs, 0.02);
        line.ends[1] = line.towards * line.offset + along * Percentile(alongs, 0.98);
        line.colour = averageColour(pieces[k].colourSum, pieces[k].points.size());
        lines.push_back(line);
    }
    for (size_t i = 0; i < lines.size(); i++)
        for (size_t j = i + 1; j < lines.size(); j++)
        {
            cv::Point2f a = lines[i].towards, b = lines[j].towards;
            float determinant = a.x * b.y - a.y * b.x;
            if (std::abs(determinant) < 0.5f) continue;
            cv::Point2f crossing((lines[i].offset * b.y - lines[j].offset * a.y) / determinant,
                                 (a.x * lines[j].offset - b.x * lines[i].offset) / determinant);
            auto nearestEnd = [&](Line& line) -> cv::Point2f&
            { return cv::norm(line.ends[0] - crossing) < cv::norm(line.ends[1] - crossing) ? line.ends[0] : line.ends[1]; };
            cv::Point2f& endI = nearestEnd(lines[i]);
            cv::Point2f& endJ = nearestEnd(lines[j]);
            if (cv::norm(endI - crossing) < CornerSnapMeters && cv::norm(endJ - crossing) < CornerSnapMeters)
                endI = endJ = crossing;
        }
    for (const Line& line : lines)
    {
        cv::Point2f e0 = line.ends[0], e1 = line.ends[1];
        AddQuad(room.mesh, "clean_walls", { e0.x, room.floorY, e0.y }, { e1.x, room.floorY, e1.y },
                { e1.x, room.ceilingY, e1.y }, { e0.x, room.ceilingY, e0.y }, line.colour,
                { line.towards.x, 0, line.towards.y });
        room.walls.emplace_back(e0, e1);
    }

    // Floor and ceiling: rectangles over the walls, most of the points, and every camera.
    float minX = Percentile(roomX, 0.02), maxX = Percentile(roomX, 0.98);
    float minZ = Percentile(roomZ, 0.02), maxZ = Percentile(roomZ, 0.98);
    std::vector<cv::Point2f> outline;
    for (const auto& wall : room.walls) { outline.push_back(wall.first); outline.push_back(wall.second); }
    for (const Scan& scan : scans) { cv::Vec3f c = ToRoom(cv::Vec3f(scan.pose.shift)); outline.emplace_back(c[0], c[2]); }
    for (const cv::Point2f& e : outline)
    {
        minX = std::min(minX, e.x); maxX = std::max(maxX, e.x);
        minZ = std::min(minZ, e.y); maxZ = std::max(maxZ, e.y);
    }
    const cv::Vec3b grey(128, 128, 128);
    for (bool isFloor : { true, false })
    {
        float y = isFloor ? room.floorY : room.ceilingY;
        int k = isFloor ? floorPiece : ceilingPiece;
        AddQuad(room.mesh, isFloor ? "clean_floor" : "clean_ceiling", { minX, y, minZ }, { maxX, y, minZ }, { maxX, y, maxZ },
                { minX, y, maxZ }, k >= 0 ? averageColour(pieces[k].colourSum, pieces[k].points.size()) : grey,
                { 0, isFloor ? 1.0f : -1.0f, 0 });
    }

    // --- Furniture (09c step 8), per scan, then boxes merged across scans (new) ---
    struct Box { cv::Vec3f lo, hi; cv::Vec3d colourSum; size_t count; };
    std::vector<Box> boxes;
    const float margin = 0.05f;
    for (size_t s = 0; s < scans.size(); s++)
    {
        cv::Mat left(gridRows, gridColumns, CV_8U, cv::Scalar(0));
        for (int i = 0; i < cells; i++)
        {
            const cv::Vec3f& p = roomGrid[s][i];
            if (std::isnan(p[2]) || taken[s][i] || p[1] < room.floorY + margin || p[1] > room.ceilingY - margin) continue;
            // New: a point on a wall's plane is wall, even if this scan saw too little of it to find
            // the plane itself (a sliver at the edge of the view).
            bool onWall = false;
            for (const Line& line : lines)
                onWall = onWall || std::abs(line.towards.dot(cv::Point2f(p[0], p[2])) - line.offset) < InlierThicknessMeters;
            if (!onWall) left.data[i] = 255;
        }
        for (int row = 0; row + 1 < gridRows; row++)
            for (int column = 0; column + 1 < gridColumns; column++)
            {
                int i = row * gridColumns + column;
                for (int neighbour : { i + 1, i + gridColumns })
                {
                    float z = gridDepth[s][i], zn = gridDepth[s][neighbour];
                    if (!left.data[i] || std::isnan(zn)) continue;
                    if (std::abs(z - zn) > 0.03f * std::min(z, zn)) left.data[z < zn ? i : neighbour] = 0;
                }
            }
        cv::Mat labels, stats, centroids;
        int count = cv::connectedComponentsWithStats(left, labels, stats, centroids);
        std::vector<std::vector<int>> blobCells(count);
        for (int i = 0; i < cells; i++) if (labels.ptr<int>()[i] > 0) blobCells[labels.ptr<int>()[i]].push_back(i);
        for (int label = 1; label < count; label++)
        {
            if ((int)blobCells[label].size() < MinBlockCells) continue;
            std::vector<float> xs, ys, zs;
            Box box{ {}, {}, cv::Vec3d(0, 0, 0), blobCells[label].size() };
            for (int i : blobCells[label])
            {
                xs.push_back(roomGrid[s][i][0]);
                ys.push_back(roomGrid[s][i][1]);
                zs.push_back(roomGrid[s][i][2]);
                box.colourSum += cv::Vec3d(gridColour[s][i]);
            }
            box.lo = cv::Vec3f(Percentile(xs, 0.02), room.floorY, Percentile(zs, 0.02));
            box.hi = cv::Vec3f(Percentile(xs, 0.98), Percentile(ys, 0.98), Percentile(zs, 0.98));
            // New: wider than any furniture is a wall piece no plane was found for, not a block.
            if (box.hi[0] - box.lo[0] <= MaxBlockMeters && box.hi[2] - box.lo[2] <= MaxBlockMeters) boxes.push_back(box);
        }
    }
    // The same couch seen from two spots gives two overlapping boxes: join any that touch.
    for (bool joined = true; joined; )
    {
        joined = false;
        for (size_t i = 0; i < boxes.size() && !joined; i++)
            for (size_t j = i + 1; j < boxes.size() && !joined; j++)
            {
                bool touch = true;
                for (int a = 0; a < 3; a++)
                    touch = touch && boxes[i].lo[a] <= boxes[j].hi[a] + 0.02f && boxes[j].lo[a] <= boxes[i].hi[a] + 0.02f;
                // New: and only while the result stays furniture-sized; touching boxes chained through a
                // whole flat into one box that hid everything inside it.
                for (int a : { 0, 2 })
                    touch = touch && std::max(boxes[i].hi[a], boxes[j].hi[a]) - std::min(boxes[i].lo[a], boxes[j].lo[a]) <= MaxBlockMeters;
                if (!touch) continue;
                for (int a = 0; a < 3; a++)
                {
                    boxes[i].lo[a] = std::min(boxes[i].lo[a], boxes[j].lo[a]);
                    boxes[i].hi[a] = std::max(boxes[i].hi[a], boxes[j].hi[a]);
                }
                boxes[i].colourSum += boxes[j].colourSum;
                boxes[i].count += boxes[j].count;
                boxes.erase(boxes.begin() + j);
                joined = true;
            }
    }
    for (const Box& box : boxes)
    {
        AddBox(room.mesh, "furniture_blocks", box.lo, box.hi, averageColour(box.colourSum, box.count));
        room.blocks.emplace_back(box.lo, box.hi);
    }
    return room;
}

// As in 09c.
bool SaveObj(const std::string& path, const Mesh& mesh)
{
    std::ofstream file(path);
    file << "# Spotter 13: seen_room = every scan merged, clean_* and furniture_blocks = the tidy (partly guessed) model\n";
    for (size_t i = 0; i < mesh.points.size(); i++)
    {
        const cv::Vec3f& p = mesh.points[i];
        const cv::Vec3b& c = mesh.colours[i];
        file << cv::format("v %.4f %.4f %.4f %.3f %.3f %.3f\n", p[0], p[1], p[2], c[2] / 255.0, c[1] / 255.0, c[0] / 255.0);
    }
    for (const auto& [name, faces] : mesh.groups)
    {
        file << "g " << name << "\n";
        for (const cv::Vec3i& f : faces) file << cv::format("f %d %d %d\n", f[0] + 1, f[1] + 1, f[2] + 1);
    }
    return (bool)file;
}

// Step 4: merge every scan, build the clean model, save. Prints what it made.
void Export(const std::vector<Scan>& allScans, const Lens& lens, std::mt19937& random, const std::vector<Landmark>& landmarks)
{
    std::vector<Scan> scans;   // new: excluded scans stay out
    for (const Scan& s : allScans) if (!s.excluded) scans.push_back(s);
    if (scans.empty()) { std::printf("Nothing to export yet: SPACE scans (excluded scans don't count).\n"); return; }
    std::printf("Merging %zu scans...\n", scans.size());
    auto start = std::chrono::steady_clock::now();
    float voxel;
    Mesh seen = FuseScans(scans, lens, voxel);
    Room room = BuildRoom(scans, seen, random);
    // New in 13d: the far landmarks, as 10 cm magenta boxes in their own group (turned into the room like everything else).
    for (const Landmark& l : landmarks)
    {
        cv::Point3f q = room.firstCamera * l.point;
        cv::Vec3f centre(q.x, q.y, q.z), half(0.05f, 0.05f, 0.05f);
        AddBox(room.mesh, "far_landmarks", centre - half, centre + half, cv::Vec3b(255, 0, 255));
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::string path = (std::filesystem::current_path() / cv::format("room_scan_%lld.obj", (long long)std::time(nullptr))).string();
    if (!SaveObj(path, room.mesh)) { std::printf("Could not write %s\n", path.c_str()); return; }
    // How scan 1's camera sat in the room, for presentation/Demo4_Room_Twin (an OBJ comment, ignored elsewhere).
    const cv::Matx33f& t = room.firstCamera;
    std::ofstream(path, std::ios::app) << cv::format("# first camera turn: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                                                     t(0, 0), t(0, 1), t(0, 2), t(1, 0), t(1, 1), t(1, 2), t(2, 0), t(2, 1), t(2, 2));
    std::printf("Saved %s (%.1f MB, %.1f s). A capture of the room: never commit it.\n", path.c_str(),
                std::filesystem::file_size(path) / 1e6, seconds);
    std::printf("  merged mesh: %zu triangles, %.1f cm cubes; %zu far landmarks (group far_landmarks)\n",
                room.mesh.groups["seen_room"].size(), voxel * 100, landmarks.size());
    std::printf("  up from %s; floor %.2f m below the first camera (%s), ceiling %.2f m above it (%s): %.2f m high\n",
                room.upFrom.c_str(), -room.floorY, room.floorSeen ? "seen" : "guessed",
                room.ceilingY, room.ceilingSeen ? "seen" : "guessed", room.ceilingY - room.floorY);
    for (const auto& [a, b] : room.walls)
        std::printf("  wall  %.2f m long, from (%+.2f, %.2f) to (%+.2f, %.2f) m\n", cv::norm(b - a), a.x, -a.y, b.x, -b.y);
    for (const auto& [lo, hi] : room.blocks)
        std::printf("  block %.2f wide x %.2f deep x %.2f high, at (%+.2f, %.2f) m\n", hi[0] - lo[0], hi[2] - lo[2],
                    hi[1] - lo[1], (lo[0] + hi[0]) / 2, -(lo[2] + hi[2]) / 2);
    std::printf("  (positions: X right, Z ahead, along the biggest wall, from the first camera)\n");
}

// ---------------------------------------------------------------------
//  Keeping the scans: K writes a folder that 13z rebuilds from (added, as in 13c)
// ---------------------------------------------------------------------

// Every point with its own colour, as a binary PLY. 08 wrote text, but a room's 5 million points
// would be about 150 MB of it. MeshLab and Blender read both.
bool SavePly(const std::string& path, const std::vector<cv::Point3f>& points, const std::vector<cv::Vec3b>& colours)
{
    std::ofstream file(path, std::ios::binary);
    file << "ply\nformat binary_little_endian 1.0\n"
         << "comment Spotter: every scan's own points. A capture of the room: never commit it.\n"
         << "element vertex " << points.size() << "\n"
         << "property float x\nproperty float y\nproperty float z\n"
         << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (size_t i = 0; i < points.size(); i++)
    {
        file.write((const char*)&points[i], 3 * sizeof(float));
        const unsigned char rgb[3] = { colours[i][2], colours[i][1], colours[i][0] };   // OpenCV keeps BGR
        file.write((const char*)rgb, 3);
    }
    return (bool)file;
}

// A line typed in the console, without the spaces or quotes round it (Explorer's "Copy as path" adds quotes).
std::string Ask(const std::string& question)
{
    std::printf("%s", question.c_str());
    std::string answer;
    if (!std::getline(std::cin, answer)) return "cancel";
    while (!answer.empty() && (answer.back() == ' ' || answer.back() == '"' || answer.back() == '\r')) answer.pop_back();
    size_t start = answer.find_first_not_of(" \"");
    return start == std::string::npos ? "" : answer.substr(start);
}

// Where K saves: a folder (Enter = SaveFolder) and a name (Enter = room_<date>_<time>). A saved room
// of the same name is overwritten only if you say so; any other folder already there is never
// touched. Returns an empty path if you type cancel.
std::filesystem::path AskWhere()
{
    namespace fs = std::filesystem;
    std::printf("\n=== K: keep the scans. Answer here, in the console (the windows wait). cancel stops. ===\n");
    std::string folder = Ask(cv::format("Folder [%s]: ", fs::absolute(SaveFolder).string().c_str()));
    if (folder == "cancel") return {};
    if (folder.empty()) folder = SaveFolder;
    char stamp[32];
    std::time_t now = std::time(nullptr);
    std::strftime(stamp, sizeof stamp, "room_%Y-%m-%d_%H%M", std::localtime(&now));
    while (true)
    {
        std::string name = Ask(cv::format("Name [%s]: ", stamp));
        if (name == "cancel") return {};
        if (name.empty()) name = stamp;
        if (name.find_first_of("\\/:*?\"<>|") != std::string::npos)
        {
            std::printf("A name can't hold \\ / : * ? \" < > |. Again.\n");
            continue;
        }
        fs::path where = fs::absolute(fs::path(folder) / name);
        std::error_code error;
        if (!fs::exists(where)) return where;
        if (!fs::exists(where / "room.txt"))
        {
            if (fs::is_directory(where) && fs::is_empty(where, error)) return where;
            std::printf("%s is there already and isn't a saved room, so it's left alone. Another name.\n", where.string().c_str());
            continue;
        }
        if (Ask(cv::format("%s is a saved room already. Overwrite it? [y/N]: ", where.string().c_str())) == "y")
        {
            fs::remove_all(where, error);
            if (!error) return where;
            std::printf("Couldn't remove it (%s). Another name.\n", error.message().c_str());
        }
    }
}

// The folder K writes. Plain files, so 13z (and you) can read it without this program:
//   .gitignore      "*": git never takes anything in this folder, wherever it is
//   room.txt        what this is, both cameras' factory settings, and how scan 1's camera sat in the room
//   points.ply      every included scan's own points and colours, in the same coordinates as E's .obj
//   scan_001\ ...   cloud.bin        the averaged depth as x, y, z floats per depth pixel (NaN = none)
//                   colour.bmp       the colour picture on the depth grid (512 x 424)
//                   colour_full.bmp  the full colour picture (1920 x 1080)
//                   colour_index.bin which full-picture pixel each depth pixel landed on (int, -1 = none)
//                   features.yml     its keypoints and ORB descriptors, and its far ones with their rays
//                   pose.txt         where it sits in the room, excluded, suspicious, how it was placed
//   landmarks.txt   13d: the far landmarks, in scan 1's camera coordinates, with their descriptors
// The pictures are BMP: lossless, and the only format this OpenCV writes (vcpkg.json leaves out
// its png and jpeg features). About 11 MB per scan.
void KeepScans(const std::vector<Scan>& scans, const libfreenect2::Freenect2Device::IrCameraParams& ir,
               const libfreenect2::Freenect2Device::ColorCameraParams& colour, const std::vector<Landmark>& landmarks)
{
    namespace fs = std::filesystem;
    if (scans.empty()) { std::printf("Nothing to keep yet: SPACE scans.\n"); return; }
    fs::path where = AskWhere();
    if (where.empty()) { std::printf("K cancelled: nothing written.\n"); return; }
    std::error_code error;
    fs::create_directories(where, error);
    if (error) { std::printf("Could not make %s: %s\n", where.string().c_str(), error.message().c_str()); return; }
    auto start = std::chrono::steady_clock::now();
    std::ofstream(where / ".gitignore") << "# A capture of a room (Spotter's K): never commit it.\n*\n";

    // The room's coordinates as E's .obj has them (levelled by the biggest floor or ceiling). Only
    // the clean model's planes decide them, so the merged mesh isn't needed.
    std::vector<Scan> included;
    for (const Scan& s : scans) if (!s.excluded) included.push_back(s);
    std::mt19937 random(1);
    Room room = included.empty() ? Room() : BuildRoom(included, Mesh(), random);
    const cv::Matx33f& t = room.firstCamera;

    char saved[32];
    std::time_t now = std::time(nullptr);
    std::strftime(saved, sizeof saved, "%Y-%m-%d %H:%M", std::localtime(&now));
    {
        std::ofstream file(where / "room.txt");
        file << "# A Spotter room scan folder, written by K. 13z rebuilds the room from it. A capture of a room: never commit it.\n"
             << "format 1\nprogram 13d_Kinect_Room_Scan_Far\nsaved " << saved << "\nscans " << scans.size() << "\n"
             << "depth_size " << Width << " " << Height << "\ncolour_size " << ColorWidth << " " << ColorHeight << "\n";
        // The cameras' factory settings as libfreenect2 reads them, in its own order: the depth camera's
        // fx fy cx cy k1 k2 k3 p1 p2, then the colour camera's fx fy cx cy shift_d shift_m mx_... my_...
        file << "ir_params";
        for (size_t i = 0; i < sizeof ir / sizeof(float); i++) file << cv::format(" %.9g", (&ir.fx)[i]);
        file << "\ncolour_params";
        for (size_t i = 0; i < sizeof colour / sizeof(float); i++) file << cv::format(" %.9g", (&colour.fx)[i]);
        file << cv::format("\nfirst_camera_turn %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                           t(0, 0), t(0, 1), t(0, 2), t(1, 0), t(1, 1), t(1, 2), t(2, 0), t(2, 1), t(2, 2))
             << "up_from " << room.upFrom << "\n";
    }

    int failed = 0;
    for (size_t s = 0; s < scans.size(); s++)
    {
        const Scan& scan = scans[s];
        fs::path dir = where / cv::format("scan_%03zu", s + 1);
        fs::create_directories(dir, error);
        std::ofstream(dir / "cloud.bin", std::ios::binary).write((const char*)scan.cloud.data(), scan.cloud.size() * sizeof(cv::Point3f));
        if (!cv::imwrite((dir / "colour.bmp").string(), scan.picture)) failed++;
        if (!scan.fullPicture.empty() && !cv::imwrite((dir / "colour_full.bmp").string(), scan.fullPicture)) failed++;
        if (!scan.colourIndex.empty())
            std::ofstream(dir / "colour_index.bin", std::ios::binary).write((const char*)scan.colourIndex.data(), scan.colourIndex.size() * sizeof(int));
        cv::FileStorage features((dir / "features.yml").string(), cv::FileStorage::WRITE);
        features << "keypoints" << scan.keypoints << "descriptors" << scan.descriptors;
        features << "far_keypoints" << scan.farKeypoints << "far_descriptors" << scan.farDescriptors   // 13d
                 << "far_rays" << scan.farRays << "far_origin" << scan.farOrigin;
        features.release();
        const cv::Matx33f& r = scan.pose.turn;
        std::ofstream(dir / "pose.txt")
            << cv::format("# scan %zu's camera -> the room (scan 1's camera): room point = turn * point + shift\n", s + 1)
            << cv::format("turn %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f\n", r(0, 0), r(0, 1), r(0, 2), r(1, 0), r(1, 1), r(1, 2),
                          r(2, 0), r(2, 1), r(2, 2))
            << cv::format("shift %.6f %.6f %.6f\n", scan.pose.shift.x, scan.pose.shift.y, scan.pose.shift.z)
            << "excluded " << (int)scan.excluded << "\nsuspicious " << (int)scan.suspicious << "\nstats " << scan.stats << "\n";
        std::printf("  scan %zu of %zu written\r", s + 1, scans.size());
    }

    // 13d: the landmarks, one a line: x y z (scan 1's camera), times seen, the 32-byte descriptor in hex.
    {
        std::ofstream file(where / "landmarks.txt");
        file << "# far landmarks: x y z (metres, scan 1's camera), times seen, ORB descriptor (hex)\n";
        for (const Landmark& l : landmarks)
        {
            file << cv::format("%.4f %.4f %.4f %d ", l.point.x, l.point.y, l.point.z, l.seen);
            for (int b = 0; b < l.descriptor.cols; b++) file << cv::format("%02x", l.descriptor.at<uchar>(0, b));
            file << "\n";
        }
    }

    // points.ply: no averaging, no cubes. Every included scan's points, as the viewer shows them.
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colours;
    for (const Scan& scan : included)
        for (int i = 0; i < Width * Height; i++)
        {
            const cv::Point3f& p = scan.cloud[i];
            if (std::isnan(p.z)) continue;
            points.push_back(t * scan.pose(p));
            colours.push_back(scan.picture.at<cv::Vec3b>(i / Width, i % Width));
        }
    if (!points.empty() && !SavePly((where / "points.ply").string(), points, colours)) failed++;

    uintmax_t bytes = 0;
    for (const auto& entry : fs::recursive_directory_iterator(where, error))
        if (entry.is_regular_file(error)) bytes += entry.file_size(error);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("Kept %zu scans (%zu included) and %zu landmarks in %s: %.0f MB, %.1f s.%s\n", scans.size(), included.size(),
                landmarks.size(), where.string().c_str(),
                bytes / 1e6, seconds, failed ? cv::format(" %d files could NOT be written.", failed).c_str() : "");
    std::printf("  points.ply: %zu points, each with its own colour. Rebuild the room any time: 13z_Kinect_Room_Rebuild.\n"
                "  A capture of the room: never commit it (the folder has its own .gitignore).\n", points.size());
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // -----------------------------------------------------------------
    //  Setup: depth and colour, registered (as in 11b)
    // -----------------------------------------------------------------

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
    libfreenect2::Freenect2Device::IrCameraParams ir = device->getIrCameraParams();
    const Lens lens{ ir.fx, ir.fy, ir.cx, ir.cy };
    const libfreenect2::Freenect2Device::ColorCameraParams colourParams = device->getColorCameraParams();   // added: K saves both
    libfreenect2::Registration registration(ir, colourParams);
    libfreenect2::Frame undistorted(Width, Height, 4), registered(Width, Height, 4), averaged(Width, Height, 4);
    std::vector<int> colourIndex(Width * Height);   // new: the colour pixel each depth pixel maps to (05c)
    libfreenect2::Frame bigdepth(ColorWidth, ColorHeight + 2, 4);   // 13d: depth per colour pixel (a row of padding above and below)
    ColourModel colourModel;                                       // 13d: measured from each scan
    std::vector<Landmark> landmarks;                               // 13d: far details placed in the room
    // Where the colour picture has no depth (255), from bigdepth.
    auto NoDepthMask = [&]
    {
        cv::Mat inner = cv::Mat(ColorHeight + 2, ColorWidth, CV_32F, bigdepth.data).rowRange(1, ColorHeight + 1), hasDepth, noDepth;
        cv::bitwise_and(inner > 0, inner < 100000, hasDepth);
        cv::bitwise_not(hasDepth, noDepth);
        return noDepth;
    };

    std::vector<Scan> scans;
    std::vector<std::vector<cv::Point3f>> shownPoints;   // what the viewer draws of each scan, in the room
    std::vector<std::vector<cv::Vec3b>> shownColours;
    int framesCaptured = -1;   // -1 = not scanning
    std::vector<double> sumMm(Width * Height);
    std::vector<int> readings(Width * Height);
    std::string message = "Set the Kinect down where you'll start, then press SPACE (in a window) to scan.";
    bool colourByScan = false;
    std::mt19937 random(1);
    int selected = -1;   // new: the scan clicked in the viewer
    const int cores = (int)std::max(2u, std::thread::hardware_concurrency());

    // New: three jobs run in the background, so the windows never freeze. They only READ "scans";
    // the main loop changes it only when no job is reading (WaitForReaders, and no SPACE/U/X/P while placing).
    struct Placement
    {
        bool placed = false;
        Scan scan;
        std::string report;
        int again = -1;                                // P's scan
        size_t matched = 0;                            // 13d: the scan its features matched...
        std::vector<std::pair<int, int>> farPairs;     // ...and the far matches that agreed (new landmarks)
    };
    std::future<Placement> placing;
    std::future<LiveCheck> checking;
    std::future<Recommendation> recommending;
    LiveCheck live;
    Recommendation next;
    auto lastCheck = std::chrono::steady_clock::now();
    auto Running = [](const auto& job) { return job.valid() && job.wait_for(std::chrono::seconds(0)) != std::future_status::ready; };
    auto WaitForReaders = [&]
    {
        if (checking.valid()) live = checking.get();
        if (recommending.valid()) next = recommending.get();
    };
    auto Recommend = [&]
    {
        if (recommending.valid()) next = recommending.get();
        recommending = std::async(std::launch::async, RecommendNext, std::cref(scans), lens);
    };
    auto Shown = [&](const Scan& scan, std::vector<cv::Point3f>& points, std::vector<cv::Vec3b>& colours)
    {
        points.clear();
        colours.clear();
        for (int row = 0; row < Height; row += ShowStep)
            for (int column = 0; column < Width; column += ShowStep)
            {
                const cv::Point3f& p = scan.cloud[row * Width + column];
                if (std::isnan(p.z)) continue;
                points.push_back(scan.pose(p));
                colours.push_back(scan.picture.at<cv::Vec3b>(row, column));
            }
    };
    auto StartPlacing = [&](Scan scan, int again)
    {
        cv::setNumThreads(cores - 2);   // leave libfreenect2's USB and depth threads a core or two
        placing = std::async(std::launch::async, [&scans, &landmarks, lens, again](Scan s) {
            Placement p;
            std::mt19937 rng((unsigned)scans.size() + 11);
            p.placed = PlaceScan(s, scans, lens, rng, p.report, again, &landmarks, &p.matched, &p.farPairs);
            p.scan = std::move(s);
            p.again = again;
            return p;
        }, std::move(scan));
    };

    const std::string viewer = "13d room scan far - the room so far  (drag/WASD turn, wheel zoom, click a camera to select)";
    const std::string camera = "13d room scan far - camera  (SPACE scan, U undo, X exclude, P place again, E export, K keep, Q quit)";
    cv::namedWindow(viewer, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(viewer, OnMouse);
    cv::namedWindow(camera, cv::WINDOW_AUTOSIZE);
    std::printf("%s\n", message.c_str());

    // -----------------------------------------------------------------
    //  Main operation
    // -----------------------------------------------------------------

    libfreenect2::FrameMap frames;
    cv::Mat frame, display;
    while (true)
    {
        int key = cv::waitKey(1);
        if (key == 'q') break;
        if (key == 27) selected = -1;   // Esc deselects (in 13b it quit; Q still does)
        if (key == 'a') yawDegrees -= TurnStepDegrees;
        if (key == 'd') yawDegrees += TurnStepDegrees;
        if (key == 'w') pitchDegrees += TurnStepDegrees;
        if (key == 's') pitchDegrees -= TurnStepDegrees;
        if (key == '=' || key == '+') viewDistance -= ZoomStepMeters;
        if (key == '-') viewDistance += ZoomStepMeters;
        if (key == 'r') { yawDegrees = 0; pitchDegrees = 0; viewDistance = PivotMeters; }
        if (key == 'c') colourByScan = !colourByScan;
        pitchDegrees = std::clamp(pitchDegrees, -89.0f, 89.0f);
        viewDistance = std::max(viewDistance, 0.1f);
        UpdateView();

        // --- New: a click in the viewer selects the nearest camera circle ---
        if (clicked)
        {
            clicked = false;
            selected = -1;
            double nearest = 15;
            for (size_t s = 0; s < scans.size(); s++)
            {
                cv::Point pixel;
                float depth;
                if (!Project(scans[s].pose.shift, pixel, depth)) continue;
                double d = cv::norm(pixel - cv::Point(clickX, clickY));
                if (d < nearest) { nearest = d; selected = (int)s; }
            }
            if (selected >= 0)
                std::printf("Selected scan %d%s: %s\n", selected + 1, scans[selected].excluded ? " (excluded)" : "",
                            scans[selected].stats.c_str());
        }

        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }
        libfreenect2::Frame* colorFrame = frames[libfreenect2::Frame::Color];
        cv::cvtColor(cv::Mat(ColorHeight, ColorWidth, CV_8UC4, colorFrame->data), frame, cv::COLOR_BGRA2BGR);
        // 13d: depth per colour pixel (for the far features' mask), only on the frames that need it.
        const bool liveDue = !Running(placing) && framesCaptured < 0 && !scans.empty() && !Running(checking) &&
                             std::chrono::steady_clock::now() - lastCheck > std::chrono::milliseconds(LiveEveryMs);
        const bool needBig = framesCaptured == AverageFrames - 1 || liveDue;
        registration.apply(colorFrame, frames[libfreenect2::Frame::Depth], &undistorted, &registered, true,
                           needBig ? &bigdepth : nullptr, colourIndex.data());
        listener.release(frames);
        cv::Mat colourGrey, noDepth;
        if (needBig)
        {
            cv::cvtColor(frame, colourGrey, cv::COLOR_BGR2GRAY);
            noDepth = NoDepthMask();
        }

        // --- New: a background placement finished ---
        if (placing.valid() && !Running(placing))
        {
            Placement done = placing.get();
            cv::setNumThreads(-1);   // all cores again
            WaitForReaders();
            if (done.again >= 0)
            {
                if (done.placed)
                {
                    Scan& s = scans[done.again];
                    s.pose = done.scan.pose;
                    s.suspicious = done.scan.suspicious;
                    s.stats = done.scan.stats;
                    Shown(s, shownPoints[done.again], shownColours[done.again]);
                    AddLandmarks(s, scans[done.matched], done.farPairs, landmarks);
                    message = cv::format("Scan %d placed again.%s", done.again + 1, s.suspicious ? " Still suspicious: X excludes it." : "");
                }
                else message = cv::format("Couldn't place scan %d again (see the console). X excludes it.", done.again + 1);
                std::printf("Scan %d again: %s.\n", done.again + 1, done.report.c_str());
            }
            else if (done.placed)
            {
                scans.push_back(std::move(done.scan));
                shownPoints.emplace_back();
                shownColours.emplace_back();
                Shown(scans.back(), shownPoints.back(), shownColours.back());
                int added = AddLandmarks(scans.back(), scans[done.matched], done.farPairs, landmarks);
                if (added > 0) std::printf("  %d new far landmarks (%zu in all)\n", added, landmarks.size());
                message = cv::format("Scan %zu placed%s. Keep the green anchors in view as you move, or follow the Next line.",
                                     scans.size(), scans.back().suspicious ? " - SUSPICIOUS (red): check it" : "");
                std::printf("Scan %zu: %zu near + %zu far features; %s.\n", scans.size(), scans.back().keypoints.size(),
                            scans.back().farKeypoints.size(), done.report.c_str());
            }
            else
            {
                message = "Couldn't place that scan (see the console). Aim back at scanned parts until the anchors are green.";
                std::printf("Scan refused: %s.\n", done.report.c_str());
            }
            std::printf("%s\n", message.c_str());
            Recommend();
        }
        const bool busy = Running(placing);
        const bool idle = !busy && framesCaptured < 0;

        // --- U: undo the last scan ---
        if (key == 'u' && idle && !scans.empty())
        {
            WaitForReaders();
            scans.pop_back();
            shownPoints.pop_back();
            shownColours.pop_back();
            if (selected >= (int)scans.size()) selected = -1;
            message = cv::format("Removed scan %zu. %zu left.", scans.size() + 1, scans.size());
            std::printf("%s\n", message.c_str());
            Recommend();
        }

        // --- New: X excludes the selected scan (or puts it back) ---
        if (key == 'x' && idle && selected >= 0)
        {
            WaitForReaders();
            scans[selected].excluded = !scans[selected].excluded;
            message = cv::format("Scan %d %s.", selected + 1, scans[selected].excluded ? "excluded: left out of placing, advice and E"
                                                                                       : "included again");
            std::printf("%s\n", message.c_str());
            Recommend();
        }

        // --- New: P places the selected scan again, against all the others ---
        if (key == 'p' && idle && selected >= 0)
        {
            if (scans[selected].excluded) message = "Include it first (X), then P.";
            else
            {
                WaitForReaders();
                StartPlacing(scans[selected], selected);
                message = cv::format("Placing scan %d again...", selected + 1);
            }
        }

        // --- E: merge and save (excluded scans stay out) ---
        if (key == 'e' && idle)
        {
            WaitForReaders();
            Export(scans, lens, random, landmarks);
            message = scans.empty() ? "Nothing to export yet." : "Exported (see the console). Scan more and export again, or Q.";
        }

        // --- Added: K keeps the scans in a folder, for 13z (the questions are asked in the console) ---
        if (key == 'k' && idle)
        {
            WaitForReaders();
            if (!display.empty())
            {
                cv::Mat note = display.clone();
                cv::putText(note, "K: answer the questions in the CONSOLE window", cv::Point(10, 160), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                            cv::Scalar(0, 0, 255), 2);
                cv::imshow(camera, note);
                cv::waitKey(1);
            }
            KeepScans(scans, ir, colourParams, landmarks);
            message = scans.empty() ? "Nothing to keep yet." : "K done (see the console). Scan more and K again, or Q.";
        }

        // --- Step 1: SPACE starts a scan; average AverageFrames frames (07) ---
        if (key == ' ' && idle)
        {
            framesCaptured = 0;
            std::fill(sumMm.begin(), sumMm.end(), 0.0);
            std::fill(readings.begin(), readings.end(), 0);
        }
        if (framesCaptured >= 0)
        {
            const float* depthMm = (const float*)undistorted.data;
            for (int i = 0; i < Width * Height; i++)
            {
                if (!(depthMm[i] > 0) || std::isinf(depthMm[i])) continue;
                sumMm[i] += depthMm[i];
                readings[i]++;
            }
            framesCaptured++;
        }
        if (framesCaptured == AverageFrames)
        {
            framesCaptured = -1;
            float* averageMm = (float*)averaged.data;
            for (int i = 0; i < Width * Height; i++)
                averageMm[i] = readings[i] >= AverageFrames / 2 ? (float)(sumMm[i] / readings[i]) : 0;
            std::vector<cv::Point3f> cloud(Width * Height);
            for (int row = 0; row < Height; row++)
                for (int column = 0; column < Width; column++)
                {
                    float x, y, z;
                    registration.getPointXYZ(&averaged, row, column, x, y, z);
                    cloud[row * Width + column] = cv::Point3f(x, y, z);
                }
            cv::Mat picture;
            cv::cvtColor(cv::Mat(Height, Width, CV_8UC4, registered.data), picture, cv::COLOR_BGRA2BGR);
            Scan scan = MakeScan(cloud, picture);
            scan.fullPicture = frame.clone();   // added: for K
            scan.colourIndex = colourIndex;
            // 13d: the far features, from this frame's full colour picture where depth reads nothing.
            ColourModel fitted = FitColourModel(cloud, colourIndex);
            if (fitted.ok) colourModel = fitted;
            AddFarFeatures(scan, colourGrey, noDepth, colourModel, lens);

            // --- Step 2: place it, in the background (new). Without features no later scan could be matched to it. ---
            if ((int)(scan.keypoints.size() + scan.farKeypoints.size()) < MinScanFeatures)
            {
                message = "Scan refused: too few features. Turn the lights on, or aim at something with more pattern.";
                std::printf("Scan refused: only %zu near + %zu far features (need %d): the colour picture is too dark or too plain.\n",
                            scan.keypoints.size(), scan.farKeypoints.size(), MinScanFeatures);
            }
            else if (scans.empty())
            {
                scan.stats = "the room's starting point";
                scans.push_back(std::move(scan));
                shownPoints.emplace_back();
                shownColours.emplace_back();
                Shown(scans.back(), shownPoints.back(), shownColours.back());
                message = "Scan 1: the room's starting point. Move or turn, keeping the green anchors in view, SPACE again.";
                std::printf("Scan 1: %zu near + %zu far features; the room's starting point.\n%s\n", scans.back().keypoints.size(),
                            scans.back().farKeypoints.size(), message.c_str());
                Recommend();
            }
            else
            {
                WaitForReaders();
                StartPlacing(std::move(scan), -1);
                message = cv::format("Placing scan %zu...", scans.size() + 1);
            }
        }

        // --- New: the live anchors, about twice a second, in the background ---
        if (checking.valid() && !Running(checking)) live = checking.get();
        if (recommending.valid() && !Running(recommending))
        {
            next = recommending.get();
            std::printf("%s\n", next.text.c_str());
        }
        auto now = std::chrono::steady_clock::now();
        if (idle && liveDue && !colourGrey.empty() && !Running(checking) && now - lastCheck > std::chrono::milliseconds(LiveEveryMs))
        {
            std::vector<cv::Point3f> cloud(Width * Height);
            for (int row = 0; row < Height; row++)
                for (int column = 0; column < Width; column++)
                {
                    float x, y, z;
                    registration.getPointXYZ(&undistorted, row, column, x, y, z);
                    cloud[row * Width + column] = cv::Point3f(x, y, z);
                }
            cv::Mat picture;
            cv::cvtColor(cv::Mat(Height, Width, CV_8UC4, registered.data), picture, cv::COLOR_BGRA2BGR);
            if (!colourModel.ok) { ColourModel fitted = FitColourModel(cloud, colourIndex); if (fitted.ok) colourModel = fitted; }
            checking = std::async(std::launch::async, CheckLive, std::move(cloud), picture, colourIndex, colourGrey, noDepth,
                                  colourModel, lens, std::cref(scans), std::cref(landmarks));
            lastCheck = now;
        }

        // -----------------------------------------------------------------
        //  Output: the room so far, and the camera with instructions
        // -----------------------------------------------------------------

        std::vector<char> excluded(scans.size());
        for (size_t s = 0; s < scans.size(); s++) excluded[s] = scans[s].excluded;
        cv::Mat image = Render(shownPoints, shownColours, excluded, colourByScan);
        // A camera: a circle where it stood, and four arrows to the corners of what it saw (13b).
        auto DrawCamera = [&](const Pose& pose, const cv::Scalar& colour, int thickness, const std::string& label)
        {
            cv::Point pixel;
            float depth;
            if (!Project(pose.shift, pixel, depth)) return;
            cv::circle(image, pixel, 6, colour, thickness + 1);
            for (cv::Point2f corner : { cv::Point2f(0, 0), cv::Point2f(Width, 0), cv::Point2f(0, Height), cv::Point2f(Width, Height) })
            {
                cv::Point3f ahead((corner.x - lens.cx) / lens.fx * ScanArrowMeters, (corner.y - lens.cy) / lens.fy * ScanArrowMeters,
                                  ScanArrowMeters);
                cv::Point tip;
                float tipDepth;
                if (Project(pose(ahead), tip, tipDepth)) cv::arrowedLine(image, pixel, tip, colour, thickness, cv::LINE_AA, 0, 0.15);
            }
            cv::putText(image, label, pixel + cv::Point(8, -8), cv::FONT_HERSHEY_SIMPLEX, 0.5, colour, 1);
        };
        int excludedCount = 0;
        for (size_t s = 0; s < scans.size(); s++)
        {
            excludedCount += scans[s].excluded;
            cv::Scalar colour = (int)s == selected ? cv::Scalar(255, 255, 255) : scans[s].excluded ? cv::Scalar(110, 110, 110)
                              : scans[s].suspicious ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 255);
            DrawCamera(scans[s].pose, colour, (int)s == selected ? 2 : 1, std::to_string(s + 1));
        }
        if (next.found && !busy) DrawCamera(next.pose, cv::Scalar(0, 255, 0), 2, "next");
        for (const Landmark& l : landmarks)   // 13d: the far landmarks
        {
            cv::Point pixel;
            float depth;
            if (Project(l.point, pixel, depth)) cv::circle(image, pixel, 3, cv::Scalar(255, 0, 255), cv::FILLED);
        }
        cv::putText(image, cv::format("%zu scans (%d excluded), %zu far landmarks", scans.size(), excludedCount, landmarks.size()),
                    cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
        cv::imshow(viewer, image);

        cv::resize(frame, display, cv::Size(ColorWidth / 2, ColorHeight / 2));
        // New: the live anchors. Green = shared with the scanned room: the next scan is placed by these.
        if (idle && !scans.empty())
        {
            for (const cv::Point2f& p : live.otherPixels) cv::circle(display, p * 0.5f, 2, cv::Scalar(140, 140, 140), cv::FILLED);
            for (const cv::Point2f& p : live.anchorPixels) cv::circle(display, p * 0.5f, 4, cv::Scalar(0, 255, 0), cv::FILLED);
            for (const cv::Point2f& p : live.farAnchorPixels) cv::circle(display, p * 0.5f, 4, cv::Scalar(255, 0, 255), cv::FILLED);
            cv::Scalar countColour = live.anchors < MinFeatureMatches ? cv::Scalar(0, 0, 255)
                                   : live.anchors < 2 * MinFeatureMatches ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0);
            std::string count = live.scan >= 0 ? cv::format("anchors: %d (need %d; %d far) - matches scan %d", live.anchors, MinFeatureMatches,
                                                            live.farAnchors, live.scan + 1)
                                               : cv::format("anchors: 0 (need %d) - aim at something already scanned", MinFeatureMatches);
            cv::putText(display, count, cv::Point(10, 65), cv::FONT_HERSHEY_SIMPLEX, 0.7, countColour, 2);
            cv::putText(display, next.text, cv::Point(10, 95), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 1);
        }
        if (selected >= 0)
            cv::putText(display, cv::format("Selected scan %d%s: X %s   P place again   Esc deselect", selected + 1,
                                            scans[selected].excluded ? " (excluded)" : "", scans[selected].excluded ? "include" : "exclude"),
                        cv::Point(10, 125), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
        std::string top = busy ? std::string("PLACING... (the windows keep working)")
                        : framesCaptured >= 0 ? cv::format("HOLD STILL  %d / %d", framesCaptured, AverageFrames)
                                              : cv::format("%zu scans   SPACE scan here   U undo   E export   K keep   Q quit", scans.size());
        cv::putText(display, top, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        cv::putText(display, message, cv::Point(10, display.rows - 15), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 1);
        cv::imshow(camera, display);
    }

    if (placing.valid()) placing.wait();
    WaitForReaders();
    device->stop();
    device->close();
    cv::destroyAllWindows();
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Scan one spot, export, and compare with 09c's file: same room, but the merged mesh has no
//     torn edges at depth jumps, only gaps where nothing was seen.
//  2. Scan the same spot twice and export: the TSDF averages them, so the mesh gets smoother.
//  3. Press C after a few scans. Each scan's colour should sit on the others' surfaces; a scan
//     that shows as a separate, shifted copy was placed wrong: U removes it.
//  4. VoxelMeters = 0.015 versus 0.06: detail, file size and merge time.
//  5. Walk the Kinect all the way round the room, back to the start. Does the last scan meet the
//     first, or is there a step where small errors added up ("drift")? Real scanners fix that
//     with "loop closure"; here, ICP against all earlier scans keeps it small.
//  6. Scan a bare wall from 2 m, then slide 1 m along it. Refused (no features) or placed wrong
//     (ICP slides along a flat wall, 11)? Put a poster up and try again.
//  7. Hide groups in Blender: seen_room (what was scanned) against the clean_* groups and
//     furniture_blocks (the tidy model). Which boxes are real furniture?
//
//  New in 13b:
//  8. Set EqualiseContrast = false and FastThreshold = 20 (13's recognition) and scan the same
//     dim spot: compare the "N features" in the console.
//  9. ClaheClipLimit = 4: more features again, but in a dark corner some are sensor noise, which
//     matches nothing. Watch the "features agree" count rather than the feature count.
// 10. Scan the same room with 13 and 13b and open both files: are the walls flatter (60 frames),
//     the edges sharper (2 cm cubes)? How much longer did E take?
//
//  New in 13c:
// 11. Walk away from the scanned part with the camera window open: watch the anchor counter fall
//     and the green dots thin out. Where does it turn red? That's how far you can go from here.
// 12. Face a scanned wall from 60 degrees to the side: the anchors drop, although the wall is in
//     view. ORB doesn't recognise a surface seen from too different an angle.
// 13. Set PlaceCandidates = 1, then 27: how long does placing take, and how often is a scan refused?
// 14. Follow only the Next line, scan after scan. How many scans until "these spots have seen it all"?
// 15. Exclude a good scan with X and export: the hole it leaves is what only that scan saw.
//
//  New in 13d:
// 16. Stand at the end of a long hallway and scan, then step 2 m in and scan again. The console
//     says how many far features agreed, and "turned by far rays": is the second scan red (distance
//     unknown)? Go back, step 0.5 m sideways and scan: landmarks appear (magenta). Now step in again.
// 17. Watch the magenta live anchors on a far doorway while you turn on the spot: they stay; walk
//     towards it: they thin out as the doorway gets bigger (ORB's 3x scale limit again).
// 18. Set MinParallaxDegrees = 0.5: more landmarks, but further off. Open the .obj and look at the
//     far_landmarks group: do the boxes sit on the far doorway, or scatter along the line of sight?
// 19. FarAgreeDegrees = 2: more far matches agree, including wrong ones. When does a scan go wrong?
//
//  Added (K, as in 13c):
// 20. Scan a hallway, press K, and open the folder's points.ply: the far doorway has no points
//     (no depth), but 13z's .obj of the same folder still has its far_landmarks boxes.
