// =====================================================================
//  Demo 3 · Booth Radar               (presentation; built from 06c)
//
//  One screen, two halves: on the left the camera, with a box, a number
//  and the distance on everyone and everything YOLO knows; on the right
//  the same scene from ABOVE, like a radar: each one a dot with a trail,
//  and the walls of the stand as grey lines.
//
//  Everything is 06c's (YOLO, Kalman tracking, depth for distance, walls
//  by RANSAC with a filter per wall). New here: both views in one
//  fullscreen window.
//
//  Set the Kinect up at chest height, roughly level (the walls are found
//  as "upright for the camera"). Nothing is saved.
//
//  Keys:   C  clear the trails      F  fullscreen on/off      Q or Esc  quit
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/registration.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video/tracking.hpp>   // cv::KalmanFilter

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// --- Detection, tracking and Kalman filter: same meaning as in example 05 ---
const std::string ModelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
const int ModelInputSize = 640;
const float MinConfidence = 0.40f;
const double BoxShrink = 0.5;
const int MinDepthSamples = 20;
const double MinOverlap = 0.3;
const int MaxMissedFrames = 12;
const double MeasurementNoisePixels = 15.0;
const double MeasurementNoiseMeters = 0.03;
const double AccelerationPixels = 500.0;
const double AccelerationMeters = 1.0;

// --- Radar (as in 06) ---
// How far the radar reaches, and its size on screen. Walls are often far away, so the depth
// cut-off is raised from 4.5 m to 6 m (07b); past 6 m almost nothing reads.
const float MaxDepthMeters = 6.0f;
const double RadarRangeMeters = 6.0;
const int RadarSizePixels = 700;

// How many past positions each object's trail keeps. At about 6 frames per second, 60 = 10 s.
const int TrailLength = 60;

// Only show these classes on the radar. Empty = show everything YOLO finds.
// Try { "person" } to see just people.
const std::vector<std::string> RadarClasses = {};

// --- Walls (as in 06b) ---
// The biggest upright planes to look at each frame. The camera rarely sees more than 3 walls.
const int MaxWalls = 4;

// RANSAC, as in 09: random planes to try, and how close (metres) a point must be to count as on it.
const int RansacTries = 200;
const float InlierThicknessMeters = 0.03f;

// Only every Nth depth pixel in each direction goes in (4 -> 1/16 of them), and RANSAC's three
// points lie within NeighbourPixels of each other (09: neighbours usually share a surface).
const int SampleStep = 4;
const int NeighbourPixels = 40;

// "Upright": the plane may lean at most this far from vertical, as the camera sees it.
const float MaxWallTiltDegrees = 20;

// A wall needs at least this share of the depth points, and must span at least this height.
// 0.8 m keeps a couch back out, but you see only about 1.15 m of a wall's height at 1 m away.
const double MinWallShare = 0.05;
const float MinWallHeightMeters = 0.8f;

// --- Wall filter (new): an extended Kalman filter per wall ---
// How much one frame's wall wobbles: its distance, its angle, and its ends (the ends jump more,
// e.g. when someone walks in front of the wall).
const double WallNoiseMeters = 0.02;
const double WallNoiseDegrees = 1.5;
const double WallEndNoiseMeters = 0.05;

// How far a wall may really move per frame. Walls don't, but the camera may get nudged.
// Smaller = smoother, but slower to follow when something did move.
const double WallDriftMeters = 0.002;

// A measured wall is the same as a tracked one when it's this close in distance and angle.
const double MatchMeters = 0.3;
const double MatchDegrees = 10;

// A wall is drawn once it's been seen ConfirmFrames frames (one-off surfaces never get there),
// and forgotten after ForgetFrames unseen (30 = about 5 s).
const int ConfirmFrames = 3;
const int ForgetFrames = 30;

// --- Output ---
const double FpsSmoothing = 0.9;
const int ReportEveryMs = 1000;

const bool StartFullscreen = true;
// The keys along the bottom of the screen, short (the user's ask).
const char* KeyLegend = "c - clear trails   f - fullscreen   q - quit";

// The 80 COCO classes yolo26n.pt was trained on, in classId order.
const char* ClassNames[] =
{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
};

struct Detection
{
    int classId;
    cv::Rect2d box;
    double rawMeters;   // 0 = no depth
};

// As in example 05, plus a trail of floor positions.
struct Track
{
    int id;
    int classId;
    cv::KalmanFilter kalman;     // state: centreX, centreY (px), metres, and their speeds
    cv::Size2d boxSize;
    int missedFrames = 0;
    std::deque<cv::Point2d> trail;   // past (X, Z) floor positions in metres, newest last

    double CentreX() const { return kalman.statePost.at<float>(0); }
    double CentreY() const { return kalman.statePost.at<float>(1); }
    double Meters() const { return kalman.statePost.at<float>(2); }
    cv::Rect2d Box() const
    {
        return cv::Rect2d(CentreX() - boxSize.width / 2, CentreY() - boxSize.height / 2, boxSize.width, boxSize.height);
    }
};

float Median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) return values[middle];
    return (values[middle - 1] + values[middle]) / 2;
}

double Overlap(const cv::Rect2d& a, const cv::Rect2d& b)
{
    double intersection = (a & b).area();
    return intersection / (a.area() + b.area() - intersection);
}

cv::Scalar ColorForId(int id)
{
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar((id * 47) % 180, 255, 255)), bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(c[0], c[1], c[2]);
}

// As in example 05.
cv::KalmanFilter NewKalman(double centreX, double centreY, double meters)
{
    cv::KalmanFilter kalman(6, 3, 0, CV_32F);
    kalman.measurementMatrix = cv::Mat::eye(3, 6, CV_32F);
    kalman.measurementNoiseCov = (cv::Mat_<float>(3, 3) <<
        MeasurementNoisePixels * MeasurementNoisePixels, 0, 0,
        0, MeasurementNoisePixels * MeasurementNoisePixels, 0,
        0, 0, MeasurementNoiseMeters * MeasurementNoiseMeters);
    kalman.statePost = (cv::Mat_<float>(6, 1) << centreX, centreY, meters, 0, 0, 0);
    double metersVariance = meters > 0 ? MeasurementNoiseMeters * MeasurementNoiseMeters : 100.0;
    kalman.errorCovPost = cv::Mat::diag((cv::Mat_<float>(6, 1) <<
        MeasurementNoisePixels * MeasurementNoisePixels, MeasurementNoisePixels * MeasurementNoisePixels,
        metersVariance, 1000 * 1000, 1000 * 1000, 2 * 2));
    return kalman;
}

// As in example 05.
void SetTimeStep(cv::KalmanFilter& kalman, double dt)
{
    cv::setIdentity(kalman.transitionMatrix);
    for (int i = 0; i < 3; i++) kalman.transitionMatrix.at<float>(i, i + 3) = (float)dt;
    double accelerations[3] = { AccelerationPixels, AccelerationPixels, AccelerationMeters };
    kalman.processNoiseCov = cv::Mat::zeros(6, 6, CV_32F);
    for (int i = 0; i < 3; i++)
    {
        double a = accelerations[i];
        kalman.processNoiseCov.at<float>(i, i) = (float)std::pow(a * dt * dt / 2, 2);
        kalman.processNoiseCov.at<float>(i + 3, i + 3) = (float)std::pow(a * dt, 2);
    }
}

bool ShownOnRadar(int classId)
{
    return RadarClasses.empty()
        || std::find(RadarClasses.begin(), RadarClasses.end(), ClassNames[classId]) != RadarClasses.end();
}

// A plane, as in 09: normal . point + distance = 0, with the normal pointing towards the camera.
// So distance = how far the camera is from the plane.
struct Plane
{
    cv::Vec3f normal;
    float distance;
    float DistanceTo(const cv::Vec3f& point) const { return normal.dot(point) + distance; }
};

// As in 09: the plane through three points; false if they're (nearly) in a line.
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

// As in 09: least-squares plane through many points (the direction they spread least in is the normal).
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

// A wall seen from above: a line on the floor, in the radar's X, Z metres.
struct Wall
{
    cv::Point2d ends[2];
    double meters;         // the camera's distance from the wall (at right angles to it)
    double degrees;        // 0 = facing the camera squarely, + = its right end is farther away
    double heightMeters;   // how much of its height is in view
};

// Value at a share (0-1) of the way through the sorted values: 0.5 = the median.
float Percentile(std::vector<float>& values, double share)
{
    size_t i = std::min(values.size() - 1, (size_t)(share * values.size()));
    std::nth_element(values.begin(), values.begin() + i, values.end());
    return values[i];
}

// Steps 10-12: the biggest upright planes in a grid of 3D points (NaN = no reading), as walls.
std::vector<Wall> FindWalls(const std::vector<cv::Vec3f>& grid, int gridColumns, std::mt19937& random)
{
    const int gridRows = (int)grid.size() / gridColumns, reach = NeighbourPixels / SampleStep;
    const float maxUpness = std::sin(MaxWallTiltDegrees * (float)CV_PI / 180);
    const cv::Vec3f cameraUp(0, -1, 0);   // the camera's Y axis points DOWN
    std::uniform_int_distribution<int> pickOffset(-reach, reach);

    // free[i]: cell i has a reading and no plane has taken it yet.
    std::vector<bool> free(grid.size());
    int readings = 0;
    for (size_t i = 0; i < grid.size(); i++)
    {
        free[i] = !std::isnan(grid[i][2]);
        if (free[i]) readings++;
    }

    std::vector<Wall> walls;
    std::vector<int> freeCells;
    for (int search = 0; search < MaxWalls; search++)
    {
        freeCells.clear();
        for (int i = 0; i < (int)grid.size(); i++) if (free[i]) freeCells.push_back(i);
        if (freeCells.size() < 3) break;

        // --- Step 10: RANSAC, as in 09 but for upright planes ---
        std::uniform_int_distribution<size_t> pickCell(0, freeCells.size() - 1);
        Plane best{};
        int bestCount = 0;
        for (int t = 0; t < RansacTries; t++)
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
            if (std::abs(plane.normal.dot(cameraUp)) > maxUpness) continue;   // floor, table top or ceiling
            int count = 0;
            for (int i : freeCells)
                if (std::abs(plane.DistanceTo(grid[i])) < InlierThicknessMeters) count++;
            if (count > bestCount) { bestCount = count; best = plane; }
        }
        if (bestCount < MinWallShare * readings) break;   // no big upright plane left

        // --- Step 11: keep its biggest connected piece, take that away for the next search, refit ---
        // A plane runs on for ever, so it also slices thin strips off the floor, the ceiling and
        // other walls. The wall itself is one connected piece in the image (as 08b's biggest blob).
        cv::Mat onPlane(gridRows, gridColumns, CV_8U, cv::Scalar(0)), labels, stats, centroids;
        for (int i : freeCells)
            if (std::abs(best.DistanceTo(grid[i])) < InlierThicknessMeters) onPlane.data[i] = 255;
        int pieces = cv::connectedComponentsWithStats(onPlane, labels, stats, centroids);
        int biggest = 1;
        for (int piece = 2; piece < pieces; piece++)
            if (stats.at<int>(piece, cv::CC_STAT_AREA) > stats.at<int>(biggest, cv::CC_STAT_AREA)) biggest = piece;
        if (pieces < 2 || stats.at<int>(biggest, cv::CC_STAT_AREA) < MinWallShare * readings) continue;
        std::vector<cv::Vec3f> inliers;
        for (int i : freeCells)
            if (labels.ptr<int>()[i] == biggest) { inliers.push_back(grid[i]); free[i] = false; }
        Plane plane = FitPlane(inliers);

        // --- Step 12: the wall as a line on the floor ---
        // Drop the normal's up/down part: what's left points from the wall to the camera, across
        // the floor. "along" runs along the wall; "foot" is the wall's nearest point to the camera.
        double across = std::hypot(plane.normal[0], plane.normal[2]);
        cv::Point2d normal(plane.normal[0] / across, plane.normal[2] / across), along(-normal.y, normal.x);
        // ponytail: drawn on the camera's own level, like 06's objects, so a camera tilted 10 deg
        // draws the walls 1.5% too far; 09's floor plane would give the true level.
        cv::Point2d foot = -normal * (plane.distance / across);

        // How far along the wall each point is, and its height. Percentiles ignore stray points.
        std::vector<float> alongs, heights;
        for (const cv::Vec3f& p : inliers)
        {
            alongs.push_back((float)(along.x * p[0] + along.y * p[2]));
            heights.push_back(p[1]);
        }
        float heightMeters = Percentile(heights, 0.98) - Percentile(heights, 0.02);
        if (heightMeters < MinWallHeightMeters) continue;   // too short: furniture, not a wall
        walls.push_back({ { foot + along * Percentile(alongs, 0.02), foot + along * Percentile(alongs, 0.98) },
                          plane.distance, std::atan2(normal.x, -normal.y) * 180 / CV_PI, heightMeters });
    }
    return walls;
}

// A tracked wall, with its own extended Kalman filter (new).
struct WallTrack
{
    int id;
    cv::Vec4d state;           // foot X, foot Z (metres), then each end's distance along the wall from the foot
    cv::Matx44d uncertainty;   // the filter's P: how unsure it is of the state (variances, metres squared)
    int seenFrames = 0;
    int missedFrames = 0;
    double rawMeters = 0;      // this frame's measured distance, to compare in the console

    cv::Point2d Foot() const { return { state[0], state[1] }; }
    double Meters() const { return cv::norm(Foot()); }
    double Degrees() const { return -std::atan2(state[0], state[1]) * 180 / CV_PI; }   // as 06b's: 0 = facing the camera
    cv::Point2d End(int i) const
    {
        cv::Point2d along(state[1] / Meters(), -state[0] / Meters());   // the foot direction turned 90 deg, as in 06b
        return Foot() + along * state[2 + i];
    }
};

// Step 13: a found wall as the 4 numbers the filter measures: the distance to its foot, the
// direction to the foot (radians, 0 = straight ahead, + = right), and how far along the wall each
// end is from the foot. (06b drew the line from foot + along * end, so this undoes that.)
// The distance is the plane's own, which a tilted camera doesn't change, so 06b's "1.5% too far
// at 10 deg" is gone for the walls (the objects are still drawn on the camera's level).
cv::Vec4d MeasureWall(const Wall& wall)
{
    cv::Point2d along = (wall.ends[1] - wall.ends[0]) / cv::norm(wall.ends[1] - wall.ends[0]);
    cv::Point2d foot = wall.ends[0] - along * wall.ends[0].dot(along);
    return { wall.meters, std::atan2(foot.x, foot.y), wall.ends[0].dot(along), wall.ends[1].dot(along) };
}

// Step 14: walls stand still, so the state stays; the filter only gets a little less sure of it.
// (05's transition matrix, with no speeds.)
void PredictWall(WallTrack& wall)
{
    wall.uncertainty += cv::Matx44d::eye() * (WallDriftMeters * WallDriftMeters);
}

// Step 16: the extended Kalman filter's correct step.
void CorrectWall(WallTrack& wall, const cv::Vec4d& measured)
{
    double x = wall.state[0], z = wall.state[1], r2 = x * x + z * z, r = std::sqrt(r2);

    // h(state): what the wall finder WOULD measure if the state were exactly right. The curved part.
    cv::Vec4d expected(r, std::atan2(x, z), wall.state[2], wall.state[3]);

    // H, the Jacobian: how much each measured number changes per metre of each state number, at
    // the current guess. Row 1: distance = hypot(x, z). Row 2: angle = atan2(x, z). The ends are
    // measured directly (1s). In 05 this was the plain identity matrix, the same everywhere.
    cv::Matx44d H(x / r,   z / r,  0, 0,
                  z / r2, -x / r2, 0, 0,
                  0,       0,      1, 0,
                  0,       0,      0, 1);
    double angleNoise = WallNoiseDegrees * CV_PI / 180;
    cv::Matx44d R = cv::Matx44d::diag(cv::Vec4d(WallNoiseMeters * WallNoiseMeters, angleNoise * angleNoise,
                                                WallEndNoiseMeters * WallEndNoiseMeters,
                                                WallEndNoiseMeters * WallEndNoiseMeters));

    // From here on it's the ordinary Kalman filter: the gain weighs the filter's own uncertainty
    // against the measurement's, and moves the state that share of the way.
    cv::Vec4d innovation = measured - expected;
    innovation[1] = std::remainder(innovation[1], 2 * CV_PI);   // -179 and +179 deg are 2 deg apart, not 358
    cv::Matx44d gain = wall.uncertainty * H.t() * (H * wall.uncertainty * H.t() + R).inv();
    wall.state += gain * innovation;
    wall.uncertainty = (cv::Matx44d::eye() - gain * H) * wall.uncertainty;
}

// Steps 14-16 for every wall: predict, match the found walls (closest pair first, as in 05),
// correct; start new walls, forget lost ones.
void UpdateWallTracks(std::vector<WallTrack>& tracked, const std::vector<Wall>& walls, int& nextId)
{
    for (WallTrack& wall : tracked) PredictWall(wall);

    std::vector<cv::Vec4d> measured;
    for (const Wall& wall : walls) measured.push_back(MeasureWall(wall));
    std::vector<bool> trackMatched(tracked.size(), false), wallMatched(walls.size(), false);
    while (true)
    {
        double bestCost = 1;   // 1 = right at the edge of MatchMeters or MatchDegrees
        int bestTrack = -1, bestWall = -1;
        for (size_t t = 0; t < tracked.size(); t++)
        {
            for (size_t w = 0; w < walls.size(); w++)
            {
                if (trackMatched[t] || wallMatched[w]) continue;
                double meters = std::abs(measured[w][0] - tracked[t].Meters());
                double degrees = std::abs(std::remainder(measured[w][1] * 180 / CV_PI + tracked[t].Degrees(), 360));
                double cost = std::max(meters / MatchMeters, degrees / MatchDegrees);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestTrack = (int)t;
                    bestWall = (int)w;
                }
            }
        }
        if (bestTrack < 0) break;
        trackMatched[bestTrack] = wallMatched[bestWall] = true;
        WallTrack& wall = tracked[bestTrack];
        CorrectWall(wall, measured[bestWall]);
        wall.seenFrames++;
        wall.missedFrames = 0;
        wall.rawMeters = measured[bestWall][0];
    }

    for (size_t t = 0; t < trackMatched.size(); t++)
        if (!trackMatched[t]) tracked[t].missedFrames++;
    // A wall not yet confirmed is forgotten at its first miss: a one-off surface, not a wall.
    tracked.erase(std::remove_if(tracked.begin(), tracked.end(), [](const WallTrack& wall)
                  { return wall.missedFrames > (wall.seenFrames >= ConfirmFrames ? ForgetFrames : 0); }),
                  tracked.end());

    // New walls: start very unsure (1 m either way), then take the first measurement.
    for (size_t w = 0; w < walls.size(); w++)
    {
        if (wallMatched[w]) continue;
        const cv::Vec4d& m = measured[w];
        WallTrack wall{ nextId++, cv::Vec4d(m[0] * std::sin(m[1]), m[0] * std::cos(m[1]), m[2], m[3]), cv::Matx44d::eye() };
        CorrectWall(wall, m);
        wall.seenFrames = 1;
        wall.rawMeters = m[0];
        tracked.push_back(wall);
    }
}

void ToggleFullscreen(const std::string& window)
{
    bool full = cv::getWindowProperty(window, cv::WND_PROP_FULLSCREEN) == cv::WINDOW_FULLSCREEN;
    cv::setWindowProperty(window, cv::WND_PROP_FULLSCREEN, full ? cv::WINDOW_NORMAL : cv::WINDOW_FULLSCREEN);
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // print immediately, even when piped (see example 01)

    // -----------------------------------------------------------------
    //  Setup: detector and Kinect (as in example 05)
    // -----------------------------------------------------------------

    if (!std::filesystem::exists(ModelPath))
    {
        std::printf("Model not found: %s\n", ModelPath.c_str());
        return 1;
    }
    cv::dnn::Net detector = cv::dnn::readNetFromONNX(ModelPath);

    libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Warning));
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device* device = freenect2.enumerateDevices() > 0 ? freenect2.openDefaultDevice() : nullptr;
    if (device == nullptr)
    {
        std::printf("No Kinect v2 found, or it is in use. See the main README's troubleshooting table.\n");
        return 1;
    }
    libfreenect2::Freenect2Device::Config config;   // the 6 m cut-off (07b), as in 06b
    config.MaxDepth = MaxDepthMeters;
    device->setConfiguration(config);
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Depth);
    device->setColorFrameListener(&listener);
    device->setIrAndDepthFrameListener(&listener);
    device->startStreams(true, true);

    const int colorWidth = 1920, colorHeight = 1080, depthWidth = 512, depthHeight = 424;
    libfreenect2::Freenect2Device::ColorCameraParams colorIntrinsics = device->getColorCameraParams();
    libfreenect2::Registration registration(device->getIrCameraParams(), colorIntrinsics);
    libfreenect2::Frame undistorted(depthWidth, depthHeight, 4), registered(depthWidth, depthHeight, 4);
    std::vector<int> colorIndexOfDepthPixel(depthWidth * depthHeight);

    // As in 06: the COLOUR camera's intrinsics, because the boxes live in the colour image.
    std::printf("Colour intrinsics: fx %.1f  fy %.1f  cx %.1f  cy %.1f  (pixels)\n",
                colorIntrinsics.fx, colorIntrinsics.fy, colorIntrinsics.cx, colorIntrinsics.cy);

    double modelScale = std::min((double)ModelInputSize / colorWidth, (double)ModelInputSize / colorHeight);
    int scaledWidth = (int)std::round(colorWidth * modelScale);
    int scaledHeight = (int)std::round(colorHeight * modelScale);
    int padLeft = (ModelInputSize - scaledWidth) / 2;
    int padTop = (ModelInputSize - scaledHeight) / 2;

    // The radar's layout. The camera sits at the bottom centre, looking up the screen.
    double pixelsPerMeter = (RadarSizePixels - 40) / RadarRangeMeters;
    cv::Point radarCamera(RadarSizePixels / 2, RadarSizePixels - 20);
    auto ToRadar = [&](double xMeters, double zMeters)
    {
        return cv::Point((int)std::lround(radarCamera.x + xMeters * pixelsPerMeter),
                         (int)std::lround(radarCamera.y - zMeters * pixelsPerMeter));
    };
    // Half the camera's horizontal field of view: the image edge is (width / 2) pixels from the
    // centre, at a focal length of fx pixels.
    double halfFovRadians = std::atan((colorWidth / 2.0) / colorIntrinsics.fx);
    std::printf("Colour camera's horizontal field of view: %.0f degrees\n", 2 * halfFovRadians * 180 / CV_PI);

    std::vector<Track> tracks;
    int nextId = 1;

    // As in 06b: the sparse grid of 3D points the walls are found in.
    const int gridColumns = depthWidth / SampleStep, gridRows = depthHeight / SampleStep;
    std::vector<cv::Vec3f> grid(gridColumns * gridRows);
    std::mt19937 random(1);

    // New: the walls being tracked, each with its own filter.
    std::vector<WallTrack> wallTracks;
    int nextWallId = 1;

    // One window: the camera scaled to the radar's height, then the radar (new).
    const std::string window = "Demo 3 - booth radar";
    const int cameraWidth = colorWidth * RadarSizePixels / colorHeight;
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, cameraWidth + RadarSizePixels, RadarSizePixels);
    if (StartFullscreen) ToggleFullscreen(window);

    auto reportStart = std::chrono::steady_clock::now();
    auto lastFrameTime = std::chrono::steady_clock::now();
    double liveFps = 0;
    cv::Mat frame;

    // -----------------------------------------------------------------
    //  Main operation
    // -----------------------------------------------------------------

    libfreenect2::FrameMap frames;
    while (true)
    {
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;   // 27 = Esc
        if (key == 'c') for (Track& track : tracks) track.trail.clear();
        if (key == 'f') ToggleFullscreen(window);

        // --- Steps 1-3: capture, register, detect, sample (as in example 05) ---
        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }
        libfreenect2::Frame* colorFrame = frames[libfreenect2::Frame::Color];
        cv::cvtColor(cv::Mat(colorHeight, colorWidth, CV_8UC4, colorFrame->data), frame, cv::COLOR_BGRA2BGR);
        registration.apply(colorFrame, frames[libfreenect2::Frame::Depth], &undistorted, &registered,
                           false, nullptr, colorIndexOfDepthPixel.data());
        const float* depthMm = (const float*)undistorted.data;
        listener.release(frames);

        // --- Steps 9-12: the walls (as in 06b) ---
        for (int row = 0; row < gridRows; row++)
        {
            for (int column = 0; column < gridColumns; column++)
            {
                float x, y, z;   // NaN = no reading
                registration.getPointXYZ(&undistorted, row * SampleStep, column * SampleStep, x, y, z);
                grid[row * gridColumns + column] = cv::Vec3f(x, y, z);
            }
        }
        std::vector<Wall> walls = FindWalls(grid, gridColumns, random);

        // --- Steps 13-16: filter the walls (new) ---
        UpdateWallTracks(wallTracks, walls, nextWallId);

        auto frameTime = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(frameTime - lastFrameTime).count();
        lastFrameTime = frameTime;

        cv::Mat scaled, letterboxed;
        cv::resize(frame, scaled, cv::Size(scaledWidth, scaledHeight));
        cv::copyMakeBorder(scaled, letterboxed, padTop, ModelInputSize - scaledHeight - padTop,
                           padLeft, ModelInputSize - scaledWidth - padLeft, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        detector.setInput(cv::dnn::blobFromImage(letterboxed, 1.0 / 255.0, cv::Size(ModelInputSize, ModelInputSize),
                                                 cv::Scalar(0, 0, 0), true, false));
        cv::Mat output = detector.forward();
        cv::Mat rows(output.size[1], output.size[2], CV_32F, output.ptr<float>());

        std::vector<Detection> detections;
        for (int row = 0; row < rows.rows; row++)
        {
            if (rows.at<float>(row, 4) < MinConfidence) continue;
            double left = (rows.at<float>(row, 0) - padLeft) / modelScale;
            double top = (rows.at<float>(row, 1) - padTop) / modelScale;
            double right = (rows.at<float>(row, 2) - padLeft) / modelScale;
            double bottom = (rows.at<float>(row, 3) - padTop) / modelScale;
            cv::Rect2d box(left, top, right - left, bottom - top);

            cv::Rect2d sample(left + box.width * (1 - BoxShrink) / 2, top + box.height * (1 - BoxShrink) / 2,
                              box.width * BoxShrink, box.height * BoxShrink);
            std::vector<float> samplesMm;
            for (int i = 0; i < depthWidth * depthHeight; i++)
            {
                int colorIndex = colorIndexOfDepthPixel[i];
                if (colorIndex < 0 || !(depthMm[i] > 0) || std::isinf(depthMm[i])) continue;
                if (sample.contains(cv::Point2d(colorIndex % colorWidth, colorIndex / colorWidth)))
                    samplesMm.push_back(depthMm[i]);
            }
            double rawMeters = (int)samplesMm.size() >= MinDepthSamples ? Median(samplesMm) / 1000.0 : 0;
            detections.push_back({ (int)rows.at<float>(row, 5), box, rawMeters });
        }

        // --- Steps 4-6: predict, match against predicted boxes, correct (as in example 05) ---
        for (Track& track : tracks)
        {
            SetTimeStep(track.kalman, dt);
            track.kalman.predict();
        }

        std::vector<bool> trackMatched(tracks.size(), false), detectionMatched(detections.size(), false);
        while (true)
        {
            double bestOverlap = MinOverlap;
            int bestTrack = -1, bestDetection = -1;
            for (size_t t = 0; t < tracks.size(); t++)
            {
                for (size_t d = 0; d < detections.size(); d++)
                {
                    if (trackMatched[t] || detectionMatched[d] || tracks[t].classId != detections[d].classId) continue;
                    double overlap = Overlap(tracks[t].Box(), detections[d].box);
                    if (overlap >= bestOverlap)
                    {
                        bestOverlap = overlap;
                        bestTrack = (int)t;
                        bestDetection = (int)d;
                    }
                }
            }
            if (bestTrack < 0) break;
            trackMatched[bestTrack] = detectionMatched[bestDetection] = true;

            Track& track = tracks[bestTrack];
            const Detection& detection = detections[bestDetection];
            double measuredMeters = detection.rawMeters;
            bool haveDepth = measuredMeters > 0;
            if (!haveDepth)
            {
                measuredMeters = track.Meters();
                track.kalman.measurementNoiseCov.at<float>(2, 2) = 1e6f;
            }
            track.kalman.correct((cv::Mat_<float>(3, 1) << detection.box.x + detection.box.width / 2,
                                  detection.box.y + detection.box.height / 2, measuredMeters));
            track.kalman.measurementNoiseCov.at<float>(2, 2) = (float)(MeasurementNoiseMeters * MeasurementNoiseMeters);
            track.boxSize = detection.box.size();
            track.missedFrames = 0;
        }

        for (size_t d = 0; d < detections.size(); d++)
        {
            if (detectionMatched[d]) continue;
            const Detection& detection = detections[d];
            Track track;
            track.id = nextId++;
            track.classId = detection.classId;
            track.boxSize = detection.box.size();
            track.kalman = NewKalman(detection.box.x + detection.box.width / 2, detection.box.y + detection.box.height / 2,
                                     detection.rawMeters);
            tracks.push_back(std::move(track));
        }

        for (size_t t = 0; t < trackMatched.size(); t++)
        {
            if (!trackMatched[t]) tracks[t].missedFrames++;
        }
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                    [](const Track& t) { return t.missedFrames > MaxMissedFrames; }),
                     tracks.end());

        // --- Step 7: Deproject each object to a floor position (as in 06) -------------
        // Pinhole model (§5.3) on the filtered estimate: how far left/right of the image centre the
        // box centre is, in pixels, times distance / focal length = metres left/right of the camera.
        // ponytail: uses the colour camera as the origin (it sits a few cm beside the depth camera),
        // and ignores lens distortion; both are small next to box jitter. Example 09 adds the floor.
        for (Track& track : tracks)
        {
            if (track.Meters() <= 0) continue;   // no distance yet, so no position
            double zMeters = track.Meters();
            double xMeters = (track.CentreX() - colorIntrinsics.cx) * zMeters / colorIntrinsics.fx;
            track.trail.push_back(cv::Point2d(xMeters, zMeters));
            if ((int)track.trail.size() > TrailLength) track.trail.pop_front();
        }

        // -----------------------------------------------------------------
        //  Output: the camera view (as in example 05)
        // -----------------------------------------------------------------

        for (const Track& track : tracks)
        {
            bool coasting = track.missedFrames > 0;
            cv::Scalar color = ColorForId(track.id);
            std::string label = "#" + std::to_string(track.id) + " " + ClassNames[track.classId]
                              + (track.Meters() > 0 ? cv::format(" %.2fm", track.Meters()) : "");
            cv::rectangle(frame, track.Box(), color, coasting ? 1 : 3);
            cv::putText(frame, label, cv::Point((int)track.Box().x, std::max((int)track.Box().y - 10, 30)),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, color, coasting ? 1 : 2);
        }
        liveFps = (liveFps == 0) ? 1.0 / dt : FpsSmoothing * liveFps + (1 - FpsSmoothing) * (1.0 / dt);
        cv::putText(frame, cv::format("%.1f fps", liveFps), cv::Point(10, frame.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

        // -----------------------------------------------------------------
        //  Step 8 + Output: the radar, seen from above (as in 06b; filtered walls new)
        // -----------------------------------------------------------------

        cv::Mat radar(RadarSizePixels, RadarSizePixels, CV_8UC3, cv::Scalar(25, 25, 25));
        cv::Scalar grid(80, 80, 80);

        // A ring every metre, labelled with its distance.
        for (int meters = 1; meters <= (int)RadarRangeMeters; meters++)
        {
            int radius = (int)(meters * pixelsPerMeter);
            cv::ellipse(radar, radarCamera, cv::Size(radius, radius), 0, 180, 360, grid, 1);
            cv::putText(radar, std::to_string(meters) + " m", radarCamera + cv::Point(4, -radius - 4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, grid, 1);
        }
        // The two edges of what the camera can see, and the 0.5 m minimum range (guide §2).
        for (int side : { -1, 1 })
        {
            cv::line(radar, radarCamera,
                     ToRadar(side * RadarRangeMeters * std::sin(halfFovRadians), RadarRangeMeters * std::cos(halfFovRadians)),
                     cv::Scalar(0, 120, 160), 1);
        }
        cv::ellipse(radar, radarCamera, cv::Size((int)(0.5 * pixelsPerMeter), (int)(0.5 * pixelsPerMeter)),
                    0, 180, 360, cv::Scalar(0, 0, 140), 1);
        cv::drawMarker(radar, radarCamera, cv::Scalar(0, 255, 255), cv::MARKER_TRIANGLE_UP, 14, 2);

        // The walls, under the objects: this frame's raw ones thin and dark (as found in 06b), the
        // filtered ones thick and grey (new), thinner while not seen.
        // ponytail: depth X comes from the depth camera, object X from the colour camera; they sit
        // about 5 cm apart, so walls and objects can be off by that much.
        for (const Wall& wall : walls)
        {
            cv::line(radar, ToRadar(wall.ends[0].x, wall.ends[0].y), ToRadar(wall.ends[1].x, wall.ends[1].y),
                     cv::Scalar(100, 100, 100), 1);
        }
        for (const WallTrack& wall : wallTracks)
        {
            if (wall.seenFrames < ConfirmFrames) continue;
            cv::Point a = ToRadar(wall.End(0).x, wall.End(0).y), b = ToRadar(wall.End(1).x, wall.End(1).y);
            cv::line(radar, a, b, cv::Scalar(190, 190, 190), wall.missedFrames > 0 ? 2 : 4);
            cv::putText(radar, cv::format("wall #%d %.2f m", wall.id, wall.Meters()), (a + b) / 2 + cv::Point(6, -6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(190, 190, 190), 1);
        }

        for (const Track& track : tracks)
        {
            if (track.trail.empty() || !ShownOnRadar(track.classId)) continue;
            cv::Scalar color = ColorForId(track.id);

            // The trail: where this object has been, oldest first.
            for (size_t i = 1; i < track.trail.size(); i++)
            {
                cv::line(radar, ToRadar(track.trail[i - 1].x, track.trail[i - 1].y),
                         ToRadar(track.trail[i].x, track.trail[i].y), color, 1);
            }

            // The object now: a filled dot if measured this frame, hollow if only predicted.
            cv::Point now = ToRadar(track.trail.back().x, track.trail.back().y);
            cv::circle(radar, now, 7, color, track.missedFrames > 0 ? 1 : cv::FILLED);
            cv::putText(radar, "#" + std::to_string(track.id) + " " + ClassNames[track.classId],
                        now + cv::Point(10, 4), cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1);
        }
        cv::putText(radar, "seen from above", cv::Point(12, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        cv::putText(radar, "live only: nothing is recorded", cv::Point(12, 55), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(160, 160, 160), 1);
        cv::Mat camera, both;
        cv::resize(frame, camera, cv::Size(cameraWidth, RadarSizePixels));
        cv::hconcat(camera, radar, both);
        // The keys, in a strip under the pictures (new: the user's ask).
        cv::Mat keys(36, both.cols, CV_8UC3, cv::Scalar(20, 20, 20));
        cv::putText(keys, KeyLegend, cv::Point(12, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(220, 220, 220), 1);
        cv::vconcat(both, keys, both);
        cv::imshow(window, both);

        if (frameTime - reportStart >= std::chrono::milliseconds(ReportEveryMs))
        {
            std::printf("%.1f fps | %d objects tracked | %d walls found this frame, %d tracked\n", liveFps,
                        (int)tracks.size(), (int)walls.size(), (int)wallTracks.size());
            for (const WallTrack& wall : wallTracks)
            {
                if (wall.seenFrames < ConfirmFrames) continue;
                std::printf("  wall #%-2d %.3f m away (%s), %.2f m long, turned %+.1f deg, seen %d frames\n",
                            wall.id, wall.Meters(),
                            wall.missedFrames > 0 ? "not seen this frame" : cv::format("raw %.3f", wall.rawMeters).c_str(),
                            cv::norm(wall.End(1) - wall.End(0)), wall.Degrees(), wall.seenFrames);
            }
            for (const Track& track : tracks)
            {
                if (track.trail.empty() || !ShownOnRadar(track.classId)) continue;
                cv::Point2d p = track.trail.back();
                // Straight-line distance across the floor: Pythagoras on X and Z.
                std::printf("  #%-3d %-12s X %+.2f m (%s), Z %.2f m ahead | straight-line %.2f m%s\n",
                            track.id, ClassNames[track.classId], p.x, p.x < 0 ? "left" : "right", p.y,
                            std::hypot(p.x, p.y), track.missedFrames > 0 ? "  (predicted)" : "");
            }
            reportStart = frameTime;
        }
    }

    device->stop();
    device->close();
    cv::destroyAllWindows();
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  (06, 06b and 06c's lists cover the method. These are for showing it.)
//  1. Ask a visitor to walk a circle: the trail on the radar draws it. C clears it for the next one.
//  2. Two visitors walk past each other: do they keep their numbers? (05's Kalman predicts both.)
//  3. Busy stand? Set RadarClasses = { "person" } so chairs and bottles don't clutter the radar.
//  4. Hold up a cup, a phone, a book: YOLO names 80 kinds of thing, each with its distance.
