// =====================================================================
//  Demo 3 · Booth Radar               (presentation; built from 06c)
//
//  One screen, two halves: on the left the camera, with a box, a number
//  and the distance on everyone and everything YOLO knows; on the right
//  the same scene from ABOVE, like a radar: each one a dot with a trail,
//  and the walls of the stand as grey lines.
//
//  Everything is 06c's (YOLO, Kalman tracking, depth for distance, walls
//  by RANSAC with an extended Kalman filter per wall; copied, the demos
//  stand on their own). New here: both views in one fullscreen window,
//  and RadarClasses to show only some kinds of thing on the radar.
//
//  Set the Kinect up at chest height, roughly level (the walls are found
//  as "upright for the camera"). Nothing is saved.
//
//  Keys:   C  clear the trails      F  fullscreen on/off      Q or Esc  quit
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/radar.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/track.hpp"
#include "spotter/view.hpp"
#include "spotter/walls.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "C  clear the trails   F  fullscreen   Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

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

// A wall is drawn once seen this many frames; a confirmed wall is forgotten after this many missed.
const int ConfirmFrames = 3;
const int ForgetFrames = 30;

// Only show these classes on the radar. Empty = everything YOLO finds. On a busy stand try
// { "person" }, so chairs and bottles don't clutter it.
const std::vector<std::string> RadarClasses = {};

const int ReportEveryMs = 1000;
const bool StartFullscreen = true;
const std::string Window = "Demo 3 - booth radar";

spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = 0.15f;
    return options;
}

// ---------------------------------------------------------------------
//  The new idea: an extended Kalman filter per wall
// ---------------------------------------------------------------------

struct WallTrack
{
    int id;
    cv::Vec4d state;           // foot X, foot Z (metres), then each end's distance along the wall from the foot
    cv::Matx44d uncertainty;   // the filter's P: how unsure it is of the state (variances)
    int seenFrames = 0, missedFrames = 0;
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

struct WallTracks
{
    std::vector<WallTrack> list;
    int nextId = 1;
};

// Step 12: a found wall as the 4 numbers the filter measures: the distance to its foot, the
// direction to the foot (radians, 0 = straight ahead, + = right), and how far along the wall each
// end is from the foot. False for a wall too short to have a direction.
bool MeasureWall(const spotter::Wall& wall, cv::Vec4d& measured)
{
    double length = cv::norm(wall.ends[1] - wall.ends[0]);
    if (length < 1e-3) return false;   // a wall of length 0 would put NaN in the filter for good (review bug 9)
    cv::Point2d along = (wall.ends[1] - wall.ends[0]) / length;
    cv::Point2d foot = wall.ends[0] - along * wall.ends[0].dot(along);
    measured = { wall.meters, std::atan2(foot.x, foot.y), wall.ends[0].dot(along), wall.ends[1].dot(along) };
    return true;
}

// Step 13: walls stand still, so the state stays; the filter only gets a little less sure of it.
void PredictWall(WallTrack& wall) { wall.uncertainty += cv::Matx44d::eye() * (WallDriftMeters * WallDriftMeters); }

// Step 15: the extended Kalman filter's correct step.
void CorrectWall(WallTrack& wall, const cv::Vec4d& measured)
{
    double x = wall.state[0], z = wall.state[1], r2 = x * x + z * z, r = std::sqrt(r2);

    // h(state): what the wall finder WOULD measure if the state were exactly right. The curved part.
    cv::Vec4d expected(r, std::atan2(x, z), wall.state[2], wall.state[3]);

    // H, the Jacobian: how much each measured number changes per metre of each state number, at the
    // current guess. Row 1: distance = hypot(x, z). Row 2: angle = atan2(x, z). The ends are measured
    // directly (1s). In 05 this was the plain identity matrix, the same everywhere.
    cv::Matx44d H(x / r,   z / r,  0, 0,
                  z / r2, -x / r2, 0, 0,
                  0,       0,      1, 0,
                  0,       0,      0, 1);
    double angleNoise = WallNoiseDegrees * CV_PI / 180;
    cv::Matx44d R = cv::Matx44d::diag(cv::Vec4d(WallNoiseMeters * WallNoiseMeters, angleNoise * angleNoise,
                                                WallEndNoiseMeters * WallEndNoiseMeters, WallEndNoiseMeters * WallEndNoiseMeters));

    // From here on it's the ordinary Kalman filter: the gain weighs the filter's own uncertainty
    // against the measurement's, and moves the state that share of the way.
    cv::Vec4d innovation = measured - expected;
    innovation[1] = std::remainder(innovation[1], 2 * CV_PI);   // -179 and +179 deg are 2 deg apart, not 358
    cv::Matx44d gain = wall.uncertainty * H.t() * (H * wall.uncertainty * H.t() + R).inv();
    wall.state += gain * innovation;
    // The Joseph form of P = (I - KH) P: the same in exact maths, but it stays symmetric and
    // positive with rounding errors (review bug 9).
    cv::Matx44d keep = cv::Matx44d::eye() - gain * H;
    wall.uncertainty = keep * wall.uncertainty * keep.t() + gain * R * gain.t();
}

// Steps 13-15 for every wall: predict, match the found walls (closest pair first, as in 05),
// correct; start new walls, forget lost ones.
void UpdateWallTracks(WallTracks& tracks, const std::vector<spotter::Wall>& walls)
{
    for (WallTrack& wall : tracks.list) PredictWall(wall);
    std::vector<cv::Vec4d> measured;
    for (const spotter::Wall& wall : walls)
    {
        cv::Vec4d m;
        if (MeasureWall(wall, m)) measured.push_back(m);
    }
    std::vector<bool> trackMatched(tracks.list.size(), false), wallMatched(measured.size(), false);
    while (true)
    {
        double bestCost = 1;   // 1 = right at the edge of MatchMeters or MatchDegrees
        int bestTrack = -1, bestWall = -1;
        for (size_t t = 0; t < tracks.list.size(); t++)
            for (size_t w = 0; w < measured.size(); w++)
            {
                if (trackMatched[t] || wallMatched[w]) continue;
                double meters = std::abs(measured[w][0] - tracks.list[t].Meters());
                double degrees = std::abs(std::remainder(measured[w][1] * 180 / CV_PI + tracks.list[t].Degrees(), 360));
                double cost = std::max(meters / MatchMeters, degrees / MatchDegrees);
                if (cost < bestCost) { bestCost = cost; bestTrack = (int)t; bestWall = (int)w; }
            }
        if (bestTrack < 0) break;
        trackMatched[bestTrack] = wallMatched[bestWall] = true;
        WallTrack& wall = tracks.list[bestTrack];
        CorrectWall(wall, measured[bestWall]);
        wall.seenFrames++;
        wall.missedFrames = 0;
        wall.rawMeters = measured[bestWall][0];
    }
    for (size_t t = 0; t < trackMatched.size(); t++) if (!trackMatched[t]) tracks.list[t].missedFrames++;
    // A wall not yet confirmed is forgotten at its first miss: a one-off surface, not a wall.
    tracks.list.erase(std::remove_if(tracks.list.begin(), tracks.list.end(), [](const WallTrack& wall)
                      { return wall.missedFrames > (wall.seenFrames >= ConfirmFrames ? ForgetFrames : 0); }), tracks.list.end());

    // New walls: start very unsure (1 m either way), then take the first measurement.
    for (size_t w = 0; w < measured.size(); w++)
    {
        if (wallMatched[w]) continue;
        const cv::Vec4d& m = measured[w];
        WallTrack wall{ tracks.nextId++, cv::Vec4d(m[0] * std::sin(m[1]), m[0] * std::cos(m[1]), m[2], m[3]), cv::Matx44d::eye() };
        CorrectWall(wall, m);
        wall.seenFrames = 1;
        wall.rawMeters = m[0];
        tracks.list.push_back(wall);
    }
}

// Steps 1-7, as in 06: YOLO on its own thread, the tracker, and each object's trail on the floor.
void TrackObjects(const spotter::RgbdFrame& frame, double dt, spotter::AsyncDetector& detector, spotter::Tracker& tracker,
                  spotter::BoxDepthSampler& boxDepth, spotter::TrackTrails& trails)
{
    boxDepth.Build(frame);
    tracker.Predict(dt);
    if (auto batch = detector.TryTake()) tracker.Match(*batch, boxDepth);
    detector.Offer(frame.colour);
    tracker.Remeasure(boxDepth);
    trails.Update(tracker, frame.colourCamera);
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

bool OnRadar(const std::string& name) { return RadarClasses.empty() || std::find(RadarClasses.begin(), RadarClasses.end(), name) != RadarClasses.end(); }

// The camera half: a box, a number and the distance on each tracked thing.
void DrawCamera(cv::Mat& picture, const spotter::Tracker& tracker, const spotter::ClassList& classes, const spotter::FpsMeter& fps)
{
    for (const spotter::Track& track : tracker.Tracks())
    {
        cv::Scalar colour = spotter::ColourForId(track.id);
        double meters = track.filter.Meters();
        std::string label = "#" + std::to_string(track.id) + " " + classes.Name(track.classId) + (meters > 0 ? cv::format(" %.2fm", meters) : "");
        cv::rectangle(picture, track.Box(), colour, track.Coasting() ? 1 : 3);
        cv::putText(picture, label, cv::Point((int)track.Box().x, std::max((int)track.Box().y - 10, 30)), cv::FONT_HERSHEY_SIMPLEX, 1.0, colour,
                    track.Coasting() ? 1 : 2);
    }
    fps.Draw(picture);
}

// The radar half, then both halves in one window (new).
void Show(cv::Mat& camera, spotter::RadarView& radar, const std::vector<spotter::Wall>& walls, const WallTracks& wallTracks, const spotter::Tracker& tracker,
          const spotter::TrackTrails& trails, const spotter::ClassList& classes, const spotter::PinholeCamera& colourCamera)
{
    radar.Begin(colourCamera);
    for (const spotter::Wall& wall : walls) radar.DrawLine(wall.ends[0], wall.ends[1], cv::Scalar(100, 100, 100), 1);   // raw, as 06b found them
    for (const WallTrack& wall : wallTracks.list)
        if (wall.seenFrames >= ConfirmFrames)
            radar.DrawLine(wall.End(0), wall.End(1), cv::Scalar(190, 190, 190), wall.missedFrames > 0 ? 2 : 4, cv::format("wall #%d %.2f m", wall.id, wall.Meters()));
    for (const spotter::Track& track : tracker.Tracks())
        if (OnRadar(classes.Name(track.classId)))
            radar.DrawTrail(trails.Of(track.id), spotter::ColourForId(track.id), track.Coasting(), "#" + std::to_string(track.id) + " " + classes.Name(track.classId));
    cv::Mat& map = radar.Image();
    cv::putText(map, "seen from above", cv::Point(12, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::putText(map, "live only: nothing is recorded", cv::Point(12, 55), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(160, 160, 160), 1);
    cv::Mat left, both;
    cv::resize(camera, left, cv::Size(camera.cols * map.rows / camera.rows, map.rows));
    cv::hconcat(left, map, both);
    cv::imshow(Window, spotter::WithKeys(both, Keys));
}

void Report(const std::vector<spotter::Wall>& walls, const WallTracks& wallTracks, const spotter::Tracker& tracker)
{
    std::printf("%zu objects | %zu walls found this frame, %zu tracked\n", tracker.Tracks().size(), walls.size(), wallTracks.list.size());
    for (const WallTrack& wall : wallTracks.list)
        if (wall.seenFrames >= ConfirmFrames)
            std::printf("  wall #%-2d %.3f m away (%s), %.2f m long, turned %+.1f deg, seen %d frames\n", wall.id, wall.Meters(),
                        wall.missedFrames > 0 ? "not seen this frame" : cv::format("raw %.3f", wall.rawMeters).c_str(),
                        cv::norm(wall.End(1) - wall.End(0)), wall.Degrees(), wall.seenFrames);
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    spotter::AsyncDetector detector(yolo);
    spotter::Tracker tracker(spotter::Tracker::Options{});
    spotter::BoxDepthSampler boxDepth;
    spotter::TrackTrails trails;
    spotter::RadarView radar;
    WallTracks wallTracks;
    std::mt19937 random(1);
    spotter::OpenExhibitWindow(Window, cv::Size(1944, 736), StartFullscreen);
    spotter::FpsMeter fps;
    spotter::Every report(ReportEveryMs);
    spotter::FrameTimer timer;
    spotter::RgbdFrame frame;

    while (kinect.Next(frame))
    {
        TrackObjects(frame, timer.Step(frame), detector, tracker, boxDepth, trails);   // steps 1-7
        std::vector<spotter::Wall> walls = spotter::FindWalls(spotter::ThinCloud(frame.depthCamera.Cloud(frame.depthMm), 512, 4), spotter::WallSearch(), random);
        UpdateWallTracks(wallTracks, walls);                                  // steps 12-15

        fps.Tick();
        DrawCamera(frame.colour, tracker, yolo.Classes(), fps);
        Show(frame.colour, radar, walls, wallTracks, tracker, trails, yolo.Classes(), frame.colourCamera);
        if (report.Due()) Report(walls, wallTracks, tracker);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 'c') trails.Clear();
        if (key == 'f') spotter::ToggleFullscreen(Window);
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas (06, 06b and 06c's lists cover the method; these are for showing it)
// ---------------------------------------------------------------------
//  1. Ask a visitor to walk a circle: the trail on the radar draws it. C clears it for the next one.
//  2. Two visitors walk past each other: do they keep their numbers? (05's Kalman predicts both.)
//  3. Busy stand? Set RadarClasses = { "person" } so chairs and bottles don't clutter the radar.
//  4. Hold up a cup, a phone, a book: YOLO names 80 kinds of thing, each with its distance.
