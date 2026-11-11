// =====================================================================
//  06c · Wall Filter
//
//  Example 06b finds the walls afresh every frame, so they wobble: RANSAC
//  is random, and the depth is noisy. But walls don't move. This one gives
//  each wall its own EXTENDED Kalman filter (EKF), the way 05 gave each
//  object a Kalman filter:
//
//    1-11. Objects, radar, walls                as in 06b (library: spotter::FindWalls)
//    12.   Measure    each wall found this frame becomes 4 numbers: its
//                     distance, the direction to its nearest point (the
//                     "foot"), and how far along the wall its two ends are.
//    13.   Predict    walls stand still: same state, a bit less sure.
//    14.   Match      each measured wall to the tracked wall nearest in
//                     distance and direction (as 05 matched boxes).
//    15.   Correct    nudge the tracked wall towards the measurement, by
//                     how sure each one is.
//
//  Why EXTENDED: the filter keeps the foot as a point on the floor, X and Z
//  in metres (later, when the camera moves, a point is what you can move).
//  But the wall finder measures a distance and an angle, like a radar. Going
//  from X, Z to distance and angle is curved maths (hypot, atan2), and a
//  plain Kalman filter only handles straight-line maths (05's measurement
//  was the state itself). The EKF straightens the curve out around the
//  current guess, every frame: that slope table is the Jacobian, H.
//
//  Windows: the camera view, and the radar: this frame's raw walls thin and
//           dark, the filtered walls thick and grey, objects with trails.
//  Keys:   C  clear the trails      Q or Esc  quit
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
const std::string Keys = "C  clear the trails      Q or Esc  quit";

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

const int ReportEveryMs = 1000;
const std::string CameraWindow = "06c camera  (C clears trails, Q quits)", RadarWindow = "06c radar";

//spotter::YoloDetector::Options YoloSettings(){
//
//@brief - the detector's options: yolo26n.onnx and its names file from the repo, minimum confidence 0.15 (the
//         tracker's weak threshold, 05b). O(1).
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

    //cv::Point2d Foot(){
    //
    //@brief - the wall's foot (its nearest point to the camera) as (X right, Z ahead): the first two state numbers.
    //         O(1).
    cv::Point2d Foot() const { return { state[0], state[1] }; }
    //double Meters(){
    //
    //@brief - the camera's distance from the wall: the length of the foot vector. O(1).
    double Meters() const { return cv::norm(Foot()); }
    //double Degrees(){
    //
    //@brief - the wall's angle as in 06b: 0 = facing the camera squarely, from the foot's direction. O(1).
    double Degrees() const { return -std::atan2(state[0], state[1]) * 180 / CV_PI; }   // as 06b's: 0 = facing the camera
    //cv::Point2d End(int i){
    //
    //@param1 - i: 0 or 1: which end of the wall
    //
    //@brief - that end's spot on the floor: the foot plus the end's distance along the wall (the foot direction turned
    //         90 degrees, as in 06b). O(1).
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

//bool MeasureWall(const spotter::Wall& wall, cv::Vec4d& measured){
//
//@param1 - wall: a wall found by FindWalls
//@param2 - measured: set to the 4 numbers the filter measures
//
//@brief - step 12: a found wall as the 4 numbers the filter measures: the distance to its foot, the direction to
//         the foot (radians, 0 = straight ahead, + = right) and how far along the wall each end is from the foot.
//         False for a wall too short (under 1 mm) to have a direction: it would put NaN in the filter for good
//         (review bug 9). O(1).
bool MeasureWall(const spotter::Wall& wall, cv::Vec4d& measured)
{
    double length = cv::norm(wall.ends[1] - wall.ends[0]);
    if (length < 1e-3) return false;   // a wall of length 0 would put NaN in the filter for good (review bug 9)
    cv::Point2d along = (wall.ends[1] - wall.ends[0]) / length;
    cv::Point2d foot = wall.ends[0] - along * wall.ends[0].dot(along);
    measured = { wall.meters, std::atan2(foot.x, foot.y), wall.ends[0].dot(along), wall.ends[1].dot(along) };
    return true;
}

//void PredictWall(WallTrack& wall){
//
//@param1 - wall: the wall's filter
//
//@brief - the filter's predict step: a wall doesn't move, so the state stays and the uncertainty grows by
//         WallDriftMeters squared on every diagonal entry. O(1).
void PredictWall(WallTrack& wall) { wall.uncertainty += cv::Matx44d::eye() * (WallDriftMeters * WallDriftMeters); }

//void CorrectWall(WallTrack& wall, const cv::Vec4d& measured){
//
//@param1 - wall: the wall's filter; its state and uncertainty are updated
//@param2 - measured: the 4 numbers from MeasureWall
//
//@brief - step 15: the extended Kalman filter's correct step. h(state) is what the wall finder would measure if
//         the state were exactly right; H is its Jacobian (distance = hypot(x, z), angle = atan2(x, z), the ends
//         measured directly); the angle difference is wrapped so -179 and +179 degrees are 2 apart; then the
//         ordinary Kalman gain moves the state that share of the way, and the uncertainty is updated in the Joseph
//         form (stays symmetric and positive despite rounding, review bug 9). O(1) (4 x 4 matrices).
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

//void UpdateWallTracks(WallTracks& tracks, const std::vector<spotter::Wall>& walls){
//
//@param1 - tracks: the wall filters; updated in place
//@param2 - walls: the walls found this frame
//
//@brief - steps 13-15 for every wall: predict, match the found walls (closest pair first, as in 05, within
//         MatchMeters and MatchDegrees), correct. A wall not yet confirmed is forgotten at its first miss, a
//         confirmed one after ForgetFrames; unmatched walls start new filters, very unsure (1 m either way). O(T x
//         W x min(T, W)).
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

//void TrackObjects(const spotter::RgbdFrame& frame, double dt, spotter::AsyncDetector& detector, spotter::Tracker& tracker, spotter::BoxDepthSampler& boxDepth, spotter::TrackTrails& trails){
//
//@param1 - frame: this frame
//@param2 - dt: seconds since the last frame
//@param3 - detector: YOLO on its own thread
//@param4 - tracker: the tracker
//@param5 - boxDepth: the depth lookup, rebuilt here
//@param6 - trails: each object's floor trail
//
//@brief - steps 1-7, as in 06: build the depth lookup, predict, take a finished YOLO batch if any and match it,
//         offer the new picture, re-measure distances, and add each object's floor spot to its trail. O(Build +
//         tracks).
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

//void ShowCamera(cv::Mat& picture, const spotter::Tracker& tracker, const spotter::ClassList& classes, const spotter::FpsMeter& fps){
//
//@param1 - picture: the colour picture; drawn on
//@param2 - tracker: the tracked objects
//@param3 - classes: the class names
//@param4 - fps: the frame-rate meter
//
//@brief - a box per track in its ID's colour (thin while coasting) with its ID and class, and the fps; shown with
//         the key strip. O(tracks).
void ShowCamera(cv::Mat& picture, const spotter::Tracker& tracker, const spotter::ClassList& classes, const spotter::FpsMeter& fps)
{
    for (const spotter::Track& track : tracker.Tracks())
    {
        cv::Scalar colour = spotter::ColourForId(track.id);
        cv::rectangle(picture, track.Box(), colour, track.Coasting() ? 1 : 3);
        cv::putText(picture, "#" + std::to_string(track.id) + " " + classes.Name(track.classId), track.Box().tl() + cv::Point2d(4, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, colour, 2);
    }
    fps.Draw(picture);
    cv::imshow(CameraWindow, spotter::WithKeys(picture, Keys));
}

//void ShowRadar(spotter::RadarView& radar, const std::vector<spotter::Wall>& walls, const WallTracks& wallTracks, const spotter::Tracker& tracker, const spotter::TrackTrails& trails, const spotter::ClassList& classes, const spotter::PinholeCamera& colourCamera){
//
//@param1 - radar: the bird's-eye map
//@param2 - walls: the raw walls found this frame
//@param3 - wallTracks: the filtered walls
//@param4 - tracker: the tracked objects
//@param5 - trails: each object's floor trail
//@param6 - classes: the class names
//@param7 - colourCamera: the colour camera's model
//
//@brief - a fresh radar map: the raw walls thin and dark grey (as 06b found them), each confirmed filtered wall
//         thick and light (thinner when not seen this frame) with its label, then each object's trail; shown with
//         the key strip. O(pixels + trail points).
void ShowRadar(spotter::RadarView& radar, const std::vector<spotter::Wall>& walls, const WallTracks& wallTracks, const spotter::Tracker& tracker,
               const spotter::TrackTrails& trails, const spotter::ClassList& classes, const spotter::PinholeCamera& colourCamera)
{
    radar.Begin(colourCamera);
    for (const spotter::Wall& wall : walls) radar.DrawLine(wall.ends[0], wall.ends[1], cv::Scalar(100, 100, 100), 1);   // raw, as 06b found them
    for (const WallTrack& wall : wallTracks.list)
        if (wall.seenFrames >= ConfirmFrames)
            radar.DrawLine(wall.End(0), wall.End(1), cv::Scalar(190, 190, 190), wall.missedFrames > 0 ? 2 : 4, cv::format("wall #%d %.2f m", wall.id, wall.Meters()));
    for (const spotter::Track& track : tracker.Tracks())
        radar.DrawTrail(trails.Of(track.id), spotter::ColourForId(track.id), track.Coasting(), "#" + std::to_string(track.id) + " " + classes.Name(track.classId));
    cv::imshow(RadarWindow, spotter::WithKeys(radar.Image(), Keys));
}

//void Report(const std::vector<spotter::Wall>& walls, const WallTracks& wallTracks, const spotter::Tracker& tracker){
//
//@param1 - walls: the raw walls found this frame
//@param2 - wallTracks: the filtered walls
//@param3 - tracker: the tracked objects
//
//@brief - console lines: object count, walls found this frame and tracked, then each confirmed wall's filtered
//         distance (and the raw one), length, angle and frames seen. O(walls).
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

//int main(){
//
//@brief - the program: 06c · Wall Filter. Example 06b finds the walls afresh every frame, so they wobble. RANSAC
//         is random, and the depth is noisy. But walls don't move. This one gives each wall its own EXTENDED
//         Kalman filter (EKF), the way 05 gave each object a Kalman filter. Setup, main loop and clean-up are laid
//         out in the file's sections. Returns 0 on a normal exit, 1 if it couldn't start.
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
    cv::namedWindow(CameraWindow, cv::WINDOW_NORMAL);
    cv::resizeWindow(CameraWindow, 960, 540);
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
        ShowCamera(frame.colour, tracker, yolo.Classes(), fps);
        ShowRadar(radar, walls, wallTracks, tracker, trails, yolo.Classes(), frame.colourCamera);
        if (report.Due()) Report(walls, wallTracks, tracker);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 'c') trails.Clear();
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Compare the console's filtered distance with the raw one: the raw one wobbles a few mm,
//     the filtered one barely moves (on the sensor: 1.7 -> 0.3 mm for a wall straight ahead).
//  2. Set WallDriftMeters = 0.05: the filter trusts each frame more and wobbles like 06b.
//  3. Walk in front of a wall: its raw ends jump about; the filtered ends hardly notice.
//  4. Nudge the Kinect a little: the filtered wall follows within a second or so.
//  5. Set MatchDegrees = 1: tracked walls keep losing their match and start over with new IDs.
