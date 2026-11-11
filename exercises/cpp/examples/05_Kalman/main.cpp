// =====================================================================
//  05 · Kalman
//
//  Example 04 matched each box to where the object WAS last frame, and
//  smoothed its distance with a running average. Two problems:
//    - at ~6 frames per second, a walking person moves so far between frames
//      that the old and new boxes barely overlap -> new ID
//    - averaging lags behind anything that moves
//
//  One new idea fixes both: a Kalman filter per object. It keeps a guess of
//  where the object is AND how fast it's moving, plus how sure it is, and
//  repeats two steps every frame:
//
//    Predict   move the guess forward: position += speed * time. Being a
//              guess, it becomes less sure.
//    Correct   blend in this frame's measurement. Whichever is more certain,
//              the measurement or the prediction, gets more weight.
//
//  Matching now uses the PREDICTED box, so fast movers keep their ID. When
//  YOLO misses an object, the prediction keeps it moving ("coasting"), and
//  its speed comes for free: how fast it approaches or moves away.
//  (spotter::BoxFilter is this same filter, for the later examples.)
//
//  Window: thick box = measured this frame, thin box = predicted only.
//  Console: raw vs filtered jitter (compare with example 04), ID changes.
//  Keys:   Q or Esc  quit
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>   // cv::KalmanFilter

#include "spotter/depth.hpp"
#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/track.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// Tracking: as in example 04, so the two compare fairly.
const float MinConfidence = 0.40f;
const double MinOverlap = 0.3;
const int MaxMissedFrames = 12;

// How much the MEASUREMENTS wobble: YOLO's box centre (pixels) and the sampled depth (metres).
// Bigger = trust measurements less = smoother but slower to react.
const double MeasurementNoisePixels = 15.0;
const double MeasurementNoiseMeters = 0.03;

// How suddenly objects can change speed (acceleration). Bigger = trust the prediction less =
// follows sudden moves faster but jitters more. A walking person: about 1 m/s^2, which is
// roughly 500 px/s^2 in the 1920-wide colour picture at 2 m.
const double AccelerationPixels = 500.0;
const double AccelerationMeters = 1.0;

const int ReportEveryMs = 1000;
const std::string Window = "05 kalman  (Q quits)";

//spotter::YoloDetector::Options YoloSettings(){
//
//@brief - the detector's options: yolo26n.onnx and its names file from the repo, minimum confidence MinConfidence.
//         O(1).
spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = MinConfidence;
    return options;
}

// ---------------------------------------------------------------------
//  The new idea: a Kalman filter per object
// ---------------------------------------------------------------------

// The filter's state, 6 numbers:   [ centreX, centreY, metres,  speedX, speedY, speedMetres ]
//                                    (pixels) (pixels) (depth)  (px/s)  (px/s)  (m/s)
// It measures 3 of them each frame: centreX, centreY, metres.
struct Track
{
    int id;
    int classId;
    cv::KalmanFilter kalman;
    cv::Size2d boxSize;       // the filter follows the centre; the size is copied from the last detection
    int missedFrames = 0;
    double rawMeters = 0;
    double lastFilteredMeters = 0, rawChangeSum = 0, filteredChangeSum = 0;   // jitter, as in 04
    int changeCount = 0;

    //double CentreX(){
    //
    //@brief - the filtered box centre's x, in colour pixels (state 0). O(1).
    double CentreX() const { return kalman.statePost.at<float>(0); }
    //double CentreY(){
    //
    //@brief - the filtered box centre's y, in colour pixels (state 1). O(1).
    double CentreY() const { return kalman.statePost.at<float>(1); }
    //double Meters(){
    //
    //@brief - the filtered distance in metres (state 2); 0 until one has been measured. O(1).
    double Meters() const { return kalman.statePost.at<float>(2); }
    //double MetersPerSecond(){
    //
    //@brief - the filtered speed along the distance (state 5): + = moving away. O(1).
    double MetersPerSecond() const { return kalman.statePost.at<float>(5); }
    //cv::Rect2d Box(){
    //
    //@brief - where the box should be NOW: the filtered centre with the last detection's size. O(1).
    cv::Rect2d Box() const { return cv::Rect2d(CentreX() - boxSize.width / 2, CentreY() - boxSize.height / 2, boxSize.width, boxSize.height); }
};

struct Tracks
{
    std::vector<Track> list;
    int nextId = 1;
};

struct Measured
{
    spotter::Detection detection;
    double rawMeters;   // 0 = no depth this frame
};

//cv::KalmanFilter NewKalman(double centreX, double centreY, double meters){
//
//@param1 - centreX: the box centre's x, pixels
//@param2 - centreY: its y, pixels
//@param3 - meters: its distance; <= 0 = not measured yet
//
//@brief - a new 6-state filter (x, y, distance and their speeds, 3 measured) for an object first seen here,
//         standing still. H picks the first three, R is the measurement noises squared, P starts at about one
//         measurement's noise for the position and "no idea" for the speeds (very unsure of the distance if none
//         yet). O(1).
cv::KalmanFilter NewKalman(double centreX, double centreY, double meters)
{
    cv::KalmanFilter kalman(6, 3, 0, CV_32F);   // 6 state numbers, 3 measured
    kalman.measurementMatrix = cv::Mat::eye(3, 6, CV_32F);   // H: the measurement is the first 3 state numbers

    // R: how noisy each measurement is (a variance = noise squared).
    kalman.measurementNoiseCov = (cv::Mat_<float>(3, 3) <<
        MeasurementNoisePixels * MeasurementNoisePixels, 0, 0,
        0, MeasurementNoisePixels * MeasurementNoisePixels, 0,
        0, 0, MeasurementNoiseMeters * MeasurementNoiseMeters);

    kalman.statePost = (cv::Mat_<float>(6, 1) << centreX, centreY, meters, 0, 0, 0);   // where we saw it, not moving

    // P: how unsure the starting guess is. Position: about one measurement's noise. Speed: no idea
    // yet (up to ~1000 px/s, ~2 m/s). No depth yet (0)? Then very unsure, so the first real depth
    // replaces the 0 at once instead of being averaged with it.
    double metersVariance = meters > 0 ? MeasurementNoiseMeters * MeasurementNoiseMeters : 100.0;
    kalman.errorCovPost = cv::Mat::diag((cv::Mat_<float>(6, 1) <<
        MeasurementNoisePixels * MeasurementNoisePixels, MeasurementNoisePixels * MeasurementNoisePixels,
        metersVariance, 1000 * 1000, 1000 * 1000, 2 * 2));
    return kalman;
}

//void SetTimeStep(cv::KalmanFilter& kalman, double dt){
//
//@param1 - kalman: the filter; its F and Q are rebuilt
//@param2 - dt: seconds since the last frame
//
//@brief - the time between frames changes, so F (position += speed x dt, speed unchanged) and Q (how far real
//         motion can drift from constant speed: a dt^2 / 2 for positions, a dt for speeds, from the Acceleration
//         settings) are rebuilt every frame. O(1).
void SetTimeStep(cv::KalmanFilter& kalman, double dt)
{
    // F: position += speed * dt, speed stays the same (constant velocity).
    cv::setIdentity(kalman.transitionMatrix);
    for (int i = 0; i < 3; i++) kalman.transitionMatrix.at<float>(i, i + 3) = (float)dt;

    // Q: how far the real motion can drift from "constant speed" during dt, caused by an
    // acceleration a: position by about a*dt^2/2, speed by about a*dt.
    // ponytail: diagonal approximation of the textbook Q; the full one adds position-speed correlation terms.
    const double accelerations[3] = { AccelerationPixels, AccelerationPixels, AccelerationMeters };
    kalman.processNoiseCov = cv::Mat::zeros(6, 6, CV_32F);
    for (int i = 0; i < 3; i++)
    {
        kalman.processNoiseCov.at<float>(i, i) = (float)std::pow(accelerations[i] * dt * dt / 2, 2);
        kalman.processNoiseCov.at<float>(i + 3, i + 3) = (float)std::pow(accelerations[i] * dt, 2);
    }
}

//void Correct(Track& track, const Measured& m){
//
//@param1 - track: the track to correct
//@param2 - m: this frame's detection with its distance
//
//@brief - blends the measurement in. No depth this frame? It feeds in the prediction's own distance with a huge
//         noise, so the filter ignores it and keeps its guess. Also resets the missed count and records how much
//         the raw and filtered distances changed. O(1).
void Correct(Track& track, const Measured& m)
{
    double meters = m.rawMeters;
    bool haveDepth = meters > 0;
    if (!haveDepth)
    {
        meters = track.Meters();
        track.kalman.measurementNoiseCov.at<float>(2, 2) = 1e6f;
    }
    const cv::Rect2d& box = m.detection.box;
    track.kalman.correct((cv::Mat_<float>(3, 1) << box.x + box.width / 2, box.y + box.height / 2, meters));
    track.kalman.measurementNoiseCov.at<float>(2, 2) = (float)(MeasurementNoiseMeters * MeasurementNoiseMeters);
    track.boxSize = box.size();
    track.missedFrames = 0;
    if (!haveDepth) return;
    if (track.rawMeters > 0)
    {
        track.rawChangeSum += std::abs(m.rawMeters - track.rawMeters);
        track.filteredChangeSum += std::abs(track.Meters() - track.lastFilteredMeters);
        track.changeCount++;
    }
    track.rawMeters = m.rawMeters;
    track.lastFilteredMeters = track.Meters();
}

//void Predict(Tracks& tracks, double dt){
//
//@param1 - tracks: the tracks; moved on in place
//@param2 - dt: seconds since the last frame
//
//@brief - moves every object forward in time (SetTimeStep, predict). Afterwards the filter's best guess
//         (statePost) IS the prediction, so an object YOLO misses keeps moving ("coasting"). O(tracks).
void Predict(Tracks& tracks, double dt)
{
    for (Track& track : tracks.list)
    {
        SetTimeStep(track.kalman, dt);
        track.kalman.predict();
    }
}

//void StartTrack(Tracks& tracks, const Measured& m, const spotter::ClassList& classes){
//
//@param1 - tracks: the tracks; one is added
//@param2 - m: the unmatched detection
//@param3 - classes: the class names, for the console line
//
//@brief - a new ID for it, with a fresh filter at its centre and distance; prints "new ID". O(1).
void StartTrack(Tracks& tracks, const Measured& m, const spotter::ClassList& classes)
{
    Track track;
    track.id = tracks.nextId++;
    track.classId = m.detection.classId;
    track.boxSize = m.detection.box.size();
    track.kalman = NewKalman(m.detection.box.x + m.detection.box.width / 2, m.detection.box.y + m.detection.box.height / 2, m.rawMeters);
    track.rawMeters = track.lastFilteredMeters = m.rawMeters;
    std::printf("  new ID #%d %s\n", track.id, classes.Name(track.classId).c_str());
    tracks.list.push_back(std::move(track));
}

//void Match(Tracks& tracks, const std::vector<Measured>& measured, const spotter::ClassList& classes){
//
//@param1 - tracks: the tracks; updated in place
//@param2 - measured: this frame's detections with their distances
//@param3 - classes: the class names
//
//@brief - 04's greedy matching; the only change: Box() is where the object should be NOW (predicted). Matched
//         tracks are corrected, unmatched detections start tracks, unmatched tracks count a missed frame and are
//         dropped after MaxMissedFrames. O(T x D x min(T, D)).
void Match(Tracks& tracks, const std::vector<Measured>& measured, const spotter::ClassList& classes)
{
    std::vector<bool> trackMatched(tracks.list.size(), false), detectionMatched(measured.size(), false);
    while (true)
    {
        double bestOverlap = MinOverlap;
        int bestTrack = -1, bestDetection = -1;
        for (size_t t = 0; t < tracks.list.size(); t++)
            for (size_t d = 0; d < measured.size(); d++)
            {
                if (trackMatched[t] || detectionMatched[d] || tracks.list[t].classId != measured[d].detection.classId) continue;
                double overlap = spotter::Overlap(tracks.list[t].Box(), measured[d].detection.box);
                if (overlap >= bestOverlap) { bestOverlap = overlap; bestTrack = (int)t; bestDetection = (int)d; }
            }
        if (bestTrack < 0) break;
        trackMatched[bestTrack] = detectionMatched[bestDetection] = true;
        Correct(tracks.list[bestTrack], measured[bestDetection]);
    }
    for (size_t d = 0; d < measured.size(); d++)
        if (!detectionMatched[d]) StartTrack(tracks, measured[d], classes);

    // Unmatched tracks coast on their prediction. Forget them after MaxMissedFrames.
    for (size_t t = 0; t < trackMatched.size(); t++)
    {
        Track& track = tracks.list[t];
        if (!trackMatched[t] && ++track.missedFrames > MaxMissedFrames)
            std::printf("  dropped #%d %s: unseen for %d frames in a row\n", track.id, classes.Name(track.classId).c_str(), track.missedFrames);
    }
    tracks.list.erase(std::remove_if(tracks.list.begin(), tracks.list.end(), [](const Track& t) { return t.missedFrames > MaxMissedFrames; }),
                      tracks.list.end());
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

//void Show(cv::Mat& picture, const Tracks& tracks, const spotter::ClassList& classes, const spotter::FpsMeter& fps){
//
//@param1 - picture: the colour picture; drawn on
//@param2 - tracks: the tracked objects
//@param3 - classes: the class names
//@param4 - fps: the frame-rate meter
//
//@brief - a box per track in its ID's colour (thin and "(predicted)" while coasting), labelled with its class,
//         distance and whether it's approaching or moving away, and the fps; shown with the key strip. O(tracks).
void Show(cv::Mat& picture, const Tracks& tracks, const spotter::ClassList& classes, const spotter::FpsMeter& fps)
{
    for (const Track& track : tracks.list)
    {
        bool coasting = track.missedFrames > 0;
        cv::Scalar colour = spotter::ColourForId(track.id);
        std::string label = "#" + std::to_string(track.id) + " " + classes.Name(track.classId);
        if (track.Meters() > 0)
        {
            double speed = track.MetersPerSecond();
            label += cv::format(" %.2fm %s %.1fm/s", track.Meters(), speed < 0 ? "approaching" : "away", std::abs(speed));
        }
        if (coasting) label += " (predicted)";
        cv::rectangle(picture, track.Box(), colour, coasting ? 1 : 3);
        cv::putText(picture, label, cv::Point((int)track.Box().x, std::max((int)track.Box().y - 10, 30)),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, colour, coasting ? 1 : 2);
    }
    fps.Draw(picture);
    cv::imshow(Window, spotter::WithKeys(picture, Keys));
}

//void Report(Tracks& tracks, const spotter::ClassList& classes, const spotter::FpsMeter& fps){
//
//@param1 - tracks: the tracked objects; their change counters are reset
//@param2 - classes: the class names
//@param3 - fps: the frame-rate meter
//
//@brief - fps, how many objects are tracked, and per track its raw and filtered distance, speed and the average
//         frame-to-frame change of each in mm; then the counters restart. O(tracks).
void Report(Tracks& tracks, const spotter::ClassList& classes, const spotter::FpsMeter& fps)
{
    std::printf("%.1f fps | %zu objects tracked\n", fps.Fps(), tracks.list.size());
    for (Track& track : tracks.list)
    {
        if (track.changeCount == 0) continue;
        std::printf("  #%-3d %-12s raw %.3f m, filtered %.3f m, speed %+.2f m/s | change per frame: raw %5.1f mm, filtered %5.1f mm\n",
                    track.id, classes.Name(track.classId).c_str(), track.rawMeters, track.Meters(), track.MetersPerSecond(),
                    1000 * track.rawChangeSum / track.changeCount, 1000 * track.filteredChangeSum / track.changeCount);
        track.rawChangeSum = track.filteredChangeSum = 0;
        track.changeCount = 0;
    }
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

//int main(){
//
//@brief - the program: 05 · Kalman. Example 04 matched each box to where the object WAS last frame, and smoothed
//         its distance with a running average. Setup, main loop and clean-up are laid out in the file's sections.
//         Returns 0 on a normal exit, 1 if it couldn't start.
int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    cv::namedWindow(Window, cv::WINDOW_NORMAL);
    cv::resizeWindow(Window, 960, 540);
    spotter::BoxDepthSampler boxDepth;
    Tracks tracks;
    spotter::FpsMeter fps;
    spotter::Every report(ReportEveryMs);
    spotter::RgbdFrame frame;
    spotter::FrameTimer timer;

    while (kinect.Next(frame))
    {
        double dt = timer.Step(frame);   // the filter needs the real time step

        boxDepth.Build(frame);
        std::vector<Measured> measured;
        for (const spotter::Detection& d : yolo.Detect(frame.colour)) measured.push_back({ d, boxDepth.MedianMeters(d.box).value_or(0) });
        Predict(tracks, dt);
        Match(tracks, measured, yolo.Classes());

        fps.Tick();
        Show(frame.colour, tracks, yolo.Classes(), fps);
        if (report.Due()) Report(tracks, yolo.Classes(), fps);
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Run example 04, walk across the view, and count the ID changes. Then do the same here.
//  2. Walk toward the camera and back. Watch "approaching" / "away" and the speed. Normal walking
//     is about 1-1.5 m/s. 04's smoothed distance trails behind; this one keeps up.
//  3. Step behind something for a second. Your box keeps moving thin ("predicted") in the
//     direction you were going, then snaps back to thick when you reappear.
//  4. Set MeasurementNoiseMeters = 0.3: very smooth but slow. 0.003: follows every wobble, like raw.
//  5. Set AccelerationMeters = 0.1: the filter believes objects barely change speed. Stop
//     suddenly and watch the predicted distance overshoot before it corrects.
//  6. Stand still and read the speed: how far from 0.00 m/s does it wander? That's the noise
//     left in the speed estimate.
