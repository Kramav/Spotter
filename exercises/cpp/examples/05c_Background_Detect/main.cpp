// =====================================================================
//  05c · Background Detect
//
//  Examples 02-05b run YOLO on every frame and wait for it: ~150 ms, so
//  everything (picture, tracking, distances) runs at ~6 fps, while the
//  Kinect delivers 30. YOLO is ~95% of each frame.
//
//  One idea: stop waiting for YOLO. Run it on a second thread.
//
//    Detector thread   takes the newest frame, runs YOLO (~6 times a second),
//                      hands back boxes, takes the next frame, repeat.
//    Main loop         runs at the camera's 30 fps. Every frame it:
//                        - predicts every box forward (the Kalman filter, 05)
//                        - re-measures each object's distance from FRESH depth
//                        - folds in YOLO's boxes whenever a new batch arrives
//                          (two-pass matching, as in 05b)
//
//  The thread is this example's idea and is written out below. The
//  tracking is 05b's, from the library (spotter::Tracker); later examples
//  use spotter::AsyncDetector, this same thread packaged.
//
//  Two settings to compare speed:
//    DetectInBackground = false   -> waits for YOLO every frame, like 05b
//    ModelInputSize     = 320     -> a smaller YOLO, ~4x faster, misses small things
//
//  Window: thick box = seen in YOLO's latest batch, thin = predicted only.
//          Bottom-left: display fps and YOLO's time per batch.
//  Keys:   Q or Esc  quit
// =====================================================================

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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

// true = YOLO on its own thread, main loop at camera speed. false = wait for YOLO every frame.
const bool DetectInBackground = true;

// YOLO's input size: 640 or 320. 320 is ~4x faster but misses small and far-away objects.
// Each size is its own model file (an ONNX model's input size is fixed when it's exported).
const int ModelInputSize = 640;
const std::string ModelPath = std::string(SPOTTER_REPO_ROOT) + (ModelInputSize == 320 ? "/yolo26n_320.onnx" : "/yolo26n.onnx");

// In a dim room the colour camera's automatic exposure needs more than one frame's time, and the
// Kinect drops to 15 fps. true = cap the exposure so it keeps 30 fps: a darker, grainier picture.
const bool CapColorExposure = true;

const double MaxUnseenSeconds = 5.0;   // seconds now, because frames and YOLO batches differ
const int ReportEveryMs = 1000;
const std::string Window = "05c background detect  (Q quits)";

//spotter::YoloDetector::Options YoloSettings(){
//
//@brief - the detector's options: ModelPath, the repo's names file, ModelInputSize, and 05b's weak threshold 0.15
//         (the tracker sorts sure from unsure). O(1).
spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = ModelPath;
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.inputSize = ModelInputSize;
    options.minConfidence = 0.15f;   // 05b's weak threshold: the tracker sorts sure from unsure
    return options;
}

//spotter::KinectSource::Options KinectSettings(){
//
//@brief - the Kinect's options: the defaults, with the colour exposure capped if CapColorExposure is set (a steady
//         frame rate in dim light). O(1).
spotter::KinectSource::Options KinectSettings()
{
    spotter::KinectSource::Options options;
    options.capExposure = CapColorExposure;
    return options;
}

//spotter::Tracker::Options TrackerSettings(){
//
//@brief - the tracker's options: 05b's two thresholds (0.40 / 0.15), overlap 0.3 and the fading coast, with
//         MaxUnseenSeconds. O(1).
spotter::Tracker::Options TrackerSettings()
{
    spotter::Tracker::Options options;   // 05b's two thresholds (0.40 / 0.15), overlap 0.3, fading coast
    options.maxUnseenSeconds = MaxUnseenSeconds;
    return options;
}

// ---------------------------------------------------------------------
//  The new idea: YOLO on its own thread
// ---------------------------------------------------------------------

// The two threads share only these, and touch them only while holding the mutex (a lock: one
// thread at a time). The condition variable lets the detector thread sleep until it has work.
struct Handover
{
    std::mutex lock;
    std::condition_variable frameHandedOver;
    cv::Mat frame;                       // main -> detector
    bool frameWaiting = false;
    std::vector<spotter::Detection> result;   // detector -> main
    bool resultWaiting = false;
    double lastDetectMs = 0;
    bool quit = false;
    bool busy = false;                   // main loop only: handed a frame over, no answer yet
    int batchesSinceReport = 0;          // main loop only
};

//void DetectorThread(spotter::YoloDetector& yolo, Handover& handover){
//
//@param1 - yolo: a ready detector
//@param2 - handover: what this thread and the main loop share; touched only under its lock
//
//@brief - the detector thread: sleep until a frame is handed over, run YOLO on it without the lock (the slow
//         part), time it, and hand the boxes back; returns when told to quit. O(detector run) per frame.
void DetectorThread(spotter::YoloDetector& yolo, Handover& handover)
{
    while (true)
    {
        cv::Mat image;
        {
            std::unique_lock<std::mutex> lock(handover.lock);
            handover.frameHandedOver.wait(lock, [&] { return handover.frameWaiting || handover.quit; });
            if (handover.quit) return;
            image = handover.frame;
            handover.frameWaiting = false;
        }
        auto start = std::chrono::steady_clock::now();
        std::vector<spotter::Detection> result = yolo.Detect(image);   // the slow part, done without the lock
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::lock_guard<std::mutex> lock(handover.lock);
        handover.result = std::move(result);
        handover.resultWaiting = true;
        handover.lastDetectMs = ms;
    }
}

//std::optional<std::vector<spotter::Detection>> ExchangeWithDetector(Handover& handover, const cv::Mat& colour){
//
//@param1 - handover: what the thread and the main loop share
//@param2 - colour: this frame's colour picture
//
//@brief - main loop side, every frame: collects a finished batch if there is one, and if the detector is idle
//         gives it this frame (a copy: the main loop keeps drawing on its own) and wakes it. Returns the batch or
//         nothing. O(1) plus the copy when handing over.
std::optional<std::vector<spotter::Detection>> ExchangeWithDetector(Handover& handover, const cv::Mat& colour)
{
    std::lock_guard<std::mutex> lock(handover.lock);
    std::optional<std::vector<spotter::Detection>> batch;
    if (handover.resultWaiting)
    {
        batch = std::move(handover.result);
        handover.resultWaiting = handover.busy = false;
    }
    if (!handover.busy)
    {
        handover.frame = colour.clone();
        handover.frameWaiting = handover.busy = true;
        handover.frameHandedOver.notify_one();
    }
    return batch;
}

double LastDetectMs(Handover& handover)
{
    std::lock_guard<std::mutex> lock(handover.lock);
    return handover.lastDetectMs;
}

//void StopDetector(Handover& handover, std::thread& thread){
//
//@param1 - handover: what the thread and the main loop share
//@param2 - thread: the detector thread
//
//@brief - tells the thread to quit and joins it, so the program ends cleanly. ponytail: an exception would skip
//         this, and a running std::thread's destructor ends the program; spotter::AsyncDetector (the library's
//         copy) stops itself in its destructor. Waits for a run in progress.
void StopDetector(Handover& handover, std::thread& thread)
{
    if (!thread.joinable()) return;
    {
        std::lock_guard<std::mutex> lock(handover.lock);
        handover.quit = true;
    }
    handover.frameHandedOver.notify_one();
    thread.join();
}

//std::optional<std::vector<spotter::Detection>> Detections(spotter::YoloDetector& yolo, Handover& handover, const cv::Mat& colour){
//
//@param1 - yolo: a ready detector, used directly when not in the background
//@param2 - handover: what the thread and the main loop share
//@param3 - colour: this frame's colour picture
//
//@brief - either way, one batch of detections for this frame, or none yet: ExchangeWithDetector with
//         DetectInBackground, else a direct yolo.Detect that waits (like 02-05b). Counts the batch and notes how
//         long it took. O(1) in the background, O(detector run) otherwise.
std::optional<std::vector<spotter::Detection>> Detections(spotter::YoloDetector& yolo, Handover& handover, const cv::Mat& colour)
{
    std::optional<std::vector<spotter::Detection>> batch;
    if (DetectInBackground) batch = ExchangeWithDetector(handover, colour);
    else
    {
        auto start = std::chrono::steady_clock::now();
        batch = yolo.Detect(colour);   // wait for it, like 02-05b
        handover.lastDetectMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    if (batch) handover.batchesSinceReport++;
    return batch;
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

//std::string Label(const spotter::Track& track, const spotter::ClassList& classes){
//
//@param1 - track: a tracked object
//@param2 - classes: the class names
//
//@brief - "#id class 1.23m approaching 0.5m/s", with "(predicted)" added while coasting. O(1).
std::string Label(const spotter::Track& track, const spotter::ClassList& classes)
{
    std::string label = "#" + std::to_string(track.id) + " " + classes.Name(track.classId);
    double meters = track.filter.Meters(), speed = track.filter.MetersPerSecond();
    if (meters > 0) label += cv::format(" %.2fm %s %.1fm/s", meters, speed < 0 ? "approaching" : "away", std::abs(speed));
    return track.Coasting() ? label + " (predicted)" : label;
}

//void Show(cv::Mat& picture, const spotter::Tracker& tracker, const spotter::ClassList& classes, const spotter::FpsMeter& fps, double detectMs){
//
//@param1 - picture: the colour picture; drawn on
//@param2 - tracker: the tracked objects
//@param3 - classes: the class names
//@param4 - fps: the frame-rate meter
//@param5 - detectMs: how long the last YOLO run took, shown beside the fps
//
//@brief - a box per track in its ID's colour (thin while coasting) with its label, and the fps with the YOLO time;
//         shown with the key strip. O(tracks).
void Show(cv::Mat& picture, const spotter::Tracker& tracker, const spotter::ClassList& classes, const spotter::FpsMeter& fps, double detectMs)
{
    for (const spotter::Track& track : tracker.Tracks())
    {
        cv::Scalar colour = spotter::ColourForId(track.id);
        cv::rectangle(picture, track.Box(), colour, track.Coasting() ? 1 : 3);
        cv::putText(picture, Label(track, classes), cv::Point((int)track.Box().x, std::max((int)track.Box().y - 10, 30)),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, colour, track.Coasting() ? 1 : 2);
    }
    fps.Draw(picture, cv::format(" | YOLO %d ms", (int)detectMs));
    cv::imshow(Window, spotter::WithKeys(picture, Keys));
}

//void Report(const spotter::Tracker& tracker, const spotter::ClassList& classes, const spotter::FpsMeter& fps, Handover& handover){
//
//@param1 - tracker: the tracked objects
//@param2 - classes: the class names
//@param3 - fps: the frame-rate meter
//@param4 - handover: for the batch count and the last YOLO time; its count restarts
//
//@brief - console lines: display fps, YOLO batches per second and time each, the object count, then each track's
//         label (with how long a coasting one has been unseen). O(tracks).
void Report(const spotter::Tracker& tracker, const spotter::ClassList& classes, const spotter::FpsMeter& fps, Handover& handover)
{
    std::printf("%.1f fps display | YOLO: %d batches/s, %.0f ms each (input %d) | %zu objects\n",
                fps.Fps(), handover.batchesSinceReport, LastDetectMs(handover), ModelInputSize, tracker.Tracks().size());
    handover.batchesSinceReport = 0;
    for (const spotter::Track& track : tracker.Tracks())
        std::printf("  %s%s\n", Label(track, classes).c_str(), track.Coasting() ? cv::format(" (unseen %.1f s)", track.secondsUnseen).c_str() : "");
}

//void PrintEvents(spotter::Tracker& tracker, const spotter::ClassList& classes){
//
//@param1 - tracker: the tracker; its events are taken
//@param2 - classes: the class names
//
//@brief - prints each "new ID" and "dropped" (with the seconds unseen) since the last call (Tracker::TakeEvents).
//         O(events).
void PrintEvents(spotter::Tracker& tracker, const spotter::ClassList& classes)
{
    for (const spotter::TrackEvent& e : tracker.TakeEvents())
        if (e.started) std::printf("  new ID #%d %s\n", e.id, classes.Name(e.classId).c_str());
        else std::printf("  dropped #%d %s: unseen for %.1f s\n", e.id, classes.Name(e.classId).c_str(), e.secondsUnseen);
}

// ---------------------------------------------------------------------
//  Setup and main operation: runs at the camera's speed
// ---------------------------------------------------------------------

//int main(){
//
//@brief - the program: 05c · Background Detect. Examples 02-05b run YOLO on every frame and wait for it: ~150 ms,
//         so everything (picture, tracking, distances) runs at ~6 fps, while the Kinect delivers 30. YOLO is ~95%
//         of each frame. Setup, main loop and clean-up are laid out in the file's sections. Returns 0 on a normal
//         exit, 1 if it couldn't start.
int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(KinectSettings());
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    Handover handover;
    std::thread detectorThread;
    if (DetectInBackground) detectorThread = std::thread(DetectorThread, std::ref(yolo), std::ref(handover));

    cv::namedWindow(Window, cv::WINDOW_NORMAL);
    cv::resizeWindow(Window, 960, 540);
    spotter::Tracker tracker(TrackerSettings());
    spotter::BoxDepthSampler boxDepth;
    spotter::FpsMeter fps;
    spotter::Every report(ReportEveryMs);
    spotter::RgbdFrame frame;
    spotter::FrameTimer timer;

    while (kinect.Next(frame))
    {
        boxDepth.Build(frame);
        tracker.Predict(timer.Step(frame));                                                          // every frame: boxes move on
        if (auto batch = Detections(yolo, handover, frame.colour)) tracker.Match(*batch, boxDepth);   // a few times a second
        tracker.Remeasure(boxDepth);                                                                 // every frame: fresh distances
        // ponytail: a background batch describes a frame ~one YOLO run old (~150 ms at 640), treated
        // as current here; the fix would be to correct the filter at that older time.

        fps.Tick();
        Show(frame.colour, tracker, yolo.Classes(), fps, LastDetectMs(handover));
        PrintEvents(tracker, yolo.Classes());
        if (report.Due()) Report(tracker, yolo.Classes(), fps, handover);
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    StopDetector(handover, detectorThread);
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Note the fps, then set DetectInBackground = false. Same YOLO speed, but now everything
//     waits for it: the fps drops to YOLO's rate.
//  2. Set ModelInputSize = 320 (needs yolo26n_320.onnx). YOLO gets ~4x faster: more batches per
//     second, so boxes snap to objects more often. What does it stop finding? Something small?
//  3. Walk across the view. Between YOLO batches your box keeps moving with the prediction; each
//     batch nudges it back. Is the box behind you? That's the ~150 ms age of each batch.
//  4. Sway toward and away from the camera: the distance updates every frame from fresh depth,
//     not just when YOLO runs.
//  5. In a dim room, set CapColorExposure = false: the display drops to 15 fps, because the
//     camera itself slows down to collect more light. Does YOLO find more with the brighter picture?
