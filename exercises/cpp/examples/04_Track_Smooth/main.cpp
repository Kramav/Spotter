// =====================================================================
//  04 · Track + Smooth
//
//  Example 02 measures every box from scratch each frame, so the distances
//  jitter and nothing connects "the person" in one frame to the next.
//  This example adds one idea on top: follow each object over time.
//
//    1-3. Detect, register, sample   as in example 02 (the library does them)
//    4.   Track    match this frame's boxes to last frame's by how much they
//                  overlap, so each object keeps an ID number:  #3 person
//    5.   Smooth   a running average of each object's own distance. Because
//                  it follows the OBJECT, not a pixel, it still works when
//                  the object or the camera moves (docs/decisions.md D12).
//
//  Window: one colour per ID. Label = smoothed distance, raw value in brackets.
//  Console: per object, how much the raw and smoothed distance change from
//           frame to frame. That's the jitter, and the smoothing should cut it.
//  Keys:   Q or Esc  quit
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/track.hpp"
#include "spotter/view.hpp"

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

const float MinConfidence = 0.40f;   // as in example 02

// Two boxes of the same class count as the same object when they overlap at least this much.
// Too high: a moving object keeps getting a new ID. Too low: two neighbours swap IDs.
const double MinOverlap = 0.3;

// How many frames an object may go unseen (hidden, or missed by YOLO) before its ID is dropped.
// At about 6 frames per second, 12 frames is about 2 seconds.
const int MaxMissedFrames = 12;

// Each frame, the smoothed distance moves only (1 - DistanceSmoothing) of the way toward the
// new raw measurement. 0 = no smoothing (raw), 0.8 = calm, 0.95 = very calm but slow to follow.
const double DistanceSmoothing = 0.8;

const int ReportEveryMs = 1000;
const std::string Window = "04 track + smooth  (Q quits)";

spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = MinConfidence;
    return options;
}

// ---------------------------------------------------------------------
//  The new idea: tracks, matched by overlap, with a smoothed distance
// ---------------------------------------------------------------------

struct Measured
{
    spotter::Detection detection;
    double rawMeters;   // 0 = too few depth readings in the box
};

// One object being followed over time.
struct Track
{
    int id;
    int classId;
    cv::Rect2d box;             // where it was last seen
    int missedFrames = 0;       // frames in a row without a matching detection
    double rawMeters = 0;       // latest measurement (0 = none yet)
    double smoothedMeters = 0;  // running average (0 = none yet)
    double rawChangeSum = 0, smoothedChangeSum = 0;   // jitter since the last report
    int changeCount = 0;
};

struct Tracks
{
    std::vector<Track> list;
    int nextId = 1;
};

// Intersection over union: the overlapping area divided by the area the two boxes cover together.
// 1 = identical boxes, 0 = not touching.
double Overlap(const cv::Rect2d& a, const cv::Rect2d& b)
{
    double intersection = (a & b).area();
    double together = a.area() + b.area() - intersection;
    return together > 0 ? intersection / together : 0;
}

// Step 5: the smoothed distance moves part of the way toward each new measurement.
void Smooth(Track& track, double rawMeters)
{
    if (rawMeters <= 0) return;
    if (track.smoothedMeters == 0) track.smoothedMeters = rawMeters;   // first measurement: nothing to average yet
    else
    {
        double smoothed = DistanceSmoothing * track.smoothedMeters + (1 - DistanceSmoothing) * rawMeters;
        track.rawChangeSum += std::abs(rawMeters - track.rawMeters);
        track.smoothedChangeSum += std::abs(smoothed - track.smoothedMeters);
        track.changeCount++;
        track.smoothedMeters = smoothed;
    }
    track.rawMeters = rawMeters;
}

// Why a new ID? If an unmatched object of the same class exists, it was probably the same thing:
// print how much they overlapped (below MinOverlap = it moved too far, or was gone too long).
void ExplainNewId(const Measured& m, const Tracks& tracks, const std::vector<bool>& trackMatched, const spotter::ClassList& classes)
{
    const Track* closest = nullptr;
    double closestOverlap = -1;
    for (size_t t = 0; t < trackMatched.size(); t++)
    {
        const Track& track = tracks.list[t];
        if (trackMatched[t] || track.classId != m.detection.classId) continue;
        double overlap = Overlap(track.box, m.detection.box);
        if (overlap > closestOverlap) { closest = &track; closestOverlap = overlap; }
    }
    const char* name = classes.Name(m.detection.classId).c_str();
    if (closest)
        std::printf("  new ID #%d %s | unmatched %s #%d: overlap %.2f (needs %.2f), unseen for %d frame(s)\n",
                    tracks.nextId, name, name, closest->id, closestOverlap, MinOverlap, closest->missedFrames + 1);
}

// Step 4: greedy matching. Of all (track, detection) pairs with the same class, take the pair that
// overlaps most, pair them up, and repeat until no pair overlaps at least MinOverlap.
void Update(Tracks& tracks, const std::vector<Measured>& measured, const spotter::ClassList& classes)
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
                double overlap = Overlap(tracks.list[t].box, measured[d].detection.box);
                if (overlap >= bestOverlap) { bestOverlap = overlap; bestTrack = (int)t; bestDetection = (int)d; }
            }
        if (bestTrack < 0) break;
        trackMatched[bestTrack] = detectionMatched[bestDetection] = true;
        Track& track = tracks.list[bestTrack];
        track.box = measured[bestDetection].detection.box;
        track.missedFrames = 0;
        Smooth(track, measured[bestDetection].rawMeters);
    }

    // Detections nobody matched are new objects: each gets a new ID.
    for (size_t d = 0; d < measured.size(); d++)
    {
        if (detectionMatched[d]) continue;
        ExplainNewId(measured[d], tracks, trackMatched, classes);
        Track track;
        track.id = tracks.nextId++;
        track.classId = measured[d].detection.classId;
        track.box = measured[d].detection.box;
        track.rawMeters = track.smoothedMeters = measured[d].rawMeters;
        tracks.list.push_back(track);
    }

    // Tracks nobody matched weren't seen this frame. Forget them after MaxMissedFrames.
    // (New tracks were added after trackMatched was sized, so they are skipped here.)
    for (size_t t = 0; t < trackMatched.size(); t++)
        if (!trackMatched[t]) tracks.list[t].missedFrames++;
    tracks.list.erase(std::remove_if(tracks.list.begin(), tracks.list.end(), [](const Track& t) { return t.missedFrames > MaxMissedFrames; }),
                      tracks.list.end());
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

void Show(cv::Mat& picture, const Tracks& tracks, const spotter::ClassList& classes, const spotter::FpsMeter& fps)
{
    for (const Track& track : tracks.list)
    {
        if (track.missedFrames > 0) continue;   // only what is visible right now
        cv::Scalar colour = spotter::ColourForId(track.id);
        std::string label = "#" + std::to_string(track.id) + " " + classes.Name(track.classId) + " "
                          + (track.smoothedMeters > 0 ? cv::format("%.2fm (raw %.2f)", track.smoothedMeters, track.rawMeters) : "?m");
        cv::rectangle(picture, track.box, colour, 3);
        cv::putText(picture, label, cv::Point((int)track.box.x, std::max((int)track.box.y - 10, 30)), cv::FONT_HERSHEY_SIMPLEX, 1.0, colour, 2);
    }
    fps.Draw(picture);
    cv::imshow(Window, picture);
}

// Average frame-to-frame change in mm: how much each number jumps around.
void Report(Tracks& tracks, const spotter::ClassList& classes, const spotter::FpsMeter& fps)
{
    std::printf("%.1f fps | %zu objects tracked, next new ID will be #%d\n", fps.Fps(), tracks.list.size(), tracks.nextId);
    for (Track& track : tracks.list)
    {
        if (track.changeCount == 0) continue;
        std::printf("  #%-3d %-12s raw %.3f m, smoothed %.3f m | change per frame: raw %5.1f mm, smoothed %5.1f mm\n",
                    track.id, classes.Name(track.classId).c_str(), track.rawMeters, track.smoothedMeters,
                    1000 * track.rawChangeSum / track.changeCount, 1000 * track.smoothedChangeSum / track.changeCount);
        track.rawChangeSum = track.smoothedChangeSum = 0;
        track.changeCount = 0;
    }
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

    cv::namedWindow(Window, cv::WINDOW_NORMAL);
    cv::resizeWindow(Window, 960, 540);
    spotter::BoxDepthSampler boxDepth;
    Tracks tracks;
    spotter::FpsMeter fps;
    spotter::Every report(ReportEveryMs);
    spotter::RgbdFrame frame;

    while (kinect.Next(frame))
    {
        boxDepth.Build(frame);
        std::vector<Measured> measured;
        for (const spotter::Detection& d : yolo.Detect(frame.colour))   // steps 1-3
            measured.push_back({ d, boxDepth.MedianMeters(d.box).value_or(0) });
        Update(tracks, measured, yolo.Classes());                       // steps 4-5

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
//  1. Sit still. Watch the console: the smoothed "change per frame" should be a fraction of the
//     raw one. Try DistanceSmoothing = 0, 0.5, 0.8, 0.95 and compare.
//  2. Walk slowly toward the camera with DistanceSmoothing = 0.95. The smoothed value lags behind
//     the raw one: smoothing trades jitter for delay. Example 05's Kalman filter predicts instead.
//  3. Walk quickly across the view. If your box jumps further than it overlaps between frames
//     (about 6 per second here), you get a new ID. Lower MinOverlap and try again.
//  4. Walk behind a chair for a second and come back. Within MaxMissedFrames you keep your ID;
//     after that you're a new object. Set MaxMissedFrames = 0 to see the difference.
//  5. Two people cross paths: do they keep their IDs, or swap? Matching only by overlap can't
//     tell them apart. That's what appearance features (or a Kalman prediction) fix.
