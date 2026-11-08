// =====================================================================
//  05b · Two Thresholds
//
//  05's logs showed why a still TV kept getting new IDs: YOLO's confidence
//  for it flickered around 0.40, so it dropped out for seconds at a time,
//  longer than MaxMissedFrames. A lower threshold everywhere would add
//  false alarms. So use two:
//
//    StartConfidence (0.40)  only a sure detection may start a NEW ID
//    KeepConfidence  (0.15)  a weak detection may only CONTINUE an existing
//                            ID, if it overlaps that object's predicted box
//
//  Matching runs in two passes: sure detections first, then the weak ones
//  for whatever is left, so a weak box never steals a sure one's object.
//  And a coasting object's speed fades (CoastSlowdownSeconds), so an
//  object unseen for longer doesn't drift off across the picture.
//
//  Window: thick = matched by a sure detection, medium = by a weak one,
//          thin = predicted only.
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

const float StartConfidence = 0.40f;       // new IDs only from detections at least this sure
const float KeepConfidence = 0.15f;        // weaker ones (down to this) only continue an ID
const int MaxMissedFrames = 30;            // was 12 in 05: ~5 s at ~6 frames a second
const double CoastSlowdownSeconds = 0.5;   // an unseen object's speed halves about every 0.35 s
const double MinOverlap = 0.3;
const int ReportEveryMs = 1000;
const std::string Window = "05b two thresholds  (Q quits)";

spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = KeepConfidence;   // YOLO hands back everything down to the weak threshold
    return options;
}

// ---------------------------------------------------------------------
//  The new idea: two passes, two thresholds, and a fading coast
// ---------------------------------------------------------------------

struct Track
{
    int id, classId;
    spotter::BoxFilter filter;   // 05's Kalman filter (centre, distance and their speeds)
    cv::Size2d boxSize;
    int missedFrames = 0;
    float lastConfidence = 0;    // the detection that last matched it
    int weakMatches = 0;         // since the last report: matches only a weak detection gave

    cv::Rect2d Box() const
    {
        cv::Point2d c = filter.Centre();
        return cv::Rect2d(c.x - boxSize.width / 2, c.y - boxSize.height / 2, boxSize.width, boxSize.height);
    }
};

struct Tracks
{
    std::vector<Track> list;
    int nextId = 1;
};

cv::Point2d Centre(const cv::Rect2d& box) { return { box.x + box.width / 2, box.y + box.height / 2 }; }

void Correct(Track& track, const spotter::Detection& d, double meters)
{
    cv::Point2d centre = Centre(d.box);
    track.filter.Correct(&centre, meters);
    track.boxSize = d.box.size();
    track.missedFrames = 0;
    track.lastConfidence = d.confidence;
    if (d.confidence < StartConfidence) track.weakMatches++;
}

// One greedy matching pass (as in 05), but only with detections in [minConfidence, maxConfidence).
void MatchPass(Tracks& tracks, const std::vector<spotter::Detection>& detections, const spotter::BoxDepthSampler& depth,
               std::vector<bool>& trackMatched, std::vector<bool>& detectionMatched, float minConfidence, float maxConfidence)
{
    while (true)
    {
        double bestOverlap = MinOverlap;
        int bestTrack = -1, bestDetection = -1;
        for (size_t t = 0; t < tracks.list.size(); t++)
            for (size_t d = 0; d < detections.size(); d++)
            {
                const spotter::Detection& detection = detections[d];
                if (detection.confidence < minConfidence || detection.confidence >= maxConfidence) continue;
                if (trackMatched[t] || detectionMatched[d] || tracks.list[t].classId != detection.classId) continue;
                double overlap = spotter::Overlap(tracks.list[t].Box(), detection.box);
                if (overlap >= bestOverlap) { bestOverlap = overlap; bestTrack = (int)t; bestDetection = (int)d; }
            }
        if (bestTrack < 0) return;
        trackMatched[bestTrack] = detectionMatched[bestDetection] = true;
        Correct(tracks.list[bestTrack], detections[bestDetection], depth.MedianMeters(detections[bestDetection].box).value_or(0));
    }
}

void Predict(Tracks& tracks, double dt)
{
    for (Track& track : tracks.list)
    {
        track.filter.Predict(dt);
        if (track.missedFrames > 0) track.filter.Fade(std::exp(-dt / CoastSlowdownSeconds));   // new: coasting slows down
    }
}

void Match(Tracks& tracks, const std::vector<spotter::Detection>& detections, const spotter::BoxDepthSampler& depth, const spotter::ClassList& classes)
{
    std::vector<bool> trackMatched(tracks.list.size(), false), detectionMatched(detections.size(), false);
    MatchPass(tracks, detections, depth, trackMatched, detectionMatched, StartConfidence, 2.0f);       // pass 1: sure ones
    MatchPass(tracks, detections, depth, trackMatched, detectionMatched, KeepConfidence, StartConfidence);   // pass 2: weak ones

    // Only SURE leftovers start a new object; weak leftovers are dropped.
    for (size_t d = 0; d < detections.size(); d++)
    {
        const spotter::Detection& detection = detections[d];
        if (detectionMatched[d] || detection.confidence < StartConfidence) continue;
        Track track;
        track.id = tracks.nextId++;
        track.classId = detection.classId;
        track.boxSize = detection.box.size();
        track.lastConfidence = detection.confidence;
        track.filter = spotter::BoxFilter(Centre(detection.box), depth.MedianMeters(detection.box).value_or(0), spotter::BoxFilter::Noise());
        std::printf("  new ID #%d %s (confidence %.2f)\n", track.id, classes.Name(track.classId).c_str(), detection.confidence);
        tracks.list.push_back(track);
    }
    for (size_t t = 0; t < trackMatched.size(); t++)
    {
        Track& track = tracks.list[t];
        if (!trackMatched[t] && ++track.missedFrames > MaxMissedFrames)
            std::printf("  dropped #%d %s: unseen for %d frames\n", track.id, classes.Name(track.classId).c_str(), track.missedFrames);
    }
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
        bool coasting = track.missedFrames > 0, weak = track.lastConfidence < StartConfidence;
        int thickness = coasting ? 1 : weak ? 2 : 3;
        cv::Scalar colour = spotter::ColourForId(track.id);
        std::string label = "#" + std::to_string(track.id) + " " + classes.Name(track.classId)
                          + (track.filter.Meters() > 0 ? cv::format(" %.2fm", track.filter.Meters()) : "")
                          + (coasting ? " (predicted)" : weak ? cv::format(" (weak %.2f)", track.lastConfidence) : "");
        cv::rectangle(picture, track.Box(), colour, thickness);
        cv::putText(picture, label, cv::Point((int)track.Box().x, std::max((int)track.Box().y - 10, 30)), cv::FONT_HERSHEY_SIMPLEX, 1.0, colour, std::min(thickness, 2));
    }
    fps.Draw(picture);
    cv::imshow(Window, picture);
}

void Report(Tracks& tracks, const spotter::ClassList& classes, const spotter::FpsMeter& fps)
{
    std::printf("%.1f fps | %zu objects\n", fps.Fps(), tracks.list.size());
    for (Track& track : tracks.list)
    {
        std::printf("  #%-3d %-12s %s, kept alive by %d weak detection(s) this second%s\n", track.id, classes.Name(track.classId).c_str(),
                    track.filter.Meters() > 0 ? cv::format("%.2f m", track.filter.Meters()).c_str() : "? m", track.weakMatches,
                    track.missedFrames > 0 ? cv::format(", unseen %d frames", track.missedFrames).c_str() : "");
        track.weakMatches = 0;
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
    spotter::FrameTimer timer;

    while (kinect.Next(frame))
    {
        double dt = timer.Step(frame);

        boxDepth.Build(frame);
        std::vector<spotter::Detection> detections = yolo.Detect(frame.colour);
        Predict(tracks, dt);
        Match(tracks, detections, boxDepth, yolo.Classes());

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
//  1. Point it at the TV (or anything YOLO is unsure of) for a minute: one ID, with weak
//     detections keeping it alive between the sure ones. 05 gave the same TV 7 IDs in 60 s.
//  2. Set KeepConfidence = StartConfidence: back to 05's behaviour (with the longer memory).
//  3. Set KeepConfidence = 0.05: weak boxes start sticking to the wrong things nearby.
//  4. Walk out of view: your box slides on, slows down and stops, instead of drifting off.
//     Try CoastSlowdownSeconds = 5 to see the drift 05 had.
