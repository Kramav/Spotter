// =====================================================================
//  04 · step 2 of 3 · Keep an ID
//
//  A "track" is an object being followed: an ID number and its last box.
//  Each frame, pair tracks with detections greedily: take the same-class
//  pair that overlaps most, pair them, repeat while some pair overlaps at
//  least MinOverlap. Paired tracks move to the new box; unpaired
//  detections become new tracks with new IDs; unpaired tracks are dropped.
//
//  New in this step: tracks, greedy matching, an ID per object.
//  Keys:  Q or Esc  quit
//  Next:  04_s3_Remember_Missing (don't drop an ID the moment YOLO misses it)
// =====================================================================

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/track.hpp"   // Overlap (step 1's), ColourForId
#include "spotter/view.hpp"

const double MinOverlap = 0.3;

struct Track
{
    int id;
    int classId;
    cv::Rect2d box;
};

spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = 0.4f;
    return options;
}

std::vector<Track> Match(const std::vector<Track>& tracks, const std::vector<spotter::Detection>& detections, int& nextId)
{
    std::vector<Track> kept;
    std::vector<bool> trackUsed(tracks.size(), false), detectionUsed(detections.size(), false);
    while (true)
    {
        double best = MinOverlap;
        int bestTrack = -1, bestDetection = -1;
        for (size_t t = 0; t < tracks.size(); t++)
            for (size_t d = 0; d < detections.size(); d++)
            {
                if (trackUsed[t] || detectionUsed[d] || tracks[t].classId != detections[d].classId) continue;
                double overlap = spotter::Overlap(tracks[t].box, detections[d].box);
                if (overlap >= best) { best = overlap; bestTrack = (int)t; bestDetection = (int)d; }
            }
        if (bestTrack < 0) break;
        trackUsed[bestTrack] = detectionUsed[bestDetection] = true;
        kept.push_back({ tracks[bestTrack].id, tracks[bestTrack].classId, detections[bestDetection].box });   // same ID, new box
    }
    for (size_t d = 0; d < detections.size(); d++)
        if (!detectionUsed[d]) kept.push_back({ nextId++, detections[d].classId, detections[d].box });       // a new object
    return kept;   // unpaired tracks are simply not kept
}

int main()
{
    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    const char* window = "04 step 2: an ID per object";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, 960, 540);
    std::vector<Track> tracks;
    int nextId = 1;
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        tracks = Match(tracks, yolo.Detect(frame.colour), nextId);
        for (const Track& t : tracks)
        {
            cv::Scalar colour = spotter::ColourForId(t.id);   // one colour per ID: a change shows
            cv::rectangle(frame.colour, t.box, colour, 3);
            cv::putText(frame.colour, "#" + std::to_string(t.id) + " " + yolo.Classes().Name(t.classId), t.box.tl() + cv::Point2d(4, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, colour, 2);
        }
        cv::imshow(window, frame.colour);
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: watch a still object for a minute. Its ID climbs now and then: whenever YOLO misses it for
// a single frame, the track is dropped, and the next detection starts a new one. Step 3 fixes that.
