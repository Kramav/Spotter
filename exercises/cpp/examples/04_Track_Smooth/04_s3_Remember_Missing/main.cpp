// =====================================================================
//  04 · step 3 of 3 · Remember an object YOLO missed
//
//  Step 2 dropped a track the first frame YOLO didn't see it. Now each
//  track counts the frames in a row it went unmatched, and is dropped only
//  after MaxMissedFrames. A missed object keeps its ID and comes back to it.
//  Unseen tracks aren't drawn.
//
//  New in this step: missedFrames, MaxMissedFrames.
//  Keys:  Q or Esc  quit
//  Next:  04_Track_Smooth (the full example smooths each object's distance)
// =====================================================================

#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/track.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "Q or Esc  quit";

const double MinOverlap = 0.3;
const int MaxMissedFrames = 12;   // about 2 s at ~6 frames per second

struct Track
{
    int id;
    int classId;
    cv::Rect2d box;
    int missedFrames = 0;
};

//spotter::YoloDetector::Options YoloSettings(){
//
//@brief - the detector's options: yolo26n.onnx and its names file from the repo and a minimum confidence. O(1).
spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = 0.4f;
    return options;
}

//void Match(std::vector<Track>& tracks, const std::vector<spotter::Detection>& detections, int& nextId){
//
//@param1 - tracks: the tracks so far; updated in place
//@param2 - detections: this frame's detections
//@param3 - nextId: the next free ID; used up for each new object
//
//@brief - greedy matching as in step 2 (the pair overlapping most, at least MinOverlap, repeated). Matched tracks
//         take the new box and forget being missed; unmatched detections start new IDs; unmatched tracks count a
//         missed frame and are dropped after MaxMissedFrames. O(T x D x min(T, D)).
void Match(std::vector<Track>& tracks, const std::vector<spotter::Detection>& detections, int& nextId)
{
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
        tracks[bestTrack].box = detections[bestDetection].box;
        tracks[bestTrack].missedFrames = 0;
    }
    for (size_t t = 0; t < trackUsed.size(); t++)
        if (!trackUsed[t]) tracks[t].missedFrames++;                                    // new: count, don't drop
    for (size_t d = 0; d < detections.size(); d++)
        if (!detectionUsed[d]) tracks.push_back({ nextId++, detections[d].classId, detections[d].box });
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [](const Track& t) { return t.missedFrames > MaxMissedFrames; }),
                 tracks.end());                                                         // new: drop only after a while
}

//int main(){
//
//@brief - the program: 04 · step 3 of 3 · Remember an object YOLO missed. Step 2 dropped a track the first frame
//         YOLO didn't see it. Now each track counts the frames in a row it went unmatched, and is dropped only
//         after MaxMissedFrames. A missed object keeps its ID and comes back to it. Unseen tracks aren't drawn.
//         Setup, main loop and clean-up are laid out in the file's sections. Returns 0 on a normal exit, 1 if it
//         couldn't start.
int main()
{
    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    const char* window = "04 step 3: IDs survive a missed frame";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, 960, 540);
    std::vector<Track> tracks;
    int nextId = 1;
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        Match(tracks, yolo.Detect(frame.colour), nextId);
        for (const Track& t : tracks)
        {
            if (t.missedFrames > 0) continue;   // only what is visible right now
            cv::Scalar colour = spotter::ColourForId(t.id);
            cv::rectangle(frame.colour, t.box, colour, 3);
            cv::putText(frame.colour, "#" + std::to_string(t.id) + " " + yolo.Classes().Name(t.classId), t.box.tl() + cv::Point2d(4, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, colour, 2);
        }
        cv::imshow(window, spotter::WithKeys(frame.colour, Keys));
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: walk behind a chair for a second and come back: same ID. Stay hidden longer than ~2 s:
// new ID. Set MaxMissedFrames = 0 and it behaves like step 2 again.
