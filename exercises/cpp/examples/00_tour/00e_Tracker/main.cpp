// =====================================================================
//  00e · Tour: Tracker and BoxFilter                 (no Kinect needed)
//
//  What it's for: following detected objects over time. Each object keeps
//  an ID, a smooth box and distance, and a speed, even when the detector
//  misses it for a while. Examples 05c, 06 and the demos use it.
//
//  How it works (spotter/src/track.cpp):
//    BoxFilter   a Kalman filter per object (example 05): state = box centre
//                x, y (pixels) and distance (m), plus their speeds.
//    Every frame   Predict(dt): every box moves on by its speed. An object
//                  that isn't seen "coasts", and its speed fades (05b).
//    Per detector batch   Match(): pair detections with PREDICTED boxes by
//                  overlap, same class, sure detections (0.40+) first, then
//                  unsure ones (0.15+), which may only continue an ID (05b).
//                  Unpaired sure detections start new IDs.
//    Every frame   Remeasure(): objects seen in the latest batch get a fresh
//                  distance from this frame's depth; objects unseen for
//                  maxUnseenSeconds are dropped.
//
//  This program feeds it made-up detections (a person walking right, a TV
//  the detector misses for a while, an unsure false alarm) and prints what
//  it does with them, one line per detector batch.
// =====================================================================

#include <cstdio>

#include "spotter/track.hpp"

// A depth picture that is 1 m on the left and 3.5 m on the right, mapped simply into colour.
spotter::BoxDepthSampler MadeUpDepth()
{
    cv::Mat depthMm(424, 512, CV_32F);
    std::vector<int> colourIndex(512 * 424);
    for (int row = 0; row < 424; row++)
        for (int column = 0; column < 512; column++)
        {
            depthMm.at<float>(row, column) = 1000 + column * 5.0f;
            colourIndex[row * 512 + column] = (int)(row * 2.55) * 1920 + (int)(column * 3.75);
        }
    spotter::BoxDepthSampler depth;
    depth.Build(depthMm.ptr<float>(), colourIndex, 1920, 1080);
    return depth;
}

std::vector<spotter::Detection> Batch(int frame)
{
    int batch = frame / 5;
    std::vector<spotter::Detection> found;
    found.push_back({ 0, batch % 4 == 3 ? 0.2f : 0.7f, cv::Rect2d(200 + frame * 10, 300, 200, 500) });   // walking right; every 4th batch unsure
    if (batch < 6 || batch > 9) found.push_back({ 62, 0.55f, cv::Rect2d(1400, 200, 300, 180) });         // a TV, missed for batches 6-9
    found.push_back({ 41, 0.2f, cv::Rect2d(100, 100, 50, 50) });                                         // an unsure "cup" on nothing
    return found;
}

void PrintBatch(int frame, const spotter::Tracker& tracker)
{
    std::printf("t=%.2fs ", frame / 30.0);
    for (const spotter::Track& t : tracker.Tracks())
        std::printf(" | #%d %s x%.0f %.2fm %+.2fm/s%s", t.id, t.classId == 0 ? "person" : "tv", t.filter.Centre().x, t.filter.Meters(),
                    t.filter.MetersPerSecond(), t.Coasting() ? cv::format(" COASTING %.1fs", t.secondsUnseen).c_str() : "");
    std::printf("\n");
}

int main()
{
    spotter::BoxDepthSampler depth = MadeUpDepth();
    spotter::Tracker::Options options;
    options.maxUnseenSeconds = 2.0;
    spotter::Tracker tracker(options);
    std::printf("A detector batch every 5th frame (30 fps), as with YOLO on its own thread.\n");
    for (int frame = 0; frame < 90; frame++)
    {
        tracker.Predict(1.0 / 30);
        if (frame % 5 == 0) tracker.Match(Batch(frame), depth);
        tracker.Remeasure(depth);
        for (const spotter::TrackEvent& e : tracker.TakeEvents())
            std::printf("  -> %s ID #%d\n", e.started ? "new" : "dropped", e.id);
        if (frame % 5 == 0) PrintBatch(frame, tracker);
    }
    std::printf("\nThe person's distance grows as it walks right (the made-up depth does too), and its speed\n"
                "settles at +0.40 m/s. The TV coasts while missed and keeps ID #2. The unsure cup never\n"
                "gets an ID, and the person's unsure batches don't break its ID.\n\nPress Enter to close.\n");
    std::getchar();
    return 0;
}
