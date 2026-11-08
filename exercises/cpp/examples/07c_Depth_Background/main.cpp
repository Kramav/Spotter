// =====================================================================
//  07c · Depth Background
//
//  A fixed camera sees the same room over and over. Remember it, and
//  anything that comes in front of what you remember is new: a person,
//  a moved chair, a cup put down. No neural network, just depth:
//
//      foreground = clearly NEARER than the remembered background
//
//  The memory keeps itself up to date, per pixel:
//    - same distance as remembered     -> blend in (a slow running average, 07)
//    - clearly FARTHER for LeaveFrames -> something left: take the new value
//    - clearly NEARER for AbsorbFrames -> something new stayed: take it
//                                         (a put-down cup joins the room)
//  So it needs no empty-room start: whatever stood there at the start
//  heals a third of a second after it moves away (it reads "farther").
//
//  Window: live depth in grey, foreground in green, each blob boxed.
//  Keys:   SPACE  forget the background and start over     Q or Esc  quit
// =====================================================================

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// How much nearer than the background a pixel must be to count as foreground. Far bigger than
// the noise (a few mm, 07), smaller than a hand in front of a wall.
const float NearerMeters = 0.10f;

// How long something new must stay put before it becomes part of the room. In frames: ~30 a second.
const int AbsorbFrames = 300;

// How long something must read farther before it counts as gone. One frame isn't enough: a
// ceiling fan's blades uncover the ceiling for a moment, the memory would jump to it, and the
// next blade would show as "new". With 10 frames the fan just becomes part of the room.
const int LeaveFrames = 10;

// How fast the background blends towards what it sees (0..1 per frame). Small = steadier.
const float BlendRate = 0.05f;

// Foreground blobs smaller than this are noise: flying pixels at edges (08), flicker.
const int MinBlobPixels = 500;

const float MaxDepthMeters = 6.0f;   // 07's far range
const std::string Window = "07c depth background  (SPACE forgets the background, Q quits)";

// ---------------------------------------------------------------------
//  The new idea: a background memory, and what's in front of it
// ---------------------------------------------------------------------

struct BackgroundMemory
{
    std::vector<float> depthMm;                     // the remembered room, 0 = nothing yet
    std::vector<int> nearerFrames, fartherFrames;   // how long each pixel has disagreed

    void Forget(size_t pixels)
    {
        depthMm.assign(pixels, 0);
        nearerFrames.assign(pixels, 0);
        fartherFrames.assign(pixels, 0);
    }
};

// Feed one frame (mm, 0 = no reading) into the memory. Returns the foreground mask (255 = nearer).
cv::Mat UpdateBackground(const cv::Mat& liveMm, BackgroundMemory& memory)
{
    if (memory.depthMm.size() != liveMm.total()) memory.Forget(liveMm.total());
    cv::Mat foreground(liveMm.size(), CV_8U, cv::Scalar(0));
    const float* live = liveMm.ptr<float>();
    for (size_t i = 0; i < liveMm.total(); i++)
    {
        float& remembered = memory.depthMm[i];
        if (live[i] <= 0) continue;                                      // no reading: learn nothing
        if (remembered <= 0) { remembered = live[i]; continue; }         // first reading here
        if (live[i] < remembered - NearerMeters * 1000)                  // something in front
        {
            memory.fartherFrames[i] = 0;
            if (++memory.nearerFrames[i] < AbsorbFrames) { foreground.data[i] = 255; continue; }
            remembered = live[i];                                        // it stayed: it's part of the room now
        }
        else if (live[i] > remembered + NearerMeters * 1000)             // something left?
        {
            memory.nearerFrames[i] = 0;
            if (++memory.fartherFrames[i] < LeaveFrames) continue;
            remembered = live[i];
        }
        else remembered += BlendRate * (live[i] - remembered);           // the same: blend in
        memory.nearerFrames[i] = memory.fartherFrames[i] = 0;
    }
    return foreground;
}

// Drop blobs smaller than MinBlobPixels from the mask. Returns the boxes of the ones left.
std::vector<cv::Rect> KeepBigBlobs(cv::Mat& mask)
{
    cv::Mat labels, stats, centroids;
    int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids);
    std::vector<cv::Rect> boxes;
    std::vector<bool> keep(count, false);
    for (int label = 1; label < count; label++)   // label 0 = everything else
    {
        if (stats.at<int>(label, cv::CC_STAT_AREA) < MinBlobPixels) continue;
        keep[label] = true;
        boxes.emplace_back(stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP),
                           stats.at<int>(label, cv::CC_STAT_WIDTH), stats.at<int>(label, cv::CC_STAT_HEIGHT));
    }
    for (int i = 0; i < (int)mask.total(); i++) mask.data[i] = keep[labels.at<int>(i)] ? 255 : 0;
    return boxes;
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

void Show(const cv::Mat& liveMm, const cv::Mat& foreground, const std::vector<cv::Rect>& blobs)
{
    // Grey over 0..MaxDepth, black where there's no reading, foreground green (at least a bit bright).
    cv::Mat picture(liveMm.size(), CV_8UC3);
    for (int i = 0; i < (int)liveMm.total(); i++)
    {
        float mm = liveMm.at<float>(i);
        uchar grey = mm > 0 ? (uchar)(255 * (1 - std::min(mm / (MaxDepthMeters * 1000), 1.0f))) : 0;
        picture.at<cv::Vec3b>(i) = foreground.data[i] ? cv::Vec3b(0, std::max<uchar>(grey, 80), 0) : cv::Vec3b(grey, grey, grey);
    }
    const cv::Scalar yellow(0, 255, 255);
    for (const cv::Rect& blob : blobs)
    {
        cv::rectangle(picture, blob, yellow, 1);
        cv::putText(picture, cv::format("%d px", cv::countNonZero(foreground(blob))), blob.tl() + cv::Point(2, 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, yellow, 1);
    }
    cv::putText(picture, cv::format("%zu blob(s) in front of the background", blobs.size()), cv::Point(10, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    cv::imshow(Window, picture);
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    spotter::KinectSource::Options kinectOptions;
    kinectOptions.colour = false;
    kinectOptions.maxDepthMeters = MaxDepthMeters;
    spotter::KinectSource kinect(kinectOptions);
    if (!kinect.IsOpen()) return 1;

    cv::namedWindow(Window, cv::WINDOW_NORMAL);
    cv::resizeWindow(Window, 1024, 848);
    BackgroundMemory memory;
    spotter::Every report(1000);
    spotter::RgbdFrame frame;

    while (kinect.Next(frame))
    {
        cv::Mat foreground = UpdateBackground(frame.depthMm, memory);   // step 1: compare with the memory, update it
        std::vector<cv::Rect> blobs = KeepBigBlobs(foreground);        // step 2: only blobs big enough to be something
        Show(frame.depthMm, foreground, blobs);
        if (report.Due()) std::printf("%zu blob(s), %d foreground pixels\n", blobs.size(), cv::countNonZero(foreground));

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == ' ')
        {
            memory.Forget(frame.depthMm.total());
            std::printf("Background forgotten: the next frame is the new background.\n");
        }
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Start it with you in view: you ARE the background. Step away: the wall behind reads
//     "farther" for LeaveFrames, so your spot heals within a third of a second.
//  2. Stand still. After AbsorbFrames (~10 s) you fade into the background; move and you turn
//     green again. 07d stops this for people by asking YOLO who is a person.
//  3. Put a cup on a table: green for ~10 s, then it's part of the room. Pick it up again:
//     the hand and cup are green, and the table under it is taken back after LeaveFrames.
//  4. Set NearerMeters = 0.02: the far wall's noise (07) starts to sparkle. Set 0.30: a hand
//     flat on a table disappears.
//  5. Black clothes or hair often give no depth reading at all. They never turn green: the
//     method can't see what the depth camera can't.
//  6. Set MinBlobPixels = 0 and watch object edges: thin green outlines from flying pixels.
//  7. Set LeaveFrames = 1 under a spinning ceiling fan: with nobody in view, 2-6 blobs appear on
//     it. Each gap between blades pulled the memory up to the ceiling, and the next blade looked
//     "nearer". With 10: 0 blobs. Anything that keeps moving in place is handled the same way.
