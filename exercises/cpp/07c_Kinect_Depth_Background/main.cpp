// =====================================================================
//  07c · Kinect Depth Background
//
//  A fixed camera sees the same room over and over. Remember it, and
//  anything that comes in front of what you remember is new: a person,
//  a moved chair, a cup put down. No neural network, just depth:
//
//      foreground = clearly NEARER than the remembered background
//
//  The memory keeps itself up to date, per pixel:
//    - same distance as remembered  -> blend in (a slow running average, 07)
//    - clearly FARTHER for LeaveFrames -> something left: take the new value
//    - clearly NEARER for a long time -> something new stayed: take it after
//                                        AbsorbFrames (a put-down cup joins the room)
//  So it needs no empty-room start: whatever stood there at the start
//  heals a third of a second after it moves away (it reads "farther").
//
//  Window: live depth in grey, foreground in green, each blob boxed.
//  Keys:   SPACE  forget the background and start over     Q or Esc  quit
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <opencv2/opencv.hpp>

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// How much nearer than the background a pixel must be to count as foreground. Far bigger than
// the noise (a few mm, 07/07b), smaller than a hand in front of a wall.
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

const float MaxDepthMeters = 6.0f;   // 07b
const int Width = 512, Height = 424;

// ---------------------------------------------------------------------
//  The new idea: a background memory, and what's in front of it
// ---------------------------------------------------------------------

// Feed one frame (mm, 0 = no reading) into the memory. Returns the foreground mask (255 = nearer).
cv::Mat UpdateBackground(const std::vector<float>& liveMm, std::vector<float>& backgroundMm, std::vector<int>& nearerFrames, std::vector<int>& fartherFrames)
{
    cv::Mat foreground(Height, Width, CV_8U, cv::Scalar(0));
    for (int i = 0; i < Width * Height; i++)
    {
        float live = liveMm[i], remembered = backgroundMm[i];
        if (live <= 0) continue;                                         // no reading: learn nothing
        if (remembered <= 0) { backgroundMm[i] = live; continue; }       // first reading here
        if (live < remembered - NearerMeters * 1000)                     // something in front
        {
            fartherFrames[i] = 0;
            if (++nearerFrames[i] < AbsorbFrames) { foreground.data[i] = 255; continue; }
            backgroundMm[i] = live;                                      // it stayed: it's part of the room now
        }
        else if (live > remembered + NearerMeters * 1000)                // something left?
        {
            nearerFrames[i] = 0;
            if (++fartherFrames[i] < LeaveFrames) continue;
            backgroundMm[i] = live;
        }
        else backgroundMm[i] += BlendRate * (live - remembered);         // the same: blend in
        nearerFrames[i] = fartherFrames[i] = 0;
    }
    return foreground;
}

// Drop blobs smaller than MinBlobPixels. Returns the boxes of the ones left.
std::vector<cv::Rect> KeepBigBlobs(cv::Mat& mask)
{
    cv::Mat labels, stats, centroids;
    int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids);
    std::vector<cv::Rect> boxes;
    std::vector<bool> keep(count, false);
    for (int label = 1; label < count; label++)   // label 0 = the background
    {
        if (stats.at<int>(label, cv::CC_STAT_AREA) < MinBlobPixels) continue;
        keep[label] = true;
        boxes.emplace_back(stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP),
                           stats.at<int>(label, cv::CC_STAT_WIDTH), stats.at<int>(label, cv::CC_STAT_HEIGHT));
    }
    for (int i = 0; i < Width * Height; i++) mask.data[i] = keep[labels.at<int>(i / Width, i % Width)] ? 255 : 0;
    return boxes;
}

int main()
{
    // -----------------------------------------------------------------
    //  Setup: depth only
    // -----------------------------------------------------------------

    std::setvbuf(stdout, nullptr, _IONBF, 0);
    libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Warning));
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device* device = freenect2.enumerateDevices() > 0 ? freenect2.openDefaultDevice() : nullptr;
    if (device == nullptr)
    {
        std::printf("No Kinect v2 found, or it is in use. See the main README's troubleshooting table.\n");
        return 1;
    }
    libfreenect2::Freenect2Device::Config config;
    config.MaxDepth = MaxDepthMeters;
    device->setConfiguration(config);
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Depth);
    device->setIrAndDepthFrameListener(&listener);
    device->startStreams(false, true);

    std::vector<float> liveMm(Width * Height), backgroundMm(Width * Height, 0);
    std::vector<int> nearerFrames(Width * Height, 0), fartherFrames(Width * Height, 0);

    const std::string window = "07c depth background  (SPACE forgets the background, Q quits)";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, Width * 2, Height * 2);
    int frameCount = 0;

    // -----------------------------------------------------------------
    //  Main operation
    // -----------------------------------------------------------------

    libfreenect2::FrameMap frames;
    while (true)
    {
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == ' ')
        {
            std::fill(backgroundMm.begin(), backgroundMm.end(), 0.0f);
            std::fill(nearerFrames.begin(), nearerFrames.end(), 0);
            std::fill(fartherFrames.begin(), fartherFrames.end(), 0);
            std::printf("Background forgotten: the next frame is the new background.\n");
        }

        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }
        const float* frameMm = (const float*)frames[libfreenect2::Frame::Depth]->data;
        for (int i = 0; i < Width * Height; i++)
            liveMm[i] = (frameMm[i] > 0 && !std::isinf(frameMm[i])) ? frameMm[i] : 0;
        listener.release(frames);

        // --- Step 1: compare with the memory, and update it ------------------
        cv::Mat foreground = UpdateBackground(liveMm, backgroundMm, nearerFrames, fartherFrames);

        // --- Step 2: keep only blobs big enough to be something ------------
        std::vector<cv::Rect> blobs = KeepBigBlobs(foreground);

        // -----------------------------------------------------------------
        //  Output
        // -----------------------------------------------------------------

        cv::Mat picture(Height, Width, CV_8UC3);
        for (int i = 0; i < Width * Height; i++)
        {
            uchar grey = liveMm[i] > 0 ? (uchar)(255 * (1 - std::min(liveMm[i] / (MaxDepthMeters * 1000), 1.0f))) : 0;
            picture.at<cv::Vec3b>(i / Width, i % Width) = foreground.data[i] ? cv::Vec3b(0, std::max<uchar>(grey, 80), 0)
                                                                             : cv::Vec3b(grey, grey, grey);
        }
        for (const cv::Rect& blob : blobs)
        {
            cv::rectangle(picture, blob, cv::Scalar(0, 255, 255), 1);
            cv::putText(picture, cv::format("%d px", cv::countNonZero(foreground(blob))), blob.tl() + cv::Point(2, 12),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
        }
        cv::putText(picture, cv::format("%zu blob(s) in front of the background", blobs.size()), cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        cv::imshow(window, picture);

        if (++frameCount % 30 == 0)
            std::printf("%zu blob(s), %d foreground pixels\n", blobs.size(), cv::countNonZero(foreground));
    }

    device->stop();
    device->close();
    cv::destroyAllWindows();
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
//  4. Set NearerMeters = 0.02: the far wall's noise (07b) starts to sparkle. Set 0.30: a hand
//     flat on a table disappears.
//  5. Black clothes or hair often give no depth reading at all. They never turn green: the
//     method can't see what the depth camera can't.
//  6. Set MinBlobPixels = 0 and watch object edges: thin green outlines from flying pixels.
//  7. Set LeaveFrames = 1 under a spinning ceiling fan: with nobody in view, 2-6 blobs appear on
//     it (first test: blades 3 m, ceiling 4.7 m). Each gap between blades pulled the memory up to
//     the ceiling, and the next blade looked "nearer". With 10: 0 blobs. Anything that keeps
//     moving in place (a fan, curtains) is handled the same way.
