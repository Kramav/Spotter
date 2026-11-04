// =====================================================================
//  Demo 5 · How Tall Are You?         (presentation; built from 03)
//
//  Step in front of the Kinect and your height appears over your head,
//  in metres, from depth alone (no AI, works in the dark). The tallest
//  visitor so far stays in the corner: a number only, no picture.
//
//  As in 03: remember the empty view's depth, everything clearly nearer
//  is a blob. New: the height is the 3D distance from the blob's top row
//  to its bottom row, each with its own depth, so it holds when the
//  Kinect is tilted (03 multiplied the box's pixel height by one depth).
//
//  Set-up: the Kinect at about chest height, visitors 2.5-3.5 m away so
//  head AND feet are in view (mark the spot). Start it with nobody in
//  view; B re-records the empty view. Nothing is saved.
//
//  Keys:   B  re-record the background (clear the view first)
//          T  reset "tallest today"     F  fullscreen on/off     Q or Esc  quit
// =====================================================================

#include <algorithm>
#include <chrono>
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

// A pixel is foreground when it is at least this much closer than the background.
// Too small: sensor noise shows up as blobs. Too big: a hand flat on a table disappears.
const int ForegroundThresholdMm = 100;

// Blobs with fewer pixels than this are ignored (a person at 3 m covers a few thousand).
const int MinBlobPixels = 400;

// Size of the "opening" filter that removes isolated foreground speckles, in pixels.
// 1 = no cleanup. Try 1, then 5, then 11 and watch the green.
const int CleanupSizePixels = 5;

// Seconds to step out of view before the background is recorded at startup.
const int SecondsBeforeBackground = 3;

// The grey scale for the picture, as in example 01.
const int NearMm = 500;
const int FarMm = 4500;

// How often the console report prints, in milliseconds.
const int ReportEveryMs = 1000;

// The live frame rate in the window's corner is smoothed so it doesn't flicker:
// each frame moves it only (1 - FpsSmoothing) of the way toward the newest value.
// 0 = raw and jumpy, 0.9 = steady, 0.99 = very steady but slow to react.
const double FpsSmoothing = 0.9;

// Heights outside this range are not counted for "tallest today" (two people merged, a chair).
const double MinPersonMeters = 1.0, MaxPersonMeters = 2.3;

const bool StartFullscreen = true;
// The keys along the bottom of the screen, short (the user's ask).
const char* KeyLegend = "b - relearn room   t - reset tallest   f - fullscreen   q - quit";

// Middle value of a list. For an even count, the average of the two middle values (like numpy's median).
float Median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) return values[middle];
    return (values[middle - 1] + values[middle]) / 2;
}

// libfreenect2 marks "no reading" as 0, and sometimes NaN or infinity. Treat all of them as 0.
float CleanDepth(float mm)
{
    return (mm > 0 && !std::isinf(mm)) ? mm : 0;   // mm > 0 is false for NaN
}

void ToggleFullscreen(const std::string& window)
{
    bool full = cv::getWindowProperty(window, cv::WND_PROP_FULLSCREEN) == cv::WINDOW_FULLSCREEN;
    cv::setWindowProperty(window, cv::WND_PROP_FULLSCREEN, full ? cv::WINDOW_NORMAL : cv::WINDOW_FULLSCREEN);
}

int main()
{
    // -----------------------------------------------------------------
    //  Setup
    // -----------------------------------------------------------------

    std::setvbuf(stdout, nullptr, _IONBF, 0);   // print immediately, even when piped (see example 01)
    libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Warning));

    libfreenect2::Freenect2 freenect2;
    if (freenect2.enumerateDevices() == 0)
    {
        std::printf("No Kinect v2 found. Check the adapter's power light, use a USB 3.0 port,\n");
        std::printf("and check the USB driver is UsbDk or libusbK (Windows) - see README.\n");
        return 1;
    }
    libfreenect2::Freenect2Device* device = freenect2.openDefaultDevice();
    if (device == nullptr)
    {
        std::printf("Found a Kinect but could not open it. Is another program using it?\n");
        return 1;
    }

    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Depth);
    device->setIrAndDepthFrameListener(&listener);
    device->startStreams(false, true);   // (colour, depth)

    // Depth intrinsics for the real-size math: focal lengths and principal point, in pixels.
    // libfreenect2 reads the factory calibration from the Kinect while starting the streams.
    libfreenect2::Freenect2Device::IrCameraParams intrinsics = device->getIrCameraParams();
    std::printf("Depth intrinsics: fx %.1f  fy %.1f  cx %.1f  cy %.1f  (pixels)\n",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy);

    const int depthWidth = 512, depthHeight = 424;
    std::vector<float> depthMm(depthWidth * depthHeight);        // this frame, millimetres, 0 = no reading
    std::vector<float> backgroundMm(depthWidth * depthHeight);   // the remembered empty room

    cv::Mat foregroundMask(depthHeight, depthWidth, CV_8UC1);    // 255 = foreground, 0 = not
    cv::Mat cleanedMask;
    cv::Mat picture(depthHeight, depthWidth, CV_8UC3);
    cv::Mat cleanupShape = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(CleanupSizePixels, CleanupSizePixels));

    bool backgroundRecorded = false;
    auto backgroundCountdownStart = std::chrono::steady_clock::now();
    std::printf("Recording the background in %d s: step out of view.\n", SecondsBeforeBackground);

    const std::string windowName = "Demo 5 - how tall are you";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, depthWidth * 2, depthHeight * 2);
    if (StartFullscreen) ToggleFullscreen(windowName);
    double tallestToday = 0;

    auto reportStart = std::chrono::steady_clock::now();

    // Live frame rate: the time between two shown frames, smoothed (see FpsSmoothing).
    auto lastFrameTime = std::chrono::steady_clock::now();
    double liveFps = 0;

    // -----------------------------------------------------------------
    //  Main operation
    // -----------------------------------------------------------------

    libfreenect2::FrameMap frames;
    while (true)
    {
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;   // 27 = Esc
        if (key == 'f') ToggleFullscreen(windowName);
        if (key == 't') tallestToday = 0;
        if (key == 'b')
        {
            backgroundRecorded = false;
            backgroundCountdownStart = std::chrono::steady_clock::now();
            std::printf("Recording the background in %d s: step out of view.\n", SecondsBeforeBackground);
        }

        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }
        const float* frameMm = (const float*)frames[libfreenect2::Frame::Depth]->data;
        for (int i = 0; i < depthWidth * depthHeight; i++) depthMm[i] = CleanDepth(frameMm[i]);
        listener.release(frames);

        // --- Step 1: remember the empty room --------------------------------
        double secondsSinceCountdown = std::chrono::duration<double>(std::chrono::steady_clock::now() - backgroundCountdownStart).count();
        if (!backgroundRecorded && secondsSinceCountdown >= SecondsBeforeBackground)
        {
            backgroundMm = depthMm;
            backgroundRecorded = true;
            std::printf("Background recorded. Walk in!\n");
        }

        // --- Step 2: foreground = closer than the background ----------------
        // Same rule as the guide:  (background - depth) > threshold  AND  depth has a reading.
        // Where the background had no reading (0), background - depth is negative, so those
        // pixels never count as foreground. Experiment idea 3 is about that.
        for (int y = 0; y < depthHeight; y++)
        {
            for (int x = 0; x < depthWidth; x++)
            {
                int i = y * depthWidth + x;
                float closerByMm = backgroundMm[i] - depthMm[i];
                bool isForeground = backgroundRecorded && depthMm[i] > 0 && closerByMm > ForegroundThresholdMm;
                foregroundMask.at<uchar>(y, x) = isForeground ? 255 : 0;

                // Draw this pixel: green if foreground, otherwise grey by distance, black for no reading.
                uchar brightness = 0;
                if (depthMm[i] > 0)
                {
                    double fraction = std::clamp((depthMm[i] - NearMm) / (double)(FarMm - NearMm), 0.0, 1.0);
                    brightness = (uchar)(255 * (1.0 - fraction));
                }
                if (isForeground)
                    picture.at<cv::Vec3b>(y, x) = cv::Vec3b(0, brightness, 0);   // B, G, R: green, still brighter when nearer
                else
                    picture.at<cv::Vec3b>(y, x) = cv::Vec3b(brightness, brightness, brightness);
            }
        }

        // --- Step 3: clean up and group into blobs ---------------------------
        // "Opening" = shrink every foreground region a little, then grow it back. Specks smaller
        // than the shape vanish in the shrink and never come back. Big regions keep their outline.
        cv::morphologyEx(foregroundMask, cleanedMask, cv::MORPH_OPEN, cleanupShape);

        // Connected components: label every group of touching foreground pixels 1, 2, 3...
        // Label 0 is always the background (all the zero pixels), so it is skipped below.
        cv::Mat labels, stats, centroids;
        int labelCount = cv::connectedComponentsWithStats(cleanedMask, labels, stats, centroids, 8);

        // --- Step 4: measure each blob ---------------------------------------
        std::vector<std::string> reportLines;
        for (int label = 1; label < labelCount; label++)
        {
            int area = stats.at<int>(label, cv::CC_STAT_AREA);
            if (area < MinBlobPixels) continue;
            cv::Rect box(stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP),
                         stats.at<int>(label, cv::CC_STAT_WIDTH), stats.at<int>(label, cv::CC_STAT_HEIGHT));

            // Median depth of the pixels that belong to THIS blob (not the whole rectangle around it).
            std::vector<float> blobDepthsMm;
            for (int y = box.y; y < box.y + box.height; y++)
            {
                for (int x = box.x; x < box.x + box.width; x++)
                {
                    float mm = depthMm[y * depthWidth + x];
                    if (labels.at<int>(y, x) == label && mm > 0) blobDepthsMm.push_back(mm);
                }
            }
            if (blobDepthsMm.empty()) continue;
            double zMeters = Median(blobDepthsMm) / 1000.0;

            // Real size (§5.3): an object w pixels wide at depth Z is about  w * Z / fx  wide.
            double widthMeters = box.width * zMeters / intrinsics.fx;

            // Height (new): the blob's top row and bottom row as 3D points, each from its own depth
            // (the pinhole maths below, per pixel), averaged along the row. Rows with fewer than 3
            // readings are skipped: those are ragged edges.
            auto RowPoint = [&](int y, cv::Point3d& point)
            {
                cv::Point3d sum(0, 0, 0);
                int count = 0;
                for (int x = box.x; x < box.x + box.width; x++)
                {
                    float mm = depthMm[y * depthWidth + x];
                    if (labels.at<int>(y, x) != label || mm <= 0) continue;
                    double z = mm / 1000.0;
                    sum += cv::Point3d((x - intrinsics.cx) * z / intrinsics.fx, (y - intrinsics.cy) * z / intrinsics.fy, z);
                    count++;
                }
                if (count >= 3) point = sum / count;
                return count >= 3;
            };
            cv::Point3d top, bottom;
            int topRow = box.y, bottomRow = box.y + box.height - 1;
            while (topRow < bottomRow && !RowPoint(topRow, top)) topRow++;
            while (bottomRow > topRow && !RowPoint(bottomRow, bottom)) bottomRow--;
            double heightMeters = cv::norm(top - bottom);
            bool wholeInView = box.y > 2 && box.y + box.height < depthHeight - 2;   // head and feet both seen
            if (wholeInView && heightMeters >= MinPersonMeters && heightMeters <= MaxPersonMeters)
                tallestToday = std::max(tallestToday, heightMeters);

            // Where the blob's centre is: pinhole model on its centroid (x right, y down).
            double xMeters = (centroids.at<double>(label, 0) - intrinsics.cx) * zMeters / intrinsics.fx;
            double yMeters = (centroids.at<double>(label, 1) - intrinsics.cy) * zMeters / intrinsics.fy;

            char text[64], line[160];
            std::snprintf(text, sizeof text, "%.2fm  %.2f x %.2fm%s", zMeters, widthMeters, heightMeters, wholeInView ? "" : " (cut off)");
            std::snprintf(line, sizeof line, "  blob %2d: %6d px  %-22s  centre x%+.2f y%+.2f", label, area, text, xMeters, yMeters);
            reportLines.push_back(line);

            // -----------------------------------------------------------------
            //  Output: draw
            // -----------------------------------------------------------------
            cv::Scalar yellow(0, 255, 255);
            cv::rectangle(picture, box, yellow, 1);
            std::string big = wholeInView ? cv::format("%.2f m", heightMeters) : "step back";
            cv::putText(picture, big, cv::Point(box.x, std::max(box.y - 8, 24)), cv::FONT_HERSHEY_SIMPLEX, 0.9, yellow, 2);
        }

        // -----------------------------------------------------------------
        //  Output
        // -----------------------------------------------------------------
        if (!backgroundRecorded)
        {
            std::string countdown = "background in " + std::to_string((int)std::ceil(SecondsBeforeBackground - secondsSinceCountdown)) + " s";
            cv::putText(picture, countdown, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }
        if (tallestToday > 0)
            cv::putText(picture, cv::format("tallest today: %.2f m", tallestToday), cv::Point(depthWidth - 215, 22),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        cv::putText(picture, "live only: nothing is recorded", cv::Point(depthWidth - 200, depthHeight - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(160, 160, 160), 1);
        // Live frame rate, bottom-left corner.
        auto frameTime = std::chrono::steady_clock::now();
        double newestFps = 1.0 / std::chrono::duration<double>(frameTime - lastFrameTime).count();
        lastFrameTime = frameTime;
        liveFps = (liveFps == 0) ? newestFps : FpsSmoothing * liveFps + (1 - FpsSmoothing) * newestFps;
        cv::putText(picture, cv::format("%.1f fps", liveFps), cv::Point(10, picture.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);   // yellow
        // The keys, in a strip under the picture (new: the user's ask).
        cv::Mat keys(22, picture.cols, CV_8UC3, cv::Scalar(20, 20, 20)), shown;
        cv::putText(keys, KeyLegend, cv::Point(8, 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(220, 220, 220), 1);
        cv::vconcat(picture, keys, shown);
        cv::imshow(windowName, shown);

        auto now = std::chrono::steady_clock::now();
        if (backgroundRecorded && now - reportStart >= std::chrono::milliseconds(ReportEveryMs))
        {
            std::printf("foreground pixels: %d raw, %d after cleanup | %d groups, %d big enough\n",
                        cv::countNonZero(foregroundMask), cv::countNonZero(cleanedMask),
                        labelCount - 1, (int)reportLines.size());
            for (const std::string& l : reportLines) std::printf("%s\n", l.c_str());
            reportStart = now;
        }
    }

    device->stop();
    device->close();
    cv::destroyAllWindows();
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Check it on someone who knows their height. Usually a few cm short: hair gives no depth
//     reading, and the soles of the feet are within ForegroundThresholdMm of the floor.
//  2. Tilt the Kinect down 15 deg and measure again: the 3D height barely changes. The console's
//     "x" size (03's box method) does.
//  3. Raise an arm: you "grow". The blob measures the outline, not the person.
//  4. Two people side by side, touching: one blob, one height, and no "tallest" (MaxPersonMeters).
//  5. Turn the lights off: it still works. Then try Demo 3 in the dark.
