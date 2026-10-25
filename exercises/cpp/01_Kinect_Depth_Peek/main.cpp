// =====================================================================
//  01 · Kinect Depth Peek
//
//  Read the Kinect v2 depth camera and look at what it actually measures.
//  Uses libfreenect2 (open source driver) - no Microsoft SDK.
//
//  What you'll see:
//    - a grey picture of the room: near = bright, far = dark
//    - RED pixels where the sensor got NO reading
//      (closer than ~0.5 m, too far, black / shiny / glass surfaces)
//    - a green crosshair, with the distance under it in millimetres
//
//  Keys:   W A S D   move the crosshair
//          Q or Esc  quit
//
//  Guide sections: §2 (Kinect v2 specs), §5.5 (what depth gets wrong).
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <opencv2/opencv.hpp>

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// The grey scale. Depth at NearMm is drawn white, depth at FarMm (or beyond) is drawn black.
// Try 1000 / 2000 to stretch the contrast over one metre of the room.
const int NearMm = 500;
const int FarMm = 4500;

// libfreenect2 cleans up depth with two filters: bilateral (smooths noise) and edge-aware
// (removes "flying pixels" along object edges). On the CPU they roughly halve the frame rate.
// false = faster, but noisier edges, AND no clipping to 0.5 .. 4.5 m: expect junk like 0 mm or 17 m.
const bool UseDepthFilters = true;

// How far one key press moves the crosshair, in depth pixels.
const int CrosshairStepPixels = 10;

// How often the console report prints, in milliseconds.
const int ReportEveryMs = 1000;

// The live frame rate in the window's corner is smoothed so it doesn't flicker:
// each frame moves it only (1 - FpsSmoothing) of the way toward the newest value.
// 0 = raw and jumpy, 0.9 = steady, 0.99 = very steady but slow to react.
const double FpsSmoothing = 0.9;

int main()
{
    // -----------------------------------------------------------------
    //  Setup
    // -----------------------------------------------------------------

    // Print immediately, even when the output is piped to a file (otherwise it waits in a buffer).
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // libfreenect2 is chatty. Warnings and errors are all we want to see.
    libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Warning));

    libfreenect2::Freenect2 freenect2;
    if (freenect2.enumerateDevices() == 0)
    {
        std::printf("No Kinect v2 found. Check the adapter's power light, use a USB 3.0 port,\n");
        std::printf("and check the USB driver is UsbDk or libusbK (Windows) - see README.\n");
        return 1;
    }

    // Opens the first Kinect, with the fastest depth decoder this build of libfreenect2 has.
    libfreenect2::Freenect2Device* device = freenect2.openDefaultDevice();
    if (device == nullptr)
    {
        std::printf("Found a Kinect but could not open it. Is another program using it?\n");
        return 1;
    }

    // A listener collects frames from the device. We only ask for depth.
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Depth);
    device->setIrAndDepthFrameListener(&listener);

    libfreenect2::Freenect2Device::Config config;   // defaults: 0.5 .. 4.5 m, both filters on
    config.EnableBilateralFilter = UseDepthFilters;
    config.EnableEdgeAwareFilter = UseDepthFilters;
    device->setConfiguration(config);
    device->startStreams(false, true);   // (colour, depth)
    std::printf("Kinect %s opened.\n", device->getSerialNumber().c_str());

    // The picture we draw into: 3 bytes (blue, green, red) per pixel. Sized on the first frame.
    cv::Mat picture;

    const std::string windowName = "01 Kinect depth peek  (WASD moves, Q quits)";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);   // NORMAL = you can drag the window bigger

    int crosshairX = -1, crosshairY = -1;   // centred on the first frame

    auto reportStart = std::chrono::steady_clock::now();

    // Live frame rate: the time between two shown frames, smoothed (see FpsSmoothing).
    auto lastFrameTime = std::chrono::steady_clock::now();
    double liveFps = 0;
    int framesSinceReport = 0;

    // -----------------------------------------------------------------
    //  Main operation
    // -----------------------------------------------------------------

    libfreenect2::FrameMap frames;
    while (true)
    {
        // Blocks until the next depth frame (30 per second). False = nothing for 10 s.
        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }
        libfreenect2::Frame* depth = frames[libfreenect2::Frame::Depth];
        int width = (int)depth->width;     // 512
        int height = (int)depth->height;   // 424
        framesSinceReport++;

        if (picture.empty())
        {
            picture.create(height, width, CV_8UC3);
            cv::resizeWindow(windowName, width * 2, height * 2);
            crosshairX = width / 2;
            crosshairY = height / 2;
            std::printf("Depth image: %d x %d pixels\n", width, height);
        }

        // One float per pixel = distance in millimetres. 0 (or not a number) means "no reading".
        // Row by row: pixel (x, y) lives at index  y * width + x.
        const float* depthMm = (const float*)depth->data;

        // Turn every depth value into a colour, and collect a few statistics on the way.
        int noReadingCount = 0;
        float nearestMm = 1e9f;
        float farthestMm = 0;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float mm = depthMm[y * width + x];

                if (!(mm > 0) || std::isinf(mm))   // !(mm > 0) also catches NaN
                {
                    noReadingCount++;
                    picture.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);   // OpenCV order is B, G, R: red
                    continue;
                }

                nearestMm = std::min(nearestMm, mm);
                farthestMm = std::max(farthestMm, mm);

                // 0.0 at NearMm, 1.0 at FarMm, clamped so values outside the range don't wrap around.
                double fraction = (mm - NearMm) / (double)(FarMm - NearMm);
                fraction = std::clamp(fraction, 0.0, 1.0);
                uchar brightness = (uchar)(255 * (1.0 - fraction));
                picture.at<cv::Vec3b>(y, x) = cv::Vec3b(brightness, brightness, brightness);
            }
        }

        float crosshairMm = depthMm[crosshairY * width + crosshairX];

        // Done with this frame's data. Hand it back, or the listener stops delivering new ones.
        listener.release(frames);

        // -------------------------------------------------------------
        //  Output
        // -------------------------------------------------------------

        bool crosshairHasReading = crosshairMm > 0 && !std::isinf(crosshairMm);
        std::string crosshairText = crosshairHasReading ? std::to_string((int)crosshairMm) + " mm" : "no reading";

        cv::Scalar green(0, 255, 0);
        cv::drawMarker(picture, cv::Point(crosshairX, crosshairY), green, cv::MARKER_CROSS, 20, 1);
        cv::putText(picture, crosshairText, cv::Point(crosshairX + 8, crosshairY - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, green, 1);
        // Live frame rate, bottom-left corner.
        auto frameTime = std::chrono::steady_clock::now();
        double newestFps = 1.0 / std::chrono::duration<double>(frameTime - lastFrameTime).count();
        lastFrameTime = frameTime;
        liveFps = (liveFps == 0) ? newestFps : FpsSmoothing * liveFps + (1 - FpsSmoothing) * newestFps;
        cv::putText(picture, cv::format("%.1f fps", liveFps), cv::Point(10, picture.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);   // yellow
        cv::imshow(windowName, picture);

        auto now = std::chrono::steady_clock::now();
        long long elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - reportStart).count();
        if (elapsedMs >= ReportEveryMs)
        {
            double framesPerSecond = framesSinceReport * 1000.0 / elapsedMs;
            double noReadingPercent = 100.0 * noReadingCount / (width * height);
            std::printf("%4.1f fps | crosshair (%d,%d) = %-10s | nearest %d mm, farthest %d mm | no reading: %.1f%%\n",
                        framesPerSecond, crosshairX, crosshairY, crosshairText.c_str(),
                        (int)nearestMm, (int)farthestMm, noReadingPercent);
            reportStart = now;
            framesSinceReport = 0;
        }

        // waitKey(1) also lets OpenCV redraw the window, so it must run every pass of the loop.
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;   // 27 = Esc
        if (key == 'w') crosshairY -= CrosshairStepPixels;
        if (key == 's') crosshairY += CrosshairStepPixels;
        if (key == 'a') crosshairX -= CrosshairStepPixels;
        if (key == 'd') crosshairX += CrosshairStepPixels;
        crosshairX = std::clamp(crosshairX, 0, width - 1);
        crosshairY = std::clamp(crosshairY, 0, height - 1);
    }

    device->stop();
    device->close();
    cv::destroyAllWindows();
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Put your hand closer and closer to the sensor. Below ~50 cm it turns red, not brighter:
//     inside the minimum range you get holes, not small numbers (guide §5.5).
//  2. Aim the crosshair at a black monitor, a window or a mirror. Watch "no reading" climb.
//  3. Turn the lights off. Depth keeps working: the Kinect lights the room with its own infrared.
//  4. Set NearMm = 1000 and FarMm = 2000 to see fine depth steps (and the noise) on one object.
//  5. Hold the crosshair on a wall and watch the console: how much does one pixel jitter
//     from frame to frame at 1 m? At 3 m?
//  6. Compare with the C# version (exercises/csharp/01): libfreenect2 decodes depth itself,
//     so its holes and noise are not identical to the Microsoft SDK's.
//  7. Replace the grey with false colour: try
//     cv::applyColorMap(picture, picture, cv::COLORMAP_JET) before imshow.
