// =====================================================================
//  07 · Kinect Capture + Average
//
//  First step toward a 3D map of the room (stage B). A map is built from
//  still captures, so depth noise matters more than speed here. With the
//  camera still, the simplest noise filter of all works perfectly:
//
//    average each pixel over many frames.
//
//  Random noise shrinks with the square root of the frame count: 30 frames
//  -> about 1/5.5 of the noise. And a pixel that only SOMETIMES gets a
//  reading (a red speckle in example 01) can be filled from the frames
//  where it did.
//
//  This example measures both, instead of just claiming them:
//    - one capture:      the noise of a single frame, per pixel, by distance
//    - the next capture: the difference between two averages, which is the
//                        averaged result's REAL noise (times sqrt 2)
//
//  Windows: live depth (as in 01), and the latest averaged capture.
//  Keys:    SPACE  capture and average     W A S D  move the crosshair
//           Q or Esc  quit
//
//  Keep the camera completely still (and people out of view) while capturing.
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

// How many frames one capture averages (30 = one second). Noise falls with sqrt(CaptureFrames).
const int CaptureFrames = 30;

// A pixel only counts in the average if it had a reading in at least this fraction of the
// frames. Lower = fills more holes, but trusts pixels that are often wrong (edges, black things).
const double MinValidFraction = 0.5;

// The grey scale and crosshair, as in example 01.
const int NearMm = 500;
const int FarMm = 4500;
const int CrosshairStepPixels = 10;

const int Width = 512, Height = 424;

// The middle value of a list (it reorders the list).
double MedianOf(std::vector<double>& values)
{
    if (values.empty()) return 0;
    size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    return values[middle];
}

// Depth in mm -> grey (near = bright), red where there's no reading. As in example 01.
cv::Mat Picture(const std::vector<float>& depthMm)
{
    cv::Mat picture(Height, Width, CV_8UC3);
    for (int i = 0; i < Width * Height; i++)
    {
        float mm = depthMm[i];
        cv::Vec3b& pixel = picture.at<cv::Vec3b>(i / Width, i % Width);
        if (!(mm > 0)) { pixel = cv::Vec3b(0, 0, 255); continue; }
        double fraction = std::clamp((mm - NearMm) / (double)(FarMm - NearMm), 0.0, 1.0);
        uchar brightness = (uchar)(255 * (1.0 - fraction));
        pixel = cv::Vec3b(brightness, brightness, brightness);
    }
    return picture;
}

double HolePercent(const std::vector<float>& depthMm)
{
    int holes = 0;
    for (float mm : depthMm) if (!(mm > 0)) holes++;
    return 100.0 * holes / depthMm.size();
}

int main()
{
    // -----------------------------------------------------------------
    //  Setup (as in example 01)
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
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Depth);
    device->setIrAndDepthFrameListener(&listener);
    device->startStreams(false, true);

    // Running totals for the capture in progress, per pixel. Sum and sum of squares are enough to
    // get the mean and the spread (standard deviation) without keeping every frame.
    std::vector<double> sumMm(Width * Height), sumSquaresMm(Width * Height);
    std::vector<int> readings(Width * Height);
    int framesCaptured = -1;   // -1 = not capturing

    std::vector<float> liveMm(Width * Height);
    std::vector<float> averageMm, previousAverageMm;   // results of the last two captures (0 = hole)
    std::vector<float> spreadMm(Width * Height);       // single-frame noise per pixel, last capture

    const std::string liveWindow = "07 live  (SPACE captures, WASD crosshair, Q quits)";
    const std::string averageWindow = "07 averaged capture";
    cv::namedWindow(liveWindow, cv::WINDOW_NORMAL);
    cv::resizeWindow(liveWindow, Width * 2, Height * 2);
    int crosshairX = Width / 2, crosshairY = Height / 2;

    std::printf("Keep the camera still and press SPACE to capture %d frames.\n", CaptureFrames);

    // -----------------------------------------------------------------
    //  Main operation
    // -----------------------------------------------------------------

    libfreenect2::FrameMap frames;
    while (true)
    {
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == 'w') crosshairY -= CrosshairStepPixels;
        if (key == 's') crosshairY += CrosshairStepPixels;
        if (key == 'a') crosshairX -= CrosshairStepPixels;
        if (key == 'd') crosshairX += CrosshairStepPixels;
        crosshairX = std::clamp(crosshairX, 0, Width - 1);
        crosshairY = std::clamp(crosshairY, 0, Height - 1);
        if (key == ' ' && framesCaptured < 0)
        {
            std::fill(sumMm.begin(), sumMm.end(), 0.0);
            std::fill(sumSquaresMm.begin(), sumSquaresMm.end(), 0.0);
            std::fill(readings.begin(), readings.end(), 0);
            framesCaptured = 0;
            std::printf("Capturing...\n");
        }

        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }
        const float* frameMm = (const float*)frames[libfreenect2::Frame::Depth]->data;
        for (int i = 0; i < Width * Height; i++)
        {
            float mm = frameMm[i];
            liveMm[i] = (mm > 0 && !std::isinf(mm)) ? mm : 0;   // 0, NaN, infinity = no reading
        }
        listener.release(frames);

        // --- Step 1: add this frame to the running totals -----------------
        if (framesCaptured >= 0)
        {
            for (int i = 0; i < Width * Height; i++)
            {
                if (liveMm[i] <= 0) continue;   // holes don't count, they'd drag the average to 0
                sumMm[i] += liveMm[i];
                sumSquaresMm[i] += (double)liveMm[i] * liveMm[i];
                readings[i]++;
            }
            framesCaptured++;
        }

        // --- Step 2: capture complete -> average, and measure the noise ---------
        if (framesCaptured == CaptureFrames)
        {
            framesCaptured = -1;
            previousAverageMm = averageMm;
            averageMm.assign(Width * Height, 0);

            // Single-frame noise, in distance bands: noise grows with distance (guide §5.5).
            const char* bandNames[4] = { "0.5-1 m", "1-2 m", "2-3 m", "3+ m" };
            std::vector<double> bandSpreads[4], allSpreads;
            for (int i = 0; i < Width * Height; i++)
            {
                spreadMm[i] = 0;
                if (readings[i] < MinValidFraction * CaptureFrames) continue;   // too often a hole: leave it a hole
                double mean = sumMm[i] / readings[i];
                // Standard deviation = sqrt(mean of squares - square of mean): how far one frame
                // typically strays from the average.
                double spread = std::sqrt(std::max(0.0, sumSquaresMm[i] / readings[i] - mean * mean));
                averageMm[i] = (float)mean;
                spreadMm[i] = (float)spread;
                allSpreads.push_back(spread);
                int band = mean < 1000 ? 0 : mean < 2000 ? 1 : mean < 3000 ? 2 : 3;
                bandSpreads[band].push_back(spread);
            }

            double oneFrameNoise = MedianOf(allSpreads);
            std::printf("Captured %d frames.\n", CaptureFrames);
            std::printf("  noise of ONE frame (typical pixel): +-%.1f mm   by distance:", oneFrameNoise);
            for (int b = 0; b < 4; b++)
            {
                if (!bandSpreads[b].empty()) std::printf("  %s +-%.1f", bandNames[b], MedianOf(bandSpreads[b]));
            }
            std::printf("\n  noise of the AVERAGE, expected: +-%.1f mm  (one frame / sqrt(%d))\n",
                        oneFrameNoise / std::sqrt((double)CaptureFrames), CaptureFrames);
            std::printf("  holes: one frame %.1f%%, average %.1f%%  (a pixel needs a reading in %.0f%% of frames)\n",
                        HolePercent(liveMm), HolePercent(averageMm), 100 * MinValidFraction);

            // Two averages of the same still scene differ only by their noise. The difference of two
            // equally noisy things has sqrt(2) times their noise, so divide by sqrt(2).
            if (!previousAverageMm.empty())
            {
                std::vector<double> differences;
                for (int i = 0; i < Width * Height; i++)
                {
                    if (averageMm[i] > 0 && previousAverageMm[i] > 0)
                        differences.push_back(std::abs(averageMm[i] - previousAverageMm[i]));
                }
                // ponytail: median |difference| underestimates the standard deviation by about 1/1.48
                // for normal noise; scaled below so it compares with the numbers above.
                double measured = 1.4826 * MedianOf(differences) / std::sqrt(2.0);
                std::printf("  noise of the AVERAGE, measured against the previous capture: +-%.1f mm\n", measured);
            }
            else
            {
                std::printf("  press SPACE again (camera still!) to MEASURE the average's noise.\n");
            }

            cv::Mat averagePicture = Picture(averageMm);
            cv::namedWindow(averageWindow, cv::WINDOW_NORMAL);
            cv::resizeWindow(averageWindow, Width * 2, Height * 2);
            cv::imshow(averageWindow, averagePicture);
        }

        // -----------------------------------------------------------------
        //  Output: live view, with the crosshair readout
        // -----------------------------------------------------------------

        cv::Mat picture = Picture(liveMm);
        int i = crosshairY * Width + crosshairX;
        std::string text = liveMm[i] > 0 ? std::to_string((int)liveMm[i]) + " mm" : "no reading";
        if (!averageMm.empty() && averageMm[i] > 0)
            text += cv::format(" | avg %.1f (1 frame +-%.1f)", averageMm[i], spreadMm[i]);
        if (framesCaptured >= 0)
            cv::putText(picture, cv::format("capturing %d/%d", framesCaptured, CaptureFrames), cv::Point(10, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        cv::Scalar green(0, 255, 0);
        cv::drawMarker(picture, cv::Point(crosshairX, crosshairY), green, cv::MARKER_CROSS, 20, 1);
        cv::putText(picture, text, cv::Point(std::min(crosshairX + 8, Width - 230), std::max(crosshairY - 8, 40)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, green, 1);
        cv::imshow(liveWindow, picture);
    }

    device->stop();
    device->close();
    cv::destroyAllWindows();
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Capture twice without touching anything. Is the measured noise of the average close to
//     the expected one? If it's much bigger, something moved (or the sensor drifted as it warmed).
//  2. Compare the distance bands: how much noisier is 3+ m than 1-2 m? That's why merged
//     maps (stage C) should prefer each camera's NEAR points.
//  3. Set CaptureFrames = 5, then 120. Does the noise follow 1/sqrt(frames)? Where does
//     waiting longer stop being worth it?
//  4. Point the crosshair at a black or shiny object: big single-frame noise, and it may stay
//     a hole at MinValidFraction = 0.5. Try 0.2. Is the filled-in value believable?
//  5. Walk through the view during a capture. The average smears you into the background:
//     averaging only works for things that stay still (see docs/decisions.md D12).
//  6. Leave the Kinect running for 20 minutes, then capture again. Has the average shifted?
//     (Time-of-flight sensors drift a few mm while warming up.)
