// =====================================================================
//  07b · Kinect Far Range
//
//  Every example so far stops at 4.5 m. That isn't the sensor: it's
//  libfreenect2's default cut-off (Config::MaxDepth). The time-of-flight
//  camera can measure further, just worse. How much worse?
//
//  This example raises the cut-off to MaxDepthMeters, captures frames
//  like 07, and reports per 1 m band:
//    - how much of the view lands in that band
//    - the noise of one frame (mm, and as % of the distance)
//    - how flaky it is: pixels that read in some frames but not most
//
//  Window: live depth. Grey up to 4.5 m (what we had), blue beyond it
//          (what the higher cut-off adds). Red = no reading.
//  Keys:   SPACE  capture and report     W A S D  move the crosshair
//          Q or Esc  quit
//
//  Point the Kinect down a hallway or across the longest room you have,
//  keep it still, and press SPACE.
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

// The new cut-off. libfreenect2's default is 4.5. Try 4.5 to see the old behaviour.
const float MaxDepthMeters = 8.0f;
const float OldCutoffMeters = 4.5f;

// As in 07: frames per capture, and how often a pixel must read to count as reliable.
const int CaptureFrames = 30;
const double MinValidFraction = 0.5;

const int CrosshairStepPixels = 10;
const int Width = 512, Height = 424;
const int Bands = 8;   // 1 m bands: 0-1, 1-2, ... 7-8 m (anything further lands in the last)

// The middle value of a list (it reorders the list).
double MedianOf(std::vector<double>& values)
{
    if (values.empty()) return 0;
    size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    return values[middle];
}

// Grey (near = bright) up to the old cut-off, blue beyond it, red where there's no reading.
cv::Mat Picture(const std::vector<float>& depthMm)
{
    cv::Mat picture(Height, Width, CV_8UC3);
    for (int i = 0; i < Width * Height; i++)
    {
        float mm = depthMm[i];
        cv::Vec3b& pixel = picture.at<cv::Vec3b>(i / Width, i % Width);
        if (!(mm > 0)) { pixel = cv::Vec3b(0, 0, 255); continue; }
        double meters = mm / 1000.0;
        if (meters <= OldCutoffMeters)
        {
            uchar grey = (uchar)(255 * (1.0 - std::clamp((meters - 0.5) / (OldCutoffMeters - 0.5), 0.0, 1.0)));
            pixel = cv::Vec3b(grey, grey, grey);
        }
        else
        {
            double fraction = std::clamp((meters - OldCutoffMeters) / (MaxDepthMeters - OldCutoffMeters), 0.0, 1.0);
            pixel = cv::Vec3b((uchar)(255 - 155 * fraction), (uchar)(120 * (1 - fraction)), 0);   // light -> dark blue
        }
    }
    return picture;
}

int main()
{
    // -----------------------------------------------------------------
    //  Setup (as in example 07, plus the cut-off)
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

    // The one new line: depth further than MaxDepth is thrown away by the decoder.
    libfreenect2::Freenect2Device::Config config;   // defaults: MinDepth 0.5, MaxDepth 4.5, both filters on
    config.MaxDepth = MaxDepthMeters;
    device->setConfiguration(config);

    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Depth);
    device->setIrAndDepthFrameListener(&listener);
    device->startStreams(false, true);

    std::vector<double> sumMm(Width * Height), sumSquaresMm(Width * Height);
    std::vector<int> readings(Width * Height);
    int framesCaptured = -1;   // -1 = not capturing
    std::vector<float> liveMm(Width * Height);

    const std::string window = "07b far range  (SPACE captures, WASD crosshair, Q quits)";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, Width * 2, Height * 2);
    int crosshairX = Width / 2, crosshairY = Height / 2;

    std::printf("Cut-off at %.1f m. Keep the camera still and press SPACE to capture %d frames.\n",
                MaxDepthMeters, CaptureFrames);

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

        // --- Step 1: add this frame to the running totals (as in 07) --------
        if (framesCaptured >= 0)
        {
            for (int i = 0; i < Width * Height; i++)
            {
                if (liveMm[i] <= 0) continue;
                sumMm[i] += liveMm[i];
                sumSquaresMm[i] += (double)liveMm[i] * liveMm[i];
                readings[i]++;
            }
            framesCaptured++;
        }

        // --- Step 2: capture complete -> report per distance band -----------
        if (framesCaptured == CaptureFrames)
        {
            framesCaptured = -1;
            std::vector<double> spreads[Bands];
            int reliable[Bands] = {}, flaky[Bands] = {}, holes = 0, beyondOld = 0;
            for (int i = 0; i < Width * Height; i++)
            {
                if (readings[i] == 0) { holes++; continue; }
                double mean = sumMm[i] / readings[i];
                int band = std::min((int)(mean / 1000), Bands - 1);
                if (readings[i] < MinValidFraction * CaptureFrames) { flaky[band]++; continue; }
                reliable[band]++;
                if (mean > OldCutoffMeters * 1000) beyondOld++;
                spreads[band].push_back(std::sqrt(std::max(0.0, sumSquaresMm[i] / readings[i] - mean * mean)));
            }

            const double pixels = Width * Height;
            std::printf("Captured %d frames, cut-off %.1f m. Never read: %.1f%% of the view.\n",
                        CaptureFrames, MaxDepthMeters, 100 * holes / pixels);
            std::printf("  band    | reliable | flaky  | one-frame noise\n");
            for (int b = 0; b < Bands; b++)
            {
                if (reliable[b] + flaky[b] == 0) continue;
                double noise = MedianOf(spreads[b]);
                std::printf("  %d-%d m%s | %6.1f%%  | %5.1f%% | ", b, b + 1, b + 1 > OldCutoffMeters ? "*" : " ",
                            100 * reliable[b] / pixels, 100 * flaky[b] / pixels);
                if (spreads[b].empty()) std::printf("-\n");
                else std::printf("+-%.1f mm (%.2f%% of distance)\n", noise, 100 * noise / (b * 1000 + 500));
            }
            std::printf("  * = past the old %.1f m cut-off: %.1f%% of the view gained reliable readings.\n",
                        OldCutoffMeters, 100 * beyondOld / pixels);
        }

        // -----------------------------------------------------------------
        //  Output: live view, with the crosshair readout
        // -----------------------------------------------------------------

        cv::Mat picture = Picture(liveMm);
        int i = crosshairY * Width + crosshairX;
        std::string text = liveMm[i] > 0 ? cv::format("%.2f m", liveMm[i] / 1000.0) : "no reading";
        if (framesCaptured >= 0)
            cv::putText(picture, cv::format("capturing %d/%d", framesCaptured, CaptureFrames), cv::Point(10, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        cv::Scalar green(0, 255, 0);
        cv::drawMarker(picture, cv::Point(crosshairX, crosshairY), green, cv::MARKER_CROSS, 20, 1);
        cv::putText(picture, text, cv::Point(std::min(crosshairX + 8, Width - 100), std::max(crosshairY - 8, 40)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, green, 1);
        cv::imshow(window, picture);
    }

    device->stop();
    device->close();
    cv::destroyAllWindows();
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Set MaxDepthMeters = 4.5 and capture the same scene: the blue disappears and becomes red.
//     That red was never "out of range" for the sensor, only for the decoder.
//  2. Watch the noise column: does it grow steadily with distance, or jump past some range?
//     Time-of-flight noise should grow roughly with the square of the distance (less light returns).
//  3. Put the crosshair on a far wall and on a far dark object at the same distance. Dark things
//     return even less light: they go flaky first.
//  4. Try MaxDepthMeters = 12. Does anything read past 8 m, and is it believable? (Compare
//     with a tape measure or the room's size.)
//  5. Set the 8 m cut-off in example 07 and capture twice: does averaging tame the far noise
//     as well as it tamed the near noise?
