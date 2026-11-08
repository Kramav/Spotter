// =====================================================================
//  07 · Capture + Average   (and far range, which was 07b)
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
//    - one capture:      the noise of a single frame, per 1 m band, and how
//                        much of the view reads reliably or only sometimes
//    - the next capture: the difference between two averages, which is the
//                        averaged result's REAL noise (times sqrt 2)
//
//  Far range (07b): the 4.5 m limit isn't the sensor, it's libfreenect2's
//  default cut-off. Set MaxDepthMeters = 8 and capture down a hallway.
//
//  Windows: live depth (as in 01), and the latest averaged capture.
//  Keys:    SPACE  capture and average     W A S D  move the crosshair
//           Q or Esc  quit
//  Keep the camera completely still (and people out of view) while capturing.
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/stats.hpp"
#include "spotter/view.hpp"

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// How many frames one capture averages (30 = one second). Noise falls with sqrt(CaptureFrames).
const int CaptureFrames = 30;

// A pixel only counts in the average if it had a reading in at least this fraction of the
// frames. Lower = fills more holes, but trusts pixels that are often wrong (edges, black things).
const double MinValidFraction = 0.5;

// libfreenect2's cut-off. 4.5 is its default; the sensor reads to about 6 m, worse with distance.
const float MaxDepthMeters = 4.5f;

const float NearMm = 500;
const int CrosshairStepPixels = 10;
const std::string LiveWindow = "07 live  (SPACE captures, WASD crosshair, Q quits)";
const std::string AverageWindow = "07 averaged capture";

// ---------------------------------------------------------------------
//  The new idea: per-pixel running totals
// ---------------------------------------------------------------------

// Sum and sum of squares are enough for the mean and the spread (standard deviation),
// without keeping every frame.
struct Capture
{
    std::vector<double> sumMm, sumSquaresMm;
    std::vector<int> readings;
    int frames = -1;   // -1 = not capturing

    bool Running() const { return frames >= 0; }
    void Start(size_t pixels)
    {
        sumMm.assign(pixels, 0);
        sumSquaresMm.assign(pixels, 0);
        readings.assign(pixels, 0);
        frames = 0;
        std::printf("Capturing...\n");
    }
    void Add(const cv::Mat& depthMm)
    {
        const float* mm = depthMm.ptr<float>();
        for (size_t i = 0; i < readings.size(); i++)
        {
            if (mm[i] <= 0) continue;   // holes don't count, they'd drag the average to 0
            sumMm[i] += mm[i];
            sumSquaresMm[i] += (double)mm[i] * mm[i];
            readings[i]++;
        }
        frames++;
    }
};

struct Averaged
{
    cv::Mat averageMm, spreadMm;   // 0 = a hole (too few readings)
};

Averaged Finish(const Capture& capture, cv::Size size)
{
    Averaged result{ cv::Mat(size, CV_32F, cv::Scalar(0)), cv::Mat(size, CV_32F, cv::Scalar(0)) };
    for (size_t i = 0; i < capture.readings.size(); i++)
    {
        if (capture.readings[i] < MinValidFraction * CaptureFrames) continue;   // too often a hole: leave it a hole
        double mean = capture.sumMm[i] / capture.readings[i];
        // Standard deviation = sqrt(mean of squares - square of mean): how far one frame
        // typically strays from the average.
        result.averageMm.at<float>((int)i) = (float)mean;
        result.spreadMm.at<float>((int)i) = (float)std::sqrt(std::max(0.0, capture.sumSquaresMm[i] / capture.readings[i] - mean * mean));
    }
    return result;
}

// Per 1 m band: how much of the view reads reliably, how much only sometimes (flaky), and the
// noise of ONE frame there (noise grows with distance, guide §5.5).
void ReportBands(const Capture& capture, const Averaged& result)
{
    const int bands = (int)std::ceil(MaxDepthMeters);
    std::vector<std::vector<float>> spreads(bands);
    std::vector<int> reliable(bands, 0), flaky(bands, 0);
    int holes = 0;
    std::vector<float> allSpreads;
    for (size_t i = 0; i < capture.readings.size(); i++)
    {
        if (capture.readings[i] == 0) { holes++; continue; }
        double mean = capture.sumMm[i] / capture.readings[i];
        int band = std::min((int)(mean / 1000), bands - 1);
        float spread = result.spreadMm.at<float>((int)i);
        if (result.averageMm.at<float>((int)i) <= 0) { flaky[band]++; continue; }
        reliable[band]++;
        spreads[band].push_back(spread);
        allSpreads.push_back(spread);
    }
    const double pixels = (double)capture.readings.size();
    double oneFrame = spotter::Median(allSpreads);
    std::printf("Captured %d frames, cut-off %.1f m. Never read: %.1f%% of the view.\n", CaptureFrames, MaxDepthMeters, 100 * holes / pixels);
    std::printf("  band  | reliable | flaky  | one-frame noise\n");
    for (int b = 0; b < bands; b++)
    {
        if (reliable[b] + flaky[b] == 0) continue;
        std::printf("  %d-%d m | %6.1f%%  | %5.1f%% | ", b, b + 1, 100 * reliable[b] / pixels, 100 * flaky[b] / pixels);
        if (spreads[b].empty()) { std::printf("-\n"); continue; }
        double noise = spotter::Median(spreads[b]);
        std::printf("+-%.1f mm (%.2f%% of distance)\n", noise, 100 * noise / (b * 1000 + 500));
    }
    std::printf("  noise of the AVERAGE, expected: +-%.1f mm  (one frame +-%.1f / sqrt(%d))\n",
                oneFrame / std::sqrt((double)CaptureFrames), oneFrame, CaptureFrames);
}

// Two averages of the same still scene differ only by their noise. The difference of two
// equally noisy things has sqrt(2) times their noise, so divide by sqrt(2).
void ReportAgainstPrevious(const Averaged& now, const Averaged& previous)
{
    if (previous.averageMm.empty())
    {
        std::printf("  press SPACE again (camera still!) to MEASURE the average's noise.\n");
        return;
    }
    std::vector<float> differences;
    for (int i = 0; i < (int)now.averageMm.total(); i++)
    {
        float a = now.averageMm.at<float>(i), b = previous.averageMm.at<float>(i);
        if (a > 0 && b > 0) differences.push_back(std::abs(a - b));
    }
    // ponytail: median |difference| underestimates the standard deviation by about 1/1.48
    // for normal noise; scaled so it compares with the numbers above.
    std::printf("  noise of the AVERAGE, measured against the previous capture: +-%.1f mm\n",
                1.4826 * spotter::Median(differences) / std::sqrt(2.0));
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

void ShowLive(const cv::Mat& liveMm, const Averaged& last, const Capture& capture, cv::Point crosshair)
{
    cv::Mat picture = spotter::DepthPicture(liveMm, NearMm, MaxDepthMeters * 1000);
    float live = liveMm.at<float>(crosshair);
    std::string text = live > 0 ? cv::format("%.0f mm", live) : "no reading";
    if (!last.averageMm.empty() && last.averageMm.at<float>(crosshair) > 0)
        text += cv::format(" | avg %.1f (1 frame +-%.1f)", last.averageMm.at<float>(crosshair), last.spreadMm.at<float>(crosshair));
    if (capture.Running())
        cv::putText(picture, cv::format("capturing %d/%d", capture.frames, CaptureFrames), cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    cv::drawMarker(picture, crosshair, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 20, 1);
    cv::putText(picture, text, cv::Point(std::min(crosshair.x + 8, picture.cols - 230), std::max(crosshair.y - 8, 40)),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
    cv::imshow(LiveWindow, picture);
}

void ShowAverage(const Averaged& result)
{
    cv::namedWindow(AverageWindow, cv::WINDOW_NORMAL);
    cv::resizeWindow(AverageWindow, 1024, 848);
    cv::imshow(AverageWindow, spotter::DepthPicture(result.averageMm, NearMm, MaxDepthMeters * 1000));
}

void MoveCrosshair(int key, cv::Point& crosshair, cv::Size picture)
{
    if (key == 'w') crosshair.y -= CrosshairStepPixels;
    if (key == 's') crosshair.y += CrosshairStepPixels;
    if (key == 'a') crosshair.x -= CrosshairStepPixels;
    if (key == 'd') crosshair.x += CrosshairStepPixels;
    crosshair.x = std::clamp(crosshair.x, 0, picture.width - 1);
    crosshair.y = std::clamp(crosshair.y, 0, picture.height - 1);
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

    cv::namedWindow(LiveWindow, cv::WINDOW_NORMAL);
    cv::resizeWindow(LiveWindow, 1024, 848);
    cv::Point crosshair(256, 212);
    Capture capture;
    Averaged last, previous;
    spotter::RgbdFrame frame;
    std::printf("Keep the camera still and press SPACE to capture %d frames.\n", CaptureFrames);

    while (kinect.Next(frame))
    {
        if (capture.Running()) capture.Add(frame.depthMm);
        if (capture.frames == CaptureFrames)
        {
            capture.frames = -1;
            previous = last;
            last = Finish(capture, frame.depthMm.size());
            ReportBands(capture, last);
            ReportAgainstPrevious(last, previous);
            ShowAverage(last);
        }
        ShowLive(frame.depthMm, last, capture, crosshair);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == ' ' && !capture.Running()) capture.Start(frame.depthMm.total());
        MoveCrosshair(key, crosshair, frame.depthMm.size());
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Capture twice without touching anything. Is the measured noise of the average close to
//     the expected one? If it's much bigger, something moved (or the sensor drifted as it warmed).
//  2. Compare the distance bands: how much noisier is 3-4 m than 1-2 m? That's why merged
//     maps (stage C) should prefer each camera's NEAR points.
//  3. Set CaptureFrames = 5, then 120. Does the noise follow 1/sqrt(frames)?
//  4. Point the crosshair at a black or shiny object: big single-frame noise, and it may stay
//     a hole at MinValidFraction = 0.5. Try 0.2. Is the filled-in value believable?
//  5. Walk through the view during a capture. The average smears you into the background:
//     averaging only works for things that stay still (docs/decisions.md D12).
//  6. Far range (07b): set MaxDepthMeters = 8 and point it down a hallway. Past 4.5 m the
//     "never read" share drops, and past ~6 m almost nothing is reliable.
//  7. Leave the Kinect running for 20 minutes, then capture again. Has the average shifted?
//     (Time-of-flight sensors drift a few mm while warming up.)
