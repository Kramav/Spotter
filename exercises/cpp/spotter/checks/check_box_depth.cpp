// Self-check for spotter::BoxDepthSampler, on a synthetic frame (no Kinect needed).
// It must give exactly what example 05c's SampleMeters gives (the old loop over all 217k depth
// pixels per box, copied below), for thousands of random boxes, and the time per frame of both.
// Run: build\check_box_depth.exe. Exit code 0 = passed.

#include "spotter/depth.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

const int DepthWidth = 512, DepthHeight = 424, ColourWidth = 1920, ColourHeight = 1080;
const double BoxShrink = 0.5;
const int MinDepthSamples = 20;

// --- The reference: example 05c's Median and SampleMeters, unchanged -------------------------
float Median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) return values[middle];
    return (values[middle - 1] + values[middle]) / 2;
}

double SampleMeters(const cv::Rect2d& box, const float* depthMm, const std::vector<int>& colorIndexOfDepthPixel, int colorWidth)
{
    cv::Rect2d sample(box.x + box.width * (1 - BoxShrink) / 2, box.y + box.height * (1 - BoxShrink) / 2,
                      box.width * BoxShrink, box.height * BoxShrink);
    std::vector<float> samplesMm;
    for (size_t i = 0; i < colorIndexOfDepthPixel.size(); i++)
    {
        int colorIndex = colorIndexOfDepthPixel[i];
        if (colorIndex < 0 || !(depthMm[i] > 0) || std::isinf(depthMm[i])) continue;
        if (sample.contains(cv::Point2d(colorIndex % colorWidth, colorIndex / colorWidth))) samplesMm.push_back(depthMm[i]);
    }
    return (int)samplesMm.size() >= MinDepthSamples ? Median(samplesMm) / 1000.0 : 0;
}

// --- A synthetic frame shaped like the Kinect's ----------------------------------------------
// Depth 0.5-4.5 m with holes (0), a few inf and NaN. Each depth pixel lands about 2.9 colour
// pixels from its neighbour, mirrored like the real thing, shifted a little by distance
// (parallax), and the top and bottom depth rows fall outside the colour picture (-1).
void MakeFrame(std::mt19937& random, std::vector<float>& depthMm, std::vector<int>& colourIndex)
{
    std::uniform_real_distribution<float> distance(500, 4500), chance(0, 1);
    depthMm.resize(DepthWidth * DepthHeight);
    colourIndex.resize(DepthWidth * DepthHeight);
    for (int row = 0; row < DepthHeight; row++)
        for (int column = 0; column < DepthWidth; column++)
        {
            int i = row * DepthWidth + column;
            float roll = chance(random);
            depthMm[i] = roll < 0.05f ? 0 : roll < 0.06f ? std::numeric_limits<float>::infinity()
                       : roll < 0.07f ? std::numeric_limits<float>::quiet_NaN() : distance(random);
            float z = std::isfinite(depthMm[i]) && depthMm[i] > 0 ? depthMm[i] : 2000;
            int x = (int)std::lround(960 - (column - 256) * 2.9 + 52000 / z);
            int y = (int)std::lround(540 + (row - 212) * 2.9);
            bool inside = x >= 0 && x < ColourWidth && y >= 0 && y < ColourHeight;
            colourIndex[i] = inside && chance(random) > 0.02f ? y * ColourWidth + x : -1;
        }
}

int failures = 0;
void Check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAILED: %s\n", what); failures++; }
}

int main()
{
    std::mt19937 random(1);
    std::vector<float> depthMm;
    std::vector<int> colourIndex;
    spotter::BoxDepthSampler sampler;

    Check(!sampler.MedianMeters(cv::Rect2d(0, 0, 100, 100)), "no answer before Build()");

    // 1. Same answer as the old loop: 20 frames x 200 random boxes, fractional edges included.
    std::uniform_real_distribution<double> x(-200, ColourWidth + 50), y(-200, ColourHeight + 50), size(0, 900);
    int compared = 0, answered = 0;
    for (int frame = 0; frame < 20; frame++)
    {
        MakeFrame(random, depthMm, colourIndex);
        sampler.Build(depthMm.data(), colourIndex, ColourWidth, ColourHeight);
        for (int b = 0; b < 200; b++)
        {
            cv::Rect2d box(x(random), y(random), size(random), size(random) * (b % 10 == 0 ? 0.01 : 1));
            double expected = SampleMeters(box, depthMm.data(), colourIndex, ColourWidth);
            std::optional<double> got = sampler.MedianMeters(box, BoxShrink, MinDepthSamples);
            Check(got.value_or(0) == expected, "same median as 05c's SampleMeters");
            compared++;
            if (got) answered++;
        }
    }
    std::printf("%d boxes compared with 05c's SampleMeters (%d with an answer, %d too few readings)\n",
                compared, answered, compared - answered);

    // 2. Edge cases.
    Check(!sampler.MedianMeters(cv::Rect2d(5000, 5000, 100, 100)), "a box off the picture has no answer");
    Check(!sampler.MedianMeters(cv::Rect2d(900, 500, 0, 0)), "an empty box has no answer");
    Check(!sampler.MedianMeters(cv::Rect2d(900, 500, 40, 40), 0.5, 100000), "too few readings -> no answer");
    Check(sampler.MedianMeters(cv::Rect2d(-1e9, -1e9, 2e9, 2e9), 1.0).has_value(), "a huge box doesn't overflow");

    // 3. Time per frame with 5 boxes (typical YOLO output), the old way and the new way.
    const int Frames = 50;
    std::vector<cv::Rect2d> boxes = { { 700, 300, 300, 500 }, { 200, 400, 150, 150 }, { 1300, 200, 400, 700 },
                                      { 900, 700, 120, 200 }, { 500, 100, 250, 250 } };
    volatile double sink = 0;   // keeps the compiler from skipping the timed work
    auto start = std::chrono::steady_clock::now();
    for (int f = 0; f < Frames; f++)
        for (const cv::Rect2d& box : boxes) sink += SampleMeters(box, depthMm.data(), colourIndex, ColourWidth);
    double oldMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / Frames;
    start = std::chrono::steady_clock::now();
    for (int f = 0; f < Frames; f++)
    {
        sampler.Build(depthMm.data(), colourIndex, ColourWidth, ColourHeight);
        for (const cv::Rect2d& box : boxes) sink += sampler.MedianMeters(box).value_or(0);
    }
    double newMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / Frames;
    start = std::chrono::steady_clock::now();
    for (int f = 0; f < Frames; f++)
        for (const cv::Rect2d& box : boxes) sink += sampler.MedianMeters(box).value_or(0);
    double queryMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / Frames;
    std::printf("5 boxes per frame: old %.2f ms, new %.2f ms (Build %.2f + boxes %.2f)\n",
                oldMs, newMs, newMs - queryMs, queryMs);

    std::printf(failures == 0 ? "PASSED\n" : "%d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
