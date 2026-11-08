// =====================================================================
//  00c · Tour: BoxDepthSampler                      (no Kinect needed)
//
//  What it's for: "how far away is the thing in this box?", for boxes drawn
//  on the colour picture (YOLO's). The answer is the median depth of the
//  depth pixels that land in the middle of the box (example 02, steps 3-4).
//
//  How it works:
//    The slow way (example 02): for each box, visit all 217k depth pixels,
//    look up where each lands in colour, keep those inside. 5 boxes = 1M visits.
//    The fast way: once per frame, Build() files every depth reading under
//    its colour row, in cells of 32 colour columns (a counting sort, two
//    passes). A box then reads only the cells it covers, row by row, and
//    checks each reading's column. Every reading inside the box is used,
//    none are skipped, so the answer is the same as the slow way's.
//
//  This program builds a made-up Kinect-shaped frame, answers a few boxes
//  both ways, and times them. spotter/checks/check_box_depth.cpp proves
//  "the same" on 4,000 random boxes.
// =====================================================================

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

#include "spotter/depth.hpp"
#include "spotter/stats.hpp"

const int DepthWidth = 512, DepthHeight = 424, ColourWidth = 1920, ColourHeight = 1080;

// A wall 3 m away with a "person" 1.5 m away in the middle, 5% holes. Each depth pixel lands
// about 2.9 colour pixels from its neighbour, shifted a little by distance (parallax).
void MakeFrame(cv::Mat& depthMm, std::vector<int>& colourIndex)
{
    std::mt19937 random(1);
    std::uniform_real_distribution<float> chance(0, 1), noise(-10, 10);
    depthMm.create(DepthHeight, DepthWidth, CV_32F);
    colourIndex.assign(DepthWidth * DepthHeight, -1);
    for (int row = 0; row < DepthHeight; row++)
        for (int column = 0; column < DepthWidth; column++)
        {
            bool person = std::abs(column - 256) < 40 && std::abs(row - 212) < 120;
            float mm = chance(random) < 0.05f ? 0 : (person ? 1500 : 3000) + noise(random);
            depthMm.at<float>(row, column) = mm;
            int x = (int)std::lround(960 + (column - 256) * 2.9 + 52000 / (mm > 0 ? mm : 2000));
            int y = (int)std::lround(540 + (row - 212) * 2.9);
            if (x >= 0 && x < ColourWidth && y >= 0 && y < ColourHeight) colourIndex[row * DepthWidth + column] = y * ColourWidth + x;
        }
}

// The slow way, as example 02 does it.
double SlowMedian(const cv::Rect2d& box, const cv::Mat& depthMm, const std::vector<int>& colourIndex)
{
    cv::Rect2d middle(box.x + box.width / 4, box.y + box.height / 4, box.width / 2, box.height / 2);
    std::vector<float> mm;
    for (size_t i = 0; i < colourIndex.size(); i++)
        if (colourIndex[i] >= 0 && depthMm.at<float>((int)i) > 0 && middle.contains(cv::Point2d(colourIndex[i] % ColourWidth, colourIndex[i] / ColourWidth)))
            mm.push_back(depthMm.at<float>((int)i));
    return mm.size() >= 20 ? spotter::Median(mm) / 1000 : 0;
}

double Milliseconds(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

int main()
{
    cv::Mat depthMm;
    std::vector<int> colourIndex;
    MakeFrame(depthMm, colourIndex);

    spotter::BoxDepthSampler sampler;
    auto start = std::chrono::steady_clock::now();
    sampler.Build(depthMm.ptr<float>(), colourIndex, ColourWidth, ColourHeight);   // once per frame
    std::printf("Build(): %.2f ms, once per frame (1080 rows x 60 cells of 32 columns)\n", Milliseconds(start));

    struct Named { const char* name; cv::Rect2d box; };
    for (const Named& n : { Named{ "the person", { 860, 180, 250, 720 } }, Named{ "wall, left", { 100, 300, 300, 300 } },
                            Named{ "off the picture", { 2500, 100, 100, 100 } }, Named{ "a sliver", { 900, 500, 3, 3 } } })
    {
        start = std::chrono::steady_clock::now();
        double slow = SlowMedian(n.box, depthMm, colourIndex);
        double slowMs = Milliseconds(start);
        start = std::chrono::steady_clock::now();
        std::optional<double> fast = sampler.MedianMeters(n.box);   // the middle half (shrink 0.5), at least 20 readings
        double fastMs = Milliseconds(start);
        std::printf("  %-16s slow %s (%.2f ms)   fast %s (%.3f ms)\n", n.name,
                    slow > 0 ? cv::format("%.3f m", slow).c_str() : "none   ", slowMs,
                    fast ? cv::format("%.3f m", *fast).c_str() : "none   ", fastMs);
    }
    std::printf("\"none\" = fewer than 20 readings there: black, shiny, too close, or outside the depth camera's view.\n");
    return 0;
}
