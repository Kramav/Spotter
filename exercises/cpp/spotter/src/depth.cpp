#include "spotter/depth.hpp"
#include "spotter/stats.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace spotter
{

namespace
{
//bool HasReading(float mm){
//
//@param1 - mm: a depth reading in millimetres
//
//@brief - true for a usable reading: above 0 and not infinity (NaN fails `mm > 0`). O(1).
bool HasReading(float mm) { return mm > 0 && !std::isinf(mm); }   // NaN fails mm > 0
}

cv::Mat DepthPicture(const cv::Mat& depthMm, float nearMm, float farMm)
{
    cv::Mat picture(depthMm.size(), CV_8UC3);
    for (int row = 0; row < depthMm.rows; row++)
    {
        const float* mm = depthMm.ptr<float>(row);
        cv::Vec3b* pixel = picture.ptr<cv::Vec3b>(row);
        for (int column = 0; column < depthMm.cols; column++)
        {
            if (!HasReading(mm[column])) { pixel[column] = cv::Vec3b(0, 0, 255); continue; }   // B, G, R: red
            float fraction = std::clamp((mm[column] - nearMm) / (farMm - nearMm), 0.0f, 1.0f);
            uchar grey = (uchar)(255 * (1 - fraction));
            pixel[column] = cv::Vec3b(grey, grey, grey);
        }
    }
    return picture;
}

void DepthAverager::Start()
{
    sumMm.release();
    readings.release();
    added = 0;
}

bool DepthAverager::Add(const cv::Mat& depthMm)
{
    if (!Running()) return false;
    if (sumMm.empty())
    {
        sumMm = cv::Mat::zeros(depthMm.size(), CV_64F);
        readings = cv::Mat::zeros(depthMm.size(), CV_32S);
    }
    const float* mm = depthMm.ptr<float>();
    double* sum = sumMm.ptr<double>();
    int* count = readings.ptr<int>();
    for (size_t i = 0; i < depthMm.total(); i++)
        if (mm[i] > 0) { sum[i] += mm[i]; count[i]++; }   // holes don't count, they'd drag the average to 0
    return ++added == frames;
}

cv::Mat DepthAverager::Mean(double minShare) const
{
    cv::Mat mean = cv::Mat::zeros(sumMm.size(), CV_32F);
    for (size_t i = 0; i < sumMm.total(); i++)
    {
        int count = readings.ptr<int>()[i];
        if (count > 0 && count >= minShare * frames) mean.ptr<float>()[i] = (float)(sumMm.ptr<double>()[i] / count);
    }
    return mean;
}

cv::Mat AdaptiveBackground::Update(const cv::Mat& liveMm, const cv::Mat& neverAbsorb)
{
    if (rememberedMm.size() != liveMm.size())
    {
        rememberedMm = cv::Mat::zeros(liveMm.size(), CV_32F);
        nearerFrames = cv::Mat::zeros(liveMm.size(), CV_32S);
        fartherFrames = cv::Mat::zeros(liveMm.size(), CV_32S);
    }
    cv::Mat foreground = cv::Mat::zeros(liveMm.size(), CV_8U);
    const float* live = liveMm.ptr<float>();
    float* remembered = rememberedMm.ptr<float>();
    int* nearer = nearerFrames.ptr<int>();
    int* farther = fartherFrames.ptr<int>();
    for (size_t i = 0; i < liveMm.total(); i++)
    {
        if (live[i] <= 0) continue;                                        // no reading: learn nothing
        if (remembered[i] <= 0) { remembered[i] = live[i]; continue; }     // first reading here
        if (live[i] < remembered[i] - options.nearerMm)                    // something in front
        {
            farther[i] = 0;
            bool keep = !neverAbsorb.empty() && neverAbsorb.data[i];
            if (++nearer[i] < options.absorbFrames || keep) { foreground.data[i] = 255; continue; }
            remembered[i] = live[i];                                       // it stayed: part of the room now
        }
        else if (live[i] > remembered[i] + options.nearerMm)               // something left?
        {
            nearer[i] = 0;
            if (++farther[i] < options.leaveFrames) continue;
            remembered[i] = live[i];
        }
        else remembered[i] += options.blendRate * (live[i] - remembered[i]);   // the same: blend in
        nearer[i] = farther[i] = 0;
    }
    return foreground;
}

void AdaptiveBackground::Forget() { rememberedMm.release(); }

//double NowSeconds(){
//
//@brief - seconds on the steady clock, for timing and countdowns. O(1).
static double NowSeconds() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

StillBackground::StillBackground(double countdownSeconds) : countdownSeconds(countdownSeconds) { Restart(); }

void StillBackground::Restart()
{
    depthMm.release();
    startSeconds = NowSeconds();
    std::printf("Recording the empty view in %.0f s: step out of view.\n", countdownSeconds);
}

double StillBackground::SecondsLeft() const { return countdownSeconds - (NowSeconds() - startSeconds); }

bool StillBackground::RecordWhenDue(const cv::Mat& liveMm)
{
    if (Recorded() || SecondsLeft() > 0) return false;
    depthMm = liveMm.clone();
    std::printf("Empty view recorded. Walk in!\n");
    return true;
}

cv::Mat StillBackground::Foreground(const cv::Mat& liveMm, float nearerMm, int cleanupPixels, int minPixels, std::vector<cv::Rect>& boxes) const
{
    boxes.clear();
    if (!Recorded()) return cv::Mat(liveMm.size(), CV_8U, cv::Scalar(0));
    // Where the empty view had no reading (0), it minus live is negative: never foreground.
    cv::Mat mask = (depthMm - liveMm > nearerMm) & (liveMm > 0);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(cleanupPixels, cleanupPixels)));
    boxes = KeepBigBlobs(mask, minPixels);
    return mask;
}

std::vector<cv::Rect> KeepBigBlobs(cv::Mat& mask, int minPixels)
{
    cv::Mat labels, stats, centroids;
    int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids);
    std::vector<cv::Rect> boxes;
    std::vector<uchar> keep(count, 0);
    for (int label = 1; label < count; label++)   // label 0 = everything else
    {
        if (stats.at<int>(label, cv::CC_STAT_AREA) < minPixels) continue;
        keep[label] = 255;
        boxes.emplace_back(stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP),
                           stats.at<int>(label, cv::CC_STAT_WIDTH), stats.at<int>(label, cv::CC_STAT_HEIGHT));
    }
    const int* label = labels.ptr<int>();
    for (size_t i = 0; i < mask.total(); i++) mask.data[i] = keep[label[i]];
    return boxes;
}

void BoxDepthSampler::Build(const float* depthMm, const std::vector<int>& colourIndex, int colourWidth, int colourHeight)
{
    this->colourWidth = colourWidth;
    this->colourHeight = colourHeight;
    const int cellsPerRow = (colourWidth + CellColumns - 1) / CellColumns;
    const int colourPixels = colourWidth * colourHeight;

    // Pass 1: which cell each reading goes in (-1 = no reading), and how many each cell gets.
    // Cells run left to right along a row, then row by row, so a row's cells sit side by side.
    cellOf.resize(colourIndex.size());
    cellStart.assign((size_t)cellsPerRow * colourHeight + 1, 0);
    for (size_t i = 0; i < colourIndex.size(); i++)
    {
        int index = colourIndex[i];
        bool usable = index >= 0 && index < colourPixels && HasReading(depthMm[i]);
        cellOf[i] = usable ? (index / colourWidth) * cellsPerRow + (index % colourWidth) / CellColumns : -1;
        if (usable) cellStart[cellOf[i] + 1]++;
    }
    for (size_t cell = 1; cell < cellStart.size(); cell++) cellStart[cell] += cellStart[cell - 1];

    // Pass 2: file each reading in its cell.
    readings.resize(cellStart.back());
    std::vector<int> next(cellStart.begin(), cellStart.end() - 1);
    for (size_t i = 0; i < colourIndex.size(); i++)
        if (cellOf[i] >= 0) readings[next[cellOf[i]]++] = { colourIndex[i] % colourWidth, depthMm[i] };
}

std::optional<double> BoxDepthSampler::MedianMeters(const cv::Rect2d& box, double shrink, int minSamples) const
{
    if (cellStart.empty()) return std::nullopt;   // Build() not called yet
    const int cellsPerRow = (colourWidth + CellColumns - 1) / CellColumns;

    // The middle part of the box, and the whole pixels inside it: x <= p < x + width (as cv::Rect2d::contains).
    cv::Rect2d sample(box.x + box.width * (1 - shrink) / 2, box.y + box.height * (1 - shrink) / 2,
                      box.width * shrink, box.height * shrink);
    auto firstPixel = [](double edge) { return (int)std::ceil(std::clamp(edge, -1.0, 1e6)); };
    int top = std::max(0, firstPixel(sample.y)), bottom = std::min(colourHeight, firstPixel(sample.y + sample.height));
    int left = std::max(0, firstPixel(sample.x)), right = std::min(colourWidth, firstPixel(sample.x + sample.width));
    if (left >= right) return std::nullopt;

    // Per row, the cells the box overlaps are one stretch of readings; only the two end cells
    // hold readings outside the box, so check each reading's column.
    std::vector<float> mm;
    for (int row = top; row < bottom; row++)
    {
        int first = cellStart[row * cellsPerRow + left / CellColumns];
        int last = cellStart[row * cellsPerRow + (right - 1) / CellColumns + 1];
        for (int r = first; r < last; r++)
            if (readings[r].column >= left && readings[r].column < right) mm.push_back(readings[r].mm);
    }
    if (mm.empty() || (int)mm.size() < minSamples) return std::nullopt;
    return Median(std::move(mm)) / 1000.0;
}

}
