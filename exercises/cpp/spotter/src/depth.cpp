#include "spotter/depth.hpp"

#include <algorithm>
#include <cmath>

namespace spotter
{

namespace
{
// The middle value; for an even count, the mean of the two middle ones (as the examples' Median).
// nth_element instead of a full sort: only the middle has to be in place.
float Median(std::vector<float> values)
{
    size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    if (values.size() % 2 == 1) return values[middle];
    float below = *std::max_element(values.begin(), values.begin() + middle);
    return (below + values[middle]) / 2;
}

bool HasReading(float mm) { return mm > 0 && !std::isinf(mm); }   // NaN fails mm > 0
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
