#include "spotter/stats.hpp"

#include <algorithm>

namespace spotter
{

// nth_element instead of a full sort: only the middle has to be in place.
float Median(std::vector<float> values)
{
    if (values.empty()) return 0;
    size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    if (values.size() % 2 == 1) return values[middle];
    float below = *std::max_element(values.begin(), values.begin() + middle);
    return (below + values[middle]) / 2;
}

float Percentile(std::vector<float> values, double share)
{
    if (values.empty()) return 0;
    size_t i = std::min(values.size() - 1, (size_t)(std::max(0.0, share) * values.size()));
    std::nth_element(values.begin(), values.begin() + i, values.end());
    return values[i];
}

}
