// spotter/stats.hpp: the middle of a list, and other spots along it.
#pragma once

#include <vector>

namespace spotter
{

// The middle value. For an even count, the mean of the two middle values (as numpy's median).
// Takes a copy, so the caller's list keeps its order. An empty list gives 0.
float Median(std::vector<float> values);

// The value `share` of the way along the sorted list: 0 = smallest, 0.5 = about the middle,
// 0.98 = near the top. Takes a copy. An empty list gives 0.
float Percentile(std::vector<float> values, double share);

}
