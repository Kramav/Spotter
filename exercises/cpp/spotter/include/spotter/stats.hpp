// spotter/stats.hpp: the middle of a list, and other spots along it.
#pragma once

#include <vector>

namespace spotter
{

//float Median(std::vector<float> values){
//
//@param1 - values: the numbers; taken as a copy, so the caller's list keeps its order
//
//@brief - the middle value; for an even count the mean of the two middle values (as numpy's median).
//         An empty list gives 0. std::nth_element puts only the middle in place: O(n), not a sort's O(n log n).
float Median(std::vector<float> values);

//float Percentile(std::vector<float> values, double share){
//
//@param1 - values: the numbers; taken as a copy
//@param2 - share: how far along the sorted list: 0 = smallest, 0.5 = about the middle, 0.98 = near the top
//
//@brief - the value `share` of the way along the sorted list. An empty list gives 0. O(n) (std::nth_element).
float Percentile(std::vector<float> values, double share);

}
