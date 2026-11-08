// spotter/features_orb.hpp: colour features that can be found again from another spot (examples
// 11b, 13e), with OpenCV's ORB. An adapter: another detector would be a sibling file with the same
// two functions.
#pragma once

#include <opencv2/core.hpp>
#include <utility>
#include <vector>

namespace spotter
{

struct Features
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;   // one row of 32 bytes (256 yes/no answers) per keypoint
};

struct FeatureOptions
{
    int count = 3000;          // at most this many, the strongest first
    int fastThreshold = 7;     // a corner's ring must be this much brighter or darker (0-255; ORB's default 20)
    bool evenContrast = true;  // CLAHE first: each tile's contrast stretched, so dim corners count (13b, D24)
    double claheClipLimit = 2.0;
    int claheTiles = 8;
};

// ORB features of a grey picture, only where `mask` is non-zero (an empty mask = everywhere).
// ponytail: ORB and CLAHE are made per call (microseconds), so any thread may call this at once.
Features FindFeatures(const cv::Mat& grey, const cv::Mat& mask, const FeatureOptions& options);

// Each `from` descriptor's nearest `to` descriptor (Hamming distance), kept when it's clearly
// nearer than the second nearest: distance <= ratio x second's (Lowe's ratio test). (from, to) rows.
std::vector<std::pair<int, int>> MatchFeatures(const cv::Mat& from, const cv::Mat& to, float ratio = 0.8f);

}
