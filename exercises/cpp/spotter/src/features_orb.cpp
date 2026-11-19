#include "spotter/features_orb.hpp"

#include <algorithm>
#include <cfloat>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace spotter
{

namespace
{
// OpenCV's BFMatcher refuses a train set of 2^18 = 262,144 rows or more (IMGIDX_ONE in matchers.cpp: the
// row number shares an int with the image number). 13e's Shortlist stacks every scan's features, so 80 scans
// of about 3,300 each crashed it (2026-11-16). Bigger sets are matched in blocks of this many rows.
const int MatchBlockRows = 200000;
}

Features FindFeatures(const cv::Mat& grey, const cv::Mat& mask, const FeatureOptions& options)
{
    Features found;
    cv::Mat picture = grey;
    if (options.evenContrast) cv::createCLAHE(options.claheClipLimit, cv::Size(options.claheTiles, options.claheTiles))->apply(grey, picture);
    cv::Ptr<cv::ORB> orb = cv::ORB::create(options.count, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, options.fastThreshold);
    if (mask.empty()) orb->detectAndCompute(picture, cv::noArray(), found.keypoints, found.descriptors);
    else orb->detectAndCompute(picture, mask, found.keypoints, found.descriptors);
    return found;
}

std::vector<std::pair<int, int>> MatchFeatures(const cv::Mat& from, const cv::Mat& to, float ratio)
{
    std::vector<std::pair<int, int>> pairs;
    if (from.empty() || to.rows < 2) return pairs;
    // The two nearest in the whole set, block by block (one block when it's small enough).
    std::vector<std::vector<cv::DMatch>> best(from.rows);
    for (int start = 0; start < to.rows; start += MatchBlockRows)
    {
        const int rows = std::min(MatchBlockRows, to.rows - start);
        std::vector<std::vector<cv::DMatch>> candidates;
        cv::BFMatcher(cv::NORM_HAMMING).knnMatch(from, to.rowRange(start, start + rows), candidates, 2);
        for (const std::vector<cv::DMatch>& found : candidates)
            for (cv::DMatch m : found)
            {
                m.trainIdx += start;
                std::vector<cv::DMatch>& two = best[m.queryIdx];
                two.push_back(m);
                std::sort(two.begin(), two.end(), [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });
                if (two.size() > 2) two.pop_back();
            }
    }
    for (const std::vector<cv::DMatch>& two : best)
        if (two.size() == 2 && two[0].distance <= ratio * two[1].distance) pairs.emplace_back(two[0].queryIdx, two[0].trainIdx);
    return pairs;
}

std::vector<cv::DMatch> NearestFeatures(const cv::Mat& from, const cv::Mat& to)
{
    std::vector<cv::DMatch> nearest;
    if (from.empty() || to.empty()) return nearest;
    std::vector<cv::DMatch> best(from.rows, cv::DMatch(-1, -1, FLT_MAX));
    for (int start = 0; start < to.rows; start += MatchBlockRows)
    {
        const int rows = std::min(MatchBlockRows, to.rows - start);
        std::vector<cv::DMatch> matches;
        cv::BFMatcher(cv::NORM_HAMMING).match(from, to.rowRange(start, start + rows), matches);
        for (cv::DMatch m : matches)
        {
            m.trainIdx += start;
            if (m.distance < best[m.queryIdx].distance) best[m.queryIdx] = m;
        }
    }
    for (const cv::DMatch& m : best)
        if (m.queryIdx >= 0) nearest.push_back(m);
    return nearest;
}

}
