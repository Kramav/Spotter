#include "spotter/features_orb.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace spotter
{

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
    std::vector<std::vector<cv::DMatch>> candidates;
    cv::BFMatcher(cv::NORM_HAMMING).knnMatch(from, to, candidates, 2);
    for (const std::vector<cv::DMatch>& best : candidates)
        if (best.size() == 2 && best[0].distance <= ratio * best[1].distance) pairs.emplace_back(best[0].queryIdx, best[0].trainIdx);
    return pairs;
}

std::vector<cv::DMatch> NearestFeatures(const cv::Mat& from, const cv::Mat& to)
{
    std::vector<cv::DMatch> matches;
    if (!from.empty() && !to.empty()) cv::BFMatcher(cv::NORM_HAMMING).match(from, to, matches);
    return matches;
}

}
