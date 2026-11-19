// Self-check for features_orb's matching, with no Kinect (2026-11-16: 13e crashed at scan 81 of a real room).
// OpenCV's BFMatcher refuses a train set of 2^18 = 262,144 rows or more; 13e's Shortlist stacks every scan's
// features, so 80 scans of about 3,300 each threw "trainDescCollection[iIdx].rows < IMGIDX_ONE".
//   1. NearestFeatures and MatchFeatures against 300,000 random descriptors: twins planted on both sides of the
//      200,000-row block edge (and at its ends) are found, with no exception.
//   2. Against a small set (one block) both give exactly what a plain BFMatcher gives.
//   3. Shortlist over 90 scans of 3,300 features (297,000 rows: the crash): the scan the query came from wins.
// Run: build\check_features.exe. Exit code 0 = passed. Writes nothing.

#include "spotter/features_orb.hpp"
#include "spotter/room_scan.hpp"

#include <cstdio>
#include <opencv2/features2d.hpp>
#include <random>

int failures = 0;
void Check(bool ok, const char* what)
{
    std::printf("  %s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

// Random 32-byte ORB-like descriptors (two random ones differ by about 128 of 256 bits).
cv::Mat RandomDescriptors(int rows, std::mt19937& random)
{
    cv::Mat d(rows, 32, CV_8U);
    std::uniform_int_distribution<int> byte(0, 255);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < 32; j++) d.at<uchar>(i, j) = (uchar)byte(random);
    return d;
}

// A copy of row `row` of `from` with `flips` random bits flipped: a near twin.
cv::Mat Twin(const cv::Mat& from, int row, int flips, std::mt19937& random)
{
    cv::Mat t = from.row(row).clone();
    std::uniform_int_distribution<int> bit(0, 255);
    for (int f = 0; f < flips; f++) { int b = bit(random); t.at<uchar>(0, b / 8) ^= (uchar)(1 << (b % 8)); }
    return t;
}

int main()
{
    std::mt19937 random(7);

    // 1. A set over the limit, twins planted around the block edge and at both ends.
    const cv::Mat big = RandomDescriptors(300000, random);
    const std::vector<int> planted = { 0, 5, 123456, 199998, 199999, 200000, 200001, 262143, 262144, 299999 };
    cv::Mat queries;
    for (int row : planted) queries.push_back(Twin(big, row, 3, random));
    bool threw = false, nearestRight = true, matchRight = true;
    try
    {
        std::vector<cv::DMatch> nearest = spotter::NearestFeatures(queries, big);
        nearestRight = nearest.size() == planted.size();
        for (size_t i = 0; nearestRight && i < nearest.size(); i++)
            nearestRight = nearest[i].queryIdx == (int)i && nearest[i].trainIdx == planted[i] && nearest[i].distance <= 3;
        std::vector<std::pair<int, int>> pairs = spotter::MatchFeatures(queries, big, 0.8f);
        matchRight = pairs.size() == planted.size();
        for (size_t i = 0; matchRight && i < pairs.size(); i++) matchRight = pairs[i].first == (int)i && pairs[i].second == planted[i];
    }
    catch (const cv::Exception& e) { threw = true; std::printf("  threw: %s\n", e.what()); }
    std::printf("1. 300,000 descriptors, %zu twins planted around the 200,000-row block edge\n", planted.size());
    Check(!threw, "no exception over OpenCV's 262,144-row limit");
    Check(!threw && nearestRight, "NearestFeatures finds every planted twin, with its index in the whole set");
    Check(!threw && matchRight, "MatchFeatures (ratio 0.8) pairs every planted twin, with its index in the whole set");

    // 2. One block: exactly a plain BFMatcher's answers.
    const cv::Mat small = RandomDescriptors(2000, random);
    cv::Mat few = RandomDescriptors(150, random);   // 150 with no twin (fail the ratio test), then 150 twins at 0-40 bits (pass it)
    for (int i = 0; i < 150; i++) few.push_back(Twin(small, i * 13, (i * 7) % 41, random));
    std::vector<cv::DMatch> plain, ours = spotter::NearestFeatures(few, small);
    cv::BFMatcher(cv::NORM_HAMMING).match(few, small, plain);
    bool same = plain.size() == ours.size();
    for (size_t i = 0; same && i < plain.size(); i++) same = plain[i].queryIdx == ours[i].queryIdx && plain[i].trainIdx == ours[i].trainIdx;
    std::vector<std::vector<cv::DMatch>> knn;
    cv::BFMatcher(cv::NORM_HAMMING).knnMatch(few, small, knn, 2);
    std::vector<std::pair<int, int>> plainPairs, ourPairs = spotter::MatchFeatures(few, small, 0.8f);
    for (const auto& two : knn)
        if (two.size() == 2 && two[0].distance <= 0.8f * two[1].distance) plainPairs.emplace_back(two[0].queryIdx, two[0].trainIdx);
    std::printf("2. 300 against 2,000 (one block): %zu nearest, %zu ratio pairs\n", ours.size(), ourPairs.size());
    Check(same, "NearestFeatures gives exactly BFMatcher::match's answers");
    Check(plainPairs == ourPairs, "MatchFeatures gives exactly BFMatcher::knnMatch's ratio-test answers");

    // 3. Shortlist over 90 scans of 3,300 features each: 297,000 rows stacked, as 13e's 81st scan saw.
    std::vector<spotter::Scan> scans(90);
    for (spotter::Scan& s : scans) s.near.descriptors = RandomDescriptors(3300, random);
    cv::Mat fromScan57;
    for (int i = 0; i < 400; i++) fromScan57.push_back(Twin(scans[57].near.descriptors, i * 8, 10, random));
    std::vector<size_t> shortlist;
    threw = false;
    try { shortlist = spotter::Shortlist(fromScan57, scans, 4, -1, spotter::ScanOptions()); }
    catch (const cv::Exception& e) { threw = true; std::printf("  threw: %s\n", e.what()); }
    std::printf("3. Shortlist over 90 scans x 3,300 features: %s\n", shortlist.empty() ? "nothing" : cv::format("scan %zu first", shortlist[0]).c_str());
    Check(!threw && !shortlist.empty() && shortlist[0] == 57, "no exception, and the scan the features came from wins");

    std::printf(failures ? "check_features: %d FAILED\n" : "check_features: all passed\n", failures);
    return failures ? 1 : 0;
}
