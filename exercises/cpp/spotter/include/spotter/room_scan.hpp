// spotter/room_scan.hpp: scanning a room from many spots (example 13e): one averaged depth frame +
// colour per spot (a Scan), placed in the room by colour features and ICP against the earlier
// scans, far details kept as rays and landmarks, the live check ("anchors") and the next-shot advice.
// The room's coordinates are scan 1's camera's: metres, X right, Y down, Z ahead.
#pragma once

#include "spotter/features_orb.hpp"
#include "spotter/frame.hpp"
#include "spotter/geometry.hpp"
#include "spotter/register.hpp"

#include <random>
#include <string>
#include <utility>
#include <vector>

namespace spotter
{

// 13e's settings, as its defaults. (Example 13e explains each one at its Settings.)
struct ScanOptions
{
    FeatureOptions features;                 // near features: 3000, FAST 7, CLAHE
    FeatureOptions farFeatures{ 1500 };      // far ones, in the full colour picture
    int icpStep = 2;                         // ICP uses every 2nd pixel each way (SampleSurface)
    float matchRatio = 0.8f;
    int ransacTries = 5000;
    float featureInlierMeters = 0.05f;
    int minFeatureMatches = 20;              // fewer agreeing = not placed
    int minScanFeatures = 100;               // fewer (a dark room) = refused before placing
    IcpOptions icp;                          // 50 rounds, 5 cm -> 2 cm, weak 5% (register.hpp)
    float minOverlapShare = 0.25f;           // after ICP, this share of points must have a partner
    float maxDepthMeters = 6.0f;
    int placeCandidates = 4;                 // full matching only against the 4 scans that win a quick vote...
    int shortlistHamming = 50;               // ...votes from twins within 50 bits of 256
    float thinMeters = 0.01f;                // ICP's target: one point per 1 cm cube...
    float cropMarginShare = 0.1f;            // ...of what the rough pose can see, plus 10% round the edges
    float suspiciousMoveMeters = 0.10f;      // red: ICP moved it this far from the features' guess...
    float suspiciousDegrees = 5;             // ...or turned it this much,
    float suspiciousRmsMeters = 0.008f;      // ...or the points sit this far apart,
    float suspiciousOverlap = 0.4f;          // ...or under 40% of them paired.
    float checkStartPairMeters = 0.3f;       // the second opinion: ICP against the room without the matched scan
    float checkMinOverlap = 0.2f;
    float farAgreeDegrees = 0.5f;            // far rays agree if they miss by less than this...
    float farMinMeters = 4.0f;               // ...and meet at least this far away
    float farMaxMeters = 30.0f;
    float minParallaxDegrees = 2.0f;         // two rays this steep make a landmark
    int landmarkMatchHamming = 40;
    float landmarkMergeMeters = 1.0f;
    float bearingMinSpread = 0.001f;
    float farIcpStartPairMeters = 0.5f;      // ICP's first gate when far rays placed the scan
    int liveRansacTries = 500;
    float recommendMinOverlap = 0.5f;        // next shot: share at least half its view with what's seen
    float recommendDistancePenalty = 0.05f;  // per metre from the last spot
    float coverageCellMeters = 0.05f;
    float seenFromMeters = 0.5f;
    float recommendMinNew = 0.1f;
};

// One spot's scan.
struct Scan
{
    std::vector<cv::Point3f> cloud;   // a 3D point per depth pixel (NaN = none), this scan's camera coordinates
    cv::Mat picture;                  // colour on the depth grid (black = none)
    SurfacePoints surface;            // every icpStep-th pixel's point and facing (SampleSurface), for ICP
    Features near;                    // features with depth under them (on `picture`)
    Features far;                     // 13d: features where depth reads nothing (colour pixels)...
    std::vector<cv::Point3f> farRays; // ...each as a direction from farOrigin
    cv::Point3f farOrigin = cv::Point3f(0, 0, 0);   // the colour camera, in the depth camera's coordinates
    Pose pose;                        // this camera -> the room
    bool excluded = false;            // left out of placing, the advice and merging
    bool suspicious = false;          // its placement looked wrong
    std::string stats;                // how its placement went
    cv::Mat fullPicture;              // saved with it: the full colour picture...
    std::vector<int> colourIndex;     // ...and which of its pixels each depth pixel landed on
};

// The point cloud, colour picture and colour index of a frame (or an averaged one) as a Scan, with
// its ICP points and near features. The far features need AddFarFeatures as well.
Scan MakeScan(const std::vector<cv::Point3f>& cloud, const cv::Mat& picture, const ScanOptions& options);

// How a depth-camera point lands in the colour picture: column = a x/z + e/z + b, row = c y/z + f/z + d.
// Measured from the pixels that have depth (least squares), so far rays can be cast from the
// colour camera, which sits at (-e/a, -f/c, 0).
struct ColourModel { double a = 1, b = 0, c = 1, d = 0, e = 0, f = 0; bool ok = false; };
ColourModel FitColourModel(const std::vector<cv::Point3f>& cloud, const std::vector<int>& colourIndex, int colourWidth);

// Features in the full colour picture where depth reads nothing (noDepth = 255 there), inside the
// depth camera's view, each as a ray (13d).
void AddFarFeatures(Scan& scan, const cv::Mat& colourGrey, const cv::Mat& noDepth, const ColourModel& model,
                    const PinholeCamera& depthCamera, const ScanOptions& options);

// A far detail placed in the room by two scans' rays to it (13d).
struct Landmark
{
    cv::Point3f point;
    cv::Mat descriptor;
    int seen = 1;
    int scans[2] = { -1, -1 };   // the two scans whose rays made it (bug 14: left out when re-placing either)
};

// The pose moving scan b onto scan a from matched features: near (3D-3D), far (ray-ray) and
// landmark (ray-point) matches all vote in RANSAC. count = how many agree.
struct FeatureMatch
{
    int count = 0, near = 0;
    Pose pose;                                       // b's camera -> a's
    std::vector<int> nearKeypoints, farKeypoints;    // b's keypoints that agree
    std::vector<std::pair<int, int>> farPairs;       // agreeing far matches (b's, a's): new landmarks
    std::vector<std::pair<int, int>> landmarkPairs;  // agreeing (b's far feature, landmark)
};
FeatureMatch FeaturePose(const Scan& b, const Scan& a, std::mt19937& random, int tries, const std::vector<Landmark>& landmarks,
                         const ScanOptions& options);

// Which earlier scans share most features with these descriptors: each votes for the scan holding
// its nearest twin. Up to k scan indices, best first. Excluded scans and `skip` get no votes.
// ponytail: stacks every scan's descriptors per call (fine for tens of scans); keep a persistent
// index if sessions grow to hundreds.
std::vector<size_t> Shortlist(const cv::Mat& descriptors, const std::vector<Scan>& scans, int k, int skip, const ScanOptions& options);

struct Placement
{
    bool placed = false;
    Pose pose;
    bool suspicious = false;
    std::string stats, report;                   // stats: short, kept with the scan; report: for the console
    size_t matched = 0;                          // the scan its features matched...
    std::vector<std::pair<int, int>> farPairs;   // ...and the far matches that agreed (for AddLandmarks)
};

// Places `scan` among `earlier` (scans placed already): features against the shortlisted scans
// give a rough pose, ICP against what that pose can see of every earlier scan polishes it, and
// a second ICP against all but the matched scan checks it. skip = a scan to leave out (placing
// it again); its own landmarks are left out too (bug 14).
Placement PlaceScan(const Scan& scan, const std::vector<Scan>& earlier, const PinholeCamera& depthCamera, std::mt19937& random,
                    int skip, const std::vector<Landmark>& landmarks, const ScanOptions& options);

// After scan b (index bIndex) is placed: its agreeing far matches with scan a become landmarks
// where the rays cross far away at a clear angle; the same detail seen again refines its landmark.
// Returns how many are new.
int AddLandmarks(const Scan& b, int bIndex, const Scan& a, int aIndex, const std::vector<std::pair<int, int>>& pairs,
                 std::vector<Landmark>& landmarks, const ScanOptions& options);

// The live check: what the next scan would be placed by ("anchors"), from a frame made into a Scan
// (with its far features). Pixels are in the full colour picture.
struct LiveCheck
{
    int anchors = 0, farAnchors = 0;
    int scan = -1;   // the scan it matches best
    std::vector<cv::Point2f> anchorPixels, farAnchorPixels, otherPixels;
};
LiveCheck CheckLive(const Scan& live, const std::vector<Scan>& scans, const std::vector<Landmark>& landmarks, int colourWidth,
                    const ScanOptions& options);

// The next shot: from which spot to turn which way to see the most that's new, still sharing at
// least recommendMinOverlap of the view (or it couldn't be placed).
struct Recommendation
{
    bool found = false;
    Pose pose;
    std::string text = "Next: (after the first scan)";
};
Recommendation RecommendNext(const std::vector<Scan>& scans, const PinholeCamera& depthCamera, const ScanOptions& options);

// A number for the little cube of space a point is in (21 bits per axis), to keep one point per cube.
long long CubeKey(const cv::Point3f& p, float size);

}
