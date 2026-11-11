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

//Scan MakeScan(const std::vector<cv::Point3f>& cloud, const cv::Mat& picture, const ScanOptions& options){
//
//@param1 - cloud: a depth cloud, one point per depth pixel (NaN = none), e.g. an averaged frame's
//@param2 - picture: the registered colour on the depth grid (black = none)
//@param3 - options: the ICP step and the near features' settings
//
//@brief - a frame as a Scan: its ICP points (SampleSurface) and near features (FindFeatures where there's
//         depth and colour, eroded 3 pixels clear of holes' edges). The far features need AddFarFeatures
//         as well; the pose is the caller's. O(pixels).
Scan MakeScan(const std::vector<cv::Point3f>& cloud, const cv::Mat& picture, const ScanOptions& options);

// How a depth-camera point lands in the colour picture: column = a x/z + e/z + b, row = c y/z + f/z + d.
// The colour camera sits at (-e/a, -f/c, 0), so far rays can be cast from it.
struct ColourModel { double a = 1, b = 0, c = 1, d = 0, e = 0, f = 0; bool ok = false; };

//ColourModel FitColourModel(const std::vector<cv::Point3f>& cloud, const std::vector<int>& colourIndex, int colourWidth){
//
//@param1 - cloud: a depth cloud, one point per depth pixel (NaN = none)
//@param2 - colourIndex: per depth pixel, the colour pixel it landed on (row * colourWidth + column), -1 = none
//@param3 - colourWidth: the colour picture's width
//
//@brief - fits the ColourModel's six numbers by least squares (SVD) on every 7th pixel that has depth and
//         a colour pixel. ok = false with fewer than 200 such pixels. O(pixels).
ColourModel FitColourModel(const std::vector<cv::Point3f>& cloud, const std::vector<int>& colourIndex, int colourWidth);

//void AddFarFeatures(Scan& scan, const cv::Mat& colourGrey, const cv::Mat& noDepth, const ColourModel& model, const PinholeCamera& depthCamera, const ScanOptions& options){
//
//@param1 - scan: the scan to add them to (its far, farRays and farOrigin are replaced)
//@param2 - colourGrey: the full colour picture, grey
//@param3 - noDepth: CV_8U, the colour picture's size, 255 where depth reads nothing
//@param4 - model: the scan's colour model (FitColourModel); nothing happens if it isn't ok
//@param5 - depthCamera: the depth camera, to keep to the part of the colour picture it also sees
//@param6 - options: the far features' settings
//
//@brief - features where depth reads nothing (13d: too far, a window, a hallway's end), eroded 8 pixels
//         clear of where depth starts, inside the depth camera's view; each becomes a unit ray from the
//         colour camera (farOrigin). O(colour pixels).
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

// What FeaturePose found: the pose and the matches that agree with it. count = how many agree.
struct FeatureMatch
{
    int count = 0, near = 0;
    Pose pose;                                       // b's camera -> a's
    std::vector<int> nearKeypoints, farKeypoints;    // b's keypoints that agree
    std::vector<std::pair<int, int>> farPairs;       // agreeing far matches (b's, a's): new landmarks
    std::vector<std::pair<int, int>> landmarkPairs;  // agreeing (b's far feature, landmark)
};

//FeatureMatch FeaturePose(const Scan& b, const Scan& a, std::mt19937& random, int tries, const std::vector<Landmark>& landmarks, const ScanOptions& options){
//
//@param1 - b: the scan to move
//@param2 - a: the scan to move it onto (its pose turns the landmarks into its coordinates)
//@param3 - random: the random number generator for RANSAC
//@param4 - tries: RANSAC tries
//@param5 - landmarks: far landmarks so far (room coordinates); may be empty
//@param6 - options: the match ratio, inlier distance and far-ray angles
//
//@brief - the pose moving b's camera onto a's from matched features (MatchFeatures): near (3D-3D), far
//         (ray-ray) and landmark (ray-point) matches all vote in RANSAC, and the best guess is refitted to
//         the matches that agree. O(n x m) to match the descriptors, then O(tries x k) for k matches.
FeatureMatch FeaturePose(const Scan& b, const Scan& a, std::mt19937& random, int tries, const std::vector<Landmark>& landmarks,
                         const ScanOptions& options);

//std::vector<size_t> Shortlist(const cv::Mat& descriptors, const std::vector<Scan>& scans, int k, int skip, const ScanOptions& options){
//
//@param1 - descriptors: the new scan's descriptors (near and far)
//@param2 - scans: the earlier scans
//@param3 - k: at most this many scans come back
//@param4 - skip: a scan index that gets no votes (-1 = none)
//@param5 - options: shortlistHamming, how near a twin must be to vote
//
//@brief - which earlier scans share most features with these descriptors: each descriptor votes for the
//         scan holding its nearest twin (NearestFeatures over every scan's descriptors stacked), if within
//         shortlistHamming bits. Up to k scan indices with votes, best first. Excluded scans and `skip` get
//         no votes. O(n x M) for n descriptors and M descriptors in all the scans.
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

//Placement PlaceScan(const Scan& scan, const std::vector<Scan>& earlier, const PinholeCamera& depthCamera, std::mt19937& random, int skip, const std::vector<Landmark>& landmarks, const ScanOptions& options){
//
//@param1 - scan: the new scan (MakeScan + AddFarFeatures)
//@param2 - earlier: the scans placed already
//@param3 - depthCamera: the depth camera, to crop ICP's target to what the rough pose can see
//@param4 - random: the random number generator for RANSAC
//@param5 - skip: an earlier scan to leave out (placing it again), or -1; the landmarks it made are left
//          out too (bug 14)
//@param6 - landmarks: far landmarks so far
//@param7 - options: every placing setting (ScanOptions)
//
//@brief - features against the shortlisted scans (Shortlist, FeaturePose) give a rough pose, and rays to
//         landmarks pin it when near matches are few (13d); ICP against what that pose can see of every
//         earlier scan, one point per thinMeters cube, polishes it; a second ICP against all but the
//         matched scan checks it, and a big move, gap or little overlap marks it suspicious. Not placed
//         when the features or the overlap fall short. The ICPs dominate: O(rounds x n log m).
Placement PlaceScan(const Scan& scan, const std::vector<Scan>& earlier, const PinholeCamera& depthCamera, std::mt19937& random,
                    int skip, const std::vector<Landmark>& landmarks, const ScanOptions& options);

//int AddLandmarks(const Scan& b, int bIndex, const Scan& a, int aIndex, const std::vector<std::pair<int, int>>& pairs, std::vector<Landmark>& landmarks, const ScanOptions& options){
//
//@param1 - b: the scan just placed
//@param2 - bIndex: its index
//@param3 - a: the scan its features matched
//@param4 - aIndex: that scan's index
//@param5 - pairs: the agreeing far matches (b's far feature, a's far feature), from Placement::farPairs
//@param6 - landmarks: the landmark list, added to or refined
//@param7 - options: the parallax, distance and merge limits
//
//@brief - each pair of rays that crosses between farMinMeters and farMaxMeters, at least minParallaxDegrees
//         apart, becomes a landmark at the middle of the rays' closest points; one within
//         landmarkMergeMeters with a similar descriptor is averaged in instead. Returns how many are new.
//         O(pairs x landmarks).
int AddLandmarks(const Scan& b, int bIndex, const Scan& a, int aIndex, const std::vector<std::pair<int, int>>& pairs,
                 std::vector<Landmark>& landmarks, const ScanOptions& options);

// The live check's answer: what the next scan would be placed by ("anchors"). Pixels are in the full
// colour picture.
struct LiveCheck
{
    int anchors = 0, farAnchors = 0;
    int scan = -1;   // the scan it matches best
    std::vector<cv::Point2f> anchorPixels, farAnchorPixels, otherPixels;
};

//LiveCheck CheckLive(const Scan& live, const std::vector<Scan>& scans, const std::vector<Landmark>& landmarks, int colourWidth, const ScanOptions& options){
//
//@param1 - live: the current frame made into a Scan, with its far features
//@param2 - scans: the scans placed so far
//@param3 - landmarks: far landmarks so far
//@param4 - colourWidth: the full colour picture's width, to turn colour indices into pixels
//@param5 - options: liveRansacTries (fewer than placing's, to keep the frame rate)
//
//@brief - FeaturePose against the 2 shortlisted scans; the best one's agreeing features are the anchors.
//         Returns their pixels and the other features' pixels, for drawing. O(n x M) for the shortlist,
//         then O(liveRansacTries x k).
LiveCheck CheckLive(const Scan& live, const std::vector<Scan>& scans, const std::vector<Landmark>& landmarks, int colourWidth,
                    const ScanOptions& options);

// The next shot's advice.
struct Recommendation
{
    bool found = false;
    Pose pose;
    std::string text = "Next: (after the first scan)";
};

//Recommendation RecommendNext(const std::vector<Scan>& scans, const PinholeCamera& depthCamera, const ScanOptions& options){
//
//@param1 - scans: the scans so far (excluded ones are ignored)
//@param2 - depthCamera: the depth camera, to project what's been seen into a candidate view
//@param3 - options: the overlap, distance penalty, coverage cube and "new enough" settings
//
//@brief - from which spot to turn which way to see the most that's new, still sharing at least
//         recommendMinOverlap of the view (or it couldn't be placed). Tries 7 turns x 3 tilts at every
//         spot; a cell of the candidate's view (16 x 12) is known if a seen point (one per coverage cube)
//         lands in it, or a scan from nearly that spot already looked that way (a window reads nothing).
//         Score = the new share minus a penalty per metre from the last spot. O(S x 21 x P) for S spots
//         and P coverage points.
Recommendation RecommendNext(const std::vector<Scan>& scans, const PinholeCamera& depthCamera, const ScanOptions& options);

//long long CubeKey(const cv::Point3f& p, float size){
//
//@param1 - p: a point
//@param2 - size: the cube's edge in metres
//
//@brief - a number for the little cube of space the point is in (21 bits per axis, so it repeats every
//         2^21 cubes), to keep one point per cube with an unordered_set. O(1).
long long CubeKey(const cv::Point3f& p, float size);

}
