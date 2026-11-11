// =====================================================================
//  11b · step 3 of 3 · A camera move from the matches (RANSAC)
//
//  Each match from step 2 has a depth under both ends, so it's a pair of
//  3D points: where the spot is for B, and for A. Step 2 of 11's ladder
//  (spotter::Kabsch) turns pairs into a pose, but some matches are wrong
//  and pull it off. RANSAC: fit a pose to just 3 random pairs; count how
//  many other pairs it moves to within 5 cm of their partner; repeat 1,000
//  times; keep the guess most pairs agree with, and refit to those.
//
//  New in this step: lifting matches to 3D; RANSAC.
//  Keys:  SPACE  capture A, then B (prints B's move), then start over    Q or Esc  quit
//  Next:  11b_Feature_Pose (adds ICP to polish the pose, and the 3D viewer)
// =====================================================================

#include <cmath>
#include <cstdio>
#include <random>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/geometry.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "SPACE  capture A, then B (prints B's move), then start over    Q or Esc  quit";

const float MatchRatio = 0.8f, InlierMeters = 0.05f;
const int RansacTries = 1000;

struct View
{
    cv::Mat picture, descriptors;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Point3f> cloud;   // a 3D point per depth pixel (NaN = none)
};

//View Capture(const spotter::RgbdFrame& frame, cv::ORB& orb){
//
//@param1 - frame: this frame
//@param2 - orb: the ORB detector
//
//@brief - a view: the registered colour picture with its ORB features, and the depth cloud. O(pixels).
View Capture(const spotter::RgbdFrame& frame, cv::ORB& orb)
{
    View v{ frame.registered.clone() };
    cv::Mat grey;
    cv::cvtColor(v.picture, grey, cv::COLOR_BGR2GRAY);
    orb.detectAndCompute(grey, cv::noArray(), v.keypoints, v.descriptors);
    v.cloud = frame.depthCamera.Cloud(frame.depthMm);
    return v;
}

//cv::Point3f Under(const View& v, const cv::KeyPoint& k){
//
//@param1 - v: a view with its cloud and picture
//@param2 - k: a keypoint
//
//@brief - the view's 3D point under the keypoint (NaN where depth read nothing). O(1).
cv::Point3f Under(const View& v, const cv::KeyPoint& k) { return v.cloud[std::lround(k.pt.y) * v.picture.cols + std::lround(k.pt.x)]; }

//std::vector<cv::DMatch> RansacPose(const View& b, const View& a, std::mt19937& random, spotter::Pose& pose){
//
//@param1 - b: the view to move
//@param2 - a: the view to move it onto
//@param3 - random: the random number generator
//@param4 - pose: set to the winning pose
//
//@brief - match (step 2), lift each match to a pair of 3D points (new), then RANSAC: a guess from 3 random pairs
//         (Kabsch), the pairs within InlierMeters of it agree; the guess with most agreeing wins and is refitted
//         to them. Returns the matches it agrees with. O(n x m + tries x pairs).
std::vector<cv::DMatch> RansacPose(const View& b, const View& a, std::mt19937& random, spotter::Pose& pose)
{
    std::vector<std::vector<cv::DMatch>> candidates;
    if (!b.descriptors.empty() && !a.descriptors.empty()) cv::BFMatcher(cv::NORM_HAMMING).knnMatch(b.descriptors, a.descriptors, candidates, 2);
    std::vector<cv::Point3f> from, to;
    std::vector<cv::DMatch> matches, agree;
    for (const std::vector<cv::DMatch>& best : candidates)
    {
        if (best.size() < 2 || best[0].distance >= MatchRatio * best[1].distance) continue;
        cv::Point3f p = Under(b, b.keypoints[best[0].queryIdx]), q = Under(a, a.keypoints[best[0].trainIdx]);   // new: lift to 3D
        if (std::isnan(p.z) || std::isnan(q.z)) continue;
        from.push_back(p); to.push_back(q); matches.push_back(best[0]);
    }
    std::vector<size_t> bestAgree;
    for (int t = 0; t < RansacTries && from.size() >= 3; t++)   // new: RANSAC
    {
        std::uniform_int_distribution<size_t> pick(0, from.size() - 1);
        size_t i = pick(random), j = pick(random), k = pick(random);
        spotter::Pose guess = spotter::Kabsch({ from[i], from[j], from[k] }, { to[i], to[j], to[k] });
        std::vector<size_t> agreeing;
        for (size_t n = 0; n < from.size(); n++) if (cv::norm(guess(from[n]) - to[n]) < InlierMeters) agreeing.push_back(n);
        if (agreeing.size() > bestAgree.size()) bestAgree = agreeing;
    }
    std::vector<cv::Point3f> fromAgree, toAgree;
    for (size_t n : bestAgree) { fromAgree.push_back(from[n]); toAgree.push_back(to[n]); agree.push_back(matches[n]); }
    if (agree.size() >= 3) pose = spotter::Kabsch(fromAgree, toAgree);   // refit to all that agree
    std::printf("%zu matches with depth, %zu agree. B: right %+.3f m, up %+.3f m, forward %+.3f m, turned %.1f deg\n",
                from.size(), agree.size(), pose.shift.x, -pose.shift.y, pose.shift.z, pose.Degrees());
    return agree;
}

//int main(){
//
//@brief - the program: 11b · step 3 of 3 · A camera move from the matches (RANSAC). Each match from step 2 has a
//         depth under both ends, so it's a pair of 3D points: where the spot is for B, and for A. Step 2 of 11's
//         ladder (spotter::Kabsch) turns pairs into a pose, but some matches are wrong and pull it off. Setup,
//         main loop and clean-up are laid out in the file's sections. Returns 0 on a normal exit, 1 if it couldn't
//         start.
int main()
{
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!kinect.IsOpen()) return 1;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
    std::mt19937 random(1);
    View a, b;
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        cv::imshow("11b step 3: SPACE captures A, then B", spotter::WithKeys(frame.registered, Keys));
        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key != ' ') continue;
        if (a.picture.empty()) { a = Capture(frame, *orb); std::printf("A captured. Move the Kinect, SPACE captures B.\n"); continue; }
        if (!b.picture.empty()) { a = b = View(); cv::destroyWindow("agreeing matches"); std::printf("Start over: SPACE captures A.\n"); continue; }
        b = Capture(frame, *orb);
        spotter::Pose pose;
        cv::Mat shown;
        cv::drawMatches(b.picture, b.keypoints, a.picture, a.keypoints, RansacPose(b, a, random, pose), shown, cv::Scalar(0, 255, 0));
        cv::imshow("agreeing matches", spotter::WithKeys(shown, Keys));
    }
    return 0;
}

// Try: capture A, slide the Kinect 20 cm right, capture B: "right" should read about +0.20. The
// agreeing lines should all be parallel; the wrong matches from step 2 are gone.
