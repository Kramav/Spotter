// =====================================================================
//  11 · step 3 of 3 · Pair up, fit, repeat: ICP          (no Kinect needed)
//
//  Now nobody tells you the pairs. Guess them: each B point's partner is
//  the nearest A point (step 1). At first many guesses are wrong, but the
//  best fit to them (step 2) still moves B the right way, so the next
//  guesses are better. Repeat until nothing moves. Pairs farther apart
//  than a gate are skipped (parts only one view saw); the gate shrinks
//  each round, from 20 cm to 3 cm.
//
//  This step: two separate samples of the same made-up walls (two cameras
//  never see the exact same spots). B moved a little, then a lot.
//
//  New in this step: the loop (Iterative Closest Point).
//  Next:  11_ICP (the same on the Kinect, with 10's viewer and keys)
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <opencv2/flann.hpp>

#include "spotter/geometry.hpp"

//std::vector<cv::Point3f> Sample(cv::RNG& random){
//
//@param1 - random: the random number generator
//
//@brief - 30,000 random spots on the floor, the back wall and the left wall of a made-up room (about 3 cm apart).
//         O(30,000).
std::vector<cv::Point3f> Sample(cv::RNG& random)
{
    std::vector<cv::Point3f> points;
    for (int i = 0; i < 30000; i++)
    {
        float u = random.uniform(0.f, 1.f), v = random.uniform(0.f, 1.f);
        points.push_back(i % 3 == 0 ? cv::Point3f(-1.5f + 3.5f * u, 1.2f, 0.5f + 2.5f * v)
                       : i % 3 == 1 ? cv::Point3f(-1.5f + 3.5f * u, -1.3f + 2.5f * v, 3.0f)
                                    : cv::Point3f(-1.5f, -1.3f + 2.5f * v, 0.5f + 2.5f * u));
    }
    return points;
}

//spotter::Pose Icp(cv::flann::Index& tree, const std::vector<cv::Point3f>& a, const std::vector<cv::Point3f>& b, bool print){
//
//@param1 - tree: A's points in a k-d tree
//@param2 - a: A's points
//@param3 - b: B's points, in B's coordinates
//@param4 - print: true = print each round
//
//@brief - new: ICP from "not moved", B's points onto A's: 30 rounds of (step 1) the nearest A point for each moved
//         B point through the tree, kept if within a gate that shrinks from 20 cm to 3 cm, then (step 2) Kabsch on
//         the pairs, on top of the pose so far; stops when it settles. O(rounds x n log m).
spotter::Pose Icp(cv::flann::Index& tree, const std::vector<cv::Point3f>& a, const std::vector<cv::Point3f>& b, bool print)
{
    spotter::Pose pose;
    for (int round = 0; round < 30; round++)
    {
        std::vector<cv::Point3f> moved;
        for (const cv::Point3f& p : b) moved.push_back(pose(p));
        cv::Mat query((int)moved.size(), 3, CV_32F, moved.data()), nearest, squared;
        tree.knnSearch(query, nearest, squared, 1, cv::flann::SearchParams(32));   // step 1: guess the pairs
        const float gate = std::max(0.03f, 0.20f * std::pow(0.7f, (float)round));
        std::vector<cv::Point3f> from, to;
        for (int i = 0; i < query.rows; i++)
            if (squared.at<float>(i) < gate * gate) { from.push_back(moved[i]); to.push_back(a[nearest.at<int>(i)]); }
        spotter::Pose step = spotter::Kabsch(from, to);   // step 2: fit them
        pose = pose.Then(step);
        if (print) std::printf("  round %2d: %5zu pairs within %4.1f cm; moved %5.1f mm, %.2f deg\n", round + 1, from.size(),
                               gate * 100, cv::norm(step.shift) * 1000, step.Degrees());
        if (gate == 0.03f && cv::norm(step.shift) < 1e-4) break;   // settled
    }
    return pose;
}

//void Try(float rightMeters, float leftDegrees, bool print){
//
//@param1 - rightMeters: how far B moved right
//@param2 - leftDegrees: how far B turned left
//@param3 - print: true = print each round
//
//@brief - makes A and B samples of the same room with B moved by that much, runs Icp and prints the pose it found.
//         O(Icp).
void Try(float rightMeters, float leftDegrees, bool print)
{
    const spotter::Pose truth{ spotter::TurnAbout(cv::Vec3d(0, -leftDegrees * CV_PI / 180, 0)), cv::Point3f(rightMeters, 0, 0) };
    const spotter::Pose undo{ truth.turn.t(), -(truth.turn.t() * truth.shift) };
    cv::RNG random(1);
    std::vector<cv::Point3f> a = Sample(random), b = Sample(random);
    for (cv::Point3f& p : b) p = undo(p);   // B's sample, in B's coordinates
    cv::Mat data((int)a.size(), 3, CV_32F, a.data());
    cv::flann::Index tree(data, cv::flann::KDTreeIndexParams(4));
    std::printf("B moved %.0f cm right and turned %.0f deg left:\n", rightMeters * 100, leftDegrees);
    spotter::Pose found = Icp(tree, a, b, print);
    std::printf("  found: right %+.3f m, forward %+.3f m, turned %.2f deg\n", found.shift.x, found.shift.z, found.Degrees());
}

//int main(){
//
//@brief - the program: 11 · step 3 of 3 · Pair up, fit, repeat: ICP (no Kinect needed). Now nobody tells you the
//         pairs. Guess them: each B point's partner is the nearest A point (step 1). At first many guesses are
//         wrong, but the best fit to them (step 2) still moves B the right way, so the next guesses are better.
//         Setup, main loop and clean-up are laid out in the file's sections. Returns 0 on a normal exit.
int main()
{
    Try(0.05f, 3, true);
    Try(0.15f, 8, false);
    std::printf("The big move stalls: early on most guessed pairs are wrong, and the fit creeps along the walls\n"
                "while the gate closes. ICP needs a start that's already close; 11b finds one from features.\n");
    return 0;
}
