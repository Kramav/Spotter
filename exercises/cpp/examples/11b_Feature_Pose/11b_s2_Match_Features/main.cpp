// =====================================================================
//  11b · step 2 of 3 · Match features between two views
//
//  SPACE keeps view A's picture and features. Then, every frame, each live
//  feature looks for the A feature with the most similar description (the
//  fewest of the 256 answers different: the Hamming distance). It counts
//  only if it's clearly better than the second best (Lowe's ratio test): a
//  repeated pattern, like a row of tiles, has many near-equal partners and
//  is dropped. The matches are drawn as lines, A on the right.
//
//  New in this step: BFMatcher, knnMatch with 2 partners, the ratio test.
//  Keys:  SPACE  capture A (again: capture a new A)    Q or Esc  quit
//  Next:  11b_s3_Ransac_Pose (a camera move from the matches)
// =====================================================================

#include <cstdio>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "SPACE  capture A (again: capture a new A)    Q or Esc  quit";

const int FeatureCount = 1000;
const float MatchRatio = 0.8f;   // best must differ by under 80% of the second best's difference

struct Features
{
    cv::Mat picture, descriptors;
    std::vector<cv::KeyPoint> keypoints;
};

Features Find(const cv::Mat& picture, cv::ORB& orb)
{
    Features f;
    f.picture = picture.clone();
    cv::Mat grey;
    cv::cvtColor(picture, grey, cv::COLOR_BGR2GRAY);
    orb.detectAndCompute(grey, cv::noArray(), f.keypoints, f.descriptors);
    return f;
}

// New: each live feature's best A partner, if clearly better than its second best.
std::vector<cv::DMatch> Match(const Features& live, const Features& a)
{
    std::vector<cv::DMatch> kept;
    if (live.descriptors.empty() || a.descriptors.empty()) return kept;
    std::vector<std::vector<cv::DMatch>> candidates;
    cv::BFMatcher(cv::NORM_HAMMING).knnMatch(live.descriptors, a.descriptors, candidates, 2);
    for (const std::vector<cv::DMatch>& best : candidates)
        if (best.size() == 2 && best[0].distance < MatchRatio * best[1].distance) kept.push_back(best[0]);
    return kept;
}

int main()
{
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!kinect.IsOpen()) return 1;

    cv::Ptr<cv::ORB> orb = cv::ORB::create(FeatureCount);
    Features a;
    spotter::Every report(1000);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        Features live = Find(frame.registered, *orb);
        cv::Mat display = live.picture;
        if (!a.picture.empty())
        {
            std::vector<cv::DMatch> matches = Match(live, a);
            cv::drawMatches(live.picture, live.keypoints, a.picture, a.keypoints, matches, display, cv::Scalar(0, 255, 0));
            if (report.Due()) std::printf("%zu live features, %zu in A, %zu matched\n", live.keypoints.size(), a.keypoints.size(), matches.size());
        }
        cv::imshow("11b step 2: live (left) matched to A (right); SPACE captures A", spotter::WithKeys(display, Keys));
        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == ' ') a = live;
    }
    return 0;
}

// Try: capture A, then slowly turn the Kinect. Most lines stay parallel; the odd crossing line is a
// wrong match, which step 3 has to throw out. MatchRatio = 0.95: many more lines, many more wrong.
