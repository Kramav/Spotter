// =====================================================================
//  11 · step 1 of 3 · The nearest point, fast          (no Kinect needed)
//
//  ICP pairs every point of view B with the NEAREST point of view A. 13e
//  pairs 13,000 against 50,000: checking every pair is 650 million distances, every
//  round. A k-d tree splits space in half (across x, then y, then z, then
//  x again...) until each box holds a few points; a search then only opens
//  the boxes near the question.
//
//  This step: 50,000 made-up room points, 13,000 questions (one round), answered by
//  checking every point and by the tree. Timed and compared.
//
//  New in this step: cv::flann::Index (a k-d tree) and knnSearch.
//  Next:  11_s2_Best_Fit (the best move for pairs you already know)
// =====================================================================

#include <cstdio>
#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>

double MsSince(int64 tick) { return (cv::getTickCount() - tick) * 1000.0 / cv::getTickFrequency(); }

//int main(){
//
//@brief - the program: 11 · step 1 of 3 · The nearest point, fast (no Kinect needed). ICP pairs every point of
//         view B with the NEAREST point of view A. 13e pairs 13,000 against 50,000: checking every pair is 650
//         million distances, every round. Setup, main loop and clean-up are laid out in the file's sections.
//         Returns 0 on a normal exit.
int main()
{
    cv::RNG random(1);
    std::vector<cv::Point3f> room(50000), questions(13000);
    for (cv::Point3f& p : room) p = cv::Point3f(random.uniform(-2.f, 2.f), random.uniform(-1.3f, 1.2f), random.uniform(0.5f, 4.f));
    for (cv::Point3f& q : questions) q = cv::Point3f(random.uniform(-2.f, 2.f), random.uniform(-1.3f, 1.2f), random.uniform(0.5f, 4.f));

    // 1. Check every point.
    int64 tick = cv::getTickCount();
    std::vector<int> nearest(questions.size());
    for (size_t i = 0; i < questions.size(); i++)
    {
        float best = 1e9f;
        for (size_t j = 0; j < room.size(); j++)
        {
            cv::Point3f d = room[j] - questions[i];
            if (d.dot(d) < best) { best = d.dot(d); nearest[i] = (int)j; }
        }
    }
    double everyMs = MsSince(tick);

    // 2. The k-d tree (new): build it once, then ask all the questions at once.
    tick = cv::getTickCount();
    cv::Mat data((int)room.size(), 3, CV_32F, room.data());   // one row per point: x y z
    cv::flann::Index tree(data, cv::flann::KDTreeIndexParams(4));
    double buildMs = MsSince(tick);
    tick = cv::getTickCount();
    cv::Mat query((int)questions.size(), 3, CV_32F, questions.data()), found, squaredDistance;
    tree.knnSearch(query, found, squaredDistance, 1, cv::flann::SearchParams(32));   // open at most 32 boxes per question
    double searchMs = MsSince(tick);

    int same = 0;
    for (size_t i = 0; i < questions.size(); i++) same += found.at<int>((int)i) == nearest[i];
    std::printf("Every point: %.1f ms.  k-d tree: built in %.1f ms, searched in %.2f ms.\n", everyMs, buildMs, searchMs);
    std::printf("Same answer for %d of %zu questions. The rest got a point almost as near: the tree opens\n"
                "at most 32 boxes, so it's approximate. ICP doesn't mind; it repeats anyway.\n", same, questions.size());
    return 0;
}
