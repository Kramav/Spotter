// =====================================================================
//  11 · step 2 of 3 · The best move for known pairs     (no Kinect needed)
//
//  If you know which point of B goes with which point of A, the turn +
//  shift that brings each pair closest has an exact answer (Kabsch):
//    1. move both sets so their centres sit at 0: what's left is a turn
//    2. a 3x3 table of how the two centred sets vary together; its SVD
//       (a standard matrix split) gives the turn
//    3. shift = A's centre - turn(B's centre)
//  spotter::Kabsch does exactly that (example 11 writes it out as BestFit).
//
//  This step: points on made-up walls, moved by a known pose, with 3 mm of
//  noise. Then 10% of the pairs are swapped for wrong ones.
//
//  New in this step: Kabsch; what wrong pairs do to it.
//  Next:  11_s3_Pair_And_Repeat (when nobody tells you the pairs)
// =====================================================================

#include <cstdio>

#include "spotter/geometry.hpp"

//void Print(const char* what, const spotter::Pose& pose){
//
//@param1 - what: a label
//@param2 - pose: the pose to show
//
//@brief - one line: the label, the shift (right, forward in metres) and the turn in degrees. O(1).
void Print(const char* what, const spotter::Pose& pose)
{
    std::printf("%-26s right %+.3f  forward %+.3f m, turned %5.2f deg\n", what, pose.shift.x, pose.shift.z, pose.Degrees());
}

//int main(){
//
//@brief - the program: 11 · step 2 of 3 · The best move for known pairs (no Kinect needed). If you know which
//         point of B goes with which point of A, the turn + shift that brings each pair closest has an exact
//         answer (Kabsch). Setup, main loop and clean-up are laid out in the file's sections. Returns 0 on a
//         normal exit.
int main()
{
    // Camera B: 30 cm right, 10 cm forward, turned 12 degrees left (about the vertical axis).
    const spotter::Pose truth{ spotter::TurnAbout(cv::Vec3d(0, -12 * CV_PI / 180, 0)), cv::Point3f(0.30f, 0, 0.10f) };
    const spotter::Pose undo{ truth.turn.t(), -(truth.turn.t() * truth.shift) };   // A's coordinates -> B's

    cv::RNG random(1);
    std::vector<cv::Point3f> a, b;   // a[i] and b[i] are the same spot, seen by A and by B
    for (int i = 0; i < 2000; i++)
    {
        float u = random.uniform(0.f, 1.f), v = random.uniform(0.f, 1.f);
        cv::Point3f p = i % 3 == 0 ? cv::Point3f(-1.5f + 3.5f * u, 1.2f, 0.5f + 2.5f * v)     // floor
                      : i % 3 == 1 ? cv::Point3f(-1.5f + 3.5f * u, -1.3f + 2.5f * v, 3.0f)   // back wall
                                   : cv::Point3f(-1.5f, -1.3f + 2.5f * v, 0.5f + 2.5f * u);  // left wall
        a.push_back(p);
        cv::Point3f noise(random.gaussian(0.003), random.gaussian(0.003), random.gaussian(0.003));
        b.push_back(undo(p) + noise);
    }
    Print("true pose", truth);
    Print("Kabsch, right pairs", spotter::Kabsch(b, a));

    for (size_t i = 0; i < a.size(); i += 10) a[i] = a[(i * 7 + 500) % a.size()];   // 1 in 10 paired with a wrong spot
    Print("Kabsch, 10% wrong pairs", spotter::Kabsch(b, a));
    std::printf("A few wrong pairs pull the whole answer off, so ICP drops pairs that are too far apart,\n"
                "and 11b's RANSAC throws the wrong ones out before fitting.\n");
    return 0;
}
