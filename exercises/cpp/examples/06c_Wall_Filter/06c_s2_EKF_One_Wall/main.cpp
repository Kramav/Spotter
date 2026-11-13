// =====================================================================
//  06c · step 2 of 2 · An extended Kalman filter for one wall  (no Kinect)
//
//  A made-up wall whose nearest point (its "foot") is at X = 1.0, Z = 3.0 m.
//  Each frame the wall finder measures, with noise, what a radar would: the
//  DISTANCE to the foot and the ANGLE to it. The filter keeps the foot as
//  X, Z. From X, Z to distance and angle is curved maths:
//      distance = hypot(X, Z)      angle = atan2(X, Z)
//  A plain Kalman filter can only use straight-line maths. The EXTENDED
//  one straightens the curve around its current guess every frame: the
//  Jacobian H, how much each measurement changes per metre of X and of Z.
//  Then it's 05's correct step.
//
//  Printed every 10 frames: the raw distance (wobbling) and the filtered
//  one (settling on 3.162 m, the true hypot(1, 3)).
//
//  New in this step: a measurement that isn't the state; the Jacobian.
//  Next:  06c_Wall_Filter (one such filter per wall, plus the wall's two ends)
// =====================================================================

#include <cmath>
#include <cstdio>
#include <random>
#include <opencv2/core.hpp>

const double NoiseMeters = 0.02, NoiseDegrees = 1.5, DriftMeters = 0.002;

//int main(){
//
//@brief - the program: 06c · step 2 of 2 · An extended Kalman filter for one wall (no Kinect). A made-up wall
//         whose nearest point (its "foot") is at X = 1.0, Z = 3.0 m. Each frame the wall finder measures, with
//         noise, what a radar would: the DISTANCE to the foot and the ANGLE to it. The filter keeps the foot as X,
//         Z. Setup, main loop and clean-up are laid out in the file's sections. Returns 0 on a normal exit.
int main()
{
    const cv::Point2d trueFoot(1.0, 3.0);
    std::mt19937 random(1);
    std::normal_distribution<double> distanceNoise(0, NoiseMeters), angleNoise(0, NoiseDegrees * CV_PI / 180);

    cv::Vec2d state(0.5, 2.5);                         // a poor first guess of the foot's X, Z
    cv::Matx22d P = cv::Matx22d::eye();                // unsure: about 1 m either way
    const double a = NoiseDegrees * CV_PI / 180;
    const cv::Matx22d R(NoiseMeters * NoiseMeters, 0, 0, a * a);   // how noisy the two measurements are

    for (int frame = 1; frame <= 60; frame++)
    {
        cv::Vec2d measured(cv::norm(trueFoot) + distanceNoise(random), std::atan2(trueFoot.x, trueFoot.y) + angleNoise(random));

        P += cv::Matx22d::eye() * (DriftMeters * DriftMeters);   // predict: walls stand still, a little less sure

        double x = state[0], z = state[1], r2 = x * x + z * z, r = std::sqrt(r2);
        cv::Vec2d expected(r, std::atan2(x, z));                 // what we'd measure if the guess were right
        cv::Matx22d H(x / r,   z / r,                            // d distance / dX, d distance / dZ
                      z / r2, -x / r2);                          // d angle / dX,    d angle / dZ
        cv::Matx22d gain = P * H.t() * (H * P * H.t() + R).inv();
        state += gain * (measured - expected);
        cv::Matx22d keep = cv::Matx22d::eye() - gain * H;
        P = keep * P * keep.t() + gain * R * gain.t();           // the Joseph form: stays symmetric

        if (frame % 10 == 0)
            std::printf("frame %2d: raw %.3f m, filtered %.3f m (foot at X %.3f, Z %.3f)\n", frame, measured[0], cv::norm(cv::Point2d(state[0], state[1])), state[0], state[1]);
    }
    std::printf("true: %.3f m (foot at X 1.000, Z 3.000)\n", cv::norm(trueFoot));
    return 0;
}

// Try: NoiseMeters = 0.2 (but measure with it too): the raw distance jumps by 20 cm; the filtered
// one still settles, just more slowly. DriftMeters = 0.1: it barely filters at all.
