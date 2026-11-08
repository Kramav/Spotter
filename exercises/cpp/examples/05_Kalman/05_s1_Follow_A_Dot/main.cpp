// =====================================================================
//  05 · step 1 of 3 · A Kalman filter follows a noisy dot     (no Kinect)
//
//  A dot moves in a circle (white, the truth). We only get noisy
//  measurements of it (red), like YOLO's wobbly box centres. A Kalman
//  filter (green) keeps a guess of the position AND speed, and each frame:
//    Predict  move the guess on by its speed; it gets less sure
//    Correct  blend in the measurement, weighted by how sure each one is
//  The green dot is calmer than the red, and stays close to the white.
//
//  New in this step: cv::KalmanFilter, predict, correct.
//  Keys:  Q or Esc  quit
//  Next:  05_s2_Coast (what happens when measurements stop)
// =====================================================================

#include <cmath>
#include <random>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "Q or Esc  quit";

const double NoisePixels = 15;          // how much the measurements wobble
const double AccelerationPixels = 300;  // how suddenly the filter thinks the dot can change speed
const double Dt = 1.0 / 30;             // 30 frames a second

// State: x, y, speed x, speed y. Measured: x, y.
cv::KalmanFilter NewFilter(cv::Point2f start)
{
    cv::KalmanFilter kalman(4, 2, 0, CV_32F);
    kalman.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, Dt, 0,   0, 1, 0, Dt,   0, 0, 1, 0,   0, 0, 0, 1);   // position += speed * dt
    kalman.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0,   0, 1, 0, 0);                              // we measure x and y
    cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar(NoisePixels * NoisePixels));                          // R: measurement noise
    double p = AccelerationPixels * Dt * Dt / 2, v = AccelerationPixels * Dt;
    kalman.processNoiseCov = cv::Mat::diag((cv::Mat_<float>(4, 1) << p * p, p * p, v * v, v * v));              // Q: how far reality drifts
    kalman.statePost = (cv::Mat_<float>(4, 1) << start.x, start.y, 0, 0);
    cv::setIdentity(kalman.errorCovPost, cv::Scalar(1000));                                                      // P: unsure to begin with
    return kalman;
}

int main()
{
    std::mt19937 random(1);
    std::normal_distribution<float> noise(0, (float)NoisePixels);
    cv::KalmanFilter kalman = NewFilter(cv::Point2f(500, 240));
    for (int frame = 0;; frame++)
    {
        double angle = frame * Dt;                                                   // one turn every ~6 s
        cv::Point2f truth(320 + 180 * (float)std::cos(angle), 240 + 180 * (float)std::sin(angle));
        cv::Point2f measured(truth.x + noise(random), truth.y + noise(random));

        kalman.predict();
        cv::Mat estimate = kalman.correct((cv::Mat_<float>(2, 1) << measured.x, measured.y));
        cv::Point2f filtered(estimate.at<float>(0), estimate.at<float>(1));

        cv::Mat picture(480, 640, CV_8UC3, cv::Scalar(30, 30, 30));
        cv::circle(picture, truth, 6, cv::Scalar(255, 255, 255), 2);
        cv::circle(picture, measured, 4, cv::Scalar(0, 0, 255), cv::FILLED);
        cv::circle(picture, filtered, 6, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(picture, "white = truth   red = measured   green = Kalman", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        cv::imshow("05 step 1: a Kalman filter follows a noisy dot", spotter::WithKeys(picture, Keys));
        if (spotter::IsQuitKey(cv::waitKey(33))) break;
    }
    return 0;
}

// Try: NoisePixels = 50: the red jumps wildly, the green stays near the truth. AccelerationPixels
// = 20: very calm, but it cuts the corner of the circle (it believes the dot barely turns).
