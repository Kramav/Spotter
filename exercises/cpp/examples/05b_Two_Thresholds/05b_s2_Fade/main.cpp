// =====================================================================
//  05b · step 2 of 2 · A coasting object slows down        (no Kinect)
//
//  05_s2's bouncing dot. Hold SPACE to hide the measurements: the filter
//  coasts. With 05b's longer memory (seconds, not one), a coasting object
//  at constant speed drifts far off. So while it is unseen, its speed
//  fades: multiplied by exp(-dt / CoastSlowdownSeconds) each frame. The
//  green dot slows and stops near where the object went out of sight.
//  F switches the fade off, to compare.
//
//  New in this step: fading the speed while coasting.
//  Keys:  SPACE (hold)  hide the measurements    F  fade on/off    Q or Esc  quit
//  Next:  05b_Two_Thresholds (both ideas in a tracker)
// =====================================================================

#include <cmath>
#include <random>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "SPACE (hold)  hide the measurements    F  fade on/off    Q or Esc  quit";

const double NoisePixels = 15, AccelerationPixels = 300, Dt = 1.0 / 30;
const double CoastSlowdownSeconds = 0.5;   // the speed halves about every 0.35 s while unseen

cv::KalmanFilter NewFilter(cv::Point2f start)
{
    cv::KalmanFilter kalman(4, 2, 0, CV_32F);
    kalman.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, Dt, 0,   0, 1, 0, Dt,   0, 0, 1, 0,   0, 0, 0, 1);
    kalman.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0,   0, 1, 0, 0);
    cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar(NoisePixels * NoisePixels));
    double p = AccelerationPixels * Dt * Dt / 2, v = AccelerationPixels * Dt;
    kalman.processNoiseCov = cv::Mat::diag((cv::Mat_<float>(4, 1) << p * p, p * p, v * v, v * v));
    kalman.statePost = (cv::Mat_<float>(4, 1) << start.x, start.y, 0, 0);
    cv::setIdentity(kalman.errorCovPost, cv::Scalar(1000));
    return kalman;
}

// Multiply the speeds (state 2 and 3), in both copies of the state OpenCV keeps.
void FadeSpeed(cv::KalmanFilter& kalman, float factor)
{
    for (int i = 2; i < 4; i++)
    {
        kalman.statePost.at<float>(i) *= factor;
        kalman.statePre.at<float>(i) *= factor;
    }
}

int main()
{
    std::mt19937 random(1);
    std::normal_distribution<float> noise(0, (float)NoisePixels);
    cv::Point2f truth(100, 100), speed(150, 90);
    cv::KalmanFilter kalman = NewFilter(truth);
    bool hidden = false, fade = true;
    while (true)
    {
        truth += speed * (float)Dt;
        if (truth.x < 20 || truth.x > 620) speed.x = -speed.x;
        if (truth.y < 20 || truth.y > 460) speed.y = -speed.y;

        kalman.predict();
        cv::Mat picture(480, 640, CV_8UC3, cv::Scalar(30, 30, 30));
        if (!hidden)
        {
            cv::Point2f measured(truth.x + noise(random), truth.y + noise(random));
            kalman.correct((cv::Mat_<float>(2, 1) << measured.x, measured.y));
            cv::circle(picture, measured, 4, cv::Scalar(0, 0, 255), cv::FILLED);
        }
        else if (fade) FadeSpeed(kalman, (float)std::exp(-Dt / CoastSlowdownSeconds));   // new

        cv::circle(picture, truth, 6, cv::Scalar(255, 255, 255), 2);
        cv::circle(picture, cv::Point2f(kalman.statePost.at<float>(0), kalman.statePost.at<float>(1)), 6, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(picture, cv::format("hold SPACE to hide | fade %s (F)", fade ? "ON" : "off"), cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        cv::imshow("05b step 2: a coasting object slows down", spotter::WithKeys(picture, Keys));

        int key = cv::waitKey(33);
        if (spotter::IsQuitKey(key)) break;
        if (key == 'f') fade = !fade;
        hidden = key == ' ';
    }
    return 0;
}

// Try: hide it for 3 seconds with the fade off: the green dot flies off the screen. With it on,
// it stops a short way past where it disappeared. Which would you rather match against later?
