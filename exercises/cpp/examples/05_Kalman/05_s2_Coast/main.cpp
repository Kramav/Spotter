// =====================================================================
//  05 · step 2 of 3 · Coasting when the measurements stop   (no Kinect)
//
//  Step 1's dot, now moving in a straight line and bouncing off the walls.
//  Hold SPACE to hide the measurements, like YOLO missing an object. The
//  filter then only predicts: the green dot keeps going at its last speed
//  ("coasting"), and its uncertainty circle grows. Let go: it snaps back.
//
//  New in this step: predicting without correcting; the filter's uncertainty.
//  Keys:  SPACE (hold)  hide the measurements     Q or Esc  quit
//  Next:  05_s3_Kalman_Per_Box (a filter per YOLO box)
// =====================================================================

#include <cmath>
#include <random>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "spotter/view.hpp"

const double NoisePixels = 15, AccelerationPixels = 300, Dt = 1.0 / 30;

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

int main()
{
    std::mt19937 random(1);
    std::normal_distribution<float> noise(0, (float)NoisePixels);
    cv::Point2f truth(100, 100), speed(150, 90);   // pixels per second
    cv::KalmanFilter kalman = NewFilter(truth);
    bool hidden = false;
    while (true)
    {
        truth += speed * (float)Dt;
        if (truth.x < 20 || truth.x > 620) speed.x = -speed.x;   // bounce
        if (truth.y < 20 || truth.y > 460) speed.y = -speed.y;

        kalman.predict();   // always
        cv::Mat picture(480, 640, CV_8UC3, cv::Scalar(30, 30, 30));
        if (!hidden)        // correct only when there is a measurement
        {
            cv::Point2f measured(truth.x + noise(random), truth.y + noise(random));
            kalman.correct((cv::Mat_<float>(2, 1) << measured.x, measured.y));
            cv::circle(picture, measured, 4, cv::Scalar(0, 0, 255), cv::FILLED);
        }
        // No correct()? predict() already left its guess in statePost, so the filter simply coasts.

        cv::Point2f filtered(kalman.statePost.at<float>(0), kalman.statePost.at<float>(1));
        float unsure = std::sqrt(kalman.errorCovPost.at<float>(0, 0));   // one standard deviation, pixels
        cv::circle(picture, truth, 6, cv::Scalar(255, 255, 255), 2);
        cv::circle(picture, filtered, 6, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::circle(picture, filtered, (int)unsure, cv::Scalar(0, 120, 0), 1);
        cv::putText(picture, hidden ? "measurements hidden: coasting" : "hold SPACE to hide the measurements", cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        cv::imshow("05 step 2: coasting", picture);

        int key = cv::waitKey(33);
        if (spotter::IsQuitKey(key)) break;
        hidden = key == ' ';   // a held key repeats, so this stays true while SPACE is down
    }
    return 0;
}

// Try: hide the measurements just before the dot bounces. The filter sails on through the wall:
// it only knows "constant speed". When you let go, the first measurements pull it back.
