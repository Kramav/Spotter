// =====================================================================
//  05c · step 1 of 2 · What does YOLO cost?
//
//  Show the colour camera with the frame rate in the corner. Press D to
//  run YOLO on every frame, and again to stop. Without YOLO the Kinect
//  gives ~30 frames a second; with it, everything waits ~150 ms per frame
//  and drops to ~6. YOLO is nearly all of each frame's time.
//
//  New in this step: FpsMeter; measuring where the time goes.
//  Keys:  D  YOLO on/off     Q or Esc  quit
//  Next:  05c_s2_Detector_Thread (stop waiting for it)
// =====================================================================

#include <chrono>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = 0.4f;
    return options;
}

int main()
{
    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    const char* window = "05c step 1: D = YOLO on/off";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, 960, 540);
    spotter::FpsMeter fps;
    bool detecting = false;
    double detectMs = 0;
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        if (detecting)
        {
            auto start = std::chrono::steady_clock::now();
            for (const spotter::Detection& d : yolo.Detect(frame.colour)) cv::rectangle(frame.colour, d.box, cv::Scalar(0, 255, 0), 3);
            detectMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        fps.Tick();
        fps.Draw(frame.colour, detecting ? cv::format(" | YOLO %.0f ms per frame", detectMs) : " | YOLO off (D)");
        cv::imshow(window, frame.colour);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 'd') detecting = !detecting;
    }
    return 0;
}

// Try: 1000 / (YOLO's ms) is about the fps you get with it on. On a Raspberry Pi expect over 1 s
// per frame. The camera keeps sending 30 frames a second either way; with YOLO we drop most of them.
