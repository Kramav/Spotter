// =====================================================================
//  05c · step 2 of 2 · YOLO on its own thread
//
//  Run YOLO on a second thread, so the main loop never waits for it:
//    main loop       every frame: if YOLO finished, take its boxes; if it's
//                    idle, give it a copy of this frame. Show the picture.
//    detector thread sleep until given a frame, run YOLO, hand back boxes.
//  The picture now runs at ~30 fps. The boxes still change only ~6 times a
//  second, and lag behind a moving person: each batch is ~150 ms old.
//
//  The two threads share a few variables, guarded by a mutex (only one
//  thread at a time); a condition variable wakes the detector up.
//
//  New in this step: std::thread, std::mutex, std::condition_variable.
//  Keys:  Q or Esc  quit
//  Next:  05c_Background_Detect (the full example predicts the boxes between batches)
// =====================================================================

#include <condition_variable>
#include <mutex>
#include <thread>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// Everything the two threads share. Touch it only while holding `lock`.
struct Shared
{
    std::mutex lock;
    std::condition_variable wake;
    cv::Mat frame;                              // main -> detector
    bool frameWaiting = false, busy = false, quit = false;
    std::vector<spotter::Detection> boxes;      // detector -> main
    bool boxesWaiting = false;
};

spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = 0.4f;
    return options;
}

void Detector(spotter::YoloDetector& yolo, Shared& shared)
{
    while (true)
    {
        cv::Mat image;
        {
            std::unique_lock<std::mutex> hold(shared.lock);
            shared.wake.wait(hold, [&] { return shared.frameWaiting || shared.quit; });   // sleeps here, using no CPU
            if (shared.quit) return;
            image = shared.frame;
            shared.frameWaiting = false;
        }
        std::vector<spotter::Detection> boxes = yolo.Detect(image);   // ~150 ms, without holding the lock
        std::lock_guard<std::mutex> hold(shared.lock);
        shared.boxes = boxes;
        shared.boxesWaiting = true;
    }
}

int main()
{
    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    Shared shared;
    std::thread detector(Detector, std::ref(yolo), std::ref(shared));
    const char* window = "05c step 2: YOLO on its own thread";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, 960, 540);
    spotter::FpsMeter fps;
    std::vector<spotter::Detection> latest;   // the newest batch, drawn on every frame until the next
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        {
            std::lock_guard<std::mutex> hold(shared.lock);
            if (shared.boxesWaiting) { latest = shared.boxes; shared.boxesWaiting = shared.busy = false; }
            if (!shared.busy) { shared.frame = frame.colour.clone(); shared.frameWaiting = shared.busy = true; shared.wake.notify_one(); }
        }
        for (const spotter::Detection& d : latest) cv::rectangle(frame.colour, d.box, cv::Scalar(0, 255, 0), 3);
        fps.Tick();
        fps.Draw(frame.colour);
        cv::imshow(window, frame.colour);
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }

    { std::lock_guard<std::mutex> hold(shared.lock); shared.quit = true; }   // stop the thread before leaving
    shared.wake.notify_one();
    detector.join();
    return 0;
}

// Try: walk across the view. The picture is smooth now, but your box jumps along behind you a few
// times a second. The full example fixes that with step 05's prediction between batches.
