// =====================================================================
//  13e · step 4 of 4 · Live anchors: will the next scan be placed?
//
//  In step 1 you only learn a scan can't be placed after holding still for
//  it. Now, about twice a second, the live frame is made into a Scan and
//  matched against the scans so far (spotter::CheckLive), on a background
//  thread so the window stays smooth. Its "anchors" are the features that
//  agree with the best pose: what the next scan would be placed by. Green
//  dots on the camera picture; at least 20 of them = safe to scan here.
//
//  New in this step: a check running in the background (std::async), and
//  its result drawn when it's ready.
//  Keys:  SPACE scan here   Q quit
//  Next:  13e_Room_Scan (all of it: far features, placing in the
//         background, the next shot, X / P / U, saving on K / Q, the rescue)
// =====================================================================

#include <chrono>
#include <cstdio>
#include <future>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/room_scan.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "SPACE scan here   Q quit";

// New: start a check when the last one is done and half a second has passed.
void StartCheck(std::future<spotter::LiveCheck>& checking, const spotter::RgbdFrame& frame, const std::vector<spotter::Scan>& scans,
                const spotter::ScanOptions& options, std::chrono::steady_clock::time_point& last)
{
    if (scans.empty() || (checking.valid() && checking.wait_for(std::chrono::seconds(0)) != std::future_status::ready)) return;
    if (std::chrono::steady_clock::now() - last < std::chrono::milliseconds(500)) return;
    last = std::chrono::steady_clock::now();
    spotter::Scan live = spotter::MakeScan(frame.depthCamera.Cloud(frame.depthMm), frame.registered, options);
    live.colourIndex = frame.colourIndex;
    checking = std::async(std::launch::async, [&scans, &options, width = frame.colour.cols](spotter::Scan live) {
        return spotter::CheckLive(live, scans, {}, width, options);
    }, std::move(live));
}

int main()
{
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!kinect.IsOpen()) return 1;
    spotter::DepthAverager averager(30);
    spotter::ScanOptions options;
    std::mt19937 random(1);
    std::vector<spotter::Scan> scans;
    std::future<spotter::LiveCheck> checking;
    spotter::LiveCheck live;
    auto last = std::chrono::steady_clock::now();
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        if (checking.valid() && checking.wait_for(std::chrono::seconds(0)) == std::future_status::ready) live = checking.get();
        if (averager.Running() && averager.Add(frame.depthMm))
        {
            if (checking.valid()) live = checking.get();   // the check reads `scans`: wait for it before changing them
            spotter::Scan scan = spotter::MakeScan(frame.depthCamera.Cloud(averager.Mean()), frame.registered, options);
            spotter::Placement placed = scans.empty() ? spotter::Placement{ true } : spotter::PlaceScan(scan, scans, frame.depthCamera, random, -1, {}, options);
            std::printf("Scan %zu %s %s\n", scans.size() + 1, placed.placed ? "placed." : "REFUSED:", placed.report.c_str());
            if (placed.placed) { scan.pose = placed.pose; scans.push_back(std::move(scan)); }
        }
        if (!averager.Running()) StartCheck(checking, frame, scans, options, last);
        cv::Mat display;
        cv::resize(frame.colour, display, frame.colour.size() / 2);
        for (const cv::Point2f& p : live.otherPixels) cv::circle(display, p * 0.5f, 2, cv::Scalar(140, 140, 140), cv::FILLED);
        for (const cv::Point2f& p : live.anchorPixels) cv::circle(display, p * 0.5f, 4, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(display, scans.empty() ? std::string("SPACE: the first scan") : cv::format("%zu scans; anchors: %d (need %d)", scans.size(), live.anchors,
                    options.minFeatureMatches), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, live.anchors >= options.minFeatureMatches ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        cv::imshow("13e step 4: live anchors (green = what the next scan is placed by)", spotter::WithKeys(display, Keys));
        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == ' ' && !averager.Running()) averager.Start();
    }
    if (checking.valid()) checking.wait();
    return 0;
}

// Try: scan, then walk slowly away, watching the count. Where it drops under 20, step back and scan:
// the next scan will be placed.
