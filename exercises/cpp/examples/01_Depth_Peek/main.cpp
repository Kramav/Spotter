// =====================================================================
//  01 · Depth Peek
//
//  Read the Kinect v2 depth camera and look at what it actually measures.
//
//  What you'll see:
//    - a grey picture of the room: near = bright, far = dark
//    - RED pixels where the sensor got NO reading
//      (closer than ~0.5 m, too far, black / shiny / glass surfaces)
//    - a green crosshair, with the distance under it in millimetres
//
//  Keys:   W A S D   move the crosshair
//          Q or Esc  quit
//
//  The new idea: a depth picture is just a grid of distances, and 0 means
//  "no reading". The library opens the Kinect (spotter::KinectSource) and
//  draws the grey picture (spotter::DepthPicture); this file looks at the
//  numbers. Guide sections: §2 (Kinect v2 specs), §5.5 (what depth gets wrong).
//  The original, with the Kinect setup written out: legacy/01_Kinect_Depth_Peek.
// =====================================================================

#include <algorithm>
#include <cstdio>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// The grey scale. Depth at NearMm is drawn white, depth at FarMm (or beyond) is drawn black.
// Try 1000 / 2000 to stretch the contrast over one metre of the room.
const float NearMm = 500;
const float FarMm = 4500;

// libfreenect2 cleans up depth with two filters: bilateral (smooths noise) and edge-aware
// (removes "flying pixels" along object edges). On the CPU they roughly halve the frame rate.
// false = faster, but noisier edges, AND no clipping to 0.5 .. 4.5 m: expect junk like 17 m.
const bool UseDepthFilters = true;

const int CrosshairStepPixels = 10;   // how far one key press moves the crosshair
const int ReportEveryMs = 1000;       // how often the console report prints

const std::string Window = "01 depth peek  (WASD moves, Q quits)";

// ---------------------------------------------------------------------
//  The new idea: what's in a depth picture
// ---------------------------------------------------------------------

struct DepthStats
{
    int noReading = 0;
    float nearestMm = 1e9f, farthestMm = 0;
};

// Every pixel is a distance in mm; 0 means "no reading" (the source already turned NaN and
// infinity into 0). Row by row: pixel (x, y) is depthMm.at<float>(y, x).
DepthStats Measure(const cv::Mat& depthMm)
{
    DepthStats stats;
    for (int y = 0; y < depthMm.rows; y++)
        for (int x = 0; x < depthMm.cols; x++)
        {
            float mm = depthMm.at<float>(y, x);
            if (mm <= 0) { stats.noReading++; continue; }
            stats.nearestMm = std::min(stats.nearestMm, mm);
            stats.farthestMm = std::max(stats.farthestMm, mm);
        }
    return stats;
}

std::string ReadingAt(const cv::Mat& depthMm, cv::Point crosshair)
{
    float mm = depthMm.at<float>(crosshair);
    return mm > 0 ? std::to_string((int)mm) + " mm" : "no reading";
}

// W A S D move the crosshair, kept inside the picture.
void MoveCrosshair(int key, cv::Point& crosshair, cv::Size picture)
{
    if (key == 'w') crosshair.y -= CrosshairStepPixels;
    if (key == 's') crosshair.y += CrosshairStepPixels;
    if (key == 'a') crosshair.x -= CrosshairStepPixels;
    if (key == 'd') crosshair.x += CrosshairStepPixels;
    crosshair.x = std::clamp(crosshair.x, 0, picture.width - 1);
    crosshair.y = std::clamp(crosshair.y, 0, picture.height - 1);
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

void Show(const cv::Mat& depthMm, cv::Point crosshair, const spotter::FpsMeter& fps)
{
    cv::Mat picture = spotter::DepthPicture(depthMm, NearMm, FarMm);
    cv::Scalar green(0, 255, 0);
    cv::drawMarker(picture, crosshair, green, cv::MARKER_CROSS, 20, 1);
    cv::putText(picture, ReadingAt(depthMm, crosshair), crosshair + cv::Point(8, -8), cv::FONT_HERSHEY_SIMPLEX, 0.5, green, 1);
    fps.Draw(picture);
    cv::imshow(Window, picture);
}

void Report(const cv::Mat& depthMm, cv::Point crosshair, const spotter::FpsMeter& fps)
{
    DepthStats stats = Measure(depthMm);
    std::printf("%4.1f fps | crosshair (%d,%d) = %-10s | nearest %d mm, farthest %d mm | no reading: %.1f%%\n",
                fps.Fps(), crosshair.x, crosshair.y, ReadingAt(depthMm, crosshair).c_str(),
                (int)stats.nearestMm, (int)stats.farthestMm, 100.0 * stats.noReading / depthMm.total());
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // print at once, even when piped to a file

    spotter::KinectSource::Options kinectOptions;
    kinectOptions.colour = false;   // depth only
    kinectOptions.depthFilters = UseDepthFilters;
    spotter::KinectSource kinect(kinectOptions);
    if (!kinect.IsOpen()) return 1;

    cv::namedWindow(Window, cv::WINDOW_NORMAL);   // NORMAL = you can drag the window bigger
    cv::resizeWindow(Window, 1024, 848);
    cv::Point crosshair(256, 212);
    spotter::FpsMeter fps;
    spotter::Every report(ReportEveryMs);
    spotter::RgbdFrame frame;

    while (kinect.Next(frame))
    {
        fps.Tick();
        Show(frame.depthMm, crosshair, fps);
        if (report.Due()) Report(frame.depthMm, crosshair, fps);

        int key = cv::waitKey(1);   // also lets OpenCV redraw the window
        if (spotter::IsQuitKey(key)) break;
        MoveCrosshair(key, crosshair, frame.depthMm.size());
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Put your hand closer and closer to the sensor. Below ~50 cm it turns red, not brighter:
//     inside the minimum range you get holes, not small numbers (guide §5.5).
//  2. Aim the crosshair at a black monitor, a window or a mirror. Watch "no reading" climb.
//  3. Turn the lights off. Depth keeps working: the Kinect lights the room with its own infrared.
//  4. Set NearMm = 1000 and FarMm = 2000 to see fine depth steps (and the noise) on one object.
//  5. Hold the crosshair on a wall and watch the console: how much does one pixel jitter
//     from frame to frame at 1 m? At 3 m?
//  6. Raise your right hand: it shows on the right. The Kinect itself sends a mirror image
//     (like a selfie camera); KinectSource flips it. Set kinectOptions.unmirror = false to see.
//  7. Replace the grey with false colour: cv::applyColorMap(picture, picture, cv::COLORMAP_JET).
