// =====================================================================
//  Demo 1 · You in 3D                 (presentation; built from 08)
//
//  Whoever stands in front of the Kinect sees themselves as a live 3D
//  point cloud, slowly swinging round so they see themselves from the
//  side. The surprise: from the side you're a thin shell. The camera only
//  sees your FRONT; your back is missing.
//
//  Same maths as 08 (a 3D point per depth pixel, the library's viewer),
//  plus: fullscreen, the view swings by itself, and C switches between
//  true colour and colour by distance (works in a dim hall too).
//  Nothing is saved: 08's P key is gone (privacy, visitors are in view).
//
//  Keys:   C  true colour / colour by distance    O  swing on/off
//          [ ]  level: the Kinect's tilt, 1° per press (see LevelDegrees)
//          mouse drag / W A S D  turn   wheel / Z X  zoom   R reset
//          F  fullscreen on/off    Q or Esc  quit
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip along the bottom of the screen (spotter::WithKeys).
const std::string Keys = "C  colour / by distance   O  swing on/off   [ ]  level   drag / W A S D  turn   wheel / Z X  zoom   R  reset   "
                         "F  fullscreen   Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// Where visitors stand, straight ahead of the Kinect. The view swings round this point, so a
// visitor standing here turns in place. Mark the spot on the floor.
const float PivotMeters = 2.0f;

// The swing: this many degrees either side, one full left-right-left every OrbitSeconds.
const float OrbitSwingDegrees = 70;
const float OrbitSeconds = 16;

// Everything farther than this is dropped. 4.5 m keeps the back wall as a backdrop; 3 m shows
// only the visitors (the shell effect is clearer with nothing behind them).
const float MaxDepthMeters = 4.5f;

// The Kinect's own tilt, upwards positive (on the floor ours looks up about 14°). The cloud is
// turned back by this, so the room stands level and the swing turns about the room's real up.
// [ and ] change it by 1° while it runs (the console prints it); put the value that looks level
// here. 0 = the Kinect stands level (on a tripod at the stand).
const float LevelDegrees = 14;

// 3 px points: the viewer puts neighbouring depth pixels about 2 px apart at the pivot, so 2 px
// leaves gaps where the wall behind shows through.
const int PointSizePixels = 3;
const bool StartFullscreen = true;
const std::string Window = "Demo 1 - you in 3D";

// ---------------------------------------------------------------------
//  The cloud, levelled, in true colour or by distance
// ---------------------------------------------------------------------

struct Cloud
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colours;
};

//cv::Mat DistanceColours(){
//
//@brief - 256 colours from near (red) to far (blue): OpenCV's "turbo" map, reversed. O(256).
cv::Mat DistanceColours()
{
    cv::Mat steps(1, 256, CV_8U), colours;
    for (int i = 0; i < 256; i++) steps.at<uchar>(i) = (uchar)(255 - i);
    cv::applyColorMap(steps, colours, cv::COLORMAP_TURBO);
    return colours;
}

//Cloud MakeCloud(const spotter::RgbdFrame& frame, float levelDegrees, bool byDistance, const cv::Mat& distanceColours){
//
//@param1 - frame: this frame
//@param2 - levelDegrees: the Kinect's tilt, to stand the cloud level
//@param3 - byDistance: true = colour by distance, false = the camera's colours
//@param4 - distanceColours: the distance colour table
//
//@brief - level (new): turns every point about the Kinect's left-right axis by its tilt so a level line comes out
//         level; points with no reading are dropped; coloured by distance (0.5 m to MaxDepthMeters) or by the
//         registered colour. O(pixels).
Cloud MakeCloud(const spotter::RgbdFrame& frame, float levelDegrees, bool byDistance, const cv::Mat& distanceColours)
{
    const float c = std::cos(levelDegrees * (float)CV_PI / 180), s = std::sin(levelDegrees * (float)CV_PI / 180);
    std::vector<cv::Point3f> all = frame.depthCamera.Cloud(frame.depthMm);
    Cloud cloud;
    for (size_t i = 0; i < all.size(); i++)
    {
        const cv::Point3f& p = all[i];
        if (std::isnan(p.z)) continue;
        cloud.points.emplace_back(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
        int step = std::clamp((int)(255 * (p.z - 0.5f) / (MaxDepthMeters - 0.5f)), 0, 255);
        cloud.colours.push_back(byDistance ? distanceColours.at<cv::Vec3b>(step) : frame.registered.at<cv::Vec3b>((int)i));
    }
    return cloud;
}

//float SwingDegrees(std::chrono::steady_clock::time_point start){
//
//@param1 - start: when the demo started
//
//@brief - the swing (new): a slow sine of OrbitSwingDegrees amplitude and OrbitSeconds period, so the view pauses
//         at each side, where the shell shows best. O(1).
float SwingDegrees(std::chrono::steady_clock::time_point start)
{
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return OrbitSwingDegrees * (float)std::sin(2 * CV_PI * seconds / OrbitSeconds);
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

//void Show(const Cloud& cloud, spotter::OrbitView& view, spotter::PointRenderer& renderer){
//
//@param1 - cloud: the points and colours
//@param2 - view: the viewer
//@param3 - renderer: draws the points
//
//@brief - the cloud on a dark background with a yellow circle where the camera is, a title and the point count
//         ("live only: nothing is recorded"); shown with the key strip. O(points).
void Show(const Cloud& cloud, spotter::OrbitView& view, spotter::PointRenderer& renderer)
{
    cv::Mat& image = renderer.Begin(view, cv::Scalar(20, 20, 20));
    renderer.Draw(cloud.points, cloud.colours, PointSizePixels);
    cv::Point kinect;
    float depth;
    if (view.Project(cv::Point3f(0, 0, 0), kinect, depth))
    {
        cv::circle(image, kinect, 8, cv::Scalar(0, 255, 255), 2);
        cv::putText(image, "camera", kinect + cv::Point(12, 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
    }
    cv::putText(image, "This is you, in 3D. The camera only sees your front.", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.85,
                cv::Scalar(255, 255, 255), 2);
    cv::putText(image, cv::format("%zu depth points, live   (live only: nothing is recorded)", cloud.points.size()),
                cv::Point(20, image.rows - 15), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(160, 160, 160), 1);
    cv::imshow(Window, spotter::WithKeys(image, Keys));
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

//int main(){
//
//@brief - the program: Demo 1 · You in 3D (presentation; built from 08). Whoever stands in front of the Kinect
//         sees themselves as a live 3D point cloud, slowly swinging round so they see themselves from the side.
//         The surprise: from the side you're a thin shell. Setup, main loop and clean-up are laid out in the
//         file's sections. Returns 0 on a normal exit, 1 if it couldn't start.
int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    spotter::KinectSource::Options kinectOptions;
    kinectOptions.maxDepthMeters = MaxDepthMeters;
    spotter::KinectSource kinect(kinectOptions);
    if (!kinect.IsOpen()) return 1;

    spotter::OrbitView view(spotter::OrbitView::Options(), cv::Point3f(0, 0, PivotMeters), PivotMeters);
    spotter::OpenExhibitWindow(Window, cv::Size(view.options.width, view.options.height), StartFullscreen);
    view.Attach(Window);
    spotter::PointRenderer renderer;
    spotter::RgbdFrame frame;
    const cv::Mat distanceColours = DistanceColours();
    float level = LevelDegrees, swing = 0;
    bool byDistance = false, orbit = true;
    const auto start = std::chrono::steady_clock::now();

    while (kinect.Next(frame))
    {
        float newSwing = orbit ? SwingDegrees(start) : 0;
        view.yawDegrees += newSwing - swing;   // the swing rides on top of whatever the visitor dragged
        swing = newSwing;
        Show(MakeCloud(frame, level, byDistance, distanceColours), view, renderer);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 'c') byDistance = !byDistance;
        if (key == 'o') orbit = !orbit;
        if (key == 'f') spotter::ToggleFullscreen(Window);
        if (key == '[' || key == ']') std::printf("LevelDegrees = %.0f\n", level += key == ']' ? 1.0f : -1.0f);
        view.HandleKey(key);
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas (and things to show visitors)
// ---------------------------------------------------------------------
//  1. Ask a visitor to hold out an arm towards the camera, then watch from the side: the arm is
//     a hollow trough, and there's an empty "shadow" on the wall behind it.
//  2. Set MaxDepthMeters = 3: only the visitors are left, floating in black.
//  3. Press C in a dim hall: colour by distance still works (depth brings its own infrared light).
//  4. Two visitors, one behind the other: the one behind has a hole the shape of the one in front.
//  5. Set OrbitSwingDegrees = 180 and OrbitSeconds = 30: the view goes right round behind them.
