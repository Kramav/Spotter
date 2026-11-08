// =====================================================================
//  08 · Point Cloud
//
//  Every depth pixel is a point in 3D. Example 03 already did the maths
//  for one pixel:  X = (u + 0.5 - cx) * Z / fx,  Y = (v + 0.5 - cy) * Z / fy.
//  Do it for all 217,088 pixels, give each the colour seen at that spot,
//  and you have a POINT CLOUD: the room as 3D dots.
//
//  It's drawn with the library's small 3D viewer (spotter::OrbitView and
//  PointRenderer; tour 00g shows how it works): turn each point around a
//  pivot, then project it onto a virtual camera, the pinhole maths in
//  reverse. The view starts where the Kinect is, so it looks like the
//  camera picture. Drag it and the room turns out to be a thin shell:
//  the camera only sees the FRONT of everything.
//
//  Window: the point cloud. Yellow circle = where the Kinect is.
//  Keys:   mouse drag / W A S D  turn      mouse wheel / Z X  zoom
//          R  reset view    SPACE  freeze/unfreeze    P  save as .ply
//          Q or Esc  quit
//
//  Privacy: a .ply file is a 3D capture of the room. Saved only when you
//  press P, into the folder you ran from. *.ply is gitignored: never
//  commit one.
// =====================================================================

#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/mesh.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "mouse drag / W A S D  turn      mouse wheel / Z X  zoom   R  reset view    SPACE  freeze/unfreeze    P  save as .ply   Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// The depth cut-off. 07: readings to about 6 m are fine, past that almost nothing is right.
const float MaxDepthMeters = 6.0f;

// The view turns around this point, straight ahead of the Kinect. Put it near the middle of the room.
const float PivotMeters = 2.5f;

const int PointSizePixels = 2;   // 1 = sharp but gappy up close, 3 = solid but blocky

const std::string Window = "08 point cloud  (drag/WASD turn, wheel/ZX zoom, R reset, SPACE freeze, P save, Q quit)";

// ---------------------------------------------------------------------
//  The new idea: every depth pixel -> a coloured 3D point
// ---------------------------------------------------------------------

struct Cloud
{
    std::vector<cv::Point3f> points;   // one per depth pixel; NaN where there was no reading
    std::vector<cv::Vec3b> colours;    // the colour camera's view of the same spot (black if it can't see it)
};

Cloud MakeCloud(const spotter::RgbdFrame& frame)
{
    Cloud cloud;
    cloud.points.resize(frame.depthMm.total());
    cloud.colours.resize(frame.depthMm.total());
    for (int row = 0; row < frame.depthMm.rows; row++)
        for (int column = 0; column < frame.depthMm.cols; column++)
        {
            int i = row * frame.depthMm.cols + column;
            float mm = frame.depthMm.at<float>(row, column);
            cloud.points[i] = mm > 0 ? frame.depthCamera.Unproject(cv::Point2f((float)column, (float)row), mm / 1000)
                                     : cv::Point3f(NAN, NAN, NAN);
            cloud.colours[i] = frame.registered.at<cv::Vec3b>(row, column);
        }
    return cloud;
}

// PLY files are usually Y up and Z towards the viewer (the camera's frame turned 180° about X).
void Save(const Cloud& cloud)
{
    std::vector<cv::Point3f> yUp;
    for (const cv::Point3f& p : cloud.points) yUp.emplace_back(p.x, -p.y, -p.z);
    std::string path = (std::filesystem::current_path() / ("cloud_" + std::to_string(std::time(nullptr)) + ".ply")).string();
    if (spotter::SavePly(path, yUp, cloud.colours)) std::printf("Saved %s (a capture of the room: never commit it)\n", path.c_str());
    else std::printf("Could not write %s\n", path.c_str());
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

void Show(const Cloud& cloud, spotter::OrbitView& view, spotter::PointRenderer& renderer, bool frozen)
{
    cv::Mat& image = renderer.Begin(view);
    renderer.Draw(cloud.points, cloud.colours, PointSizePixels);

    cv::Point kinectPixel;
    float kinectDepth;
    if (view.Project(cv::Point3f(0, 0, 0), kinectPixel, kinectDepth)) cv::circle(image, kinectPixel, 6, cv::Scalar(0, 255, 255), 2);
    cv::putText(image, cv::format("%s  turn %.0f, tilt %.0f deg", frozen ? "FROZEN" : "live", view.yawDegrees, view.pitchDegrees),
                cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
    cv::imshow(Window, spotter::WithKeys(image, Keys));
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    spotter::KinectSource::Options kinectOptions;
    kinectOptions.maxDepthMeters = MaxDepthMeters;
    spotter::KinectSource kinect(kinectOptions);
    if (!kinect.IsOpen()) return 1;

    cv::namedWindow(Window, cv::WINDOW_AUTOSIZE);
    spotter::OrbitView view(spotter::OrbitView::Options(), cv::Point3f(0, 0, PivotMeters), PivotMeters);
    view.Attach(Window);
    spotter::PointRenderer renderer;
    spotter::RgbdFrame frame;
    Cloud cloud;
    bool frozen = false;

    while (kinect.Next(frame))
    {
        if (!frozen) cloud = MakeCloud(frame);
        Show(cloud, view, renderer, frozen);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == ' ') frozen = !frozen;
        if (key == 'p') Save(cloud);
        view.HandleKey(key);
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Turn the view 90° (side on). Flat walls should be straight lines. Are they, or do they
//     bow or fan out? That's lens distortion or depth bias, and it matters when merging views.
//  2. Look behind a chair from the side: an empty "shadow" where the camera couldn't see.
//     That's why stage C needs several views.
//  3. Look at object edges side on: a trail of points between the object and the wall behind
//     ("flying pixels"). They come from pixels that saw half of each.
//  4. Set MaxDepthMeters = 8 and look at the far end: 07's far-range noise, now in 3D.
//  5. Freeze, then save with P and open the file in MeshLab or CloudCompare. Delete it after.
//  6. Combine with 07: build the cloud from a 30-frame average instead of one frame. How much
//     thinner does a flat wall get when you look at it edge on?
