// =====================================================================
//  09c · step 1 of 3 · Every big flat surface
//
//  Press S for an averaged scan (09b_s1). Then 06b's repeated plane search
//  (spotter::FindPlanes), this time facing ANY way: floor, ceiling, walls,
//  table tops. Each plane's sampled points are dotted on the picture in
//  their own colour, and listed with which way they face.
//
//  New in this step: FindPlanes for planes facing any way.
//  Keys:  S  scan     Q or Esc  quit
//  Next:  09c_s2_Sort_Planes (which plane is the floor, the ceiling, a wall)
// =====================================================================

#include <cstdio>
#include <random>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/geometry.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "S  scan     Q or Esc  quit";

const int Step = 4;   // planes are searched on every 4th pixel each way

int main()
{
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!kinect.IsOpen()) return 1;

    const cv::Scalar colours[] = { { 0, 255, 0 }, { 255, 128, 0 }, { 0, 128, 255 }, { 255, 0, 255 }, { 0, 255, 255 }, { 255, 255, 0 }, { 128, 0, 255 }, { 255, 255, 255 } };
    spotter::DepthAverager averager(30);
    std::vector<std::pair<cv::Point, cv::Scalar>> dots;
    std::mt19937 random(1);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        if (averager.Running() && averager.Add(frame.depthMm))
        {
            spotter::PointGrid grid = spotter::ThinCloud(frame.depthCamera.Cloud(averager.Mean()), 512, Step);
            std::vector<spotter::PlanePiece> pieces = spotter::FindPlanes(grid, spotter::PlaneSearch(), random);
            dots.clear();
            for (size_t k = 0; k < pieces.size(); k++)
            {
                const cv::Point3f& n = pieces[k].plane.normal;
                std::printf("plane %zu: %zu points, %.2f m from the camera, normal (%+.2f, %+.2f, %+.2f)\n", k + 1, pieces[k].cells.size(), pieces[k].plane.distance, n.x, n.y, n.z);
                for (int cell : pieces[k].cells)
                {
                    int c = frame.colourIndex[(cell / grid.columns) * Step * 512 + (cell % grid.columns) * Step];   // the cell's depth pixel -> colour pixel
                    if (c >= 0) dots.emplace_back(cv::Point(c % 1920, c / 1920), colours[k % 8]);
                }
            }
        }
        cv::Mat display;
        cv::resize(frame.colour, display, frame.colour.size() / 2);
        for (const auto& [p, colour] : dots) cv::circle(display, p / 2, 2, colour, cv::FILLED);
        cv::imshow("09c step 1: S scans, then each plane in its own colour", spotter::WithKeys(display, Keys));
        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 's' && !averager.Running()) { averager.Start(); std::printf("Scanning: hold still...\n"); }
    }
    return 0;
}

// Try: read the normals. A floor's points up at the camera: about (0, -1, 0) (the camera's Y
// axis points down). A wall straight ahead: about (0, 0, -1).
