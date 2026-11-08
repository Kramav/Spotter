// =====================================================================
//  09b · Room Mesh
//
//  Example 08b turned ONE object into a 3D model. This one does the
//  same for EVERYTHING the Kinect sees: the first recreation of the room,
//  from a single view.
//
//    1. Press S. Keep the camera still.
//    2. Average 30 depth frames (07, spotter::DepthAverager).
//    3. Every depth pixel becomes a 3D point (08), coloured from the
//       colour camera.
//    4. Mesh: the depth picture is already a grid. Each square of points
//       becomes two triangles (08b), unless its corners jump in depth:
//       that's an edge (a chair in front of a wall), not a surface.
//    5. Save room_<time>.obj in the folder you ran from.
//
//  You'll see the room from where the camera stood, and HOLES behind
//  everything: the camera can't see behind a chair. Those holes are why
//  the next steps (11-13) take several views and merge them.
//  (spotter::MeshFromGrid is step 4, for the later examples.)
//
//  Window: colour (half size). After a scan, the pixels that went into
//          the mesh are tinted green: untinted = no depth there.
//  Keys:   S  scan     C  clear the tint     Q or Esc  quit
//  Privacy: an .obj is a 3D capture of the room, in colour: never commit
//  one. Scan with nobody in view.
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/mesh.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "S  scan     C  clear the tint     Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

const int AverageFrames = 30;   // one second (07)

// Use every Nth depth pixel in each direction. 1 = all of them (about 400k triangles, a 25 MB
// file); 2 = a quarter of that, which still looks the same from across the room.
const int MeshStep = 2;

// A triangle is skipped when its corners differ in depth by more than this share of their
// distance. A share, not a fixed number of mm as in 08b: far away, neighbouring pixels are farther
// apart, so a wall seen at a slant 4 m away steps about 3 cm per pixel.
const float MaxJumpShare = 0.03f;

const std::string Window = "09b room mesh  (S scan, C clear, Q quit)";

// ---------------------------------------------------------------------
//  The new idea: the whole view as one triangle mesh
// ---------------------------------------------------------------------

// Step 4: every MeshStep-th point, two triangles per square, none across a depth jump.
spotter::Mesh MeshFromCloud(const std::vector<cv::Point3f>& cloud, const std::vector<cv::Vec3b>& colours)
{
    const int width = 512, columns = (width + MeshStep - 1) / MeshStep, rows = ((int)cloud.size() / width + MeshStep - 1) / MeshStep;
    spotter::Mesh mesh;
    std::vector<int> pointOf(columns * rows, -1);
    for (int row = 0; row < rows; row++)
        for (int column = 0; column < columns; column++)
        {
            int pixel = row * MeshStep * width + column * MeshStep;
            if (std::isnan(cloud[pixel].z)) continue;
            pointOf[row * columns + column] = (int)mesh.points.size();
            mesh.points.push_back(cloud[pixel]);
            mesh.colours.push_back(colours[pixel]);
        }
    std::vector<cv::Vec3i>& faces = mesh.groups["seen_room"];
    auto AddTriangle = [&](int a, int b, int c)
    {
        if (pointOf[a] < 0 || pointOf[b] < 0 || pointOf[c] < 0) return;
        float za = mesh.points[pointOf[a]].z, zb = mesh.points[pointOf[b]].z, zc = mesh.points[pointOf[c]].z;
        float nearest = std::min({ za, zb, zc }), farthest = std::max({ za, zb, zc });
        if (farthest - nearest > MaxJumpShare * nearest) return;   // an edge between two things
        faces.emplace_back(pointOf[a], pointOf[b], pointOf[c]);
    };
    for (int row = 0; row + 1 < rows; row++)
        for (int column = 0; column + 1 < columns; column++)
        {
            int topLeft = row * columns + column, topRight = topLeft + 1, bottomLeft = topLeft + columns, bottomRight = bottomLeft + 1;
            AddTriangle(topLeft, bottomLeft, topRight);
            AddTriangle(topRight, bottomLeft, bottomRight);
        }
    return mesh;
}

// Steps 3-5. The colour comes from the last frame (black where the colour camera doesn't reach).
// Returns the colour pixels that had depth, for the tint.
std::vector<cv::Point> ScanAndSave(const cv::Mat& averageMm, const spotter::RgbdFrame& frame)
{
    std::vector<cv::Vec3b> colours = frame.registered.reshape(0, 1);   // one colour per depth pixel
    spotter::Mesh mesh = MeshFromCloud(frame.depthCamera.Cloud(averageMm), colours);
    std::vector<cv::Point> tint;
    for (int i = 0; i < (int)averageMm.total(); i++)
        if (averageMm.at<float>(i) > 0 && frame.colourIndex[i] >= 0) tint.emplace_back(frame.colourIndex[i] % frame.colour.cols, frame.colourIndex[i] / frame.colour.cols);
    size_t triangles = mesh.groups["seen_room"].size();
    if (triangles == 0) { std::printf("Nothing to mesh: no depth readings.\n"); return tint; }
    std::string path = (std::filesystem::current_path() / cv::format("room_%lld.obj", (long long)std::time(nullptr))).string();
    spotter::TurnToYUp(mesh);   // 3D programs expect Y up
    if (spotter::SaveObj(path, mesh, "Spotter 09b: the room from one view"))
        std::printf("Saved %zu points and %zu triangles, %.1f MB, to %s\n  %.0f%% of the view had no depth. (A capture of the room: never commit it.)\n",
                    mesh.points.size(), triangles, std::filesystem::file_size(path) / 1e6, path.c_str(), 100.0 - 100.0 * tint.size() / averageMm.total());
    else std::printf("Could not write %s\n", path.c_str());
    return tint;
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

void Show(const cv::Mat& colour, const std::vector<cv::Point>& tint, const spotter::DepthAverager& averager, const spotter::FpsMeter& fps)
{
    cv::Mat display;
    cv::resize(colour, display, colour.size() / 2);
    for (const cv::Point& p : tint)
    {
        cv::Vec3b& pixel = display.at<cv::Vec3b>(p.y / 2, p.x / 2);
        pixel = cv::Vec3b(pixel[0] / 2, (pixel[1] + 255) / 2, pixel[2] / 2);
    }
    std::string status = averager.Running() ? cv::format("HOLD STILL  %d / %d", averager.Added(), averager.Frames()) : "S to scan the room";
    cv::putText(display, status, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    fps.Draw(display);
    cv::imshow(Window, spotter::WithKeys(display, Keys));
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    spotter::KinectSource kinect(spotter::KinectSource::Options{});   // colour + depth, 6 m
    if (!kinect.IsOpen()) return 1;

    cv::namedWindow(Window, cv::WINDOW_AUTOSIZE);
    spotter::DepthAverager averager(AverageFrames);
    std::vector<cv::Point> tint;
    spotter::FpsMeter fps;
    spotter::RgbdFrame frame;

    while (kinect.Next(frame))
    {
        if (averager.Running() && averager.Add(frame.depthMm)) tint = ScanAndSave(averager.Mean(), frame);   // steps 2-5
        fps.Tick();
        Show(frame.colour, tint, averager, fps);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 's' && !averager.Running()) { averager.Start(); std::printf("Scanning: hold still...\n"); }   // step 1
        if (key == 'c') tint.clear();
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Open the .obj in Blender (File > Import > Wavefront; show vertex colours) or MeshLab. Turn
//     it round: where are the holes, and what made each one?
//  2. Look at the room from above. Every hole is a "shadow" pointing away from the camera. Which
//     spot would you scan from next to fill the biggest one?
//  3. MaxJumpShare = 0.005 versus 0.2: a torn room versus webs from every chair to the wall behind.
//  4. MeshStep = 1 versus 4: file size and look.
//  5. AverageFrames = 1: look at a flat wall edge-on. How bumpy is one frame?
//  6. Scan, move the Kinect 1 m, scan again, and import both files. They overlap in the wrong
//     place: each was saved from where its own camera stood. Lining them up is example 11's job.
