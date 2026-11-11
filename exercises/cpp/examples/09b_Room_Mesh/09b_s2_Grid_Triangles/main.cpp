// =====================================================================
//  09b · step 2 of 2 · Every square becomes two triangles
//
//  Step 1's scan as 3D points (08), coloured. The depth picture is already
//  a grid, so neighbouring points are neighbours in 3D too: each 2x2
//  square of points (every 2nd pixel) becomes two triangles. No jump rule
//  yet: every square with four points is joined, also from the edge of a
//  chair to the wall behind it. Saved as room_<time>.obj.
//
//  New in this step: a grid of points -> triangles.
//  Keys:  S  scan and save     Q or Esc  quit
//  Next:  09b_Room_Mesh (the full example skips triangles across depth jumps)
//  Privacy: an .obj is a 3D capture of the room: never commit one. Scan with nobody in view.
// =====================================================================

#include <cmath>
#include <cstdio>
#include <ctime>
#include <opencv2/highgui.hpp>

#include "spotter/depth.hpp"
#include "spotter/mesh.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "S  scan and save     Q or Esc  quit";

const int Step = 2;

//spotter::Mesh Triangles(const std::vector<cv::Point3f>& cloud, const std::vector<cv::Vec3b>& colours){
//
//@param1 - cloud: a depth cloud, 512 wide (NaN = no reading)
//@param2 - colours: one colour per cloud point
//
//@brief - a point per Step-th pixel with a reading, and two triangles per square of neighbouring kept pixels
//         (group seen_room); this by-hand version is what MeshFromGrid does, without the jump test yet. O(pixels /
//         Step^2).
spotter::Mesh Triangles(const std::vector<cv::Point3f>& cloud, const std::vector<cv::Vec3b>& colours)
{
    const int columns = 512 / Step, rows = 424 / Step;
    spotter::Mesh mesh;
    std::vector<int> pointOf(columns * rows, -1);
    for (int row = 0; row < rows; row++)
        for (int column = 0; column < columns; column++)
        {
            int pixel = row * Step * 512 + column * Step;
            if (std::isnan(cloud[pixel].z)) continue;
            pointOf[row * columns + column] = (int)mesh.points.size();
            mesh.points.push_back(cloud[pixel]);
            mesh.colours.push_back(colours[pixel]);
        }
    for (int row = 0; row + 1 < rows; row++)
        for (int column = 0; column + 1 < columns; column++)
        {
            int a = pointOf[row * columns + column], b = pointOf[row * columns + column + 1];
            int c = pointOf[(row + 1) * columns + column], d = pointOf[(row + 1) * columns + column + 1];
            if (a >= 0 && b >= 0 && c >= 0) mesh.groups["seen_room"].emplace_back(a, c, b);   // top-left, bottom-left, top-right
            if (b >= 0 && c >= 0 && d >= 0) mesh.groups["seen_room"].emplace_back(b, c, d);
        }
    return mesh;
}

//int main(){
//
//@brief - the program: 09b · step 2 of 2 · Every square becomes two triangles. Step 1's scan as 3D points (08),
//         coloured. The depth picture is already a grid, so neighbouring points are neighbours in 3D too: each 2x2
//         square of points (every 2nd pixel) becomes two triangles. Setup, main loop and clean-up are laid out in
//         the file's sections. Returns 0 on a normal exit, 1 if it couldn't start.
int main()
{
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!kinect.IsOpen()) return 1;

    spotter::DepthAverager averager(30);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        if (averager.Running() && averager.Add(frame.depthMm))
        {
            spotter::Mesh mesh = Triangles(frame.depthCamera.Cloud(averager.Mean()), frame.registered.reshape(0, 1));
            spotter::TurnToYUp(mesh);
            std::string path = cv::format("room_%lld.obj", (long long)std::time(nullptr));
            if (spotter::SaveObj(path, mesh, "Spotter 09b step 2: no jump rule"))
                std::printf("Saved %zu triangles to %s (never commit it)\n", mesh.groups["seen_room"].size(), path.c_str());
        }
        cv::imshow("09b step 2: S scans and saves", spotter::WithKeys(frame.colour, Keys));
        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 's' && !averager.Running()) { averager.Start(); std::printf("Scanning: hold still...\n"); }
    }
    return 0;
}

// Try: open it and look from the side: every object is joined to the wall behind it by a sheet of
// stretched triangles. The full example skips a triangle when its corners jump in depth.
