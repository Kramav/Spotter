// =====================================================================
//  00g · Tour: Mesh, SaveObj, OrbitView and PointRenderer   (no Kinect needed)
//
//  What they're for:
//    MeshFromGrid   a depth cloud -> triangles (examples 08b, 09b, 09c)
//    SaveObj        triangles -> an .obj file for Blender or MeshLab
//    OrbitView      where a 3D viewer looks from; the mouse turns it
//    PointRenderer  draws points through an OrbitView, nearest wins
//
//  How they work: a depth cloud is a grid, so each square of 4 neighbouring
//  pixels is two triangles. Where the corners' depths jump (one thing in
//  front of another) the square is skipped, or a skin would join them.
//  The viewer: move the pivot to the middle, turn about the vertical axis
//  (yaw), tilt about the sideways axis (pitch), step back, then the
//  pinhole maths of tour 00b. A z-buffer keeps the nearest point per pixel.
//
//  This program makes a depth picture by hand (a wall, a box in front, a
//  dome), meshes it, saves tour_00g.obj where it runs (made-up shapes, no
//  camera picture), checks one viewer projection by hand, then opens the viewer.
//  Keys:  mouse drag / W A S D  turn    wheel / Z X  zoom    R  reset    Q or Esc  quit
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/frame.hpp"
#include "spotter/mesh.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "mouse drag / W A S D  turn    wheel / Z X  zoom    R  reset    Q or Esc  quit";

const spotter::PinholeCamera DepthCamera{ 365.5f, 365.5f, 256.0f, 212.0f, 512, 424 };

// A wall 2.5 m away, a box front 1.8 m away, and a dome bulging out of the wall.
cv::Mat MadeUpDepthMm()
{
    cv::Mat depthMm(424, 512, CV_32F, cv::Scalar(2500));
    depthMm(cv::Rect(60, 220, 140, 150)).setTo(1800);
    for (int row = 0; row < 424; row++)
        for (int column = 0; column < 512; column++)
        {
            float r = (float)cv::norm(cv::Point2f(column - 350.f, row - 170.f)) / 90;
            if (r < 1) depthMm.at<float>(row, column) -= 400 * std::sqrt(1 - r * r);
        }
    return depthMm;
}

spotter::Mesh MeshIt(const std::vector<cv::Point3f>& cloud)
{
    std::printf("1. MeshFromGrid: every 2nd pixel, squares skipped where depth jumps over 3%%:\n");
    std::vector<cv::Vec3b> colours;
    for (const cv::Point3f& p : cloud) colours.push_back(cv::Vec3b(80, (uchar)std::clamp(400 - p.z * 140, 0.f, 255.f), 200));   // nearer = more green
    spotter::Mesh mesh;
    spotter::MeshFromGrid(cloud, colours, 512, 2, 0.03f, mesh, "made_up");
    size_t triangles = mesh.groups["made_up"].size(), full = 2 * (512 / 2 - 1) * (424 / 2 - 1);
    std::printf("   %zu points, %zu triangles of %zu possible: the %zu missing are where the depth jumps:\n"
                "   the box's outline (wall 2.5 m, box 1.8 m) and the dome's steep rim. The rest of the dome stays whole.\n",
                mesh.points.size(), triangles, full, full - triangles);
    return mesh;
}

void Save(spotter::Mesh mesh)
{
    spotter::TurnToYUp(mesh);   // Blender and MeshLab expect Y up
    bool ok = spotter::SaveObj("tour_00g.obj", mesh, "Spotter tour 00g: made-up shapes");
    std::printf("2. SaveObj: %s tour_00g.obj (Y turned up for 3D programs). Open it in Blender or MeshLab.\n", ok ? "wrote" : "could NOT write");
}

void ProjectByHand(spotter::OrbitView& view)
{
    std::printf("3. OrbitView: where does the box's corner land when the view is turned 30 deg?\n");
    view.yawDegrees = 30;
    view.Update();
    cv::Point3f p = DepthCamera.Unproject(cv::Point2f(60, 220), 1.8f);
    const float yaw = 30 * (float)CV_PI / 180;
    float x = p.x - view.pivot.x, y = p.y - view.pivot.y, z = p.z - view.pivot.z;   // 1. pivot to the middle
    float x1 = std::cos(yaw) * x + std::sin(yaw) * z, z1 = -std::sin(yaw) * x + std::cos(yaw) * z;   // 2. yaw (pitch is 0)
    float depth = z1 + view.distance;                                                // 3. step back
    cv::Point byHand((int)(view.options.focalPixels * x1 / depth + view.options.width / 2),
                     (int)(view.options.focalPixels * y / depth + view.options.height / 2));   // 4. pinhole
    cv::Point pixel;
    float viewDepth;
    view.Project(p, pixel, viewDepth);
    std::printf("   by hand: pixel (%d, %d), %.2f m from the viewer;  OrbitView::Project: (%d, %d), %.2f m\n",
                byHand.x, byHand.y, depth, pixel.x, pixel.y, viewDepth);
    view.Reset();
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<cv::Point3f> cloud = DepthCamera.Cloud(MadeUpDepthMm());
    spotter::Mesh mesh = MeshIt(cloud);
    Save(mesh);

    const std::string window = "00g viewer: the mesh's points (drag/WASD turn, wheel/ZX zoom, R reset, Q quit)";
    cv::namedWindow(window);
    spotter::OrbitView view(spotter::OrbitView::Options(), cv::Point3f(0, 0, 2.2f), 2.2f);
    view.Attach(window);
    ProjectByHand(view);
    std::printf("4. The viewer: drag it side on to see the box stand out from the wall. Q closes.\n");

    spotter::PointRenderer renderer;
    while (true)
    {
        renderer.Begin(view);
        renderer.Draw(mesh.points, mesh.colours);
        cv::imshow(window, spotter::WithKeys(renderer.Image(), Keys));
        int key = cv::waitKey(15);
        if (spotter::IsQuitKey(key) || cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1) break;
        view.HandleKey(key);
    }
    return 0;
}
