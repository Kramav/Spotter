// Self-check for mesh.hpp and DepthAverager (depth.hpp), with no Kinect: a made-up depth grid
// becomes the right triangles (none across a depth jump), a box faces outwards, and averaging
// fills sometimes-holes but not mostly-holes.
// Run: build\check_mesh.exe. Exit code 0 = passed.

#include "spotter/depth.hpp"
#include "spotter/mesh.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>

int failures = 0;
void Check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAILED: %s\n", what); failures++; }
}

int main()
{
    // 1. A flat 10 x 6 grid at 2 m: 9 x 5 squares, two triangles each. Then a step in depth down the
    // middle (2 m -> 3 m): the squares across it are skipped.
    const int width = 10, height = 6;
    std::vector<cv::Point3f> cloud(width * height);
    std::vector<cv::Vec3b> colours(width * height, cv::Vec3b(10, 20, 30));
    for (int i = 0; i < width * height; i++) cloud[i] = cv::Point3f((i % width) * 0.01f, (i / width) * 0.01f, 2.0f);
    spotter::Mesh flat;
    spotter::MeshFromGrid(cloud, colours, width, 1, 0.03f, flat, "seen");
    Check(flat.points.size() == 60 && flat.groups["seen"].size() == 2 * 9 * 5, "a flat grid: two triangles per square");
    for (int i = 0; i < width * height; i++) if (i % width >= 5) cloud[i].z = 3.0f;
    spotter::Mesh stepped;
    spotter::MeshFromGrid(cloud, colours, width, 1, 0.03f, stepped, "seen");
    Check(stepped.groups["seen"].size() == 2 * 8 * 5, "no triangles across a depth jump");
    cloud[0].z = std::numeric_limits<float>::quiet_NaN();
    spotter::Mesh holed;
    spotter::MeshFromGrid(cloud, colours, width, 1, 0.03f, holed, "seen");
    Check(holed.points.size() == 59 && holed.groups["seen"].size() == 2 * 8 * 5 - 1, "a hole loses its point and the one triangle using it alone");

    // 2. A box: 12 triangles, each facing away from the box's centre.
    spotter::Mesh box;
    spotter::AddBox(box, "box", { 0, 0, 0 }, { 1, 2, 3 }, cv::Vec3b(1, 2, 3));
    bool outwards = box.groups["box"].size() == 12;
    for (const cv::Vec3i& f : box.groups["box"])
    {
        cv::Point3f a = box.points[f[0]], b = box.points[f[1]], c = box.points[f[2]];
        outwards = outwards && (b - a).cross(c - a).dot((a + b + c) / 3 - cv::Point3f(0.5f, 1, 1.5f)) > 0;
    }
    Check(outwards, "a box's 12 triangles all face out");

    // 3. SaveObj: one v line per point, one f line per triangle, one g line per group.
    std::string path = (std::filesystem::temp_directory_path() / "spotter_check_mesh.obj").string();
    spotter::TurnToYUp(box);
    Check(spotter::SaveObj(path, box, "check"), "SaveObj writes");
    std::ifstream file(path);
    int v = 0, f = 0, g = 0;
    for (std::string line; std::getline(file, line);) { v += line[0] == 'v'; f += line[0] == 'f'; g += line[0] == 'g'; }
    file.close();
    std::filesystem::remove(path);
    Check(v == 24 && f == 12 && g == 1, "the OBJ has every point, triangle and group");

    // 4. Averaging: pixel 0 reads 1000/1010/1020 mm (mean 1010); pixel 1 reads once in 3 (a hole);
    // pixel 2 reads twice in 3 (filled).
    spotter::DepthAverager averager(3);
    averager.Start();
    float frames[3][3] = { { 1000, 0, 500 }, { 1010, 0, 0 }, { 1020, 900, 700 } };
    bool done = false;
    for (auto& frame : frames) done = averager.Add(cv::Mat(1, 3, CV_32F, frame));
    cv::Mat mean = averager.Mean(0.5);
    Check(done && !averager.Running(), "done after the 3rd frame");
    Check(std::abs(mean.at<float>(0) - 1010) < 1e-3 && mean.at<float>(1) == 0 && std::abs(mean.at<float>(2) - 600) < 1e-3,
          "the mean of the readings; a pixel that read in under half the frames stays a hole");

    std::printf(failures == 0 ? "PASSED\n" : "%d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
