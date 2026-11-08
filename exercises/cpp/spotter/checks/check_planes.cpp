// Self-check for geometry.hpp and walls.hpp, with no Kinect: a made-up room seen by a made-up
// depth camera (raycast, with a little noise). FindPlanes must find its floor, ceiling and walls,
// and FindWalls its two walls (not the couch). Plus the review's bug 8: a plane whose points are too
// scattered to be one surface must not hide the real planes behind it.
// Run: build\check_planes.exe. Exit code 0 = passed.

#include "spotter/frame.hpp"
#include "spotter/walls.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

int failures = 0;
void Check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAILED: %s\n", what); failures++; }
}

const float NaN = std::numeric_limits<float>::quiet_NaN();

// The room, in camera coordinates (Y down): back wall 3 m ahead, left wall 1.5 m to the left,
// floor 1.2 m below, ceiling 1.3 m above, and a couch front 0.6 m tall standing 2 m ahead.
float Raycast(const cv::Point3f& ray)
{
    float best = 1e9f;
    auto Hit = [&](float t) { if (t > 0 && t < best) best = t; };
    if (ray.z > 0) Hit(3.0f / ray.z);
    if (ray.x < 0) Hit(-1.5f / ray.x);
    if (ray.y > 0) Hit(1.2f / ray.y);
    if (ray.y < 0) Hit(-1.3f / ray.y);
    if (ray.z > 0)
    {
        float t = 2.0f / ray.z;
        cv::Point3f p = ray * t;
        if (p.x > 0.2f && p.x < 1.0f && p.y > 0.6f && p.y < 1.2f) Hit(t);
    }
    return best;
}

int main()
{
    spotter::PinholeCamera camera{ 365.5f, 365.5f, 256.0f, 212.0f, 512, 424 };
    cv::RNG noise(7);
    std::vector<cv::Point3f> cloud(512 * 424);
    for (int row = 0; row < 424; row++)
        for (int column = 0; column < 512; column++)
        {
            cv::Point3f ray = camera.Unproject(cv::Point2f((float)column, (float)row), 1.0f);   // a point at z = 1 on this pixel's ray
            float t = Raycast(ray) + (float)noise.gaussian(0.003);
            cloud[row * 512 + column] = ray * t;
        }
    spotter::PointGrid grid = spotter::ThinCloud(cloud, 512, 4);
    Check(grid.columns == 128 && grid.rows == 106, "thinned to every 4th pixel");

    // 1. The planes, facing any way.
    std::mt19937 random(1);
    std::vector<spotter::PlanePiece> pieces = spotter::FindPlanes(grid, spotter::PlaneSearch(), random);
    auto Found = [&](cv::Point3f normal, float distance)
    {
        for (const spotter::PlanePiece& piece : pieces)
            if (piece.plane.normal.dot(normal) > 0.999f && std::abs(piece.plane.distance - distance) < 0.01f) return true;
        return false;
    };
    Check(Found({ 0, -1, 0 }, 1.2f), "the floor: normal up (towards the camera), 1.2 m below");
    Check(Found({ 0, 1, 0 }, 1.3f), "the ceiling, 1.3 m above");
    Check(Found({ 0, 0, -1 }, 3.0f), "the back wall, 3 m ahead");
    Check(Found({ 1, 0, 0 }, 1.5f), "the left wall, 1.5 m to the left");
    std::printf("%zu planes found\n", pieces.size());

    // 2. Walls: upright, tall enough. The couch front (0.6 m) is not one.
    random.seed(1);
    std::vector<spotter::Wall> walls = spotter::FindWalls(grid, spotter::WallSearch(), random);
    bool back = false, left = false;
    for (const spotter::Wall& wall : walls)
    {
        std::printf("  wall %.3f m away, %+.1f deg, %.2f m tall, from (%+.2f, %.2f) to (%+.2f, %.2f)\n", wall.meters, wall.degrees,
                    wall.heightMeters, wall.ends[0].x, wall.ends[0].y, wall.ends[1].x, wall.ends[1].y);
        back = back || (std::abs(wall.meters - 3.0) < 0.01 && std::abs(wall.degrees) < 1);
        left = left || std::abs(wall.meters - 1.5) < 0.01;
    }
    Check(walls.size() == 2 && back && left, "two walls (back 3 m facing the camera, left 1.5 m), no couch");

    // 3. Bug 8: plane A's points are isolated cells spread over the whole view, more of them than
    // plane B's solid patch. RANSAC picks A first; its biggest piece is one cell. Before the fix A
    // stayed in play and was found again on every search, so B was never found.
    spotter::PointGrid scattered;
    scattered.columns = 128;
    scattered.rows = 106;
    scattered.points.assign(128 * 106, cv::Point3f(NaN, NaN, NaN));
    for (int row = 0; row < 106; row++)
        for (int column = 0; column < 128; column++)
        {
            cv::Point3f ray = camera.Unproject(cv::Point2f(column * 4.0f, row * 4.0f), 1.0f);
            if (row % 3 == 0 && column % 3 == 0) scattered.points[row * 128 + column] = ray * 2.0f;   // A: z = 2, isolated cells
            else if (row >= 40 && row < 70 && column >= 40 && column < 70) scattered.points[row * 128 + column] = ray * 3.0f;   // B: z = 3, solid
        }
    random.seed(1);
    spotter::PlaneSearch search;
    search.maxPlanes = 3;
    pieces = spotter::FindPlanes(scattered, search, random);
    Check(pieces.size() == 1 && std::abs(pieces[0].plane.distance - 3.0f) < 0.01f, "a scattered plane doesn't hide the solid one behind it");

    // 4. The basics.
    spotter::Plane plane;
    Check(!spotter::PlaneThrough({ 0, 0, 1 }, { 1, 0, 1 }, { 2, 0, 1 }, plane), "three points in a line make no plane");
    Check(spotter::PlaneThrough({ 0, 1, 2 }, { 1, 1, 2 }, { 0, 1, 3 }, plane) && plane.distance > 0 && plane.DistanceTo({ 0, 0, 0 }) > 0,
          "the normal points to the camera's side");

    std::printf(failures == 0 ? "PASSED\n" : "%d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
