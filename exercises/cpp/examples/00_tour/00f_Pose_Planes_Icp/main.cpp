// =====================================================================
//  00f · Tour: Pose, FindPlanes and Icp                (no Kinect needed)
//
//  What they're for:
//    Pose        where one camera stood relative to another: a turn and a
//                shift. Any point it sees maps across as  turn * p + shift.
//    Kabsch      the pose that best moves a set of points onto known partners.
//    FindPlanes  the room's flat surfaces (floor, ceiling, walls) in a cloud.
//    Icp         the pose between two views of the same room, with NO known
//                partners: pair each point with its nearest, fit, repeat.
//
//  How they work: Kabsch centres both sets; what's left is a turn, and the
//  SVD of how the sets vary together gives it. FindPlanes is RANSAC, again
//  and again (examples 06b, 09). Icp (example 11, then 13e's version) pairs
//  each point with the nearest target point, measures the gap along the
//  target's surface normal, and solves for the small move that closes the
//  gaps. A direction the room can't pin (sliding along a bare wall) is left
//  where it started: the result says how many there were ("weak").
//
//  This program builds a made-up room, "photographs" it with a made-up
//  depth camera from two spots (raycasting), and runs all four on it.
// =====================================================================

#include <cmath>
#include <cstdio>
#include <random>

#include "spotter/frame.hpp"
#include "spotter/geometry.hpp"
#include "spotter/register.hpp"

// The room in the first camera's coordinates (metres; X right, Y down, Z ahead): x -1.5 to 2.0,
// ceiling 1.3 m above, floor 1.2 m below, z -1.0 to 3.0, and a couch 2 m ahead on the right.
float Raycast(const cv::Point3f& from, const cv::Point3f& ray)
{
    const float low[3] = { -1.5f, -1.3f, -1.0f }, high[3] = { 2.0f, 1.2f, 3.0f };
    const float boxLow[3] = { 0.2f, 0.6f, 2.0f }, boxHigh[3] = { 1.0f, 1.2f, 2.5f };
    const float o[3] = { from.x, from.y, from.z }, d[3] = { ray.x, ray.y, ray.z };
    float wall = 1e9f, enter = 0, leave = 1e9f;
    for (int a = 0; a < 3; a++)
    {
        if (d[a] == 0) continue;
        wall = std::min(wall, ((d[a] > 0 ? high[a] : low[a]) - o[a]) / d[a]);   // from inside: the wall it heads for
        float t1 = (boxLow[a] - o[a]) / d[a], t2 = (boxHigh[a] - o[a]) / d[a];  // the couch, from outside
        enter = std::max(enter, std::min(t1, t2));
        leave = std::min(leave, std::max(t1, t2));
    }
    return enter < leave && enter > 0 ? std::min(wall, enter) : wall;
}

// What a 512 x 424 depth camera standing at `pose` sees, as a cloud in its own coordinates.
std::vector<cv::Point3f> Photograph(const spotter::Pose& pose)
{
    const spotter::PinholeCamera camera{ 365.5f, 365.5f, 256.0f, 212.0f, 512, 424 };
    cv::RNG noise(1);
    std::vector<cv::Point3f> cloud;
    for (int row = 0; row < 424; row++)
        for (int column = 0; column < 512; column++)
        {
            cv::Point3f ray = camera.Unproject(cv::Point2f((float)column, (float)row), 1.0f);
            cloud.push_back(ray * (Raycast(pose.shift, pose.turn * ray) + (float)noise.gaussian(0.002)));   // 2 mm of noise
        }
    return cloud;
}

void Print(const char* name, const spotter::Pose& pose)
{
    std::printf("   %-10s right %+.3f  down %+.3f  forward %+.3f m, turned %.2f deg\n", name, pose.shift.x, pose.shift.y,
                pose.shift.z, pose.Degrees());
}

void PosesAndKabsch(const spotter::Pose& truth)
{
    std::printf("1. A pose, and Kabsch:\n");
    cv::Point3f ahead(0, 0, 2);
    cv::Point3f moved = truth(ahead);
    std::printf("   B's camera sees a point 2 m straight ahead; in A's coordinates it's at (%+.3f, %+.3f, %.3f):\n"
                "   B turned left, so B's 'ahead' is a little to A's left (and B stands 10 cm right of A).\n",
                moved.x, moved.y, moved.z);
    std::vector<cv::Point3f> from = { { 0, 0, 2 }, { 1, 0, 3 }, { 0, 1, 2.5f }, { -1, -0.5f, 1.5f } }, to;
    for (const cv::Point3f& p : from) to.push_back(truth(p));
    Print("true", truth);
    Print("Kabsch", spotter::Kabsch(from, to));
    std::printf("   Four exact pairs are plenty; with real data every pair is a little off, and more pairs average it out.\n");
}

void Planes(const std::vector<cv::Point3f>& cloud)
{
    std::printf("2. FindPlanes on what camera A sees (every 4th pixel each way):\n");
    std::mt19937 random(1);
    for (const spotter::PlanePiece& piece : spotter::FindPlanes(spotter::ThinCloud(cloud, 512, 4), spotter::PlaneSearch(), random))
    {
        const cv::Point3f& n = piece.plane.normal;
        const char* name = n.y < -0.9f ? "floor" : n.y > 0.9f ? "ceiling" : std::abs(n.y) < 0.1f ? "wall (or couch side)" : "slanted";
        std::printf("   %-20s normal (%+.2f, %+.2f, %+.2f), %.3f m from the camera, %4zu points\n", name, n.x, n.y, n.z,
                    piece.plane.distance, piece.cells.size());
    }
    std::printf("   The normal points at the camera, so a floor's is (0, -1, 0): Y points down.\n");
}

void IcpBetweenViews(const std::vector<cv::Point3f>& a, const std::vector<cv::Point3f>& b, const spotter::Pose& truth)
{
    std::printf("3. Icp: B's view onto A's, starting from 'not moved':\n");
    spotter::SurfacePoints surfaceA = spotter::SampleSurface(a, 512, 4), surfaceB = spotter::SampleSurface(b, 512, 4);
    spotter::IcpOptions options;
    options.startPairMeters = 0.20f;   // B is 12 cm off: allow pairs that far apart at first
    spotter::IcpResult fit = spotter::Icp(surfaceB.points, surfaceA.points, surfaceA.normals, spotter::Pose(), options);
    Print("true", truth);
    Print("Icp", fit.pose);
    std::printf("   %zu points each; %.0f%% paired at the end, %.1f mm apart (the 2 mm noise), %d weak directions\n",
                surfaceB.points.size(), fit.pairedShare * 100, fit.rmsMeters * 1000, fit.weakDirections);

    std::printf("4. The same with only the back wall (a bare wall, nothing else):\n");
    spotter::SurfacePoints wallA, wallB;
    for (size_t i = 0; i < surfaceA.points.size(); i++)
        if (surfaceA.points[i].z > 2.95f) { wallA.points.push_back(surfaceA.points[i]); wallA.normals.push_back(surfaceA.normals[i]); }
    for (size_t i = 0; i < surfaceB.points.size(); i++)
        if (truth(surfaceB.points[i]).z > 2.95f) wallB.points.push_back(surfaceB.points[i]);
    fit = spotter::Icp(wallB.points, wallA.points, wallA.normals, spotter::Pose(), options);
    Print("Icp", fit.pose);
    std::printf("   %d weak directions: sliding along the wall (left/right, up/down) and turning about the line of\n"
                "   sight fit a flat wall equally well, so Icp doesn't move along them: right is %.0f cm off, down %.0f cm.\n"
                "   The distance and the turn come out. That's why 11b finds a rough pose from colour features first.\n",
                fit.weakDirections, 100 * std::abs(fit.pose.shift.x - truth.shift.x), 100 * std::abs(fit.pose.shift.y - truth.shift.y));
}

int main()
{
    // Camera B: 10 cm right, 5 cm forward, 3 cm up (Y points down), turned 5 deg left.
    const spotter::Pose b{ spotter::TurnAbout(cv::Vec3d(0, -5 * CV_PI / 180, 0)), cv::Point3f(0.10f, -0.03f, 0.05f) };
    PosesAndKabsch(b);
    std::vector<cv::Point3f> seenByA = Photograph(spotter::Pose()), seenByB = Photograph(b);
    Planes(seenByA);
    IcpBetweenViews(seenByA, seenByB, b);
    std::printf("\nPress Enter to close.\n");
    std::getchar();
    return 0;
}
