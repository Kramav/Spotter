// Self-check for Pose/Kabsch (geometry.hpp) and register.hpp, with no Kinect:
//   1. Kabsch recovers a known turn + shift from exact pairs.
//   2. ICP lines up two raycast views of a made-up room, taken from camera spots 12 cm and 6 deg apart.
//   3. Bug 2: a NaN normal (one bad target point) mustn't make the whole pose NaN, and
//      SampleSurface never hands out one.
//   4. Bug 13: a small corner 0.8 m away (radius about 0.1 m) still pins all 6 directions.
// Run: build\check_icp.exe. Exit code 0 = passed.

#include "spotter/frame.hpp"
#include "spotter/register.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

int failures = 0;
void Check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAILED: %s\n", what); failures++; }
}

const float NaN = std::numeric_limits<float>::quiet_NaN();

// How far apart two poses are: the shift in mm, the turn in degrees.
void Off(const spotter::Pose& a, const spotter::Pose& b, double& mm, double& degrees)
{
    mm = cv::norm(a.shift - b.shift) * 1000;
    degrees = spotter::Pose{ a.turn.t() * b.turn, cv::Point3f() }.Degrees();
}

// The room in camera A's coordinates (Y down): a box from x -1.5 to 2.0, y -1.3 (ceiling) to 1.2
// (floor), z -1.0 to 3.0, with a couch (0.8 x 0.6 x 0.5 m) on the floor 2 m ahead.
float Raycast(const cv::Point3f& from, const cv::Point3f& ray)
{
    const float low[3] = { -1.5f, -1.3f, -1.0f }, high[3] = { 2.0f, 1.2f, 3.0f };
    const float o[3] = { from.x, from.y, from.z }, d[3] = { ray.x, ray.y, ray.z };
    float best = 1e9f;
    for (int a = 0; a < 3; a++)   // from inside the room: the wall each way the ray heads
        if (d[a] != 0) best = std::min(best, ((d[a] > 0 ? high[a] : low[a]) - o[a]) / d[a]);
    const float boxLow[3] = { 0.2f, 0.6f, 2.0f }, boxHigh[3] = { 1.0f, 1.2f, 2.5f };
    float enter = 0, leave = 1e9f;   // the couch, from outside: the slab method
    for (int a = 0; a < 3; a++)
    {
        if (d[a] == 0) { if (o[a] < boxLow[a] || o[a] > boxHigh[a]) leave = -1; continue; }
        float t1 = (boxLow[a] - o[a]) / d[a], t2 = (boxHigh[a] - o[a]) / d[a];
        enter = std::max(enter, std::min(t1, t2));
        leave = std::min(leave, std::max(t1, t2));
    }
    if (enter < leave && enter > 0) best = std::min(best, enter);
    return best;
}

// What a depth camera standing at `pose` (its coordinates -> A's) sees, in its own coordinates.
std::vector<cv::Point3f> View(const spotter::Pose& pose, int seed)
{
    spotter::PinholeCamera camera{ 365.5f, 365.5f, 256.0f, 212.0f, 512, 424 };
    cv::RNG noise(seed);
    std::vector<cv::Point3f> cloud(512 * 424);
    for (int row = 0; row < 424; row++)
        for (int column = 0; column < 512; column++)
        {
            cv::Point3f ray = camera.Unproject(cv::Point2f((float)column, (float)row), 1.0f);
            float t = Raycast(pose.shift, pose.turn * ray) + (float)noise.gaussian(0.002);
            cloud[row * 512 + column] = ray * t;
        }
    return cloud;
}

int main()
{
    // 1. Kabsch on exact pairs.
    spotter::Pose truth{ spotter::TurnAbout(cv::Vec3d(0.1, -0.3, 0.05)), cv::Point3f(0.4f, -0.1f, 0.25f) };
    cv::RNG random(3);
    std::vector<cv::Point3f> from, to;
    for (int i = 0; i < 50; i++)
    {
        from.emplace_back(random.uniform(-2.f, 2.f), random.uniform(-1.f, 1.f), random.uniform(0.5f, 4.f));
        to.push_back(truth(from.back()));
    }
    double mm, degrees;
    Off(spotter::Kabsch(from, to), truth, mm, degrees);
    std::printf("1. Kabsch: %.4f mm, %.5f deg off\n", mm, degrees);
    Check(mm < 0.01 && degrees < 0.001, "Kabsch recovers an exact move");

    // 2. ICP between two raycast views: B stands 10 cm right, 5 cm forward and 3 cm up, turned
    //    5 deg left and tipped 2 deg down. ICP starts from "not moved" (as 11 does).
    spotter::Pose b{ spotter::TurnAbout(cv::Vec3d(0, -5, 0) * (CV_PI / 180)) * spotter::TurnAbout(cv::Vec3d(-2, 0, 0) * (CV_PI / 180)),
                     cv::Point3f(0.10f, -0.03f, 0.05f) };
    spotter::SurfacePoints a = spotter::SampleSurface(View(spotter::Pose(), 1), 512, 4);
    spotter::SurfacePoints seenByB = spotter::SampleSurface(View(b, 2), 512, 4);
    spotter::IcpOptions wide;
    wide.startPairMeters = 0.20f;   // the start is 12 cm off: 13e's 5 cm assumes the features came first
    spotter::IcpResult fit = spotter::Icp(seenByB.points, a.points, a.normals, spotter::Pose(), wide);
    Off(fit.pose, b, mm, degrees);
    std::printf("2. ICP: %zu and %zu points; %.1f mm, %.3f deg off; %.0f%% paired, %.1f mm rms, %d weak\n", a.points.size(),
                seenByB.points.size(), mm, degrees, fit.pairedShare * 100, fit.rmsMeters * 1000, fit.weakDirections);
    Check(mm < 3 && degrees < 0.1, "ICP recovers B's pose within 3 mm and 0.1 deg");
    Check(fit.weakDirections == 0, "a room with a couch pins all 6 directions");

    // 3. Bug 2. Every 10th target normal NaN: the pose must stay a number, and still be right.
    std::vector<cv::Point3f> spoilt = a.normals;
    for (size_t i = 0; i < spoilt.size(); i += 10) spoilt[i] = cv::Point3f(NaN, NaN, NaN);
    fit = spotter::Icp(seenByB.points, a.points, spoilt, spotter::Pose(), wide);
    Off(fit.pose, b, mm, degrees);
    std::printf("3. NaN normals: %.1f mm, %.3f deg off\n", mm, degrees);
    Check(std::isfinite(mm) && mm < 3 && degrees < 0.1, "a NaN normal doesn't spoil the pose");
    std::vector<cv::Point3f> flat(64 * 64, cv::Point3f(0, 0, 1));   // every neighbour the same point: no facing
    spotter::SurfacePoints none = spotter::SampleSurface(flat, 64, 2);
    Check(none.points.empty(), "SampleSurface leaves out points whose neighbours give no facing");

    // 4. Bug 13. A corner of three 20 cm squares, 0.8 m ahead, moved 1.5 deg and 6 mm. 13e scored
    //    turns in metres squared at the lever arm (0.1 m -> 0.01): all three turns came out "weak".
    std::vector<cv::Point3f> corner, cornerNormals, moved;
    const cv::Point3f origin(-0.1f, -0.1f, 0.8f);
    for (int face = 0; face < 3; face++)
        for (int i = 0; i < 20; i++)
            for (int j = 0; j < 20; j++)
            {
                cv::Point3f along[3] = { { 0, 1, 0 }, { 1, 0, 0 }, { 1, 0, 0 } }, across[3] = { { 0, 0, 1 }, { 0, 0, 1 }, { 0, 1, 0 } };
                cv::Point3f facing[3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, -1 } };
                corner.push_back(origin + along[face] * (0.01f * (i + 1)) + across[face] * (0.01f * (j + 1)));   // 1 cm clear of the seams
                cornerNormals.push_back(facing[face]);
                moved.push_back(origin + along[face] * (0.01f * (i + 1) + 0.005f) + across[face] * (0.01f * (j + 1) + 0.005f));   // sampled between
            }
    spotter::Pose nudge{ spotter::TurnAbout(cv::Vec3d(0.6, -1.0, 0.8) * (1.5 * CV_PI / 180 / std::sqrt(2.0))),
                         cv::Point3f(0.004f, -0.003f, 0.003f) };
    spotter::Pose back{ nudge.turn.t(), -(nudge.turn.t() * nudge.shift) };   // nudge undone
    for (cv::Point3f& p : moved) p = back(p);
    fit = spotter::Icp(moved, corner, cornerNormals, spotter::Pose());
    Off(fit.pose, nudge, mm, degrees);
    std::printf("4. small corner: %.2f mm, %.3f deg off, %d weak directions\n", mm, degrees, fit.weakDirections);
    Check(fit.weakDirections == 0, "a small near corner pins all 6 directions");
    Check(mm < 0.5 && degrees < 0.05, "and ICP recovers its move");

    std::printf(failures ? "check_icp: %d FAILED\n" : "check_icp: all passed\n", failures);
    return failures ? 1 : 0;
}
