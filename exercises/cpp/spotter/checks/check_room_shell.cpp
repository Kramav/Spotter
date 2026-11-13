// Self-check for 13w's surfaces, texture and fill (examples/13w_Room_Shell/room_shell.hpp), with no Kinect
// (review 2026-11-11, initiative 6: this replaces the temporary harness behind D42).
//   1. FindCeiling: the highest BIG level surface. A small patch above the ceiling and a fan below it don't
//      win; with no big level surface above 1.8 m there is none.
//   2. FindSurfaces on check_room's patterned box room (couch included), from 5 shots at their true poses:
//      the floor 1.2 m below the first camera, the ceiling 2.5 m above the floor, the walls.
//   3. Texture: every seen texel has its true surface's colour, also on the wall and floor behind the couch
//      (the depth must agree: the couch's colour never lands on the wall).
//   4. Fill: seen texels are left alone; the floor under the couch is filled with the floor's own pattern,
//      not a smooth smear (its colours vary about as much as the seen floor's).
// Not covered: the full colour picture's path (Sight with a colour model), since the shots here have none.
// Run: build\check_room_shell.exe. Exit code 0 = passed. Writes nothing.

#include "../../examples/13w_Room_Shell/room_shell.hpp"

#include <cstdio>

int failures = 0;
void Check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAILED: %s\n", what); failures++; }
}

const float Low[3] = { -1.5f, -1.3f, -1.0f }, High[3] = { 2.0f, 1.2f, 3.0f };            // the room (Y down), as check_room
const float CouchLow[3] = { 0.2f, 0.6f, 2.0f }, CouchHigh[3] = { 1.0f, 1.2f, 2.5f };
const spotter::PinholeCamera Camera{ 365.5f, 365.5f, 256.0f, 212.0f, 512, 424 };

float Raycast(const cv::Point3f& from, const cv::Point3f& ray, cv::Vec3b& colour)
{
    const float o[3] = { from.x, from.y, from.z }, d[3] = { ray.x, ray.y, ray.z };
    float best = 1e9f;
    int face = 0;
    for (int a = 0; a < 3; a++)
        if (d[a] != 0)
        {
            float t = ((d[a] > 0 ? High[a] : Low[a]) - o[a]) / d[a];
            if (t < best) { best = t; face = 2 * a + (d[a] > 0); }
        }
    float enter = 0, leave = 1e9f;
    int enterFace = 0;
    for (int a = 0; a < 3; a++)
    {
        float t1 = (CouchLow[a] - o[a]) / d[a], t2 = (CouchHigh[a] - o[a]) / d[a];
        if (std::min(t1, t2) > enter) { enter = std::min(t1, t2); enterFace = 6 + a; }
        leave = std::min(leave, std::max(t1, t2));
    }
    if (enter < leave && enter > 0 && enter < best) { best = enter; face = enterFace; }
    // The block from 1 mm in front of the surface (check_room's uses the hit point itself, where y = 1.2 falls in
    // block 11 or 12 by rounding: two rays to one spot could disagree, and step 3 compares two rays).
    cv::Point3f p = from + ray * (best - 0.001f / (float)cv::norm(ray));
    int i = (int)std::floor(p.x / 0.1f), j = (int)std::floor(p.y / 0.1f), k = (int)std::floor(p.z / 0.1f);
    unsigned h = (unsigned)(face * 73856093) ^ (unsigned)(i * 19349663) ^ (unsigned)(j * 83492791) ^ (unsigned)(k * 2654435761u);
    h ^= h >> 13; h *= 0x5bd1e995; h ^= h >> 15;
    colour = cv::Vec3b((uchar)(h & 255), (uchar)((h >> 8) & 255), (uchar)((h >> 16) & 255));
    return best;
}

Shot Photograph(const spotter::Pose& pose, int seed)
{
    cv::RNG noise(seed);
    std::vector<cv::Point3f> cloud(512 * 424);
    cv::Mat picture(424, 512, CV_8UC3);
    for (int row = 0; row < 424; row++)
        for (int column = 0; column < 512; column++)
        {
            cv::Point3f ray = Camera.Unproject(cv::Point2f((float)column, (float)row), 1.0f);
            cv::Vec3b colour;
            float t = Raycast(pose.shift, pose.turn * ray, colour);
            cloud[row * 512 + column] = ray * (t * (1 + (float)noise.gaussian(0.0005)));
            picture.at<cv::Vec3b>(row, column) = colour;
        }
    Shot shot;
    shot.scan = spotter::MakeScan(cloud, picture, spotter::ScanOptions());
    shot.scan.pose = pose;
    return shot;   // no full colour picture: Sight takes the depth grid's colour
}

spotter::Pose Spot(float right, float forward, float leftDegrees, float downDegrees)
{
    return { spotter::TurnAbout(cv::Vec3d(0, -leftDegrees * CV_PI / 180, 0)) * spotter::TurnAbout(cv::Vec3d(-downDegrees * CV_PI / 180, 0, 0)),
             cv::Point3f(right, 0, forward) };
}

// Texels this close to another surface (a corner, the couch's foot) are left out of step 3: the depth test's
// 4 cm + 1% lets the neighbour's colour in there. Measured: with none left out, 88% of the texels behind the
// couch are right (a strip under 3 cm wide at its foot takes the couch's colour); with 3 cm, 99.7%.
const float ClearMeters = 0.03f;

// A level patch of points every 3 cm, facing down, at height y over [x0, x1] x [z0, z1].
void AddLevel(std::vector<cv::Point3f>& points, std::vector<cv::Point3f>& normals, float y, float x0, float x1, float z0, float z1)
{
    for (float x = x0; x < x1; x += 0.03f)
        for (float z = z0; z < z1; z += 0.03f) { points.emplace_back(x, y, z); normals.emplace_back(0.0f, -1.0f, 0.0f); }
}

int main()
{
    // 1. FindCeiling, the floor at 0.
    std::vector<cv::Point3f> points, normals;
    AddLevel(points, normals, 2.5f, -1, 1, -1, 1);           // the ceiling, 4 m^2
    AddLevel(points, normals, 2.7f, 0, 0.3f, 0, 0.3f);       // a small patch higher up (0.09 m^2: a skylight's edge, noise)
    AddLevel(points, normals, 2.2f, -0.2f, 0.4f, -0.2f, 0.4f);   // a fan's blades (0.36 m^2)
    float ceilingY = 0;
    bool found = FindCeiling(points, normals, 0, ceilingY);
    std::printf("1. ceiling at %.3f m\n", ceilingY);
    Check(found && std::abs(ceilingY - 2.5f) < 0.02f, "the ceiling is the highest big level surface (not the small patch above, not the fan)");
    points.clear();
    normals.clear();
    AddLevel(points, normals, 2.2f, -0.2f, 0.4f, -0.2f, 0.4f);
    AddLevel(points, normals, 1.0f, -1, 1, -1, 1);           // a big table top, under 1.8 m
    Check(!FindCeiling(points, normals, 0, ceilingY), "no big level surface above 1.8 m: no ceiling");

    // 2. The surfaces, from 5 shots: 3 looking down at the floor, one up at the ceiling, one back.
    Shell shell;
    shell.info.depthCamera = Camera;
    const spotter::Pose spots[] = { Spot(0, 0, 0, 15), Spot(0.3f, 0.2f, 40, 15), Spot(-0.3f, 0.3f, -50, 15), Spot(0, 0.5f, 0, -30), Spot(0.3f, 1.0f, 180, 10) };
    int seed = 1;
    for (const spotter::Pose& pose : spots) shell.shots.push_back(Photograph(pose, seed++));
    found = FindSurfaces(shell);
    std::printf("2. floor %.3f, ceiling %.3f (%.3f above the floor), %zu surfaces\n", shell.floorY, shell.ceilingY, shell.ceilingY - shell.floorY, shell.surfaces.size());
    Check(found && std::abs(shell.floorY + 1.2f) < 0.03f, "the floor 1.2 m below the first camera");
    Check(std::abs(shell.ceilingY - shell.floorY - 2.5f) < 0.03f, "the ceiling 2.5 m above the floor");
    Check(shell.surfaces.size() >= 5, "the floor, the ceiling and at least 3 walls");

    // 3. Texture: each seen texel against its true colour. The levelled room -> the raycast room is
    // firstCamera.t(); the truth is what a ray from 2 cm inside the room hits, straight at the surface, at the
    // texel's centre or 1.5 cm to any side (the 10 cm colour blocks' edges).
    std::mt19937 random(1);
    int seen = 0, right = 0, behind = 0, behindRight = 0;
    double seenSpread = 0, filledSpread = 0;
    bool untouched = true;
    for (Surface& s : shell.surfaces)
    {
        Texture(s, shell.shots, Camera);
        const cv::Point3f in = shell.firstCamera.t() * s.inward;
        for (int row = 0; row < s.inside.rows; row++)
            for (int column = 0; column < s.inside.cols; column++)
            {
                if (!s.seen.at<uchar>(row, column)) continue;
                const cv::Point3f w = shell.firstCamera.t() * Centre(s, row, column);
                // Clear: at least ClearMeters from the couch and from every other wall, the floor and the ceiling.
                const float q[3] = { w.x, w.y, w.z };
                float box = 0, nearest = 1e9f;
                for (int a = 0; a < 3; a++)
                {
                    for (float plane : { Low[a], High[a] })
                        if (std::abs(q[a] - plane) > 0.02f) nearest = std::min(nearest, std::abs(q[a] - plane));
                    const float out = std::max({ CouchLow[a] - q[a], q[a] - CouchHigh[a], 0.0f });
                    box += out * out;
                }
                if (std::min(nearest, std::sqrt(box)) < ClearMeters) continue;
                bool match = false;
                for (int a = -2; a <= 2 && !match; a++)
                    for (int b = -2; b <= 2 && !match; b++)
                    {
                        cv::Vec3b truth;
                        const cv::Point3f nudge = shell.firstCamera.t() * (s.across * (a * 0.015f) + s.down * (b * 0.015f));
                        Raycast(w + nudge + in * 0.02f, -in, truth);
                        match = cv::norm(cv::Vec3i(truth) - cv::Vec3i(s.colour.at<cv::Vec3b>(row, column)), cv::NORM_INF) <= 2;
                    }
                seen++;
                right += match;
                const bool behindCouch = w.x > CouchLow[0] - 0.1f && w.x < CouchHigh[0] + 0.1f && w.z > CouchLow[2] && w.y > CouchLow[1] - 0.1f;
                behind += behindCouch;
                behindRight += behindCouch && match;
            }

        // 4. Fill: seen texels untouched; the pattern's spread in the filled texels against the seen ones (floor only).
        cv::Mat before = s.colour.clone();
        Fill(s, random);
        untouched = untouched && cv::norm(before, s.colour, cv::NORM_INF, s.seen) == 0;
        if (s.name != "floor") continue;
        cv::Scalar mean, deviation;
        cv::meanStdDev(s.colour, mean, deviation, s.seen & s.inside);
        seenSpread = (deviation[0] + deviation[1] + deviation[2]) / 3;
        cv::meanStdDev(s.colour, mean, deviation, s.filled);
        filledSpread = (deviation[0] + deviation[1] + deviation[2]) / 3;
        std::printf("4. floor: %d texels filled; colour spread %.0f filled vs %.0f seen, from a %d texel square\n", cv::countNonZero(s.filled),
                    filledSpread, seenSpread, s.exemplar);
    }
    std::printf("3. %d seen texels clear of the corners and the couch, %.2f%% their true colour; behind the couch %d, %.2f%%\n", seen,
                100.0 * right / std::max(seen, 1), behind, 100.0 * behindRight / std::max(behind, 1));
    Check(seen > 100000 && right > 0.99 * seen, "every seen texel has its surface's true colour (99%)");
    Check(behind > 1000 && behindRight > 0.99 * behind, "also behind the couch: the couch's colour never lands on the wall or floor behind it");
    Check(untouched, "Fill leaves the seen texels alone");
    Check(filledSpread > 0.5 * seenSpread, "the floor under the couch gets the floor's own pattern, not a smooth smear");

    std::printf(failures ? "check_room_shell: %d FAILED\n" : "check_room_shell: all passed\n", failures);
    return failures ? 1 : 0;
}
