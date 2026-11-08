// =====================================================================
//  09c · Room Model
//
//  Example 09b's mesh is every surface exactly as the camera saw it:
//  lumpy, full of holes, and tilted if the camera was. This one also
//  builds a CLEAN model next to it, the way you'd draw the room:
//
//    1-3.  Scan, average, points                as in 09b
//    4.    Planes     the biggest flat surfaces, facing any way (06b's
//                     repeated RANSAC: spotter::FindPlanes).
//    5.    Sort them  level and below = floor (the lowest one), level and
//                     high above = ceiling, upright and tall = wall.
//                     Everything else is furniture.
//    6.    Level it   the floor (or ceiling, or two walls) says which way
//                     is UP, whatever the camera's tilt. The biggest wall
//                     says which way is "ahead". The whole model is turned
//                     to stand straight.
//    7.    Walls      each one a flat rectangle from floor to ceiling;
//                     two walls that nearly meet are joined at the corner.
//    8.    Furniture  the left-over points, split where the depth jumps,
//                     each piece a box standing on the floor.
//    9.    Save       one .obj: the raw mesh (09b) and the clean model, in
//                     separate groups, each painted in its own average colour.
//
//  The raw mesh is what was SEEN. The clean model is partly GUESSED: walls
//  run behind the furniture, boxes go down to the floor, and the floor and
//  ceiling are grey where none was seen. Hide groups in Blender or MeshLab.
//
//  Window: colour (half size). After a scan, each sampled point is dotted
//          by what it became: wall blue, floor green, ceiling yellow,
//          furniture red, nothing (holes, slivers) undotted.
//  Keys:   S  scan     C  clear the dots     Q or Esc  quit
//  Privacy: an .obj is a 3D capture of the room: never commit one. Scan
//  with nobody in view.
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/geometry.hpp"
#include "spotter/mesh.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/stats.hpp"
#include "spotter/view.hpp"

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

const int AverageFrames = 30;
const int MeshStep = 2;                    // the raw mesh, as in 09b
const float MaxJumpShare = 0.03f;

const int SampleStep = 4;                  // planes are searched on every 4th pixel each way
const float InlierThicknessMeters = 0.03f;

// Level or upright within this many degrees, as the camera sees it. A camera tilted more than this
// needs a bigger number. Planes in between (a slanted roof) count as furniture.
const float MaxTiltDegrees = 25;

// A wall spans at least this height (06b: a couch back doesn't). A ceiling is at least this far
// above the lowest point seen (else it's a table top, seen from below the table's height).
const float MinWallHeightMeters = 0.8f;
const float MinCeilingHeightMeters = 1.8f;

const float CornerSnapMeters = 0.5f;       // two walls whose ends come this close to their crossing are joined there
const int MinBlockCells = 30;              // a furniture piece needs at least this many sampled points

const std::string Window = "09c room model  (S scan, C clear, Q quit)";

spotter::PlaneSearch PlaneSettings()
{
    spotter::PlaneSearch search;   // facing any way
    search.maxPlanes = 8;          // a room view rarely has more big flat surfaces
    search.thicknessMeters = InlierThicknessMeters;
    search.neighbourCells = 40 / SampleStep;
    search.minShare = 0.03;
    return search;
}

// ---------------------------------------------------------------------
//  The new idea: from planes to a tidy model of the room
// ---------------------------------------------------------------------

// What the model builder found, for the console and the window.
struct Room
{
    spotter::Mesh mesh;            // room coordinates: X right, Y up, Z towards the camera; camera at 0,0,0
    std::string upFrom;            // what said which way is up
    float floorY = 0, ceilingY = 0;
    bool floorSeen = false, ceilingSeen = false;
    float tiltDegrees = 0, rollDegrees = 0;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> walls;   // each wall's ends on the floor plan (X, Z)
    std::vector<std::pair<cv::Point3f, cv::Point3f>> blocks;  // each box's lowest and highest corner
    std::vector<char> cellKind;    // per grid cell: 'w' wall, 'f' floor, 'c' ceiling, 'b' furniture, 0 nothing
};

// A wall seen from above: towards . (x, z) = offset.
struct Line
{
    cv::Point2f towards, ends[2];
    float offset;
    cv::Vec3b colour;
};

// Step 5: floor, ceiling and walls among the planes, judged by the camera's own up.
void SortPlanes(const std::vector<spotter::PlanePiece>& pieces, const spotter::PointGrid& grid, int& floorPiece, int& ceilingPiece, std::vector<int>& wallPieces)
{
    const cv::Point3f cameraUp(0, -1, 0);   // the camera's Y axis points DOWN
    const float levelCos = std::cos(MaxTiltDegrees * (float)CV_PI / 180), uprightSin = std::sin(MaxTiltDegrees * (float)CV_PI / 180);
    std::vector<float> heights;
    for (const cv::Point3f& p : grid.points) if (!std::isnan(p.z)) heights.push_back(p.dot(cameraUp));
    float lowest = spotter::Percentile(heights, 0.02);
    floorPiece = ceilingPiece = -1;
    for (int k = 0; k < (int)pieces.size(); k++)
    {
        const spotter::Plane& plane = pieces[k].plane;
        float upness = plane.normal.dot(cameraUp);   // the normal points at the camera: + = the plane is below it
        if (upness > levelCos)
        {
            if (floorPiece < 0 || plane.distance > pieces[floorPiece].plane.distance) floorPiece = k;   // the lowest
        }
        else if (-upness > levelCos)
        {
            bool highEnough = plane.distance - lowest >= MinCeilingHeightMeters;
            if (highEnough && (ceilingPiece < 0 || plane.distance > pieces[ceilingPiece].plane.distance)) ceilingPiece = k;
        }
        else if (std::abs(upness) < uprightSin)
        {
            std::vector<float> wallHeights;
            for (int i : pieces[k].cells) wallHeights.push_back(grid.points[i].dot(cameraUp));
            if (spotter::Percentile(wallHeights, 0.98) - spotter::Percentile(wallHeights, 0.02) >= MinWallHeightMeters) wallPieces.push_back(k);
        }
        // Anything else (a table top, a low or slanted piece) stays for the furniture.
    }
    std::sort(wallPieces.begin(), wallPieces.end(), [&](int a, int b) { return pieces[a].cells.size() > pieces[b].cells.size(); });
}

// Step 6: which way is up: the floor, else the ceiling, else two walls that aren't parallel.
cv::Point3f FindUp(const std::vector<spotter::PlanePiece>& pieces, int floorPiece, int ceilingPiece, const std::vector<int>& wallPieces, std::string& from)
{
    const cv::Point3f cameraUp(0, -1, 0);
    from = "the camera (nothing level found, under 2 walls)";
    if (floorPiece >= 0) { from = "the floor"; return pieces[floorPiece].plane.normal; }
    if (ceilingPiece >= 0) { from = "the ceiling"; return -pieces[ceilingPiece].plane.normal; }
    cv::Point3f up = cameraUp;
    float bestAcross = 0.5f;   // at least 30 degrees apart
    for (size_t i = 0; i < wallPieces.size(); i++)
        for (size_t j = i + 1; j < wallPieces.size(); j++)
        {
            cv::Point3f across = pieces[wallPieces[i]].plane.normal.cross(pieces[wallPieces[j]].plane.normal);
            if (cv::norm(across) <= bestAcross) continue;
            bestAcross = (float)cv::norm(across);
            up = across / bestAcross;
            if (up.dot(cameraUp) < 0) up = -up;
            from = "two walls";
        }
    return up;
}

// Step 7's corners: where two wall lines cross, if each has an end near the crossing, move both there.
void JoinCorners(std::vector<Line>& lines)
{
    for (size_t i = 0; i < lines.size(); i++)
        for (size_t j = i + 1; j < lines.size(); j++)
        {
            cv::Point2f a = lines[i].towards, b = lines[j].towards;
            float determinant = a.x * b.y - a.y * b.x;   // sin of the angle between them
            if (std::abs(determinant) < 0.5f) continue;   // nearly parallel: no corner
            cv::Point2f crossing((lines[i].offset * b.y - lines[j].offset * a.y) / determinant, (a.x * lines[j].offset - b.x * lines[i].offset) / determinant);
            auto NearestEnd = [&](Line& line) -> cv::Point2f& { return cv::norm(line.ends[0] - crossing) < cv::norm(line.ends[1] - crossing) ? line.ends[0] : line.ends[1]; };
            cv::Point2f& endI = NearestEnd(lines[i]);
            cv::Point2f& endJ = NearestEnd(lines[j]);
            if (cv::norm(endI - crossing) < CornerSnapMeters && cv::norm(endJ - crossing) < CornerSnapMeters) endI = endJ = crossing;
        }
}

// Steps 4-8: the clean model, plus 09b's raw mesh, all turned to stand straight.
Room BuildRoom(const std::vector<cv::Point3f>& cloud, const std::vector<cv::Vec3b>& colours, std::mt19937& random)
{
    Room room;
    spotter::PointGrid grid = spotter::ThinCloud(cloud, 512, SampleStep);
    std::vector<cv::Vec3b> gridColours(grid.points.size());
    for (size_t i = 0; i < grid.points.size(); i++) gridColours[i] = colours[(i / grid.columns) * SampleStep * 512 + (i % grid.columns) * SampleStep];
    auto AverageColour = [&](const std::vector<int>& cells)
    {
        cv::Vec3d sum(0, 0, 0);
        for (int i : cells) sum += cv::Vec3d(gridColours[i]);
        return cv::Vec3b(sum / (double)std::max<size_t>(1, cells.size()));
    };
    room.cellKind.assign(grid.points.size(), 0);
    if (std::none_of(grid.points.begin(), grid.points.end(), [](const cv::Point3f& p) { return !std::isnan(p.z); })) return room;

    // Steps 4-6: the planes, sorted; up; and "towards the camera" from the biggest wall.
    std::vector<spotter::PlanePiece> pieces = spotter::FindPlanes(grid, PlaneSettings(), random);
    int floorPiece, ceilingPiece;
    std::vector<int> wallPieces;
    SortPlanes(pieces, grid, floorPiece, ceilingPiece, wallPieces);
    cv::Point3f up = FindUp(pieces, floorPiece, ceilingPiece, wallPieces, room.upFrom);
    auto Flatten = [&](cv::Point3f v) { v -= up * up.dot(v); return v / (float)cv::norm(v); };   // drop the up/down part
    cv::Point3f towards = wallPieces.empty() ? Flatten(cv::Point3f(0, 0, -1)) : Flatten(pieces[wallPieces[0]].plane.normal);
    cv::Point3f right = up.cross(towards);
    auto ToRoom = [&](const cv::Point3f& p) { return cv::Point3f(right.dot(p), up.dot(p), towards.dot(p)); };
    room.tiltDegrees = std::asin(up.z) * 180 / (float)CV_PI;   // + = the camera looks up
    room.rollDegrees = std::asin(up.x) * 180 / (float)CV_PI;

    // The raw mesh (09b), turned to stand straight.
    spotter::MeshFromGrid(cloud, colours, 512, MeshStep, MaxJumpShare, room.mesh, "seen_room");
    for (cv::Point3f& p : room.mesh.points) p = ToRoom(p);

    // Floor and ceiling heights: the planes', or else the lowest and highest points seen.
    std::vector<cv::Point3f> roomGrid(grid.points.size());
    std::vector<float> roomHeights, roomX, roomZ;
    for (size_t i = 0; i < grid.points.size(); i++)
    {
        roomGrid[i] = ToRoom(grid.points[i]);
        if (std::isnan(grid.points[i].z)) continue;
        roomHeights.push_back(roomGrid[i].y);
        roomX.push_back(roomGrid[i].x);
        roomZ.push_back(roomGrid[i].z);
    }
    room.floorSeen = floorPiece >= 0;
    room.ceilingSeen = ceilingPiece >= 0;
    room.floorY = room.floorSeen ? -pieces[floorPiece].plane.distance : spotter::Percentile(roomHeights, 0.02);
    room.ceilingY = room.ceilingSeen ? pieces[ceilingPiece].plane.distance : spotter::Percentile(roomHeights, 0.98);
    if (room.floorSeen) for (int i : pieces[floorPiece].cells) room.cellKind[i] = 'f';
    if (room.ceilingSeen) for (int i : pieces[ceilingPiece].cells) room.cellKind[i] = 'c';

    // Step 7: walls as lines on the floor plan, joined at corners, then stood up.
    std::vector<Line> lines;
    for (int k : wallPieces)
    {
        cv::Point3f n = ToRoom(Flatten(pieces[k].plane.normal));
        Line line;
        line.towards = cv::Point2f(n.x, n.z) / std::hypot(n.x, n.z);
        cv::Point2f along(-line.towards.y, line.towards.x);
        std::vector<float> offsets, alongs;
        for (int i : pieces[k].cells)
        {
            cv::Point2f q(roomGrid[i].x, roomGrid[i].z);
            offsets.push_back(line.towards.dot(q));
            alongs.push_back(along.dot(q));
            room.cellKind[i] = 'w';
        }
        line.offset = spotter::Percentile(offsets, 0.5);
        line.ends[0] = line.towards * line.offset + along * spotter::Percentile(alongs, 0.02);
        line.ends[1] = line.towards * line.offset + along * spotter::Percentile(alongs, 0.98);
        line.colour = AverageColour(pieces[k].cells);
        lines.push_back(line);
    }
    JoinCorners(lines);
    for (const Line& line : lines)
    {
        cv::Point2f e0 = line.ends[0], e1 = line.ends[1];
        spotter::AddQuad(room.mesh, "clean_walls", { e0.x, room.floorY, e0.y }, { e1.x, room.floorY, e1.y }, { e1.x, room.ceilingY, e1.y },
                         { e0.x, room.ceilingY, e0.y }, line.colour, { line.towards.x, 0, line.towards.y });
        room.walls.emplace_back(e0, e1);
    }

    // Floor and ceiling: rectangles reaching the walls, most of the points, and the camera.
    // ponytail: a rectangle, not the room's true outline. Grey = not seen, a guess.
    float minX = std::min(0.0f, spotter::Percentile(roomX, 0.02)), maxX = std::max(0.0f, spotter::Percentile(roomX, 0.98));
    float minZ = std::min(0.0f, spotter::Percentile(roomZ, 0.02)), maxZ = std::max(0.0f, spotter::Percentile(roomZ, 0.98));
    for (const auto& wall : room.walls)
        for (const cv::Point2f& e : { wall.first, wall.second })
        {
            minX = std::min(minX, e.x); maxX = std::max(maxX, e.x);
            minZ = std::min(minZ, e.y); maxZ = std::max(maxZ, e.y);
        }
    for (bool isFloor : { true, false })
    {
        float y = isFloor ? room.floorY : room.ceilingY;
        int k = isFloor ? floorPiece : ceilingPiece;
        spotter::AddQuad(room.mesh, isFloor ? "clean_floor" : "clean_ceiling", { minX, y, minZ }, { maxX, y, minZ }, { maxX, y, maxZ }, { minX, y, maxZ },
                         k >= 0 ? AverageColour(pieces[k].cells) : cv::Vec3b(128, 128, 128), { 0, isFloor ? 1.0f : -1.0f, 0 });
    }

    // Step 8: furniture, from what's left between floor and ceiling and off every wall's plane.
    const float margin = 0.05f;
    cv::Mat left(grid.rows, grid.columns, CV_8U, cv::Scalar(0));
    for (size_t i = 0; i < grid.points.size(); i++)
    {
        if (std::isnan(grid.points[i].z) || room.cellKind[i] != 0 || roomGrid[i].y < room.floorY + margin || roomGrid[i].y > room.ceilingY - margin) continue;
        bool onWall = false;   // a strip cut off a wall (by a door frame, say) is still wall
        for (const Line& line : lines) onWall = onWall || std::abs(line.towards.dot(cv::Point2f(roomGrid[i].x, roomGrid[i].z)) - line.offset) < InlierThicknessMeters;
        if (!onWall) left.data[i] = 255;
    }
    // Split pieces where the depth jumps (a chair in front of a couch): clear the nearer side's cell.
    for (int row = 0; row + 1 < grid.rows; row++)
        for (int column = 0; column + 1 < grid.columns; column++)
        {
            int i = row * grid.columns + column;
            for (int neighbour : { i + 1, i + grid.columns })
            {
                if (!left.data[i] || std::isnan(grid.points[neighbour].z)) continue;
                float z = grid.points[i].z, zn = grid.points[neighbour].z;
                if (std::abs(z - zn) > MaxJumpShare * std::min(z, zn)) left.data[z < zn ? i : neighbour] = 0;
            }
        }
    cv::Mat labels, stats, centroids;
    int count = cv::connectedComponentsWithStats(left, labels, stats, centroids);
    std::vector<std::vector<int>> blobCells(count);
    for (size_t i = 0; i < grid.points.size(); i++) if (labels.ptr<int>()[i] > 0) blobCells[labels.ptr<int>()[i]].push_back((int)i);
    for (int label = 1; label < count; label++)
    {
        if ((int)blobCells[label].size() < MinBlockCells) continue;
        std::vector<float> xs, ys, zs;
        for (int i : blobCells[label])
        {
            xs.push_back(roomGrid[i].x); ys.push_back(roomGrid[i].y); zs.push_back(roomGrid[i].z);
            room.cellKind[i] = 'b';
        }
        // ponytail: boxes line up with the biggest wall and stand on the floor ("most furniture
        // does"); a picture on the wall becomes a floor-to-picture box. Only the front was seen.
        cv::Point3f low(spotter::Percentile(xs, 0.02), room.floorY, spotter::Percentile(zs, 0.02));
        cv::Point3f high(spotter::Percentile(xs, 0.98), spotter::Percentile(ys, 0.98), spotter::Percentile(zs, 0.98));
        spotter::AddBox(room.mesh, "furniture_blocks", low, high, AverageColour(blobCells[label]));
        room.blocks.emplace_back(low, high);
    }
    return room;
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

void Report(const Room& room, const std::string& path)
{
    std::printf("Saved %s (%.1f MB). A capture of the room: never commit it.\n", path.c_str(), std::filesystem::file_size(path) / 1e6);
    std::printf("  up from %s: camera tilted %+.1f deg (+ = looking up), rolled %+.1f deg\n", room.upFrom.c_str(), room.tiltDegrees, room.rollDegrees);
    std::printf("  floor %.2f m below the camera (%s), ceiling %.2f m above it (%s): %.2f m high\n", -room.floorY,
                room.floorSeen ? "seen" : "guessed: lowest point", room.ceilingY, room.ceilingSeen ? "seen" : "guessed: highest point", room.ceilingY - room.floorY);
    for (const auto& [a, b] : room.walls)
        std::printf("  wall  %.2f m long, from (%+.2f, %.2f) to (%+.2f, %.2f) m (X right, Z ahead)\n", cv::norm(b - a), a.x, -a.y, b.x, -b.y);
    for (const auto& [low, high] : room.blocks)
        std::printf("  block %.2f wide x %.2f deep x %.2f high, at X %+.2f, Z %.2f m\n", high.x - low.x, high.z - low.z, high.y - low.y,
                    (low.x + high.x) / 2, -(low.z + high.z) / 2);
}

// Each sampled point's colour pixel, dotted by what it became.
std::vector<std::pair<cv::Point, cv::Scalar>> Dots(const Room& room, const spotter::RgbdFrame& frame)
{
    std::vector<std::pair<cv::Point, cv::Scalar>> dots;
    const int gridColumns = 512 / SampleStep;
    for (size_t cell = 0; cell < room.cellKind.size(); cell++)
    {
        int pixel = (int)(cell / gridColumns) * SampleStep * 512 + (int)(cell % gridColumns) * SampleStep, c = frame.colourIndex[pixel];
        char kind = room.cellKind[cell];
        if (!kind || c < 0) continue;
        cv::Scalar colour = kind == 'w' ? cv::Scalar(255, 80, 0) : kind == 'f' ? cv::Scalar(0, 200, 0) : kind == 'c' ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255);
        dots.emplace_back(cv::Point(c % frame.colour.cols, c / frame.colour.cols), colour);
    }
    return dots;
}

std::vector<std::pair<cv::Point, cv::Scalar>> ScanAndSave(const cv::Mat& averageMm, const spotter::RgbdFrame& frame, std::mt19937& random)
{
    std::vector<cv::Vec3b> colours = frame.registered.reshape(0, 1);
    Room room = BuildRoom(frame.depthCamera.Cloud(averageMm), colours, random);
    if (room.mesh.points.empty()) { std::printf("Nothing to model: no depth readings.\n"); return {}; }
    std::string path = (std::filesystem::current_path() / cv::format("room_model_%lld.obj", (long long)std::time(nullptr))).string();
    if (spotter::SaveObj(path, room.mesh, "Spotter 09c: seen_room = scanned, clean_* and furniture_blocks = the tidy (partly guessed) model")) Report(room, path);
    else std::printf("Could not write %s\n", path.c_str());
    return Dots(room, frame);
}

void Show(const cv::Mat& colour, const std::vector<std::pair<cv::Point, cv::Scalar>>& dots, const spotter::DepthAverager& averager, const spotter::FpsMeter& fps)
{
    cv::Mat display;
    cv::resize(colour, display, colour.size() / 2);
    for (const auto& [p, c] : dots) cv::circle(display, p / 2, 2, c, cv::FILLED);
    std::string status = averager.Running() ? cv::format("HOLD STILL  %d / %d", averager.Added(), averager.Frames()) : "S to scan the room";
    cv::putText(display, status, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    fps.Draw(display);
    cv::imshow(Window, display);
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!kinect.IsOpen()) return 1;

    cv::namedWindow(Window, cv::WINDOW_AUTOSIZE);
    spotter::DepthAverager averager(AverageFrames);
    std::vector<std::pair<cv::Point, cv::Scalar>> dots;
    std::mt19937 random(1);
    spotter::FpsMeter fps;
    spotter::RgbdFrame frame;

    while (kinect.Next(frame))
    {
        if (averager.Running() && averager.Add(frame.depthMm)) dots = ScanAndSave(averager.Mean(), frame, random);   // steps 2-9
        fps.Tick();
        Show(frame.colour, dots, averager, fps);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 's' && !averager.Running()) { averager.Start(); std::printf("Scanning: hold still...\n"); }
        if (key == 'c') dots.clear();
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Open the .obj and hide seen_room: the clean model alone. Hide the clean groups: the raw scan.
//  2. Tilt the camera 10 degrees down and scan: the console's tilt says so, and the model still
//     stands straight (it's levelled by the floor, not the camera).
//  3. Scan with no floor in view: "up from the ceiling" or "two walls". With neither: the camera.
//  4. A table: is it a furniture box, or did its top become the floor? (Which plane is lowest?)
//  5. CornerSnapMeters = 0: walls stop short of each other in the corners.
//  6. A picture on the wall becomes a box from the floor up to it. Why? (Step 8's ponytail.)
