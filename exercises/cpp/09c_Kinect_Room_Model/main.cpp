// =====================================================================
//  09c · Kinect Room Model
//
//  Example 09b's mesh is every surface exactly as the camera saw it:
//  lumpy, full of holes, and tilted if the camera was. This one also
//  builds a CLEAN model next to it, the way you'd draw the room:
//
//    1-3.  Scan, average, points                as in example 09b
//    4.    Planes     the biggest flat surfaces, facing any way (06b's
//                     repeated RANSAC, without the "upright" rule).
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
//  The raw mesh is what was SEEN. The clean model is partly GUESSED:
//  walls run behind the furniture, boxes go down to the floor, and the
//  floor/ceiling are grey where none was seen. Hide groups in Blender or
//  MeshLab to compare.
//
//  Window: colour (half size). After a scan, each sampled point is dotted
//          by what it became: wall blue, floor green, ceiling yellow,
//          furniture red, nothing (holes, slivers) undotted.
//  Keys:   S  scan     C  clear the dots     Q or Esc  quit
//
//  Privacy: an .obj is a 3D capture of the room, in colour. *.obj is
//  gitignored: never commit one. Scan with nobody in view.
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/registration.h>
#include <opencv2/opencv.hpp>

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// --- Scan and raw mesh (as in 09b) ---
const int AverageFrames = 30;
const float MaxDepthMeters = 6.0f;
const int MeshStep = 2;
const float MaxJumpShare = 0.03f;
const bool CapColorExposure = false;

// --- Planes (new): 06b's search, for planes facing any way ---
const int MaxPlanes = 8;                   // a room view rarely has more big flat surfaces
const int RansacTries = 200;
const float InlierThicknessMeters = 0.03f;
const int SampleStep = 4;                  // every 4th depth pixel each way (1/16 of them)
const int NeighbourPixels = 40;
const double MinPlaneShare = 0.03;         // a plane needs at least this share of the points

// Level or upright within this many degrees, as the camera sees it (06b, 09). A camera tilted
// more than this needs a bigger number. Planes in between (a slanted roof) count as furniture.
const float MaxTiltDegrees = 25;

// A wall spans at least this height (06b: a couch back doesn't). A ceiling is at least this far
// above the lowest point seen (else it's a table top, seen from below the table's height).
const float MinWallHeightMeters = 0.8f;
const float MinCeilingHeightMeters = 1.8f;

// --- Clean model (new) ---
// Two walls whose ends come this close to where their lines cross are joined at that corner.
const float CornerSnapMeters = 0.5f;

// A furniture piece needs at least this many sampled points (each stands for 4x4 pixels).
const int MinBlockCells = 30;

const int Width = 512, Height = 424, ColorWidth = 1920, ColorHeight = 1080;

// A triangle mesh, its triangles sorted into named groups (OBJ "g" lines).
struct Mesh
{
    std::vector<cv::Vec3f> points;
    std::vector<cv::Vec3b> colours;                           // blue, green, red
    std::map<std::string, std::vector<cv::Vec3i>> groups;     // group name -> three point indices each
};

// As in 06b: normal . point + distance = 0, the normal pointing towards the camera.
struct Plane
{
    cv::Vec3f normal;
    float distance;
    float DistanceTo(const cv::Vec3f& point) const { return normal.dot(point) + distance; }
};

// A plane and the grid cells of its biggest connected piece.
struct Piece
{
    Plane plane;
    std::vector<int> cells;
};

// What the model builder found, for the console and the window.
struct Room
{
    Mesh mesh;                     // room coordinates: X right, Y up, Z towards the camera; camera at 0,0,0
    std::string upFrom;            // what said which way is up
    float floorY, ceilingY;        // heights of the floor and ceiling (camera = 0)
    bool floorSeen, ceilingSeen;
    float tiltDegrees, rollDegrees;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> walls;   // each wall's ends on the floor plan (X, Z)
    std::vector<std::pair<cv::Vec3f, cv::Vec3f>> blocks;      // each box's lowest and highest corner
    std::vector<char> cellKind;    // per grid cell: 'w' wall, 'f' floor, 'c' ceiling, 'b' furniture, 0 nothing
};

// As in 09b.
void MeshFromCloud(const std::vector<cv::Vec3f>& cloud, const std::vector<cv::Vec3b>& colours, Mesh& mesh)
{
    const int columns = (Width + MeshStep - 1) / MeshStep, rows = (Height + MeshStep - 1) / MeshStep;
    std::vector<int> pointOf(columns * rows, -1);
    std::vector<float> depthOf(columns * rows);
    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            int pixel = row * MeshStep * Width + column * MeshStep;
            if (std::isnan(cloud[pixel][2])) continue;
            pointOf[row * columns + column] = (int)mesh.points.size();
            depthOf[row * columns + column] = cloud[pixel][2];
            mesh.points.push_back(cloud[pixel]);
            mesh.colours.push_back(colours[pixel]);
        }
    }
    std::vector<cv::Vec3i>& faces = mesh.groups["seen_room"];
    auto addTriangle = [&](int a, int b, int c)
    {
        if (pointOf[a] < 0 || pointOf[b] < 0 || pointOf[c] < 0) return;
        float nearest = std::min({ depthOf[a], depthOf[b], depthOf[c] });
        float farthest = std::max({ depthOf[a], depthOf[b], depthOf[c] });
        if (farthest - nearest > MaxJumpShare * nearest) return;
        faces.emplace_back(pointOf[a], pointOf[b], pointOf[c]);
    };
    for (int row = 0; row + 1 < rows; row++)
    {
        for (int column = 0; column + 1 < columns; column++)
        {
            int topLeft = row * columns + column, topRight = topLeft + 1;
            int bottomLeft = topLeft + columns, bottomRight = bottomLeft + 1;
            addTriangle(topLeft, bottomLeft, topRight);
            addTriangle(topRight, bottomLeft, bottomRight);
        }
    }
}

// As in 06b.
bool PlaneThrough(const cv::Vec3f& a, const cv::Vec3f& b, const cv::Vec3f& c, Plane& plane)
{
    cv::Vec3f normal = (b - a).cross(c - a);
    float length = (float)cv::norm(normal);
    if (length < 1e-6f) return false;
    plane.normal = normal / length;
    plane.distance = -plane.normal.dot(a);
    if (plane.distance < 0) { plane.normal = -plane.normal; plane.distance = -plane.distance; }
    return true;
}

// As in 06b.
Plane FitPlane(const std::vector<cv::Vec3f>& points)
{
    cv::Mat data((int)points.size(), 3, CV_32F, (void*)points.data());
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Vec3f mean(pca.mean.at<float>(0), pca.mean.at<float>(1), pca.mean.at<float>(2));
    Plane plane{ cv::Vec3f(pca.eigenvectors.row(2)), 0 };
    plane.distance = -plane.normal.dot(mean);
    if (plane.distance < 0) { plane.normal = -plane.normal; plane.distance = -plane.distance; }
    return plane;
}

// As in 06b.
float Percentile(std::vector<float> values, double share)
{
    size_t i = std::min(values.size() - 1, (size_t)(share * values.size()));
    std::nth_element(values.begin(), values.begin() + i, values.end());
    return values[i];
}

// Step 4: 06b's FindWalls without the upright rule and the height rule: the biggest planes,
// each as its biggest connected piece.
std::vector<Piece> FindPlanes(const std::vector<cv::Vec3f>& grid, int gridColumns, std::mt19937& random)
{
    const int gridRows = (int)grid.size() / gridColumns, reach = NeighbourPixels / SampleStep;
    std::uniform_int_distribution<int> pickOffset(-reach, reach);
    std::vector<bool> free(grid.size());
    int readings = 0;
    for (size_t i = 0; i < grid.size(); i++)
    {
        free[i] = !std::isnan(grid[i][2]);
        if (free[i]) readings++;
    }

    std::vector<Piece> pieces;
    std::vector<int> freeCells;
    for (int search = 0; search < MaxPlanes; search++)
    {
        freeCells.clear();
        for (int i = 0; i < (int)grid.size(); i++) if (free[i]) freeCells.push_back(i);
        if (freeCells.size() < 3) break;

        std::uniform_int_distribution<size_t> pickCell(0, freeCells.size() - 1);
        Plane best{};
        int bestCount = 0;
        for (int t = 0; t < RansacTries; t++)
        {
            int a = freeCells[pickCell(random)];
            int row = a / gridColumns, column = a % gridColumns;
            int bRow = row + pickOffset(random), bColumn = column + pickOffset(random);
            int cRow = row + pickOffset(random), cColumn = column + pickOffset(random);
            if (std::min({ bRow, bColumn, cRow, cColumn }) < 0 || std::max(bRow, cRow) >= gridRows ||
                std::max(bColumn, cColumn) >= gridColumns) continue;
            int b = bRow * gridColumns + bColumn, c = cRow * gridColumns + cColumn;
            if (!free[b] || !free[c]) continue;
            Plane plane;
            if (!PlaneThrough(grid[a], grid[b], grid[c], plane)) continue;
            int count = 0;
            for (int i : freeCells)
                if (std::abs(plane.DistanceTo(grid[i])) < InlierThicknessMeters) count++;
            if (count > bestCount) { bestCount = count; best = plane; }
        }
        if (bestCount < MinPlaneShare * readings) break;

        cv::Mat onPlane(gridRows, gridColumns, CV_8U, cv::Scalar(0)), labels, stats, centroids;
        for (int i : freeCells)
            if (std::abs(best.DistanceTo(grid[i])) < InlierThicknessMeters) onPlane.data[i] = 255;
        int count = cv::connectedComponentsWithStats(onPlane, labels, stats, centroids);
        int biggest = 1;
        for (int piece = 2; piece < count; piece++)
            if (stats.at<int>(piece, cv::CC_STAT_AREA) > stats.at<int>(biggest, cv::CC_STAT_AREA)) biggest = piece;
        if (count < 2 || stats.at<int>(biggest, cv::CC_STAT_AREA) < MinPlaneShare * readings) continue;
        Piece piece;
        std::vector<cv::Vec3f> inliers;
        for (int i : freeCells)
        {
            if (labels.ptr<int>()[i] != biggest) continue;
            piece.cells.push_back(i);
            inliers.push_back(grid[i]);
            free[i] = false;
        }
        piece.plane = FitPlane(inliers);
        pieces.push_back(piece);
    }
    return pieces;
}

// Two triangles for the corners a b c d (in order round the edge), turned to face "facing".
void AddQuad(Mesh& mesh, const std::string& group, const cv::Vec3f& a, const cv::Vec3f& b, const cv::Vec3f& c,
             const cv::Vec3f& d, const cv::Vec3b& colour, const cv::Vec3f& facing)
{
    int first = (int)mesh.points.size();
    for (const cv::Vec3f& p : { a, b, c, d })
    {
        mesh.points.push_back(p);
        mesh.colours.push_back(colour);
    }
    bool flip = (b - a).cross(c - a).dot(facing) < 0;
    std::vector<cv::Vec3i>& faces = mesh.groups[group];
    faces.push_back(flip ? cv::Vec3i(first, first + 2, first + 1) : cv::Vec3i(first, first + 1, first + 2));
    faces.push_back(flip ? cv::Vec3i(first, first + 3, first + 2) : cv::Vec3i(first, first + 2, first + 3));
}

// A box from its lowest to its highest corner, its six sides facing out.
void AddBox(Mesh& mesh, const std::string& group, const cv::Vec3f& lo, const cv::Vec3f& hi, const cv::Vec3b& colour)
{
    cv::Vec3f corner[8];
    for (int i = 0; i < 8; i++) corner[i] = cv::Vec3f(i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]);
    const int sides[6][4] = { { 0, 2, 6, 4 }, { 1, 3, 7, 5 }, { 0, 1, 5, 4 }, { 2, 3, 7, 6 }, { 0, 1, 3, 2 }, { 4, 5, 7, 6 } };
    cv::Vec3f centre = (lo + hi) / 2;
    for (const auto& s : sides)
    {
        cv::Vec3f middle = (corner[s[0]] + corner[s[1]] + corner[s[2]] + corner[s[3]]) / 4;
        AddQuad(mesh, group, corner[s[0]], corner[s[1]], corner[s[2]], corner[s[3]], colour, middle - centre);
    }
}

// Steps 4-8: the clean model, plus 09b's raw mesh, all turned to stand straight.
Room BuildRoom(const std::vector<cv::Vec3f>& cloud, const std::vector<cv::Vec3b>& colours, std::mt19937& random)
{
    Room room;
    const int gridColumns = Width / SampleStep, gridRows = Height / SampleStep;
    std::vector<cv::Vec3f> grid(gridColumns * gridRows);
    std::vector<cv::Vec3b> gridColours(grid.size());
    for (int row = 0; row < gridRows; row++)
        for (int column = 0; column < gridColumns; column++)
        {
            grid[row * gridColumns + column] = cloud[row * SampleStep * Width + column * SampleStep];
            gridColours[row * gridColumns + column] = colours[row * SampleStep * Width + column * SampleStep];
        }
    auto averageColour = [&](const std::vector<int>& cells)
    {
        cv::Vec3d sum(0, 0, 0);
        for (int i : cells) sum += cv::Vec3d(gridColours[i]);
        return cv::Vec3b(sum / (double)std::max<size_t>(1, cells.size()));
    };

    // --- Step 4: the planes ---
    std::vector<Piece> pieces = FindPlanes(grid, gridColumns, random);

    // --- Step 5: sort them, first by the camera's own up (as in 06b and 09) ---
    const cv::Vec3f cameraUp(0, -1, 0);   // the camera's Y axis points DOWN
    const float levelCos = std::cos(MaxTiltDegrees * (float)CV_PI / 180), uprightSin = std::sin(MaxTiltDegrees * (float)CV_PI / 180);
    std::vector<float> heights;           // every point's height (camera = 0), for the ceiling rule
    for (const cv::Vec3f& p : grid) if (!std::isnan(p[2])) heights.push_back(p.dot(cameraUp));
    if (heights.empty()) return room;
    float lowest = Percentile(heights, 0.02);
    int floorPiece = -1, ceilingPiece = -1;
    std::vector<int> wallPieces;
    for (int k = 0; k < (int)pieces.size(); k++)
    {
        const Plane& plane = pieces[k].plane;
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
            for (int i : pieces[k].cells) wallHeights.push_back(grid[i].dot(cameraUp));
            if (Percentile(wallHeights, 0.98) - Percentile(wallHeights, 0.02) >= MinWallHeightMeters) wallPieces.push_back(k);
        }
        // Anything else (a table top, a low or slanted piece) stays for the furniture.
    }

    // --- Step 6: which way is up, and which way is ahead ---
    cv::Vec3f up = cameraUp;
    room.upFrom = "the camera (nothing level found, under 2 walls)";
    if (floorPiece >= 0) { up = pieces[floorPiece].plane.normal; room.upFrom = "the floor"; }
    else if (ceilingPiece >= 0) { up = -pieces[ceilingPiece].plane.normal; room.upFrom = "the ceiling"; }
    else
    {
        // Two walls that aren't parallel: up is at right angles to both.
        float bestAcross = 0.5f;   // at least 30 degrees apart
        for (size_t i = 0; i < wallPieces.size(); i++)
            for (size_t j = i + 1; j < wallPieces.size(); j++)
            {
                cv::Vec3f across = pieces[wallPieces[i]].plane.normal.cross(pieces[wallPieces[j]].plane.normal);
                if (cv::norm(across) <= bestAcross) continue;
                bestAcross = (float)cv::norm(across);
                up = across / bestAcross;
                if (up.dot(cameraUp) < 0) up = -up;
                room.upFrom = "two walls";
            }
    }
    std::sort(wallPieces.begin(), wallPieces.end(), [&](int a, int b) { return pieces[a].cells.size() > pieces[b].cells.size(); });
    auto Flatten = [&](cv::Vec3f v) { v -= up * up.dot(v); return v / (float)cv::norm(v); };   // drop the up/down part
    // "Towards the camera" from the biggest wall, else straight back along the camera's view.
    cv::Vec3f towards = wallPieces.empty() ? Flatten(cv::Vec3f(0, 0, -1)) : Flatten(pieces[wallPieces[0]].plane.normal);
    cv::Vec3f right = up.cross(towards);
    auto ToRoom = [&](const cv::Vec3f& p) { return cv::Vec3f(right.dot(p), up.dot(p), towards.dot(p)); };
    room.tiltDegrees = std::asin(up[2]) * 180 / (float)CV_PI;    // + = the camera looks up
    room.rollDegrees = std::asin(up[0]) * 180 / (float)CV_PI;

    // The raw mesh (09b), turned to stand straight.
    MeshFromCloud(cloud, colours, room.mesh);
    for (cv::Vec3f& p : room.mesh.points) p = ToRoom(p);

    // Floor and ceiling heights: the planes', or else the lowest and highest points seen.
    std::vector<cv::Vec3f> roomGrid(grid.size());
    std::vector<float> roomHeights, roomX, roomZ;
    for (size_t i = 0; i < grid.size(); i++)
    {
        roomGrid[i] = ToRoom(grid[i]);
        if (std::isnan(grid[i][2])) continue;
        roomHeights.push_back(roomGrid[i][1]);
        roomX.push_back(roomGrid[i][0]);
        roomZ.push_back(roomGrid[i][2]);
    }
    room.floorSeen = floorPiece >= 0;
    room.ceilingSeen = ceilingPiece >= 0;
    room.floorY = room.floorSeen ? -pieces[floorPiece].plane.distance : Percentile(roomHeights, 0.02);
    room.ceilingY = room.ceilingSeen ? pieces[ceilingPiece].plane.distance : Percentile(roomHeights, 0.98);
    room.cellKind.assign(grid.size(), 0);
    if (room.floorSeen) for (int i : pieces[floorPiece].cells) room.cellKind[i] = 'f';
    if (room.ceilingSeen) for (int i : pieces[ceilingPiece].cells) room.cellKind[i] = 'c';

    // --- Step 7: walls as lines on the floor plan, joined at corners, then stood up ---
    struct Line { cv::Point2f towards, ends[2]; float offset; cv::Vec3b colour; };   // towards . (x, z) = offset
    std::vector<Line> lines;
    for (int k : wallPieces)
    {
        cv::Vec3f n = ToRoom(Flatten(pieces[k].plane.normal));
        Line line;
        line.towards = cv::Point2f(n[0], n[2]) / std::hypot(n[0], n[2]);
        cv::Point2f along(-line.towards.y, line.towards.x);
        std::vector<float> offsets, alongs;
        for (int i : pieces[k].cells)
        {
            cv::Point2f q(roomGrid[i][0], roomGrid[i][2]);
            offsets.push_back(line.towards.dot(q));
            alongs.push_back(along.dot(q));
            room.cellKind[i] = 'w';
        }
        line.offset = Percentile(offsets, 0.5);
        line.ends[0] = line.towards * line.offset + along * Percentile(alongs, 0.02);
        line.ends[1] = line.towards * line.offset + along * Percentile(alongs, 0.98);
        line.colour = averageColour(pieces[k].cells);
        lines.push_back(line);
    }
    // Corners: where two lines cross, if each has an end near the crossing, move both ends there.
    for (size_t i = 0; i < lines.size(); i++)
        for (size_t j = i + 1; j < lines.size(); j++)
        {
            cv::Point2f a = lines[i].towards, b = lines[j].towards;
            float determinant = a.x * b.y - a.y * b.x;   // sin of the angle between them
            if (std::abs(determinant) < 0.5f) continue;   // nearly parallel: no corner
            cv::Point2f crossing((lines[i].offset * b.y - lines[j].offset * a.y) / determinant,
                                 (a.x * lines[j].offset - b.x * lines[i].offset) / determinant);
            auto nearestEnd = [&](Line& line) -> cv::Point2f&
            { return cv::norm(line.ends[0] - crossing) < cv::norm(line.ends[1] - crossing) ? line.ends[0] : line.ends[1]; };
            cv::Point2f& endI = nearestEnd(lines[i]);
            cv::Point2f& endJ = nearestEnd(lines[j]);
            if (cv::norm(endI - crossing) < CornerSnapMeters && cv::norm(endJ - crossing) < CornerSnapMeters)
                endI = endJ = crossing;
        }
    for (const Line& line : lines)
    {
        cv::Point2f e0 = line.ends[0], e1 = line.ends[1];
        AddQuad(room.mesh, "clean_walls", { e0.x, room.floorY, e0.y }, { e1.x, room.floorY, e1.y },
                { e1.x, room.ceilingY, e1.y }, { e0.x, room.ceilingY, e0.y }, line.colour,
                { line.towards.x, 0, line.towards.y });
        room.walls.emplace_back(e0, e1);
    }

    // Floor and ceiling: rectangles reaching the walls and most of the points, and the camera.
    float minX = std::min(0.0f, Percentile(roomX, 0.02)), maxX = std::max(0.0f, Percentile(roomX, 0.98));
    float minZ = std::min(0.0f, Percentile(roomZ, 0.02)), maxZ = std::max(0.0f, Percentile(roomZ, 0.98));
    for (const auto& wall : room.walls)
        for (const cv::Point2f& e : { wall.first, wall.second })
        {
            minX = std::min(minX, e.x); maxX = std::max(maxX, e.x);
            minZ = std::min(minZ, e.y); maxZ = std::max(maxZ, e.y);
        }
    // ponytail: a rectangle, not the room's true outline; with every wall seen (13), the walls'
    // corners give the outline. Grey = not seen, a guess.
    const cv::Vec3b grey(128, 128, 128);
    for (bool isFloor : { true, false })
    {
        float y = isFloor ? room.floorY : room.ceilingY;
        int k = isFloor ? floorPiece : ceilingPiece;
        AddQuad(room.mesh, isFloor ? "clean_floor" : "clean_ceiling", { minX, y, minZ }, { maxX, y, minZ }, { maxX, y, maxZ },
                { minX, y, maxZ }, k >= 0 ? averageColour(pieces[k].cells) : grey, { 0, isFloor ? 1.0f : -1.0f, 0 });
    }

    // --- Step 8: furniture, from what's left between floor and ceiling ---
    const float margin = 0.05f;
    cv::Mat left(gridRows, gridColumns, CV_8U, cv::Scalar(0));
    for (size_t i = 0; i < grid.size(); i++)
    {
        if (std::isnan(grid[i][2]) || room.cellKind[i] != 0 ||
            roomGrid[i][1] < room.floorY + margin || roomGrid[i][1] > room.ceilingY - margin) continue;
        // A point on a wall's plane is wall, even outside the piece RANSAC kept (a strip cut off
        // by a door frame, say): not furniture.
        bool onWall = false;
        for (const Line& line : lines)
            onWall = onWall || std::abs(line.towards.dot(cv::Point2f(roomGrid[i][0], roomGrid[i][2])) - line.offset) < InlierThicknessMeters;
        if (!onWall) left.data[i] = 255;
    }
    // Split pieces where the depth jumps (a chair in front of a couch): clear the nearer side's cell.
    for (int row = 0; row + 1 < gridRows; row++)
        for (int column = 0; column + 1 < gridColumns; column++)
        {
            int i = row * gridColumns + column;
            for (int neighbour : { i + 1, i + gridColumns })
            {
                if (!left.data[i] || std::isnan(grid[neighbour][2])) continue;
                float z = grid[i][2], zn = grid[neighbour][2];
                if (std::abs(z - zn) > MaxJumpShare * std::min(z, zn)) left.data[z < zn ? i : neighbour] = 0;
            }
        }
    cv::Mat labels, stats, centroids;
    int count = cv::connectedComponentsWithStats(left, labels, stats, centroids);
    std::vector<std::vector<int>> blobCells(count);
    for (size_t i = 0; i < grid.size(); i++) if (labels.ptr<int>()[i] > 0) blobCells[labels.ptr<int>()[i]].push_back((int)i);
    for (int label = 1; label < count; label++)
    {
        if ((int)blobCells[label].size() < MinBlockCells) continue;
        std::vector<float> xs, ys, zs;
        for (int i : blobCells[label])
        {
            xs.push_back(roomGrid[i][0]);
            ys.push_back(roomGrid[i][1]);
            zs.push_back(roomGrid[i][2]);
            room.cellKind[i] = 'b';
        }
        // ponytail: boxes line up with the biggest wall and stand on the floor ("most furniture
        // does"); a picture on the wall becomes a floor-to-picture box. Only the front is seen, so
        // the depth is just what was visible.
        cv::Vec3f lo(Percentile(xs, 0.02), room.floorY, Percentile(zs, 0.02));
        cv::Vec3f hi(Percentile(xs, 0.98), Percentile(ys, 0.98), Percentile(zs, 0.98));
        AddBox(room.mesh, "furniture_blocks", lo, hi, averageColour(blobCells[label]));
        room.blocks.emplace_back(lo, hi);
    }
    return room;
}

// Step 9: as in 09b, with a "g" line before each group. The points are already in room
// coordinates (Y up, Z towards the viewer), so no flip.
bool SaveObj(const std::string& path, const Mesh& mesh)
{
    std::ofstream file(path);
    file << "# Spotter 09c: seen_room = scanned, clean_* and furniture_blocks = the tidy (partly guessed) model\n";
    for (size_t i = 0; i < mesh.points.size(); i++)
    {
        const cv::Vec3f& p = mesh.points[i];
        const cv::Vec3b& c = mesh.colours[i];
        file << cv::format("v %.4f %.4f %.4f %.3f %.3f %.3f\n", p[0], p[1], p[2], c[2] / 255.0, c[1] / 255.0, c[0] / 255.0);
    }
    for (const auto& [name, faces] : mesh.groups)
    {
        file << "g " << name << "\n";
        for (const cv::Vec3i& f : faces) file << cv::format("f %d %d %d\n", f[0] + 1, f[1] + 1, f[2] + 1);
    }
    return (bool)file;
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // -----------------------------------------------------------------
    //  Setup: Kinect and registration (as in 09b)
    // -----------------------------------------------------------------

    libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Warning));
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device* device = freenect2.enumerateDevices() > 0 ? freenect2.openDefaultDevice() : nullptr;
    if (device == nullptr)
    {
        std::printf("No Kinect v2 found, or it is in use. See the main README's troubleshooting table.\n");
        return 1;
    }
    libfreenect2::Freenect2Device::Config config;
    config.MaxDepth = MaxDepthMeters;
    device->setConfiguration(config);
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Depth);
    device->setColorFrameListener(&listener);
    device->setIrAndDepthFrameListener(&listener);
    device->startStreams(true, true);
    if (CapColorExposure) device->setColorSemiAutoExposure(16.0f);

    libfreenect2::Registration registration(device->getIrCameraParams(), device->getColorCameraParams());
    libfreenect2::Frame undistorted(Width, Height, 4), registered(Width, Height, 4), averaged(Width, Height, 4);
    std::vector<int> colorIndexOfDepthPixel(Width * Height);

    int framesCaptured = -1;
    std::vector<double> sumMm(Width * Height);
    std::vector<int> readings(Width * Height);
    std::vector<cv::Vec3f> cloud(Width * Height);
    std::vector<cv::Vec3b> colours(Width * Height);
    std::vector<std::pair<cv::Point, cv::Scalar>> dots;   // colour pixel and what it became, from the last scan
    std::mt19937 random(1);
    double fps = 0;
    auto lastFrameTime = std::chrono::steady_clock::now();

    const std::string window = "09c room model  (S scan, C clear, Q quit)";
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    std::printf("Press S to scan the room (in the window). Keep the camera still for %d frames.\n", AverageFrames);

    // -----------------------------------------------------------------
    //  Main operation
    // -----------------------------------------------------------------

    libfreenect2::FrameMap frames;
    cv::Mat frame;
    while (true)
    {
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == 'c') dots.clear();

        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }
        libfreenect2::Frame* colorFrame = frames[libfreenect2::Frame::Color];
        cv::cvtColor(cv::Mat(ColorHeight, ColorWidth, CV_8UC4, colorFrame->data), frame, cv::COLOR_BGRA2BGR);
        registration.apply(colorFrame, frames[libfreenect2::Frame::Depth], &undistorted, &registered,
                           false, nullptr, colorIndexOfDepthPixel.data());
        listener.release(frames);

        auto now = std::chrono::steady_clock::now();
        fps = 0.9 * fps + 0.1 / std::max(1e-3, std::chrono::duration<double>(now - lastFrameTime).count());
        lastFrameTime = now;

        // --- Steps 1-2: scan and average (as in 09b) ---
        if (key == 's' && framesCaptured < 0)
        {
            framesCaptured = 0;
            std::fill(sumMm.begin(), sumMm.end(), 0.0);
            std::fill(readings.begin(), readings.end(), 0);
            std::printf("Scanning: hold still...\n");
        }
        if (framesCaptured >= 0)
        {
            const float* depthMm = (const float*)undistorted.data;
            for (int i = 0; i < Width * Height; i++)
            {
                if (!(depthMm[i] > 0) || std::isinf(depthMm[i])) continue;
                sumMm[i] += depthMm[i];
                readings[i]++;
            }
            framesCaptured++;
        }

        // --- Steps 3-9: points (as in 09b), the model (new), save ---
        if (framesCaptured == AverageFrames)
        {
            framesCaptured = -1;
            auto start = std::chrono::steady_clock::now();
            float* averageMm = (float*)averaged.data;
            for (int i = 0; i < Width * Height; i++)
                averageMm[i] = readings[i] >= AverageFrames / 2 ? (float)(sumMm[i] / readings[i]) : 0;
            for (int row = 0; row < Height; row++)
            {
                for (int column = 0; column < Width; column++)
                {
                    float x, y, z;
                    registration.getPointXYZ(&averaged, row, column, x, y, z);
                    int i = row * Width + column;
                    cloud[i] = cv::Vec3f(x, y, z);
                    const unsigned char* bgrx = registered.data + 4 * i;
                    colours[i] = cv::Vec3b(bgrx[0], bgrx[1], bgrx[2]);
                }
            }

            Room room = BuildRoom(cloud, colours, random);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

            dots.clear();
            const int gridColumns = Width / SampleStep;
            for (size_t cell = 0; cell < room.cellKind.size(); cell++)
            {
                int pixel = (int)(cell / gridColumns) * SampleStep * Width + (int)(cell % gridColumns) * SampleStep;
                if (!room.cellKind[cell] || colorIndexOfDepthPixel[pixel] < 0) continue;
                char kind = room.cellKind[cell];
                cv::Scalar colour = kind == 'w' ? cv::Scalar(255, 80, 0) : kind == 'f' ? cv::Scalar(0, 200, 0)
                                  : kind == 'c' ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255);
                dots.emplace_back(cv::Point(colorIndexOfDepthPixel[pixel] % ColorWidth, colorIndexOfDepthPixel[pixel] / ColorWidth), colour);
            }

            std::string path = (std::filesystem::current_path() / cv::format("room_model_%lld.obj", (long long)std::time(nullptr))).string();
            if (room.mesh.points.empty())
                std::printf("Nothing to model: no depth readings.\n");
            else if (SaveObj(path, room.mesh))
            {
                std::printf("Saved %s (%.1f MB, %.0f ms). A capture of the room: never commit it.\n", path.c_str(),
                            std::filesystem::file_size(path) / 1e6, ms);
                std::printf("  up from %s: camera tilted %+.1f deg (+ = looking up), rolled %+.1f deg\n",
                            room.upFrom.c_str(), room.tiltDegrees, room.rollDegrees);
                std::printf("  floor %.2f m below the camera (%s), ceiling %.2f m above it (%s): %.2f m high\n",
                            -room.floorY, room.floorSeen ? "seen" : "guessed: lowest point",
                            room.ceilingY, room.ceilingSeen ? "seen" : "guessed: highest point", room.ceilingY - room.floorY);
                for (const auto& [a, b] : room.walls)
                    std::printf("  wall  %.2f m long, from (%+.2f, %.2f) to (%+.2f, %.2f) m (X right, Z ahead)\n",
                                cv::norm(b - a), a.x, -a.y, b.x, -b.y);
                for (const auto& [lo, hi] : room.blocks)
                    std::printf("  block %.2f wide x %.2f deep x %.2f high, at X %+.2f, Z %.2f m\n", hi[0] - lo[0],
                                hi[2] - lo[2], hi[1] - lo[1], (lo[0] + hi[0]) / 2, -(lo[2] + hi[2]) / 2);
            }
            else
                std::printf("Could not write %s\n", path.c_str());
        }

        // -----------------------------------------------------------------
        //  Output: half-size colour, the last scan's points dotted by kind
        // -----------------------------------------------------------------

        cv::Mat display;
        cv::resize(frame, display, cv::Size(ColorWidth / 2, ColorHeight / 2));
        for (const auto& [p, colour] : dots) cv::circle(display, p / 2, 2, colour, cv::FILLED);
        std::string status = framesCaptured >= 0 ? cv::format("HOLD STILL  %d / %d", framesCaptured, AverageFrames)
                                                 : cv::format("%.1f fps   S to scan the room", fps);
        cv::putText(display, status, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        cv::imshow(window, display);
    }

    device->stop();
    device->close();
    cv::destroyAllWindows();
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Open the .obj in Blender or MeshLab and hide groups: seen_room alone is 09b, levelled; the
//     clean_* groups and furniture_blocks alone are the tidy room. Where do the two disagree?
//  2. Is "up" right? The console says what it came from. Tilt the Kinect and scan again: the
//     model should still stand straight (the camera's tilt is printed instead).
//  3. Point the Kinect into a corner. Are the two walls joined, and about 90 deg apart in Blender?
//     CornerSnapMeters = 0.05: do they still meet?
//  4. Which blocks are real furniture, and which are bits of wall edge or clutter? Try
//     MinBlockCells = 10 and 100.
//  5. A chair in front of a couch: one block or two? (MaxJumpShare splits them where the depth jumps.)
//  6. MaxTiltDegrees = 5: what stops counting as a wall or a floor? Which way do your walls lean?
//  7. Compare a block's printed size with something of known size (an A4 sheet is 297 x 210 mm).
//  8. Scan from two spots. The two models are each turned to their own biggest wall, so if both
//     saw the same wall they're already turned alike; only the shift is left. Example 11 finds both.
