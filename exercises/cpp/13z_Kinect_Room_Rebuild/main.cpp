// =====================================================================
//  13z · Kinect Room Rebuild
//
//  Rebuilds rooms from the folders that 13c's and 13d's K key keeps (or
//  their rescue, if the Kinect was lost), and that 13e writes as it scans,
//  any time later, without the Kinect. No scanning and no placing here:
//  each scan's points, pictures and placement are read back from its
//  folder. A scan without its pose.txt (cut off by a crash) is skipped.
//
//    1. Find     every saved room under SearchFolder (or a folder you
//                give), listed with its scans, date and size.
//    2. Choose   which rooms, and in each one which scans to load
//                (1-12,15). Only those are read, so memory stays low.
//    3. Do       view one (13c's "room so far" viewer, where X excludes
//                a scan), or build files from it, or both:
//                  .obj  13c's E: TSDF merge + surface net + clean model
//                        (plus far_landmarks for a 13d room)
//                  .ply  every scan's own points and colours (K's points.ply)
//                Builds run without a window, so many rooms go in one go.
//
//  The files land in each room's own folder: room_scan_<time>.obj (with
//  the line Demo 4 needs) and points_<time>.ply. K's own points.ply is
//  never overwritten. The merge settings below can be finer than live
//  (VoxelMeters 0.01): time doesn't matter offline.
//
//  Run:      13z_Kinect_Room_Rebuild [folder]   (default: SearchFolder)
//            then answer in the console.
//  Viewer:   mouse drag / W A S D turn    wheel / + - zoom    R reset view
//            C colour by scan    click a camera circle to select it
//            X exclude / include (saved to the scan's pose.txt)   Esc deselect
//            E build .obj   K build .ply   N next room   Q quit
//
//  Privacy: the folders and what this builds are captures of a room in
//  colour. room_scans/, *.obj and *.ply are gitignored: never commit them.
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// --- Finding the rooms (new) ---
const std::string SearchFolder = "room_scans";   // where K saves by default; relative = the folder you run from
const int SearchDepth = 3;                       // how many folders deep to look for saved rooms

// --- Loading (new) ---
const int BoundsStep = 2;   // every 2nd pixel's point sets the merge's box (13c took its ICP points)

// --- Merging (as in 13c's E) ---
// The TSDF grid's cube size. 3 cm: walls look flat and a room needs a few million cubes (about
// 20 bytes each). Smaller = finer and slower; it grows by itself if the room would need more
// than MaxVoxels.
const float VoxelMeters = 0.02f;     // 13: 0.03
const int MaxVoxels = 80000000;      // 13: 30M. About 24 bytes each: up to 1.9 GB (fine on the PC, too much for a Pi)

// How far in front of and behind a surface a scan updates the grid, in cubes. Behind that, the
// scan can't know what's there (it might be the far side of a wall), so it leaves it alone.
const int TruncationVoxels = 3;

// --- Clean model (09c) ---
const int MaxPlanes = 8;
const float InlierThicknessMeters = 0.03f;
const int SampleStep = 4;
const int NeighbourPixels = 40;
const double MinPlaneShare = 0.03;
const float MaxTiltDegrees = 25;
const float MinWallHeightMeters = 0.8f;
const float MinCeilingHeightMeters = 1.8f;
const float CornerSnapMeters = 0.5f;
const int MinBlockCells = 30;
const float LevelDegrees = 10;       // New: once "up" is known, floor/ceiling/walls must be this close to level/upright
const float MaxBlockMeters = 3.0f;   // New: furniture boxes that touch join only up to this size (else one box swallows the flat)

// New: the same plane seen from two spots is merged into one when it's this close in angle and
// distance. Boxes from different scans that touch or overlap become one box.
const float MergeDegrees = 10;
const float MergeMeters = 0.05f;

// --- Viewer (as in 13c) ---
const int ViewWidth = 960, ViewHeight = 720;
const float ViewFocalPixels = 700;
const float PivotMeters = 2.5f;
const int PointSizePixels = 2;
const float TurnStepDegrees = 5;
const float ZoomStepMeters = 0.25f;
const int ShowStep = 2;   // the viewer draws every 2nd pixel of each scan, to stay quick
const float ScanArrowMeters = 0.6f;   // the 4 arrows from each camera circle: its field of view, this far ahead
const int MaxShownPoints = 600000;    // new: the viewer draws at most this many (every k-th point of each scan)
const int Width = 512, Height = 424;   // the depth picture, as K saved it (room.txt says so too)

// ---------------------------------------------------------------------
//  The viewer (copied from 13c)
// ---------------------------------------------------------------------

float yawDegrees = 0, pitchDegrees = 0;
float viewDistance = PivotMeters;
bool dragging = false;
int lastMouseX = 0, lastMouseY = 0;
int pressX = 0, pressY = 0;          // new: where the button went down...
bool clicked = false;                // ...and a click = released within 4 px of it (not a drag)
int clickX = 0, clickY = 0;
float viewCosYaw = 1, viewSinYaw = 0, viewCosPitch = 1, viewSinPitch = 0;   // new: once per frame, not per point

void UpdateView()
{
    const float yaw = yawDegrees * (float)CV_PI / 180, pitch = pitchDegrees * (float)CV_PI / 180;
    viewCosYaw = std::cos(yaw);
    viewSinYaw = std::sin(yaw);
    viewCosPitch = std::cos(pitch);
    viewSinPitch = std::sin(pitch);
}

void OnMouse(int event, int x, int y, int flags, void*)
{
    if (event == cv::EVENT_LBUTTONDOWN) { dragging = true; lastMouseX = pressX = x; lastMouseY = pressY = y; }
    if (event == cv::EVENT_LBUTTONUP)
    {
        dragging = false;
        if (std::abs(x - pressX) + std::abs(y - pressY) < 4) { clicked = true; clickX = x; clickY = y; }
    }
    if (event == cv::EVENT_MOUSEMOVE && dragging)
    {
        yawDegrees += (x - lastMouseX) * 0.3f;
        pitchDegrees -= (y - lastMouseY) * 0.3f;
        lastMouseX = x;
        lastMouseY = y;
    }
    if (event == cv::EVENT_MOUSEWHEEL)
        viewDistance -= cv::getMouseWheelDelta(flags) / 120.0f * ZoomStepMeters;
}

bool Project(const cv::Point3f& point, cv::Point& pixel, float& depth)
{
    float x = point.x, y = point.y, z = point.z - PivotMeters;
    float x1 = viewCosYaw * x + viewSinYaw * z;
    float z1 = -viewSinYaw * x + viewCosYaw * z;
    float y2 = viewCosPitch * y - viewSinPitch * z1;
    float z2 = viewSinPitch * y + viewCosPitch * z1;
    depth = z2 + viewDistance;
    if (depth < 0.05f) return false;
    pixel = cv::Point((int)(ViewFocalPixels * x1 / depth + ViewWidth / 2),
                      (int)(ViewFocalPixels * y2 / depth + ViewHeight / 2));
    return pixel.x >= 0 && pixel.y >= 0 && pixel.x < ViewWidth && pixel.y < ViewHeight;
}

cv::Vec3b ColorForId(int id)   // as in 04
{
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar((id * 47) % 180, 255, 255)), bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr.at<cv::Vec3b>(0, 0);
}

// As in 08, but drawn straight from each scan's own points (13b copied them all into one list every
// frame), and only every stride-th point once there are more than MaxShownPoints. Excluded scans
// are drawn dim grey.
// ponytail: the stride drops detail evenly everywhere; a merged, thinned display cloud would keep it.
cv::Mat Render(const std::vector<std::vector<cv::Point3f>>& points, const std::vector<std::vector<cv::Vec3b>>& colours,
               const std::vector<char>& excluded, bool byScan)
{
    cv::Mat image(ViewHeight, ViewWidth, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::Mat nearest(ViewHeight, ViewWidth, CV_32F, cv::Scalar(1e9));
    size_t total = 0;
    for (const auto& p : points) total += p.size();
    const size_t stride = std::max<size_t>(1, (total + MaxShownPoints - 1) / MaxShownPoints);
    for (size_t s = 0; s < points.size(); s++)
    {
        const cv::Vec3b scanColour = excluded[s] ? cv::Vec3b(70, 70, 70) : ColorForId((int)s + 1);
        for (size_t i = 0; i < points[s].size(); i += stride)
        {
            cv::Point pixel;
            float depth;
            if (!Project(points[s][i], pixel, depth)) continue;
            const cv::Vec3b& colour = excluded[s] || byScan ? scanColour : colours[s][i];
            for (int dy = 0; dy < PointSizePixels; dy++)
            {
                for (int dx = 0; dx < PointSizePixels; dx++)
                {
                    int u = pixel.x + dx, v = pixel.y + dy;
                    if (u >= ViewWidth || v >= ViewHeight || depth >= nearest.at<float>(v, u)) continue;
                    nearest.at<float>(v, u) = depth;
                    image.at<cv::Vec3b>(v, u) = colour;
                }
            }
        }
    }
    return image;
}

// ---------------------------------------------------------------------
//  A saved scan (new): what 13c kept in memory, read back from its folder
// ---------------------------------------------------------------------

struct Pose   // as in 13c: this scan's camera -> the room (= scan 1's camera)
{
    cv::Matx33f turn = cv::Matx33f::eye();
    cv::Point3f shift = cv::Point3f(0, 0, 0);
    cv::Point3f operator()(const cv::Point3f& p) const { return turn * p + shift; }
};

// The depth camera's lens, to find which pixel sees a point (as in 10).
struct Lens { float fx, fy, cx, cy; };

struct Scan
{
    int number = 0;                   // its number in the room (scan_005 = 5), also when only some are loaded
    fs::path folder;
    std::vector<cv::Point3f> cloud;   // a 3D point per depth pixel (NaN = none), in this scan's camera coordinates
    cv::Mat picture;                  // the colour picture on the depth grid (black = no colour)
    std::vector<cv::Point3f> sparse;  // every BoundsStep-th pixel's point, for the merge's box
    Pose pose;
    bool excluded = false;
    bool suspicious = false;
    std::string stats;
};

// A far landmark from 13d: only its place is needed here.
struct Landmark { cv::Point3f point; };

// ---------------------------------------------------------------------
//  Merging: a TSDF grid and a surface net (new)
// ---------------------------------------------------------------------

// A triangle mesh with named groups (09c).
struct Mesh
{
    std::vector<cv::Vec3f> points;
    std::vector<cv::Vec3b> colours;
    std::map<std::string, std::vector<cv::Vec3i>> groups;
};


// Step 4a: every scan into one grid of little cubes ("voxels"), then one mesh where the surface is.
// Each voxel keeps a signed distance to the nearest surface along the camera's view: + in front
// (empty air), - just behind, cut off ("truncated") at TruncationVoxels cubes, and averaged over
// every scan that saw it. Noise averages out; two scans of the same wall become one wall.
Mesh FuseScans(const std::vector<Scan>& scans, const Lens& lens, float& voxel)
{
    // The box to fill: around the scans' points, 1-99% so stray points don't blow it up.
    std::vector<float> xs, ys, zs;
    for (const Scan& scan : scans)
        for (const cv::Point3f& p : scan.sparse)
        {
            cv::Point3f q = scan.pose(p);
            xs.push_back(q.x); ys.push_back(q.y); zs.push_back(q.z);
        }
    auto percentile = [](std::vector<float>& v, double share)
    {
        size_t i = std::min(v.size() - 1, (size_t)(share * v.size()));
        std::nth_element(v.begin(), v.begin() + i, v.end());
        return v[i];
    };
    const float margin = 0.3f;
    cv::Point3f lo(percentile(xs, 0.01) - margin, percentile(ys, 0.01) - margin, percentile(zs, 0.01) - margin);
    cv::Point3f hi(percentile(xs, 0.99) + margin, percentile(ys, 0.99) + margin, percentile(zs, 0.99) + margin);
    voxel = VoxelMeters;
    int nx, ny, nz;
    while (true)
    {
        nx = (int)std::ceil((hi.x - lo.x) / voxel);
        ny = (int)std::ceil((hi.y - lo.y) / voxel);
        nz = (int)std::ceil((hi.z - lo.z) / voxel);
        if ((double)nx * ny * nz <= MaxVoxels) break;
        voxel *= 1.25f;   // too big a room for this cube size: coarser cubes
    }
    const size_t count = (size_t)nx * ny * nz;
    std::vector<float> distance(count, 0), weight(count, 0), colourWeight(count, 0);
    std::vector<cv::Vec3f> colourSum(count, cv::Vec3f(0, 0, 0));
    const float truncation = TruncationVoxels * voxel;
    auto index = [&](int x, int y, int z) { return ((size_t)z * ny + y) * nx + x; };

    for (const Scan& scan : scans)
    {
        cv::Matx33f back = scan.pose.turn.t();   // room -> this scan's camera
        cv::parallel_for_(cv::Range(0, nz), [&](const cv::Range& slices)
        {
            for (int z = slices.start; z < slices.end; z++)
                for (int y = 0; y < ny; y++)
                    for (int x = 0; x < nx; x++)
                    {
                        cv::Point3f centre(lo.x + (x + 0.5f) * voxel, lo.y + (y + 0.5f) * voxel, lo.z + (z + 0.5f) * voxel);
                        cv::Point3f p = back * (centre - scan.pose.shift);
                        if (p.z < 0.3f) continue;   // behind or right at the camera
                        int column = (int)std::lround(p.x * lens.fx / p.z + lens.cx - 0.5f);
                        int row = (int)std::lround(p.y * lens.fy / p.z + lens.cy - 0.5f);
                        if (column < 0 || row < 0 || column >= Width || row >= Height) continue;
                        float seen = scan.cloud[row * Width + column].z;
                        if (std::isnan(seen)) continue;
                        float signedDistance = seen - p.z;          // + = in front of what the camera saw
                        if (signedDistance < -truncation) continue;   // hidden behind it: unknown
                        size_t i = index(x, y, z);
                        float value = std::min(1.0f, signedDistance / truncation);
                        distance[i] = (distance[i] * weight[i] + value) / (weight[i] + 1);
                        weight[i] += 1;
                        if (signedDistance < truncation)
                        {
                            cv::Vec3b c = scan.picture.at<cv::Vec3b>(row, column);
                            if (c != cv::Vec3b(0, 0, 0)) { colourSum[i] += cv::Vec3f(c[0], c[1], c[2]); colourWeight[i] += 1; }
                        }
                    }
        });
    }

    // Step 4b: a surface net. Each cube of 8 neighbouring voxels where the sign changes gets one
    // point, at the average of where the surface crosses its edges. Then every voxel edge that
    // crosses the surface joins the 4 cubes around it into a square (two triangles).
    Mesh mesh;
    std::vector<int> pointOf(count, -1);
    auto corner = [](int c) { return cv::Vec3f((float)(c & 1), (float)((c >> 1) & 1), (float)((c >> 2) & 1)); };
    for (int z = 0; z + 1 < nz; z++)
        for (int y = 0; y + 1 < ny; y++)
            for (int x = 0; x + 1 < nx; x++)
            {
                float value[8];
                int inside = 0;
                bool seenAll = true;
                cv::Vec3f colour(0, 0, 0);
                float colourCount = 0;
                for (int c = 0; c < 8 && seenAll; c++)
                {
                    size_t i = index(x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1));
                    seenAll = weight[i] > 0;
                    value[c] = distance[i];
                    if (value[c] < 0) inside++;
                    colour += colourSum[i];
                    colourCount += colourWeight[i];
                }
                if (!seenAll || inside == 0 || inside == 8) continue;   // never seen, or no surface here
                cv::Vec3f sum(0, 0, 0);
                int crossings = 0;
                for (int bit : { 1, 2, 4 })
                    for (int c = 0; c < 8; c++)
                    {
                        if (c & bit) continue;
                        float a = value[c], b = value[c | bit];
                        if ((a < 0) == (b < 0)) continue;
                        sum += corner(c) + (corner(c | bit) - corner(c)) * (a / (a - b));   // where it crosses zero
                        crossings++;
                    }
                cv::Vec3f inCube = sum / (float)crossings;
                pointOf[index(x, y, z)] = (int)mesh.points.size();
                mesh.points.emplace_back(lo.x + (x + 0.5f + inCube[0]) * voxel, lo.y + (y + 0.5f + inCube[1]) * voxel,
                                         lo.z + (z + 0.5f + inCube[2]) * voxel);
                mesh.colours.push_back(colourCount > 0 ? cv::Vec3b(colour / colourCount) : cv::Vec3b(128, 128, 128));
            }
    std::vector<cv::Vec3i>& faces = mesh.groups["seen_room"];
    const int size[3] = { nx, ny, nz };
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
            {
                size_t i = index(x, y, z);
                if (weight[i] == 0) continue;
                for (int axis = 0; axis < 3; axis++)
                {
                    int at[3] = { x, y, z };
                    if (at[axis] + 1 >= size[axis]) continue;
                    at[axis]++;
                    size_t j = index(at[0], at[1], at[2]);
                    if (weight[j] == 0 || (distance[i] < 0) == (distance[j] < 0)) continue;
                    // The 4 cubes around this edge: one step back along the other two axes, or not.
                    int u = (axis + 1) % 3, w = (axis + 2) % 3, quad[4];
                    const int du[4] = { -1, 0, 0, -1 }, dw[4] = { -1, -1, 0, 0 };
                    bool complete = true;
                    for (int k = 0; k < 4 && complete; k++)
                    {
                        int c[3] = { x, y, z };
                        c[u] += du[k];
                        c[w] += dw[k];
                        complete = c[u] >= 0 && c[w] >= 0 && c[u] + 1 < size[u] && c[w] + 1 < size[w];
                        if (complete) { quad[k] = pointOf[index(c[0], c[1], c[2])]; complete = quad[k] >= 0; }
                    }
                    if (!complete) continue;
                    // Face the empty side (+): towards voxel i if it's the + one.
                    cv::Vec3f facing(0, 0, 0);
                    facing[axis] = distance[i] > 0 ? -1.0f : 1.0f;
                    cv::Vec3f normal = (mesh.points[quad[1]] - mesh.points[quad[0]]).cross(mesh.points[quad[2]] - mesh.points[quad[0]]);
                    if (normal.dot(facing) < 0) std::swap(quad[1], quad[3]);
                    faces.emplace_back(quad[0], quad[1], quad[2]);
                    faces.emplace_back(quad[0], quad[2], quad[3]);
                }
            }
    return mesh;
}

// ---------------------------------------------------------------------
//  The clean model, from all scans (09c, plus merging planes and boxes)
// ---------------------------------------------------------------------

struct Plane
{
    cv::Vec3f normal;
    float distance;
    float DistanceTo(const cv::Vec3f& point) const { return normal.dot(point) + distance; }
};

struct Piece { Plane plane; std::vector<int> cells; };   // one scan's plane and its grid cells (09c)

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

float Percentile(std::vector<float> values, double share)
{
    size_t i = std::min(values.size() - 1, (size_t)(share * values.size()));
    std::nth_element(values.begin(), values.begin() + i, values.end());
    return values[i];
}

// As in 09c.
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
        for (int t = 0; t < 200; t++)
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

struct Room
{
    Mesh mesh;                 // room coordinates: X right, Y up, Z towards the first camera; first camera at 0,0,0
    std::string upFrom;
    float floorY = 0, ceilingY = 0;
    bool floorSeen = false, ceilingSeen = false;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> walls;
    std::vector<std::pair<cv::Vec3f, cv::Vec3f>> blocks;
    cv::Matx33f firstCamera = cv::Matx33f::eye();   // scan 1's camera (X right, Y down, Z ahead) -> room
};

// Step 4c: 09c's clean model from every scan. Each scan's planes are found as in 09c, moved into
// the room, and merged with the same plane seen from other spots; then sorted, levelled, and
// built as in 09c. The merged mesh from step 4b is levelled with it.
Room BuildRoom(const std::vector<Scan>& scans, const Mesh& seen, std::mt19937& random)
{
    Room room;
    room.mesh = seen;
    const int gridColumns = Width / SampleStep, gridRows = Height / SampleStep, cells = gridColumns * gridRows;
    struct RoomPiece { Plane plane; std::vector<cv::Vec3f> points; cv::Vec3d colourSum; std::vector<std::pair<int, int>> cells; };
    std::vector<RoomPiece> pieces;
    std::vector<std::vector<cv::Vec3f>> roomGrid(scans.size());   // each scan's sampled points, in the room (first camera)
    std::vector<std::vector<cv::Vec3b>> gridColour(scans.size());
    std::vector<std::vector<float>> gridDepth(scans.size());        // ...and their distance from their own camera

    for (size_t s = 0; s < scans.size(); s++)
    {
        std::vector<cv::Vec3f> grid(cells);
        roomGrid[s].resize(cells);
        gridColour[s].resize(cells);
        gridDepth[s].resize(cells);
        for (int cell = 0; cell < cells; cell++)
        {
            int pixel = (cell / gridColumns) * SampleStep * Width + (cell % gridColumns) * SampleStep;
            cv::Point3f p = scans[s].cloud[pixel];
            grid[cell] = cv::Vec3f(p.x, p.y, p.z);
            roomGrid[s][cell] = std::isnan(p.z) ? cv::Vec3f(NAN, NAN, NAN) : cv::Vec3f(scans[s].pose(p));
            gridColour[s][cell] = scans[s].picture.at<cv::Vec3b>(pixel / Width, pixel % Width);
            gridDepth[s][cell] = p.z;
        }
        for (const Piece& found : FindPlanes(grid, gridColumns, random))
        {
            RoomPiece piece{ {}, {}, cv::Vec3d(0, 0, 0), {} };
            cv::Vec3f centre(0, 0, 0);
            for (int i : found.cells)
            {
                piece.points.push_back(roomGrid[s][i]);
                piece.colourSum += cv::Vec3d(gridColour[s][i]);
                piece.cells.emplace_back((int)s, i);
                centre += roomGrid[s][i];
            }
            centre /= (float)found.cells.size();
            piece.plane = FitPlane(piece.points);
            // New: the same plane seen from an earlier spot? Then it's one piece.
            bool merged = false;
            for (RoomPiece& other : pieces)
            {
                if (other.plane.normal.dot(piece.plane.normal) < std::cos(MergeDegrees * (float)CV_PI / 180) ||
                    std::abs(other.plane.DistanceTo(centre)) > MergeMeters) continue;
                other.points.insert(other.points.end(), piece.points.begin(), piece.points.end());
                other.cells.insert(other.cells.end(), piece.cells.begin(), piece.cells.end());
                other.colourSum += piece.colourSum;
                other.plane = FitPlane(other.points);
                merged = true;
                break;
            }
            if (!merged) pieces.push_back(piece);
        }
    }
    auto averageColour = [](const cv::Vec3d& sum, size_t count) { return cv::Vec3b(sum / (double)std::max<size_t>(1, count)); };

    // --- Sort the planes (09c step 5): first by the first camera's up, then again by the room's ---
    const cv::Vec3f cameraUp(0, -1, 0);
    int floorPiece = -1, ceilingPiece = -1;
    std::vector<int> wallPieces;
    auto Sort = [&](const cv::Vec3f& by, float tiltDegrees)
    {
        const float levelCos = std::cos(tiltDegrees * (float)CV_PI / 180), uprightSin = std::sin(tiltDegrees * (float)CV_PI / 180);
        std::vector<float> heights;
        for (const auto& grid : roomGrid)
            for (const cv::Vec3f& p : grid) if (!std::isnan(p[2])) heights.push_back(p.dot(by));
        float lowest = Percentile(heights, 0.02);
        floorPiece = ceilingPiece = -1;
        wallPieces.clear();
        for (int k = 0; k < (int)pieces.size(); k++)
        {
            const Plane& plane = pieces[k].plane;
            float upness = plane.normal.dot(by);
            if (upness > levelCos)
            {
                if (floorPiece < 0 || plane.distance > pieces[floorPiece].plane.distance) floorPiece = k;
            }
            else if (-upness > levelCos)
            {
                bool highEnough = plane.distance - lowest >= MinCeilingHeightMeters;
                if (highEnough && (ceilingPiece < 0 || plane.distance > pieces[ceilingPiece].plane.distance)) ceilingPiece = k;
            }
            else if (std::abs(upness) < uprightSin)
            {
                std::vector<float> wallHeights;
                for (const cv::Vec3f& p : pieces[k].points) wallHeights.push_back(p.dot(by));
                if (Percentile(wallHeights, 0.98) - Percentile(wallHeights, 0.02) >= MinWallHeightMeters) wallPieces.push_back(k);
            }
        }
    };
    bool anyPoints = false;
    for (const auto& grid : roomGrid) for (const cv::Vec3f& p : grid) anyPoints = anyPoints || !std::isnan(p[2]);
    if (!anyPoints) return room;

    // --- Up and ahead (09c step 6) ---
    // New: up comes from the BIGGEST level plane, floor or ceiling, not the farthest floor-ish one.
    // The farthest could be a ramp or a couch back (a real 34-scan export took a slanted piece as
    // the floor and came out tilted 28 degrees, with its ceiling ignored).
    cv::Vec3f up = cameraUp;
    room.upFrom = "the first camera (nothing level found, under 2 walls)";
    const float tiltCos = std::cos(MaxTiltDegrees * (float)CV_PI / 180);
    size_t biggest = 0;
    for (const RoomPiece& piece : pieces)
    {
        float upness = piece.plane.normal.dot(cameraUp);
        if (std::abs(upness) <= tiltCos || piece.points.size() <= biggest) continue;
        biggest = piece.points.size();
        up = upness > 0 ? piece.plane.normal : -piece.plane.normal;
        room.upFrom = upness > 0 ? "the floor" : "the ceiling";
    }
    Sort(cameraUp, MaxTiltDegrees);
    if (biggest == 0)
    {
        float bestAcross = 0.5f;
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
    Sort(up, LevelDegrees);   // now by the room's own up, tightly: a ramp is neither floor nor wall
    std::sort(wallPieces.begin(), wallPieces.end(), [&](int a, int b) { return pieces[a].points.size() > pieces[b].points.size(); });
    auto Flatten = [&](cv::Vec3f v) { v -= up * up.dot(v); return v / (float)cv::norm(v); };
    cv::Vec3f towards = wallPieces.empty() ? Flatten(cv::Vec3f(0, 0, -1)) : Flatten(pieces[wallPieces[0]].plane.normal);
    cv::Vec3f right = up.cross(towards);
    auto ToRoom = [&](const cv::Vec3f& p) { return cv::Vec3f(right.dot(p), up.dot(p), towards.dot(p)); };
    for (cv::Vec3f& p : room.mesh.points) p = ToRoom(p);
    room.firstCamera = cv::Matx33f(right[0], right[1], right[2], up[0], up[1], up[2], towards[0], towards[1], towards[2]);

    std::vector<float> roomHeights, roomX, roomZ;
    for (auto& grid : roomGrid)
        for (cv::Vec3f& p : grid)
        {
            if (std::isnan(p[2])) continue;
            p = ToRoom(p);
            roomHeights.push_back(p[1]);
            roomX.push_back(p[0]);
            roomZ.push_back(p[2]);
        }
    for (RoomPiece& piece : pieces) for (cv::Vec3f& p : piece.points) p = ToRoom(p);
    room.floorSeen = floorPiece >= 0;
    room.ceilingSeen = ceilingPiece >= 0;
    room.floorY = room.floorSeen ? -pieces[floorPiece].plane.distance : Percentile(roomHeights, 0.02);
    room.ceilingY = room.ceilingSeen ? pieces[ceilingPiece].plane.distance : Percentile(roomHeights, 0.98);
    std::vector<std::vector<char>> taken(scans.size(), std::vector<char>(cells, 0));   // cells used by floor, ceiling, walls
    for (int k : { floorPiece, ceilingPiece })
        if (k >= 0) for (const auto& [s, i] : pieces[k].cells) taken[s][i] = 1;

    // --- Walls (09c step 7) ---
    struct Line { cv::Point2f towards, ends[2]; float offset; cv::Vec3b colour; };
    std::vector<Line> lines;
    for (int k : wallPieces)
    {
        cv::Vec3f n = ToRoom(Flatten(pieces[k].plane.normal));
        Line line;
        line.towards = cv::Point2f(n[0], n[2]) / std::hypot(n[0], n[2]);
        cv::Point2f along(-line.towards.y, line.towards.x);
        std::vector<float> offsets, alongs;
        for (const cv::Vec3f& p : pieces[k].points)
        {
            cv::Point2f q(p[0], p[2]);
            offsets.push_back(line.towards.dot(q));
            alongs.push_back(along.dot(q));
        }
        for (const auto& [s, i] : pieces[k].cells) taken[s][i] = 1;
        line.offset = Percentile(offsets, 0.5);
        line.ends[0] = line.towards * line.offset + along * Percentile(alongs, 0.02);
        line.ends[1] = line.towards * line.offset + along * Percentile(alongs, 0.98);
        line.colour = averageColour(pieces[k].colourSum, pieces[k].points.size());
        lines.push_back(line);
    }
    for (size_t i = 0; i < lines.size(); i++)
        for (size_t j = i + 1; j < lines.size(); j++)
        {
            cv::Point2f a = lines[i].towards, b = lines[j].towards;
            float determinant = a.x * b.y - a.y * b.x;
            if (std::abs(determinant) < 0.5f) continue;
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

    // Floor and ceiling: rectangles over the walls, most of the points, and every camera.
    float minX = Percentile(roomX, 0.02), maxX = Percentile(roomX, 0.98);
    float minZ = Percentile(roomZ, 0.02), maxZ = Percentile(roomZ, 0.98);
    std::vector<cv::Point2f> outline;
    for (const auto& wall : room.walls) { outline.push_back(wall.first); outline.push_back(wall.second); }
    for (const Scan& scan : scans) { cv::Vec3f c = ToRoom(cv::Vec3f(scan.pose.shift)); outline.emplace_back(c[0], c[2]); }
    for (const cv::Point2f& e : outline)
    {
        minX = std::min(minX, e.x); maxX = std::max(maxX, e.x);
        minZ = std::min(minZ, e.y); maxZ = std::max(maxZ, e.y);
    }
    const cv::Vec3b grey(128, 128, 128);
    for (bool isFloor : { true, false })
    {
        float y = isFloor ? room.floorY : room.ceilingY;
        int k = isFloor ? floorPiece : ceilingPiece;
        AddQuad(room.mesh, isFloor ? "clean_floor" : "clean_ceiling", { minX, y, minZ }, { maxX, y, minZ }, { maxX, y, maxZ },
                { minX, y, maxZ }, k >= 0 ? averageColour(pieces[k].colourSum, pieces[k].points.size()) : grey,
                { 0, isFloor ? 1.0f : -1.0f, 0 });
    }

    // --- Furniture (09c step 8), per scan, then boxes merged across scans (new) ---
    struct Box { cv::Vec3f lo, hi; cv::Vec3d colourSum; size_t count; };
    std::vector<Box> boxes;
    const float margin = 0.05f;
    for (size_t s = 0; s < scans.size(); s++)
    {
        cv::Mat left(gridRows, gridColumns, CV_8U, cv::Scalar(0));
        for (int i = 0; i < cells; i++)
        {
            const cv::Vec3f& p = roomGrid[s][i];
            if (std::isnan(p[2]) || taken[s][i] || p[1] < room.floorY + margin || p[1] > room.ceilingY - margin) continue;
            // New: a point on a wall's plane is wall, even if this scan saw too little of it to find
            // the plane itself (a sliver at the edge of the view).
            bool onWall = false;
            for (const Line& line : lines)
                onWall = onWall || std::abs(line.towards.dot(cv::Point2f(p[0], p[2])) - line.offset) < InlierThicknessMeters;
            if (!onWall) left.data[i] = 255;
        }
        for (int row = 0; row + 1 < gridRows; row++)
            for (int column = 0; column + 1 < gridColumns; column++)
            {
                int i = row * gridColumns + column;
                for (int neighbour : { i + 1, i + gridColumns })
                {
                    float z = gridDepth[s][i], zn = gridDepth[s][neighbour];
                    if (!left.data[i] || std::isnan(zn)) continue;
                    if (std::abs(z - zn) > 0.03f * std::min(z, zn)) left.data[z < zn ? i : neighbour] = 0;
                }
            }
        cv::Mat labels, stats, centroids;
        int count = cv::connectedComponentsWithStats(left, labels, stats, centroids);
        std::vector<std::vector<int>> blobCells(count);
        for (int i = 0; i < cells; i++) if (labels.ptr<int>()[i] > 0) blobCells[labels.ptr<int>()[i]].push_back(i);
        for (int label = 1; label < count; label++)
        {
            if ((int)blobCells[label].size() < MinBlockCells) continue;
            std::vector<float> xs, ys, zs;
            Box box{ {}, {}, cv::Vec3d(0, 0, 0), blobCells[label].size() };
            for (int i : blobCells[label])
            {
                xs.push_back(roomGrid[s][i][0]);
                ys.push_back(roomGrid[s][i][1]);
                zs.push_back(roomGrid[s][i][2]);
                box.colourSum += cv::Vec3d(gridColour[s][i]);
            }
            box.lo = cv::Vec3f(Percentile(xs, 0.02), room.floorY, Percentile(zs, 0.02));
            box.hi = cv::Vec3f(Percentile(xs, 0.98), Percentile(ys, 0.98), Percentile(zs, 0.98));
            // New: wider than any furniture is a wall piece no plane was found for, not a block.
            if (box.hi[0] - box.lo[0] <= MaxBlockMeters && box.hi[2] - box.lo[2] <= MaxBlockMeters) boxes.push_back(box);
        }
    }
    // The same couch seen from two spots gives two overlapping boxes: join any that touch.
    for (bool joined = true; joined; )
    {
        joined = false;
        for (size_t i = 0; i < boxes.size() && !joined; i++)
            for (size_t j = i + 1; j < boxes.size() && !joined; j++)
            {
                bool touch = true;
                for (int a = 0; a < 3; a++)
                    touch = touch && boxes[i].lo[a] <= boxes[j].hi[a] + 0.02f && boxes[j].lo[a] <= boxes[i].hi[a] + 0.02f;
                // New: and only while the result stays furniture-sized; touching boxes chained through a
                // whole flat into one box that hid everything inside it.
                for (int a : { 0, 2 })
                    touch = touch && std::max(boxes[i].hi[a], boxes[j].hi[a]) - std::min(boxes[i].lo[a], boxes[j].lo[a]) <= MaxBlockMeters;
                if (!touch) continue;
                for (int a = 0; a < 3; a++)
                {
                    boxes[i].lo[a] = std::min(boxes[i].lo[a], boxes[j].lo[a]);
                    boxes[i].hi[a] = std::max(boxes[i].hi[a], boxes[j].hi[a]);
                }
                boxes[i].colourSum += boxes[j].colourSum;
                boxes[i].count += boxes[j].count;
                boxes.erase(boxes.begin() + j);
                joined = true;
            }
    }
    for (const Box& box : boxes)
    {
        AddBox(room.mesh, "furniture_blocks", box.lo, box.hi, averageColour(box.colourSum, box.count));
        room.blocks.emplace_back(box.lo, box.hi);
    }
    return room;
}

// As in 09c.
bool SaveObj(const std::string& path, const Mesh& mesh)
{
    std::ofstream file(path);
    file << "# Spotter 13: seen_room = every scan merged, clean_* and furniture_blocks = the tidy (partly guessed) model\n";
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

// ---------------------------------------------------------------------
//  Reading the folders (new)
// ---------------------------------------------------------------------

// Every point with its own colour, as a binary PLY. 08 wrote text, but a room's 5 million points
// would be about 150 MB of it. MeshLab and Blender read both.
bool SavePly(const std::string& path, const std::vector<cv::Point3f>& points, const std::vector<cv::Vec3b>& colours)
{
    std::ofstream file(path, std::ios::binary);
    file << "ply\nformat binary_little_endian 1.0\n"
         << "comment Spotter: every scan's own points. A capture of the room: never commit it.\n"
         << "element vertex " << points.size() << "\n"
         << "property float x\nproperty float y\nproperty float z\n"
         << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (size_t i = 0; i < points.size(); i++)
    {
        file.write((const char*)&points[i], 3 * sizeof(float));
        const unsigned char rgb[3] = { colours[i][2], colours[i][1], colours[i][0] };   // OpenCV keeps BGR
        file.write((const char*)rgb, 3);
    }
    return (bool)file;
}

// A line typed in the console, without the spaces or quotes round it (Explorer's "Copy as path" adds quotes).
std::string Ask(const std::string& question)
{
    std::printf("%s", question.c_str());
    std::string answer;
    if (!std::getline(std::cin, answer)) return "cancel";
    while (!answer.empty() && (answer.back() == ' ' || answer.back() == '"' || answer.back() == '\r')) answer.pop_back();
    size_t start = answer.find_first_not_of(" \"");
    return start == std::string::npos ? "" : answer.substr(start);
}

// A file of "key values..." lines (room.txt, pose.txt) as key -> the rest of the line. # = a comment.
std::map<std::string, std::string> ReadKeys(const fs::path& path)
{
    std::map<std::string, std::string> keys;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        size_t space = line.find(' ');
        keys[line.substr(0, space)] = space == std::string::npos ? "" : line.substr(space + 1);
    }
    return keys;
}

std::vector<float> Numbers(const std::string& text)
{
    std::vector<float> numbers;
    std::istringstream in(text);
    for (float v; in >> v; ) numbers.push_back(v);
    return numbers;
}

// One saved room, as listed.
struct RoomFolder
{
    fs::path folder;
    std::string program, saved;
    int scans = 0, excluded = 0;
    double megabytes = 0;
};

// Step 1: every folder holding a room.txt, up to SearchDepth folders down (a room's own scan_
// folders aren't searched). The folder itself counts too.
std::vector<RoomFolder> FindRooms(const fs::path& root)
{
    std::vector<fs::path> found;
    std::error_code error;
    if (fs::exists(root / "room.txt", error)) found.push_back(root);
    else if (fs::is_directory(root, error))
        for (auto it = fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied, error);
             it != fs::recursive_directory_iterator(); it.increment(error))
        {
            if (error) break;
            if (!it->is_directory(error)) continue;
            if (fs::exists(it->path() / "room.txt", error)) { found.push_back(it->path()); it.disable_recursion_pending(); }
            else if (it.depth() + 1 >= SearchDepth) it.disable_recursion_pending();
        }
    std::sort(found.begin(), found.end());
    std::vector<RoomFolder> rooms;
    for (const fs::path& folder : found)
    {
        auto keys = ReadKeys(folder / "room.txt");
        RoomFolder room{ folder, keys["program"].substr(0, 3), keys["saved"] };
        uintmax_t bytes = 0;
        for (const auto& entry : fs::recursive_directory_iterator(folder, error))
        {
            if (entry.is_regular_file(error)) bytes += entry.file_size(error);
            if (entry.is_directory(error) && entry.path().filename().string().rfind("scan_", 0) == 0)
            {
                room.scans++;
                room.excluded += ReadKeys(entry.path() / "pose.txt")["excluded"] == "1";
            }
        }
        room.megabytes = bytes / 1e6;
        rooms.push_back(room);
    }
    return rooms;
}

// "1-12,15" (or "all", or nothing = all) as numbers from 1 to count. False if it can't be read.
bool ParseList(const std::string& text, int count, std::vector<int>& numbers)
{
    std::set<int> chosen;
    std::string list = text;
    std::replace(list.begin(), list.end(), ',', ' ');
    std::istringstream in(list);
    std::string token;
    bool any = false;
    while (in >> token)
    {
        any = true;
        if (token == "all") { for (int n = 1; n <= count; n++) chosen.insert(n); continue; }
        size_t dash = token.find('-');
        try
        {
            int first = std::stoi(token.substr(0, dash)), last = dash == std::string::npos ? first : std::stoi(token.substr(dash + 1));
            if (first < 1 || last > count || first > last) return false;
            for (int n = first; n <= last; n++) chosen.insert(n);
        }
        catch (...) { return false; }
    }
    if (!any) for (int n = 1; n <= count; n++) chosen.insert(n);
    numbers.assign(chosen.begin(), chosen.end());
    return !numbers.empty();
}

// The scan folders of a room, in order: scan_001, scan_002, ...
std::vector<fs::path> ScanFolders(const fs::path& room)
{
    std::vector<fs::path> folders;
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(room, error))
        if (entry.is_directory(error) && entry.path().filename().string().rfind("scan_", 0) == 0) folders.push_back(entry.path());
    std::sort(folders.begin(), folders.end());
    return folders;
}

// Step 2: one scan back from its folder: cloud.bin, colour.bmp and pose.txt. False if a file is
// missing or the wrong size.
bool LoadScan(const fs::path& folder, Scan& scan, std::string& problem)
{
    scan.folder = folder;
    scan.number = std::atoi(folder.filename().string().substr(5).c_str());
    scan.cloud.resize(Width * Height);
    std::ifstream cloud(folder / "cloud.bin", std::ios::binary);
    cloud.read((char*)scan.cloud.data(), scan.cloud.size() * sizeof(cv::Point3f));
    if (!cloud || cloud.gcount() != (std::streamsize)(scan.cloud.size() * sizeof(cv::Point3f))) { problem = "cloud.bin missing or short"; return false; }
    scan.picture = cv::imread((folder / "colour.bmp").string(), cv::IMREAD_COLOR);
    if (scan.picture.size() != cv::Size(Width, Height)) { problem = "colour.bmp missing or the wrong size"; return false; }
    auto keys = ReadKeys(folder / "pose.txt");
    std::vector<float> turn = Numbers(keys["turn"]), shift = Numbers(keys["shift"]);
    if (turn.size() != 9 || shift.size() != 3) { problem = "pose.txt missing or unreadable"; return false; }
    for (int i = 0; i < 9; i++) scan.pose.turn(i / 3, i % 3) = turn[i];
    scan.pose.shift = cv::Point3f(shift[0], shift[1], shift[2]);
    scan.excluded = keys["excluded"] == "1";
    scan.suspicious = keys["suspicious"] == "1";
    scan.stats = keys["stats"];
    for (int row = 0; row < Height; row += BoundsStep)
        for (int column = 0; column < Width; column += BoundsStep)
        {
            const cv::Point3f& p = scan.cloud[row * Width + column];
            if (!std::isnan(p.z)) scan.sparse.push_back(p);
        }
    return true;
}

// X in the viewer: the scan's new excluded flag, written back into its pose.txt (the other lines stay).
void SaveExcluded(const Scan& scan)
{
    std::ifstream in(scan.folder / "pose.txt");
    std::string text, line;
    while (std::getline(in, line))
        text += (line.rfind("excluded ", 0) == 0 ? "excluded " + std::to_string((int)scan.excluded) : line) + "\n";
    in.close();
    std::ofstream(scan.folder / "pose.txt") << text;
}

// 13d's landmarks.txt: x y z first on each line (the rest is for matching, not needed here).
std::vector<Landmark> LoadLandmarks(const fs::path& room)
{
    std::vector<Landmark> landmarks;
    std::ifstream file(room / "landmarks.txt");
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        cv::Point3f p;
        if (in >> p.x >> p.y >> p.z) landmarks.push_back({ p });
    }
    return landmarks;
}

// ---------------------------------------------------------------------
//  Building (new): 13c's E and K, from loaded scans, into the room's folder
// ---------------------------------------------------------------------

std::string Stamp()
{
    char stamp[32];
    std::time_t now = std::time(nullptr);
    std::strftime(stamp, sizeof stamp, "%Y%m%d_%H%M%S", std::localtime(&now));
    return stamp;
}

// Step 3a: as 13c's Export, into the room's folder, plus 13d's far landmarks if it has any.
void BuildObj(const fs::path& room, const std::vector<Scan>& allScans, const Lens& lens, const std::vector<Landmark>& landmarks)
{
    std::vector<Scan> scans;
    for (const Scan& s : allScans) if (!s.excluded) scans.push_back(s);
    if (scans.empty()) { std::printf("  .obj: every loaded scan is excluded - nothing to merge.\n"); return; }
    std::printf("  Merging %zu scans...\n", scans.size());
    auto start = std::chrono::steady_clock::now();
    std::mt19937 random(1);
    float voxel;
    Mesh seen = FuseScans(scans, lens, voxel);
    Room built = BuildRoom(scans, seen, random);
    for (const Landmark& l : landmarks)   // as in 13d: 10 cm magenta boxes in their own group
    {
        cv::Point3f q = built.firstCamera * l.point;
        cv::Vec3f centre(q.x, q.y, q.z), half(0.05f, 0.05f, 0.05f);
        AddBox(built.mesh, "far_landmarks", centre - half, centre + half, cv::Vec3b(255, 0, 255));
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::string path = (room / ("room_scan_" + Stamp() + ".obj")).string();
    if (!SaveObj(path, built.mesh)) { std::printf("  Could not write %s\n", path.c_str()); return; }
    // How scan 1's camera sat in the room, for presentation/Demo4_Room_Twin (as in 13c).
    const cv::Matx33f& t = built.firstCamera;
    std::ofstream(path, std::ios::app) << cv::format("# first camera turn: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                                                     t(0, 0), t(0, 1), t(0, 2), t(1, 0), t(1, 1), t(1, 2), t(2, 0), t(2, 1), t(2, 2));
    std::printf("  Saved %s (%.1f MB, %.1f s). A capture of the room: never commit it.\n", path.c_str(),
                fs::file_size(path) / 1e6, seconds);
    std::printf("    merged mesh: %zu triangles, %.1f cm cubes%s\n", built.mesh.groups["seen_room"].size(), voxel * 100,
                landmarks.empty() ? "" : cv::format("; %zu far landmarks", landmarks.size()).c_str());
    std::printf("    up from %s; %.2f m high (floor %s, ceiling %s); %zu walls, %zu furniture blocks\n", built.upFrom.c_str(),
                built.ceilingY - built.floorY, built.floorSeen ? "seen" : "guessed", built.ceilingSeen ? "seen" : "guessed",
                built.walls.size(), built.blocks.size());
}

// Step 3b: as K's points.ply: every included scan's own points, levelled as the .obj is.
void BuildPly(const fs::path& room, const std::vector<Scan>& allScans)
{
    std::vector<Scan> scans;
    for (const Scan& s : allScans) if (!s.excluded) scans.push_back(s);
    if (scans.empty()) { std::printf("  .ply: every loaded scan is excluded - nothing to write.\n"); return; }
    std::mt19937 random(1);
    const cv::Matx33f t = BuildRoom(scans, Mesh(), random).firstCamera;
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colours;
    for (const Scan& scan : scans)
        for (int i = 0; i < Width * Height; i++)
        {
            const cv::Point3f& p = scan.cloud[i];
            if (std::isnan(p.z)) continue;
            points.push_back(t * scan.pose(p));
            colours.push_back(scan.picture.at<cv::Vec3b>(i / Width, i % Width));
        }
    std::string path = (room / ("points_" + Stamp() + ".ply")).string();
    if (!SavePly(path, points, colours)) { std::printf("  Could not write %s\n", path.c_str()); return; }
    std::printf("  Saved %s: %zu points from %zu scans (%.0f MB). A capture of the room: never commit it.\n", path.c_str(),
                points.size(), scans.size(), fs::file_size(path) / 1e6);
}

// ---------------------------------------------------------------------
//  Viewing one room (new, from 13c's viewer window)
// ---------------------------------------------------------------------

// Returns false when Q was pressed (stop everything), true for N or the window closed (next room).
bool View(const RoomFolder& room, std::vector<Scan>& scans, const Lens& lens, const std::vector<Landmark>& landmarks)
{
    std::vector<std::vector<cv::Point3f>> shownPoints(scans.size());
    std::vector<std::vector<cv::Vec3b>> shownColours(scans.size());
    for (size_t s = 0; s < scans.size(); s++)
        for (int row = 0; row < Height; row += ShowStep)
            for (int column = 0; column < Width; column += ShowStep)
            {
                const cv::Point3f& p = scans[s].cloud[row * Width + column];
                if (std::isnan(p.z)) continue;
                shownPoints[s].push_back(scans[s].pose(p));
                shownColours[s].push_back(scans[s].picture.at<cv::Vec3b>(row, column));
            }
    const std::string name = room.folder.filename().string();
    const std::string window = "13z room rebuild - " + name + "  (drag/WASD turn, wheel zoom, click a camera to select)";
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(window, OnMouse);
    yawDegrees = pitchDegrees = 0;
    viewDistance = PivotMeters;
    bool colourByScan = false;
    int selected = -1;
    std::string message = "E build .obj   K build .ply   X exclude   N next room   Q quit";
    std::printf("  Viewer open: %s\n", message.c_str());
    while (true)
    {
        int key = cv::waitKey(30);
        if (key == 'q') { cv::destroyWindow(window); return false; }
        if (key == 'n' || cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1) break;
        if (key == 27) selected = -1;
        if (key == 'a') yawDegrees -= TurnStepDegrees;
        if (key == 'd') yawDegrees += TurnStepDegrees;
        if (key == 'w') pitchDegrees += TurnStepDegrees;
        if (key == 's') pitchDegrees -= TurnStepDegrees;
        if (key == '=' || key == '+') viewDistance -= ZoomStepMeters;
        if (key == '-') viewDistance += ZoomStepMeters;
        if (key == 'r') { yawDegrees = 0; pitchDegrees = 0; viewDistance = PivotMeters; }
        if (key == 'c') colourByScan = !colourByScan;
        pitchDegrees = std::clamp(pitchDegrees, -89.0f, 89.0f);
        viewDistance = std::max(viewDistance, 0.1f);
        UpdateView();

        if (clicked)   // as in 13c: a click selects the nearest camera circle
        {
            clicked = false;
            selected = -1;
            double nearest = 15;
            for (size_t s = 0; s < scans.size(); s++)
            {
                cv::Point pixel;
                float depth;
                if (!Project(scans[s].pose.shift, pixel, depth)) continue;
                double d = cv::norm(pixel - cv::Point(clickX, clickY));
                if (d < nearest) { nearest = d; selected = (int)s; }
            }
            if (selected >= 0)
                std::printf("  Selected scan %d%s: %s\n", scans[selected].number, scans[selected].excluded ? " (excluded)" : "",
                            scans[selected].stats.c_str());
        }
        if (key == 'x' && selected >= 0)
        {
            scans[selected].excluded = !scans[selected].excluded;
            SaveExcluded(scans[selected]);
            message = cv::format("Scan %d %s (saved to its pose.txt).", scans[selected].number,
                                 scans[selected].excluded ? "excluded: left out of E and K" : "included again");
            std::printf("  %s\n", message.c_str());
        }
        if (key == 'e') { BuildObj(room.folder, scans, lens, landmarks); message = "Built the .obj (see the console)."; }
        if (key == 'k') { BuildPly(room.folder, scans); message = "Built the .ply (see the console)."; }

        // Output: 13c's room so far, with the cameras and (13d) the far landmarks.
        std::vector<char> excluded(scans.size());
        for (size_t s = 0; s < scans.size(); s++) excluded[s] = scans[s].excluded;
        cv::Mat image = Render(shownPoints, shownColours, excluded, colourByScan);
        for (size_t s = 0; s < scans.size(); s++)
        {
            cv::Point pixel;
            float depth;
            if (!Project(scans[s].pose.shift, pixel, depth)) continue;
            cv::Scalar colour = (int)s == selected ? cv::Scalar(255, 255, 255) : scans[s].excluded ? cv::Scalar(110, 110, 110)
                              : scans[s].suspicious ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 255);
            int thickness = (int)s == selected ? 2 : 1;
            cv::circle(image, pixel, 6, colour, thickness + 1);
            for (cv::Point2f corner : { cv::Point2f(0, 0), cv::Point2f(Width, 0), cv::Point2f(0, Height), cv::Point2f(Width, Height) })
            {
                cv::Point3f ahead((corner.x - lens.cx) / lens.fx * ScanArrowMeters, (corner.y - lens.cy) / lens.fy * ScanArrowMeters,
                                  ScanArrowMeters);
                cv::Point tip;
                float tipDepth;
                if (Project(scans[s].pose(ahead), tip, tipDepth)) cv::arrowedLine(image, pixel, tip, colour, thickness, cv::LINE_AA, 0, 0.15);
            }
            cv::putText(image, std::to_string(scans[s].number), pixel + cv::Point(8, -8), cv::FONT_HERSHEY_SIMPLEX, 0.5, colour, 1);
        }
        for (const Landmark& l : landmarks)
        {
            cv::Point pixel;
            float depth;
            if (Project(l.point, pixel, depth)) cv::circle(image, pixel, 3, cv::Scalar(255, 0, 255), cv::FILLED);
        }
        int excludedCount = 0;
        for (const Scan& s : scans) excludedCount += s.excluded;
        cv::putText(image, cv::format("%s: %zu scans loaded (%d excluded)", name.c_str(), scans.size(), excludedCount), cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
        cv::putText(image, message, cv::Point(10, ViewHeight - 15), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 1);
        cv::imshow(window, image);
    }
    cv::destroyWindow(window);
    return true;
}

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // -----------------------------------------------------------------
    //  Setup: find the rooms and choose (step 1 and 2)
    // -----------------------------------------------------------------

    fs::path searchIn = fs::absolute(argc > 1 ? fs::path(argv[1]) : fs::path(SearchFolder));
    std::vector<RoomFolder> rooms;
    std::vector<int> picks;
    while (true)
    {
        rooms = FindRooms(searchIn);
        std::printf("\nSaved rooms in %s:\n", searchIn.string().c_str());
        if (rooms.empty()) std::printf("  none (13e, or K in 13c or 13d, saves them; give another folder below)\n");
        for (size_t r = 0; r < rooms.size(); r++)
            std::printf("  %2zu  %-28s %s  %-16s  %3d scans (%d excluded)  %5.0f MB\n", r + 1,
                        rooms[r].folder.filename().string().c_str(), rooms[r].program.c_str(), rooms[r].saved.c_str(),
                        rooms[r].scans, rooms[r].excluded, rooms[r].megabytes);
        std::string answer = Ask(rooms.empty() ? "Folder to look in (q quits): "
                                               : "Which rooms? (2   1,3   1-4   Enter = all; or a folder to look in; q quits): ");
        if (answer == "q" || answer == "cancel") return 0;
        std::error_code error;
        if (!answer.empty() && fs::is_directory(answer, error)) { searchIn = fs::absolute(answer); continue; }
        if (rooms.empty()) { std::printf("Not a folder: %s\n", answer.c_str()); continue; }
        if (ParseList(answer, (int)rooms.size(), picks)) break;
        std::printf("Didn't understand \"%s\": numbers from 1 to %zu, like 1,3 or 2-4.\n", answer.c_str(), rooms.size());
    }
    std::string what;
    while (true)
    {
        what = Ask("What to do? 1 view (build from the viewer)  2 build .obj  3 build .ply  4 build both  [Enter = 1]: ");
        if (what.empty()) what = "1";
        if (what == "1" || what == "2" || what == "3" || what == "4") break;
        if (what == "q") return 0;
    }

    // -----------------------------------------------------------------
    //  Main operation: each room in turn, load then view or build (step 2 and 3)
    // -----------------------------------------------------------------

    for (int pick : picks)
    {
        const RoomFolder& room = rooms[pick - 1];
        const std::string name = room.folder.filename().string();
        auto keys = ReadKeys(room.folder / "room.txt");
        std::vector<float> ir = Numbers(keys["ir_params"]);
        if (keys["format"] != "1" || ir.size() < 4)
        {
            std::printf("\n%s: room.txt is missing or from a newer format - skipped.\n", name.c_str());
            continue;
        }
        const Lens lens{ ir[0], ir[1], ir[2], ir[3] };
        std::vector<fs::path> folders = ScanFolders(room.folder);
        std::printf("\n=== %s (%s, %s): %zu scans ===\n", name.c_str(), keys["program"].c_str(), keys["saved"].c_str(), folders.size());
        std::vector<int> chosen;
        while (true)
        {
            std::string answer = Ask(cv::format("Which scans to load? (1-12,15  Enter = all; skip = next room): "));
            if (answer == "skip") break;
            if (ParseList(answer, (int)folders.size(), chosen)) break;
            std::printf("Numbers from 1 to %zu, please.\n", folders.size());
            chosen.clear();
        }
        if (chosen.empty()) continue;

        std::vector<Scan> scans;
        for (int n : chosen)
        {
            Scan scan;
            std::string problem;
            if (LoadScan(folders[n - 1], scan, problem)) scans.push_back(std::move(scan));
            else std::printf("  %s skipped: %s\n", folders[n - 1].filename().string().c_str(), problem.c_str());
            std::printf("  loaded %zu of %zu\r", scans.size(), chosen.size());
        }
        std::vector<Landmark> landmarks = LoadLandmarks(room.folder);
        int excludedCount = 0;
        for (const Scan& s : scans) excludedCount += s.excluded;
        std::printf("  loaded %zu scans (%d of them excluded)%s\n", scans.size(), excludedCount,
                    landmarks.empty() ? "" : cv::format(", %zu far landmarks", landmarks.size()).c_str());
        if (scans.empty()) continue;

        // -------------------------------------------------------------
        //  Output: the viewer, or the files
        // -------------------------------------------------------------

        if (what == "1" && !View(room, scans, lens, landmarks)) break;
        if (what == "2" || what == "4") BuildObj(room.folder, scans, lens, landmarks);
        if (what == "3" || what == "4") BuildPly(room.folder, scans);
    }
    std::printf("\nDone.\n");
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Build the same room at VoxelMeters = 0.02 (13c's) and 0.01. How much sharper is the .obj,
//     how much bigger the file, how much longer the merge? Watch MaxVoxels: a big room at 1 cm
//     may hit it and coarsen by itself (the console prints the cube size it really used).
//  2. Load only the first half of a room's scans (1-10), then only the second half (11-20), and
//     build both. Where do the two .obj files overlap, and where does each have holes?
//  3. In the viewer, press C and look for a scan that shows as a shifted copy. Exclude it (X),
//     build again (E): are the double walls gone?
//  4. Open K's points.ply and a 13z .obj of the same room side by side in MeshLab. Which keeps
//     the small details (text, picture frames), which the clean shapes?
//  5. Set TruncationVoxels = 1, then 6. Thin truncation: sharp, but holes where scans disagree by
//     more than a cube. Thick: closed, but thin things (a door) melt into their surroundings.
