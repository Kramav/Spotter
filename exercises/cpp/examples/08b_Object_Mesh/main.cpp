// =====================================================================
//  08b · Object Mesh
//
//  Example 08 turned every depth pixel into a 3D point. This example
//  keeps only the points of ONE detected object and joins them into
//  triangles: a 3D model you can open in Blender or MeshLab.
//
//    1. Click a YOLO box (02). Keep the object and the camera still.
//       No box (YOLO missed it, or named it wrong)? Click the object
//       anyway: it grows out from the clicked point instead.
//    2. Average 30 depth frames (07, spotter::DepthAverager): less noise.
//    3. Cut the object out: inside the box, near the box's middle depth
//       (02), above the surface it stands on (09's RANSAC), one blob.
//       Clicked without a box: near the click, above the surface, and
//       joined to the click without a depth jump (a flood fill).
//    4. Mesh: the depth picture is already a grid. Each 2x2 square of
//       object pixels becomes two triangles, unless its corners jump in
//       depth (that's an edge, not a surface).
//    5. The camera only sees the FRONT. The back is GUESSED (mirrored or
//       flat), coloured grey and kept in its own group, so what was seen
//       and what was made up never mix.
//    6. Save object_<class>_<time>.obj in the folder you ran from.
//
//  Window: colour (half size) with YOLO's boxes. After a capture, the
//          pixels that went into the mesh are tinted green.
//  Keys:   click an object  capture it     C  clear the tint     Q or Esc  quit
//
//  Privacy: an .obj is a 3D capture of part of the room. *.obj is
//  gitignored: never commit one. The person class is refused: a coloured
//  3D scan of a person is an image of a person. A scan made without a box
//  is refused if it lies mostly inside a person's box.
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/geometry.hpp"
#include "spotter/mesh.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/stats.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "click an object  capture it     C  clear the tint     Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

const int AverageFrames = 30;             // depth frames averaged (07). 1 = a single noisy frame.

// Keep pixels within this distance (metres) of the box's middle depth. It must cover the object's
// own depth: 0.3 fits a chair, not a couch. Too big and the wall behind comes along.
const float DepthBandMeters = 0.3f;

// Clicked without a box: the object grows out from the click, at most this far (metres).
const float ClickReachMeters = 0.5f;
const int ClickSnapPixels = 30;           // the click snaps to the nearest depth reading this close (colour pixels)
const double PersonShare = 0.3;           // refused if this share of it lies inside person boxes (07d)

// The surface the object stands on (a table, the floor), found by RANSAC as in 09.
const bool RemoveSupportPlane = true;
const float PlaneMarginMeters = 0.01f;    // "on the plane" = within this distance
const int RansacTries = 200;
const int NeighbourPixels = 10;           // RANSAC's 3 points come from this close together
const float MaxPlaneSlopeDegrees = 45;    // steeper is a wall, not something to stand on
const double MinPlaneShare = 0.10;        // a plane needs at least this share of the points...
const double MaxBelowShare = 0.10;        // ...and at most this share below it (a chair seat has legs below it)

// No triangle across a depth jump bigger than this (mm): that's an edge, not a surface.
const float MaxJumpMm = 20;

// The back, which the camera can't see. Mirror = the front copied behind, assuming the object is
// as deep as it is wide. Flat = a flat back at the front's farthest point. None = the front only.
enum class Back { None, Flat, Mirror };
const Back BackGuess = Back::Mirror;

const std::string Window = "08b object mesh  (click an object to capture, C clear, Q quit)";

//spotter::YoloDetector::Options YoloSettings(){
//
//@brief - the detector's options: yolo26n.onnx and its names file from the repo, minimum confidence 0.40. O(1).
spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = 0.40f;
    return options;
}

//spotter::KinectSource::Options KinectSettings(){
//
//@brief - the Kinect's options: the defaults with the colour exposure capped (30 fps in a dim room, 05c). O(1).
spotter::KinectSource::Options KinectSettings()
{
    spotter::KinectSource::Options options;
    options.capExposure = true;   // 30 fps in a dim room (05c)
    return options;
}

// ---------------------------------------------------------------------
//  The new idea: cut one object out, mesh it, guess its back
// ---------------------------------------------------------------------

const int Width = 512, Height = 424;

// What was clicked, and what is being captured.
struct Capture
{
    cv::Point click{ -1, -1 };          // in colour pixels; set by the mouse
    spotter::Detection chosen;          // the box, if a box was clicked
    int seed = -1;                      // clicked without a box: the depth pixel clicked
    std::vector<spotter::Detection> detections;   // seen when it was clicked (for the person check)
};

//void OnMouse(int event, int x, int y, int flags, void* capture){
//
//@param1 - event: OpenCV's mouse event
//@param2 - x: the mouse's column in the half-size window
//@param3 - y: its row
//@param4 - flags: OpenCV's flags (unused)
//@param5 - capture: the Capture the click is stored in
//
//@brief - OpenCV's mouse callback: a left click stores its spot doubled (the window is half size), i.e. in colour
//         pixels. O(1).
void OnMouse(int event, int x, int y, int, void* capture)
{
    if (event == cv::EVENT_LBUTTONDOWN) ((Capture*)capture)->click = cv::Point(x * 2, y * 2);   // the window is half size
}

//bool FindSupport(const cv::Mat& mask, const std::vector<cv::Point3f>& cloud, std::mt19937& random, spotter::Plane& support){
//
//@param1 - mask: the object's pixels
//@param2 - cloud: a 3D point per depth pixel
//@param3 - random: the random number generator
//@param4 - support: set to the plane it stands on
//
//@brief - RANSAC as in 09, over the object's pixels only: the level-ish plane with the most points ON it (within
//         PlaneMarginMeters) and few BELOW it (under MaxBelowShare). That's the table or floor it stands on. True
//         if at least MinPlaneShare of the points are on it; then refitted to them. O(RansacTries x pixels).
bool FindSupport(const cv::Mat& mask, const std::vector<cv::Point3f>& cloud, std::mt19937& random, spotter::Plane& support)
{
    std::vector<int> kept;
    for (int i = 0; i < Width * Height; i++) if (mask.data[i]) kept.push_back(i);
    if (kept.size() < 3) return false;
    const float minUpness = std::cos(MaxPlaneSlopeDegrees * (float)CV_PI / 180);
    std::uniform_int_distribution<size_t> pick(0, kept.size() - 1);
    std::uniform_int_distribution<int> offset(-NeighbourPixels, NeighbourPixels);
    int bestCount = 0;
    for (int t = 0; t < RansacTries; t++)
    {
        int a = kept[pick(random)], row = a / Width, column = a % Width;
        int bRow = row + offset(random), bColumn = column + offset(random), cRow = row + offset(random), cColumn = column + offset(random);
        if (std::min({ bRow, bColumn, cRow, cColumn }) < 0 || std::max(bRow, cRow) >= Height || std::max(bColumn, cColumn) >= Width) continue;
        int b = bRow * Width + bColumn, c = cRow * Width + cColumn;
        spotter::Plane plane;
        if (!mask.data[b] || !mask.data[c] || !spotter::PlaneThrough(cloud[a], cloud[b], cloud[c], plane)) continue;
        if (plane.normal.dot(cv::Point3f(0, -1, 0)) < minUpness) continue;   // camera "up" is -Y
        int on = 0, below = 0;
        for (int k : kept)
        {
            float height = plane.DistanceTo(cloud[k]);
            if (std::abs(height) < PlaneMarginMeters) on++;
            else if (height < 0) below++;
        }
        if (below > MaxBelowShare * kept.size()) continue;
        if (on > bestCount) { bestCount = on; support = plane; }
    }
    if (bestCount < MinPlaneShare * kept.size()) return false;
    std::vector<cv::Point3f> inliers;
    for (int k : kept) if (std::abs(support.DistanceTo(cloud[k])) < PlaneMarginMeters) inliers.push_back(cloud[k]);
    support = spotter::FitPlane(inliers);
    return true;
}

//void RemoveSupport(cv::Mat& mask, const std::vector<cv::Point3f>& cloud, std::mt19937& random){
//
//@param1 - mask: the object's pixels; changed in place
//@param2 - cloud: a 3D point per depth pixel
//@param3 - random: the random number generator
//
//@brief - drops the points on or below the surface the object stands on (FindSupport), if there is one and
//         RemoveSupportPlane is on; prints how many are left. O(RansacTries x pixels).
void RemoveSupport(cv::Mat& mask, const std::vector<cv::Point3f>& cloud, std::mt19937& random)
{
    spotter::Plane support;
    if (!RemoveSupportPlane) return;
    if (!FindSupport(mask, cloud, random, support)) { std::printf("  no surface found under it (none removed)\n"); return; }
    for (int i = 0; i < Width * Height; i++) if (mask.data[i] && support.DistanceTo(cloud[i]) < PlaneMarginMeters) mask.data[i] = 0;
    std::printf("  above the surface it stands on: %d\n", cv::countNonZero(mask));
}

//cv::Mat CutOut(const cv::Rect2d& box, const std::vector<cv::Point3f>& cloud, const spotter::RgbdFrame& frame, std::mt19937& random){
//
//@param1 - box: the clicked box on the colour picture
//@param2 - cloud: a 3D point per depth pixel (the averaged depth)
//@param3 - frame: a frame with colourIndex
//@param4 - random: the random number generator
//
//@brief - the object's depth pixels: in the box, within DepthBandMeters of the middle half's median depth, above
//         its support (RemoveSupport), and only the biggest connected blob. Prints how many pixels each step
//         leaves, so you can see which step does what. O(depth pixels + RemoveSupport).
cv::Mat CutOut(const cv::Rect2d& box, const std::vector<cv::Point3f>& cloud, const spotter::RgbdFrame& frame, std::mt19937& random)
{
    cv::Mat mask = cv::Mat::zeros(Height, Width, CV_8U);
    cv::Rect2d centre(box.x + box.width / 4, box.y + box.height / 4, box.width / 2, box.height / 2);
    std::vector<float> centreZ;
    for (int i = 0; i < Width * Height; i++)
    {
        int c = frame.colourIndex[i];
        if (c < 0 || std::isnan(cloud[i].z)) continue;
        cv::Point2d pixel(c % frame.colour.cols, c / frame.colour.cols);
        if (!box.contains(pixel)) continue;
        mask.data[i] = 255;
        if (centre.contains(pixel)) centreZ.push_back(cloud[i].z);
    }
    std::printf("  in the box: %d pixels\n", cv::countNonZero(mask));
    if (centreZ.empty()) return mask * 0;
    float middle = spotter::Median(centreZ);
    for (int i = 0; i < Width * Height; i++) if (mask.data[i] && std::abs(cloud[i].z - middle) > DepthBandMeters) mask.data[i] = 0;
    std::printf("  within %.2f m of its middle depth (%.2f m): %d\n", DepthBandMeters, middle, cv::countNonZero(mask));
    RemoveSupport(mask, cloud, random);

    // Keep only the biggest connected blob (07c): stray bits of other things go.
    cv::Mat labels, stats, centroids;
    int blobs = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    int biggest = 0;
    for (int label = 1; label < blobs; label++)
        if (biggest == 0 || stats.at<int>(label, cv::CC_STAT_AREA) > stats.at<int>(biggest, cv::CC_STAT_AREA)) biggest = label;
    if (biggest == 0) return mask;
    std::printf("  biggest blob: %d\n", stats.at<int>(biggest, cv::CC_STAT_AREA));
    return labels == biggest;
}

//cv::Mat CutOutAround(int seed, const std::vector<cv::Point3f>& cloud, std::mt19937& random){
//
//@param1 - seed: the clicked depth pixel's index
//@param2 - cloud: a 3D point per depth pixel
//@param3 - random: the random number generator
//
//@brief - clicked without a box: grows the object from the clicked depth pixel. The points within ClickReachMeters
//         of the click, minus the surface it stands on, then a flood fill from the click that only steps to a
//         neighbour less than MaxJumpMm away in depth, so it stops at the object's edges. Empty (with a message)
//         if the click has no depth or is on the surface itself. O(depth pixels + RemoveSupport).
cv::Mat CutOutAround(int seed, const std::vector<cv::Point3f>& cloud, std::mt19937& random)
{
    cv::Mat mask = cv::Mat::zeros(Height, Width, CV_8U);
    if (std::isnan(cloud[seed].z)) { std::printf("  no steady depth reading at the click\n"); return mask; }
    for (int i = 0; i < Width * Height; i++)
        if (!std::isnan(cloud[i].z) && cv::norm(cloud[i] - cloud[seed]) < ClickReachMeters) mask.data[i] = 255;
    std::printf("  within %.2f m of the click (%.2f m away): %d pixels\n", ClickReachMeters, cloud[seed].z, cv::countNonZero(mask));
    RemoveSupport(mask, cloud, random);
    if (!mask.data[seed]) { std::printf("  the click was on the surface itself: click the object\n"); return mask * 0; }

    // floodFill's mask is one pixel bigger all round; it never fills where the mask isn't 0, and
    // with FLOODFILL_MASK_ONLY it marks what it fills (here with 128) instead of painting.
    cv::Mat depthMm(Height, Width, CV_32F), fill(Height + 2, Width + 2, CV_8U, cv::Scalar(255));
    for (int i = 0; i < Width * Height; i++) depthMm.at<float>(i) = mask.data[i] ? cloud[i].z * 1000 : 0;
    fill(cv::Rect(1, 1, Width, Height)).setTo(0, mask);
    cv::floodFill(depthMm, fill, cv::Point(seed % Width, seed / Width), 0, nullptr, MaxJumpMm, MaxJumpMm, 8 | cv::FLOODFILL_MASK_ONLY | (128 << 8));
    mask = fill(cv::Rect(1, 1, Width, Height)) == 128;
    std::printf("  joined to the click without a depth jump: %d\n", cv::countNonZero(mask));
    return mask;
}

//int DepthPixelAt(cv::Point click, const spotter::RgbdFrame& frame){
//
//@param1 - click: a spot on the colour picture
//@param2 - frame: a frame with depth and colourIndex
//
//@brief - the depth pixel that lands nearest to a colour pixel (within ClickSnapPixels), or -1. O(depth pixels).
int DepthPixelAt(cv::Point click, const spotter::RgbdFrame& frame)
{
    int best = -1;
    double bestDistance = ClickSnapPixels;
    for (int i = 0; i < Width * Height; i++)
    {
        int c = frame.colourIndex[i];
        if (c < 0 || frame.depthMm.at<float>(i) <= 0) continue;
        double distance = std::hypot(c % frame.colour.cols - click.x, c / frame.colour.cols - click.y);
        if (distance < bestDistance) { bestDistance = distance; best = i; }
    }
    return best;
}

//spotter::Mesh MeshFromMask(const cv::Mat& mask, const std::vector<cv::Point3f>& cloud, const cv::Mat& registered){
//
//@param1 - mask: the object's pixels
//@param2 - cloud: a 3D point per depth pixel
//@param3 - registered: colour per depth pixel
//
//@brief - step 4: a point per object pixel and two triangles per 2 x 2 square of them, unless the corners jump in
//         depth by more than MaxJumpMm; group "seen_front", facing the camera. O(pixels).
spotter::Mesh MeshFromMask(const cv::Mat& mask, const std::vector<cv::Point3f>& cloud, const cv::Mat& registered)
{
    spotter::Mesh mesh;
    std::vector<int> pointOf(Width * Height, -1);
    for (int i = 0; i < Width * Height; i++)
    {
        if (!mask.data[i]) continue;
        pointOf[i] = (int)mesh.points.size();
        mesh.points.push_back(cloud[i]);
        mesh.colours.push_back(registered.at<cv::Vec3b>(i));
    }
    std::vector<cv::Vec3i>& faces = mesh.groups["seen_front"];
    auto AddTriangle = [&](int a, int b, int c)
    {
        if (pointOf[a] < 0 || pointOf[b] < 0 || pointOf[c] < 0) return;
        float nearest = std::min({ cloud[a].z, cloud[b].z, cloud[c].z }), farthest = std::max({ cloud[a].z, cloud[b].z, cloud[c].z });
        if ((farthest - nearest) * 1000 > MaxJumpMm) return;
        faces.emplace_back(pointOf[a], pointOf[b], pointOf[c]);   // this order faces the camera
    };
    for (int row = 0; row + 1 < Height; row++)
        for (int column = 0; column + 1 < Width; column++)
        {
            int topLeft = row * Width + column, topRight = topLeft + 1, bottomLeft = topLeft + Width, bottomRight = bottomLeft + 1;
            AddTriangle(topLeft, bottomLeft, topRight);
            AddTriangle(topRight, bottomLeft, bottomRight);
        }
    return mesh;
}

//void AddGuessedBack(spotter::Mesh& mesh){
//
//@param1 - mesh: the mesh; the guessed back is added to it
//
//@brief - step 5: a grey twin of every front point pushed behind (flat at the farthest depth, or mirrored about
//         the middle as deep as it is wide, never in front of the front), its triangles reversed, plus a rim
//         joining the border edges (an edge whose reverse no triangle uses). Group "guessed_back". ponytail: "as
//         deep as it is wide" is the whole guess; a cylinder fit (docs/ideas.md) would be better, a real back
//         needs more views. O(triangles log triangles).
void AddGuessedBack(spotter::Mesh& mesh)
{
    const std::vector<cv::Vec3i>& front = mesh.groups["seen_front"];
    if (BackGuess == Back::None || front.empty()) return;
    int count = (int)mesh.points.size();
    float left = 1e9f, right = -1e9f, nearest = 1e9f, farthest = -1e9f;
    for (const cv::Point3f& p : mesh.points)
    {
        left = std::min(left, p.x); right = std::max(right, p.x);
        nearest = std::min(nearest, p.z); farthest = std::max(farthest, p.z);
    }
    // ponytail: "as deep as it is wide" is the whole guess. A cylinder fit for cups and bottles is
    // better (docs/ideas.md); a real back needs more views.
    float middle = nearest + (right - left) / 2;
    for (int i = 0; i < count; i++)
    {
        cv::Point3f p = mesh.points[i];
        p.z = BackGuess == Back::Flat ? farthest : std::max(2 * middle - p.z, p.z);   // never in front of the front
        mesh.points.push_back(p);
        mesh.colours.emplace_back(128, 128, 128);
    }
    std::vector<cv::Vec3i>& back = mesh.groups["guessed_back"];
    std::set<std::pair<int, int>> edges;
    for (const cv::Vec3i& f : front)
    {
        edges.insert({ f[0], f[1] }); edges.insert({ f[1], f[2] }); edges.insert({ f[2], f[0] });
        back.emplace_back(f[0] + count, f[2] + count, f[1] + count);   // reversed: facing away from the camera
    }
    // The rim: an edge whose reverse no triangle uses is on the border. Join it to its twin.
    // ponytail: a "pinch" pixel where two parts of the border touch gets its rim edge twice;
    // Blender and MeshLab don't mind, a 3D-print slicer would.
    for (const auto& [a, b] : edges)
    {
        if (edges.count({ b, a })) continue;
        back.emplace_back(b, a, a + count);
        back.emplace_back(b, a + count, b + count);
    }
}

//bool MostlyPerson(const cv::Mat& mask, const Capture& capture, const spotter::RgbdFrame& frame, int person){
//
//@param1 - mask: the object's pixels
//@param2 - capture: the capture, for the detections at the click
//@param3 - frame: a frame with colourIndex
//@param4 - person: the class number of "person"
//
//@brief - a scan made without a box has no class to refuse, so check it isn't mostly a person (07d): true if more
//         than PersonShare of its pixels land inside a person box. O(pixels x detections).
bool MostlyPerson(const cv::Mat& mask, const Capture& capture, const spotter::RgbdFrame& frame, int person)
{
    int inPerson = 0;
    for (int i = 0; i < Width * Height; i++)
    {
        int c = frame.colourIndex[i];
        if (!mask.data[i] || c < 0) continue;
        for (const spotter::Detection& d : capture.detections)
            if (d.classId == person && d.box.contains(cv::Point2d(c % frame.colour.cols, c / frame.colour.cols))) { inPerson++; break; }
    }
    return inPerson > PersonShare * cv::countNonZero(mask);
}

//bool StartCapture(Capture& capture, const spotter::RgbdFrame& frame, const std::vector<spotter::Detection>& detections, const spotter::ClassList& classes, spotter::DepthAverager& averager){
//
//@param1 - capture: the click and what it hit; filled in
//@param2 - frame: this frame
//@param3 - detections: this frame's boxes
//@param4 - classes: the class names
//@param5 - averager: the depth averager; started
//
//@brief - step 1: what did the click hit? The smallest box under it, else the nearest depth reading. A person is
//         refused (privacy); so is a click with neither a box nor a reading. Otherwise remembers the target and
//         starts the averager; true if it started. O(detections + depth pixels).
bool StartCapture(Capture& capture, const spotter::RgbdFrame& frame, const std::vector<spotter::Detection>& detections,
                  const spotter::ClassList& classes, spotter::DepthAverager& averager)
{
    const spotter::Detection* hit = nullptr;
    for (const spotter::Detection& d : detections)
        if (d.box.contains(capture.click) && (!hit || d.box.area() < hit->box.area())) hit = &d;
    capture.seed = hit ? -1 : DepthPixelAt(capture.click, frame);
    capture.click = { -1, -1 };
    if (hit && hit->classId == classes.Find("person")) { std::printf("Not capturing a person (privacy).\n"); return false; }
    if (!hit && capture.seed < 0) { std::printf("No box and no depth reading there.\n"); return false; }
    if (hit) capture.chosen = *hit;
    capture.detections = detections;
    averager.Start();
    std::printf("Capturing %s: hold still...\n", hit ? classes.Name(hit->classId).c_str() : "the clicked object (no box)");
    return true;
}

//std::vector<cv::Point> FinishCapture(const Capture& capture, const spotter::DepthAverager& averager, const spotter::RgbdFrame& frame, const spotter::ClassList& classes, std::mt19937& random){
//
//@param1 - capture: the click and what it hit
//@param2 - averager: the finished depth average
//@param3 - frame: this frame
//@param4 - classes: the class names
//@param5 - random: the random number generator
//
//@brief - steps 3-6, on the averaged depth: the cloud, the cut-out (around the click or in the box), a privacy
//         check, the mesh with its guessed back, then SaveObj to object_<name>_<time>.obj with Y up. Returns the
//         colour pixels to tint on the screen. O(depth pixels + CutOut).
std::vector<cv::Point> FinishCapture(const Capture& capture, const spotter::DepthAverager& averager, const spotter::RgbdFrame& frame,
                                     const spotter::ClassList& classes, std::mt19937& random)
{
    std::vector<cv::Point3f> cloud = frame.depthCamera.Cloud(averager.Mean());
    cv::Mat mask = capture.seed >= 0 ? CutOutAround(capture.seed, cloud, random) : CutOut(capture.chosen.box, cloud, frame, random);
    if (capture.seed >= 0 && MostlyPerson(mask, capture, frame, classes.Find("person"))) { std::printf("Mostly inside a person's box: not saved (privacy).\n"); return {}; }
    spotter::Mesh mesh = MeshFromMask(mask, cloud, frame.registered);
    size_t seenPoints = mesh.points.size(), seenFaces = mesh.groups["seen_front"].size();
    AddGuessedBack(mesh);

    std::vector<cv::Point> tint;
    for (int i = 0; i < Width * Height; i++)
        if (mask.data[i] && frame.colourIndex[i] >= 0) tint.emplace_back(frame.colourIndex[i] % frame.colour.cols, frame.colourIndex[i] / frame.colour.cols);
    if (seenFaces == 0) { std::printf("Nothing left to mesh. Too far, too dark or shiny, or try a bigger DepthBandMeters / ClickReachMeters.\n"); return tint; }

    std::string name = capture.seed >= 0 ? "clicked" : classes.Name(capture.chosen.classId);
    std::replace(name.begin(), name.end(), ' ', '_');   // "potted plant" -> potted_plant
    std::string path = (std::filesystem::current_path() / cv::format("object_%s_%lld.obj", name.c_str(), (long long)std::time(nullptr))).string();
    spotter::TurnToYUp(mesh);
    if (spotter::SaveObj(path, mesh, "Spotter 08b. seen_front = scanned by the Kinect, guessed_back = made up"))
        std::printf("Saved %zu seen points, %zu seen + %zu guessed triangles to %s\n  (a capture of the room: never commit it)\n",
                    seenPoints, seenFaces, mesh.groups["guessed_back"].size(), path.c_str());
    else std::printf("Could not write %s\n", path.c_str());
    return tint;
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

//void Show(const spotter::RgbdFrame& frame, const std::vector<spotter::Detection>& detections, const std::vector<cv::Point>& tint, const spotter::DepthAverager& averager, const spotter::ClassList& classes, const spotter::FpsMeter& fps){
//
//@param1 - frame: this frame
//@param2 - detections: this frame's boxes
//@param3 - tint: the colour pixels of the last object, tinted green
//@param4 - averager: for the hold-still count
//@param5 - classes: the class names
//@param6 - fps: the frame-rate meter
//
//@brief - the half-size colour picture with the last object tinted green, a box and name per detection (person
//         red), "HOLD STILL n / N" or "click an object to capture" and the fps; shown with the key strip.
//         O(pixels).
void Show(const spotter::RgbdFrame& frame, const std::vector<spotter::Detection>& detections, const std::vector<cv::Point>& tint,
          const spotter::DepthAverager& averager, const spotter::ClassList& classes, const spotter::FpsMeter& fps)
{
    cv::Mat display;
    cv::resize(frame.colour, display, frame.colour.size() / 2);
    for (const cv::Point& p : tint)
    {
        cv::Vec3b& pixel = display.at<cv::Vec3b>(p.y / 2, p.x / 2);
        pixel = cv::Vec3b(pixel[0] / 2, (pixel[1] + 255) / 2, pixel[2] / 2);
    }
    for (const spotter::Detection& d : detections)
    {
        cv::Scalar colour = d.classId == classes.Find("person") ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::Rect box(cv::Point((int)d.box.x / 2, (int)d.box.y / 2), cv::Point((int)d.box.br().x / 2, (int)d.box.br().y / 2));
        cv::rectangle(display, box, colour, 2);
        cv::putText(display, classes.Name(d.classId), box.tl() + cv::Point(3, 16), cv::FONT_HERSHEY_SIMPLEX, 0.5, colour, 1);
    }
    std::string status = averager.Running() ? cv::format("HOLD STILL  %d / %d", averager.Added(), averager.Frames()) : "click an object to capture";
    cv::putText(display, status, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    fps.Draw(display);
    cv::imshow(Window, spotter::WithKeys(display, Keys));
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

//int main(){
//
//@brief - the program: 08b · Object Mesh. Example 08 turned every depth pixel into a 3D point. This example keeps
//         only the points of ONE detected object and joins them into triangles: a 3D model you can open in Blender
//         or MeshLab. Setup, main loop and clean-up are laid out in the file's sections. Returns 0 on a normal
//         exit, 1 if it couldn't start.
int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(KinectSettings());
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    Capture capture;
    cv::namedWindow(Window, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(Window, OnMouse, &capture);
    spotter::DepthAverager averager(AverageFrames);
    std::vector<spotter::Detection> detections;
    std::vector<cv::Point> tint;
    std::mt19937 random(1);
    spotter::FpsMeter fps;
    spotter::RgbdFrame frame;
    std::printf("Click an object (its box, or the object itself if it has none). Keep it and the camera still for %d frames.\n", AverageFrames);

    while (kinect.Next(frame))
    {
        if (!averager.Running())
        {
            detections = yolo.Detect(frame.colour);                                              // YOLO waits; paused during a capture
            if (capture.click.x >= 0) StartCapture(capture, frame, detections, yolo.Classes(), averager);   // step 1
        }
        else if (averager.Add(frame.depthMm)) tint = FinishCapture(capture, averager, frame, yolo.Classes(), random);   // steps 2-6

        fps.Tick();
        Show(frame, averager.Running() ? std::vector<spotter::Detection>{} : detections, tint, averager, yolo.Classes(), fps);
        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 'c') tint.clear();
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Open the .obj in Blender (File > Import > Wavefront) or MeshLab. Hide the group
//     guessed_back: what's left is all the camera really saw.
//  2. MaxJumpMm = 5 versus 50: a torn mesh versus webs stretched to the wall behind.
//  3. AverageFrames = 1 versus 30: look at a flat side edge on. How bumpy is one frame?
//  4. RemoveSupportPlane = false on a cup on a table: the tabletop inside the box comes along.
//  5. The same cup at 1 m and at 2 m. How many triangles each? (One depth pixel is about 2.7 mm
//     across at 1 m and 5.4 mm at 2 m.)
//  6. BackGuess = Mirror on a cup, a box and a chair. Where does "as deep as it is wide" go wrong?
//  7. Click an object YOLO has no box for. Then click the same object's box, if it has one: do the
//     two scans match?
//  8. Click the floor or the table itself: the support plane removes it, nothing to scan.
