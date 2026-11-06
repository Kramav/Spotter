// =====================================================================
//  08b · Kinect Object Mesh
//
//  Example 08 turned every depth pixel into a 3D point. This example
//  keeps only the points of ONE detected object and joins them into
//  triangles: a 3D model you can open in Blender or MeshLab.
//
//    1. Click a YOLO box (02). Keep the object and the camera still.
//       No box (YOLO missed it, or named it wrong)? Click the object
//       anyway: it grows out from the clicked point instead.
//    2. Average 30 depth frames (07): less noise, a smoother surface.
//    3. Cut the object out: inside the box, near the box's middle depth
//       (02), above the surface it stands on (09's RANSAC), one blob.
//       Clicked without a box: near the click, above the surface, and
//       joined to the click without a depth jump (a flood fill).
//    4. Mesh: the depth image is already a grid. Each 2x2 square of
//       object pixels becomes two triangles, unless its corners jump in
//       depth (that's an edge, not a surface).
//    5. The camera only sees the FRONT. The back is GUESSED (mirrored or
//       flat), coloured grey and kept in its own group, so what was seen
//       and what was made up never mix. Hide "guessed_back" in Blender
//       or MeshLab to see only the scan.
//    6. Save object_<class>_<time>.obj in the folder you ran from.
//
//  Window: colour (half size) with YOLO's boxes. After a capture, the
//          pixels that went into the mesh are tinted green.
//  Keys:   click an object  capture it     C  clear the tint
//          Q or Esc  quit
//
//  Privacy: an .obj is a 3D capture of part of the room. *.obj is
//  gitignored: never commit one. The person class is refused: a
//  coloured 3D scan of a person is an image of a person. A scan made
//  without a box is refused if it lies mostly inside a person's box,
//  which only works if YOLO saw the person.
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/registration.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// --- Capture ---
// Depth frames averaged per pixel (07). 1 = a single noisy frame: compare the surfaces.
const int AverageFrames = 30;

// The depth cut-off (07b, as in 08). At the 4.5 m default, a TV at 4.47 m lost its far half.
const float MaxDepthMeters = 6.0f;

// --- Cut-out ---
// Keep pixels within this distance (metres) of the box's middle depth. It must cover the object's
// own depth: 0.3 fits a chair, not a couch. Too big and the wall behind comes along.
const float DepthBandMeters = 0.3f;
const double BoxShrink = 0.5;   // the middle depth comes from the centre half of the box (02)

// Clicked without a box: the object grows out from the click, at most this far (metres). Big
// enough for the object; too big and it can creep along a wall the object touches.
const float ClickReachMeters = 0.5f;
const int ClickSnapPixels = 30;     // the click snaps to the nearest depth reading this close (colour pixels)
const double PersonShare = 0.3;     // refused if this share of it lies inside person boxes (07d)

// The surface the object stands on (a table, the floor), found by RANSAC as in 09. Points on it
// or below it are dropped.
const bool RemoveSupportPlane = true;
const float PlaneMarginMeters = 0.01f;   // "on the plane" = within this distance
const int RansacTries = 200;
const int NeighbourPixels = 10;          // RANSAC's 3 points come from this close together (09)
const float MaxPlaneSlopeDegrees = 45;   // steeper is a wall, not something to stand on
const double MinPlaneShare = 0.10;       // a plane needs at least this share of the points...
const double MaxBelowShare = 0.10;       // ...and at most this share below it. A chair seat has
                                         // its legs below it, so it isn't taken for the floor.

// --- Mesh ---
// No triangle across a depth jump bigger than this (mm): that's an edge, not a surface.
// Small = torn on slanted surfaces, big = webs to whatever is behind.
const float MaxJumpMm = 20;

// The back, which the camera can't see. Mirror = the front copied behind, assuming the object is
// as deep as it is wide (right for cups and bottles, a guess for boxes). Flat = a flat back at the
// front's farthest point. None = the front shell only.
enum class Back { None, Flat, Mirror };
const Back BackGuess = Back::Mirror;

// --- Detector (as in 05c) ---
const int ModelInputSize = 640;
const std::string ModelPath = std::string(SPOTTER_REPO_ROOT) + (ModelInputSize == 320 ? "/yolo26n_320.onnx" : "/yolo26n.onnx");
const float Confidence = 0.40f;
const bool CapColorExposure = true;   // 05c: keeps 30 fps in a dim room, at the cost of a darker picture

const int Width = 512, Height = 424, ColorWidth = 1920, ColorHeight = 1080;

// The 80 COCO classes yolo26n.pt was trained on, in classId order.
const char* ClassNames[] =
{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
};

// ---------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------

struct Detection
{
    int classId;
    float confidence;
    cv::Rect2d box;
};

// A triangle mesh: points in the Kinect's coordinates (metres; X right, Y down, Z forward).
struct Mesh
{
    std::vector<cv::Vec3f> points;
    std::vector<cv::Vec3b> colours;                    // blue, green, red
    std::vector<cv::Vec3i> seenFaces, guessedFaces;    // three point indices each
};

// A plane, as in 09: normal . point + distance = height above it.
struct Plane
{
    cv::Vec3f normal;
    float distance;
    float HeightOf(const cv::Vec3f& point) const { return normal.dot(point) + distance; }
};

int clickX = -1, clickY = -1;   // set by the mouse, in full-size colour pixels

void OnMouse(int event, int x, int y, int, void*)
{
    if (event == cv::EVENT_LBUTTONDOWN) { clickX = x * 2; clickY = y * 2; }   // the window is half size
}

float Median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) return values[middle];
    return (values[middle - 1] + values[middle]) / 2;
}

// YOLO on one colour frame (letterbox, forward, undo letterbox: as in 05c).
std::vector<Detection> Detect(cv::dnn::Net& detector, const cv::Mat& frame)
{
    double scale = std::min((double)ModelInputSize / frame.cols, (double)ModelInputSize / frame.rows);
    int scaledWidth = (int)std::round(frame.cols * scale), scaledHeight = (int)std::round(frame.rows * scale);
    int padLeft = (ModelInputSize - scaledWidth) / 2, padTop = (ModelInputSize - scaledHeight) / 2;
    cv::Mat scaled, letterboxed;
    cv::resize(frame, scaled, cv::Size(scaledWidth, scaledHeight));
    cv::copyMakeBorder(scaled, letterboxed, padTop, ModelInputSize - scaledHeight - padTop,
                       padLeft, ModelInputSize - scaledWidth - padLeft, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    detector.setInput(cv::dnn::blobFromImage(letterboxed, 1.0 / 255.0, cv::Size(ModelInputSize, ModelInputSize),
                                             cv::Scalar(0, 0, 0), true, false));
    cv::Mat output = detector.forward();
    cv::Mat rows(output.size[1], output.size[2], CV_32F, output.ptr<float>());

    std::vector<Detection> detections;
    for (int row = 0; row < rows.rows; row++)
    {
        float confidence = rows.at<float>(row, 4);
        if (confidence < Confidence) continue;
        double left = (rows.at<float>(row, 0) - padLeft) / scale, top = (rows.at<float>(row, 1) - padTop) / scale;
        double right = (rows.at<float>(row, 2) - padLeft) / scale, bottom = (rows.at<float>(row, 3) - padTop) / scale;
        detections.push_back({ (int)rows.at<float>(row, 5), confidence, cv::Rect2d(left, top, right - left, bottom - top) });
    }
    return detections;
}

// The plane through three points, normal towards the camera (09).
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

// Least-squares plane through many points (09).
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

// RANSAC as in 09, but only over the object's pixels: the level-ish plane with the most points ON
// it and few BELOW it. That's the table or floor the object stands on.
bool FindSupport(const cv::Mat& mask, const std::vector<cv::Vec3f>& cloud, std::mt19937& random, Plane& support)
{
    std::vector<int> kept;
    for (int i = 0; i < Width * Height; i++)
        if (mask.data[i]) kept.push_back(i);
    if (kept.size() < 3) return false;

    const float minUpness = std::cos(MaxPlaneSlopeDegrees * (float)CV_PI / 180);
    std::uniform_int_distribution<size_t> pick(0, kept.size() - 1);
    std::uniform_int_distribution<int> offset(-NeighbourPixels, NeighbourPixels);
    int bestCount = 0;
    for (int t = 0; t < RansacTries; t++)
    {
        int a = kept[pick(random)], row = a / Width, column = a % Width;
        int bRow = row + offset(random), bColumn = column + offset(random);
        int cRow = row + offset(random), cColumn = column + offset(random);
        if (std::min({ bRow, bColumn, cRow, cColumn }) < 0 || std::max(bRow, cRow) >= Height ||
            std::max(bColumn, cColumn) >= Width) continue;
        int b = bRow * Width + bColumn, c = cRow * Width + cColumn;
        if (!mask.data[b] || !mask.data[c]) continue;
        Plane plane;
        if (!PlaneThrough(cloud[a], cloud[b], cloud[c], plane)) continue;
        if (plane.normal.dot(cv::Vec3f(0, -1, 0)) < minUpness) continue;   // camera "up" is -Y
        int on = 0, below = 0;
        for (int k : kept)
        {
            float height = plane.HeightOf(cloud[k]);
            if (std::abs(height) < PlaneMarginMeters) on++;
            else if (height < 0) below++;
        }
        if (below > MaxBelowShare * kept.size()) continue;
        if (on > bestCount) { bestCount = on; support = plane; }
    }
    if (bestCount < MinPlaneShare * kept.size()) return false;

    std::vector<cv::Vec3f> inliers;
    for (int k : kept)
        if (std::abs(support.HeightOf(cloud[k])) < PlaneMarginMeters) inliers.push_back(cloud[k]);
    support = FitPlane(inliers);
    return true;
}

// Drop the points on or below the surface the object stands on, if there is one.
void RemoveSupport(cv::Mat& mask, const std::vector<cv::Vec3f>& cloud, std::mt19937& random)
{
    Plane support;
    if (!RemoveSupportPlane) return;
    if (!FindSupport(mask, cloud, random, support))
    {
        std::printf("  no surface found under it (none removed)\n");
        return;
    }
    for (int i = 0; i < Width * Height; i++)
        if (mask.data[i] && support.HeightOf(cloud[i]) < PlaneMarginMeters) mask.data[i] = 0;
    std::printf("  above the surface it stands on: %d\n", cv::countNonZero(mask));
}

// The object's depth pixels: in the box, near its middle depth, above its support, one blob.
// Prints how many pixels each step leaves, so you can see which step does what.
cv::Mat CutOut(const cv::Rect2d& box, const std::vector<cv::Vec3f>& cloud,
               const std::vector<int>& colorIndexOfDepthPixel, std::mt19937& random)
{
    cv::Mat mask(Height, Width, CV_8U, cv::Scalar(0));
    cv::Rect2d centre(box.x + box.width * (1 - BoxShrink) / 2, box.y + box.height * (1 - BoxShrink) / 2,
                      box.width * BoxShrink, box.height * BoxShrink);
    std::vector<float> centreZ;
    int count = 0;
    for (int i = 0; i < Width * Height; i++)
    {
        int colorIndex = colorIndexOfDepthPixel[i];
        if (colorIndex < 0 || std::isnan(cloud[i][2])) continue;
        cv::Point2d pixel(colorIndex % ColorWidth, colorIndex / ColorWidth);
        if (!box.contains(pixel)) continue;
        mask.data[i] = 255;
        count++;
        if (centre.contains(pixel)) centreZ.push_back(cloud[i][2]);
    }
    std::printf("  in the box: %d pixels\n", count);
    if (centreZ.empty()) return mask * 0;

    float middle = Median(centreZ);
    count = 0;
    for (int i = 0; i < Width * Height; i++)
    {
        if (mask.data[i] && std::abs(cloud[i][2] - middle) > DepthBandMeters) mask.data[i] = 0;
        count += mask.data[i] ? 1 : 0;
    }
    std::printf("  within %.2f m of its middle depth (%.2f m): %d\n", DepthBandMeters, middle, count);
    RemoveSupport(mask, cloud, random);

    // Keep only the biggest connected blob (07c): stray bits of other things go.
    cv::Mat labels, stats, centroids;
    int blobs = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    int biggest = 0;
    for (int label = 1; label < blobs; label++)
        if (biggest == 0 || stats.at<int>(label, cv::CC_STAT_AREA) > stats.at<int>(biggest, cv::CC_STAT_AREA)) biggest = label;
    if (biggest == 0) return mask;
    mask = labels == biggest;
    std::printf("  biggest blob: %d\n", stats.at<int>(biggest, cv::CC_STAT_AREA));
    return mask;
}

// Clicked without a box: grow the object out from the clicked depth pixel. The points near the
// click, minus the surface it stands on, then a flood fill from the click that only steps to a
// neighbour less than MaxJumpMm away in depth. It stops at the object's edges, where the depth
// jumps to whatever is behind.
cv::Mat CutOutAround(int seed, const std::vector<cv::Vec3f>& cloud, std::mt19937& random)
{
    cv::Mat mask(Height, Width, CV_8U, cv::Scalar(0));
    if (std::isnan(cloud[seed][2]))
    {
        std::printf("  no steady depth reading at the click\n");
        return mask;
    }
    for (int i = 0; i < Width * Height; i++)
        if (!std::isnan(cloud[i][2]) && cv::norm(cloud[i] - cloud[seed]) < ClickReachMeters) mask.data[i] = 255;
    std::printf("  within %.2f m of the click (%.2f m away): %d pixels\n", ClickReachMeters, cloud[seed][2], cv::countNonZero(mask));
    RemoveSupport(mask, cloud, random);
    if (!mask.data[seed])
    {
        std::printf("  the click was on the surface itself: click the object\n");
        return mask * 0;
    }

    // floodFill's mask is one pixel bigger all round; it never fills where the mask isn't 0, and
    // with FLOODFILL_MASK_ONLY it marks what it fills there (here with 128) instead of painting.
    cv::Mat depthMm(Height, Width, CV_32F), fill(Height + 2, Width + 2, CV_8U, cv::Scalar(255));
    for (int i = 0; i < Width * Height; i++) depthMm.at<float>(i) = mask.data[i] ? cloud[i][2] * 1000 : 0;
    fill(cv::Rect(1, 1, Width, Height)).setTo(0, mask);
    cv::floodFill(depthMm, fill, cv::Point(seed % Width, seed / Width), 0, nullptr, MaxJumpMm, MaxJumpMm,
                  8 | cv::FLOODFILL_MASK_ONLY | (128 << 8));
    mask = fill(cv::Rect(1, 1, Width, Height)) == 128;
    std::printf("  joined to the click without a depth jump: %d\n", cv::countNonZero(mask));
    return mask;
}

// The depth pixel seen nearest to a colour pixel (within ClickSnapPixels), or -1.
int DepthPixelAt(int x, int y, const float* depthMm, const std::vector<int>& colorIndexOfDepthPixel)
{
    int best = -1;
    double bestDistance = ClickSnapPixels;
    for (int i = 0; i < Width * Height; i++)
    {
        int colorIndex = colorIndexOfDepthPixel[i];
        if (colorIndex < 0 || !(depthMm[i] > 0) || std::isinf(depthMm[i])) continue;
        double distance = std::hypot(colorIndex % ColorWidth - x, colorIndex / ColorWidth - y);
        if (distance < bestDistance) { bestDistance = distance; best = i; }
    }
    return best;
}

// The depth image is already a grid: each 2x2 square of object pixels gives two triangles,
// unless the corners jump in depth.
Mesh MeshFromMask(const cv::Mat& mask, const std::vector<cv::Vec3f>& cloud, const unsigned char* registeredBgrx)
{
    Mesh mesh;
    std::vector<int> pointOf(Width * Height, -1);
    for (int i = 0; i < Width * Height; i++)
    {
        if (!mask.data[i]) continue;
        pointOf[i] = (int)mesh.points.size();
        mesh.points.push_back(cloud[i]);
        const unsigned char* bgrx = registeredBgrx + 4 * i;
        mesh.colours.emplace_back(bgrx[0], bgrx[1], bgrx[2]);
    }
    auto addTriangle = [&](int a, int b, int c)
    {
        int pa = pointOf[a], pb = pointOf[b], pc = pointOf[c];
        if (pa < 0 || pb < 0 || pc < 0) return;
        float nearest = std::min({ cloud[a][2], cloud[b][2], cloud[c][2] });
        float farthest = std::max({ cloud[a][2], cloud[b][2], cloud[c][2] });
        if ((farthest - nearest) * 1000 > MaxJumpMm) return;
        mesh.seenFaces.emplace_back(pa, pb, pc);   // this order faces the camera
    };
    for (int row = 0; row + 1 < Height; row++)
    {
        for (int column = 0; column + 1 < Width; column++)
        {
            int topLeft = row * Width + column, topRight = topLeft + 1;
            int bottomLeft = topLeft + Width, bottomRight = bottomLeft + 1;
            addTriangle(topLeft, bottomLeft, topRight);
            addTriangle(topRight, bottomLeft, bottomRight);
        }
    }
    return mesh;
}

// Guess the back: a grey twin of every front point, pushed behind, plus a rim joining the two.
void AddGuessedBack(Mesh& mesh)
{
    if (BackGuess == Back::None || mesh.seenFaces.empty()) return;
    int count = (int)mesh.points.size();
    float left = 1e9f, right = -1e9f, nearest = 1e9f, farthest = -1e9f;
    for (const cv::Vec3f& p : mesh.points)
    {
        left = std::min(left, p[0]);
        right = std::max(right, p[0]);
        nearest = std::min(nearest, p[2]);
        farthest = std::max(farthest, p[2]);
    }
    // ponytail: "as deep as it is wide" is the whole guess. A cylinder fit for cups and bottles
    // is better (docs/ideas.md); a real back needs more views (rung D).
    float middle = nearest + (right - left) / 2;
    for (int i = 0; i < count; i++)
    {
        cv::Vec3f p = mesh.points[i];
        p[2] = BackGuess == Back::Flat ? farthest : std::max(2 * middle - p[2], p[2]);   // never in front of the front
        mesh.points.push_back(p);
        mesh.colours.emplace_back(128, 128, 128);
    }

    // The back's triangles, in reverse order so they face away from the camera.
    std::set<std::pair<int, int>> edges;
    for (const cv::Vec3i& f : mesh.seenFaces)
    {
        edges.insert({ f[0], f[1] });
        edges.insert({ f[1], f[2] });
        edges.insert({ f[2], f[0] });
        mesh.guessedFaces.emplace_back(f[0] + count, f[2] + count, f[1] + count);
    }
    // The rim: an edge whose reverse no triangle uses is on the border. Join it to its twin.
    // ponytail: where two parts of the border touch at one pixel ("pinch"), that pixel's rim edge
    // is used twice. Still closed, and Blender and MeshLab don't mind; a 3D-print slicer would.
    // Split pinch points if that matters.
    for (const auto& [a, b] : edges)
    {
        if (edges.count({ b, a })) continue;
        mesh.guessedFaces.emplace_back(b, a, a + count);
        mesh.guessedFaces.emplace_back(b, a + count, b + count);
    }
}

// OBJ: text, one "v x y z red green blue" line per point (colours 0-1, which Blender, MeshLab and
// CloudCompare read), then one "f a b c" line per triangle (counting from 1). Saved Y up, Z towards
// the viewer, as 08's .ply. The "g" lines put seen and guessed triangles in separate groups.
bool SaveObj(const std::string& path, const Mesh& mesh)
{
    std::ofstream file(path);
    file << "# Spotter 08b. seen_front = scanned by the Kinect, guessed_back = made up\n";
    for (size_t i = 0; i < mesh.points.size(); i++)
    {
        const cv::Vec3f& p = mesh.points[i];
        const cv::Vec3b& c = mesh.colours[i];
        file << cv::format("v %.4f %.4f %.4f %.3f %.3f %.3f\n", p[0], -p[1], -p[2], c[2] / 255.0, c[1] / 255.0, c[0] / 255.0);
    }
    file << "g seen_front\n";
    for (const cv::Vec3i& f : mesh.seenFaces) file << cv::format("f %d %d %d\n", f[0] + 1, f[1] + 1, f[2] + 1);
    if (!mesh.guessedFaces.empty()) file << "g guessed_back\n";
    for (const cv::Vec3i& f : mesh.guessedFaces) file << cv::format("f %d %d %d\n", f[0] + 1, f[1] + 1, f[2] + 1);
    return (bool)file;
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // -----------------------------------------------------------------
    //  Setup: detector, Kinect, registration (as in 05c)
    // -----------------------------------------------------------------

    if (!std::filesystem::exists(ModelPath))
    {
        std::printf("Model not found: %s\n", ModelPath.c_str());
        return 1;
    }
    cv::dnn::Net detector = cv::dnn::readNetFromONNX(ModelPath);

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

    std::vector<Detection> detections;
    Detection chosen{};
    int seedPixel = -1;                            // clicked without a box: the depth pixel clicked, else -1
    cv::Point clicked;
    int framesCaptured = -1;                       // -1 = not capturing (07)
    std::vector<double> sumMm(Width * Height);
    std::vector<int> readings(Width * Height);
    std::vector<cv::Vec3f> cloud(Width * Height);
    std::vector<cv::Point> tintPixels;             // colour pixels of the last capture's object
    std::mt19937 random(1);
    double fps = 0;
    auto lastFrameTime = std::chrono::steady_clock::now();

    const std::string window = "08b object mesh  (click an object to capture, C clear, Q quit)";
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(window, OnMouse);
    std::printf("Click an object (its box, or the object itself if it has none). Keep it and the camera still for %d frames.\n", AverageFrames);

    // -----------------------------------------------------------------
    //  Main operation
    // -----------------------------------------------------------------

    libfreenect2::FrameMap frames;
    cv::Mat frame;
    while (true)
    {
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == 'c') tintPixels.clear();

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

        // --- Step 1: pick (YOLO waits, ~6 fps; paused during a capture) ----------
        if (framesCaptured < 0) detections = Detect(detector, frame);
        if (clickX >= 0 && framesCaptured < 0)
        {
            const Detection* hit = nullptr;   // the smallest box under the click
            for (const Detection& d : detections)
                if (d.box.contains(cv::Point2d(clickX, clickY)) && (hit == nullptr || d.box.area() < hit->box.area())) hit = &d;
            seedPixel = hit == nullptr ? DepthPixelAt(clickX, clickY, (const float*)undistorted.data, colorIndexOfDepthPixel) : -1;
            if (hit != nullptr && hit->classId == 0)
                std::printf("Not capturing a person (privacy).\n");
            else if (hit == nullptr && seedPixel < 0)
                std::printf("No box and no depth reading there.\n");
            else
            {
                if (hit != nullptr) chosen = *hit;
                clicked = cv::Point(clickX, clickY);
                framesCaptured = 0;
                std::fill(sumMm.begin(), sumMm.end(), 0.0);
                std::fill(readings.begin(), readings.end(), 0);
                std::printf("Capturing %s: hold still...\n", hit != nullptr ? ClassNames[chosen.classId] : "the clicked object (no box)");
            }
        }
        clickX = -1;

        // --- Step 2: average the depth (07) -----------------------------------------
        if (framesCaptured >= 0)
        {
            const float* depthMm = (const float*)undistorted.data;
            for (int i = 0; i < Width * Height; i++)
            {
                if (!(depthMm[i] > 0) || std::isinf(depthMm[i])) continue;   // holes don't count
                sumMm[i] += depthMm[i];
                readings[i]++;
            }
            framesCaptured++;
        }

        // --- Steps 3-6: cut out, mesh, guess the back, save ---------------------------
        if (framesCaptured == AverageFrames)
        {
            framesCaptured = -1;
            auto start = std::chrono::steady_clock::now();
            float* averageMm = (float*)averaged.data;
            for (int i = 0; i < Width * Height; i++)
                averageMm[i] = readings[i] >= AverageFrames / 2 ? (float)(sumMm[i] / readings[i]) : 0;   // too often a hole: a hole
            for (int row = 0; row < Height; row++)
                for (int column = 0; column < Width; column++)
                {
                    float x, y, z;
                    registration.getPointXYZ(&averaged, row, column, x, y, z);   // NaN = no reading
                    cloud[row * Width + column] = cv::Vec3f(x, y, z);
                }

            cv::Mat mask = seedPixel >= 0 ? CutOutAround(seedPixel, cloud, random)
                                          : CutOut(chosen.box, cloud, colorIndexOfDepthPixel, random);

            // No box means no class to refuse, so check the scan isn't mostly a person (07d's idea).
            if (seedPixel >= 0)
            {
                int inPerson = 0;
                for (int i = 0; i < Width * Height; i++)
                {
                    if (!mask.data[i] || colorIndexOfDepthPixel[i] < 0) continue;
                    cv::Point2d pixel(colorIndexOfDepthPixel[i] % ColorWidth, colorIndexOfDepthPixel[i] / ColorWidth);
                    for (const Detection& d : detections)
                        if (d.classId == 0 && d.box.contains(pixel)) { inPerson++; break; }
                }
                if (inPerson > PersonShare * cv::countNonZero(mask))
                {
                    std::printf("Mostly inside a person's box: not saved (privacy).\n");
                    continue;
                }
            }

            Mesh mesh = MeshFromMask(mask, cloud, registered.data);
            size_t seenPoints = mesh.points.size();
            AddGuessedBack(mesh);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

            tintPixels.clear();
            for (int i = 0; i < Width * Height; i++)
                if (mask.data[i] && colorIndexOfDepthPixel[i] >= 0)
                    tintPixels.emplace_back(colorIndexOfDepthPixel[i] % ColorWidth, colorIndexOfDepthPixel[i] / ColorWidth);

            if (mesh.seenFaces.empty())
                std::printf("Nothing left to mesh. Too far, too dark or shiny, or try a bigger DepthBandMeters / ClickReachMeters.\n");
            else
            {
                std::string name = seedPixel >= 0 ? "clicked" : ClassNames[chosen.classId];
                std::replace(name.begin(), name.end(), ' ', '_');   // "potted plant" -> potted_plant
                std::string path = (std::filesystem::current_path() /
                                    cv::format("object_%s_%lld.obj", name.c_str(), (long long)std::time(nullptr))).string();
                if (SaveObj(path, mesh))
                    std::printf("Saved %zu seen points, %zu seen + %zu guessed triangles (%.0f ms) to %s\n"
                                "  (a capture of the room: never commit it)\n",
                                seenPoints, mesh.seenFaces.size(), mesh.guessedFaces.size(), ms, path.c_str());
                else
                    std::printf("Could not write %s\n", path.c_str());
            }
        }

        // -----------------------------------------------------------------
        //  Output: half-size colour, boxes, the last capture tinted green
        // -----------------------------------------------------------------

        cv::Mat display;
        cv::resize(frame, display, cv::Size(ColorWidth / 2, ColorHeight / 2));
        for (const cv::Point& p : tintPixels)
        {
            cv::Vec3b& pixel = display.at<cv::Vec3b>(p.y / 2, p.x / 2);
            pixel = cv::Vec3b(pixel[0] / 2, (pixel[1] + 255) / 2, pixel[2] / 2);
        }
        auto drawBox = [&](const Detection& d, const cv::Scalar& colour)
        {
            cv::Rect box(cv::Point((int)d.box.x / 2, (int)d.box.y / 2), cv::Point((int)d.box.br().x / 2, (int)d.box.br().y / 2));
            cv::rectangle(display, box, colour, 2);
            cv::putText(display, ClassNames[d.classId], box.tl() + cv::Point(3, 16), cv::FONT_HERSHEY_SIMPLEX, 0.5, colour, 1);
        };
        if (framesCaptured >= 0 && seedPixel >= 0)
            cv::circle(display, clicked / 2, 8, cv::Scalar(0, 255, 255), 2);
        else if (framesCaptured >= 0)
            drawBox(chosen, cv::Scalar(0, 255, 255));
        else
            for (const Detection& d : detections) drawBox(d, d.classId == 0 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0));
        std::string status = framesCaptured >= 0 ? cv::format("HOLD STILL  %d / %d", framesCaptured, AverageFrames)
                                                 : cv::format("%.1f fps   click an object to capture", fps);
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
//  1. Open the .obj in Blender (File > Import > Wavefront) or MeshLab. Hide the group
//     guessed_back: what's left is all the camera really saw.
//  2. MaxJumpMm = 5 versus 50: a torn mesh versus webs stretched to the wall behind.
//  3. AverageFrames = 1 versus 30: look at a flat side edge on. How bumpy is one frame?
//  4. RemoveSupportPlane = false on a cup on a table: the tabletop inside the box comes along.
//  5. The same cup at 1 m and at 2 m. How many triangles each? (One depth pixel is about 2.7 mm
//     across at 1 m and 5.4 mm at 2 m.)
//  6. A black or shiny object: where do the holes come from?
//  7. BackGuess = Mirror on a cup, a box and a chair. Where does "as deep as it is wide" go wrong?
//     Try Flat on the box.
//  8. Tape-measure the object's real depth and compare it with the guess (half its width behind
//     the nearest point, times two).
//  9. Click an object YOLO has no box for (a cup on the floor seen side-on came out as "fire
//     hydrant 0.05"). Then click the same object's box, if it has one: do the two scans match?
// 10. Click-mode on something touching a wall: ClickReachMeters 0.2 versus 1.0. Where does the
//     flood fill creep along the wall, and where does MaxJumpMm stop it?
// 11. Click the floor or the table itself: the support plane removes it, nothing to scan.
