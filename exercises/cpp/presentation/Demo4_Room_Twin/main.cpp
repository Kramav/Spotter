// =====================================================================
//  Demo 4 · Room Twin                 (presentation; from 13, 08 and 03)
//
//  The room you scanned with 13, as a slowly turning doll's house (the
//  ceiling cut off), with the visitors walking around INSIDE it, live,
//  in 3D. A first cut of roadmap step 14 (objects in the map), for people.
//
//    1. Load      13's room_scan_<time>.obj (only its "seen_room" mesh
//                 points: what was really seen), and how scan 1's camera
//                 sat in the room (13 writes that as a comment).
//    2. Learn     the empty view's depth, as in 03.
//    3. Locate    where the Kinect stands in the room: start where scan 1
//                 was, then 13's ICP (point to plane, along the room's
//                 surface normals). The empty view is matched, so
//                 visitors don't confuse it.
//    4. Live      every frame: what is clearly nearer than the empty view
//                 (03's blobs) becomes 3D points, moved into the room by
//                 step 3's pose, and drawn in it.
//
//  Before the show: scan the stand with 13 (README, "Recreate a room"),
//  press E, then put the Kinect back where scan 1 was taken, aimed the
//  same way (within about 50 cm and 15°: ICP finds the rest). Run this
//  from the folder the .obj is in, with nobody in view.
//
//  Nothing is saved. The room file holds no people (13 is run with
//  nobody in view); the visitors are only drawn on screen.
//
//  Keys:   L  learn the empty view and locate again (clear the view first)
//          H  visitors highlighted / true colour     O  turning on/off
//          mouse drag / W A S D  turn   wheel / Z X  zoom   R reset
//          F  fullscreen on/off      Q or Esc  quit
//  Run:    Demo4_Room_Twin [room.obj]   (default: the newest room_scan_*.obj here)
//  No room file? It takes the Kinect's own empty view as the room instead (OneViewRoom.h): one
//  view only, good for rehearsing.
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/registration.h>
#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>

#include "OneViewRoom.h"   // the fallback when there's no room file (its own file: the user's call, D23)

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// --- The doll's house ---
const float CutAboveFloorMeters = 1.6f;   // room points higher than this above the floor are hidden
const float OrbitSeconds = 60;            // one full turn
const float ViewPitchDegrees = 35;        // looking down from above

// --- Visitors: 03's blobs ---
const int SecondsBeforeBackground = 3;
const int ForegroundThresholdMm = 100;
const int CleanupSizePixels = 5;
const int MinBlobPixels = 400;
const float MaxDepthMeters = 6.0f;

// --- Locating the Kinect in the room: 13's ICP ---
const int LocateStep = 8;                  // every 8th pixel of the empty view
const int IcpRounds = 30;
const float IcpStartPairMeters = 0.50f;    // 13 used 0.05 after its features; here the Kinect may be put back 50 cm off
const float IcpFinalPairMeters = 0.02f;
const float MinLocatedShare = 0.6f;        // below this share of matched points, warn (lost: 40%, found: 100% in the test)

// --- Viewer ---
const int ViewWidth = 1280, ViewHeight = 720;
const float ViewFocalPixels = 930;
// Each point is drawn as big as the patch it stands for, so nothing behind shows through the
// gaps (a fixed 2 px did: the room's points are 3 cm apart, at 9 m that's 3 px on screen).
const float RoomPointMeters = 0.03f;    // 13's cube size: its export prints it ("3.0 cm cubes")
const float DepthFocalPixels = 365;     // a visitor's point covers distance / this (03 prints it)
const int MaxPointPixels = 8;
const float TurnStepDegrees = 5;
const float ZoomStepMeters = 0.25f;

const bool StartFullscreen = true;
// The keys along the bottom of the screen, short (the user's ask).
const char* KeyLegend = "l - relocate   h - highlight   o - orbit   wasd - turn   z/x - zoom   r - reset   f - fullscreen   q - quit";
const int Width = 512, Height = 424;

// ---------------------------------------------------------------------
//  The viewer: as in 08, but in room coordinates (X right, Y UP, Z towards the first camera),
//  turning about the room's middle and looking down at it
// ---------------------------------------------------------------------

float yawDegrees = 0, pitchDegrees = 0, orbitDegrees = 0;
float viewDistance = 5;
cv::Point3f pivot(0, 0, 0);
bool dragging = false;
int lastMouseX = 0, lastMouseY = 0;

void OnMouse(int event, int x, int y, int flags, void*)
{
    if (event == cv::EVENT_LBUTTONDOWN) { dragging = true; lastMouseX = x; lastMouseY = y; }
    if (event == cv::EVENT_LBUTTONUP) dragging = false;
    if (event == cv::EVENT_MOUSEMOVE && dragging)
    {
        yawDegrees += (x - lastMouseX) * 0.3f;
        pitchDegrees += (y - lastMouseY) * 0.3f;
        lastMouseX = x;
        lastMouseY = y;
    }
    if (event == cv::EVENT_MOUSEWHEEL)
        viewDistance -= cv::getMouseWheelDelta(flags) / 120.0f * ZoomStepMeters;
}

bool Project(const cv::Point3f& point, cv::Point& pixel, float& depth)
{
    const float yaw = (yawDegrees + orbitDegrees) * (float)CV_PI / 180;
    const float pitch = std::clamp(ViewPitchDegrees + pitchDegrees, -89.0f, 89.0f) * (float)CV_PI / 180;
    cv::Point3f p = point - pivot;
    float x1 = std::cos(yaw) * p.x + std::sin(yaw) * p.z;
    float z1 = -std::sin(yaw) * p.x + std::cos(yaw) * p.z;
    float y2 = std::cos(pitch) * p.y - std::sin(pitch) * z1;   // tip the far side up: seen from above
    float z2 = std::sin(pitch) * p.y + std::cos(pitch) * z1;
    depth = viewDistance - z2;                                  // the viewer sits on +Z, looking back
    if (depth < 0.05f) return false;
    pixel = cv::Point((int)(ViewFocalPixels * x1 / depth + ViewWidth / 2),
                      (int)(ViewHeight / 2 - ViewFocalPixels * y2 / depth));
    return pixel.x >= 0 && pixel.y >= 0 && pixel.x < ViewWidth && pixel.y < ViewHeight;
}

// As 08's Render, drawing into an image and z-buffer shared by the room and the visitors. Each
// point is a square as big as its patch (patchMeters) looks from the viewer.
void Render(cv::Mat& image, cv::Mat& nearest, const std::vector<cv::Point3f>& points,
            const std::vector<cv::Vec3b>& colours, const std::vector<float>& patchMeters)
{
    for (size_t i = 0; i < points.size(); i++)
    {
        cv::Point pixel;
        float depth;
        if (!Project(points[i], pixel, depth)) continue;
        int size = std::clamp((int)std::ceil(ViewFocalPixels * patchMeters[i] / depth), 1, MaxPointPixels);
        for (int dy = -size / 2; dy < size - size / 2; dy++)
        {
            for (int dx = -size / 2; dx < size - size / 2; dx++)
            {
                int u = pixel.x + dx, v = pixel.y + dy;
                if (u < 0 || v < 0 || u >= ViewWidth || v >= ViewHeight || depth >= nearest.at<float>(v, u)) continue;
                nearest.at<float>(v, u) = depth;
                image.at<cv::Vec3b>(v, u) = colours[i];
            }
        }
    }
}

// ---------------------------------------------------------------------
//  Step 1: the room file
// ---------------------------------------------------------------------

// The points of 13's OBJ that belong to its "seen_room" faces (the clean model's boxes and walls
// are only corners); 09b's files have that group too. "firstCamera" = 13's
// comment line: the turn from scan 1's camera (X right, Y down, Z ahead) into the room. Without
// it (09b), the file is the camera's own view with Y and Z flipped.
// "normals" = each point's surface direction: the sum of its faces' (13's ICP measures along it).
bool LoadRoom(const std::string& path, std::vector<cv::Point3f>& points, std::vector<cv::Vec3b>& colours,
              std::vector<cv::Point3f>& normals, cv::Matx33f& firstCamera)
{
    firstCamera = cv::Matx33f(1, 0, 0, 0, -1, 0, 0, 0, -1);
    std::ifstream file(path);
    if (!file) return false;
    std::vector<cv::Point3f> v;
    std::vector<cv::Vec3b> c;
    std::vector<char> seen;
    std::vector<cv::Point3f> faceNormals;
    std::string line, group;
    while (std::getline(file, line))
    {
        float x, y, z, r = 0.6f, g = 0.6f, b = 0.6f;
        int f[3];
        if (line.rfind("v ", 0) == 0 && std::sscanf(line.c_str(), "v %f %f %f %f %f %f", &x, &y, &z, &r, &g, &b) >= 3)
        {
            v.emplace_back(x, y, z);
            c.emplace_back((uchar)(b * 255), (uchar)(g * 255), (uchar)(r * 255));
        }
        else if (line.rfind("g ", 0) == 0) group = line.substr(2);
        else if (line.rfind("# first camera turn:", 0) == 0)
        {
            cv::Matx33f m;
            if (std::sscanf(line.c_str() + 20, "%f %f %f %f %f %f %f %f %f", &m(0, 0), &m(0, 1), &m(0, 2),
                            &m(1, 0), &m(1, 1), &m(1, 2), &m(2, 0), &m(2, 1), &m(2, 2)) == 9) firstCamera = m;
        }
        else if (group == "seen_room" && std::sscanf(line.c_str(), "f %d %d %d", &f[0], &f[1], &f[2]) == 3)
        {
            if (std::min({ f[0], f[1], f[2] }) < 1 || std::max({ f[0], f[1], f[2] }) > (int)v.size()) continue;
            seen.resize(v.size(), 0);
            faceNormals.resize(v.size(), cv::Point3f(0, 0, 0));
            cv::Point3f n = (v[f[1] - 1] - v[f[0] - 1]).cross(v[f[2] - 1] - v[f[0] - 1]);
            for (int i : f) { seen[i - 1] = 1; faceNormals[i - 1] += n; }
        }
    }
    for (size_t i = 0; i < v.size(); i++)
    {
        if (seen.empty() || i >= seen.size() || !seen[i] || cv::norm(faceNormals[i]) == 0) continue;
        points.push_back(v[i]);
        colours.push_back(c[i]);
        normals.push_back(faceNormals[i] / (float)cv::norm(faceNormals[i]));
    }
    return !points.empty();
}

// The newest room_scan_*.obj in the current folder, or "" if there is none.
std::string NewestRoomFile()
{
    std::string newest;
    std::filesystem::file_time_type newestTime;
    for (const auto& entry : std::filesystem::directory_iterator(std::filesystem::current_path()))
    {
        std::string name = entry.path().filename().string();
        if (name.rfind("room_scan_", 0) != 0 || entry.path().extension() != ".obj") continue;
        if (newest.empty() || entry.last_write_time() > newestTime)
        {
            newest = entry.path().string();
            newestTime = entry.last_write_time();
        }
    }
    return newest;
}

// ---------------------------------------------------------------------
//  Step 3: where the Kinect is in the room (11's pose, BestFit and ICP)
// ---------------------------------------------------------------------

// A Kinect point (X right, Y down, Z ahead) -> the same point in the room.
struct Pose
{
    cv::Matx33f turn = cv::Matx33f::eye();
    cv::Point3f shift = cv::Point3f(0, 0, 0);
    cv::Point3f operator()(const cv::Point3f& p) const { return turn * p + shift; }
    Pose Then(const Pose& next) const { return { next.turn * turn, next.turn * shift + next.shift }; }
    float Degrees() const
    {
        float cosine = (turn(0, 0) + turn(1, 1) + turn(2, 2) - 1) / 2;
        return std::acos(std::clamp(cosine, -1.0f, 1.0f)) * 180 / (float)CV_PI;
    }
};

// As in 13: a turn of |w| radians about the axis w points along (Rodrigues' formula).
cv::Matx33f TurnAbout(const cv::Vec3d& w)
{
    double angle = cv::norm(w);
    if (angle < 1e-12) return cv::Matx33f::eye();
    cv::Vec3d k = w / angle;
    double c = std::cos(angle), s = std::sin(angle);
    cv::Matx33d turn;
    for (int row = 0; row < 3; row++)
        for (int column = 0; column < 3; column++) turn(row, column) = (row == column ? c : 0) + (1 - c) * k[row] * k[column];
    turn(0, 1) -= s * k[2]; turn(0, 2) += s * k[1];
    turn(1, 0) += s * k[2]; turn(1, 2) -= s * k[0];
    turn(2, 0) -= s * k[1]; turn(2, 1) += s * k[0];
    return cv::Matx33f(turn);
}

// 13's ICP (point to plane, directions the shape can't pin left alone; 13 explains both), with
// the room's k-d tree built once (L reuses it). 11's point-to-point ICP was tried first: put back
// 5 degrees and 11 cm off in a bare room, it ended 4 degrees and 12 cm off; this one gets it.
// "share" = how many "from" points end within IcpFinalPairMeters of the room's surface.
Pose Icp(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to, const std::vector<cv::Point3f>& toNormals,
         cv::flann::Index& tree, Pose pose, float& share)
{
    share = 0;
    if (from.size() < 10) return pose;
    cv::Mat query((int)from.size(), 3, CV_32F), nearestIndex, squaredDistance;
    for (int round = 0; round < IcpRounds; round++)
    {
        for (int i = 0; i < (int)from.size(); i++)
        {
            cv::Point3f p = pose(from[i]);
            query.at<float>(i, 0) = p.x;
            query.at<float>(i, 1) = p.y;
            query.at<float>(i, 2) = p.z;
        }
        tree.knnSearch(query, nearestIndex, squaredDistance, 1, cv::flann::SearchParams(32));
        float maxPair = std::max(IcpFinalPairMeters, IcpStartPairMeters * std::pow(0.7f, (float)round));
        float maxNeighbour = std::max(maxPair, 0.06f);
        std::vector<cv::Point3f> ps, normals;
        std::vector<float> gaps;
        cv::Point3f centre(0, 0, 0);
        for (int i = 0; i < (int)from.size(); i++)
        {
            if (squaredDistance.at<float>(i) > maxNeighbour * maxNeighbour) continue;
            cv::Point3f p(query.at<float>(i, 0), query.at<float>(i, 1), query.at<float>(i, 2));
            int j = nearestIndex.at<int>(i);
            float gap = (p - to[j]).dot(toNormals[j]);
            if (std::abs(gap) > maxPair) continue;
            ps.push_back(p);
            normals.push_back(toNormals[j]);
            gaps.push_back(gap);
            centre += p;
        }
        const int pairs = (int)ps.size();
        if (pairs < 10) break;
        centre *= 1.0f / pairs;
        cv::Matx66d A = cv::Matx66d::zeros();
        cv::Matx61d b = cv::Matx61d::zeros();
        for (int i = 0; i < pairs; i++)
        {
            const cv::Point3f& n = normals[i];
            cv::Point3f c = (ps[i] - centre).cross(n);
            const double row[6] = { c.x, c.y, c.z, n.x, n.y, n.z };
            for (int r = 0; r < 6; r++)
            {
                for (int k = 0; k < 6; k++) A(r, k) += row[r] * row[k];
                b(r) -= row[r] * gaps[i];
            }
        }
        cv::Mat values, directions;
        cv::eigen(cv::Mat(A), values, directions);
        cv::Matx61d x = cv::Matx61d::zeros();
        for (int k = 0; k < 6; k++)
        {
            double strength = values.at<double>(k);
            if (strength < 0.02 * pairs) continue;
            double along = 0;
            for (int r = 0; r < 6; r++) along += directions.at<double>(k, r) * b(r);
            for (int r = 0; r < 6; r++) x(r) += along / strength * directions.at<double>(k, r);
        }
        if (maxPair == IcpFinalPairMeters) share = (float)pairs / from.size();
        cv::Matx33f turn = TurnAbout(cv::Vec3d(x(0), x(1), x(2)));
        cv::Point3f shift((float)x(3), (float)x(4), (float)x(5));
        Pose step{ turn, centre - turn * centre + shift };
        pose = pose.Then(step);
        if (maxPair == IcpFinalPairMeters && cv::norm(shift) < 1e-4 && step.Degrees() < 0.01f) break;
    }
    return pose;
}

void ToggleFullscreen(const std::string& window)
{
    bool full = cv::getWindowProperty(window, cv::WND_PROP_FULLSCREEN) == cv::WINDOW_FULLSCREEN;
    cv::setWindowProperty(window, cv::WND_PROP_FULLSCREEN, full ? cv::WINDOW_NORMAL : cv::WINDOW_FULLSCREEN);
}

int main(int argc, char** argv)
{
    // -----------------------------------------------------------------
    //  Setup: the Kinect (depth + colour + registration, as in 08), then the room: 13's file, or
    //  without one, the Kinect's own view (OneViewRoom.h)
    // -----------------------------------------------------------------

    std::setvbuf(stdout, nullptr, _IONBF, 0);
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
    device->start();
    libfreenect2::Registration registration(device->getIrCameraParams(), device->getColorCameraParams());
    libfreenect2::Frame undistorted(Width, Height, 4), registered(Width, Height, 4);

    std::string roomPath = argc > 1 ? argv[1] : NewestRoomFile();
    std::vector<cv::Point3f> room;
    std::vector<cv::Vec3b> roomColours;
    std::vector<cv::Point3f> roomNormals;
    cv::Matx33f firstCamera;
    if (roomPath.empty() || !LoadRoom(roomPath, room, roomColours, roomNormals, firstCamera))
    {
        std::printf("%s For the real thing, scan the room with 13_Kinect_Room_Scan, press E, and run this from\n"
                    "the folder the room_scan_*.obj landed in (or give its path: Demo4_Room_Twin room.obj).\n",
                    roomPath.empty() ? "No room_scan_*.obj in this folder." : ("Could not read " + roomPath + ".").c_str());
        room.clear();
        roomColours.clear();
        roomNormals.clear();
        roomPath = "the Kinect's own view";
        if (!OneViewRoom(listener, registration, room, roomColours, roomNormals, firstCamera))
        {
            std::printf("The Kinect saw nothing.\n");
            device->stop();
            device->close();
            return 1;
        }
    }

    // The doll's house: the floor is the lowest 2% of the room; everything above the cut is hidden.
    std::vector<float> heights;
    for (const cv::Point3f& p : room) heights.push_back(p.y);
    std::nth_element(heights.begin(), heights.begin() + heights.size() / 50, heights.end());
    float floorY = heights[heights.size() / 50];
    std::vector<cv::Point3f> house;
    std::vector<cv::Vec3b> houseColours;
    cv::Point3f middle(0, 0, 0);
    for (size_t i = 0; i < room.size(); i++)
    {
        if (room[i].y > floorY + CutAboveFloorMeters) continue;
        house.push_back(room[i]);
        houseColours.push_back(roomColours[i]);
        middle += room[i];
    }
    middle *= 1.0f / std::max<size_t>(house.size(), 1);
    std::vector<float> reach;
    for (const cv::Point3f& p : house) reach.push_back(std::hypot(p.x - middle.x, p.z - middle.z));
    std::nth_element(reach.begin(), reach.begin() + reach.size() * 9 / 10, reach.end());
    float roomRadius = reach.empty() ? 3 : reach[reach.size() * 9 / 10];
    pivot = cv::Point3f(middle.x, floorY + CutAboveFloorMeters / 2, middle.z);
    const float startDistance = 3 * roomRadius;
    viewDistance = startDistance;
    std::printf("Loaded %s: %zu points, %zu below the cut, about %.1f m across\n", roomPath.c_str(), room.size(),
                house.size(), 2 * roomRadius);

    cv::Mat roomData((int)room.size(), 3, CV_32F, room.data());
    cv::flann::Index tree(roomData, cv::flann::KDTreeIndexParams(4));


    std::vector<float> backgroundMm(Width * Height, 0);
    bool located = false;
    Pose pose;
    float locatedShare = 0;
    auto countdownStart = std::chrono::steady_clock::now(), lastFrame = countdownStart;
    std::printf("Learning the empty view in %d s: clear the view.\n", SecondsBeforeBackground);

    cv::Mat cleanupShape = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(CleanupSizePixels, CleanupSizePixels));
    bool highlight = true, orbit = true;
    std::vector<cv::Point3f> visitors;
    std::vector<cv::Vec3b> visitorColours;
    std::vector<float> visitorPatches, housePatches(house.size(), RoomPointMeters);

    const std::string window = "Demo 4 - room twin";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, ViewWidth, ViewHeight);
    if (StartFullscreen) ToggleFullscreen(window);
    cv::setMouseCallback(window, OnMouse);

    // -----------------------------------------------------------------
    //  Main operation
    // -----------------------------------------------------------------

    libfreenect2::FrameMap frames;
    while (true)
    {
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == 'a') yawDegrees -= TurnStepDegrees;
        if (key == 'd') yawDegrees += TurnStepDegrees;
        if (key == 'w') pitchDegrees += TurnStepDegrees;
        if (key == 's') pitchDegrees -= TurnStepDegrees;
        if (key == 'z') viewDistance -= ZoomStepMeters;
        if (key == 'x') viewDistance += ZoomStepMeters;
        if (key == 'r') { yawDegrees = 0; pitchDegrees = 0; viewDistance = startDistance; }
        if (key == 'h') highlight = !highlight;
        if (key == 'o') orbit = !orbit;
        if (key == 'f') ToggleFullscreen(window);
        if (key == 'l')
        {
            located = false;
            countdownStart = std::chrono::steady_clock::now();
            std::printf("Learning the empty view in %d s: clear the view.\n", SecondsBeforeBackground);
        }
        viewDistance = std::max(viewDistance, 0.5f);
        auto now = std::chrono::steady_clock::now();
        if (orbit) orbitDegrees += 360 * (float)std::chrono::duration<double>(now - lastFrame).count() / OrbitSeconds;
        lastFrame = now;

        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }
        registration.apply(frames[libfreenect2::Frame::Color], frames[libfreenect2::Frame::Depth], &undistorted, &registered);
        listener.release(frames);
        const float* depthMm = (const float*)undistorted.data;

        // A depth pixel -> a Kinect point -> by the pose into the room. "range" = its distance from
        // the Kinect.
        float range = 0;
        auto ToRoom = [&](int row, int column, cv::Point3f& p)
        {
            float x, y, z;
            registration.getPointXYZ(&undistorted, row, column, x, y, z);
            if (std::isnan(z)) return false;
            p = pose(cv::Point3f(x, y, z));
            range = std::sqrt(x * x + y * y + z * z);
            return true;
        };

        // --- Steps 2-3: learn the empty view, then locate the Kinect in the room ---------
        double waited = std::chrono::duration<double>(now - countdownStart).count();
        if (!located && waited >= SecondsBeforeBackground)
        {
            for (int i = 0; i < Width * Height; i++)
                backgroundMm[i] = (depthMm[i] > 0 && !std::isinf(depthMm[i])) ? depthMm[i] : 0;
            std::vector<cv::Point3f> view;
            pose = Pose();   // the view in the Kinect's own coordinates
            for (int row = 0; row < Height; row += LocateStep)
            {
                for (int column = 0; column < Width; column += LocateStep)
                {
                    cv::Point3f p;
                    if (ToRoom(row, column, p)) view.push_back(p);
                }
            }
            Pose start;
            start.turn = firstCamera;   // where scan 1 was
            pose = Icp(view, room, roomNormals, tree, start, locatedShare);
            located = true;
            cv::Point3f at = pose(cv::Point3f(0, 0, 0));
            std::printf("Located: %.0f%% of the view within %.0f cm of the room. Camera at (%+.2f, %.2f, %+.2f) m, "
                        "%.2f m above the floor\n", locatedShare * 100, IcpFinalPairMeters * 100, at.x, at.y, at.z, at.y - floorY);
            if (locatedShare < MinLocatedShare)
                std::printf("  That's a poor match: is the Kinect where scan 1 was taken, aimed the same way? Move it,\n"
                            "  clear the view, press L.\n");
        }

        // --- Step 4: visitors = what's clearly nearer than the empty view (03), into the room ---
        visitors.clear();
        visitorColours.clear();
        visitorPatches.clear();
        if (located)
        {
            cv::Mat foreground(Height, Width, CV_8U, cv::Scalar(0));
            for (int i = 0; i < Width * Height; i++)
                if (depthMm[i] > 0 && !std::isinf(depthMm[i]) && backgroundMm[i] - depthMm[i] > ForegroundThresholdMm) foreground.data[i] = 255;
            cv::morphologyEx(foreground, foreground, cv::MORPH_OPEN, cleanupShape);
            cv::Mat labels, stats, centroids;
            cv::connectedComponentsWithStats(foreground, labels, stats, centroids, 8);
            for (int row = 0; row < Height; row++)
            {
                for (int column = 0; column < Width; column++)
                {
                    int label = labels.at<int>(row, column);
                    cv::Point3f p;
                    if (label == 0 || stats.at<int>(label, cv::CC_STAT_AREA) < MinBlobPixels || !ToRoom(row, column, p)) continue;
                    const uchar* bgrx = registered.data + 4 * (row * Width + column);
                    cv::Vec3b colour(bgrx[0], bgrx[1], bgrx[2]);
                    visitors.push_back(p);
                    visitorPatches.push_back(range / DepthFocalPixels);
                    visitorColours.push_back(highlight ? colour / 2 + cv::Vec3b(0, 80, 127) : colour);   // half orange
                }
            }
        }

        // -----------------------------------------------------------------
        //  Output
        // -----------------------------------------------------------------

        cv::Mat image(ViewHeight, ViewWidth, CV_8UC3, cv::Scalar(20, 20, 20));
        cv::Mat nearest(ViewHeight, ViewWidth, CV_32F, cv::Scalar(1e9));
        Render(image, nearest, house, houseColours, housePatches);
        Render(image, nearest, visitors, visitorColours, visitorPatches);
        cv::Point pixel;
        float depth;
        if (located && Project(pose(cv::Point3f(0, 0, 0)), pixel, depth))
        {
            cv::circle(image, pixel, 8, cv::Scalar(0, 255, 255), 2);
            cv::putText(image, "camera", pixel + cv::Point(12, 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
        }
        std::string title = located ? "The room, scanned in 3D - and you, live, inside it"
                                    : cv::format("Clear the view: learning the room in %d s",
                                                 std::max(1, (int)std::ceil(SecondsBeforeBackground - waited)));
        cv::putText(image, title, cv::Point(20, 45), cv::FONT_HERSHEY_SIMPLEX, 1.1, cv::Scalar(255, 255, 255), 2);
        cv::rectangle(image, cv::Rect(0, ViewHeight - 65, ViewWidth, 65), cv::Scalar(20, 20, 20), cv::FILLED);   // keeps the keys readable
        cv::putText(image, located ? cv::format("matched %.0f%%   live only: nothing is recorded", locatedShare * 100)
                                   : std::string("live only: nothing is recorded"),
                    cv::Point(20, ViewHeight - 45), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(160, 160, 160), 1);
        cv::putText(image, KeyLegend, cv::Point(20, ViewHeight - 15), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(220, 220, 220), 1);
        cv::imshow(window, image);
    }

    device->stop();
    device->close();
    cv::destroyAllWindows();
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Walk to a corner of the stand: does your orange shell stand in the same corner of the
//     scanned room? If it floats beside it, the match was off: look at "matched %" and press L.
//  2. Move the Kinect 40 cm along and press L: it finds itself again, and the yellow "camera"
//     circle moves in the house. Turn it 30 degrees: it gets lost (and says so), because ICP
//     starts where scan 1 was. (11b's features would find it from anywhere: that's 13's way.)
//  3. Set CutAboveFloorMeters = 3: the ceiling comes back and hides everything from above.
//  4. Put a chair where there was none during the scan: it shows up orange too. Anything new is a
//     "visitor" (03 knows nearer, not who). Demo 3's YOLO could name it: that's step 14 proper.
//  5. Point the Kinect at a bare wall and press L: the match % is high, but the camera circle
//     stays where it started along the wall. A flat wall can't say where along it you are, so
//     13's ICP leaves that direction alone (its "weak directions").
