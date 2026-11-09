// =====================================================================
//  Demo 4 · Room Twin                 (presentation; from 13e, 08 and 03)
//
//  The room you scanned with 13e, as a slowly turning doll's house (the
//  ceiling cut off), with the visitors walking around INSIDE it, live,
//  in 3D. A first cut of roadmap step 15 (objects in the map), for people.
//
//    1. Load      a room folder from 13e: every scan merged into one
//                 surface (spotter::LoadTwinRoom, 13z's "seen_room"), and
//                 its floor (FindTwinFloor, D33's rule).
//    2. Learn     the empty view's depth, as in 03 (StillBackground).
//    3. Locate    where the Kinect stands in the room: ICP from several
//                 starts around scan 1's spot, turned and shifted; the
//                 best match wins (LocateKinect, D33). The empty view is
//                 matched, so visitors don't confuse it.
//    4. Live      every frame: what is clearly nearer than the empty view
//                 (03's blobs) becomes 3D points, moved into the room by
//                 step 3's pose, and drawn in it.
//
//  Before the show: scan the stand with 13e (README, "Recreate a room"),
//  then put the Kinect back where scan 1 was taken, aimed the same way
//  (within about 50 cm and 30°: the starts and ICP find the rest). Start
//  it with nobody in view.
//
//  Nothing is saved. The room folder holds no people (13e is run with
//  nobody in view); the visitors are only drawn on screen.
//
//  Keys:   L  learn the empty view and locate again (clear the view first)
//          H  visitors highlighted / true colour     O  turning on/off
//          mouse drag / W A S D  turn   wheel / Z X  zoom   R reset
//          F  fullscreen on/off      Q or Esc  quit
//  Run:    Demo4_Room_Twin [room folder]   (default: the newest room under RoomsFolder)
//  No room? It takes the Kinect's own empty view as the room instead (spotter::OneViewRoom):
//  one view only, good for rehearsing.
//  Rooms are 13e's folders (format 2, or 13c-13e's format 1). Legacy 13's room_scan_*.obj files
//  are legacy Demo 4's.
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/room_twin.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/stats.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip along the bottom of the screen (spotter::WithKeys).
const std::string Keys = "L  relearn and locate (clear the view first)   H  highlight visitors   O  turning on/off   drag / W A S D  turn   "
                         "wheel / Z X  zoom   R  reset   F  fullscreen   Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

const std::string RoomsFolder = "room_scans";   // 13e's default, beside the .exe
const float CubeMeters = 0.03f;                 // the room's points: one per 3 cm cube (13's export)

// --- The doll's house ---
const float CutAboveFloorMeters = 1.6f;   // room points higher than this above the floor are hidden
const float UnderFloorMeters = 0.10f;     // ...and readings this far under it (noise, D33)
const float OrbitSeconds = 60;            // one full turn
const float ViewPitchDegrees = 35;        // looking down from above

// --- Visitors: 03's blobs ---
const double SecondsBeforeBackground = 3;
const float ForegroundThresholdMm = 100;
const int CleanupSizePixels = 5;
const int MinBlobPixels = 400;

// --- Locating ---
const int LocateStep = 8;               // every 8th pixel of the empty view, each way
const float MinLocatedShare = 0.6f;     // below this share of matched points, warn (lost: 40%, found: 100% in the test)
// No room folder: the Kinect's tilt, upwards positive, so the one-view room stands level (as Demo 1).
const float OneViewLevelDegrees = 14;

const bool StartFullscreen = true;
const std::string Window = "Demo 4 - room twin";
const cv::Vec3b Orange(0, 160, 255);

// ---------------------------------------------------------------------
//  Step 1: the doll's house
// ---------------------------------------------------------------------

// The viewer works in camera coordinates (Y down, Z ahead); the room is Y up, Z towards the first
// camera: turned 180 degrees about X, it's seen from in front, as 09b's flip.
cv::Point3f ForView(const cv::Point3f& p) { return { p.x, -p.y, -p.z }; }

struct House
{
    std::vector<cv::Point3f> points;   // the room between the floor and the cut, ready for the viewer
    std::vector<cv::Vec3b> colours;
    float floorY = 0, radius = 3;      // how far the room reaches from its middle (90% of it)
    cv::Point3f middle;
};

House MakeHouse(const spotter::TwinRoom& room)
{
    House house;
    spotter::TwinFloor floor = spotter::FindTwinFloor(room);
    for (const std::string& line : floor.report) std::printf("%s\n", line.c_str());
    house.floorY = floor.y;
    for (size_t i = 0; i < room.points.size(); i++)
    {
        const float above = room.points[i].y - floor.y;
        if (above < -UnderFloorMeters || above > CutAboveFloorMeters) continue;
        house.points.push_back(ForView(room.points[i]));
        house.colours.push_back(room.colours[i]);
        house.middle += room.points[i];
    }
    house.middle *= 1.0f / std::max<size_t>(house.points.size(), 1);
    std::vector<float> reach;
    for (const cv::Point3f& p : house.points) reach.push_back(std::hypot(p.x - house.middle.x, -p.z - house.middle.z));
    if (!reach.empty()) house.radius = spotter::Percentile(reach, 0.9);
    house.middle.y = floor.y + CutAboveFloorMeters / 2;
    std::printf("The room: %s, %zu points, %zu below the cut, about %.1f m across\n", room.from.c_str(), room.points.size(), house.points.size(),
                2 * house.radius);
    return house;
}

// ---------------------------------------------------------------------
//  Steps 2-3: learn the empty view, then where the Kinect is in the room
// ---------------------------------------------------------------------

struct Twin
{
    spotter::TwinRoom room;   // empty until loaded or made from the first empty view
    House house;
    spotter::Located located;
    bool ready = false;       // located
    std::vector<cv::Point3f> visitors;   // step 4, this frame's, ready for the viewer
    std::vector<cv::Vec3b> visitorColours;
};

// Step 1: the room folder given on the command line, or the newest one under RoomsFolder. None: the
// room waits for the first empty view (LearnAndLocate).
Twin LoadTwin(int argc, char** argv)
{
    Twin twin;
    std::string problem;
    const std::filesystem::path folder = argc > 1 ? std::filesystem::path(argv[1]) : spotter::NewestRoomFolder(RoomsFolder);
    if (folder.empty() || !spotter::LoadTwinRoom(folder, CubeMeters, twin.room, problem))
        std::printf("%s The Kinect's own empty view becomes the room. For the real thing, scan the room with 13e_Room_Scan.\n",
                    folder.empty() ? ("No room under " + RoomsFolder + ".").c_str() : problem.c_str());
    else twin.house = MakeHouse(twin.room);
    return twin;
}

void LearnAndLocate(Twin& twin, const spotter::RgbdFrame& frame, const spotter::StillBackground& background, spotter::OrbitView& view)
{
    std::vector<cv::Point3f> cloud = frame.depthCamera.Cloud(background.DepthMm()), sample;
    if (twin.room.points.empty())
    {
        twin.room = spotter::OneViewRoom(cloud, frame.registered, frame.depthMm.cols, OneViewLevelDegrees);
        twin.house = MakeHouse(twin.room);
        view.pivot = ForView(twin.house.middle);
        view.distance = 3 * twin.house.radius;
    }
    for (int row = 0; row < frame.depthMm.rows; row += LocateStep)
        for (int column = 0; column < frame.depthMm.cols; column += LocateStep)
            if (!std::isnan(cloud[row * frame.depthMm.cols + column].z)) sample.push_back(cloud[row * frame.depthMm.cols + column]);
    auto start = std::chrono::steady_clock::now();
    twin.located = spotter::LocateKinect(sample, twin.room);
    twin.ready = true;
    cv::Point3f at = twin.located.pose.shift;
    std::printf("Located from %d starts in %.1f s: %.0f%% of the view within 2 cm of the room. Camera at (%+.2f, %.2f, %+.2f) m, "
                "%.2f m above the floor. Best fits:\n%s", twin.located.tried,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), twin.located.share * 100, at.x, at.y, at.z,
                at.y - twin.house.floorY, twin.located.report.c_str());
    if (twin.located.twoAlike) std::printf("  Two fits almost alike: the room looks the same from both. Put the Kinect nearer scan 1's spot, press L.\n");
    if (twin.located.share < MinLocatedShare)
        std::printf("  That's a poor match: is the Kinect where scan 1 was taken, aimed the same way? Move it, clear the view, press L.\n");
}

// ---------------------------------------------------------------------
//  Step 4: the visitors, into the room
// ---------------------------------------------------------------------

void Visitors(const spotter::RgbdFrame& frame, const cv::Mat& mask, bool highlight, Twin& twin)
{
    const spotter::Pose& pose = twin.located.pose;
    std::vector<cv::Point3f>& points = twin.visitors;
    std::vector<cv::Vec3b>& colours = twin.visitorColours;
    points.clear();
    colours.clear();
    if (!twin.ready) return;
    for (int i = 0; i < (int)mask.total(); i++)
    {
        if (!mask.data[i]) continue;
        const int row = i / mask.cols, column = i % mask.cols;
        cv::Point3f p = frame.depthCamera.Unproject(cv::Point2f((float)column, (float)row), frame.depthMm.at<float>(i) / 1000);
        points.push_back(ForView(pose.turn * p + pose.shift));
        cv::Vec3b colour = frame.registered.at<cv::Vec3b>(i);
        colours.push_back(highlight ? colour / 2 + Orange / 2 : colour);
    }
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

void Show(const Twin& twin, const spotter::StillBackground& background, spotter::OrbitView& view, spotter::PointRenderer& renderer)
{
    cv::Mat& image = renderer.Begin(view, cv::Scalar(20, 20, 20));
    renderer.Draw(twin.house.points, twin.house.colours, 3);   // 3 cm apart: about 2-3 px from a few metres
    renderer.Draw(twin.visitors, twin.visitorColours, 2);
    cv::Point pixel;
    float depth;
    if (twin.ready && view.Project(ForView(twin.located.pose.shift), pixel, depth))
    {
        cv::circle(image, pixel, 8, cv::Scalar(0, 255, 255), 2);
        cv::putText(image, "camera", pixel + cv::Point(12, 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
    }
    std::string title = twin.ready ? "The room, scanned in 3D - and you, live, inside it"
                                   : cv::format("Clear the view: learning the room in %.0f s", std::max(1.0, std::ceil(background.SecondsLeft())));
    cv::putText(image, title, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(255, 255, 255), 2);
    cv::putText(image, (twin.ready ? cv::format("matched %.0f%%   ", twin.located.share * 100) : std::string()) + "live only: nothing is recorded",
                cv::Point(20, image.rows - 15), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(160, 160, 160), 1);
    cv::imshow(Window, spotter::WithKeys(image, Keys));
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!kinect.IsOpen()) return 1;

    Twin twin = LoadTwin(argc, argv);                                                                   // step 1
    spotter::OrbitView view(spotter::OrbitView::Options(), ForView(twin.house.middle), 3 * twin.house.radius);
    view.pitchDegrees = ViewPitchDegrees;
    spotter::OpenExhibitWindow(Window, cv::Size(view.options.width, view.options.height), StartFullscreen);
    view.Attach(Window);
    spotter::PointRenderer renderer;
    spotter::StillBackground background(SecondsBeforeBackground);
    spotter::RgbdFrame frame;
    bool highlight = true, orbit = true;
    double last = 0;

    while (kinect.Next(frame))
    {
        if (background.RecordWhenDue(frame.depthMm)) LearnAndLocate(twin, frame, background, view);   // steps 2-3
        std::vector<cv::Rect> boxes;
        cv::Mat mask = background.Foreground(frame.depthMm, ForegroundThresholdMm, CleanupSizePixels, MinBlobPixels, boxes);
        Visitors(frame, mask, highlight, twin);                                                         // step 4
        if (orbit && last > 0) view.yawDegrees += 360 * (float)(frame.seconds - last) / OrbitSeconds;
        last = frame.seconds;
        Show(twin, background, view, renderer);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 'h') highlight = !highlight;
        if (key == 'o') orbit = !orbit;
        if (key == 'f') spotter::ToggleFullscreen(Window);
        if (key == 'l') { twin.ready = false; background.Restart(); }
        if (view.HandleKey(key) && key == 'r') view.pitchDegrees = ViewPitchDegrees;
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Walk to a corner of the stand: does your orange shell stand in the same corner of the
//     scanned room? If it floats beside it, the match was off: look at "matched %" and press L.
//  2. Move the Kinect 40 cm along and press L: it finds itself again, and the yellow "camera"
//     circle moves in the house. Turn it 45 degrees: it gets lost (and says so), because the
//     starts are all near scan 1's spot. (docs/structure-localization-findings.md: a global search.)
//  3. Set CutAboveFloorMeters = 3: the ceiling comes back and hides everything from above.
//  4. Put a chair where there was none during the scan: it shows up orange too. Anything new is a
//     "visitor" (03 knows nearer, not who). Demo 3's YOLO could name it: that's step 15 proper.
//  5. Point the Kinect at a bare wall and press L: the match % is high, but the camera circle
//     stays where it started along the wall. A flat wall can't say where along it you are, so
//     the ICP leaves that direction alone (its "weak directions").
