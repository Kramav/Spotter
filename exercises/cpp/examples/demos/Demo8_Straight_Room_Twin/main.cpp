// =====================================================================
//  Demo 8 · Straight Room Twin        (presentation; Demo 7 for big rooms)
//
//  Demo 7 (the scanned room as a turning doll's house, filled, the ceiling
//  lifted off, the visitors live inside it), made right for a big hall.
//  The user's 80-scan hall (2026-11-16) broke Demo 7 three ways:
//
//    - the far end leaned 9 degrees: each scan is placed against earlier
//      ones, so one scan's small tilt is handed on to every later one. The
//      floor, the ceiling and the walls all leaned alike (a real ramp keeps
//      its walls plumb), and the camera "climbed" 0.9 m across the hall.
//    - the ceiling isn't one height: steps, lower parts, a slope towards
//      the walls. Demo 7 took the highest (6.1 m, on the leaning end) and
//      left everything lower on show.
//    - lamps and beams hang below it.
//
//  Demo 8's three new ideas (straight_room.hpp, beside this file):
//
//    1. Straighten (before merging) every scan stood up straight: turned
//                 about its own camera until its level readings face
//                 straight up or down and its walls stand plumb (a curved
//                 wall stands plumb too); then lifted onto the room's floor
//                 where it saw the floor (the camera stands about as high
//                 over it as before), and in proportion between those where
//                 it didn't.
//    2. Ceiling map  the ceiling found spot by spot: a height per 25 cm of
//                 the plan from the readings facing down, small lower
//                 pieces left out (lamps), the gaps filled from around.
//    3. Hanging   pieces above a door's height that reach down to nothing
//                 (lamps, beams, ducts) are lifted off with the ceiling.
//
//  The rest is Demo 7's:
//
//    1. Load      a room folder from 13e (spotter::LoadTwinScans), the
//                 scans straightened (idea 1), merged into one surface
//                 (MergeTwinRoom, 13z's "seen_room"), its floor
//                 (FindTwinFloor, D33), the ceiling map (idea 2) and what
//                 hangs (idea 3), both set apart: the walls keep every point.
//    1b. Fill     once, on everything but the ceiling and what hangs, up to
//                 the highest ceiling, then cut down to each spot's own
//                 (spotter/room_fill.hpp, legacy Demo 6's filling):
//                 walls, cabinet fronts, level surfaces and table legs;
//                 what nobody saw is painted in from the seen cells around
//                 and marked with blue stripes; dark things that sent no
//                 depth back become purple cut-outs.
//    2. Learn     (Kinect) the empty view's depth, as in 03 (StillBackground).
//    3. Locate    (Kinect) where the Kinect stands in the room: ICP from
//                 several starts around scan 1's spot, turned and shifted;
//                 the best match wins (LocateKinect, D33).
//    4. Live      (Kinect) every frame: what is clearly nearer than the
//                 empty view (03's blobs) becomes 3D points, moved into the
//                 room by step 3's pose, and drawn in it.
//
//  The Kinect: looked for at the start, and every RetryKinectSeconds while
//  it's missing (the picture may pause briefly then). Unplugged mid-show,
//  the visitors go and the room stays; plugged back in, it learns the empty
//  view and locates again, as L does.
//
//  Before the show: scan the stand with 13e (README, "Recreate a room"),
//  then put the Kinect back where scan 1 was taken, aimed the same way
//  (within about 50 cm and 30°: the starts and ICP find the rest). Start
//  it with nobody in view.
//
//  Nothing is saved. The room folder holds no people (13e is run with
//  nobody in view); the visitors are only drawn on screen.
//
//  Keys:   G  fill: marked / unmarked / off
//          C  lifted off: the ceiling and what hangs / the ceiling only / nothing
//          L  learn the empty view and locate again (Kinect; clear the view first)
//          H  visitors highlighted / true colour (Kinect)    O  turning on/off
//          mouse drag / W A S D  turn   wheel / Z X  zoom   R reset
//          F  fullscreen on/off      Q or Esc  quit
//  Run:    Demo8_Straight_Room_Twin [room folder]   (none: an Open window starting in RoomsFolder, the newest room selected)
//  No room? With the Kinect, its own empty view becomes the room instead (spotter::OneViewRoom):
//  one view only, good for rehearsing. With neither, there's nothing to show: it says so and stops.
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/file_dialog.hpp"
#include "spotter/room_fill.hpp"   // step 1b: the filling (13v's)
#include "spotter/room_twin.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/stats.hpp"
#include "spotter/text.hpp"
#include "spotter/view.hpp"

#include "straight_room.hpp"   // Demo 8's new ideas: Straighten, FindCeilingMap, HangingPoints

using namespace spotter::fill;   // Filled, FillHoles, MarkColour, GhostColour

// The keys, shown in a strip along the bottom of the screen (spotter::WithKeys). Without the Kinect, L and H do nothing.
const std::string KeysLive = "G  fill: marked / unmarked / off   C  lift off: ceiling + lamps / ceiling / nothing   L  relearn and locate (clear the view first)   "
                             "H  highlight visitors   O  turning on/off   drag / W A S D  turn   wheel / Z X  zoom   R  reset   F  fullscreen   Q or Esc  quit";
const std::string KeysRoomOnly = "G  fill: marked / unmarked / off   C  lift off: ceiling + lamps / ceiling / nothing   O  turning on/off   drag / W A S D  turn   "
                                 "wheel / Z X  zoom   R  reset   F  fullscreen   Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

const std::string RoomsFolder = SPOTTER_REPO_ROOT "/room_scans";   // 13e's default (gitignored, outside build/)
const float CubeMeters = 0.03f;                 // the room's points: one per 3 cm cube (13's export)

// --- Idea 1: straighten the scans ---
const bool StraightenScans = true;        // false = merged as placed (Demo 7's room): compare the console's Floor/Ceiling lines
// Its limits: straight::StraightenOptions (straight_room.hpp).

// --- Ideas 2-3: the ceiling, spot by spot, and what hangs from it ---
const float CeilingCellMeters = 0.25f;    // one ceiling height per cell this big
const float CeilingBandMeters = 0.15f;    // readings this close under their spot's ceiling are the ceiling (a flush lamp too)
const float HangAboveFloorMeters = 2.1f;  // only pieces above a door's height can hang
const float GuessCeilingMeters = 2.4f;    // no ceiling seen: the walls are filled this high above the floor
// The other limits: straight::CeilingMapOptions and straight::HangingOptions (straight_room.hpp).

// --- The doll's house ---
const float UnderFloorMeters = 0.10f;     // readings this far under the floor are hidden (noise, D33)
const int StartLiftOff = 0;               // 0 = the ceiling and what hangs lifted off, 1 = the ceiling only, 2 = nothing
const float OrbitSeconds = 60;            // one full turn
const float ViewPitchDegrees = 35;        // looking down from above
// Filling the holes: spotter/room_fill.hpp.

// --- The Kinect ---
const int RetryKinectSeconds = 10;        // while it's missing, look for it this often

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
const std::string Window = "Demo 8 - straight room twin";
const cv::Vec3b Orange(0, 160, 255);

// ---------------------------------------------------------------------
//  Step 1: the doll's house
// ---------------------------------------------------------------------

//cv::Point3f ForView(const cv::Point3f& p){
//
//@param1 - p: a point in the room (Y up, Z towards the first camera)
//
//@brief - the room -> the viewer's coordinates (Y down, Z ahead): turned 180 degrees about X, (x, y, z) -> (x, -y,
//         -z), as 09b's flip, so the room is seen from in front. It is its own inverse. O(1).
cv::Point3f ForView(const cv::Point3f& p) { return { p.x, -p.y, -p.z }; }

struct House
{
    std::vector<cv::Point3f> points;          // the room without the ceiling, ready for the viewer
    std::vector<cv::Vec3b> colours;
    std::vector<cv::Point3f> ceilingPoints;   // the ceiling's own (C shows them), ready for the viewer
    std::vector<cv::Vec3b> ceilingColours;
    std::vector<cv::Point3f> hangingPoints;   // what hangs from it (C shows them), ready for the viewer
    std::vector<cv::Vec3b> hangingColours;
    spotter::TwinFloor floor;
    straight::CeilingMap ceiling;             // idea 2
    float topY = 0;                           // the highest ceiling (the walls are filled up to it, then cut to their spot's)
    int hangingPieces = 0;
    float radius = 3;                         // how far the room reaches from its middle (90% of it)
    cv::Point3f middle;
    Filled filled;                            // step 1b: the guessed points, as the room's
    std::vector<cv::Point3f> filledPoints, ghostPoints;   // ...ready for the viewer
    std::vector<cv::Vec3b> markedColours, ghostColours;   // the guess tinted blue in stripes; the shadows purple
};

//void Fill(House& house, const std::vector<cv::Point3f>& points, const std::vector<cv::Point3f>& normals, const std::vector<cv::Vec3b>& colours, const spotter::TwinRoom& room){
//
//@param1 - house: the house, its floor and ceiling found; its filled points are added
//@param2 - points: the room's points above the under-floor noise, the ceiling and what hangs left out (else the
//          fill's level step would take a lamp or the ceiling for a table top and grow legs up to it)
//@param3 - normals: which way each faces
//@param4 - colours: each one's colour
//@param5 - room: the loaded room (for scan 1's camera)
//
//@brief - step 1b: fills the holes once, up to the highest ceiling (FillHoles), then drops what was filled above its
//         own spot's ceiling (idea 2: a wall under a lower part of the ceiling stops there); prints the areas and time,
//         and keeps the filled points (for the viewer) with their marked colours and the purple shadow cut-outs.
//         O(FillHoles).
void Fill(House& house, const std::vector<cv::Point3f>& points, const std::vector<cv::Point3f>& normals, const std::vector<cv::Vec3b>& colours,
          const spotter::TwinRoom& room)
{
    auto start = std::chrono::steady_clock::now();
    const Filled all = FillHoles(points, normals, colours, house.floor.y, house.floor.seen, house.topY, room.firstCamera);
    Filled& filled = house.filled = all;
    auto UnderCeiling = [&](const cv::Point3f& p) { return !house.ceiling.seen || p.y <= house.ceiling.At(p.x, p.z) + 0.05f; };
    filled.points.clear(); filled.colours.clear(); filled.stripe.clear();
    filled.ghosts.clear(); filled.ghostStripe.clear(); filled.ghostPatches.clear();
    for (size_t i = 0; i < all.points.size(); i++)
        if (UnderCeiling(all.points[i])) { filled.points.push_back(all.points[i]); filled.colours.push_back(all.colours[i]); filled.stripe.push_back(all.stripe[i]); }
    for (size_t i = 0; i < all.ghosts.size(); i++)
        if (UnderCeiling(all.ghosts[i])) { filled.ghosts.push_back(all.ghosts[i]); filled.ghostStripe.push_back(all.ghostStripe[i]); filled.ghostPatches.push_back(all.ghostPatches[i]); }
    const size_t dropped = all.points.size() - filled.points.size();
    std::printf("Filled %.1f m^2 in %.1f s: floor %.1f m^2, %d walls %.1f m^2, %d other surfaces %.1f m^2, %d legs, %d dark shadows; "
                "then %.1f m^2 of it over its spot's ceiling dropped (%zu points)\n",
                filled.floorArea + filled.wallArea + filled.otherArea, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                filled.floorArea, filled.walls, filled.wallArea, filled.others, filled.otherArea, filled.legs, filled.shadows,
                dropped * FillCellMeters * FillCellMeters, dropped);
    for (const std::string& line : filled.report) std::printf("%s\n", line.c_str());
    for (size_t i = 0; i < filled.points.size(); i++)
    {
        float tint = filled.stripe[i] ? 0.65f : 0.3f;
        house.filledPoints.push_back(ForView(filled.points[i]));
        house.markedColours.push_back(cv::Vec3b(cv::Vec3f(filled.colours[i]) * (1 - tint) + cv::Vec3f(MarkColour) * tint));
    }
    for (size_t i = 0; i < filled.ghosts.size(); i++)
    {
        house.ghostPoints.push_back(ForView(filled.ghosts[i]));
        house.ghostColours.push_back(filled.ghostStripe[i] ? GhostColour : GhostColour / 2);
    }
}

//House MakeHouse(const spotter::TwinRoom& room){
//
//@param1 - room: the loaded room
//
//@brief - the room as a doll's house: the floor (FindTwinFloor), the ceiling spot by spot (idea 2, FindCeilingMap) and
//         what hangs from it (idea 3, HangingPoints); the points from just under the floor up, the ceiling's and the
//         hanging ones kept apart, all turned for the viewer; their middle (halfway up) and radius; then the fill
//         (Fill). O(points + FillHoles).
House MakeHouse(const spotter::TwinRoom& room)
{
    House house;
    house.floor = spotter::FindTwinFloor(room);
    for (const std::string& line : house.floor.report) std::printf("%s\n", line.c_str());
    straight::CeilingMapOptions ceilingOptions;
    ceilingOptions.cellMeters = CeilingCellMeters;
    ceilingOptions.bandMeters = CeilingBandMeters;
    house.ceiling = straight::FindCeilingMap(room, house.floor.y, ceilingOptions);
    for (const std::string& line : house.ceiling.report) std::printf("%s\n", line.c_str());
    house.topY = house.ceiling.seen ? house.ceiling.highest : house.floor.y + GuessCeilingMeters;
    const std::vector<char> isCeiling = straight::CeilingMapPoints(room, house.ceiling, ceilingOptions);
    straight::HangingOptions hangingOptions;
    hangingOptions.aboveFloorMeters = HangAboveFloorMeters;
    const std::vector<char> hangs = straight::HangingPoints(room, isCeiling, house.floor.y, house.hangingPieces, hangingOptions);
    std::vector<cv::Point3f> kept, keptNormals;   // for the fill, in the room's own coordinates
    std::vector<cv::Vec3b> keptColours;
    for (size_t i = 0; i < room.points.size(); i++)
    {
        if (room.points[i].y < house.floor.y - UnderFloorMeters) continue;
        if (isCeiling[i]) { house.ceilingPoints.push_back(ForView(room.points[i])); house.ceilingColours.push_back(room.colours[i]); continue; }
        if (hangs[i]) { house.hangingPoints.push_back(ForView(room.points[i])); house.hangingColours.push_back(room.colours[i]); continue; }
        house.points.push_back(ForView(room.points[i]));
        house.colours.push_back(room.colours[i]);
        house.middle += room.points[i];
        kept.push_back(room.points[i]);
        keptNormals.push_back(room.normals[i]);
        keptColours.push_back(room.colours[i]);
    }
    house.middle *= 1.0f / std::max<size_t>(kept.size(), 1);
    std::vector<float> reach;
    for (const cv::Point3f& p : kept) reach.push_back(std::hypot(p.x - house.middle.x, p.z - house.middle.z));
    if (!reach.empty()) house.radius = spotter::Percentile(reach, 0.9);
    house.middle.y = (house.floor.y + (house.ceiling.seen ? house.ceiling.lowest : house.topY)) / 2;
    std::printf("The room: %s, %zu points (%zu of them the ceiling, %zu hanging from it in %d pieces), about %.1f m across\n", room.from.c_str(),
                house.points.size() + house.ceilingPoints.size() + house.hangingPoints.size(), house.ceilingPoints.size(), house.hangingPoints.size(),
                house.hangingPieces, 2 * house.radius);
    Fill(house, kept, keptNormals, keptColours, room);
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

//Twin LoadTwin(int argc, char** argv, bool haveKinect){
//
//@param1 - argc: how many command-line words there are
//@param2 - argv: the words: an optional room folder
//@param3 - haveKinect: true if the Kinect is open (then a missing room can be made from its view)
//
//@brief - step 1: the room folder given, or one chosen in an Open window (ChooseRoom): its scans loaded
//         (LoadTwinScans), straightened (idea 1, Straighten; StraightenScans), merged (MergeTwinRoom) and made a doll's
//         house (MakeHouse). None: with the Kinect the room waits for the first empty view (LearnAndLocate); without
//         it, the room stays empty. O(scans x pixels + FuseScans).
Twin LoadTwin(int argc, char** argv, bool haveKinect)
{
    Twin twin;
    std::string problem;
    const std::filesystem::path folder = spotter::ChooseRoom(argc, argv, RoomsFolder, "Demo8");
    spotter::RoomInfo info;
    std::vector<spotter::Scan> scans;
    if (!folder.empty() && spotter::LoadTwinScans(folder, info, scans, problem))
    {
        if (StraightenScans)
        {
            std::vector<spotter::Scan*> order;
            for (spotter::Scan& scan : scans) order.push_back(&scan);
            const straight::Straightened straightened = straight::Straighten(order, info.firstCamera, straight::StraightenOptions());
            for (const std::string& line : straightened.report) std::printf("%s\n", line.c_str());
        }
        else std::printf("The scans merged as placed (StraightenScans is off)\n");
        twin.room = spotter::MergeTwinRoom(scans, info, CubeMeters, folder.filename().string());
        twin.house = MakeHouse(twin.room);
        return twin;
    }
    std::printf("%s %s For the real thing, scan the room with 13e_Room_Scan.\n",
                folder.empty() ? "No room chosen." : ("Can't load " + folder.string() + ": " + problem).c_str(),
                haveKinect ? "The Kinect's own empty view becomes the room." : "And no Kinect either: nothing to show.");
    return twin;
}

//void LearnAndLocate(Twin& twin, const spotter::RgbdFrame& frame, const spotter::StillBackground& background, spotter::OrbitView& view){
//
//@param1 - twin: the room and where the Kinect is; updated
//@param2 - frame: this frame
//@param3 - background: the recorded empty view
//@param4 - view: the viewer, repositioned for a new room
//
//@brief - with no scanned room the Kinect's own empty view becomes the room (OneViewRoom, then MakeHouse). Then
//         LocateKinect from every LocateStep-th pixel of the empty view finds where the Kinect stands in it;
//         prints the starts, time, share matched, the camera's place and height, the best fits, and warns about
//         two alike fits or a poor match. O(LocateKinect).
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
                at.y - twin.house.floor.y, twin.located.report.c_str());
    if (twin.located.twoAlike) std::printf("  Two fits almost alike: the room looks the same from both. Put the Kinect nearer scan 1's spot, press L.\n");
    if (twin.located.share < MinLocatedShare)
        std::printf("  That's a poor match: is the Kinect where scan 1 was taken, aimed the same way? Move it, clear the view, press L.\n");
}

// ---------------------------------------------------------------------
//  Step 4: the visitors, into the room
// ---------------------------------------------------------------------

//void Visitors(const spotter::RgbdFrame& frame, const cv::Mat& mask, bool highlight, Twin& twin){
//
//@param1 - frame: this frame
//@param2 - mask: the foreground mask: the visitors
//@param3 - highlight: true = tint them orange
//@param4 - twin: the room and where the Kinect is; its visitor points are replaced
//
//@brief - the foreground pixels as 3D points in the room (each from its own depth, moved by the located pose and
//         turned for the viewer) with their colour. Nothing until the room is located. O(pixels).
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
//  The Kinect, there or not
// ---------------------------------------------------------------------

//std::unique_ptr<spotter::KinectSource> OpenKinect(){
//
//@brief - the Kinect if one opens, else nullptr (KinectSource prints why). O(device enumeration).
std::unique_ptr<spotter::KinectSource> OpenKinect()
{
    auto kinect = std::make_unique<spotter::KinectSource>(spotter::KinectSource::Options{});
    if (!kinect->IsOpen()) return nullptr;
    return kinect;
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

struct Look
{
    int fillMode = 0;             // 0 marked, 1 unmarked, 2 off (Demo 4's look)
    int liftOff = StartLiftOff;   // 0 the ceiling and what hangs lifted off, 1 the ceiling only, 2 nothing (all shown)
    bool highlight = true, orbit = true;
};

//void Show(const Twin& twin, const spotter::StillBackground* background, const Look& look, spotter::OrbitView& view, spotter::PointRenderer& renderer){
//
//@param1 - twin: the room, the fill and the visitors
//@param2 - background: the empty view being learnt, for the countdown; nullptr = no Kinect (room only)
//@param3 - look: fill marked / unmarked / off, what's lifted off
//@param4 - view: the viewer
//@param5 - renderer: draws the points
//
//@brief - the doll's house (the ceiling and what hangs, if shown), the fill (marked, unmarked or off) and the purple shadows in the
//         marked view, the visitors, a circle where the camera is, the title, what the colours mean, the ceiling, and
//         the Kinect's state; shown with the key strip. O(points).
void Show(const Twin& twin, const spotter::StillBackground* background, const Look& look, spotter::OrbitView& view, spotter::PointRenderer& renderer)
{
    const bool haveKinect = background != nullptr;
    const House& house = twin.house;
    cv::Mat& image = renderer.Begin(view, cv::Scalar(20, 20, 20));
    renderer.Draw(house.points, house.colours, 3);   // 3 cm apart: about 2-3 px from a few metres
    if (look.liftOff == 2) renderer.Draw(house.ceilingPoints, house.ceilingColours, 3);
    if (look.liftOff >= 1) renderer.Draw(house.hangingPoints, house.hangingColours, 3);
    if (look.fillMode < 2) renderer.Draw(house.filledPoints, look.fillMode == 0 ? house.markedColours : house.filled.colours, 3);
    // ponytail: legacy sized each cut-out by its distance (ghostPatches); one size here.
    if (look.fillMode == 0) renderer.Draw(house.ghostPoints, house.ghostColours, 3);
    renderer.Draw(twin.visitors, twin.visitorColours, 2);
    cv::Point pixel;
    float depth;
    if (haveKinect && twin.ready && view.Project(ForView(twin.located.pose.shift), pixel, depth))
    {
        cv::circle(image, pixel, 8, cv::Scalar(0, 255, 255), 2);
        spotter::PutText(image, "camera", pixel + cv::Point(12, 5), 0.6, cv::Scalar(0, 255, 255), 1);
    }
    std::string title = !haveKinect || twin.ready ? "The room, scanned in 3D - holes filled by software"
                                                  : cv::format("Clear the view: learning the room in %.0f s", std::max(1.0, std::ceil(background->SecondsLeft())));
    spotter::PutText(image, title, cv::Point(20, 40), 0.85, cv::Scalar(255, 255, 255), 2);
    const char* fillNote[3] = { "blue stripes = filled in by software (never seen)", "filled in by software, unmarked", "only what the Kinect saw" };
    spotter::PutText(image, fillNote[look.fillMode], cv::Point(20, 70), 0.6, look.fillMode == 0 ? cv::Scalar(MarkColour) : cv::Scalar(200, 200, 200), 2);
    const std::string heights = cv::format("%.1f-%.1f m up", house.ceiling.lowest - house.floor.y, house.ceiling.highest - house.floor.y);
    const std::string ceiling = !house.ceiling.seen ? (look.liftOff == 0 ? "no ceiling found; what hangs lifted off" : "no ceiling found: the room is shown whole")
                              : look.liftOff == 0   ? "ceiling (" + heights + ") and what hangs from it lifted off so you can see in"
                              : look.liftOff == 1   ? "ceiling (" + heights + ") lifted off; lamps and beams shown"
                                                    : "ceiling shown, " + heights;
    spotter::PutText(image, ceiling, cv::Point(20, 95), 0.6, cv::Scalar(200, 200, 200), 1);
    if (look.fillMode == 0 && !house.ghostPoints.empty())
        spotter::PutText(image, "purple = something dark the Kinect got no reading from (its shadow on the wall)", cv::Point(20, 120), 0.6,
                         cv::Scalar(GhostColour), 2);
    const std::string state = !haveKinect ? cv::format("no Kinect: the room only (plug it in: it's looked for every %d s)", RetryKinectSeconds)
                            : (twin.ready ? cv::format("matched %.0f%%   ", twin.located.share * 100) : std::string()) + "live only: nothing is recorded";
    spotter::PutText(image, state, cv::Point(20, image.rows - 15), 0.55, cv::Scalar(160, 160, 160), 1);
    cv::imshow(Window, spotter::WithKeys(image, haveKinect ? KeysLive : KeysRoomOnly));
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

//int main(int argc, char** argv){
//
//@param1 - argc: how many command-line words there are
//@param2 - argv: the words; the file's header says what it accepts
//
//@brief - the program: Demo 8 · Straight Room Twin (presentation; Demo 7 for big rooms). The scanned room's
//         scans straightened, merged into a turning doll's house, filled, the ceiling found spot by spot and lifted off
//         with what hangs from it, the visitors live inside it when the Kinect is there. Setup, main loop and clean-up are laid out in the file's sections. Returns 0 on a normal exit, 1 if
//         there was nothing to show.
int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::unique_ptr<spotter::KinectSource> kinect = OpenKinect();
    if (!kinect) std::printf("No Kinect: showing the room only. Plug it in any time: it's looked for every %d s.\n", RetryKinectSeconds);

    Twin twin = LoadTwin(argc, argv, kinect != nullptr);                                                // step 1
    if (!kinect && twin.room.points.empty())
    {
        spotter::Tell("Demo 8", "No room and no Kinect: nothing to show.\n\nChoose a room folder (13e saves them), or plug the Kinect in "
                                "so its own view becomes the room.");
        return 1;
    }
    spotter::OrbitView view(spotter::OrbitView::Options(), ForView(twin.house.middle), 3 * twin.house.radius);
    view.pitchDegrees = ViewPitchDegrees;
    spotter::OpenExhibitWindow(Window, cv::Size(view.options.width, view.options.height), StartFullscreen);
    view.Attach(Window);
    spotter::PointRenderer renderer;
    // The empty view's recorder: only while there's a Kinect (it starts counting down, and says so, when made).
    std::unique_ptr<spotter::StillBackground> background;
    if (kinect) background = std::make_unique<spotter::StillBackground>(SecondsBeforeBackground);
    spotter::RgbdFrame frame;
    spotter::Every lookForKinect(RetryKinectSeconds * 1000);
    Look look;
    auto last = std::chrono::steady_clock::now();

    while (true)
    {
        if (kinect && !kinect->Next(frame))   // unplugged: the room stays, the visitors go
        {
            std::printf("The Kinect stopped sending: showing the room only. Plug it back in: it's looked for every %d s.\n", RetryKinectSeconds);
            kinect.reset();
            background.reset();
            twin.ready = false;
            twin.visitors.clear();
            twin.visitorColours.clear();
        }
        else if (kinect)
        {
            if (background->RecordWhenDue(frame.depthMm)) LearnAndLocate(twin, frame, *background, view);   // steps 2-3
            std::vector<cv::Rect> boxes;
            cv::Mat mask = background->Foreground(frame.depthMm, ForegroundThresholdMm, CleanupSizePixels, MinBlobPixels, boxes);
            Visitors(frame, mask, look.highlight, twin);                                                  // step 4
        }
        else if (lookForKinect.Due() && (kinect = OpenKinect()))
        {
            std::printf("Found the Kinect: it learns the empty view, then locates itself.\n");
            twin.ready = false;
            background = std::make_unique<spotter::StillBackground>(SecondsBeforeBackground);
        }
        const auto now = std::chrono::steady_clock::now();
        if (look.orbit) view.yawDegrees += 360 * (float)std::chrono::duration<double>(now - last).count() / OrbitSeconds;
        last = now;
        Show(twin, background.get(), look, view, renderer);

        int key = cv::waitKey(kinect ? 1 : 15);
        if (spotter::IsQuitKey(key) || cv::getWindowProperty(Window, cv::WND_PROP_VISIBLE) < 1) break;
        if (key == 'g') look.fillMode = (look.fillMode + 1) % 3;
        if (key == 'c') look.liftOff = (look.liftOff + 1) % 3;
        if (key == 'h') look.highlight = !look.highlight;
        if (key == 'o') look.orbit = !look.orbit;
        if (key == 'f') spotter::ToggleFullscreen(Window);
        if (key == 'l' && background) { twin.ready = false; background->Restart(); }
        if (view.HandleKey(key) && key == 'r') view.pitchDegrees = ViewPitchDegrees;
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Set StraightenScans = false and run the same room: read the console's Floor and Ceiling
//     lines against the straightened run's (the user's hall: the ceiling 6.1 m -> 5.3 m at most, the
//     floor's biggest flat piece 14.6 -> 16.3 m^2). Turn the view low (S): the far end leans.
//  2. Read the "Straightened" lines: the most-tilted scan, and where the tilt began. Scans
//     placed against a tilted one inherit it: the user's hall jumped 6 degrees at scan 36.
//  3. Press C twice (everything shown), then once more: the lamps and beams go with the ceiling.
//     Set HangAboveFloorMeters = 3: lamps hung lower stay.
//  4. Set CeilingCellMeters = 1.0: the ceiling's steps blur, and the walls under its lower parts
//     keep a strip of the higher ceiling's fill.
//  5. Press G three times: marked, unmarked, off. Off is Demo 4's look: count the holes it had.
//  6. Unplug the Kinect mid-show: the visitors go, the room stays. Plug it back in: within
//     RetryKinectSeconds it learns the empty view again (clear the view) and locates itself.
//  7. Read the console's list of surfaces: does every real wall show up as "wall", and how much
//     does each bend? A curved wall should say so.
//  8. A doorway nothing was seen through (a closed door, or a dark hallway the Kinect read nothing
//     in) gets filled shut: the filling can't tell a door from a hole. Open the door and scan through it.
//  9. Put a black office chair in front of a wall and scan: the Kinect reads almost nothing of it,
//     but its purple cut-out appears where its legs were read.
// 10. Stand where scan 1 was taken and walk to the far wall: on the screen, you walk under the
//     lifted-off ceiling to a wall that's as tall as the real one.
