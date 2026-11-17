// =====================================================================
//  Demo 6 · Room Filled               (presentation; Demo 4 + hole filling)
//
//  Demo 4's room twin (the scanned room as a turning doll's house, the
//  visitors live inside it), with the holes filled in. A scan only has
//  what the Kinect saw: the wall behind the couch, the floor under the
//  table, the part of a table top under a box were never seen. Here the
//  flat surfaces are found and their holes painted in, and the filled-in
//  parts are MARKED (blue stripes), so visitors can tell real from guessed.
//
//    1. Load      the room (Demo 4: a 13e room folder, or the Kinect's own view).
//    1b. Fill     (new) once, assuming a simple layout: walls upright,
//                 surfaces level (D32):
//                 a. walls: seen from above, upright points stacked over
//                    1 m high are a wall's path along the floor (straight,
//                    bent or curved). Each path is unrolled into a flat
//                    picture and filled from the floor up to the cut;
//                 b. cabinet fronts, couch backs: upright planes (RANSAC);
//                 c. level surfaces (floor, table tops, seats): crowded
//                    heights of up-facing points. The floor is the lowest
//                    one big enough to be a floor (D33, spotter::FindTwinFloor);
//                    a floor nobody saw is guessed at the walls' foot;
//                 d. table legs: thin, tall blobs of readings under a top,
//                    made whole from the floor to the top;
//                 e. a cell counts as seen if a real point is near it, or
//                    just behind it; points far behind a wall mean an
//                    opening (a door). The rest is filled, coloured from
//                    the seen cells around (cv::inpaint, 07e's plan).
//    1c. Shadows  (new) black, glossy or see-through things send no depth
//                 back, but leave their shadow on the wall behind. A hole
//                 nothing seen explains becomes a purple cut-out of
//                 "something dark", where its few readings (chair legs)
//                 put it.
//    2-4.         as Demo 4: learn the empty view, locate (from several
//                 starts around scan 1's spot, D33), live visitors.
//
//  It only fills FLAT things and legs it has readings of. It never invents
//  a chair back or a plant, and big holes get smeared colour.
//
//  Nothing is saved. The room file holds no people (scanned with nobody
//  in view); the visitors are only drawn on screen.
//
//  Keys:   G  fill: marked / unmarked / off   H  visitors highlighted / true colour
//          O  turning on/off   L  learn the empty view and locate again
//          mouse drag / W A S D  turn   wheel / Z X  zoom   R reset
//          F  fullscreen on/off      Q or Esc  quit
//  Run:    Demo6_Room_Filled [room folder]   (none: an Open window starting in RoomsFolder, the newest room selected)
//  No room? As Demo 4, the Kinect's own empty view becomes the room (spotter::OneViewRoom):
//  one view has big shadows behind things, so there's plenty to fill.
//  The filling is legacy Demo 6's (D31-D33), unchanged, shared with 13x (13x_Room_Fill/fill_holes.hpp);
//  the rest is Demo 4's.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/room_fill.hpp"   // was fill_holes.hpp beside 13x; in the library since D58
#include "spotter/depth.hpp"
#include "spotter/room_twin.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/stats.hpp"
#include "spotter/text.hpp"
#include "spotter/view.hpp"

using namespace spotter::fill;   // retired (D58): not built; this line keeps it compiling against the library

// The keys, shown in a strip along the bottom of the screen (spotter::WithKeys).
const std::string Keys = "G  fill: marked / unmarked / off   L  relearn and locate (clear the view first)   H  highlight visitors   O  turning on/off   drag / W A S D  turn   "
                         "wheel / Z X  zoom   R  reset   F  fullscreen   Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

const std::string RoomsFolder = SPOTTER_REPO_ROOT "/room_scans";   // 13e's default (gitignored, outside build/)
const float CubeMeters = 0.03f;                 // the room's points: one per 3 cm cube (13's export)

// --- The doll's house ---
const float CutAboveFloorMeters = 1.6f;   // room points higher than this above the floor are hidden
const float UnderFloorMeters = 0.10f;     // ...and readings this far under it (noise, D33)
const float OrbitSeconds = 60;            // one full turn
const float ViewPitchDegrees = 35;        // looking down from above

// --- Filling the holes (new): 13x's, its settings and code in ../../13x_Room_Fill/fill_holes.hpp. ---

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
const std::string Window = "Demo 6 - room filled";
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
    Filled filled;                     // step 1b (new): the guessed points, as the room's
    std::vector<cv::Point3f> filledPoints, ghostPoints;   // ...ready for the viewer
    std::vector<cv::Vec3b> markedColours, ghostColours;   // the guess tinted blue in stripes; the shadows purple
};

//void Fill(House& house, const spotter::TwinRoom& room, const spotter::TwinFloor& floor){
//
//@param1 - house: the house; its filled points are added
//@param2 - room: the loaded room
//@param3 - floor: its floor
//
//@brief - step 1b (new): fills the holes once, in what's above the under-floor noise (FillHoles up to the cut),
//         prints the areas and time, and keeps the filled points (for the viewer, Y turned) with their marked
//         colours and the purple shadow cut-outs. O(FillHoles).
void Fill(House& house, const spotter::TwinRoom& room, const spotter::TwinFloor& floor)
{
    std::vector<cv::Point3f> kept, keptNormals;
    std::vector<cv::Vec3b> keptColours;
    for (size_t i = 0; i < room.points.size(); i++)
        if (room.points[i].y >= floor.y - UnderFloorMeters)
        {
            kept.push_back(room.points[i]);
            keptNormals.push_back(room.normals[i]);
            keptColours.push_back(room.colours[i]);
        }
    auto start = std::chrono::steady_clock::now();
    Filled& filled = house.filled = FillHoles(kept, keptNormals, keptColours, floor.y, floor.seen, floor.y + CutAboveFloorMeters, room.firstCamera);
    std::printf("Filled %.1f m^2 in %.1f s: floor %.1f m^2, %d walls %.1f m^2, %d other surfaces %.1f m^2, %d legs, %d dark shadows\n",
                filled.floorArea + filled.wallArea + filled.otherArea, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                filled.floorArea, filled.walls, filled.wallArea, filled.others, filled.otherArea, filled.legs, filled.shadows);
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
//@brief - the room as a doll's house: the floor (FindTwinFloor), the points from just under the floor up to
//         CutAboveFloorMeters (turned for the viewer), their middle and radius, then the fill (Fill). O(points).
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
    Fill(house, room, floor);
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

//Twin LoadTwin(int argc, char** argv){
//
//@param1 - argc: how many command-line words there are
//@param2 - argv: the words: an optional room folder
//
//@brief - step 1: the room folder given, or one chosen in an Open window (ChooseRoom). None: the room waits for the first empty
//         view (LearnAndLocate). O(LoadTwinRoom).
Twin LoadTwin(int argc, char** argv)
{
    Twin twin;
    std::string problem;
    const std::filesystem::path folder = spotter::ChooseRoom(argc, argv, RoomsFolder, "Demo6");
    if (folder.empty() || !spotter::LoadTwinRoom(folder, CubeMeters, twin.room, problem))
        std::printf("%s The Kinect's own empty view becomes the room. For the real thing, scan the room with 13e_Room_Scan.\n",
                    folder.empty() ? "No room chosen." : ("Can't load " + folder.string() + ": " + problem).c_str());
    else twin.house = MakeHouse(twin.room);
    return twin;
}

//void LearnAndLocate(Twin& twin, const spotter::RgbdFrame& frame, const spotter::StillBackground& background, spotter::OrbitView& view){
//
//@param1 - twin: the room and where the Kinect is; updated
//@param2 - frame: this frame
//@param3 - background: the recorded empty view
//@param4 - view: the viewer, repositioned for a new room
//
//@brief - with no scanned room the Kinect's own empty view becomes the room (OneViewRoom, then the fill). Then
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
                at.y - twin.house.floorY, twin.located.report.c_str());
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
//  Output
// ---------------------------------------------------------------------

//void Show(const Twin& twin, const spotter::StillBackground& background, int fillMode, spotter::OrbitView& view, spotter::PointRenderer& renderer){
//
//@param1 - twin: the room, the fill and the visitors
//@param2 - background: for the countdown
//@param3 - fillMode: 0 = marked, 1 = unmarked, 2 = off
//@param4 - view: the viewer
//@param5 - renderer: draws the points
//
//@brief - the doll's house, then (new) the fill (marked, unmarked or off) and the purple shadows in the marked
//         view, the visitors, a circle where the camera is, the title, what the colours mean and the matched
//         share; shown with the key strip. O(points).
void Show(const Twin& twin, const spotter::StillBackground& background, int fillMode, spotter::OrbitView& view, spotter::PointRenderer& renderer)
{
    const House& house = twin.house;
    cv::Mat& image = renderer.Begin(view, cv::Scalar(20, 20, 20));
    renderer.Draw(house.points, house.colours, 3);   // 3 cm apart: about 2-3 px from a few metres
    if (fillMode < 2) renderer.Draw(house.filledPoints, fillMode == 0 ? house.markedColours : house.filled.colours, 3);   // new
    // ponytail: legacy sized each cut-out by its distance (ghostPatches); one size here.
    if (fillMode == 0) renderer.Draw(house.ghostPoints, house.ghostColours, 3);
    renderer.Draw(twin.visitors, twin.visitorColours, 2);
    cv::Point pixel;
    float depth;
    if (twin.ready && view.Project(ForView(twin.located.pose.shift), pixel, depth))
    {
        cv::circle(image, pixel, 8, cv::Scalar(0, 255, 255), 2);
        spotter::PutText(image, "camera", pixel + cv::Point(12, 5), 0.6, cv::Scalar(0, 255, 255), 1);
    }
    std::string title = twin.ready ? "The room, scanned in 3D - holes filled by software"
                                   : cv::format("Clear the view: learning the room in %.0f s", std::max(1.0, std::ceil(background.SecondsLeft())));
    spotter::PutText(image, title, cv::Point(20, 40), 0.85, cv::Scalar(255, 255, 255), 2);
    const char* fillNote[3] = { "blue stripes = filled in by software (never seen)", "filled in by software, unmarked", "only what the Kinect saw" };
    spotter::PutText(image, fillNote[fillMode], cv::Point(20, 70), 0.6, fillMode == 0 ? cv::Scalar(MarkColour) : cv::Scalar(200, 200, 200), 2);
    if (fillMode == 0 && !house.ghostPoints.empty())
        spotter::PutText(image, "purple = something dark the Kinect got no reading from (its shadow on the wall)", cv::Point(20, 95),
                    0.6, cv::Scalar(GhostColour), 2);
    spotter::PutText(image, (twin.ready ? cv::format("matched %.0f%%   ", twin.located.share * 100) : std::string()) + "live only: nothing is recorded",
                cv::Point(20, image.rows - 15), 0.55, cv::Scalar(160, 160, 160), 1);
    cv::imshow(Window, spotter::WithKeys(image, Keys));
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

//int main(int argc, char** argv){
//
//@param1 - argc: how many command-line words there are
//@param2 - argv: the words; the file's header says what it accepts
//
//@brief - the program: Demo 6 · Room Filled (presentation; Demo 4 + hole filling). Demo 4's room twin (the scanned
//         room as a turning doll's house, the visitors live inside it), with the holes filled in. Setup, main loop
//         and clean-up are laid out in the file's sections. Returns 0 on a normal exit, 1 if it couldn't start.
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
    int fillMode = 0;   // 0 marked, 1 unmarked, 2 off (Demo 4's look)
    double last = 0;

    while (kinect.Next(frame))
    {
        if (background.RecordWhenDue(frame.depthMm)) LearnAndLocate(twin, frame, background, view);   // steps 2-3
        std::vector<cv::Rect> boxes;
        cv::Mat mask = background.Foreground(frame.depthMm, ForegroundThresholdMm, CleanupSizePixels, MinBlobPixels, boxes);
        Visitors(frame, mask, highlight, twin);                                                         // step 4
        if (orbit && last > 0) view.yawDegrees += 360 * (float)(frame.seconds - last) / OrbitSeconds;
        last = frame.seconds;
        Show(twin, background, fillMode, view, renderer);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 'g') fillMode = (fillMode + 1) % 3;   // new
        if (key == 'h') highlight = !highlight;
        if (key == 'o') orbit = !orbit;
        if (key == 'f') spotter::ToggleFullscreen(Window);
        if (key == 'l') { twin.ready = false; background.Restart(); }
        if (view.HandleKey(key) && key == 'r') view.pitchDegrees = ViewPitchDegrees;
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas (Demo 4's apply too)
// ---------------------------------------------------------------------
//  1. Press G three times: marked, unmarked, off. Off is Demo 4: count the holes it had.
//  2. Read the console's list of surfaces: does every real wall show up as "wall", and how much
//     does each bend? A curved wall should say so.
//  3. A doorway nothing was seen through (a closed door, or a dark hallway the Kinect read nothing
//     in) gets filled shut: the filling can't tell a door from a hole. Open the door and scan through it.
//  4. Set WallBehindMeters = 0.05 (round 1's value): where the drawn wall stands a little in front
//     of the real one, the fill covers what was seen. That's why it's 0.5.
//  5. Set WallStackMeters = 0.5: chair backs and table edges start counting as walls.
//  6. Put a black office chair in front of a wall and scan: the Kinect reads almost nothing of it,
//     but its purple cut-out appears where its legs were read.
//  7. Set LegCrossShare = 0.8: the far legs, whose upper part the table top itself hides from the
//     camera, are no longer found. Push a chair under the table: its legs aren't taken for the
//     table's (its seat is on them), but the chair seat gets its own legs.
//  8. Set InpaintRadiusCells = 1: big holes get the colour of the nearest edge only, in streaks.
