// =====================================================================
//  13v · Room Fill                          (no Kinect; after 13e and 13z)
//
//  A scan only has what the Kinect saw: the wall behind the couch, the
//  floor under the table, the part of a table top under a box were never
//  seen. 13v finds the room's flat surfaces and paints their holes in,
//  and MARKS what it filled (blue stripes), so real and guessed stay
//  apart. It assumes a simple layout: walls upright, surfaces level (D32).
//
//  How it differs from 13x (retired, legacy/retired/13x_Room_Fill, D58):
//  13x hid everything over 1.6 m above the floor, which took the ceiling
//  off but cut every wall's top off with it, and filled the walls only up
//  to 1.6 m. 13v FINDS the ceiling (spotter::FindTwinCeiling, 13w's rule)
//  and hides only the ceiling itself (spotter::CeilingPoints): the walls
//  keep their full height, and are filled right up to the ceiling. C shows
//  or hides the ceiling.
//
//    1. Load      a 13e room folder: every included scan merged at 3 cm,
//                 levelled; its floor (spotter::FindTwinFloor, D33) and
//                 its ceiling (spotter::FindTwinCeiling, D58): the highest
//                 level readings covering 0.5 m^2, at least 1.8 m up.
//                 The ceiling's own points (level, within 10 cm of it, or
//                 above it) are set apart (spotter::CeilingPoints).
//    2. Fill      (new), once, on everything but the ceiling, up to the
//                 ceiling (spotter/room_fill.hpp, legacy Demo 6's filling):
//                 a. walls: seen from above, upright points stacked over
//                    1 m high are a wall's path along the floor (straight,
//                    bent or curved). Each path is unrolled into a flat
//                    picture and filled from the floor up to the ceiling;
//                 b. cabinet fronts, couch backs: upright planes (RANSAC);
//                 c. level surfaces (floor, table tops, seats): crowded
//                    heights of up-facing points; a floor nobody saw is
//                    guessed at the walls' foot;
//                 d. table legs: thin, tall blobs of readings under a top,
//                    made whole from the floor to the top;
//                 e. a cell counts as seen if a real point is near it, or
//                    just behind it; points far behind a wall mean an
//                    opening (a door). The rest is filled, coloured from
//                    the seen cells around (cv::inpaint, 07e's plan);
//                 f. shadows: black, glossy or see-through things send no
//                    depth back, but leave their shadow on the wall behind.
//                    A hole nothing seen explains becomes a purple cut-out
//                    of "something dark", where its few readings put it.
//    3. Look      the room as a doll's house, the ceiling lifted off
//                 (C puts it back); G switches marked / unmarked / off.
//    4. Save      K writes room_filled_<date>_<time>.ply into the room's
//                 folder: every real point (the ceiling too) plus the fill
//                 as shown.
//
//  It only fills FLAT things and legs it has readings of. It never invents
//  a chair back or a plant, and big holes get smeared colour.
//
//  Keys:   G  fill: marked / unmarked / off   C  ceiling shown / hidden   K  save the .ply
//          mouse drag / W A S D  turn   wheel / Z X  zoom   R reset   Q or Esc  quit
//  Run:    13v_Room_Fill [room folder]   (none: an Open window starting in RoomsFolder, the newest room selected)
//  Demo 7 shows the same room, with the visitors live inside it when the Kinect is plugged in.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/mesh.hpp"
#include "spotter/room_fill.hpp"   // step 2 (new), also used by Demo 7 and check_fill
#include "spotter/room_twin.hpp"
#include "spotter/stats.hpp"
#include "spotter/text.hpp"
#include "spotter/view.hpp"

using namespace spotter::fill;   // Filled, FillHoles, MarkColour, GhostColour

// The keys, shown in a strip below the window (spotter::WithKeys).
const std::string Keys = "G  fill: marked / unmarked / off   C  ceiling shown / hidden   K  save the .ply   drag / W A S D  turn   wheel / Z X  zoom   "
                         "R  reset   Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

const std::string RoomsFolder = SPOTTER_REPO_ROOT "/room_scans";   // 13e's default (gitignored, outside build/)
const float CubeMeters = 0.03f;                 // the room's points: one per 3 cm cube
const float UnderFloorMeters = 0.10f;           // readings this far under the floor are noise (D33)
const float CeilingBandMeters = 0.10f;          // level readings this close under the ceiling are the ceiling (a flush lamp too)
const bool StartWithCeiling = false;            // false = lifted off at the start, to look in from above
const float ViewPitchDegrees = 35;              // looking down from above
const std::string Window = "13v - room filled";

// --- Finding the ceiling: spotter::CeilingOptions (room_twin.hpp). Filling the holes: spotter/room_fill.hpp. ---

// ---------------------------------------------------------------------
//  Step 1: load the room
// ---------------------------------------------------------------------

struct Room
{
    std::filesystem::path folder;
    spotter::TwinRoom twin;            // every included scan merged, levelled (Y up)
    spotter::TwinFloor floor;
    spotter::TwinCeiling ceiling;
    std::vector<cv::Point3f> points, normals;   // above the under-floor noise, the ceiling left out
    std::vector<cv::Vec3b> colours;
    std::vector<cv::Point3f> ceilingPoints;     // the ceiling's own (C shows them)
    std::vector<cv::Vec3b> ceilingColours;
    cv::Point3f middle;                // for the view
    float radius = 3;
    Filled filled;                     // step 2
    std::vector<cv::Vec3b> marked;     // the guess tinted blue in stripes
};

//bool LoadRoom(int argc, char** argv, Room& room){
//
//@param1 - argc: how many command-line words there are
//@param2 - argv: the words: an optional room folder
//@param3 - room: filled with the points, floor, ceiling and size
//
//@brief - step 1: the room folder given, or one chosen in an Open window (ChooseRoom); LoadTwinRoom, FindTwinFloor and
//         FindTwinCeiling; keeps the points above the under-floor noise, the ceiling's (CeilingPoints) apart, and works
//         out the middle (halfway up) and the radius (90th percentile reach). False (with a message) if there is no
//         room or it won't load. O(points).
bool LoadRoom(int argc, char** argv, Room& room)
{
    room.folder = spotter::ChooseRoom(argc, argv, RoomsFolder, "13v");
    std::string problem;
    if (room.folder.empty())
    {
        std::printf("No room to fill. Scan one with 13e_Room_Scan, or name its folder: 13v_Room_Fill <folder>\n");
        return false;
    }
    if (!spotter::LoadTwinRoom(room.folder, CubeMeters, room.twin, problem)) { std::printf("Can't load %s: %s\n", room.folder.string().c_str(), problem.c_str()); return false; }
    room.floor = spotter::FindTwinFloor(room.twin);
    for (const std::string& line : room.floor.report) std::printf("%s\n", line.c_str());
    room.ceiling = spotter::FindTwinCeiling(room.twin, room.floor.y);
    for (const std::string& line : room.ceiling.report) std::printf("%s\n", line.c_str());
    const std::vector<char> isCeiling = spotter::CeilingPoints(room.twin, room.ceiling, CeilingBandMeters);
    std::vector<float> reach;
    for (size_t i = 0; i < room.twin.points.size(); i++)
    {
        const cv::Point3f& p = room.twin.points[i];
        if (p.y < room.floor.y - UnderFloorMeters) continue;
        if (isCeiling[i]) { room.ceilingPoints.push_back(p); room.ceilingColours.push_back(room.twin.colours[i]); continue; }
        room.points.push_back(p);
        room.normals.push_back(room.twin.normals[i]);
        room.colours.push_back(room.twin.colours[i]);
        room.middle += p;
    }
    room.middle *= 1.0f / std::max<size_t>(room.points.size(), 1);
    for (const cv::Point3f& p : room.points) reach.push_back(std::hypot(p.x - room.middle.x, p.z - room.middle.z));
    if (!reach.empty()) room.radius = spotter::Percentile(reach, 0.9);
    room.middle.y = (room.floor.y + room.ceiling.y) / 2;
    std::printf("The room: %s, %zu points (%zu of them the ceiling), about %.1f m across\n", room.twin.from.c_str(),
                room.points.size() + room.ceilingPoints.size(), room.ceilingPoints.size(), 2 * room.radius);
    return true;
}

// ---------------------------------------------------------------------
//  Step 2 (new): fill, once
// ---------------------------------------------------------------------

//void FillRoom(Room& room){
//
//@param1 - room: the loaded room; its fill is added
//
//@brief - step 2 (new), once: FillHoles on the room's points without the ceiling (else its level step would take the
//         ceiling for a table top and grow legs up to it), up to the ceiling; prints the filled areas, counts and time,
//         and tints each filled cell with the marking colour (stronger on one band) for the marked view. O(FillHoles).
void FillRoom(Room& room)
{
    auto start = std::chrono::steady_clock::now();
    Filled& filled = room.filled = FillHoles(room.points, room.normals, room.colours, room.floor.y, room.floor.seen, room.ceiling.y,
                                             room.twin.firstCamera);
    std::printf("Filled %.1f m^2 in %.1f s: floor %.1f m^2, %d walls %.1f m^2, %d other surfaces %.1f m^2, %d legs, %d dark shadows\n",
                filled.floorArea + filled.wallArea + filled.otherArea, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                filled.floorArea, filled.walls, filled.wallArea, filled.others, filled.otherArea, filled.legs, filled.shadows);
    for (const std::string& line : filled.report) std::printf("%s\n", line.c_str());
    for (size_t i = 0; i < filled.points.size(); i++)
    {
        float tint = filled.stripe[i] ? 0.65f : 0.3f;
        room.marked.push_back(cv::Vec3b(cv::Vec3f(filled.colours[i]) * (1 - tint) + cv::Vec3f(MarkColour) * tint));
    }
}

// ---------------------------------------------------------------------
//  Steps 3-4: what to show and save (fillMode: 0 marked, 1 unmarked, 2 off)
// ---------------------------------------------------------------------

//void Gather(const Room& room, int fillMode, bool withCeiling, std::vector<cv::Point3f>& points, std::vector<cv::Vec3b>& colours){
//
//@param1 - room: the loaded room with its fill
//@param2 - fillMode: 0 = marked, 1 = unmarked, 2 = off
//@param3 - withCeiling: true = the ceiling's points too
//@param4 - points: set to what to show or save
//@param5 - colours: their colours
//
//@brief - every real point, the ceiling if asked, then the fill (marked or plain colours, or none) and in the marked
//         view the purple shadow cut-outs. O(points).
void Gather(const Room& room, int fillMode, bool withCeiling, std::vector<cv::Point3f>& points, std::vector<cv::Vec3b>& colours)
{
    points = room.points;
    colours = room.colours;
    if (withCeiling)
    {
        points.insert(points.end(), room.ceilingPoints.begin(), room.ceilingPoints.end());
        colours.insert(colours.end(), room.ceilingColours.begin(), room.ceilingColours.end());
    }
    if (fillMode < 2)
    {
        points.insert(points.end(), room.filled.points.begin(), room.filled.points.end());
        const std::vector<cv::Vec3b>& fillColours = fillMode == 0 ? room.marked : room.filled.colours;
        colours.insert(colours.end(), fillColours.begin(), fillColours.end());
    }
    if (fillMode == 0)
        for (size_t i = 0; i < room.filled.ghosts.size(); i++)
        {
            points.push_back(room.filled.ghosts[i]);
            colours.push_back(room.filled.ghostStripe[i] ? GhostColour : GhostColour / 2);
        }
}

//cv::Point3f ForView(const cv::Point3f& p){
//
//@param1 - p: a point in the room (Y up, Z towards the first camera)
//
//@brief - the room -> the viewer's coordinates (Y down, Z ahead): turned 180 degrees about X, (x, y, z) -> (x, -y,
//         -z), as 09b's flip, so the room is seen from in front. It is its own inverse. O(1).
cv::Point3f ForView(const cv::Point3f& p) { return { p.x, -p.y, -p.z }; }

//void Show(const Room& room, int fillMode, bool withCeiling, spotter::OrbitView& view, spotter::PointRenderer& renderer){
//
//@param1 - room: the loaded room with its fill
//@param2 - fillMode: 0 = marked, 1 = unmarked, 2 = off
//@param3 - withCeiling: true = the ceiling shown
//@param4 - view: the viewer
//@param5 - renderer: draws the points
//
//@brief - the room as a doll's house (Gather, turned with ForView), a note on what the colours mean and one on the
//         ceiling; shown with the key strip. O(points).
void Show(const Room& room, int fillMode, bool withCeiling, spotter::OrbitView& view, spotter::PointRenderer& renderer)
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colours;
    Gather(room, fillMode, withCeiling, points, colours);
    for (cv::Point3f& p : points) p = ForView(p);
    cv::Mat& image = renderer.Begin(view, cv::Scalar(20, 20, 20));
    renderer.Draw(points, colours, 3);   // 3 cm apart: about 2-3 px from a few metres
    const char* note[3] = { "blue stripes = filled in (never seen); purple = something dark, no reading", "filled in, unmarked",
                            "only what the Kinect saw" };
    spotter::PutText(image, note[fillMode], cv::Point(15, 30), 0.55, fillMode == 0 ? cv::Scalar(MarkColour) : cv::Scalar(200, 200, 200), 1);
    const std::string ceiling = !room.ceiling.seen ? "no ceiling found: the room is shown whole"
                              : withCeiling        ? cv::format("ceiling shown, %.2f m above the floor", room.ceiling.y - room.floor.y)
                                                   : cv::format("ceiling lifted off (%.2f m above the floor): C puts it back", room.ceiling.y - room.floor.y);
    spotter::PutText(image, ceiling, cv::Point(15, 55), 0.55, cv::Scalar(200, 200, 200), 1);
    cv::imshow(Window, spotter::WithKeys(image, Keys));
}

//void Save(const Room& room, int fillMode){
//
//@param1 - room: the loaded room with its fill
//@param2 - fillMode: 0 = marked, 1 = unmarked, 2 = off
//
//@brief - every real point (the ceiling too) plus the fill as shown, written to room_filled_<date>_<time>.ply in the
//         room's folder; prints the point count or the failure. O(points).
void Save(const Room& room, int fillMode)
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colours;
    Gather(room, fillMode, true, points, colours);
    char stamp[32];
    std::time_t now = std::time(nullptr);
    std::strftime(stamp, sizeof stamp, "%Y-%m-%d_%H%M%S", std::localtime(&now));
    const std::filesystem::path path = room.folder / (std::string("room_filled_") + stamp + ".ply");
    if (spotter::SavePly(path.string(), points, colours)) std::printf("Saved %s (%zu points)\n", path.string().c_str(), points.size());
    else std::printf("Could not write %s\n", path.string().c_str());
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

//int main(int argc, char** argv){
//
//@param1 - argc: how many command-line words there are
//@param2 - argv: the words; the file's header says what it accepts
//
//@brief - the program: 13v · Room Fill (no Kinect; after 13e and 13z). The room's flat surfaces with their holes
//         filled and marked, the walls at full height, the ceiling found and lifted off (C). Setup, main loop and
//         clean-up are laid out in the file's sections. Returns 0 on a normal exit, 1 if it couldn't start.
int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Room room;
    if (!LoadRoom(argc, argv, room)) return 1;                                  // step 1
    FillRoom(room);                                                             // step 2 (new)

    cv::namedWindow(Window);
    spotter::OrbitView view(spotter::OrbitView::Options(), ForView(room.middle), 3 * room.radius);
    view.pitchDegrees = ViewPitchDegrees;
    view.Attach(Window);
    spotter::PointRenderer renderer;
    int fillMode = 0;
    bool withCeiling = StartWithCeiling;
    while (true)
    {
        Show(room, fillMode, withCeiling, view, renderer);                      // step 3
        int key = cv::waitKey(15);
        if (spotter::IsQuitKey(key) || cv::getWindowProperty(Window, cv::WND_PROP_VISIBLE) < 1) break;
        if (key == 'g') fillMode = (fillMode + 1) % 3;
        if (key == 'c') withCeiling = !withCeiling;
        if (key == 'k') Save(room, fillMode);                                   // step 4
        if (view.HandleKey(key) && key == 'r') view.pitchDegrees = ViewPitchDegrees;
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Press G three times: marked, unmarked, off. Off is the scan as it was: count its holes.
//  2. Press C with the view turned low (S): the ceiling's holes are where no scan looked up.
//  3. Read the console's list of surfaces: does every real wall show up as "wall", and how much
//     does each bend? A curved wall should say so.
//  4. A doorway nothing was seen through (a closed door, or a dark hallway) gets filled shut: the
//     filling can't tell a door from a hole. Open the door and scan through it.
//  5. Set WallBehindMeters = 0.05 (room_fill.hpp): where the drawn wall stands a little in front of
//     the real one, the fill covers what was seen. That's why it's 0.5.
//  6. Set CeilingBandMeters = 0.5: a ceiling lamp or a high shelf top now goes with the ceiling.
//  7. Set spotter::CeilingOptions::minAreaMeters2 = 5 (room_twin.hpp) in a small room: no ceiling is
//     found, the room is shown whole, and the walls are filled up to 2.4 m.
//  8. Exclude a scan in 13z (X), run 13v again: which holes appear, and does the fill cover them?
