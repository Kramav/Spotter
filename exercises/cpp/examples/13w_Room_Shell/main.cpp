// =====================================================================
//  13w · Room Shell                         (no Kinect; after 13e)
//
//  Just the room itself: every wall from the floor to the ceiling, the
//  floor and the ceiling, each a flat surface with its own picture
//  (a texture), as an .obj you can open in Blender or MeshLab. No
//  furniture: what stood in front of a surface is left out, and the
//  surface behind it is filled in with that surface's own pattern.
//
//    1. Load      a 13e room folder: every included scan, with its full
//                 1920x1080 colour picture.
//    2. Surfaces  (new) the walls are BuildRoom's (09c across every scan),
//                 now full height. The floor is FindTwinFloor's (D33). The
//                 ceiling is the HIGHEST height where level readings cover
//                 half a square metre: a ceiling fan or a lamp hangs lower
//                 and covers less, so it never wins. The floor and ceiling
//                 follow the walls' outline.
//    3. Texture   (new) every 1 cm of every surface asks each scan: "did you
//                 see this spot?" Yes if the scan's depth there agrees with
//                 the surface (not a couch in front of the wall, not the fan
//                 under the ceiling), the colour camera saw the same spot,
//                 and the view wasn't too slanted. Of the scans that saw it,
//                 the one looking most head-on and nearest gives the colour,
//                 from its full colour picture (about 4x sharper than the
//                 depth grid). A colour unlike the others' (a fan blade)
//                 is dropped first.
//    4. Fill      (new) what no scan saw (behind the couch, above the fan,
//                 under the table) gets two things: the slow changes of
//                 colour (light and shadow) inpainted from around, and the
//                 surface's own pattern (wood grain, carpet, plaster)
//                 copied in blocks from its plainest seen square, each
//                 block picked to continue its neighbours (image quilting).
//    5. Look      walls you'd see from outside are left out, as in a doll's
//                 house; G tints what was filled in.
//    6. Save      K writes shell_<date>_<time>\ into the room's folder:
//                 room_shell.obj, room_shell.mtl and a .bmp per surface.
//
//  Simple rooms only: walls upright, floor and ceiling level (D32). Doors
//  and windows get filled shut: "full walls".
//
//  Keys:   G  filled parts tinted / true colour   K  save the .obj
//          mouse drag / W A S D  turn   wheel / Z X  zoom   R reset   Q or Esc  quit
//  Run:    13w_Room_Shell [room folder]   (default: the newest room under RoomsFolder)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <unordered_set>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>   // cv::inpaint

#include "spotter/room.hpp"
#include "spotter/room_folder.hpp"
#include "spotter/room_twin.hpp"
#include "spotter/view.hpp"

namespace fs = std::filesystem;

// The keys, shown in a strip below the window (spotter::WithKeys).
const std::string Keys = "G  filled parts tinted / true colour   K  save the .obj   drag / W A S D  turn   wheel / Z X  zoom   R  reset   Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

const std::string RoomsFolder = "room_scans";   // 13e's default, beside the .exe

// --- The surfaces (new) ---
const float CubeMeters = 0.03f;                 // the floor and ceiling searches use one point per 3 cm cube (as Demo 4's room)
const float JoinMeters = 0.10f;                 // wall ends this close are one corner (BuildRoom snaps them)
const float MinWallMeters = 0.5f;               // shorter wall pieces are left out
const float LevelDegrees = 20;                  // a level reading faces within this of straight up or down
const float MinCeilingAboveFloorMeters = 1.8f;
const float CeilingStepMeters = 0.02f;          // the ceiling's height histogram step
const float MinCeilingAreaMeters2 = 0.5f;       // the ceiling: the HIGHEST level readings covering this much
const float AreaCellMeters = 0.05f;             // area counted in cells this big, seen from below
const float GuessCeilingMeters = 2.4f;          // above the floor, when no ceiling was seen

// --- Texturing (new) ---
const float TexelMeters = 0.01f;                // one texture pixel per 1 cm of surface
const int MaxTexels = 4096;                     // per side; a bigger surface gets coarser texels
const float AgreeMeters = 0.04f;                // a scan saw a spot if its depth there is within 4 cm...
const float AgreeShare = 0.01f;                 // ...plus 1% of the distance of the surface
const float MinFacing = 0.25f;                  // ...looking less slanted than about 75 deg (the cosine)
const int ColourAgreePixels = 8;                // ...and its colour camera saw the same spot (not something nearer it)
const int OutlierColour = 60;                   // with 3+ scans: a colour this far from their median is dropped

// --- Filling (new) ---
const float BaseCellMeters = 0.25f;             // the slow changes of colour, on cells this big (keep it over a plank's width)
const int ExemplarTexels = 96;                  // the pattern comes from the plainest seen square this big...
const int MinExemplarTexels = 16;               // ...or smaller, down to this (none: holes get the plain colour)
const int BlockTexels = 32;                     // copied in blocks this big...
const int OverlapTexels = 8;                    // ...overlapping the blocks before by this much,
const int BlockCandidates = 80;                 // ...the best of this many tries at fitting the overlap
const cv::Vec3b UnseenColour(200, 200, 200);    // a surface no scan saw at all

// --- Output ---
const float QuadMeters = 0.10f;                 // the floor and ceiling as squares this big, so any outline works
const int ViewStep = 2;                         // the viewer draws every 2nd texel each way
const float ViewPitchDegrees = 35;
const cv::Vec3b MarkColour(255, 128, 0);        // B, G, R: filled-in parts tinted blue (G)
const std::string Window = "13w - room shell";

// ---------------------------------------------------------------------
//  Step 1: load the scans
// ---------------------------------------------------------------------

struct Shot
{
    spotter::Scan scan;                 // cloud, colour on the depth grid, pose
    cv::Mat full;                       // colour_full.bmp
    std::vector<int> colourIndex;       // per depth pixel, the full picture's pixel it landed on (-1 = none)
    spotter::ColourModel model;         // how a depth-camera point lands in the full picture
    cv::Matx33f toCamera;               // the levelled room -> this camera: p = toCamera * w + offset
    cv::Point3f offset, at;             // at: where it stood in the levelled room
};

// One flat surface and its picture. Texel (row, column)'s centre is
// origin + across * (column + 0.5) * texel + down * (row + 0.5) * texel. "across" and "down" are chosen
// so the picture reads the right way round from inside the room.
struct Surface
{
    std::string name;                   // floor, ceiling, wall_01 ...
    bool wall = false;
    cv::Point3f origin, across, down, inward;
    float texel = TexelMeters;
    cv::Mat colour, seen, inside, filled;   // CV_8UC3; 255 = seen / part of the surface / filled in
    int exemplar = 0;                   // the side of the square its pattern came from (texels)
    std::vector<cv::Point3f> viewPoints;   // step 5, ready for the viewer
    std::vector<cv::Vec3b> viewColours, viewMarked;
};

struct Shell
{
    fs::path folder;
    spotter::RoomInfo info;
    std::vector<Shot> shots;
    cv::Matx33f firstCamera = cv::Matx33f::eye();   // scan 1's camera -> the levelled room
    float floorY = 0, ceilingY = 0;
    std::vector<cv::Point2f> outline;   // the floor's outline, as (x, z)
    cv::Point3f middle;
    float radius = 3;
    std::vector<Surface> surfaces;
};

//bool LoadShell(int argc, char** argv, Shell& shell){
//
//@param1 - argc: how many command-line words there are
//@param2 - argv: the words: an optional room folder
//@param3 - shell: filled with the folder, info and shots
//
//@brief - step 1: the room folder given, or the newest under RoomsFolder. Refuses an old (format 1, mirrored)
//         room. Loads every included scan (LoadScan) with its full colour picture (colour_full.bmp), its colour
//         index and the colour model fitted from them; scans without the full picture fall back to the depth
//         grid's colour. False if nothing loaded. O(scans x pixels).
bool LoadShell(int argc, char** argv, Shell& shell)
{
    shell.folder = argc > 1 ? fs::path(argv[1]) : spotter::NewestRoomFolder(RoomsFolder);
    std::string problem;
    if (shell.folder.empty())
    {
        std::printf("No room under %s. Scan one with 13e_Room_Scan, or name its folder: 13w_Room_Shell <folder>\n", RoomsFolder.c_str());
        return false;
    }
    if (!spotter::ReadRoomInfo(shell.folder, shell.info, problem)) { std::printf("%s\n", problem.c_str()); return false; }
    if (shell.info.format != 2) { std::printf("An old (format 1, mirrored) room: scan it again with 13e_Room_Scan.\n"); return false; }
    int sharp = 0;
    for (const fs::path& dir : spotter::ScanFolders(shell.folder))
    {
        Shot shot;
        int number = 0;
        if (!spotter::LoadScan(dir, shell.info, shot.scan, number, problem)) { std::printf("  %s skipped: %s\n", dir.filename().string().c_str(), problem.c_str()); continue; }
        if (shot.scan.excluded) continue;
        shot.full = cv::imread((dir / "colour_full.bmp").string(), cv::IMREAD_COLOR);
        shot.colourIndex.resize(shot.scan.cloud.size());
        std::ifstream index(dir / "colour_index.bin", std::ios::binary);
        if (!index.read((char*)shot.colourIndex.data(), shot.colourIndex.size() * sizeof(int))) shot.colourIndex.clear();
        if (!shot.full.empty() && !shot.colourIndex.empty()) shot.model = spotter::FitColourModel(shot.scan.cloud, shot.colourIndex, shot.full.cols);
        sharp += shot.model.ok;
        shell.shots.push_back(std::move(shot));
    }
    std::printf("%s: %zu scans, %d with their full colour picture (the rest use the depth grid's colour)\n", shell.folder.string().c_str(),
                shell.shots.size(), sharp);
    return !shell.shots.empty();
}

// ---------------------------------------------------------------------
//  Step 2 (new): the walls, floor and ceiling
// ---------------------------------------------------------------------

//bool FindCeiling(const std::vector<cv::Point3f>& points, const std::vector<cv::Point3f>& normals, float floorY, float& ceilingY){
//
//@param1 - points: the room's points, levelled (Y up)
//@param2 - normals: which way each faces
//@param3 - floorY: the floor's height
//@param4 - ceilingY: set to the ceiling's height
//
//@brief - the ceiling: the highest height (2 cm steps, at least MinCeilingAboveFloorMeters above the floor) where
//         level, down-facing readings cover MinCeilingAreaMeters2 of the plan (AreaCellMeters cells). A fan, a
//         lamp or a beam hangs lower and covers less, so it never wins; the ceiling it hides is step 4's job.
//         False if none qualifies. O(points).
bool FindCeiling(const std::vector<cv::Point3f>& points, const std::vector<cv::Point3f>& normals, float floorY, float& ceilingY)
{
    const float levelCos = std::cos(LevelDegrees * (float)CV_PI / 180);
    std::map<int, std::vector<cv::Point3f>> steps;   // level readings by height
    for (size_t i = 0; i < points.size(); i++)
        if (std::abs(normals[i].y) >= levelCos && points[i].y > floorY + MinCeilingAboveFloorMeters)
            steps[(int)std::floor(points[i].y / CeilingStepMeters)].push_back(points[i]);
    for (auto step = steps.rbegin(); step != steps.rend(); ++step)   // from the top down
    {
        std::set<std::pair<int, int>> cells;
        double sum = 0;
        int count = 0;
        for (int s = step->first - 1; s <= step->first + 1; s++)   // the step and its neighbours
        {
            auto found = steps.find(s);
            if (found == steps.end()) continue;
            for (const cv::Point3f& p : found->second)
            {
                cells.insert({ (int)std::floor(p.x / AreaCellMeters), (int)std::floor(p.z / AreaCellMeters) });
                sum += p.y;
                count++;
            }
        }
        if (cells.size() * AreaCellMeters * AreaCellMeters >= MinCeilingAreaMeters2) { ceilingY = (float)(sum / count); return true; }
    }
    return false;
}

//std::vector<cv::Point2f> Outline(const std::vector<std::pair<cv::Point2f, cv::Point2f>>& walls, bool& closed){
//
//@param1 - walls: the walls' ends seen from above, (x, z)
//@param2 - closed: set to true if the walls made a closed loop
//
//@brief - the walls' ends chained into a closed loop if they make one (from the first wall, each next unused wall
//         whose end lies within JoinMeters of the loop's end); otherwise their convex hull. O(walls^2).
std::vector<cv::Point2f> Outline(const std::vector<std::pair<cv::Point2f, cv::Point2f>>& walls, bool& closed)
{
    std::vector<cv::Point2f> loop{ walls[0].first, walls[0].second };
    std::vector<char> used(walls.size(), 0);
    used[0] = 1;
    for (size_t step = 1; step < walls.size(); step++)
    {
        int next = -1;
        bool flip = false;
        for (size_t w = 0; w < walls.size() && next < 0; w++)
        {
            if (used[w]) continue;
            if (cv::norm(walls[w].first - loop.back()) < JoinMeters) next = (int)w;
            else if (cv::norm(walls[w].second - loop.back()) < JoinMeters) { next = (int)w; flip = true; }
        }
        if (next < 0) break;
        used[next] = 1;
        loop.push_back(flip ? walls[next].first : walls[next].second);
    }
    closed = loop.size() >= 4 && cv::norm(loop.back() - loop.front()) < JoinMeters;
    if (closed) { loop.pop_back(); return loop; }
    std::vector<cv::Point2f> ends, hull;
    for (const auto& w : walls) ends.insert(ends.end(), { w.first, w.second });
    cv::convexHull(ends, hull);
    return hull;
}

//Surface MakeSurface(const std::string& name, const cv::Point3f& origin, const cv::Point3f& across, const cv::Point3f& down, float widthMeters, float heightMeters){
//
//@param1 - name: the surface's name, for its file
//@param2 - origin: its top-left corner in the room
//@param3 - across: unit direction along its columns
//@param4 - down: unit direction along its rows
//@param5 - widthMeters: how wide it is
//@param6 - heightMeters: how tall it is
//
//@brief - an empty surface: inward = down x across (the side it's seen from), a texel of TexelMeters (coarser if
//         it would pass MaxTexels), and every texel inside. O(texels).
Surface MakeSurface(const std::string& name, const cv::Point3f& origin, const cv::Point3f& across, const cv::Point3f& down,
                    float widthMeters, float heightMeters)
{
    Surface s;
    s.name = name;
    s.origin = origin;
    s.across = across;
    s.down = down;
    s.inward = down.cross(across);   // the side it's seen from
    s.texel = std::max(TexelMeters, std::max(widthMeters, heightMeters) / MaxTexels);
    const int columns = std::max(1, (int)std::ceil(widthMeters / s.texel)), rows = std::max(1, (int)std::ceil(heightMeters / s.texel));
    s.inside = cv::Mat(rows, columns, CV_8U, cv::Scalar(255));
    return s;
}

//void Clip(Surface& s, const std::vector<cv::Point2f>& outline, float y){
//
//@param1 - s: a floor or ceiling; its `inside` mask is clipped
//@param2 - outline: the walls' outline seen from above, (x, z)
//@param3 - y: the surface's height, to put each outline point on the surface
//
//@brief - a floor or ceiling: only the texels inside the outline polygon stay inside (each outline point is put on
//         the surface at height y, turned into texel units, and the polygon is filled into the mask). O(texels +
//         outline).
void Clip(Surface& s, const std::vector<cv::Point2f>& outline, float y)
{
    std::vector<cv::Point> corners;
    for (const cv::Point2f& o : outline)
    {
        const cv::Point3f d = cv::Point3f(o.x, y, o.y) - s.origin;
        corners.push_back(cv::Point((int)std::lround(d.dot(s.across) / s.texel), (int)std::lround(d.dot(s.down) / s.texel)));
    }
    s.inside.setTo(0);
    cv::fillPoly(s.inside, std::vector<std::vector<cv::Point>>{ corners }, cv::Scalar(255));
}

//void Level(Shell& shell, const cv::Matx33f& firstCamera){
//
//@param1 - shell: the shell; its shots get their levelled pose
//@param2 - firstCamera: scan 1's camera -> the levelled room (BuildRoom's)
//
//@brief - BuildRoom levels the room; from then on every scan is placed in the levelled room: each shot's camera <-
//         room turn and offset are worked out (the inverse of w = firstCamera * (turn * p + shift)). O(scans).
void Level(Shell& shell, const cv::Matx33f& firstCamera)
{
    shell.firstCamera = firstCamera;
    for (Shot& shot : shell.shots)
    {
        shot.toCamera = (firstCamera * shot.scan.pose.turn).t();
        shot.at = firstCamera * shot.scan.pose.shift;
        shot.offset = -(shot.toCamera * shot.at);
    }
}

//bool FindSurfaces(Shell& shell){
//
//@param1 - shell: the shell with its shots; its surfaces, floor, ceiling and outline are filled
//
//@brief - step 2: BuildRoom's walls at full height, FindTwinFloor's floor and FindCeiling's ceiling, from every
//         scan's ICP points levelled and thinned to one per CubeMeters cube (evenly spread, so a wall seen up
//         close doesn't outweigh one seen once; D43: raw points put the walls' foot, and so the floor, too high).
//         Floor and ceiling follow the walls' outline; each wall faces into the room. False if no walls.
//         O(BuildRoom + points).
bool FindSurfaces(Shell& shell)
{
    std::vector<const spotter::Scan*> scans;
    for (const Shot& shot : shell.shots) scans.push_back(&shot.scan);
    std::mt19937 random(1);
    const spotter::Room built = spotter::BuildRoom(scans, spotter::Mesh(), random, spotter::RoomOptions());
    Level(shell, built.firstCamera);

    // Every scan's ICP points, levelled, one per CubeMeters cube: for the floor and the ceiling. Evenly
    // spread, so a wall seen up close or by many scans doesn't outweigh one seen once (D43: the raw
    // points put the walls' foot, and so the floor, too high).
    spotter::TwinRoom levelled;
    levelled.firstCamera = built.firstCamera;
    std::unordered_set<long long> taken;
    for (const Shot& shot : shell.shots)
        for (size_t i = 0; i < shot.scan.surface.points.size(); i++)
        {
            const cv::Point3f p = shot.toCamera.t() * shot.scan.surface.points[i] + shot.at;
            if (!taken.insert(spotter::CubeKey(p, CubeMeters)).second) continue;
            levelled.points.push_back(p);
            levelled.normals.push_back(shot.toCamera.t() * shot.scan.surface.normals[i]);
        }
    const spotter::TwinFloor floor = spotter::FindTwinFloor(levelled);
    for (const std::string& line : floor.report) std::printf("%s\n", line.c_str());
    shell.floorY = floor.y;
    const bool ceilingSeen = FindCeiling(levelled.points, levelled.normals, floor.y, shell.ceilingY);
    if (!ceilingSeen) shell.ceilingY = floor.y + GuessCeilingMeters;
    const float height = shell.ceilingY - shell.floorY;
    std::printf("Ceiling: %s, %.2f m above the floor\n", ceilingSeen ? "seen" : "not seen, guessed", height);

    std::vector<std::pair<cv::Point2f, cv::Point2f>> walls;
    for (const auto& w : built.walls)
        if (cv::norm(w.second - w.first) >= MinWallMeters) walls.push_back(w);
    if (walls.empty()) { std::printf("No walls found: scan more of the room's walls with 13e.\n"); return false; }
    bool closed = false;
    shell.outline = Outline(walls, closed);
    cv::Point2f low(1e9f, 1e9f), high(-1e9f, -1e9f), centre(0, 0);
    for (const cv::Point2f& o : shell.outline)
    {
        low = cv::Point2f(std::min(low.x, o.x), std::min(low.y, o.y));
        high = cv::Point2f(std::max(high.x, o.x), std::max(high.y, o.y));
        centre += o * (1.0f / shell.outline.size());
    }
    shell.middle = cv::Point3f(centre.x, shell.floorY + height / 2, centre.y);
    shell.radius = 0;
    for (const cv::Point2f& o : shell.outline) shell.radius = std::max(shell.radius, (float)cv::norm(o - centre));
    std::printf("%zu walls, %s\n", walls.size(), closed ? "joined all round" : "not joined all round: the floor follows their outer outline");

    // Seen from above: X right, Z down the picture. Seen from below: the ceiling's picture runs the other way.
    shell.surfaces.push_back(MakeSurface("floor", { low.x, shell.floorY, low.y }, { 1, 0, 0 }, { 0, 0, 1 }, high.x - low.x, high.y - low.y));
    Clip(shell.surfaces.back(), shell.outline, shell.floorY);
    shell.surfaces.push_back(MakeSurface("ceiling", { low.x, shell.ceilingY, high.y }, { 1, 0, 0 }, { 0, 0, -1 }, high.x - low.x, high.y - low.y));
    Clip(shell.surfaces.back(), shell.outline, shell.ceilingY);
    for (auto [a, b] : walls)   // full height, from the floor to the ceiling
    {
        cv::Point3f across(b.x - a.x, 0, b.y - a.y);
        const cv::Point2f mid = (a + b) * 0.5f;
        if (cv::Point3f(0, -1, 0).cross(across).dot(cv::Point3f(centre.x - mid.x, 0, centre.y - mid.y)) < 0) { std::swap(a, b); across = -across; }
        const float length = (float)cv::norm(across);
        shell.surfaces.push_back(MakeSurface(cv::format("wall_%02d", (int)shell.surfaces.size() - 1), { a.x, shell.ceilingY, a.y },
                                             across * (1 / length), { 0, -1, 0 }, length, height));
        shell.surfaces.back().wall = true;
    }
    return true;
}

// ---------------------------------------------------------------------
//  Step 3 (new): each texel's colour, from the scans that saw it
// ---------------------------------------------------------------------

//cv::Point3f Centre(const Surface& s, int row, int column){
//
//@param1 - s: a surface
//@param2 - row: a texel's row
//@param3 - column: its column
//
//@brief - the room point at the middle of that texel. O(1).
cv::Point3f Centre(const Surface& s, int row, int column)
{
    return s.origin + s.across * ((column + 0.5f) * s.texel) + s.down * ((row + 0.5f) * s.texel);
}

//cv::Vec3b Bilinear(const cv::Mat& picture, double column, double row){
//
//@param1 - picture: a BGR picture
//@param2 - column: a position between pixels
//@param3 - row: likewise
//
//@brief - the colour at a fractional position, blended from the four nearest pixels. O(1).
cv::Vec3b Bilinear(const cv::Mat& picture, double column, double row)
{
    const int c = std::min((int)column, picture.cols - 2), r = std::min((int)row, picture.rows - 2);
    const double fc = column - c, fr = row - r;
    const cv::Vec3d top = cv::Vec3d(picture.at<cv::Vec3b>(r, c)) * (1 - fc) + cv::Vec3d(picture.at<cv::Vec3b>(r, c + 1)) * fc;
    const cv::Vec3d bottom = cv::Vec3d(picture.at<cv::Vec3b>(r + 1, c)) * (1 - fc) + cv::Vec3d(picture.at<cv::Vec3b>(r + 1, c + 1)) * fc;
    return cv::Vec3b(top * (1 - fr) + bottom * fr);
}

struct Sighting
{
    cv::Vec3b colour;
    float weight;   // head-on and near
};

//bool Sight(const Shot& shot, const Surface& s, const cv::Point3f& w, const spotter::PinholeCamera& lens, Sighting& out){
//
//@param1 - shot: one scan
//@param2 - s: the surface
//@param3 - w: a room point on it
//@param4 - lens: the depth camera
//@param5 - out: set to the colour and weight of what it saw
//
//@brief - what one scan saw at room point w on surface s, if it saw it: the point must land in the picture, the
//         scan's depth there must agree with the surface (within AgreeMeters: not a couch in front of the wall,
//         not the fan under the ceiling), the view must face it (at least MinFacing), and the colour camera must
//         have seen the same spot (its colour index agrees). The colour comes from the full picture (bilinear) or
//         the depth grid's. Weight = facing squared / distance squared (head-on and near). O(1).
bool Sight(const Shot& shot, const Surface& s, const cv::Point3f& w, const spotter::PinholeCamera& lens, Sighting& out)
{
    const cv::Point3f p = shot.toCamera * w + shot.offset;
    if (p.z < 0.3f) return false;
    const cv::Point3f toCamera = shot.at - w;
    const float distance = (float)cv::norm(toCamera), facing = toCamera.dot(s.inward) / distance;
    if (facing < MinFacing) return false;
    const std::optional<cv::Point2f> pixel = lens.Project(p);
    if (!pixel) return false;
    const int column = (int)std::lround(pixel->x), row = (int)std::lround(pixel->y);
    if (column < 0 || row < 0 || column >= lens.width || row >= lens.height) return false;
    const int i = row * lens.width + column;
    const cv::Point3f& reading = shot.scan.cloud[i];
    if (std::isnan(reading.z) || std::abs(reading.z - p.z) > AgreeMeters + AgreeShare * p.z) return false;   // something nearer, or a hole behind
    out.weight = facing * facing / (distance * distance);
    if (!shot.model.ok) { out.colour = shot.scan.picture.at<cv::Vec3b>(row, column); return true; }
    const spotter::ColourModel& m = shot.model;
    const double fc = m.a * p.x / p.z + m.e / p.z + m.b, fr = m.c * p.y / p.z + m.f / p.z + m.d;
    if (fc < 0 || fr < 0 || fc > shot.full.cols - 1 || fr > shot.full.rows - 1) return false;
    const int landed = shot.colourIndex[i];   // the colour camera saw this depth pixel there (-1: it saw something nearer)
    if (landed < 0 || std::abs(landed % shot.full.cols - fc) > ColourAgreePixels || std::abs(landed / shot.full.cols - fr) > ColourAgreePixels)
        return false;
    out.colour = Bilinear(shot.full, fc, fr);
    return true;
}

//void Texture(Surface& s, const std::vector<Shot>& shots, const spotter::PinholeCamera& lens){
//
//@param1 - s: the surface; its colour and seen masks are filled
//@param2 - shots: every scan
//@param3 - lens: the depth camera
//
//@brief - step 3: every texel asks every scan (Sight, in parallel). With three or more sightings, colours unlike
//         the per-channel median by more than OutlierColour (a fan blade, a reflection) are dropped first; of the
//         rest the highest weight gives the colour, and a texel with any sighting left is seen. O(texels x scans).
void Texture(Surface& s, const std::vector<Shot>& shots, const spotter::PinholeCamera& lens)
{
    s.colour = cv::Mat(s.inside.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    s.seen = cv::Mat::zeros(s.inside.size(), CV_8U);
    cv::parallel_for_(cv::Range(0, s.inside.rows), [&](const cv::Range& rows) {
        std::vector<Sighting> found;
        std::vector<uchar> channel;
        for (int row = rows.start; row < rows.end; row++)
            for (int column = 0; column < s.inside.cols; column++)
            {
                if (!s.inside.at<uchar>(row, column)) continue;
                const cv::Point3f w = Centre(s, row, column);
                found.clear();
                Sighting sighting;
                for (const Shot& shot : shots)
                    if (Sight(shot, s, w, lens, sighting)) found.push_back(sighting);
                if (found.empty()) continue;
                if (found.size() >= 3)   // drop colours unlike the rest (a fan blade, a reflection)
                {
                    cv::Vec3b median;
                    for (int k = 0; k < 3; k++)
                    {
                        channel.clear();
                        for (const Sighting& f : found) channel.push_back(f.colour[k]);
                        std::nth_element(channel.begin(), channel.begin() + channel.size() / 2, channel.end());
                        median[k] = channel[channel.size() / 2];
                    }
                    auto unlike = [&](const Sighting& f) { return cv::norm(cv::Vec3i(f.colour) - cv::Vec3i(median), cv::NORM_INF) > OutlierColour; };
                    found.erase(std::remove_if(found.begin(), found.end(), unlike), found.end());
                    if (found.empty()) continue;
                }
                const Sighting& best = *std::max_element(found.begin(), found.end(), [](const Sighting& a, const Sighting& b) { return a.weight < b.weight; });
                s.colour.at<cv::Vec3b>(row, column) = best.colour;
                s.seen.at<uchar>(row, column) = 255;
            }
    });
}

// ---------------------------------------------------------------------
//  Step 4 (new): fill what nobody saw, with the surface's own pattern
// ---------------------------------------------------------------------

//cv::Rect Exemplar(const Surface& s){
//
//@param1 - s: a textured surface
//
//@brief - the plainest fully seen square (ExemplarTexels, down to MinExemplarTexels): the one whose colour is
//         nearest the surface's usual colour and that's least busy (not a poster). Empty if there is none.
//         O(texels).
cv::Rect Exemplar(const Surface& s)
{
    cv::Mat seenInt, sum, squares;
    cv::integral(s.seen & s.inside, seenInt, CV_32S);
    cv::integral(s.colour, sum, squares, CV_64F, CV_64F);
    std::vector<uchar> channel[3];
    for (int i = 0; i < (int)s.seen.total(); i += 7)
        if (s.seen.data[i]) for (int k = 0; k < 3; k++) channel[k].push_back(s.colour.data[3 * i + k]);
    cv::Vec3d usual;
    for (int k = 0; k < 3; k++)
    {
        std::nth_element(channel[k].begin(), channel[k].begin() + channel[k].size() / 2, channel[k].end());
        usual[k] = channel[k][channel[k].size() / 2];
    }
    cv::Rect best;
    double bestScore = 1e30;
    for (int size = std::min({ ExemplarTexels, s.seen.rows, s.seen.cols }); size >= MinExemplarTexels && best.empty(); size = size * 3 / 4)
        for (int y = 0; y + size <= s.seen.rows; y += std::max(1, size / 4))
            for (int x = 0; x + size <= s.seen.cols; x += std::max(1, size / 4))
            {
                auto box = [&](const cv::Mat& integral, int k) {
                    return integral.ptr<double>(y + size)[3 * (x + size) + k] - integral.ptr<double>(y)[3 * (x + size) + k]
                         - integral.ptr<double>(y + size)[3 * x + k] + integral.ptr<double>(y)[3 * x + k];
                };
                const int seenCount = seenInt.at<int>(y + size, x + size) - seenInt.at<int>(y, x + size) - seenInt.at<int>(y + size, x) + seenInt.at<int>(y, x);
                if (seenCount < size * size * 255) continue;
                double score = 0;
                for (int k = 0; k < 3; k++)
                {
                    const double n = size * size, mean = box(sum, k) / n, variance = box(squares, k) / n - mean * mean;
                    score += std::abs(mean - usual[k]) + std::sqrt(std::max(0.0, variance));
                }
                if (score < bestScore) { bestScore = score; best = cv::Rect(x, y, size, size); }
            }
    return best;
}

//void Fill(Surface& s, std::mt19937& random){
//
//@param1 - s: a textured surface; its unseen texels are filled
//@param2 - random: the random number generator
//
//@brief - step 4: a. the slow changes (the seen colour averaged over BaseCellMeters cells, empty cells inpainted
//         with Telea, smoothed back up); b. the pattern (what's left of the seen colour); c. blocks of the pattern
//         from the exemplar square, in rows, each the best of BlockCandidates at matching what's already there
//         where it overlaps (Efros and Freeman 2001's image quilting, without its seam cut), writing only unseen
//         texels; d. slow changes plus pattern in the unseen texels only. O(texels x BlockCandidates x block
//         area).
void Fill(Surface& s, std::mt19937& random)
{
    const int rows = s.inside.rows, columns = s.inside.cols;
    s.filled = s.inside & ~s.seen;
    if (cv::countNonZero(s.seen) == 0) { s.colour.setTo(UnseenColour, s.inside); return; }
    if (cv::countNonZero(s.filled) == 0) return;

    // a. The slow changes: the seen colour averaged over BaseCellMeters cells, the empty cells
    //    inpainted, then smoothed back up to the texels.
    cv::Mat colour, weight, weights;
    s.colour.convertTo(colour, CV_32FC3);
    s.seen.convertTo(weight, CV_32F, 1.0 / 255);
    cv::merge(std::vector<cv::Mat>{ weight, weight, weight }, weights);
    const int k = std::max(1, (int)std::lround(BaseCellMeters / s.texel));
    const cv::Size coarse((columns + k - 1) / k, (rows + k - 1) / k);
    cv::Mat sum, count, base;
    cv::resize(colour.mul(weights), sum, coarse, 0, 0, cv::INTER_AREA);
    cv::resize(weight, count, coarse, 0, 0, cv::INTER_AREA);
    cv::Mat mean(coarse, CV_8UC3, cv::Scalar(0, 0, 0)), empty(coarse, CV_8U, cv::Scalar(0));
    for (int r = 0; r < coarse.height; r++)
        for (int c = 0; c < coarse.width; c++)
        {
            const float n = count.at<float>(r, c);
            if (n > 0) mean.at<cv::Vec3b>(r, c) = cv::Vec3b(sum.at<cv::Vec3f>(r, c) / n);
            if (n < 0.25f) empty.at<uchar>(r, c) = 255;
        }
    cv::inpaint(mean, empty, base, 3, cv::INPAINT_TELEA);
    cv::resize(base, base, s.inside.size(), 0, 0, cv::INTER_LINEAR);
    base.convertTo(base, CV_32FC3);

    // b. The pattern: what's left of the seen colour after the slow changes.
    cv::Mat detail = colour - base;
    detail.setTo(cv::Scalar::all(0), s.seen == 0);
    const cv::Rect square = Exemplar(s);
    s.exemplar = square.width;

    // c. Blocks of the pattern from that square, in rows; each the best of BlockCandidates at matching
    //    what's already there where it overlaps (the blocks before it, or seen texels). Only unseen
    //    texels are written. (Efros & Freeman 2001's image quilting, without its seam cut.)
    cv::Mat have = s.seen.clone();
    if (!square.empty())
    {
        const cv::Mat example = detail(square);
        const int b = std::min(BlockTexels, square.width), step = b - std::min(OverlapTexels, b / 3);
        std::uniform_int_distribution<int> pick(0, square.width - b);
        for (int y = 0; y < rows; y += step)
            for (int x = 0; x < columns; x += step)
            {
                const cv::Rect block = cv::Rect(x, y, b, b) & cv::Rect(0, 0, columns, rows);
                if (cv::countNonZero(cv::Mat(s.filled(block) & ~have(block))) == 0) continue;
                cv::Point best(pick(random), pick(random));
                double bestCost = 1e30;
                for (int t = 0; t < BlockCandidates; t++)
                {
                    const cv::Point from(pick(random), pick(random));
                    double cost = 0;
                    for (int dy = 0; dy < block.height && cost < bestCost; dy++)
                        for (int dx = 0; dx < block.width; dx++)
                            if (have.at<uchar>(y + dy, x + dx))
                            {
                                const cv::Vec3f d = detail.at<cv::Vec3f>(y + dy, x + dx) - example.at<cv::Vec3f>(from.y + dy, from.x + dx);
                                cost += d.dot(d);
                            }
                    if (cost < bestCost) { bestCost = cost; best = from; }
                }
                for (int dy = 0; dy < block.height; dy++)
                    for (int dx = 0; dx < block.width; dx++)
                        if (s.filled.at<uchar>(y + dy, x + dx) && !have.at<uchar>(y + dy, x + dx))
                        {
                            detail.at<cv::Vec3f>(y + dy, x + dx) = example.at<cv::Vec3f>(best.y + dy, best.x + dx);
                            have.at<uchar>(y + dy, x + dx) = 255;
                        }
            }
    }

    // d. Slow changes plus pattern, in the unseen texels only.
    cv::Mat together;
    cv::Mat(base + detail).convertTo(together, CV_8UC3);
    together.copyTo(s.colour, s.filled);
}

// ---------------------------------------------------------------------
//  Steps 5-6: look, save
// ---------------------------------------------------------------------

//cv::Point3f ForView(const cv::Point3f& p){
//
//@param1 - p: a point in the room (Y up, Z towards the first camera)
//
//@brief - the room -> the viewer's coordinates (Y down, Z ahead): turned 180 degrees about X, (x, y, z) -> (x, -y,
//         -z), as 09b's flip, so the room is seen from in front. It is its own inverse. O(1).
cv::Point3f ForView(const cv::Point3f& p) { return { p.x, -p.y, -p.z }; }

//void ReadyToView(Surface& s){
//
//@param1 - s: a filled surface
//
//@brief - the points the viewer draws: every ViewStep-th texel inside the surface, with its point (turned with
//         ForView) and colour, and a second colour for the marked view (a filled texel blended 40% with
//         MarkColour). O(texels / ViewStep^2).
void ReadyToView(Surface& s)
{
    for (int row = 0; row < s.inside.rows; row += ViewStep)
        for (int column = 0; column < s.inside.cols; column += ViewStep)
        {
            if (!s.inside.at<uchar>(row, column)) continue;
            const cv::Vec3b colour = s.colour.at<cv::Vec3b>(row, column);
            s.viewPoints.push_back(ForView(Centre(s, row, column)));
            s.viewColours.push_back(colour);
            s.viewMarked.push_back(s.filled.at<uchar>(row, column) ? cv::Vec3b(cv::Vec3f(colour) * 0.6f + cv::Vec3f(MarkColour) * 0.4f) : colour);
        }
}

//cv::Point3f Eye(const spotter::OrbitView& view){
//
//@param1 - view: the viewer
//
//@brief - where the viewer's eye is, in the room (OrbitView::Project, backwards): pivot + distance x (cos pitch
//         sin yaw, -sin pitch, -cos pitch cos yaw), turned back with ForView. O(1).
cv::Point3f Eye(const spotter::OrbitView& view)
{
    const float r = (float)CV_PI / 180, d = view.distance;
    const float cy = std::cos(view.yawDegrees * r), sy = std::sin(view.yawDegrees * r), cp = std::cos(view.pitchDegrees * r), sp = std::sin(view.pitchDegrees * r);
    return ForView(view.pivot + cv::Point3f(d * cp * sy, -d * sp, -d * cp * cy));
}

//void Show(const Shell& shell, bool marked, spotter::OrbitView& view, spotter::PointRenderer& renderer){
//
//@param1 - shell: the surfaces
//@param2 - marked: true = show the filled texels marked
//@param3 - view: the viewer
//@param4 - renderer: draws the points
//
//@brief - the surfaces as points, leaving out any surface seen from outside (its back, towards the eye); shown
//         with the key strip. O(shown points).
void Show(const Shell& shell, bool marked, spotter::OrbitView& view, spotter::PointRenderer& renderer)
{
    const cv::Point3f eye = Eye(view);
    cv::Mat& image = renderer.Begin(view, cv::Scalar(20, 20, 20));
    for (const Surface& s : shell.surfaces)
        if ((eye - s.origin).dot(s.inward) > 0)   // a surface seen from outside is left out: a doll's house
            renderer.Draw(s.viewPoints, marked ? s.viewMarked : s.viewColours, 2);
    cv::putText(image, marked ? "blue tint = filled in (no scan saw it)" : "true colour", cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                marked ? cv::Scalar(MarkColour) : cv::Scalar(200, 200, 200), 1);
    cv::imshow(Window, spotter::WithKeys(image, Keys));
}

//fs::path Save(const Shell& shell){
//
//@param1 - shell: the finished surfaces
//
//@brief - shell_<date>_<time>\ in the room's folder: the .obj (levelled, Y up, metres: the floor and ceiling as
//         QuadMeters quads, a wall as one quad, counter-clockwise from inside), its .mtl, and a .bmp per surface.
//         Returns the folder. O(texels + quads).
fs::path Save(const Shell& shell)
{
    char stamp[32];
    std::time_t now = std::time(nullptr);
    std::strftime(stamp, sizeof stamp, "%Y-%m-%d_%H%M%S", std::localtime(&now));
    const fs::path dir = shell.folder / (std::string("shell_") + stamp);
    fs::create_directories(dir);
    std::ofstream obj(dir / "room_shell.obj"), mtl(dir / "room_shell.mtl");
    obj << "# 13w room shell: walls, floor and ceiling. Levelled: Y up, metres; spotter format 2 (not mirrored)\nmtllib room_shell.mtl\n";
    bool ok = true;
    int vertices = 0;
    for (const Surface& s : shell.surfaces)
    {
        ok = cv::imwrite((dir / (s.name + ".bmp")).string(), s.colour) && ok;
        mtl << "newmtl " << s.name << "\nKa 1 1 1\nKd 1 1 1\nmap_Kd " << s.name << ".bmp\n\n";
        obj << "o " << s.name << "\nusemtl " << s.name << "\n";
        const int rows = s.inside.rows, columns = s.inside.cols;
        const int q = s.wall ? std::max(rows, columns) : std::max(1, (int)std::lround(QuadMeters / s.texel));   // a wall is one quad
        for (int r = 0; r < rows; r += q)
            for (int c = 0; c < columns; c += q)
            {
                const int r1 = std::min(r + q, rows), c1 = std::min(c + q, columns);
                if (!s.inside.at<uchar>((r + r1) / 2, (c + c1) / 2)) continue;
                const int corners[4][2] = { { c, r }, { c, r1 }, { c1, r1 }, { c1, r } };   // counter-clockwise from inside
                for (const auto& k : corners)
                {
                    const cv::Point3f p = s.origin + s.across * (k[0] * s.texel) + s.down * (k[1] * s.texel);
                    obj << cv::format("v %.4f %.4f %.4f\nvt %.5f %.5f\n", p.x, p.y, p.z, (double)k[0] / columns, 1.0 - (double)k[1] / rows);
                }
                obj << cv::format("f %d/%d %d/%d %d/%d %d/%d\n", vertices + 1, vertices + 1, vertices + 2, vertices + 2, vertices + 3, vertices + 3,
                                  vertices + 4, vertices + 4);
                vertices += 4;
            }
    }
    ok = ok && (bool)obj && (bool)mtl;
    std::printf(ok ? "Saved %s\n" : "Could not write everything in %s\n", dir.string().c_str());
    return ok ? dir : fs::path();
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

//int main(int argc, char** argv){
//
//@param1 - argc: how many command-line words there are
//@param2 - argv: the words; the file's header says what it accepts
//
//@brief - the program: 13w · Room Shell (no Kinect; after 13e). Just the room itself: every wall from the floor to
//         the ceiling, the floor and the ceiling, each a flat surface with its own picture (a texture), as an .obj
//         you can open in Blender or MeshLab. Setup, main loop and clean-up are laid out in the file's sections.
//         Returns 0 on a normal exit, 1 if it couldn't start.
int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Shell shell;
    if (!LoadShell(argc, argv, shell) || !FindSurfaces(shell)) return 1;                // steps 1-2
    auto start = std::chrono::steady_clock::now();
    std::mt19937 random(1);
    for (Surface& s : shell.surfaces)                                                   // steps 3-4 (new)
    {
        Texture(s, shell.shots, shell.info.depthCamera);
        Fill(s, random);
        ReadyToView(s);
        const double area = cv::countNonZero(s.inside) * s.texel * s.texel;
        std::printf("  %-8s %5.1f m^2 at %.0f mm texels: %3.0f%% seen, the rest filled from a %d cm square of its own pattern\n", s.name.c_str(),
                    area, s.texel * 1000, 100.0 * cv::countNonZero(s.seen) / std::max(1, cv::countNonZero(s.inside)), (int)std::lround(s.exemplar * s.texel * 100));
    }
    std::printf("Textured in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    shell.shots.clear();   // the pictures aren't needed any more

    cv::namedWindow(Window);
    spotter::OrbitView view(spotter::OrbitView::Options(), ForView(shell.middle), 3 * shell.radius);
    view.pitchDegrees = ViewPitchDegrees;
    view.Attach(Window);
    spotter::PointRenderer renderer;
    bool marked = true;
    while (true)
    {
        Show(shell, marked, view, renderer);                                            // step 5
        int key = cv::waitKey(15);
        if (spotter::IsQuitKey(key) || cv::getWindowProperty(Window, cv::WND_PROP_VISIBLE) < 1) break;
        if (key == 'g') marked = !marked;
        if (key == 'k') Save(shell);                                                    // step 6
        if (view.HandleKey(key) && key == 'r') view.pitchDegrees = ViewPitchDegrees;
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Press G: how much of each wall was really seen? Where would one more scan help most?
//  2. Set MinCeilingAreaMeters2 = 0.05: with a fan, the console's ceiling height may drop to the fan's.
//     That's why the ceiling is the highest BIG level surface, not just the highest.
//  3. Set BaseCellMeters = 0.05 on a wooden floor: the planks under the couch smear into stripes of
//     colour, because the planks are now "slow changes". Over a plank's width, they're pattern.
//  4. Set BlockCandidates = 1: the blocks no longer match their neighbours. Compare the floor under
//     the couch in Blender.
//  5. Set OutlierColour = 255 (nothing dropped) on a room scanned while the fan was spinning.
//  6. A door left open: the wall is filled shut. Where would a door cut-out need to come from
//     (13x's "points far behind a wall")?
//  7. Set TexelMeters = 0.005: sharper? The full colour picture has about 4 mm per pixel at 2 m.
