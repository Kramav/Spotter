// straight_room.hpp: Demo 8's three new ideas, beside it so check_straight_room can test them (D50's pattern).
//
//   1. Straighten  every scan stood up straight and set on the room's floor, before they're merged. Each scan was
//                  placed against earlier ones, so a small tilt in one is handed on to every scan placed after it.
//                  On the user's 80-scan hall the far end leaned 9 degrees: floor, ceiling and walls alike.
//   2. Ceiling map the ceiling found spot by spot (a height per 25 cm of the plan), not as one height: a hall's
//                  ceiling has steps, a lower part, and a slope.
//   3. Hanging     what hangs from the ceiling (lamps): pieces above head height that reach down to nothing.
//
// Coordinates: the room's, as spotter::TwinRoom's (X right, Y up, Z towards the first camera).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "spotter/geometry.hpp"
#include "spotter/room_scan.hpp"
#include "spotter/room_twin.hpp"
#include "spotter/stats.hpp"

namespace straight
{

// ---------------------------------------------------------------------
//  1. Straighten
// ---------------------------------------------------------------------

struct StraightenOptions
{
    float levelDegrees = 20;          // a reading facing within this of up or down is level (floor, ceiling, a table top),
                                      // within this of sideways upright (a wall, a curved wall too)
    float keepDegrees = 3;            // after the first fit, readings further off than this are left out (a slanted desk)
    int rounds = 4;
    float priorReadings = 300;        // each fix leans towards the previous scan's, worth this many readings: a scan that
                                      // sees one wall and no floor can't tell its tilt along that wall
    float floorDegrees = 10;          // a floor reading faces within this of straight up...
    float floorStepMeters = 0.02f;    // ...and the scan's floor is the lowest 2 cm step...
    int minFloorReadings = 150;       // ...holding this many of them
    float cameraSlackMeters = 0.15f;  // a floor is believed when the camera stands as high above it as over the last few
                                      // floors seen, within this (a table top is 0.7 m nearer)
    int cameraMemory = 5;             // "the last few"
    int referenceFloors = 3;          // the room's floor: the median of the first this many believed (scan 1 sets the room;
                                      // the earliest scans have handed on the least error)
};

struct ScanFix
{
    double tiltX = 0, tiltZ = 0;      // the turn about the room's X and Z axes that stood it up (radians)
    float lift = 0;                   // then moved up by this (metres)
    bool floorSeen = false;           // lift measured from its own floor (else taken from the scans either side)
    float cameraAboveFloor = NAN;     // how high the camera stood over the floor it saw
};

struct Straightened
{
    std::vector<ScanFix> fixes;       // one per scan, in order
    float floorY = 0;                 // the floor every scan was set on
    std::vector<std::string> report;
};

//cv::Vec2d FitTilt(const std::vector<cv::Point3f>& normals, const cv::Vec2d& prior, const StraightenOptions& options){
//
//@param1 - normals: a scan's facings, in the room's coordinates
//@param2 - prior: the previous scan's fix (tilt about X, about Z), which this one leans towards
//@param3 - options: the limits (StraightenOptions)
//
//@brief - the small turn w = (about X, about Z) that makes level readings face straight up or down and upright ones
//         face straight sideways, by least squares (small angles: a turned facing is n + w x n). A level reading gives
//         two equations, nx - wz ny = 0 and nz + wx ny = 0; an upright one gives one, wz nx - wx nz = -ny (a wall along
//         Z can't tell the tilt about Z: the prior holds that). Fitted `rounds` times, leaving out readings more than
//         keepDegrees off the last fit. O(rounds x normals).
inline cv::Vec2d FitTilt(const std::vector<cv::Point3f>& normals, const cv::Vec2d& prior, const StraightenOptions& options)
{
    const float levelCos = std::cos(options.levelDegrees * (float)CV_PI / 180), uprightSin = std::sin(options.levelDegrees * (float)CV_PI / 180);
    cv::Vec2d w = prior;
    for (int round = 0; round < options.rounds; round++)
    {
        const double keep = round == 0 ? 1.0 : options.keepDegrees * CV_PI / 180;
        cv::Matx22d a = cv::Matx22d::eye() * (double)options.priorReadings;
        cv::Vec2d b = prior * (double)options.priorReadings;
        for (const cv::Point3f& n : normals)
        {
            if (std::abs(n.y) > levelCos)
            {
                if (std::abs(n.x - w[1] * n.y) > keep || std::abs(n.z + w[0] * n.y) > keep) continue;
                a(0, 0) += n.y * n.y; b[0] += -n.z * n.y;   // nz + wx ny = 0
                a(1, 1) += n.y * n.y; b[1] += n.x * n.y;    // nx - wz ny = 0
            }
            else if (std::abs(n.y) < uprightSin)
            {
                if (std::abs(w[1] * n.x - w[0] * n.z + n.y) > keep) continue;
                const cv::Vec2d row(-n.z, n.x);              // wz nx - wx nz = -ny
                a += row * row.t();
                b += row * (double)-n.y;
            }
        }
        w = a.inv() * b;
    }
    return w;
}

//float FloorUnder(const std::vector<cv::Point3f>& points, const std::vector<cv::Point3f>& normals, const StraightenOptions& options){
//
//@param1 - points: a scan's points, stood up straight, in the room's coordinates
//@param2 - normals: their facings
//@param3 - options: the limits (StraightenOptions)
//
//@brief - the height of the lowest floorStepMeters step where at least minFloorReadings readings face up (a peak, not
//         a slope's foot), or NaN if none: the floor, or a table top when no floor is in view (Straighten checks which).
//         O(points).
inline float FloorUnder(const std::vector<cv::Point3f>& points, const std::vector<cv::Point3f>& normals, const StraightenOptions& options)
{
    const float upCos = std::cos(options.floorDegrees * (float)CV_PI / 180);
    std::vector<float> heights;
    for (size_t i = 0; i < points.size(); i++) if (normals[i].y > upCos) heights.push_back(points[i].y);
    if (heights.empty()) return NAN;
    const float lowest = *std::min_element(heights.begin(), heights.end());
    std::vector<int> steps(400, 0);   // 8 m of 2 cm steps
    for (float y : heights) { const int k = (int)((y - lowest) / options.floorStepMeters); if (k < (int)steps.size()) steps[k]++; }
    for (int k = 0; k < (int)steps.size(); k++)
        if (steps[k] >= options.minFloorReadings && steps[k] >= (k > 0 ? steps[k - 1] : 0) && steps[k] >= (k + 1 < (int)steps.size() ? steps[k + 1] : 0))
            return lowest + (k + 0.5f) * options.floorStepMeters;
    return NAN;
}

//Straightened Straighten(const std::vector<spotter::Scan*>& scans, const cv::Matx33f& firstCamera, const StraightenOptions& options){
//
//@param1 - scans: the placed scans in the order they were taken; their poses are changed
//@param2 - firstCamera: scan 1's camera -> the room (Y up), as room.txt's first camera turn
//@param3 - options: the limits (StraightenOptions)
//
//@brief - idea 1. Each scan, in order: its tilt fitted (FitTilt, leaning on the previous scan's) and undone by turning
//         it about its own camera; then its floor (FloorUnder). A floor is believed when the camera stands as high above
//         it as over the last cameraMemory floors believed (within cameraSlackMeters; the first guess is the median over
//         all scans), else it's a table top. The room's floor is the median of the first referenceFloors believed; each scan with a
//         believed floor is lifted onto it, and the rest by the lifts of the scans either side (in order, in proportion).
//         No floor anywhere: tilts only. O(scans x surface points).
inline Straightened Straighten(const std::vector<spotter::Scan*>& scans, const cv::Matx33f& firstCamera, const StraightenOptions& options)
{
    Straightened out;
    out.fixes.resize(scans.size());
    std::vector<float> floors(scans.size(), NAN), cameraHeights(scans.size(), NAN);
    std::vector<cv::Matx33f> turns(scans.size());
    std::vector<cv::Point3f> cameras(scans.size());
    cv::Vec2d prior(0, 0);
    for (size_t s = 0; s < scans.size(); s++)
    {
        const cv::Matx33f turn = firstCamera * scans[s]->pose.turn;
        const cv::Point3f camera = firstCamera * scans[s]->pose.shift;
        std::vector<cv::Point3f> normals, points;
        for (const cv::Point3f& n : scans[s]->surface.normals) normals.push_back(turn * n);
        const cv::Vec2d tilt = prior = FitTilt(normals, prior, options);
        const cv::Matx33f fix = spotter::TurnAbout(cv::Vec3d(tilt[0], 0, tilt[1]));
        for (size_t i = 0; i < normals.size(); i++)
        {
            normals[i] = fix * normals[i];
            points.push_back(fix * turn * scans[s]->surface.points[i] + camera);   // turned about its own camera
        }
        out.fixes[s].tiltX = tilt[0];
        out.fixes[s].tiltZ = tilt[1];
        turns[s] = fix * turn;
        cameras[s] = camera;
        floors[s] = FloorUnder(points, normals, options);
        cameraHeights[s] = camera.y - floors[s];
    }

    // Which floors to believe: the camera stands about as high over each as over the last few believed.
    std::vector<float> plausible;
    for (float h : cameraHeights) if (h > 0.3f) plausible.push_back(h);   // NaN fails too
    std::vector<float> recent;
    std::vector<float> roomFloors;
    if (!plausible.empty()) recent.push_back(spotter::Percentile(plausible, 0.5));
    for (size_t s = 0; s < scans.size(); s++)
    {
        if (recent.empty() || std::isnan(cameraHeights[s])) continue;
        const float typical = spotter::Percentile(recent, 0.5);
        if (std::abs(cameraHeights[s] - typical) > options.cameraSlackMeters) continue;
        out.fixes[s].floorSeen = true;
        out.fixes[s].cameraAboveFloor = cameraHeights[s];
        if (recent.size() == 1 && roomFloors.empty()) recent.clear();   // the first believed replaces the guess
        recent.push_back(cameraHeights[s]);
        if ((int)recent.size() > options.cameraMemory) recent.erase(recent.begin());
        if ((int)roomFloors.size() < options.referenceFloors) roomFloors.push_back(floors[s]);
    }
    if (!roomFloors.empty()) out.floorY = spotter::Percentile(roomFloors, 0.5);

    // Lifts: onto the room's floor where the floor was seen, in proportion between those scans elsewhere.
    std::vector<int> seen;
    for (size_t s = 0; s < scans.size(); s++) if (out.fixes[s].floorSeen) { seen.push_back((int)s); out.fixes[s].lift = out.floorY - floors[s]; }
    for (size_t s = 0; s < scans.size() && !seen.empty(); s++)
    {
        if (out.fixes[s].floorSeen) continue;
        auto after = std::lower_bound(seen.begin(), seen.end(), (int)s);
        if (after == seen.begin()) out.fixes[s].lift = out.fixes[*after].lift;
        else if (after == seen.end()) out.fixes[s].lift = out.fixes[seen.back()].lift;
        else
        {
            const int a = *(after - 1), b = *after;
            const float share = (float)((int)s - a) / (float)(b - a);
            out.fixes[s].lift = out.fixes[a].lift * (1 - share) + out.fixes[b].lift * share;
        }
    }

    // The new poses, back in scan 1's camera coordinates.
    const cv::Matx33f back = firstCamera.t();
    double most = 0;
    int mostScan = 0;
    float lowestLift = 0, highestLift = 0;
    std::vector<float> tilts;
    for (size_t s = 0; s < scans.size(); s++)
    {
        const ScanFix& f = out.fixes[s];
        scans[s]->pose.turn = back * turns[s];
        scans[s]->pose.shift = back * (cameras[s] + cv::Point3f(0, f.lift, 0));
        const double degrees = std::hypot(f.tiltX, f.tiltZ) * 180 / CV_PI;
        tilts.push_back((float)degrees);
        if (degrees > most) { most = degrees; mostScan = (int)s; }
        lowestLift = std::min(lowestLift, f.lift);
        highestLift = std::max(highestLift, f.lift);
    }
    std::vector<float> believed;
    for (const ScanFix& f : out.fixes) if (f.floorSeen) believed.push_back(f.cameraAboveFloor);
    out.report.push_back(cv::format("Straightened %zu scans: tilts undone, median %.1f deg, the most %.1f deg (the %d. scan); moved up or down by %+.2f to %+.2f m",
                                    scans.size(), tilts.empty() ? 0.0f : spotter::Percentile(tilts, 0.5), most, mostScan + 1, lowestLift, highestLift));
    out.report.push_back(believed.empty()
        ? std::string("  no scan saw a floor it could trust: tilts only, heights left as placed")
        : cv::format("  %zu scans saw the floor (the camera %.2f m above it, typically); the others were moved as the scans either side",
                     believed.size(), spotter::Percentile(believed, 0.5)));
    return out;
}

// ---------------------------------------------------------------------
//  2. The ceiling, spot by spot
// ---------------------------------------------------------------------

struct CeilingMapOptions
{
    float cellMeters = 0.25f;           // one ceiling height per cell this big, seen from below
    float facingDegrees = 60;           // a ceiling reading faces within this of straight down (a ceiling sloping to the walls too)
    float levelDegrees = 20;            // ...and a level one within this, either way (CeilingMapPoints)
    float minAboveFloorMeters = 1.8f;
    int minReadings = 4;                // a cell needs this many to have a height of its own
    double topShare = 0.95;             // its height: this far up its readings (a lamp under it doesn't pull it down)
    float stepMeters = 0.20f;           // neighbours this far apart in height are different pieces (a slope up to ~40 deg is one)
    float minPieceMeters2 = 0.5f;       // a piece smaller than this, lower than its surroundings or alone, is something hanging
    float bandMeters = 0.15f;           // level readings this close under the ceiling are the ceiling (a flush lamp too)
    float aboveMeters = 0.05f;          // anything this far above it is the ceiling too (recesses, stray readings)
};

struct CeilingMap
{
    bool seen = false;
    float x0 = 0, z0 = 0, cell = 0.25f;
    cv::Mat height;                     // CV_32F, rows along Z, columns along X; every cell filled once seen
    float lowest = 0, highest = 0;      // the pieces' heights (not the hanging ones')
    std::vector<std::string> report;

    //float At(float x, float z){
    //
    //@param1 - x: metres along X
    //@param2 - z: metres along Z
    //
    //@brief - the ceiling's height over that spot (the nearest cell's, outside the map). O(1).
    float At(float x, float z) const
    {
        const int c = std::clamp((int)std::floor((x - x0) / cell), 0, height.cols - 1), r = std::clamp((int)std::floor((z - z0) / cell), 0, height.rows - 1);
        return height.at<float>(r, c);
    }
};

//CeilingMap FindCeilingMap(const spotter::TwinRoom& room, float floorY, const CeilingMapOptions& options){
//
//@param1 - room: the levelled room
//@param2 - floorY: the floor's height
//@param3 - options: the limits (CeilingMapOptions)
//
//@brief - idea 2. Readings facing down (within facingDegrees), minAboveFloorMeters over the floor, binned into cells; a cell with minReadings
//         takes the height topShare up them. Neighbouring cells within stepMeters are one piece; a piece under
//         minPieceMeters2 that is lower than the cells around it (or alone) is a lamp or a beam, not ceiling: cleared.
//         Then every empty cell takes its highest filled neighbour's height, ring by ring, so the whole plan has one.
//         `report`: the pieces over 1 m^2, highest first. O(points + cells).
inline CeilingMap FindCeilingMap(const spotter::TwinRoom& room, float floorY, const CeilingMapOptions& options)
{
    CeilingMap map;
    map.cell = options.cellMeters;
    if (room.points.empty()) return map;
    float x1 = -1e9f, z1 = -1e9f;
    map.x0 = map.z0 = 1e9f;
    for (const cv::Point3f& p : room.points) { map.x0 = std::min(map.x0, p.x); map.z0 = std::min(map.z0, p.z); x1 = std::max(x1, p.x); z1 = std::max(z1, p.z); }
    const int columns = (int)((x1 - map.x0) / map.cell) + 1, rows = (int)((z1 - map.z0) / map.cell) + 1;
    std::vector<std::vector<float>> readings((size_t)columns * rows);
    const float downCos = std::cos(options.facingDegrees * (float)CV_PI / 180);
    for (size_t i = 0; i < room.points.size(); i++)
    {
        const cv::Point3f& p = room.points[i];
        if (room.normals[i].y > -downCos || p.y < floorY + options.minAboveFloorMeters) continue;
        readings[(size_t)((int)((p.z - map.z0) / map.cell) * columns + (int)((p.x - map.x0) / map.cell))].push_back(p.y);
    }
    map.height = cv::Mat(rows, columns, CV_32F, cv::Scalar(NAN));
    for (int k = 0; k < columns * rows; k++)
        if ((int)readings[k].size() >= options.minReadings) map.height.at<float>(k / columns, k % columns) = spotter::Percentile(readings[k], options.topShare);

    // Pieces: neighbouring cells within stepMeters.
    cv::Mat piece(rows, columns, CV_32S, cv::Scalar(-1));
    struct Piece { std::vector<cv::Point> cells; float height; float around = NAN; };
    std::vector<Piece> pieces;
    const cv::Point next4[4] = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < columns; c++)
        {
            if (std::isnan(map.height.at<float>(r, c)) || piece.at<int>(r, c) >= 0) continue;
            Piece p;
            std::queue<cv::Point> todo;
            todo.push({ c, r });
            piece.at<int>(r, c) = (int)pieces.size();
            std::vector<float> heights, aroundHeights;
            while (!todo.empty())
            {
                const cv::Point at = todo.front();
                todo.pop();
                p.cells.push_back(at);
                const float h = map.height.at<float>(at);
                heights.push_back(h);
                for (const cv::Point& d : next4)
                {
                    const cv::Point n = at + d;
                    if (n.x < 0 || n.y < 0 || n.x >= columns || n.y >= rows) continue;
                    const float hn = map.height.at<float>(n);
                    if (std::isnan(hn) || piece.at<int>(n) >= 0) continue;
                    if (std::abs(hn - h) >= options.stepMeters) continue;
                    piece.at<int>(n) = (int)pieces.size();
                    todo.push(n);
                }
            }
            p.height = spotter::Percentile(heights, 0.5);
            pieces.push_back(p);
        }
    // What surrounds each piece: the median of the filled cells next to it that belong to other pieces.
    for (size_t k = 0; k < pieces.size(); k++)
    {
        std::vector<float> around;
        for (const cv::Point& at : pieces[k].cells)
            for (const cv::Point& d : next4)
            {
                const cv::Point n = at + d;
                if (n.x < 0 || n.y < 0 || n.x >= columns || n.y >= rows) continue;
                if (piece.at<int>(n) >= 0 && piece.at<int>(n) != (int)k) around.push_back(map.height.at<float>(n));
            }
        if (!around.empty()) pieces[k].around = spotter::Percentile(around, 0.5);
    }
    const float cellArea = map.cell * map.cell;
    bool anyBig = false;
    for (const Piece& p : pieces) anyBig = anyBig || p.cells.size() * cellArea >= options.minPieceMeters2;
    int hanging = 0;
    map.lowest = 1e9f;
    map.highest = -1e9f;
    for (const Piece& p : pieces)
    {
        const bool small = p.cells.size() * cellArea < options.minPieceMeters2;
        const bool lower = !std::isnan(p.around) && p.around > p.height + options.stepMeters;
        if (anyBig && small && (lower || std::isnan(p.around)))
        {
            for (const cv::Point& at : p.cells) map.height.at<float>(at) = NAN;
            hanging++;
            continue;
        }
        if (small) continue;
        map.lowest = std::min(map.lowest, p.height);
        map.highest = std::max(map.highest, p.height);
    }
    map.seen = anyBig;
    if (!map.seen) { map.report.push_back("Ceiling: never seen (no piece of level readings facing down covers 0.5 m^2): nothing to hide"); return map; }

    // Fill the rest ring by ring from the filled cells: the highest neighbour (a ceiling's unseen part is not lower).
    const cv::Point next8[8] = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 }, { 1, 1 }, { -1, 1 }, { 1, -1 }, { -1, -1 } };
    for (bool grew = true; grew;)
    {
        grew = false;
        cv::Mat grown = map.height.clone();
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < columns; c++)
            {
                if (!std::isnan(map.height.at<float>(r, c))) continue;
                float best = NAN;
                for (const cv::Point& d : next8)
                {
                    const int rr = r + d.y, cc = c + d.x;
                    if (rr < 0 || cc < 0 || rr >= rows || cc >= columns) continue;
                    const float h = map.height.at<float>(rr, cc);
                    if (!std::isnan(h) && (std::isnan(best) || h > best)) best = h;
                }
                if (!std::isnan(best)) { grown.at<float>(r, c) = best; grew = true; }
            }
        map.height = grown;
    }

    std::vector<const Piece*> big;
    for (const Piece& p : pieces) if (p.cells.size() * cellArea >= 1.0f && !std::isnan(map.height.at<float>(p.cells[0]))) big.push_back(&p);
    std::sort(big.begin(), big.end(), [](const Piece* a, const Piece* b) { return a->height > b->height; });
    map.report.push_back(cv::format("Ceiling: found spot by spot, %.2f to %.2f m above the floor, in %zu pieces of 1 m^2 or more; %d small lower pieces are hanging things, not ceiling",
                                    map.lowest - floorY, map.highest - floorY, big.size(), hanging));
    for (size_t k = 0; k < big.size() && k < 8; k++)
        map.report.push_back(cv::format("  ceiling piece %.2f m above the floor: %.1f m^2 seen", big[k]->height - floorY, big[k]->cells.size() * cellArea));
    return map;
}

//std::vector<char> CeilingMapPoints(const spotter::TwinRoom& room, const CeilingMap& map, const CeilingMapOptions& options){
//
//@param1 - room: the levelled room
//@param2 - map: FindCeilingMap's
//@param3 - options: bandMeters and aboveMeters
//
//@brief - 1 for each point that is the ceiling where it stands: level (or facing down within facingDegrees) and
//         within bandMeters under the map's height there (or above it), or more than aboveMeters above it whatever its facing. Walls keep every upright point up
//         to their own spot's ceiling. All 0 if the ceiling wasn't seen. O(points).
inline std::vector<char> CeilingMapPoints(const spotter::TwinRoom& room, const CeilingMap& map, const CeilingMapOptions& options)
{
    std::vector<char> isCeiling(room.points.size(), 0);
    if (!map.seen) return isCeiling;
    const float levelCos = std::cos(options.levelDegrees * (float)CV_PI / 180), downCos = std::cos(options.facingDegrees * (float)CV_PI / 180);
    for (size_t i = 0; i < room.points.size(); i++)
    {
        const cv::Point3f& p = room.points[i];
        const float ceiling = map.At(p.x, p.z), ny = room.normals[i].y;
        const bool ceilingFacing = std::abs(ny) >= levelCos || ny <= -downCos;
        isCeiling[i] = (char)((ceilingFacing && p.y >= ceiling - options.bandMeters) || p.y > ceiling + options.aboveMeters);
    }
    return isCeiling;
}

// ---------------------------------------------------------------------
//  3. What hangs from the ceiling
// ---------------------------------------------------------------------

struct HangingOptions
{
    float aboveFloorMeters = 2.1f;   // only above a door's height
    float cubeMeters = 0.08f;        // points in touching cubes this big are one piece
    float clearMeters = 0.10f;       // a piece whose lowest point is this far above the line reaches down to nothing
};

//std::vector<char> HangingPoints(const spotter::TwinRoom& room, const std::vector<char>& isCeiling, float floorY, int& pieces, const HangingOptions& options){
//
//@param1 - room: the levelled room
//@param2 - isCeiling: the ceiling's points (CeilingMapPoints), left out
//@param3 - floorY: the floor's height
//@param4 - pieces: set to how many hanging pieces were found
//@param5 - options: the limits (HangingOptions)
//
//@brief - idea 3. The points over aboveFloorMeters (the ceiling's left out) in touching cubeMeters cubes are pieces; a
//         piece that comes no lower than clearMeters over that line hangs (a lamp, a fan, a sign). A wall's or a tall
//         shelf's top reaches down to the line, so it stays. 1 for each hanging point. O(points).
inline std::vector<char> HangingPoints(const spotter::TwinRoom& room, const std::vector<char>& isCeiling, float floorY, int& pieces,
                                       const HangingOptions& options)
{
    std::vector<char> hangs(room.points.size(), 0);
    pieces = 0;
    const float line = floorY + options.aboveFloorMeters;
    auto Key = [&](int x, int y, int z) { return ((long long)(x + 1000000) << 42) | ((long long)(y + 1000000) << 21) | (long long)(z + 1000000); };
    std::unordered_map<long long, std::vector<int>> cubes;
    for (size_t i = 0; i < room.points.size(); i++)
    {
        const cv::Point3f& p = room.points[i];
        if (isCeiling[i] || p.y <= line) continue;
        cubes[Key((int)std::floor(p.x / options.cubeMeters), (int)std::floor(p.y / options.cubeMeters), (int)std::floor(p.z / options.cubeMeters))].push_back((int)i);
    }
    std::unordered_map<long long, char> visited;
    for (const auto& [key, unused] : cubes)
    {
        if (visited.count(key)) continue;
        std::vector<long long> members;
        std::queue<long long> todo;
        todo.push(key);
        visited[key] = 1;
        float lowest = 1e9f;
        while (!todo.empty())
        {
            const long long at = todo.front();
            todo.pop();
            members.push_back(at);
            for (int i : cubes[at]) lowest = std::min(lowest, room.points[i].y);
            const int x = (int)((at >> 42) & 0x1FFFFF) - 1000000, y = (int)((at >> 21) & 0x1FFFFF) - 1000000, z = (int)(at & 0x1FFFFF) - 1000000;
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        const long long n = Key(x + dx, y + dy, z + dz);
                        if (visited.count(n) || !cubes.count(n)) continue;
                        visited[n] = 1;
                        todo.push(n);
                    }
        }
        if (lowest < line + options.clearMeters) continue;
        pieces++;
        for (long long at : members) for (int i : cubes[at]) hangs[i] = 1;
    }
    return hangs;
}

}
