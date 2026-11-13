// =====================================================================
//  13x · Room Fill                          (no Kinect; after 13e and 13z)
//
//  A scan only has what the Kinect saw: the wall behind the couch, the
//  floor under the table, the part of a table top under a box were never
//  seen. 13x finds the room's flat surfaces and paints their holes in,
//  and MARKS what it filled (blue stripes), so real and guessed stay
//  apart. It assumes a simple layout: walls upright, surfaces level (D32).
//
//    1. Load      a 13e room folder: every included scan merged at 3 cm,
//                 levelled, and its floor (spotter::FindTwinFloor, D33).
//    2. Fill      (new), once:
//                 a. walls: seen from above, upright points stacked over
//                    1 m high are a wall's path along the floor (straight,
//                    bent or curved). Each path is unrolled into a flat
//                    picture and filled from the floor up to the cut;
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
//    3. Look      the room up to the cut, as a doll's house; G switches
//                 marked / unmarked / off.
//    4. Save      K writes room_filled_<date>_<time>.ply into the room's
//                 folder: every real point plus the fill as shown.
//
//  It only fills FLAT things and legs it has readings of. It never invents
//  a chair back or a plant, and big holes get smeared colour.
//
//  Keys:   G  fill: marked / unmarked / off   K  save the .ply
//          mouse drag / W A S D  turn   wheel / Z X  zoom   R reset   Q or Esc  quit
//  Run:    13x_Room_Fill [room folder]   (none: an Open window starting in RoomsFolder, the newest room selected)
//  The filling is legacy Demo 6's (D31-D33), copied unchanged; Demo 6 shows it live.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <functional>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>   // cv::inpaint

#include "spotter/mesh.hpp"
#include "spotter/room_twin.hpp"
#include "spotter/stats.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below the window (spotter::WithKeys).
const std::string Keys = "G  fill: marked / unmarked / off   K  save the .ply   drag / W A S D  turn   wheel / Z X  zoom   R  reset   Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

const std::string RoomsFolder = SPOTTER_REPO_ROOT "/room_scans";   // 13e's default (gitignored, outside build/)
const float CubeMeters = 0.03f;                 // the room's points: one per 3 cm cube
const float CutAboveFloorMeters = 1.6f;         // walls are filled up to this; the view hides what's higher
const float UnderFloorMeters = 0.10f;           // readings this far under the floor are noise (D33)
const float ViewPitchDegrees = 35;              // looking down from above
const std::string Window = "13x - room filled";

// --- Filling the holes (new). A simple layout: walls upright (any path along the floor), surfaces
//     level (D32). Sloped ceilings, ramps or leaning walls need another mode (docs/ideas.md). ---
const float FillCellMeters = 0.03f;          // the grid on every surface: one guessed point per cell
const float FacingDegrees = 20;              // "upright" or "level": the normal within this of flat, or of up
const float SeenFrontMeters = 0.05f;         // a cell is seen if a real point is up to this far in front of it...
const float WallBehindMeters = 0.50f;        // ...or this far behind it, for a wall (our wall a little in front of the real one)
const float OtherBehindMeters = 0.10f;       // ...for a cabinet front or a table top (things stand close behind them)
const float WallStackMeters = 1.2f;          // a wall: upright points stacked this high in one spot, seen from above
const float MinStackPointsPerMeter = 15;     // ...and that densely (a few stray points 1 m apart are no wall)
const float MinWallLengthMeters = 1.0f;
const float MaxWallGapMeters = 1.2f;         // a wall's path broken by up to this (a doorway, a shadow) is joined up
const float MinWallCoverage = 0.4f;          // the share of a wall's length that must have been seen
const float PlaneThicknessMeters = 0.02f;    // a cabinet front's own points: within this of its plane
const int MaxUprightPlanes = 12;
const float LevelStepMeters = 0.02f;         // the height histogram's step, for level surfaces
const float LevelGatherMeters = 0.04f;       // a level surface's points: within this of its height (a slightly tilted top too)
const float TileMeters = 0.5f;               // a level surface's height is fitted per tile this big
const float LevelJoinMeters = 0.10f;         // parts of a level surface this close are one (a table top cut by a shadow)
const float MinSurfaceArea = 0.05f;          // m²: smaller flat pieces are left alone (a book, a lamp's top)
const float MinRectangleShare = 0.5f;        // a piece covering this much of its rectangle (a table top) fills to it
const double InpaintRadiusCells = 5;         // how far cv::inpaint looks for colour
// Legs (new): under a top higher than this, a blob of readings no wider than this, spanning this
// share of the floor-to-top gap and crossing this height up it (the top itself hides the far
// legs' upper part, a chair the near legs' lower part; a box's sides on the table don't reach
// down that far), with no seat above it (then it's a chair's leg).
const float MinTableHeightMeters = 0.3f;
const float MaxLegWidthMeters = 0.12f;
const float MinLegSpanShare = 0.15f;
const float LegCrossShare = 0.5f;
// Shadows (new): only within the first camera's view (the Kinect's field of view) and range.
const float GhostRangeMeters = 4.5f;
const float GhostMinArea = 0.03f;            // m²
// A shadow's own readings, as the first camera saw them: up to this far below it (a dark chair's
// legs, a TV's stand), and this far to either side.
const float ConeDegrees = 8, ConeSideDegrees = 2;
const float HalfWideDegrees = 35, HalfTallDegrees = 30;
const int StripeCells = 3;                   // the marking: diagonal bands this many cells wide
const cv::Vec3b MarkColour(255, 128, 0);     // B, G, R: the blue of filled-in points
const cv::Vec3b GhostColour(200, 60, 200);   // the purple of "something dark"

// ---------------------------------------------------------------------
//  Step 2 (new): how the holes are filled (parts a-f)
// ---------------------------------------------------------------------

// What the filling made: a point per filled cell (blue), and the shadows' cut-outs (purple).
struct Filled
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colours;       // the guessed colour (inpainted)
    std::vector<char> stripe;             // which marking band
    std::vector<cv::Point3f> ghosts;
    std::vector<char> ghostStripe;
    std::vector<float> ghostPatches;      // a cut-out nearer the camera is smaller
    double floorArea = 0, wallArea = 0, otherArea = 0;   // m²
    int walls = 0, others = 0, legs = 0, shadows = 0;
    bool floorSeen = false;
    std::vector<std::string> report;      // one line per surface, for the console
};

//void AddFilled(Filled& out, const cv::Point3f& p, const cv::Vec3b& colour, int row, int column, double& area){
//
//@param1 - out: the fill so far; one point is added
//@param2 - p: the filled cell's centre in the room
//@param3 - colour: its guessed colour
//@param4 - row: the cell's row
//@param5 - column: its column
//@param6 - area: the surface's filled area; grows by one cell
//
//@brief - adds one filled cell: its point, colour, and which marking band it falls in (stripes every StripeCells),
//         and counts its area. O(1).
void AddFilled(Filled& out, const cv::Point3f& p, const cv::Vec3b& colour, int row, int column, double& area)
{
    out.points.push_back(p);
    out.colours.push_back(colour);
    out.stripe.push_back((char)(((row + column) / StripeCells) % 2));
    area += FillCellMeters * FillCellMeters;
}

//cv::Mat PaintIn(const cv::Mat& sum, const cv::Mat& count){
//
//@param1 - sum: per cell, the sum of the seen colours (CV_32FC3)
//@param2 - count: per cell, how many readings (CV_32F)
//
//@brief - the colours of a surface's seen cells (count > 0, the average), and cv::inpaint (Telea) for the rest,
//         from the edges in. O(cells).
cv::Mat PaintIn(const cv::Mat& sum, const cv::Mat& count)
{
    cv::Mat picture(sum.size(), CV_8UC3, cv::Scalar::all(128)), unknown(sum.size(), CV_8U, cv::Scalar(255));
    for (int r = 0; r < sum.rows; r++)
        for (int c = 0; c < sum.cols; c++)
        {
            float n = count.at<float>(r, c);
            if (n == 0) continue;
            picture.at<cv::Vec3b>(r, c) = cv::Vec3b(sum.at<cv::Vec3f>(r, c) / n);
            unknown.at<uchar>(r, c) = 0;
        }
    if (cv::countNonZero(unknown) < (int)unknown.total()) cv::inpaint(picture, unknown, picture, InpaintRadiusCells, cv::INPAINT_TELEA);
    return picture;
}

// --- The room seen from above, and lines of sight from the first camera (at 0,0,0) ---

struct Plan
{
    float x0 = 0, z0 = 0;
    int columns = 0, rows = 0;
    //int Index(float x, float z){
    //
    //@param1 - x: metres to the right
    //@param2 - z: metres ahead
    //
    //@brief - the plan cell index for that spot, or -1 if it's outside the plan. O(1).
    int Index(float x, float z) const
    {
        int c = (int)std::floor((x - x0) / FillCellMeters), r = (int)std::floor((z - z0) / FillCellMeters);
        return c < 0 || r < 0 || c >= columns || r >= rows ? -1 : r * columns + c;
    }
};

// The kept walls, seen from above: for each plan cell within a cell and a half of a wall's path,
// which wall and which of its columns (-1: none).
struct WallMap
{
    Plan plan;
    std::vector<int> wall, column;
};

//float MeetWall(const WallMap& map, cv::Point2f to, float from, float until, int& wall, int& column){
//
//@param1 - map: the kept walls seen from above
//@param2 - to: the plan point to look towards, relative to the first camera
//@param3 - from: the plan distance to start at
//@param4 - until: the plan distance to stop at
//@param5 - wall: set to the wall met
//@param6 - column: set to which of its columns
//
//@brief - walks the first camera's line of sight towards the plan point in half cells and returns the distance
//         where it first meets a kept wall (-1: none, or it left the plan). O((until - from) / half a cell).
float MeetWall(const WallMap& map, cv::Point2f to, float from, float until, int& wall, int& column)
{
    float length = (float)cv::norm(to);
    if (length < 1e-3f || map.wall.empty()) return -1;
    cv::Point2f direction = to / length;
    for (float s = from; s < until; s += FillCellMeters / 2)
    {
        int k = map.plan.Index(direction.x * s, direction.y * s);
        if (k < 0) return -1;
        if (map.wall[k] < 0) continue;
        wall = map.wall[k];
        column = map.column[k];
        return s;
    }
    return -1;
}

// A grid of FillCellMeters cells on the plane "normal . p = offset". u runs along the columns and
// v along the rows (normal, u and v at right angles, each 1 long), starting at (uMin, vMin).
struct Grid
{
    cv::Point3f normal, u, v;
    float offset = 0, uMin = 0, vMin = 0;
    int columns = 0, rows = 0;
    //bool Cell(const cv::Point3f& p, int& row, int& column){
    //
    //@param1 - p: a point in the room
    //@param2 - row: set to its row in the grid
    //@param3 - column: set to its column
    //
    //@brief - which grid cell the point projects into (along u for columns, v for rows). False if it falls outside the
    //         grid. O(1).
    bool Cell(const cv::Point3f& p, int& row, int& column) const
    {
        column = (int)std::floor((u.dot(p) - uMin) / FillCellMeters);
        row = (int)std::floor((v.dot(p) - vMin) / FillCellMeters);
        return column >= 0 && row >= 0 && column < columns && row < rows;
    }
    //cv::Point3f Centre(int row, int column, float at){
    //
    //@param1 - row: a cell's row
    //@param2 - column: its column
    //@param3 - at: its distance along the normal
    //
    //@brief - the room point at the middle of that cell, `at` along the normal. O(1).
    cv::Point3f Centre(int row, int column, float at) const
    {
        return normal * at + u * (uMin + (column + 0.5f) * FillCellMeters) + v * (vMin + (row + 0.5f) * FillCellMeters);
    }
};

//Grid MakeGrid(const cv::Point3f& normal, const cv::Point3f& u, const cv::Point3f& v, float offset, float uMin, float uMax, float vMin, float vMax){
//
//@param1 - normal: the plane's normal (unit)
//@param2 - u: the direction of the columns (unit)
//@param3 - v: the direction of the rows (unit)
//@param4 - offset: the plane: normal . p = offset
//@param5 - uMin: where the columns start
//@param6 - uMax: where they end
//@param7 - vMin: where the rows start
//@param8 - vMax: where they end
//
//@brief - a grid of FillCellMeters cells on the plane covering [uMin, uMax] x [vMin, vMax] (at least one cell each
//         way). O(1).
Grid MakeGrid(const cv::Point3f& normal, const cv::Point3f& u, const cv::Point3f& v, float offset,
              float uMin, float uMax, float vMin, float vMax)
{
    Grid grid{ normal, u, v, offset, uMin, vMin };
    grid.columns = std::max(1, (int)std::ceil((uMax - uMin) / FillCellMeters));
    grid.rows = std::max(1, (int)std::ceil((vMax - vMin) / FillCellMeters));
    return grid;
}

//std::vector<cv::Mat> PieceDomains(const cv::Mat& on, int join){
//
//@param1 - on: a surface's cells that hold its own points (255)
//@param2 - join: parts up to this many cells apart are one piece (default 0)
//
//@brief - for a table top, a seat or a cabinet front: the cells to fill for each piece of `on` of at least
//         MinSurfaceArea. A piece covering most of its rectangle (a table top with a box on it) fills to the
//         rectangle; any other shape (an L-shaped desk) only fills the holes it encloses. Parts up to `join` cells
//         apart are one piece (a table top cut in two by the shadow of something standing on it). O(cells).
std::vector<cv::Mat> PieceDomains(const cv::Mat& on, int join = 0)
{
    std::vector<cv::Mat> domains;
    cv::Mat closed, joined, labels, stats, centroids;
    cv::morphologyEx(on, closed, cv::MORPH_CLOSE, cv::Mat());   // join specks a cell apart
    if (join > 0) cv::dilate(closed, joined, cv::Mat(), cv::Point(-1, -1), join);
    else joined = closed;
    int count = cv::connectedComponentsWithStats(joined, labels, stats, centroids, 8);
    const int minCells = (int)(MinSurfaceArea / (FillCellMeters * FillCellMeters));
    for (int label = 1; label < count; label++)
    {
        cv::Mat piece = (labels == label) & closed, domain;
        int area = cv::countNonZero(piece);
        if (area < minCells) continue;
        std::vector<cv::Point> cells;
        cv::findNonZero(piece, cells);
        cv::RotatedRect rect = cv::minAreaRect(cells);
        if (area >= MinRectangleShare * (rect.size.width + 1) * (rect.size.height + 1))
        {
            cv::Point2f corners[4];
            rect.points(corners);
            std::vector<cv::Point> polygon;
            for (const cv::Point2f& c : corners) polygon.emplace_back(cvRound(c.x), cvRound(c.y));
            domain = piece.clone();
            cv::fillConvexPoly(domain, polygon, cv::Scalar(255));
        }
        else
        {
            // The holes it encloses: whatever the outside can't reach.
            cv::Mat padded, whole;
            if (join > 0) cv::morphologyEx(piece, whole, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), join);
            else whole = piece;
            cv::copyMakeBorder(whole, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
            cv::floodFill(padded, cv::Point(0, 0), cv::Scalar(128));
            domain = padded(cv::Rect(1, 1, on.cols, on.rows)) != 128;
        }
        domains.push_back(domain);
    }
    return domains;
}

// Fills one plane's "domain" cells that no room point is near: a real point up to SeenFrontMeters
// in front of the cell (on the first camera's side) or up to "behind" behind it makes it seen.
// "at" = the plane's offset per cell (a level surface's tiles), or empty for one offset.
// Returns the area filled.
double FillSurface(const Grid& grid, const cv::Mat& domain, const std::vector<cv::Point3f>& members,
                   const std::vector<cv::Vec3b>& memberColours, const std::vector<cv::Point3f>& room,
                   float behind, const cv::Mat& at, float cutY, Filled& out)
{
    auto OffsetAt = [&](int r, int c) { return at.empty() ? grid.offset : at.at<float>(r, c); };
    const float front = grid.offset <= 0 ? 1.0f : -1.0f;   // the side the first camera (0,0,0) is on
    int row, column;
    cv::Mat seen(grid.rows, grid.columns, CV_8U, cv::Scalar(0));
    for (const cv::Point3f& p : room)
    {
        if (!grid.Cell(p, row, column)) continue;
        float gap = front * (grid.normal.dot(p) - OffsetAt(row, column));
        if (gap <= SeenFrontMeters && gap >= -behind) seen.at<uchar>(row, column) = 255;
    }
    cv::dilate(seen, seen, cv::Mat());   // a real point in this cell or the next: seen
    cv::Mat fill = domain & ~seen;
    if (cv::countNonZero(fill) == 0) return 0;

    cv::Mat sum(grid.rows, grid.columns, CV_32FC3, cv::Scalar::all(0)), count(grid.rows, grid.columns, CV_32F, cv::Scalar(0));
    for (size_t i = 0; i < members.size(); i++)
    {
        if (!grid.Cell(members[i], row, column)) continue;
        sum.at<cv::Vec3f>(row, column) += cv::Vec3f(memberColours[i]);
        count.at<float>(row, column) += 1;
    }
    cv::Mat picture = PaintIn(sum, count);
    double area = 0;
    for (int r = 0; r < grid.rows; r++)
        for (int c = 0; c < grid.columns; c++)
        {
            if (!fill.at<uchar>(r, c)) continue;
            cv::Point3f p = grid.Centre(r, c, OffsetAt(r, c));
            if (p.y > cutY) continue;   // the doll's house has no ceiling
            AddFilled(out, p, picture.at<cv::Vec3b>(r, c), r, c, area);
        }
    return area;
}

// --- Step a: walls, as paths along the floor ---

// One wall: its path's plan cells in order along it, unrolled into columns (along) x rows (height,
// from the floor up to the cut).
struct Wall
{
    std::vector<int> cells;
    std::vector<int> cellColumn;          // per path cell, its column
    std::vector<cv::Point2f> at, facing;  // per column: where it stands, and which way it faces (towards the first camera)
    std::vector<char> columnSeen;         // per column: any real point on it, at any height
    cv::Mat seen, through, explained, sum, count;
    std::vector<int> members;             // the room points seen on it
    float length = 0;
    bool kept = false;
};

//WallMap FillWalls(const std::vector<cv::Point3f>& room, const std::vector<cv::Point3f>& normals, const std::vector<cv::Vec3b>& colours, float floorY, float cutY, const cv::Matx33f& firstCamera, std::vector<char>& claimed, Filled& out){
//
//@param1 - room: the room's points, levelled (Y up)
//@param2 - normals: which way each faces
//@param3 - colours: each point's colour
//@param4 - floorY: the floor's height
//@param5 - cutY: fill up to this height
//@param6 - firstCamera: scan 1's camera -> the room, for the lines of sight
//@param7 - claimed: set to 1 for each point seen on a kept wall
//@param8 - out: the fill so far; the walls' cells and shadows are added
//
//@brief - steps a and e for walls, and step f's shadows. 1. Stacks: upright points per plan cell and how high they
//         reach; 2. thin the paths to one cell across; 3. bridge gaps up to MaxWallGapMeters between path ends
//         pointing at each other; 4. split where three paths meet and order each path from one end; 5. keep real
//         walls (long enough, seen along enough of their length, not seen only through another wall); then lines
//         of sight from the first camera find points seen through an opening (a door) and where a point casts its
//         shadow; 6. fill each kept wall, unrolled into a flat picture, from the floor up to the cut, and the big
//         unexplained shadows as "something dark stood here". Returns the kept walls' map. O(points + wall cells).
WallMap FillWalls(const std::vector<cv::Point3f>& room, const std::vector<cv::Point3f>& normals,
                  const std::vector<cv::Vec3b>& colours, float floorY, float cutY, const cv::Matx33f& firstCamera,
                  std::vector<char>& claimed, Filled& out)
{
    const float cell = FillCellMeters;
    const float uprightSin = std::sin(FacingDegrees * (float)CV_PI / 180);
    WallMap map;
    Plan& plan = map.plan;
    std::vector<float> xs, zs;
    for (const cv::Point3f& p : room) { xs.push_back(p.x); zs.push_back(p.z); }
    // The middle 99% of the room, plus the first camera (a stray point 50 m away would make it huge).
    plan.x0 = std::min(spotter::Percentile(xs, 0.005), 0.0f) - 0.5f;
    plan.z0 = std::min(spotter::Percentile(zs, 0.005), 0.0f) - 0.5f;
    plan.columns = (int)std::ceil((std::max(spotter::Percentile(xs, 0.995), 0.0f) + 0.5f - plan.x0) / cell);
    plan.rows = (int)std::ceil((std::max(spotter::Percentile(zs, 0.995), 0.0f) + 0.5f - plan.z0) / cell);
    const int cells = plan.columns * plan.rows, heightRows = std::max(1, (int)std::ceil((cutY - floorY) / cell));
    auto Centre = [&](int k) { return cv::Point2f(plan.x0 + (k % plan.columns + 0.5f) * cell, plan.z0 + (k / plan.columns + 0.5f) * cell); };
    auto RowOf = [&](float y) { return (int)std::floor((y - floorY) / cell); };

    // 1. Stacks: upright points per plan cell, and how high they reach.
    std::vector<int> count(cells, 0);
    std::vector<float> low(cells, 1e9f), high(cells, -1e9f);
    std::vector<cv::Point2f> sum(cells, cv::Point2f(0, 0)), facing(cells, cv::Point2f(0, 0));
    for (size_t i = 0; i < room.size(); i++)
    {
        if (std::abs(normals[i].y) >= uprightSin) continue;
        int k = plan.Index(room[i].x, room[i].z);
        if (k < 0) continue;
        cv::Point2f n(normals[i].x, normals[i].z), toCamera(-room[i].x, -room[i].z);
        if (n.dot(toCamera) < 0) n = -n;   // every normal turned to face the first camera
        count[k]++;
        low[k] = std::min(low[k], room[i].y);
        high[k] = std::max(high[k], room[i].y);
        sum[k] += cv::Point2f(room[i].x, room[i].z);
        facing[k] += n;
    }

    // 2. Thin: keep the cell with the most points across the wall's thickness (a merged wall is a
    //    few cm thick, or doubled), placed at its points' mean.
    auto Stack = [&](int k) { return count[k] > 0 && high[k] - low[k] >= WallStackMeters && count[k] >= MinStackPointsPerMeter * (high[k] - low[k]); };
    cv::Mat path(plan.rows, plan.columns, CV_8U, cv::Scalar(0));
    std::vector<cv::Point2f> position(cells), direction(cells);
    for (int k = 0; k < cells; k++)
    {
        if (!Stack(k)) continue;
        cv::Point2f n = facing[k];
        float length = (float)cv::norm(n);
        n = length > 1e-6f ? n / length : -Centre(k) / std::max(1e-3f, (float)cv::norm(Centre(k)));
        int r = k / plan.columns, c = k % plan.columns;
        bool best = true;
        for (int s : { -2, -1, 1, 2 })
        {
            int rr = r + (int)std::lround(n.y * s), cc = c + (int)std::lround(n.x * s);
            if (rr < 0 || cc < 0 || rr >= plan.rows || cc >= plan.columns) continue;
            int j = rr * plan.columns + cc;
            if (!Stack(j)) continue;
            if (count[j] > count[k] || (count[j] == count[k] && j < k)) best = false;
        }
        if (!best) continue;
        path.data[k] = 255;
        position[k] = sum[k] / (float)count[k];
        direction[k] = n;
    }

    // Around a path cell: how many path cells touch it, and in how many separate runs.
    const int ringR[8] = { -1, -1, -1, 0, 1, 1, 1, 0 }, ringC[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };
    auto Runs = [&](const cv::Mat& m, int k, int& touching)
    {
        int r = k / plan.columns, c = k % plan.columns, runs = 0;
        bool on[8];
        touching = 0;
        for (int i = 0; i < 8; i++)
        {
            int rr = r + ringR[i], cc = c + ringC[i];
            on[i] = rr >= 0 && cc >= 0 && rr < plan.rows && cc < plan.columns && m.data[rr * plan.columns + cc];
            touching += on[i];
        }
        for (int i = 0; i < 8; i++) if (on[i] && !on[(i + 7) % 8]) runs++;
        return runs;
    };

    // 3. Bridge the gaps: two path ends up to MaxWallGapMeters apart, pointing at each other.
    {
        cv::Mat labels;
        cv::connectedComponents(path, labels, 8, CV_32S);
        struct End { int k, label; cv::Point2f at, out; bool used; };
        std::vector<End> ends;
        for (int k = 0; k < cells; k++)
        {
            int touching;
            if (!path.data[k] || Runs(path, k, touching) != 1 || touching > 2) continue;
            int r = k / plan.columns, c = k % plan.columns, label = labels.at<int>(r, c);
            cv::Point2f out(0, 0);
            for (int dr = -5; dr <= 5; dr++)
                for (int dc = -5; dc <= 5; dc++)
                {
                    int rr = r + dr, cc = c + dc;
                    if (rr < 0 || cc < 0 || rr >= plan.rows || cc >= plan.columns || labels.at<int>(rr, cc) != label) continue;
                    out += Centre(k) - Centre(rr * plan.columns + cc);
                }
            float length = (float)cv::norm(out);
            if (length > 1e-6f) ends.push_back({ k, label, Centre(k), out / length, false });
        }
        const float aimCos = std::cos(25 * (float)CV_PI / 180);
        for (End& a : ends)
        {
            if (a.used) continue;
            End* best = nullptr;
            float bestDistance = MaxWallGapMeters;
            for (End& b : ends)
            {
                if (b.used || b.label == a.label) continue;
                cv::Point2f d = b.at - a.at;
                float distance = (float)cv::norm(d);
                if (distance > bestDistance || distance < 1e-6f) continue;
                if (d.dot(a.out) / distance < aimCos || -d.dot(b.out) / distance < aimCos) continue;
                best = &b;
                bestDistance = distance;
            }
            if (!best) continue;
            a.used = best->used = true;
            cv::Point2f bridgeFacing = direction[a.k] + direction[best->k];
            bridgeFacing /= std::max(1e-6f, (float)cv::norm(bridgeFacing));
            cv::LineIterator line(path, cv::Point(a.k % plan.columns, a.k / plan.columns),
                                  cv::Point(best->k % plan.columns, best->k / plan.columns), 8);
            for (int i = 0; i < line.count; i++, ++line)
            {
                int k = line.pos().y * plan.columns + line.pos().x;
                if (path.data[k]) continue;
                path.data[k] = 255;
                position[k] = Centre(k);   // nothing was read here: the cell's centre
                direction[k] = bridgeFacing;
            }
        }
    }

    // 4. Split where three paths meet, then order each path from one end (shortest walk along it).
    cv::Mat paths = path.clone();
    for (int k = 0; k < cells; k++)
    {
        int touching;
        if (path.data[k] && Runs(path, k, touching) >= 3) paths.data[k] = 0;
    }
    if (cv::countNonZero(paths) == 0) return map;
    cv::Mat labels;
    int pathCount = cv::connectedComponents(paths, labels, 8, CV_32S);
    std::vector<Wall> walls(pathCount - 1);
    std::vector<int> wallOf(cells, -1), indexOf(cells, -1);
    for (int k = 0; k < cells; k++)
    {
        int label = labels.ptr<int>()[k];
        if (label == 0) continue;
        wallOf[k] = label - 1;
        indexOf[k] = (int)walls[label - 1].cells.size();
        walls[label - 1].cells.push_back(k);
    }
    auto Walk = [&](const Wall& wall, int start, std::vector<float>& distance)
    {
        distance.assign(wall.cells.size(), 1e9f);
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<>> queue;
        distance[indexOf[start]] = 0;
        queue.push({ 0.0f, start });
        while (!queue.empty())
        {
            auto [d, k] = queue.top();
            queue.pop();
            if (d > distance[indexOf[k]]) continue;
            int r = k / plan.columns, c = k % plan.columns;
            for (int i = 0; i < 8; i++)
            {
                int rr = r + ringR[i], cc = c + ringC[i];
                if (rr < 0 || cc < 0 || rr >= plan.rows || cc >= plan.columns) continue;
                int j = rr * plan.columns + cc;
                if (wallOf[j] != wallOf[k]) continue;
                float next = d + (ringR[i] != 0 && ringC[i] != 0 ? 1.4142f : 1.0f) * cell;
                if (next < distance[indexOf[j]]) { distance[indexOf[j]] = next; queue.push({ next, j }); }
            }
        }
    };
    for (Wall& wall : walls)
    {
        std::vector<float> distance;
        Walk(wall, wall.cells[0], distance);
        int far = wall.cells[std::max_element(distance.begin(), distance.end()) - distance.begin()];
        Walk(wall, far, distance);
        wall.length = *std::max_element(distance.begin(), distance.end()) + cell;
        int columns = (int)std::lround(wall.length / cell);
        wall.at.assign(columns, cv::Point2f(0, 0));
        wall.facing.assign(columns, cv::Point2f(0, 0));
        std::vector<int> used(columns, 0);
        for (size_t i = 0; i < wall.cells.size(); i++)
        {
            int column = std::min(columns - 1, (int)std::lround(distance[i] / cell));
            wall.cellColumn.push_back(column);
            wall.at[column] += position[wall.cells[i]];
            wall.facing[column] += direction[wall.cells[i]];
            used[column]++;
        }
        // Columns no cell landed on (a diagonal stretch walks 1.4 cells per step): between their neighbours.
        for (int c = 0; c < columns; c++)
        {
            if (used[c]) { wall.at[c] /= (float)used[c]; continue; }
            int before = c - 1, after = c + 1;
            while (before >= 0 && !used[before]) before--;
            while (after < columns && !used[after]) after++;
            int a = before >= 0 ? before : after, b = after < columns ? after : before;
            float t = a == b ? 0 : (float)(c - a) / (b - a);
            cv::Point2f atA = wall.at[a] / (float)std::max(1, used[a]), atB = wall.at[b] / (float)std::max(1, used[b]);
            wall.at[c] = atA * (1 - t) + atB * t;
            wall.facing[c] = wall.facing[a] + wall.facing[b];
        }
        for (cv::Point2f& f : wall.facing) f /= std::max(1e-6f, (float)cv::norm(f));
        wall.columnSeen.assign(columns, 0);
        for (cv::Mat* m : { &wall.seen, &wall.through, &wall.explained }) *m = cv::Mat(heightRows, columns, CV_8U, cv::Scalar(0));
        wall.sum = cv::Mat(heightRows, columns, CV_32FC3, cv::Scalar::all(0));
        wall.count = cv::Mat(heightRows, columns, CV_32F, cv::Scalar(0));
    }
    std::vector<int> columnOf(cells, -1);
    for (Wall& wall : walls)
        for (size_t i = 0; i < wall.cells.size(); i++) columnOf[wall.cells[i]] = wall.cellColumn[i];

    // Step e for walls: each room point goes to its nearest path cell (seen from above). Seen if it
    // lies from SeenFrontMeters in front of the wall to WallBehindMeters behind it.
    cv::Mat notPath = paths == 0, distance, nearest;
    cv::distanceTransform(notPath, distance, nearest, cv::DIST_L2, cv::DIST_MASK_5, cv::DIST_LABEL_PIXEL);
    std::vector<int> cellOfLabel(cells + 1, -1);
    for (int k = 0; k < cells; k++) if (paths.data[k]) cellOfLabel[nearest.ptr<int>()[k]] = k;
    for (size_t i = 0; i < room.size(); i++)
    {
        const cv::Point3f& p = room[i];
        int k = plan.Index(p.x, p.z);
        if (k < 0 || distance.ptr<float>()[k] > (WallBehindMeters + cell) / cell) continue;
        int pathCell = cellOfLabel[nearest.ptr<int>()[k]];
        if (pathCell < 0) continue;
        Wall& wall = walls[wallOf[pathCell]];
        int column = columnOf[pathCell];
        cv::Point2f d = cv::Point2f(p.x, p.z) - wall.at[column];
        float gap = d.dot(wall.facing[column]);
        if (cv::norm(d - wall.facing[column] * gap) > 1.5f * cell || gap > SeenFrontMeters || gap < -WallBehindMeters) continue;
        wall.columnSeen[column] = 1;
        wall.members.push_back((int)i);
        int row = RowOf(p.y);
        if (row < 0 || row >= heightRows) continue;
        wall.seen.at<uchar>(row, column) = 255;
        wall.sum.at<cv::Vec3f>(row, column) += cv::Vec3f(colours[i]);
        wall.count.at<float>(row, column) += 1;
    }

    // 5. Keep real walls: long enough, seen along enough of their length, and not seen only
    //    through another wall (a hallway's wall, through a door).
    std::vector<char> candidate(walls.size(), 0);
    for (size_t w = 0; w < walls.size(); w++)
    {
        int seenColumns = (int)std::count(walls[w].columnSeen.begin(), walls[w].columnSeen.end(), 1);
        candidate[w] = walls[w].length >= MinWallLengthMeters && seenColumns >= MinWallCoverage * walls[w].columnSeen.size();
    }
    for (size_t w = 0; w < walls.size(); w++)
    {
        Wall& wall = walls[w];
        if (!candidate[w]) continue;
        int tried = 0, hidden = 0;
        const size_t step = std::max<size_t>(1, wall.members.size() / 300);
        for (size_t m = 0; m < wall.members.size(); m += step)
        {
            const cv::Point3f& p = room[wall.members[m]];
            cv::Point2f to(p.x, p.z);
            float length = (float)cv::norm(to);
            tried++;
            for (float s = cell; s < length - 3 * cell; s += cell / 2)
            {
                int k = plan.Index(to.x / length * s, to.y / length * s);
                if (k < 0) break;
                if (wallOf[k] >= 0 && wallOf[k] != (int)w && candidate[wallOf[k]]) { hidden++; break; }
            }
        }
        wall.kept = tried > 0 && hidden * 2 < tried;
    }

    // The kept walls' map, for lines of sight (a cell and a half either side of each path).
    cv::Mat notKept(plan.rows, plan.columns, CV_8U, cv::Scalar(255)), keptDistance, keptNearest;
    for (size_t w = 0; w < walls.size(); w++)
        if (walls[w].kept) for (int k : walls[w].cells) notKept.data[k] = 0;
    map.wall.assign(cells, -1);
    map.column.assign(cells, -1);
    if (cv::countNonZero(notKept) < cells)
    {
        cv::distanceTransform(notKept, keptDistance, keptNearest, cv::DIST_L2, cv::DIST_MASK_5, cv::DIST_LABEL_PIXEL);
        std::vector<int> keptCellOfLabel(cells + 1, -1);
        for (int k = 0; k < cells; k++) if (!notKept.data[k]) keptCellOfLabel[keptNearest.ptr<int>()[k]] = k;
        for (int k = 0; k < cells; k++)
        {
            if (keptDistance.ptr<float>()[k] > 1.5f) continue;
            int pathCell = keptCellOfLabel[keptNearest.ptr<int>()[k]];
            if (pathCell < 0) continue;
            map.wall[k] = wallOf[pathCell];
            map.column[k] = columnOf[pathCell];
        }
    }

    // Lines of sight from the first camera to every point: a kept wall met well before the point
    // means it was seen THROUGH an opening there; the first wall met beyond the point is where
    // the point casts its shadow (step f).
    for (const cv::Point3f& p : room)
    {
        cv::Point2f to(p.x, p.z);
        float length = (float)cv::norm(to);
        if (length < 0.1f) continue;
        int w, column;
        float s = MeetWall(map, to, cell, length - cell, w, column);
        if (s > 0 && length - s > WallBehindMeters)
        {
            int row = RowOf(p.y * s / length);   // the line of sight's height there (the camera is at height 0)
            if (row >= 0 && row < heightRows) walls[w].through.at<uchar>(row, column) = 255;
        }
        s = MeetWall(map, to, length + 2 * cell, GhostRangeMeters, w, column);
        if (s > 0)
        {
            int row = RowOf(p.y * s / length);
            if (row >= 0 && row < heightRows) walls[w].explained.at<uchar>(row, column) = 255;
        }
    }

    // Which points are wall.
    std::vector<char> onWall(room.size(), 0);
    for (const Wall& wall : walls)
        if (wall.kept) for (int i : wall.members) onWall[i] = 1;
    // Each point's direction from the first camera, in degrees (for the shadows' readings).
    std::vector<cv::Point2f> directions;
    auto Direction = [](const cv::Point3f& p)
    {
        return cv::Point2f(std::atan2(p.x, -p.z) * 180 / (float)CV_PI, std::atan2(p.y, std::hypot(p.x, p.z)) * 180 / (float)CV_PI);
    };

    // 6. Fill each kept wall, unrolled, from the floor up to the cut; then its unexplained shadows.
    const cv::Matx33f toCamera = firstCamera.t();   // room -> the first camera (X right, Y down, Z ahead)
    const float wideTan = std::tan(HalfWideDegrees * (float)CV_PI / 180), tallTan = std::tan(HalfTallDegrees * (float)CV_PI / 180);
    for (size_t w = 0; w < walls.size(); w++)
    {
        Wall& wall = walls[w];
        if (!wall.kept)
        {
            if (wall.length >= MinWallLengthMeters)
                out.report.push_back(cv::format("  upright path %.1f m long: not a wall (%s)", wall.length,
                                                candidate[w] ? "seen only through another wall" : "too little of it seen"));
            continue;
        }
        out.walls++;
        for (int i : wall.members) claimed[i] = 1;
        cv::Mat seen, through, explained;
        cv::dilate(wall.seen, seen, cv::Mat());
        cv::dilate(wall.through, through, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(wall.explained, explained, cv::Mat(), cv::Point(-1, -1), 2);
        cv::Mat fill = ~seen & ~through;
        cv::Mat picture = PaintIn(wall.sum, wall.count);
        auto PointAt = [&](int row, int column)
        { return cv::Point3f(wall.at[column].x, floorY + (row + 0.5f) * cell, wall.at[column].y); };
        double area = 0;
        cv::Mat shadow(fill.size(), CV_8U, cv::Scalar(0));
        for (int r = 0; r < fill.rows; r++)
            for (int c = 0; c < fill.cols; c++)
            {
                if (!fill.at<uchar>(r, c)) continue;
                cv::Point3f p = PointAt(r, c);
                AddFilled(out, p, picture.at<cv::Vec3b>(r, c), r, c, area);
                // Step 1c: inside the first camera's view and range, and nothing seen in the way?
                cv::Point3f q = toCamera * p;
                if (!explained.at<uchar>(r, c) && q.z > 0.1f && std::abs(q.x) < wideTan * q.z && std::abs(q.y) < tallTan * q.z &&
                    cv::norm(p) < GhostRangeMeters) shadow.at<uchar>(r, c) = 255;
            }
        out.wallArea += area;

        // Each big unexplained shadow: "something dark stood here". Its distance from the camera:
        // the readings within ConeDegrees of it as the first camera saw them, that are neither wall
        // nor floor (a dark chair's legs and armrests, which often still read); with none, on the
        // wall itself (a TV, a black poster).
        cv::Mat shadowLabels, stats, centroids;
        int shadowCount = cv::connectedComponentsWithStats(shadow, shadowLabels, stats, centroids, 8);
        int shadows = 0;
        for (int label = 1; label < shadowCount; label++)
        {
            if (stats.at<int>(label, cv::CC_STAT_AREA) * cell * cell < GhostMinArea) continue;
            cv::Mat piece = shadowLabels == label;
            if (directions.empty()) for (const cv::Point3f& p : room) directions.push_back(Direction(p));
            // The shadow as the first camera saw it: a mask of directions, half a degree per cell, widened by ConeDegrees.
            const float bin = 0.5f;
            std::vector<cv::Point2f> pieceDirections;
            std::vector<float> wallDistances;
            for (int r = 0; r < piece.rows; r++)
                for (int c = 0; c < piece.cols; c++)
                    if (piece.at<uchar>(r, c))
                    {
                        cv::Point3f p = PointAt(r, c);
                        pieceDirections.push_back(Direction(p));
                        wallDistances.push_back(std::hypot(p.x, p.z));
                    }
            cv::Rect2f around = cv::boundingRect(pieceDirections);
            around.x -= ConeSideDegrees; around.y -= ConeDegrees; around.width += 2 * ConeSideDegrees; around.height += ConeDegrees;
            cv::Mat cone((int)(around.height / bin) + 2, (int)(around.width / bin) + 2, CV_8U, cv::Scalar(0));
            for (const cv::Point2f& d : pieceDirections) cone.at<uchar>((int)((d.y - around.y) / bin), (int)((d.x - around.x) / bin)) = 255;
            // Widen it a little sideways and a lot downwards (rows count up with the elevation).
            const int side = (int)(ConeSideDegrees / bin), down = (int)(ConeDegrees / bin);
            cv::dilate(cone, cone, cv::Mat::ones(down + 1, 2 * side + 1, CV_8U), cv::Point(side, 0));
            // Distances across the floor (seen from above): a chair's legs are lower than its
            // seat, but stand at the same distance.
            const float wallDistance = spotter::Percentile(wallDistances, 0.5);
            std::vector<float> shares;
            for (size_t i = 0; i < room.size(); i++)
            {
                const cv::Point2f& d = directions[i];
                if (onWall[i] || room[i].y < floorY + 0.03f || !around.contains(d)) continue;
                if (!cone.at<uchar>((int)((d.y - around.y) / bin), (int)((d.x - around.x) / bin))) continue;
                float distance = std::hypot(room[i].x, room[i].z);
                if (distance < wallDistance - 0.2f) shares.push_back(distance / wallDistance);
            }
            float share = shares.size() >= 3 ? spotter::Percentile(shares, 0.5) : 0.98f;   // 0.98: just in front of the wall's fill
            for (int r = 0; r < piece.rows; r++)
                for (int c = 0; c < piece.cols; c++)
                {
                    if (!piece.at<uchar>(r, c)) continue;
                    out.ghosts.push_back(PointAt(r, c) * share);   // along the line of sight from the camera at 0,0,0
                    out.ghostStripe.push_back((char)(((r + c) / StripeCells) % 2));
                    out.ghostPatches.push_back(cell * share);
                }
            shadows++;
        }
        out.shadows += shadows;

        // How bent it is: the farthest its path strays from the straight line between its ends.
        cv::Point2f a = wall.at.front(), b = wall.at.back(), along = b - a;
        float chord = (float)cv::norm(along), bend = 0;
        if (chord > 1e-3f)
            for (const cv::Point2f& p : wall.at) bend = std::max(bend, (float)std::abs((p - a).cross(along)) / chord);
        int seenColumns = (int)std::count(wall.columnSeen.begin(), wall.columnSeen.end(), 1);
        out.report.push_back(cv::format("  wall %d: %.1f m long, bends up to %.0f cm, %.0f%% of it seen, filled %.2f m^2, %d dark shadows",
                                        out.walls, wall.length, bend * 100, 100.0 * seenColumns / wall.columnSeen.size(), area, shadows));
    }
    return map;
}

// The floor (new, D33): the lowest level surface big enough to be a floor. It used to be the lowest
// 2% of the points, but a Kinect at chest height sees the floor at a grazing angle, so the floor is
// a small share of a scan. Readings under it (flying pixels, multipath in the corners, a slightly
// misplaced scan) dragged that mark below the real floor on the first real run. Nothing real and
// level lies under the floor, and stray readings never make one big connected piece. The walls'
// foot (the lowest height where upright readings crowd as densely as up a wall) keeps a table top
// or the ceiling from passing for an unseen floor, and is the floor's guess when none was seen.
// "report": how it chose, for the console.


//Filled FillHoles(const std::vector<cv::Point3f>& room, const std::vector<cv::Point3f>& normals, const std::vector<cv::Vec3b>& colours, float floorY, bool floorSeen, float cutY, const cv::Matx33f& firstCamera){
//
//@param1 - room: the room's points, levelled (Y up; the first camera at 0, 0, 0)
//@param2 - normals: which way each faces
//@param3 - colours: each point's colour
//@param4 - floorY: FindTwinFloor's floor height
//@param5 - floorSeen: true if the floor was seen, false = guessed at the walls' foot
//@param6 - cutY: fill up to this height
//@param7 - firstCamera: scan 1's camera -> the room
//
//@brief - steps a-e: every flat surface of the room and its holes filled, and step f's shadows. a. walls
//         (FillWalls); b. cabinet fronts and couch backs (upright planes found by RANSAC, refitted twice seen from
//         above); c. the floor (inside the walls) and level surfaces (table tops, seats, shelf tops, each height
//         fitted per tile); d. table legs (thin tall blobs under a top, made whole from the floor to the top); e.
//         a cell counts as seen if a real point is near it or just behind it, the rest is filled with colour from
//         the seen cells around (PaintIn). Returns the filled points, marking bands, shadows and a report line per
//         surface. O(points + surface cells).
Filled FillHoles(const std::vector<cv::Point3f>& room, const std::vector<cv::Point3f>& normals,
                 const std::vector<cv::Vec3b>& colours, float floorY, bool floorSeen, float cutY, const cv::Matx33f& firstCamera)
{
    Filled out;
    const float cell = FillCellMeters;
    const float levelCos = std::cos(FacingDegrees * (float)CV_PI / 180), uprightSin = std::sin(FacingDegrees * (float)CV_PI / 180);
    const float agreeCos = std::cos(2 * FacingDegrees * (float)CV_PI / 180);   // two normals on the same plane
    const int minPoints = (int)(MinSurfaceArea / (cell * cell));
    const cv::Point3f up(0, 1, 0);
    const cv::Mat none;

    // --- a. Walls ---
    std::vector<char> claimed(room.size(), 0);
    WallMap walls = FillWalls(room, normals, colours, floorY, cutY, firstCamera, claimed, out);

    // --- b. Cabinet fronts, couch backs: upright planes in what the walls left. RANSAC proposes a
    //        plane from one point's normal, then it's refitted to its points, seen from above. ---
    std::mt19937 random(1);
    std::vector<int> upright, level;
    for (int i = 0; i < (int)room.size(); i++)
    {
        if (std::abs(normals[i].y) < uprightSin && !claimed[i]) upright.push_back(i);
        if (std::abs(normals[i].y) > levelCos && room[i].y <= cutY) level.push_back(i);
    }
    std::vector<char> used(room.size(), 0);
    for (int plane = 0; plane < MaxUprightPlanes; plane++)
    {
        std::vector<int> pool;
        for (int i : upright) if (!used[i]) pool.push_back(i);
        if ((int)pool.size() < minPoints) break;
        const size_t step = std::max<size_t>(1, pool.size() / 20000);   // score on at most 20k points (speed)
        std::uniform_int_distribution<size_t> pick(0, pool.size() - 1);
        cv::Point3f normal;
        float offset = 0;
        int bestCount = 0;
        for (int t = 0; t < 300; t++)
        {
            int i = pool[pick(random)];
            cv::Point3f h(normals[i].x, 0, normals[i].z);
            float length = (float)cv::norm(h);
            if (length < 0.5f) continue;
            h *= 1 / length;
            float d = h.dot(room[i]);
            int count = 0;
            for (size_t k = 0; k < pool.size(); k += step)
            {
                int j = pool[k];
                if (std::abs(h.dot(room[j]) - d) < PlaneThicknessMeters && std::abs(h.dot(normals[j])) > agreeCos) count++;
            }
            if (count > bestCount) { bestCount = count; normal = h; offset = d; }
        }
        if (bestCount * (int)step < minPoints) break;
        auto OnPlane = [&](int j) { return std::abs(normal.dot(room[j]) - offset) < PlaneThicknessMeters && std::abs(normal.dot(normals[j])) > agreeCos; };
        // Refit twice: the best straight line through its points seen from above (PCA in 2D, as 13's FitPlane).
        for (int round = 0; round < 2; round++)
        {
            double n = 0, mx = 0, mz = 0, sxx = 0, sxz = 0, szz = 0;
            for (int j : pool) if (OnPlane(j)) { n++; mx += room[j].x; mz += room[j].z; }
            if (n < 3) break;
            mx /= n; mz /= n;
            for (int j : pool)
                if (OnPlane(j)) { double dx = room[j].x - mx, dz = room[j].z - mz; sxx += dx * dx; sxz += dx * dz; szz += dz * dz; }
            double angle = 0.5 * std::atan2(2 * sxz, sxx - szz);   // the line's direction
            cv::Point3f refit((float)-std::sin(angle), 0, (float)std::cos(angle));
            if (refit.dot(normal) < 0) refit = -refit;
            normal = refit;
            offset = (float)(normal.x * mx + normal.z * mz);
        }
        std::vector<cv::Point3f> members;
        std::vector<cv::Vec3b> memberColours;
        for (int j : pool)
        {
            if (OnPlane(j)) { members.push_back(room[j]); memberColours.push_back(colours[j]); }
            if (std::abs(normal.dot(room[j]) - offset) < SeenFrontMeters && std::abs(normal.dot(normals[j])) > agreeCos) used[j] = 1;
        }
        if ((int)members.size() < minPoints) continue;
        const cv::Point3f along(-normal.z, 0, normal.x);
        float minA = 1e9f, maxA = -1e9f, minY = 1e9f, maxY = -1e9f;
        for (const cv::Point3f& p : members)
        {
            minA = std::min(minA, along.dot(p)); maxA = std::max(maxA, along.dot(p));
            minY = std::min(minY, p.y); maxY = std::max(maxY, p.y);
        }
        Grid grid = MakeGrid(normal, along, up, offset, minA - 2 * cell, maxA + 2 * cell, minY - 2 * cell, maxY + 2 * cell);
        cv::Mat on(grid.rows, grid.columns, CV_8U, cv::Scalar(0));
        int row, column;
        for (const cv::Point3f& p : members) if (grid.Cell(p, row, column)) on.at<uchar>(row, column) = 255;
        for (const cv::Mat& domain : PieceDomains(on))
        {
            double area = FillSurface(grid, domain, members, memberColours, room, OtherBehindMeters, none, cutY, out);
            out.otherArea += area;
            out.others++;
            out.report.push_back(cv::format("  upright surface %.1f m wide, %.1f m tall: filled %.2f m^2", maxA - minA, maxY - minY, area));
        }
    }

    // --- c. Level surfaces: heights where many up-facing points crowd together ---
    const float lowest = floorY - 0.3f;
    std::vector<int> histogram((size_t)std::max(1.0f, std::ceil((cutY - lowest) / LevelStepMeters)) + 1, 0);
    for (int i : level)
    {
        int bin = (int)((room[i].y - lowest) / LevelStepMeters);
        if (bin >= 0 && bin < (int)histogram.size()) histogram[bin]++;
    }
    std::vector<std::pair<int, float>> peaks;   // (points, height)
    for (int bin = 0; bin < (int)histogram.size(); bin++)
    {
        int before = bin > 0 ? histogram[bin - 1] : 0, after = bin + 1 < (int)histogram.size() ? histogram[bin + 1] : 0;
        if (histogram[bin] < minPoints || histogram[bin] <= before || histogram[bin] < after) continue;
        double sum = 0;
        int count = 0;
        for (int i : level)
            if (std::abs(room[i].y - (lowest + (bin + 0.5f) * LevelStepMeters)) < 1.5f * LevelStepMeters) { sum += room[i].y; count++; }
        peaks.emplace_back(count, (float)(sum / count));
    }
    std::sort(peaks.rbegin(), peaks.rend());
    std::vector<float> heights;   // the levels, biggest first, none within 6 cm of a bigger one
    for (const auto& [count, height] : peaks)
    {
        bool near = false;
        for (float h : heights) near = near || std::abs(h - height) < 1.5f * LevelGatherMeters;
        if (!near) heights.push_back(height);
    }
    // The floor's height comes from FindTwinFloor (D33): as seen, or the walls' foot.
    out.floorSeen = floorSeen;
    const float groundY = floorY;

    // Inside the walls: the first camera's line of sight to a floor cell meets no kept wall first.
    auto Inside = [&](const cv::Point3f& p)
    {
        cv::Point2f to(p.x, p.z);
        int w, column;
        return MeetWall(walls, to, cell, (float)cv::norm(to) - 2 * cell, w, column) < 0;
    };
    // The floor, seen (its points' outline) or not (everything's outline at the room's foot).
    {
        std::vector<cv::Point3f> members;
        std::vector<cv::Vec3b> memberColours;
        const float y = groundY;
        for (int i = 0; i < (int)room.size(); i++)
        {
            bool onFloor = out.floorSeen ? std::abs(normals[i].y) > levelCos && std::abs(room[i].y - y) < LevelGatherMeters
                                         : room[i].y <= cutY;   // unseen: the outline of everything
            if (onFloor) { members.push_back(room[i]); memberColours.push_back(colours[i]); }
        }
        if (!members.empty())
        {
            const Plan& plan = walls.plan;
            float x0 = 1e9f, x1 = -1e9f, z0 = 1e9f, z1 = -1e9f;
            for (const cv::Point3f& p : members) { x0 = std::min(x0, p.x); x1 = std::max(x1, p.x); z0 = std::min(z0, p.z); z1 = std::max(z1, p.z); }
            if (plan.columns > 0)   // no wider than the room seen from above
            {
                x0 = std::max(x0, plan.x0); z0 = std::max(z0, plan.z0);
                x1 = std::min(x1, plan.x0 + plan.columns * cell); z1 = std::min(z1, plan.z0 + plan.rows * cell);
            }
            Grid grid = MakeGrid(up, cv::Point3f(1, 0, 0), cv::Point3f(0, 0, 1), y, x0, x1, z0, z1);
            std::vector<cv::Point> cells;
            int row, column;
            for (const cv::Point3f& p : members) if (grid.Cell(p, row, column)) cells.emplace_back(column, row);
            cv::Mat domain(grid.rows, grid.columns, CV_8U, cv::Scalar(0));
            if (cells.size() >= 3)
            {
                std::vector<cv::Point> hull;
                cv::convexHull(cells, hull);
                cv::fillConvexPoly(domain, hull, cv::Scalar(255));
            }
            for (int r = 0; r < grid.rows; r++)
                for (int c = 0; c < grid.columns; c++)
                    if (domain.at<uchar>(r, c) && !Inside(grid.Centre(r, c, y))) domain.at<uchar>(r, c) = 0;
            std::vector<cv::Point3f> seenMembers;   // only real floor readings give colour
            std::vector<cv::Vec3b> seenColours;
            if (out.floorSeen) { seenMembers = members; seenColours = memberColours; }
            out.floorArea = FillSurface(grid, domain, seenMembers, seenColours, room, OtherBehindMeters, none, cutY, out);
            out.report.push_back(out.floorSeen ? cv::format("  floor (seen) at %.2f m: filled %.2f m^2", y, out.floorArea)
                                               : cv::format("  floor never seen: guessed at the walls' foot (%.2f m), filled %.2f m^2", y, out.floorArea));
        }
    }

    // Table tops, seats, shelf tops: each level's pieces, its height fitted per tile.
    for (int k = 0; k < (int)heights.size(); k++)
    {
        if (std::abs(heights[k] - groundY) < 1.5f * LevelGatherMeters) continue;   // the floor itself
        const float y = heights[k];
        std::vector<cv::Point3f> members;
        std::vector<cv::Vec3b> memberColours;
        for (int i : level)
            if (std::abs(room[i].y - y) < LevelGatherMeters) { members.push_back(room[i]); memberColours.push_back(colours[i]); }
        if ((int)members.size() < minPoints) continue;
        float minX = 1e9f, maxX = -1e9f, minZ = 1e9f, maxZ = -1e9f;
        for (const cv::Point3f& p : members)
        {
            minX = std::min(minX, p.x); maxX = std::max(maxX, p.x);
            minZ = std::min(minZ, p.z); maxZ = std::max(maxZ, p.z);
        }
        const float margin = 2 * cell;
        Grid grid = MakeGrid(up, cv::Point3f(1, 0, 0), cv::Point3f(0, 0, 1), y, minX - margin, maxX + margin, minZ - margin, maxZ + margin);
        cv::Mat on(grid.rows, grid.columns, CV_8U, cv::Scalar(0));
        int row, column;
        for (const cv::Point3f& p : members) if (grid.Cell(p, row, column)) on.at<uchar>(row, column) = 255;
        // Its height per TileMeters tile (the median there), so a top tilted 1-2 degrees is followed.
        const int tile = std::max(1, (int)std::lround(TileMeters / cell));
        const int tileColumns = (grid.columns + tile - 1) / tile, tileRows = (grid.rows + tile - 1) / tile;
        std::vector<std::vector<float>> tileHeights(tileColumns * tileRows);
        for (const cv::Point3f& p : members)
            if (grid.Cell(p, row, column)) tileHeights[(row / tile) * tileColumns + column / tile].push_back(p.y);
        cv::Mat at(grid.rows, grid.columns, CV_32F, cv::Scalar(y));
        for (int r = 0; r < grid.rows; r++)
            for (int c = 0; c < grid.columns; c++)
            {
                const std::vector<float>& here = tileHeights[(r / tile) * tileColumns + c / tile];
                if (here.size() >= 5) at.at<float>(r, c) = spotter::Percentile(here, 0.5);
            }

        for (const cv::Mat& domain : PieceDomains(on, (int)std::lround(LevelJoinMeters / cell)))
        {
            double area = FillSurface(grid, domain, members, memberColours, room, OtherBehindMeters, at, cutY, out);
            out.otherArea += area;
            out.others++;
            cv::Rect box = cv::boundingRect(domain);

            // --- d. Legs: under a top, thin and tall blobs of readings, made whole from the floor up ---
            int legs = 0;
            const float gap = y - groundY;
            if (gap >= MinTableHeightMeters)
            {
                cv::Mat under;
                cv::dilate(domain, under, cv::Mat(), cv::Point(-1, -1), 2);   // the top's outline plus about 5 cm
                // Cells whose readings stack up like a leg's (a chair's flat seat under the table doesn't).
                std::vector<int> below;
                cv::Mat cellLow(grid.rows, grid.columns, CV_32F, cv::Scalar(1e9f)), cellHigh(grid.rows, grid.columns, CV_32F, cv::Scalar(-1e9f));
                for (int i = 0; i < (int)room.size(); i++)
                {
                    const cv::Point3f& p = room[i];
                    if (p.y <= groundY + 0.02f || p.y >= y - 0.03f || !grid.Cell(p, row, column) || !under.at<uchar>(row, column)) continue;
                    cellLow.at<float>(row, column) = std::min(cellLow.at<float>(row, column), p.y);
                    cellHigh.at<float>(row, column) = std::max(cellHigh.at<float>(row, column), p.y);
                    below.push_back(i);
                }
                cv::Mat legCells = (cellHigh - cellLow) >= std::max(0.06f, MinLegSpanShare * gap / 2);
                cv::Mat legLabels, legStats, legCentroids;
                int blobs = cv::connectedComponentsWithStats(legCells, legLabels, legStats, legCentroids, 8);
                std::vector<float> blobLow(blobs, 1e9f), blobHigh(blobs, -1e9f);
                std::vector<cv::Vec3f> blobColour(blobs, cv::Vec3f(0, 0, 0));
                std::vector<int> blobCount(blobs, 0);
                const int legRows = std::max(1, (int)((y - 0.03f - groundY) / cell));
                for (int i : below)
                {
                    grid.Cell(room[i], row, column);
                    int b = legLabels.at<int>(row, column);
                    blobLow[b] = std::min(blobLow[b], room[i].y);
                    blobHigh[b] = std::max(blobHigh[b], room[i].y);
                    blobColour[b] += cv::Vec3f(colours[i]);
                    blobCount[b]++;
                }
                const int maxCells = (int)std::ceil(MaxLegWidthMeters / cell);
                for (int b = 1; b < blobs; b++)
                {
                    if (legStats.at<int>(b, cv::CC_STAT_WIDTH) > maxCells || legStats.at<int>(b, cv::CC_STAT_HEIGHT) > maxCells) continue;
                    const float cross = groundY + LegCrossShare * gap;
                    if (blobHigh[b] - blobLow[b] < MinLegSpanShare * gap || blobLow[b] > cross || blobHigh[b] < cross) continue;
                    // A seat on it (up-facing readings over its footprint, from its top up to below the table's): a chair's leg.
                    int seat = 0;
                    for (int i : level)
                    {
                        const cv::Point3f& p = room[i];
                        if (p.y > blobHigh[b] - 0.05f && p.y < y - 0.1f && grid.Cell(p, row, column) && legLabels.at<int>(row, column) == b) seat++;
                    }
                    if (seat >= 3) continue;
                    // A post over its own footprint: only the heights with no reading in that cell are filled.
                    cv::Mat seen(legRows, grid.rows * grid.columns, CV_8U, cv::Scalar(0));
                    for (int i : below)
                    {
                        grid.Cell(room[i], row, column);
                        if (legLabels.at<int>(row, column) != b) continue;
                        int h = (int)((room[i].y - groundY) / cell);
                        for (int dh = -1; dh <= 1; dh++)
                            if (h + dh >= 0 && h + dh < legRows) seen.at<uchar>(h + dh, row * grid.columns + column) = 255;
                    }
                    cv::Vec3b colour(blobColour[b] / (float)std::max(1, blobCount[b]));
                    double legArea = 0;
                    for (int r = 0; r < grid.rows; r++)
                        for (int c = 0; c < grid.columns; c++)
                        {
                            if (legLabels.at<int>(r, c) != b) continue;
                            for (int h = 0; h < legRows; h++)
                            {
                                if (seen.at<uchar>(h, r * grid.columns + c)) continue;
                                cv::Point3f p = grid.Centre(r, c, y);
                                p.y = groundY + (h + 0.5f) * cell;
                                AddFilled(out, p, colour, h, c, legArea);
                            }
                        }
                    out.otherArea += legArea;
                    legs++;
                }
            }
            out.legs += legs;
            out.report.push_back(cv::format("  level surface %.2f m above the floor, %.1f x %.1f m: filled %.2f m^2, %d legs",
                                            y - groundY, box.width * cell, box.height * cell, area, legs));
        }
    }
    return out;
}



// ---------------------------------------------------------------------
//  Step 1: load the room
// ---------------------------------------------------------------------

struct Room
{
    std::filesystem::path folder;
    spotter::TwinRoom twin;            // every included scan merged, levelled (Y up)
    spotter::TwinFloor floor;
    std::vector<cv::Point3f> points, normals;   // what's above the under-floor noise
    std::vector<cv::Vec3b> colours;
    cv::Point3f middle;                // for the view
    float radius = 3;
    Filled filled;                     // step 2
    std::vector<cv::Vec3b> marked;     // the guess tinted blue in stripes
};

//bool LoadRoom(int argc, char** argv, Room& room){
//
//@param1 - argc: how many command-line words there are
//@param2 - argv: the words: an optional room folder
//@param3 - room: filled with the points, floor and size
//
//@brief - step 1: the room folder given, or one chosen in an Open window (ChooseRoom); LoadTwinRoom and FindTwinFloor; keeps
//         the points above the under-floor noise and works out the middle and the radius (90th percentile reach).
//         False (with a message) if there is no room or it won't load. O(points).
bool LoadRoom(int argc, char** argv, Room& room)
{
    room.folder = spotter::ChooseRoom(argc, argv, RoomsFolder, "13x");
    std::string problem;
    if (room.folder.empty())
    {
        std::printf("No room to fill. Scan one with 13e_Room_Scan, or name its folder: 13x_Room_Fill <folder>\n");
        return false;
    }
    if (!spotter::LoadTwinRoom(room.folder, CubeMeters, room.twin, problem)) { std::printf("Can't load %s: %s\n", room.folder.string().c_str(), problem.c_str()); return false; }
    room.floor = spotter::FindTwinFloor(room.twin);
    for (const std::string& line : room.floor.report) std::printf("%s\n", line.c_str());
    std::vector<float> reach;
    for (size_t i = 0; i < room.twin.points.size(); i++)
    {
        const cv::Point3f& p = room.twin.points[i];
        if (p.y < room.floor.y - UnderFloorMeters) continue;
        room.points.push_back(p);
        room.normals.push_back(room.twin.normals[i]);
        room.colours.push_back(room.twin.colours[i]);
        room.middle += p;
    }
    room.middle *= 1.0f / std::max<size_t>(room.points.size(), 1);
    for (const cv::Point3f& p : room.points) reach.push_back(std::hypot(p.x - room.middle.x, p.z - room.middle.z));
    if (!reach.empty()) room.radius = spotter::Percentile(reach, 0.9);
    room.middle.y = room.floor.y + CutAboveFloorMeters / 2;
    std::printf("The room: %s, %zu points, about %.1f m across\n", room.twin.from.c_str(), room.points.size(), 2 * room.radius);
    return true;
}

// ---------------------------------------------------------------------
//  Step 2 (new): fill, once
// ---------------------------------------------------------------------

//void FillRoom(Room& room){
//
//@param1 - room: the loaded room; its fill is added
//
//@brief - step 2 (new), once: FillHoles on the room's points up to the cut, prints the filled areas, counts and
//         time, and tints each filled cell with the marking colour (stronger on one band) for the marked view.
//         O(FillHoles).
void FillRoom(Room& room)
{
    auto start = std::chrono::steady_clock::now();
    Filled& filled = room.filled = FillHoles(room.points, room.normals, room.colours, room.floor.y, room.floor.seen,
                                             room.floor.y + CutAboveFloorMeters, room.twin.firstCamera);
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

//void Gather(const Room& room, int fillMode, float topY, std::vector<cv::Point3f>& points, std::vector<cv::Vec3b>& colours){
//
//@param1 - room: the loaded room with its fill
//@param2 - fillMode: 0 = marked, 1 = unmarked, 2 = off
//@param3 - topY: leave out anything above this height
//@param4 - points: set to what to show or save
//@param5 - colours: their colours
//
//@brief - every real point, then the fill (marked or plain colours, or none) and in the marked view the purple
//         shadow cut-outs, all up to topY. O(points).
void Gather(const Room& room, int fillMode, float topY, std::vector<cv::Point3f>& points, std::vector<cv::Vec3b>& colours)
{
    auto add = [&](const std::vector<cv::Point3f>& from, const std::vector<cv::Vec3b>& colour, size_t i) {
        if (from[i].y > topY) return;
        points.push_back(from[i]);
        colours.push_back(colour[i]);
    };
    for (size_t i = 0; i < room.points.size(); i++) add(room.points, room.colours, i);
    if (fillMode < 2)
        for (size_t i = 0; i < room.filled.points.size(); i++) add(room.filled.points, fillMode == 0 ? room.marked : room.filled.colours, i);
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

//void Show(const Room& room, int fillMode, spotter::OrbitView& view, spotter::PointRenderer& renderer){
//
//@param1 - room: the loaded room with its fill
//@param2 - fillMode: 0 = marked, 1 = unmarked, 2 = off
//@param3 - view: the viewer
//@param4 - renderer: draws the points
//
//@brief - the room up to the cut as a doll's house (Gather, turned with ForView) and a note on what the colours
//         mean; shown with the key strip. O(points).
void Show(const Room& room, int fillMode, spotter::OrbitView& view, spotter::PointRenderer& renderer)
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colours;
    Gather(room, fillMode, room.floor.y + CutAboveFloorMeters, points, colours);
    for (cv::Point3f& p : points) p = ForView(p);
    cv::Mat& image = renderer.Begin(view, cv::Scalar(20, 20, 20));
    renderer.Draw(points, colours, 3);   // 3 cm apart: about 2-3 px from a few metres
    const char* note[3] = { "blue stripes = filled in (never seen); purple = something dark, no reading", "filled in, unmarked",
                            "only what the Kinect saw" };
    cv::putText(image, note[fillMode], cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                fillMode == 0 ? cv::Scalar(MarkColour) : cv::Scalar(200, 200, 200), 1);
    cv::imshow(Window, spotter::WithKeys(image, Keys));
}

//void Save(const Room& room, int fillMode){
//
//@param1 - room: the loaded room with its fill
//@param2 - fillMode: 0 = marked, 1 = unmarked, 2 = off
//
//@brief - every real point (no cut) plus the fill as shown, written to room_filled_<date>_<time>.ply in the room's
//         folder; prints the point count or the failure. O(points).
void Save(const Room& room, int fillMode)
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colours;
    Gather(room, fillMode, INFINITY, points, colours);
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
//@brief - the program: 13x · Room Fill (no Kinect; after 13e and 13z). A scan only has what the Kinect saw: the
//         wall behind the couch, the floor under the table, the part of a table top under a box were never seen.
//         Setup, main loop and clean-up are laid out in the file's sections. Returns 0 on a normal exit, 1 if it
//         couldn't start.
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
    while (true)
    {
        Show(room, fillMode, view, renderer);                                   // step 3
        int key = cv::waitKey(15);
        if (spotter::IsQuitKey(key) || cv::getWindowProperty(Window, cv::WND_PROP_VISIBLE) < 1) break;
        if (key == 'g') fillMode = (fillMode + 1) % 3;
        if (key == 'k') Save(room, fillMode);                                   // step 4
        if (view.HandleKey(key) && key == 'r') view.pitchDegrees = ViewPitchDegrees;
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Press G three times: marked, unmarked, off. Off is the scan as it was: count its holes.
//  2. Read the console's list of surfaces: does every real wall show up as "wall", and how much
//     does each bend? A curved wall should say so.
//  3. A doorway nothing was seen through (a closed door, or a dark hallway) gets filled shut: the
//     filling can't tell a door from a hole. Open the door and scan through it.
//  4. Set WallBehindMeters = 0.05: where the drawn wall stands a little in front of the real one,
//     the fill covers what was seen. That's why it's 0.5.
//  5. Set WallStackMeters = 0.5: chair backs and table edges start counting as walls.
//  6. Set CutAboveFloorMeters = 2.4: the walls are filled to the ceiling. Is the view still readable?
//  7. Exclude a scan in 13z (X), run 13x again: which holes appear, and does the fill cover them?
