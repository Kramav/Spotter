// spotter/structure.hpp: a room's structure (floor, ceiling, walls as planes, and the edges where
// they meet), for localising a scan without colour and for measuring what each scan sees of it.
// A prototype for the structure-localisation study (docs/structure-localization-findings.md).
// Points are in a camera's coordinates (metres; X right, Y down, Z ahead) or the room's.
#pragma once

#include "spotter/frame.hpp"
#include "spotter/geometry.hpp"
#include "spotter/register.hpp"
#include "spotter/room_scan.hpp"

#include <random>
#include <string>
#include <vector>

namespace spotter
{

enum class Surface { Floor, Ceiling, Wall, Other };

//const char* SurfaceName(Surface surface){
//
//@param1 - surface: floor, ceiling, wall or other
//
//@brief - its name for printing: "floor", "ceiling", "wall" or "other". O(1).
const char* SurfaceName(Surface surface);

struct StructurePlane
{
    Plane plane;                      // normal towards the camera (for walls seen from inside: into the room)
    Surface surface = Surface::Other;
    int cells = 0;                    // sampled grid cells on it (every step-th pixel each way)
    float areaMeters2 = 0;            // the area those cells cover on the plane
    float pointsPerMeter2 = 0;        // depth readings per square metre on it (all pixels)
    std::vector<cv::Point3f> points;  // its sampled points
};

struct ScanStructure
{
    std::vector<StructurePlane> planes;
    cv::Point3f up{ 0, -1, 0 };       // the room's up in these coordinates
    std::string upFrom;               // "given", "floor", "ceiling", "two walls", "one wall" or "camera"
    int readings = 0;                 // sampled cells with a reading
};

struct StructureOptions
{
    PlaneSearch planes;               // any facing, up to 8 planes (06b / 09c)
    int step = 4;                     // planes are searched on every 4th pixel each way
    float maxTiltDegrees = 45;        // without a given up: a level plane may be this far from the camera's level (a hand-held Kinect tilts)
    float levelDegrees = 10;          // floor, ceiling and walls: this close to level / upright once up is known
    float minWallHeightMeters = 0.5f; // a wall's points must span this much height (less than BuildRoom's 0.8: one view sees part of a wall)
};

//ScanStructure ExtractStructure(const std::vector<cv::Point3f>& cloud, const PinholeCamera& camera, const cv::Point3f& up, const StructureOptions& options, std::mt19937& random){
//
//@param1 - cloud: a depth cloud, one point per pixel (NaN = none)
//@param2 - camera: the depth camera, for the grid width and each cell's area
//@param3 - up: the room's up in this cloud's coordinates if known (e.g. from a placed scan), or (0, 0, 0)
//          to find it
//@param4 - options: the plane search, grid step and the level / upright / wall height limits
//@param5 - random: the random number generator for RANSAC
//
//@brief - the cloud's planes (FindPlanes on every step-th pixel), each with its area (each cell's patch at
//         its distance, stretched by the slant) and density, sorted into floor, ceiling, walls and others.
//         Without a given up: the biggest level-ish plane gives it, else two walls at a clear angle (their
//         cross product), else one wall (the camera's up laid into it), else the camera's -Y.
//         FindPlanes dominates: O(maxPlanes x tries x n) for n sampled cells.
ScanStructure ExtractStructure(const std::vector<cv::Point3f>& cloud, const PinholeCamera& camera, const cv::Point3f& up,
                               const StructureOptions& options, std::mt19937& random);

//ScanStructure MoveStructure(const ScanStructure& structure, const Pose& pose){
//
//@param1 - structure: planes in one camera's coordinates
//@param2 - pose: that camera -> the new coordinates (e.g. a scan's pose into the room)
//
//@brief - the same structure in the new coordinates: normals turned, distances and points moved. O(points).
ScanStructure MoveStructure(const ScanStructure& structure, const Pose& pose);

//ScanStructure MergeStructures(const std::vector<ScanStructure>& structures, float mergeDegrees, float mergeMeters){
//
//@param1 - structures: many scans' structures, all in the room's coordinates
//@param2 - mergeDegrees: two planes of the same kind this close in angle... (default 10)
//@param3 - mergeMeters: ...with one's centre this close to the other are the same plane (default 0.05)
//
//@brief - the planes merged: each plane joins the first matching merged plane (points pooled, areas summed,
//         refitted with FitPlane) or starts a new one. O(P x M x k) for P planes in, M merged planes and k
//         points per refit.
ScanStructure MergeStructures(const std::vector<ScanStructure>& structures, float mergeDegrees = 10, float mergeMeters = 0.05f);

// Where two planes meet: a line, and the stretch of it both planes' points reach.
// Floor-wall, ceiling-wall and wall-wall edges.
struct Edge
{
    int planes[2] = { -1, -1 };       // indices into the structure's planes
    cv::Point3f point, direction;     // the line: point + t direction (unit)
    cv::Point3f ends[2];              // the stretch both planes reach along it
    int support = 0;                  // points near the line, from both planes

    //std::string Kind(const ScanStructure& s){
    //
    //@param1 - s: the structure the edge's plane indices refer to
    //
    //@brief - the two planes' kinds in alphabetical order, e.g. "floor-wall" or "wall-wall". O(1).
    std::string Kind(const ScanStructure& s) const;
};

//std::vector<Edge> FindEdges(const ScanStructure& structure, float minDegrees, float nearMeters){
//
//@param1 - structure: the planes
//@param2 - minDegrees: only planes more than this far apart in angle meet in an edge (default 60)
//@param3 - nearMeters: a plane's point this close to the line supports it (default 0.05)
//
//@brief - for each pair of planes at a clear angle: the line where they meet (direction n1 x n2, through
//         the point on both), and the stretch between the 2nd and 98th percentiles of the supporting
//         points along it. Pairs with fewer than 10 supporting points are dropped (they never come near
//         each other in view). O(P^2 x k) for P planes of k points.
std::vector<Edge> FindEdges(const ScanStructure& structure, float minDegrees = 60, float nearMeters = 0.05f);

//void EdgeOffset(const Edge& a, const Edge& b, float& degrees, float& meters){
//
//@param1 - a: one edge
//@param2 - b: the other edge, same coordinates
//@param3 - degrees: set to the angle between their lines
//@param4 - meters: set to the gap from the middle of a's stretch to b's line
//
//@brief - how far apart two edges are (e.g. the same edge seen by two placed scans). O(1).
void EdgeOffset(const Edge& a, const Edge& b, float& degrees, float& meters);

// --- Relocate: a scan's pose in a known room from structure alone ---

struct RelocateOptions
{
    float wallMatchDegrees = 10;      // a scan wall pair and a map wall pair must be this close in angle
    float alongStepMeters = 0.25f;    // with one wall only, slide along it in these steps
    IcpOptions refine{ 30, 0.20f, 0.02f, 0.05f };   // each candidate polished by ICP from 20 cm
    float ambiguousScore = 0.05f;     // the two best distinct fits within this agreement score: ambiguous
    int refineAtMost = 24;            // candidates polished by ICP (the best by a quick score)
    float distinctMeters = 0.3f, distinctDegrees = 10;
};

struct RelocateCandidate
{
    Pose pose;                        // the scan's camera -> the room
    float share = 0, rmsMeters = 0;   // after ICP: the share of the scan's points paired, their gap
    float agreement = 0;              // the map seen from this pose vs the scan's depth: agreeing minus see-through pixels, as a share
    bool underConstrained = false;    // from one wall only (slid along it)
    int colourVotes = -1;             // disambiguation (i): agreeing colour feature matches (-1 = not tried)
    float offPlaneShare = -1;         // disambiguation (ii): the share of off-plane points that fit (-1 = not tried)
};

struct RelocateResult
{
    std::vector<RelocateCandidate> candidates;   // distinct fits, best agreement first (up to 4)
    bool ambiguous = false;           // the two best distinct fits score alike
    int generated = 0;                // candidates before ICP
    std::string note;                 // why it couldn't (no floor, no wall...)
};

// The map: the room's merged structure, plus its points and normals (one per 1 cm cube) for ICP.
struct RoomMap
{
    ScanStructure structure;
    SurfacePoints surface;
};

//RoomMap MakeRoomMap(const std::vector<const Scan*>& scans, const PinholeCamera& camera, const StructureOptions& options, std::mt19937& random, const cv::Point3f& roomUp){
//
//@param1 - scans: the placed scans that make the map
//@param2 - camera: the depth camera
//@param3 - options: ExtractStructure's settings
//@param4 - random: the random number generator for RANSAC
//@param5 - roomUp: the room's up in the room's coordinates
//
//@brief - each scan's structure (ExtractStructure with the room's up turned into its camera) moved into the
//         room and merged (MergeStructures), plus every scan's ICP points and normals in the room, one per
//         1 cm cube. O(scans x (maxPlanes x tries x n)) for the plane searches.
RoomMap MakeRoomMap(const std::vector<const Scan*>& scans, const PinholeCamera& camera, const StructureOptions& options, std::mt19937& random,
                    const cv::Point3f& roomUp);

//RelocateResult Relocate(const Scan& scan, const ScanStructure& structure, const RoomMap& map, const PinholeCamera& camera, const RelocateOptions& options){
//
//@param1 - scan: the scan to place
//@param2 - structure: its own structure (ExtractStructure), in its camera's coordinates
//@param3 - map: the room's map (MakeRoomMap)
//@param4 - camera: the depth camera, for ViewAgreement
//@param5 - options: the wall matching, the slide step, ICP, and the ambiguity thresholds
//
//@brief - a scan's pose from structure alone: the floor (or ceiling) turns the scan's up onto the room's
//         and sets the height; each scan wall pair x each map wall pair close in angle gives the yaw and
//         position in closed form; with one wall direction only, the scan slides along the map wall in
//         alongStepMeters steps (under-constrained). A quick score keeps the best refineAtMost, ICP polishes
//         them, and they're ranked by ViewAgreement: ICP's share alone can't tell one corner from another,
//         since every scan point lies on SOME map wall; seeing through a map wall can. Ambiguous when the
//         two best distinct fits score within ambiguousScore. O(W^2 x V^2) candidates for W scan walls and V
//         map walls, then O(refineAtMost x rounds x n log m) for the ICPs.
RelocateResult Relocate(const Scan& scan, const ScanStructure& structure, const RoomMap& map, const PinholeCamera& camera,
                        const RelocateOptions& options);

//float ViewAgreement(const Scan& scan, const Pose& pose, const RoomMap& map, const PinholeCamera& camera){
//
//@param1 - scan: the scan, with its depth cloud
//@param2 - pose: a candidate pose of the scan's camera in the room
//@param3 - map: the room's map
//@param4 - camera: the depth camera
//
//@brief - the map's points seen from `pose` (a depth image on every 4th pixel, nearest wins) against the
//         scan's depth: + where they agree (within 5 cm), - where the scan sees more than 10 cm beyond a map
//         surface (impossible at the right pose). Unmapped pixels and nearer things (furniture) don't
//         count. Returns (agree - see-through) / pixels compared, or -1 if none. O(m + pixels / 16).
float ViewAgreement(const Scan& scan, const Pose& pose, const RoomMap& map, const PinholeCamera& camera);

// Disambiguation, for the candidates of an ambiguous result:

//int ColourVotes(const Scan& scan, const Pose& pose, const std::vector<const Scan*>& mapScans, float ratio){
//
//@param1 - scan: the scan being placed
//@param2 - pose: a candidate pose for it
//@param3 - mapScans: the map's placed scans, with their near features
//@param4 - ratio: Lowe's ratio for MatchFeatures (default 0.8)
//
//@brief - (i) colour: how many of the scan's near features match a map scan's AND land within 10 cm of it
//         under the pose. O(scans x n x m) for the brute-force matching.
int ColourVotes(const Scan& scan, const Pose& pose, const std::vector<const Scan*>& mapScans, float ratio = 0.8f);

//float OffPlaneShare(const Scan& scan, const ScanStructure& structure, const Pose& pose, const RoomMap& map){
//
//@param1 - scan: the scan being placed
//@param2 - structure: its own structure, to tell which points are on the floor, ceiling or walls
//@param3 - pose: a candidate pose for it
//@param4 - map: the room's map
//
//@brief - (ii) furniture: the share of the scan's ICP points off its floor, ceiling and walls (more than
//         5 cm) that land within 3 cm of a map point under the pose (a FLANN k-d tree). 0 if there are none.
//         O(n x P + m log m + n log m) for n scan points, P planes, m map points.
float OffPlaneShare(const Scan& scan, const ScanStructure& structure, const Pose& pose, const RoomMap& map);

// --- Continuous: the next scan from a first guess, against the previous scan only ---

//IcpResult PlaceAfter(const Scan& scan, const Scan& previous, const Pose& guess, const IcpOptions& options){
//
//@param1 - scan: the new scan
//@param2 - previous: the previous scan, already placed
//@param3 - guess: the first guess of the new scan's pose (the previous pose, or the recommended next shot)
//@param4 - options: ICP's settings (default: 50 rounds, from 50 cm down to 2 cm, weak 5%)
//
//@brief - ICP of the scan's points against the previous scan's, moved into the room. No features, no
//         structure. O(m log m + rounds x n log m).
IcpResult PlaceAfter(const Scan& scan, const Scan& previous, const Pose& guess, const IcpOptions& options = IcpOptions{ 50, 0.5f, 0.02f, 0.05f });

}
