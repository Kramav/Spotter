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
    std::string upFrom;               // "given", "floor", "ceiling" or "camera"
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

// The planes of one depth cloud, sorted into floor, ceiling, walls and others. `up` = the room's up
// in this cloud's coordinates if known (e.g. from a placed scan), or (0,0,0) to find it: the biggest
// level-ish plane, else the camera's -Y.
ScanStructure ExtractStructure(const std::vector<cv::Point3f>& cloud, const PinholeCamera& camera, const cv::Point3f& up,
                               const StructureOptions& options, std::mt19937& random);

// The same structure in other coordinates (a scan's planes moved into the room by its pose).
ScanStructure MoveStructure(const ScanStructure& structure, const Pose& pose);

// Many scans' structures (in the room's coordinates) merged: the same plane seen twice (within
// mergeDegrees and mergeMeters) becomes one, refitted to all its points.
ScanStructure MergeStructures(const std::vector<ScanStructure>& structures, float mergeDegrees = 10, float mergeMeters = 0.05f);

// Where two planes meet (more than minDegrees apart): a line, and the stretch of it both planes'
// points reach (within nearMeters). Floor-wall, ceiling-wall and wall-wall edges.
struct Edge
{
    int planes[2] = { -1, -1 };       // indices into the structure's planes
    cv::Point3f point, direction;     // the line: point + t direction (unit)
    cv::Point3f ends[2];              // the stretch both planes reach along it
    int support = 0;                  // points near the line, from both planes
    std::string Kind(const ScanStructure& s) const;
};
std::vector<Edge> FindEdges(const ScanStructure& structure, float minDegrees = 60, float nearMeters = 0.05f);

// How far apart two edges are: the angle between them, and the gap between the lines at the
// middle of the first one's stretch.
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
RoomMap MakeRoomMap(const std::vector<const Scan*>& scans, const PinholeCamera& camera, const StructureOptions& options, std::mt19937& random,
                    const cv::Point3f& roomUp);

// Candidates from the structure (floor/ceiling sets up and height, wall pairs set yaw and position),
// polished by ICP, ranked by ViewAgreement: ICP's share alone can't tell one corner from another,
// since every scan point lies on SOME map wall; seeing through a map wall can.
RelocateResult Relocate(const Scan& scan, const ScanStructure& structure, const RoomMap& map, const PinholeCamera& camera,
                        const RelocateOptions& options);

// The map's points seen from `pose` (a depth image, every 4th pixel, nearest wins) against the scan's
// depth: + where they agree (within 5 cm), - where the scan sees more than 10 cm beyond a map surface
// (impossible at the right pose). Unmapped pixels don't count; nearer things (furniture) don't count.
float ViewAgreement(const Scan& scan, const Pose& pose, const RoomMap& map, const PinholeCamera& camera);

// Disambiguation, for the candidates of an ambiguous result:
// (i) colour: how many of the scan's near features match a map scan's AND land within 10 cm of it under the pose;
int ColourVotes(const Scan& scan, const Pose& pose, const std::vector<const Scan*>& mapScans, float ratio = 0.8f);
// (ii) furniture: the share of the scan's points off its structure planes that land on the map (within 3 cm).
float OffPlaneShare(const Scan& scan, const ScanStructure& structure, const Pose& pose, const RoomMap& map);

// --- Continuous: the next scan from a first guess, against the previous scan only ---
// ICP of `scan` against `previous` (placed), starting from `guess` (the previous pose, or the
// recommended next shot). No features, no structure.
IcpResult PlaceAfter(const Scan& scan, const Scan& previous, const Pose& guess, const IcpOptions& options = IcpOptions{ 50, 0.5f, 0.02f, 0.05f });

}
