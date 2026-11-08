// spotter/geometry.hpp: poses and flat surfaces in 3D (examples 06b, 09, 09c, 11).
// Points are in the camera's coordinates: metres, X right, Y down, Z ahead.
#pragma once

#include <opencv2/core.hpp>
#include <random>
#include <vector>

namespace spotter
{

// Where a camera stood and which way it pointed, relative to another (example 11): a point it
// sees maps into the other's coordinates as  turn * point + shift.
struct Pose
{
    cv::Matx33f turn = cv::Matx33f::eye();
    cv::Point3f shift = cv::Point3f(0, 0, 0);

    cv::Point3f operator()(const cv::Point3f& p) const { return turn * p + shift; }
    Pose Then(const Pose& next) const { return { next.turn * turn, next.turn * shift + next.shift }; }   // this, then next
    float Degrees() const;   // how far it turns, whatever the axis
};

// The turn + shift that best moves each `from` point onto its partner in `to` (least squares; the
// Kabsch method: centre both, the SVD of how they vary together gives the turn). Needs 3+ pairs.
Pose Kabsch(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to);

// A turn of |w| radians about the axis w points along (Rodrigues' formula).
cv::Matx33f TurnAbout(const cv::Vec3d& w);

// A plane: normal . point + distance = 0, with the normal pointing to the camera's side, so
// distance >= 0 is how far the camera is from the plane, and DistanceTo is + on the camera's side
// (for a floor: the height above it).
struct Plane
{
    cv::Point3f normal;
    float distance = 0;
    float DistanceTo(const cv::Point3f& point) const { return normal.dot(point) + distance; }
};

// The plane through three points; false if they're (nearly) in a line.
bool PlaneThrough(const cv::Point3f& a, const cv::Point3f& b, const cv::Point3f& c, Plane& plane);

// The least-squares plane through many points: the direction they spread least in is the normal.
Plane FitPlane(const std::vector<cv::Point3f>& points);

// A thinned depth cloud: every step-th pixel each way, still laid out as a grid (NaN = no reading),
// so "neighbours in the picture" stays easy to find.
struct PointGrid
{
    std::vector<cv::Point3f> points;
    int columns = 0, rows = 0;
};
PointGrid ThinCloud(const std::vector<cv::Point3f>& cloud, int width, int step);

// Repeated RANSAC (example 06b): the biggest plane, as its biggest connected piece, taken away,
// then the next biggest, and so on.
struct PlaneSearch
{
    enum class Facing { Any, Upright, Level };
    Facing facing = Facing::Any;
    float maxTiltDegrees = 25;       // how far from upright / level still counts, as the camera sees it
    int maxPlanes = 8;
    int tries = 200;                 // random planes per search
    float thicknessMeters = 0.03f;   // a point within this of the plane is on it
    int neighbourCells = 10;         // RANSAC's three points lie this close together in the grid
    double minShare = 0.03;          // a plane (and its piece) needs this share of the readings
};

struct PlanePiece
{
    Plane plane;              // refitted to the piece's points
    std::vector<int> cells;   // grid cells of its biggest connected piece
};

std::vector<PlanePiece> FindPlanes(const PointGrid& grid, const PlaneSearch& search, std::mt19937& random);

}
