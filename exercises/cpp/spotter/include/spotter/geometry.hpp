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

    //cv::Point3f operator()(const cv::Point3f& p){
    //
    //@param1 - p: a point in this camera's coordinates
    //
    //@brief - the same point in the other's coordinates: turn * p + shift. O(1).
    cv::Point3f operator()(const cv::Point3f& p) const { return turn * p + shift; }

    //Pose Then(const Pose& next){
    //
    //@param1 - next: the pose to apply after this one
    //
    //@brief - this pose followed by `next`, as one pose: turn = next.turn * turn,
    //         shift = next.turn * shift + next.shift. O(1).
    Pose Then(const Pose& next) const { return { next.turn * turn, next.turn * shift + next.shift }; }

    //float Degrees(){
    //
    //@brief - how far it turns, whatever the axis: acos((trace - 1) / 2), in degrees. O(1).
    float Degrees() const;
};

//Pose Kabsch(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to){
//
//@param1 - from: the points to move; at least 3, not all in a line
//@param2 - to: each from point's partner, same length and order
//
//@brief - the turn + shift that best moves each `from` point onto its partner (least squares; the Kabsch
//         method: centre both, the SVD of how they vary together gives the turn; a mirror image is
//         made a turn by flipping the weakest axis). O(n) for n pairs (the SVD is of a fixed 3 x 3 matrix).
Pose Kabsch(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to);

//cv::Matx33f TurnAbout(const cv::Vec3d& w){
//
//@param1 - w: the axis to turn about, its length = the angle in radians
//
//@brief - the rotation matrix of a turn of |w| radians about w (Rodrigues' formula). O(1).
cv::Matx33f TurnAbout(const cv::Vec3d& w);

// A plane: normal . point + distance = 0, with the normal pointing to the camera's side, so
// distance >= 0 is how far the camera is from the plane, and DistanceTo is + on the camera's side
// (for a floor: the height above it).
struct Plane
{
    cv::Point3f normal;
    float distance = 0;

    //float DistanceTo(const cv::Point3f& point){
    //
    //@param1 - point: any point, same coordinates as the plane
    //
    //@brief - the signed distance from the plane: normal . point + distance; + on the camera's side. O(1).
    float DistanceTo(const cv::Point3f& point) const { return normal.dot(point) + distance; }
};

//bool PlaneThrough(const cv::Point3f& a, const cv::Point3f& b, const cv::Point3f& c, Plane& plane){
//
//@param1 - a: a point on the plane
//@param2 - b: a second point
//@param3 - c: a third point
//@param4 - plane: filled with the plane through all three, normal towards the camera
//
//@brief - the plane through three points, from the cross product of two edges. False if they're (nearly)
//         in a line. O(1).
bool PlaneThrough(const cv::Point3f& a, const cv::Point3f& b, const cv::Point3f& c, Plane& plane);

//Plane FitPlane(const std::vector<cv::Point3f>& points){
//
//@param1 - points: three or more points on (roughly) one plane
//
//@brief - the least-squares plane through them: principal component analysis, and the direction they
//         spread least in is the normal (turned towards the camera). O(n).
Plane FitPlane(const std::vector<cv::Point3f>& points);

// A thinned depth cloud: every step-th pixel each way, still laid out as a grid (NaN = no reading),
// so "neighbours in the picture" stays easy to find.
struct PointGrid
{
    std::vector<cv::Point3f> points;
    int columns = 0, rows = 0;
};

//PointGrid ThinCloud(const std::vector<cv::Point3f>& cloud, int width, int step){
//
//@param1 - cloud: a depth cloud, one point per pixel row by row (NaN = no reading)
//@param2 - width: the depth picture's width (points per row)
//@param3 - step: keep every step-th pixel each way
//
//@brief - copies every step-th pixel each way into a smaller grid. O(n / step^2).
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

//std::vector<PlanePiece> FindPlanes(const PointGrid& grid, const PlaneSearch& search, std::mt19937& random){
//
//@param1 - grid: a thinned depth cloud (ThinCloud)
//@param2 - search: which way the planes may face, how many, the RANSAC tries, thickness and minimum share
//@param3 - random: the random number generator (seed it for repeatable results)
//
//@brief - repeated RANSAC: pick three free cells near each other in the grid (neighbours usually share a
//         surface), take the plane through them, count the free points on it, keep the best of `tries`;
//         then keep that plane's biggest connected piece in the picture, refit it, and take its cells away.
//         Repeat up to maxPlanes times, stopping when the best plane has under minShare of the readings.
//         A plane too scattered to be one surface has its points taken away anyway (bug 8: it was found
//         again and again). O(maxPlanes x tries x n) for n grid cells.
std::vector<PlanePiece> FindPlanes(const PointGrid& grid, const PlaneSearch& search, std::mt19937& random);

}
