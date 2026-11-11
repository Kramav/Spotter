// spotter/mesh.hpp: triangle meshes and point clouds as files (OBJ, PLY), for Blender, MeshLab
// and CloudCompare. A mesh or cloud is a capture of the room: *.obj and *.ply are gitignored.
#pragma once

#include <opencv2/core.hpp>
#include <map>
#include <string>
#include <vector>

namespace spotter
{

// Triangles sorted into named groups (OBJ "g" lines), so a viewer can hide one: e.g. what was seen
// (seen_room) apart from what was guessed (clean_walls, guessed_back).
struct Mesh
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colours;                          // BGR, one per point
    std::map<std::string, std::vector<cv::Vec3i>> groups;    // group name -> three point indices per triangle
};

//void MeshFromGrid(const std::vector<cv::Point3f>& cloud, const std::vector<cv::Vec3b>& colours, int width, int step, float maxJumpShare, Mesh& mesh, const std::string& group){
//
//@param1 - cloud: a depth cloud, one point per pixel row by row (NaN = none)
//@param2 - colours: one BGR colour per cloud point
//@param3 - width: points per row
//@param4 - step: use every step-th pixel each way
//@param5 - maxJumpShare: a triangle is skipped where its corners' depths differ by more than this share of
//          the nearest (an edge between two things, not a surface)
//@param6 - mesh: the mesh the points and triangles are added to
//@param7 - group: the group name the triangles go in
//
//@brief - the cloud as a grid of triangles facing the camera, two per square of neighbouring sampled pixels
//         (example 09b). Triangles with a missing corner or a depth jump are left out. O(n / step^2).
void MeshFromGrid(const std::vector<cv::Point3f>& cloud, const std::vector<cv::Vec3b>& colours, int width, int step,
                  float maxJumpShare, Mesh& mesh, const std::string& group);

//void AddQuad(Mesh& mesh, const std::string& group, const cv::Point3f& a, const cv::Point3f& b, const cv::Point3f& c, const cv::Point3f& d, const cv::Vec3b& colour, const cv::Point3f& facing){
//
//@param1 - mesh: the mesh to add to
//@param2 - group: the group the two triangles go in
//@param3 - a: the first corner
//@param4 - b: the second corner, round the edge from a
//@param5 - c: the third corner
//@param6 - d: the fourth corner
//@param7 - colour: BGR, for all four corners
//@param8 - facing: a direction the front of the quad should face
//
//@brief - four new points and two triangles (a b c, a c d), wound so their front faces `facing`. O(1).
void AddQuad(Mesh& mesh, const std::string& group, const cv::Point3f& a, const cv::Point3f& b, const cv::Point3f& c,
             const cv::Point3f& d, const cv::Vec3b& colour, const cv::Point3f& facing);

//void AddBox(Mesh& mesh, const std::string& group, const cv::Point3f& low, const cv::Point3f& high, const cv::Vec3b& colour){
//
//@param1 - mesh: the mesh to add to
//@param2 - group: the group the triangles go in
//@param3 - low: the box's lowest corner (smallest x, y, z)
//@param4 - high: its highest corner
//@param5 - colour: BGR
//
//@brief - an axis-aligned box: six AddQuads, each facing out. O(1).
void AddBox(Mesh& mesh, const std::string& group, const cv::Point3f& low, const cv::Point3f& high, const cv::Vec3b& colour);

//void TurnToYUp(Mesh& mesh){
//
//@param1 - mesh: changed in place
//
//@brief - the camera's coordinates (Y down, Z ahead) -> what 3D programs expect (Y up, Z towards the
//         viewer): turned 180 degrees about X, (x, y, z) -> (x, -y, -z). A turn, so triangles keep facing
//         the same way. O(points).
void TurnToYUp(Mesh& mesh);

//bool SaveObj(const std::string& path, const Mesh& mesh, const std::string& comment){
//
//@param1 - path: the .obj file to write (overwritten)
//@param2 - mesh: the points, colours and groups
//@param3 - comment: written at the top after "# "
//
//@brief - a text OBJ: "v x y z r g b" per point (colours 0-1), then per group a "g name" line and its
//         "f a b c" lines (1-based). False if the file can't be written. O(points + triangles).
bool SaveObj(const std::string& path, const Mesh& mesh, const std::string& comment);

//bool SavePly(const std::string& path, const std::vector<cv::Point3f>& points, const std::vector<cv::Vec3b>& colours){
//
//@param1 - path: the .ply file to write (overwritten)
//@param2 - points: the points, written as given (no turning)
//@param3 - colours: one BGR colour per point
//
//@brief - every point with its colour as a binary PLY; points with NaN are left out. False if the file
//         can't be written. O(points).
bool SavePly(const std::string& path, const std::vector<cv::Point3f>& points, const std::vector<cv::Vec3b>& colours);

}
