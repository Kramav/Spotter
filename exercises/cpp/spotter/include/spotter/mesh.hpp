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

// A depth cloud (one point per pixel, NaN = none, `width` per row) as a grid of triangles: every
// step-th pixel, two triangles per square, skipped where the corners' depths differ by more than
// maxJumpShare of the nearest (an edge between two things, not a surface). Example 09b.
void MeshFromGrid(const std::vector<cv::Point3f>& cloud, const std::vector<cv::Vec3b>& colours, int width, int step,
                  float maxJumpShare, Mesh& mesh, const std::string& group);

// Two triangles for the corners a b c d (in order round the edge), turned to face `facing`.
void AddQuad(Mesh& mesh, const std::string& group, const cv::Point3f& a, const cv::Point3f& b, const cv::Point3f& c,
             const cv::Point3f& d, const cv::Vec3b& colour, const cv::Point3f& facing);

// A box from its lowest to its highest corner, its six sides facing out.
void AddBox(Mesh& mesh, const std::string& group, const cv::Point3f& low, const cv::Point3f& high, const cv::Vec3b& colour);

// The camera's coordinates (Y down, Z ahead) -> what 3D programs expect (Y up, Z towards the
// viewer): turned 180 degrees about X.
void TurnToYUp(Mesh& mesh);

// OBJ: "v x y z r g b" per point (colours 0-1), then the groups' "f a b c" lines.
bool SaveObj(const std::string& path, const Mesh& mesh, const std::string& comment);

// Every point with its colour, as a binary PLY. Points with NaN are left out. Written as given.
bool SavePly(const std::string& path, const std::vector<cv::Point3f>& points, const std::vector<cv::Vec3b>& colours);

}
