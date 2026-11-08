// spotter/mesh.hpp: 3D files. Point clouds as PLY (opens in MeshLab, CloudCompare, Blender).
#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace spotter
{

// Every point with its colour (BGR, as OpenCV keeps it), as a binary PLY. Points with NaN are
// left out. Written as given: turn the points first for another axis convention.
// A point cloud is a capture of the room: *.ply is gitignored, never commit one.
bool SavePly(const std::string& path, const std::vector<cv::Point3f>& points, const std::vector<cv::Vec3b>& colours);

}
