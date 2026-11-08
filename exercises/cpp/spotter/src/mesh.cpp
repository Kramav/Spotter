#include "spotter/mesh.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace spotter
{

void MeshFromGrid(const std::vector<cv::Point3f>& cloud, const std::vector<cv::Vec3b>& colours, int width, int step,
                  float maxJumpShare, Mesh& mesh, const std::string& group)
{
    const int height = (int)cloud.size() / width;
    const int columns = (width + step - 1) / step, rows = (height + step - 1) / step;
    std::vector<int> pointOf(columns * rows, -1);
    std::vector<float> depthOf(columns * rows);
    for (int row = 0; row < rows; row++)
        for (int column = 0; column < columns; column++)
        {
            int pixel = row * step * width + column * step;
            if (std::isnan(cloud[pixel].z)) continue;
            pointOf[row * columns + column] = (int)mesh.points.size();
            depthOf[row * columns + column] = cloud[pixel].z;
            mesh.points.push_back(cloud[pixel]);
            mesh.colours.push_back(colours[pixel]);
        }
    std::vector<cv::Vec3i>& faces = mesh.groups[group];
    auto AddTriangle = [&](int a, int b, int c)
    {
        if (pointOf[a] < 0 || pointOf[b] < 0 || pointOf[c] < 0) return;
        float nearest = std::min({ depthOf[a], depthOf[b], depthOf[c] });
        float farthest = std::max({ depthOf[a], depthOf[b], depthOf[c] });
        if (farthest - nearest > maxJumpShare * nearest) return;
        faces.emplace_back(pointOf[a], pointOf[b], pointOf[c]);   // this order faces the camera
    };
    for (int row = 0; row + 1 < rows; row++)
        for (int column = 0; column + 1 < columns; column++)
        {
            int topLeft = row * columns + column, topRight = topLeft + 1, bottomLeft = topLeft + columns, bottomRight = bottomLeft + 1;
            AddTriangle(topLeft, bottomLeft, topRight);
            AddTriangle(topRight, bottomLeft, bottomRight);
        }
}

void AddQuad(Mesh& mesh, const std::string& group, const cv::Point3f& a, const cv::Point3f& b, const cv::Point3f& c,
             const cv::Point3f& d, const cv::Vec3b& colour, const cv::Point3f& facing)
{
    int first = (int)mesh.points.size();
    for (const cv::Point3f& p : { a, b, c, d })
    {
        mesh.points.push_back(p);
        mesh.colours.push_back(colour);
    }
    bool flip = (b - a).cross(c - a).dot(facing) < 0;
    std::vector<cv::Vec3i>& faces = mesh.groups[group];
    faces.push_back(flip ? cv::Vec3i(first, first + 2, first + 1) : cv::Vec3i(first, first + 1, first + 2));
    faces.push_back(flip ? cv::Vec3i(first, first + 3, first + 2) : cv::Vec3i(first, first + 2, first + 3));
}

void AddBox(Mesh& mesh, const std::string& group, const cv::Point3f& low, const cv::Point3f& high, const cv::Vec3b& colour)
{
    cv::Point3f corner[8];
    for (int i = 0; i < 8; i++) corner[i] = cv::Point3f(i & 1 ? high.x : low.x, i & 2 ? high.y : low.y, i & 4 ? high.z : low.z);
    const int sides[6][4] = { { 0, 2, 6, 4 }, { 1, 3, 7, 5 }, { 0, 1, 5, 4 }, { 2, 3, 7, 6 }, { 0, 1, 3, 2 }, { 4, 5, 7, 6 } };
    cv::Point3f centre = (low + high) / 2;
    for (const auto& s : sides)
    {
        cv::Point3f middle = (corner[s[0]] + corner[s[1]] + corner[s[2]] + corner[s[3]]) / 4;
        AddQuad(mesh, group, corner[s[0]], corner[s[1]], corner[s[2]], corner[s[3]], colour, middle - centre);
    }
}

void TurnToYUp(Mesh& mesh)
{
    for (cv::Point3f& p : mesh.points) p = cv::Point3f(p.x, -p.y, -p.z);   // turning keeps the triangles facing the same way
}

bool SaveObj(const std::string& path, const Mesh& mesh, const std::string& comment)
{
    std::ofstream file(path);
    file << "# " << comment << "\n";
    for (size_t i = 0; i < mesh.points.size(); i++)
    {
        const cv::Point3f& p = mesh.points[i];
        const cv::Vec3b& c = mesh.colours[i];
        file << cv::format("v %.4f %.4f %.4f %.3f %.3f %.3f\n", p.x, p.y, p.z, c[2] / 255.0, c[1] / 255.0, c[0] / 255.0);
    }
    for (const auto& [name, faces] : mesh.groups)
    {
        if (faces.empty()) continue;
        file << "g " << name << "\n";
        for (const cv::Vec3i& f : faces) file << cv::format("f %d %d %d\n", f[0] + 1, f[1] + 1, f[2] + 1);
    }
    return (bool)file;
}

bool SavePly(const std::string& path, const std::vector<cv::Point3f>& points, const std::vector<cv::Vec3b>& colours)
{
    size_t count = 0;
    for (const cv::Point3f& p : points) count += !std::isnan(p.z);
    // Binary, not text: a room's 5 million points would be about 150 MB of text.
    std::ofstream file(path, std::ios::binary);
    file << "ply\nformat binary_little_endian 1.0\n"
         << "comment Spotter. A capture of the room: never commit it.\n"
         << "element vertex " << count << "\n"
         << "property float x\nproperty float y\nproperty float z\n"
         << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (size_t i = 0; i < points.size() && i < colours.size(); i++)
    {
        if (std::isnan(points[i].z)) continue;
        file.write((const char*)&points[i], 3 * sizeof(float));
        const unsigned char rgb[3] = { colours[i][2], colours[i][1], colours[i][0] };
        file.write((const char*)rgb, 3);
    }
    return (bool)file;
}

}
