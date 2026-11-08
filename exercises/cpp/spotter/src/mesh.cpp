#include "spotter/mesh.hpp"

#include <cmath>
#include <fstream>

namespace spotter
{

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
