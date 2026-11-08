#include "spotter/detection.hpp"

#include <cstdio>
#include <fstream>

namespace spotter
{

ClassList::ClassList(const std::string& path)
{
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line))
    {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty()) names.push_back(line);
    }
    if (names.empty()) std::printf("Warning: no class names in %s; classes show as \"?\".\n", path.c_str());
}

const std::string& ClassList::Name(int classId) const
{
    static const std::string unknown = "?";
    return classId >= 0 && classId < (int)names.size() ? names[classId] : unknown;
}

int ClassList::Find(const std::string& name) const
{
    for (size_t i = 0; i < names.size(); i++)
        if (names[i] == name) return (int)i;
    return -1;
}

}
