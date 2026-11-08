// spotter/detection.hpp: what any object detector hands back. No YOLO in here, so a different
// detector (another model, another runtime) fills the same struct.
#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace spotter
{

struct Detection
{
    int classId = -1;
    float confidence = 0;   // 0..1
    cv::Rect2d box;         // in the picture the detector was given, pixels
};

// The class names a model was trained on, one per line in a text file (yolo26n.names: the 80
// COCO classes). A model you trained yourself brings its own file, in its data.yaml's order.
class ClassList
{
public:
    ClassList() = default;
    explicit ClassList(const std::string& path);   // prints a warning if it can't be read

    size_t Size() const { return names.size(); }
    const std::string& Name(int classId) const;    // "?" for an id outside the list
    int Find(const std::string& name) const;        // the id of "person", or -1 if there's no such class

private:
    std::vector<std::string> names;
};

}
