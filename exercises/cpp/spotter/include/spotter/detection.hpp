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

    //ClassList(const std::string& path){
    //
    //@param1 - path: the names file, one class name per line, in training order
    //
    //@brief - reads the names. Prints a warning if the file can't be read (then the list is empty). O(lines).
    explicit ClassList(const std::string& path);

    //size_t Size(){
    //
    //@brief - how many class names were read. O(1).
    size_t Size() const { return names.size(); }

    //const std::string& Name(int classId){
    //
    //@param1 - classId: a detection's class number
    //
    //@brief - that class's name, or "?" for an id outside the list (bounds-checked). O(1).
    const std::string& Name(int classId) const;

    //int Find(const std::string& name){
    //
    //@param1 - name: a class name, e.g. "person"
    //
    //@brief - that class's id, or -1 if there's no such class. Looks through the list: O(classes).
    int Find(const std::string& name) const;

private:
    std::vector<std::string> names;
};

}
