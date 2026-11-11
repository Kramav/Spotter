// spotter/radar.hpp: example 06's bird's-eye map. The camera sits at the bottom centre looking up
// the screen; X (right) and Z (ahead) are metres on the floor.
#pragma once

#include "spotter/frame.hpp"
#include "spotter/track.hpp"

#include <deque>
#include <map>
#include <string>

namespace spotter
{

class RadarView
{
public:
    //RadarView(double rangeMeters, int sizePixels){
    //
    //@param1 - rangeMeters: how far ahead the map reaches (default 6)
    //@param2 - sizePixels: the square picture's size (default 700)
    //
    //@brief - a map with the camera at the bottom centre, scaled so rangeMeters fits. O(1).
    explicit RadarView(double rangeMeters = 6.0, int sizePixels = 700);

    //cv::Mat& Begin(const PinholeCamera& colourCamera){
    //
    //@param1 - colourCamera: the colour camera's model, for its field of view
    //
    //@brief - a fresh map: a ring every metre, the colour camera's field of view, the 0.5 m minimum range.
    //         Returns the picture to draw on. O(pixels).
    cv::Mat& Begin(const PinholeCamera& colourCamera);

    //cv::Point ToPixel(cv::Point2d floor){
    //
    //@param1 - floor: a spot on the floor, (X right, Z ahead) in metres
    //
    //@brief - the map pixel it lands on. O(1).
    cv::Point ToPixel(cv::Point2d floor) const;

    //void DrawLine(cv::Point2d a, cv::Point2d b, cv::Scalar colour, int thickness, const std::string& label){
    //
    //@param1 - a: one end on the floor, metres
    //@param2 - b: the other end
    //@param3 - colour: BGR
    //@param4 - thickness: in pixels
    //@param5 - label: text by its middle (default none)
    //
    //@brief - a line on the map, e.g. a wall. O(line length in pixels).
    void DrawLine(cv::Point2d a, cv::Point2d b, cv::Scalar colour, int thickness, const std::string& label = "");

    //void DrawTrail(const std::deque<cv::Point2d>& trail, cv::Scalar colour, bool hollow, const std::string& label){
    //
    //@param1 - trail: where the object has been on the floor, oldest first
    //@param2 - colour: BGR
    //@param3 - hollow: true = a hollow dot ("predicted only", coasting), false = filled
    //@param4 - label: text beside the dot
    //
    //@brief - the trail as joined lines and a dot at its newest end. Nothing for an empty trail. O(trail length).
    void DrawTrail(const std::deque<cv::Point2d>& trail, cv::Scalar colour, bool hollow, const std::string& label);

    //cv::Mat& Image(){
    //
    //@brief - the map picture. O(1).
    cv::Mat& Image() { return image; }

private:
    double rangeMeters, pixelsPerMeter;
    int sizePixels;
    cv::Point camera;
    cv::Mat image;
};

// Where each tracked object has been on the floor: its filtered box centre and distance, through
// the colour camera's pinhole model (06). Forgets the IDs the tracker forgot.
class TrackTrails
{
public:
    //TrackTrails(int length){
    //
    //@param1 - length: how many spots each trail keeps (default 150, 5 s at 30 fps)
    //
    //@brief - no trails yet. O(1).
    explicit TrackTrails(int length = 150) : length(length) {}

    //void Update(const Tracker& tracker, const PinholeCamera& colourCamera){
    //
    //@param1 - tracker: the tracker after this frame's Predict/Match/Remeasure
    //@param2 - colourCamera: the colour camera's model, to turn a box centre and distance into X, Z
    //
    //@brief - adds each tracked object's floor spot (once it has a distance) to its trail, drops the oldest
    //         spot past `length`, and forgets the trails of IDs no longer tracked. O(tracks x log tracks).
    void Update(const Tracker& tracker, const PinholeCamera& colourCamera);

    //const std::deque<cv::Point2d>& Of(int id){
    //
    //@param1 - id: a track's ID
    //
    //@brief - that object's trail; empty until it has a distance, or for an unknown ID. O(log tracks).
    const std::deque<cv::Point2d>& Of(int id) const;

    //void Clear(){
    //
    //@brief - empties every trail. O(spots kept).
    void Clear();

private:
    int length;
    std::map<int, std::deque<cv::Point2d>> trails;
};

}
