// spotter/source_freenect2.hpp: the Kinect v2, through libfreenect2 (adapter, D34).
// The only place that includes libfreenect2. Another camera = a sibling file with the same
// shape (Options, IsOpen, Next) that fills the same RgbdFrame.
#pragma once

#include "spotter/frame.hpp"

#include <memory>
#include <string>

namespace spotter
{

class KinectSource
{
public:
    struct Options
    {
        bool colour = true;            // false = depth only (faster; no colour, registered or colourIndex)
        float maxDepthMeters = 6.0f;   // readings past this are dropped. libfreenect2's own default is 4.5;
                                       // the sensor reads to ~6 m, past that almost nothing is right (07)
        bool depthFilters = true;      // bilateral + edge-aware filters: fewer flying pixels, half the fps (01)
        bool occlusionFilter = true;   // registered colour: black where the colour camera can't see that spot
        bool capExposure = false;      // cap the colour exposure at 16 ms: a steady frame rate in dim light (D15)
        bool bigdepth = false;         // also fill RgbdFrame::bigdepth (depth per colour pixel); costs a few ms
        bool unmirror = true;          // the Kinect sends mirror images; true flips them the right way round
        int timeoutMs = 10000;         // Next() gives up after this long without a frame
    };

    //KinectSource(const Options& options){
    //
    //@param1 - options: which pictures to make, the depth range, the filters, the timeout
    //
    //@brief - opens the first Kinect and starts its streams. Prints what went wrong if it can't; then
    //         IsOpen() is false.
    explicit KinectSource(const Options& options);

    //~KinectSource(){
    //
    //@brief - stops and closes the device, also when an exception unwinds past it (RAII).
    ~KinectSource();
    KinectSource(const KinectSource&) = delete;
    KinectSource& operator=(const KinectSource&) = delete;

    //bool IsOpen(){
    //
    //@brief - true if a Kinect was opened and started. O(1).
    bool IsOpen() const;

    //bool Next(RgbdFrame& frame){
    //
    //@param1 - frame: filled with the new frame; its pictures are reused between calls
    //
    //@brief - waits for the next frame, undistorts and registers depth and colour (libfreenect2), adds
    //         the cameras' pinhole models (made at opening) and unmirrors it all (Unmirror) if asked. False when the device
    //         isn't open or no frame came within timeoutMs (unplugged?). O(pixels) per frame.
    bool Next(RgbdFrame& frame);

    //void SetBigdepth(bool on){
    //
    //@param1 - on: true to fill RgbdFrame::bigdepth, false to skip it
    //
    //@brief - turns bigdepth on or off from the next frame on (13e asks for it only now and then). O(1).
    void SetBigdepth(bool on);

    //std::string CalibrationText(){
    //
    //@brief - the factory calibration as text lines, for saving next to a scan (room.txt):
    //         "ir_params fx fy cx cy k1 k2 k3 p1 p2" and "colour_params <every number libfreenect2 has>".
    //         Before Unmirror: these are what the device reports. O(1).
    std::string CalibrationText() const;

private:
    struct Device;   // libfreenect2's objects, kept out of this header
    std::unique_ptr<Device> device;
};

}
