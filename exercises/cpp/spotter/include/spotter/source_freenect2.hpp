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
        float maxDepthMeters = 4.5f;   // libfreenect2 drops readings past this; the sensor reads to ~6 m (07b)
        bool depthFilters = true;      // bilateral + edge-aware filters: fewer flying pixels, half the fps (01)
        bool occlusionFilter = true;   // registered colour: black where the colour camera can't see that spot
        bool capExposure = false;      // cap the colour exposure at 16 ms: a steady frame rate in dim light (D15)
        bool bigdepth = false;         // also fill RgbdFrame::bigdepth (depth per colour pixel); costs a few ms
        bool unmirror = true;          // the Kinect sends mirror images; true flips them the right way round
        int timeoutMs = 10000;         // Next() gives up after this long without a frame
    };

    // Opens the first Kinect and starts it. Prints what went wrong if it can't; then IsOpen() is false.
    explicit KinectSource(const Options& options);
    ~KinectSource();   // stops and closes the device, also when an exception unwinds past it
    KinectSource(const KinectSource&) = delete;
    KinectSource& operator=(const KinectSource&) = delete;

    bool IsOpen() const;

    // Waits for the next frame and fills `frame` (its pictures are reused between calls).
    // False when the device isn't open or no frame came within timeoutMs (unplugged?).
    bool Next(RgbdFrame& frame);

    // Turn bigdepth on or off from the next frame on (13e asks for it only now and then).
    void SetBigdepth(bool on);

    // The factory calibration as text lines, for saving next to a scan (room.txt):
    // "ir_params fx fy cx cy k1 k2 k3 p1 p2" and "colour_params <every number libfreenect2 has>".
    // Before Unmirror: these are what the device reports.
    std::string CalibrationText() const;

private:
    struct Device;   // libfreenect2's objects, kept out of this header
    std::unique_ptr<Device> device;
};

}
