#include "spotter/source_freenect2.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/registration.h>
#include <opencv2/imgproc.hpp>

namespace spotter
{

namespace
{
const int DepthWidth = 512, DepthHeight = 424, ColourWidth = 1920, ColourHeight = 1080;

//void CleanCopy(const float* from, cv::Mat& to, int rows, int columns){
//
//@param1 - from: libfreenect2's depth floats, millimetres
//@param2 - to: the CV_32F picture to fill (made the right size)
//@param3 - rows: the picture's height
//@param4 - columns: the picture's width
//
//@brief - copies libfreenect2's depth with "no reading" (NaN, infinity, 0) as 0. O(rows x columns).
void CleanCopy(const float* from, cv::Mat& to, int rows, int columns)
{
    to.create(rows, columns, CV_32F);
    float* out = to.ptr<float>();
    for (int i = 0; i < rows * columns; i++) out[i] = (from[i] > 0 && !std::isinf(from[i])) ? from[i] : 0;
}
}

struct KinectSource::Device
{
    Options options;
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device* device = nullptr;
    std::unique_ptr<libfreenect2::SyncMultiFrameListener> listener;
    std::unique_ptr<libfreenect2::Registration> registration;
    libfreenect2::Frame undistorted{ DepthWidth, DepthHeight, 4 }, registered{ DepthWidth, DepthHeight, 4 };
    libfreenect2::Frame bigdepth{ ColourWidth, ColourHeight + 2, 4 };   // a blank row above and below
    PinholeCamera depthCamera, colourCamera;
};

KinectSource::KinectSource(const Options& options) : device(std::make_unique<Device>())
{
    Device& d = *device;
    d.options = options;
    libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Warning));
    if (d.freenect2.enumerateDevices() == 0)
    {
        std::printf("No Kinect v2 found. Check the adapter's power light, use a USB 3.0 port,\n"
                    "and check the USB driver is UsbDk or libusbK (Windows); see the README.\n");
        return;
    }
    d.device = d.freenect2.openDefaultDevice();
    if (d.device == nullptr)
    {
        std::printf("Found a Kinect but could not open it. Is another program using it?\n");
        return;
    }

    libfreenect2::Freenect2Device::Config config;
    config.MaxDepth = options.maxDepthMeters;
    config.EnableBilateralFilter = config.EnableEdgeAwareFilter = options.depthFilters;
    d.device->setConfiguration(config);

    unsigned types = libfreenect2::Frame::Depth | (options.colour ? libfreenect2::Frame::Color : 0);
    d.listener = std::make_unique<libfreenect2::SyncMultiFrameListener>(types);
    if (options.colour) d.device->setColorFrameListener(d.listener.get());
    d.device->setIrAndDepthFrameListener(d.listener.get());
    if (!d.device->startStreams(options.colour, true))
    {
        std::printf("The Kinect opened but would not start streaming.\n");
        d.device->close();
        d.device = nullptr;
        return;
    }
    if (options.colour && options.capExposure) d.device->setColorSemiAutoExposure(16.0f);

    // The factory calibration is read while the streams start.
    libfreenect2::Freenect2Device::IrCameraParams ir = d.device->getIrCameraParams();
    libfreenect2::Freenect2Device::ColorCameraParams colour = d.device->getColorCameraParams();
    d.registration = std::make_unique<libfreenect2::Registration>(ir, colour);
    d.depthCamera = { ir.fx, ir.fy, ir.cx, ir.cy, DepthWidth, DepthHeight };
    d.colourCamera = { colour.fx, colour.fy, colour.cx, colour.cy, ColourWidth, ColourHeight };
    std::printf("Kinect %s opened: %s, depth to %.1f m.\n", d.device->getSerialNumber().c_str(),
                options.colour ? "colour + depth" : "depth only", options.maxDepthMeters);
}

KinectSource::~KinectSource()
{
    if (device->device == nullptr) return;
    device->device->stop();
    device->device->close();
}

bool KinectSource::IsOpen() const { return device->device != nullptr; }

void KinectSource::SetBigdepth(bool on) { device->options.bigdepth = on; }

bool KinectSource::Next(RgbdFrame& frame)
{
    Device& d = *device;
    if (d.device == nullptr) return false;
    libfreenect2::FrameMap frames;
    if (!d.listener->waitForNewFrame(frames, d.options.timeoutMs))
    {
        std::printf("No frame from the Kinect for %d s: unplugged?\n", d.options.timeoutMs / 1000);
        return false;
    }
    frame.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    frame.depthCamera = d.depthCamera;
    frame.colourCamera = d.colourCamera;
    libfreenect2::Frame* depth = frames[libfreenect2::Frame::Depth];

    if (d.options.colour)
    {
        libfreenect2::Frame* colour = frames[libfreenect2::Frame::Color];
        frame.colourIndex.resize(DepthWidth * DepthHeight);
        d.registration->apply(colour, depth, &d.undistorted, &d.registered, d.options.occlusionFilter,
                              d.options.bigdepth ? &d.bigdepth : nullptr, frame.colourIndex.data());
        // The camera sends JPEG; libfreenect2 decodes it to BGRX (4 bytes per pixel, the 4th unused).
        cv::cvtColor(cv::Mat(ColourHeight, ColourWidth, CV_8UC4, colour->data), frame.colour, cv::COLOR_BGRA2BGR);
        cv::cvtColor(cv::Mat(DepthHeight, DepthWidth, CV_8UC4, d.registered.data), frame.registered, cv::COLOR_BGRA2BGR);
        if (d.options.bigdepth) CleanCopy((const float*)d.bigdepth.data + ColourWidth, frame.bigdepth, ColourHeight, ColourWidth);
        else frame.bigdepth.release();
    }
    else
    {
        d.registration->undistortDepth(depth, &d.undistorted);
        frame.colour.release();
        frame.registered.release();
        frame.bigdepth.release();
        frame.colourIndex.clear();
    }
    CleanCopy((const float*)d.undistorted.data, frame.depthMm, DepthHeight, DepthWidth);
    d.listener->release(frames);   // hand the buffers back, or the listener stops delivering

    if (d.options.unmirror) Unmirror(frame);
    return true;
}

std::string KinectSource::CalibrationText() const
{
    if (device->device == nullptr) return "";
    libfreenect2::Freenect2Device::IrCameraParams ir = device->device->getIrCameraParams();
    libfreenect2::Freenect2Device::ColorCameraParams colour = device->device->getColorCameraParams();
    // libfreenect2's own order: the depth camera's fx fy cx cy k1 k2 k3 p1 p2, then the colour
    // camera's fx fy cx cy shift_d shift_m mx_... my_... (both structs are all floats).
    std::string text = "ir_params";
    for (size_t i = 0; i < sizeof ir / sizeof(float); i++) text += cv::format(" %.9g", (&ir.fx)[i]);
    text += "\ncolour_params";
    for (size_t i = 0; i < sizeof colour / sizeof(float); i++) text += cv::format(" %.9g", (&colour.fx)[i]);
    return text + "\n";
}

}
