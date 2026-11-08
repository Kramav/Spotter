// =====================================================================
//  06 · step 2 of 2 · Draw them from above
//
//  Step 1's floor spots on a bird's-eye map: the camera at the bottom
//  centre, looking up the screen, a ring every metre. One dot per
//  detection, this frame only (no tracking yet).
//
//  New in this step: a second window, metres -> map pixels.
//  Keys:  Q or Esc  quit
//  Next:  06_Radar (the full example tracks each object and draws its trail)
// =====================================================================

#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "Q or Esc  quit";

const double RangeMeters = 6.0;
const int MapPixels = 700;
const double PixelsPerMeter = (MapPixels - 40) / RangeMeters;
const cv::Point Camera(MapPixels / 2, MapPixels - 20);   // bottom centre

// x metres to the side, z metres ahead -> a map pixel (z up the screen).
cv::Point ToMap(double x, double z) { return { (int)std::lround(Camera.x + x * PixelsPerMeter), (int)std::lround(Camera.y - z * PixelsPerMeter) }; }

cv::Mat EmptyMap()
{
    cv::Mat map(MapPixels, MapPixels, CV_8UC3, cv::Scalar(25, 25, 25));
    for (int meters = 1; meters <= (int)RangeMeters; meters++)
    {
        int radius = (int)(meters * PixelsPerMeter);
        cv::ellipse(map, Camera, cv::Size(radius, radius), 0, 180, 360, cv::Scalar(80, 80, 80), 1);   // the upper half of a circle
        cv::putText(map, std::to_string(meters) + " m", Camera + cv::Point(4, -radius - 4), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(80, 80, 80), 1);
    }
    cv::drawMarker(map, Camera, cv::Scalar(0, 255, 255), cv::MARKER_TRIANGLE_UP, 14, 2);
    return map;
}

spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = 0.4f;
    return options;
}

int main()
{
    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    cv::namedWindow("06 step 2: camera", cv::WINDOW_NORMAL);
    cv::resizeWindow("06 step 2: camera", 960, 540);
    spotter::BoxDepthSampler boxDepth;
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        boxDepth.Build(frame);
        cv::Mat map = EmptyMap();
        for (const spotter::Detection& d : yolo.Detect(frame.colour))
        {
            std::optional<double> meters = boxDepth.MedianMeters(d.box);
            if (!meters) continue;
            cv::Point3f p = frame.colourCamera.Unproject(cv::Point2f((float)(d.box.x + d.box.width / 2), (float)(d.box.y + d.box.height / 2)), (float)*meters);
            cv::circle(map, ToMap(p.x, p.z), 7, cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(map, yolo.Classes().Name(d.classId), ToMap(p.x, p.z) + cv::Point(10, 4), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 1);
            cv::rectangle(frame.colour, d.box, cv::Scalar(0, 255, 0), 3);
        }
        cv::imshow("06 step 2: camera", spotter::WithKeys(frame.colour, Keys));
        cv::imshow("06 step 2: radar (camera at the bottom)", spotter::WithKeys(map, Keys));
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: walk a circle around the room. The dot follows you on the map; it jumps a little each
// frame (box jitter). The full example smooths it with a tracker and draws where you've been.
