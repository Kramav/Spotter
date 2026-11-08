// =====================================================================
//  02 · Detect + Distance
//
//  Run YOLO on the Kinect's colour camera, then use the depth camera to
//  measure every box:   person 1.84m x-0.31 y+0.12
//
//  This is the guide's "late fusion" (§1, §5): the detector never sees depth.
//  Depth is only used AFTER detection, in four visible steps:
//
//    1. Detect      YOLO finds boxes in the 1920 x 1080 colour picture.
//    2. Register    For every depth pixel, where it lands in the colour picture
//                   (the two cameras don't line up, §5.1). KinectSource does it.
//    3. Sample      Collect the depth pixels that land in the CENTRE of each
//                   box and take their median (§5.2).
//    4. Deproject   Pixel + depth -> X, Y, Z in metres, with the depth camera's
//                   factory intrinsics (§5.3, §5.4).
//
//  Steps 3 and 4 are this example's idea and are written out below. Later
//  examples use spotter::BoxDepthSampler, which does step 3 for many boxes faster.
//
//  Window: thick box = detection, thin box = the centre area that was sampled.
//  Console: time per step, and each detection's numbers.
//  Keys:   Q or Esc  quit
//  Needs:  yolo26n.onnx and yolo26n.names in the repo root (README: the one-line export).
// =====================================================================

#include <chrono>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/stats.hpp"
#include "spotter/view.hpp"

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// SPOTTER_REPO_ROOT is the repo's folder, filled in by CMake, so this works from any folder.
// Another model (or one you trained) = change these four; nothing else (see the adapter's header).
const std::string ModelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
const std::string NamesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
const int ModelInputSize = 640;
const float MinConfidence = 0.40f;   // lower = more boxes, more mistakes

// Fraction of the box (each way) to sample depth from, around its centre. 0.5 = the middle half.
// 1.0 = the whole box, which usually measures the wall behind the object, not the object.
const double BoxShrink = 0.5;

// Fewer depth pixels than this inside the sampled area -> "?m" instead of a guess.
// Black, shiny and too-close objects end up here.
const int MinDepthSamples = 20;

// Only draw detections closer than this. Set 1.5 for a "within arm's reach" filter (guide §10).
const double MaxDistanceMeters = 100.0;

const int ReportEveryMs = 1000;
const std::string Window = "02 detect + distance  (Q quits)";

spotter::KinectSource::Options KinectSettings()
{
    spotter::KinectSource::Options options;   // colour + depth, cut-off 4.5 m
    return options;
}

spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = ModelPath;
    options.namesPath = NamesPath;
    options.inputSize = ModelInputSize;
    options.minConfidence = MinConfidence;
    return options;
}

// ---------------------------------------------------------------------
//  The new idea: steps 3 and 4, one detection at a time
// ---------------------------------------------------------------------

struct Measured
{
    spotter::Detection detection;
    cv::Rect2d sampleArea;               // the centre of the box
    int samples = 0;                     // depth pixels that landed in it
    std::optional<cv::Point3f> where;    // metres from the depth camera; empty = too few samples
};

Measured Measure(const spotter::Detection& detection, const spotter::RgbdFrame& frame)
{
    Measured result;
    result.detection = detection;
    const cv::Rect2d& box = detection.box;
    result.sampleArea = cv::Rect2d(box.x + box.width * (1 - BoxShrink) / 2, box.y + box.height * (1 - BoxShrink) / 2,
                                   box.width * BoxShrink, box.height * BoxShrink);

    // Step 3: every depth pixel whose colour position falls inside the sample area. Keep its
    // depth, and where it sits in the DEPTH picture (for step 4).
    std::vector<float> depthsMm, columns, rows;
    const int colourWidth = frame.colour.cols;
    for (int row = 0; row < frame.depthMm.rows; row++)
        for (int column = 0; column < frame.depthMm.cols; column++)
        {
            int i = row * frame.depthMm.cols + column;
            int colourIndex = frame.colourIndex[i];
            float mm = frame.depthMm.at<float>(row, column);
            if (colourIndex < 0 || mm <= 0) continue;
            cv::Point2d colourPixel(colourIndex % colourWidth, colourIndex / colourWidth);
            if (!result.sampleArea.contains(colourPixel)) continue;
            depthsMm.push_back(mm);
            columns.push_back((float)column);
            rows.push_back((float)row);
        }
    result.samples = (int)depthsMm.size();
    if (result.samples < MinDepthSamples) return result;

    // The median ignores the few background and edge pixels that are left (§5.2).
    float meters = spotter::Median(depthsMm) / 1000;
    cv::Point2f centre(spotter::Median(columns), spotter::Median(rows));   // the object's middle, in the depth picture

    // Step 4: the pinhole camera model (§5.3), on the depth camera (lens distortion already removed):
    //   X = (u + 0.5 - cx) * Z / fx,   Y = (v + 0.5 - cy) * Z / fy      (x right, y down, z ahead)
    result.where = frame.depthCamera.Unproject(centre, meters);
    return result;
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

std::string Label(const Measured& m, const spotter::ClassList& classes)
{
    const std::string& name = classes.Name(m.detection.classId);
    if (!m.where) return name + " ?m";
    return cv::format("%s %.2fm x%+.2f y%+.2f", name.c_str(), m.where->z, m.where->x, m.where->y);
}

void Show(cv::Mat& picture, const std::vector<Measured>& objects, const spotter::ClassList& classes, const spotter::FpsMeter& fps)
{
    const cv::Scalar green(0, 255, 0), yellow(0, 255, 255);
    for (const Measured& m : objects)
    {
        if (m.where && m.where->z > MaxDistanceMeters) continue;
        cv::rectangle(picture, m.detection.box, green, 3);
        cv::rectangle(picture, m.sampleArea, yellow, 1);
        cv::putText(picture, Label(m, classes), cv::Point((int)m.detection.box.x, std::max((int)m.detection.box.y - 10, 30)),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, green, 2);
    }
    fps.Draw(picture);
    cv::imshow(Window, picture);
}

struct StepTimes { double captureMs = 0, detectMs = 0, measureMs = 0; };

void Report(const std::vector<Measured>& objects, const spotter::ClassList& classes, const StepTimes& times, const spotter::FpsMeter& fps)
{
    std::printf("%.1f fps | capture+register %.0f ms, detect %.0f ms, sample+deproject %.0f ms | %zu detections\n",
                fps.Fps(), times.captureMs, times.detectMs, times.measureMs, objects.size());
    for (const Measured& m : objects)
        std::printf("  %-40s conf %.2f  samples %5d\n", Label(m, classes).c_str(), m.detection.confidence, m.samples);
}

double MillisecondsSince(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(KinectSettings());
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    cv::namedWindow(Window, cv::WINDOW_NORMAL);
    cv::resizeWindow(Window, 960, 540);
    spotter::FpsMeter fps;
    spotter::Every report(ReportEveryMs);
    spotter::RgbdFrame frame;
    StepTimes times;

    while (true)
    {
        auto start = std::chrono::steady_clock::now();
        if (!kinect.Next(frame)) break;                                            // step 2 happens in here
        times.captureMs = MillisecondsSince(start);

        start = std::chrono::steady_clock::now();
        std::vector<spotter::Detection> detections = yolo.Detect(frame.colour);    // step 1
        times.detectMs = MillisecondsSince(start);

        start = std::chrono::steady_clock::now();
        std::vector<Measured> objects;
        for (const spotter::Detection& detection : detections) objects.push_back(Measure(detection, frame));   // steps 3 + 4
        times.measureMs = MillisecondsSince(start);

        fps.Tick();
        Show(frame.colour, objects, yolo.Classes(), fps);
        if (report.Due()) Report(objects, yolo.Classes(), times, fps);
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Stand 1 m in front of a far wall and set BoxShrink = 1.0. Your distance jumps toward the
//     wall's, because most pixels in a loose box are background. That's why the centre is used.
//  2. Walk slowly sideways across the view at a fixed distance. The number should stay steady
//     even at the edges. If it jumps to the wall at the edges, registration is off (guide §7 test 3).
//  3. Check against a tape measure at 1, 2 and 3 m (guide §7 test 2). Expect a few cm.
//  4. Watch "sample+deproject" in the console with several people in view: step 3 visits all
//     217k depth pixels per box. 05c measures every track every frame, so it uses
//     spotter::BoxDepthSampler (one pass per frame, then only the pixels inside each box).
//  5. Set MaxDistanceMeters = 1.5 and walk toward the sensor: boxes appear only within reach.
//  6. Hold up a phone showing a photo of a person. YOLO says "person". The distance says 0.4 m.
//     How would you reject it? (Hint: real height = box height in pixels * Z / focal length; guide §10.)
//  7. Try another model: export yolo26n at imgsz=320, set ModelPath and ModelInputSize. Faster,
//     but it misses objects cut off at the picture's edge.
