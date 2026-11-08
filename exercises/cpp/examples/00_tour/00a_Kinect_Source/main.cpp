// =====================================================================
//  00a · Tour: KinectSource and RgbdFrame          (needs the Kinect)
//
//  What it's for: every example starts with a camera. spotter::KinectSource
//  opens the Kinect v2, and each Next() fills one spotter::RgbdFrame:
//
//    colour       1920 x 1080 BGR          what the colour camera sees
//    depthMm      512 x 424 float, mm      distance per depth pixel, 0 = no reading
//    registered   512 x 424 BGR            the colour seen AT each depth pixel
//    colourIndex  512 x 424 int            where each depth pixel lands in colour
//    depthCamera, colourCamera             the factory pinhole numbers (tour 00b)
//
//  How it works (spotter/src/source_freenect2.cpp, the only file that
//  includes libfreenect2):
//    1. Open the first device, set the depth cut-off and filters, start it.
//    2. Per frame: libfreenect2's Registration removes the depth lens's
//       distortion and maps every depth pixel into the colour picture.
//    3. NaN / infinity become 0, BGRX becomes BGR.
//    4. Unmirror: the Kinect sends mirror images (like a selfie camera). Every
//       picture is flipped, colourIndex follows, and both cameras' cx move
//       (cx -> width - cx) so 3D points come out the right way round too.
//    5. The destructor stops the device, even when an exception unwinds.
//  Another camera = a sibling file (source_<camera>.cpp) filling the same frame.
//
//  Check the mirror fix: raise your RIGHT hand. It must show on the RIGHT.
//  Window: colour (shrunk) | depth | registered colour, side by side.
//  Keys:   Q or Esc  quit
// =====================================================================

#include <cstdio>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "Q or Esc  quit";

const std::string Window = "00a KinectSource: colour | depth | registered  (raise your RIGHT hand: it shows on the right)";

void PrintFirstFrame(const spotter::RgbdFrame& frame, const spotter::KinectSource& kinect)
{
    const spotter::PinholeCamera& d = frame.depthCamera;
    const spotter::PinholeCamera& c = frame.colourCamera;
    std::printf("colour %d x %d, depth %d x %d, registered %d x %d, colourIndex %zu entries\n", frame.colour.cols, frame.colour.rows,
                frame.depthMm.cols, frame.depthMm.rows, frame.registered.cols, frame.registered.rows, frame.colourIndex.size());
    std::printf("depth camera:  fx %.1f fy %.1f cx %.1f cy %.1f (after the unmirror)\n", d.fx, d.fy, d.cx, d.cy);
    std::printf("colour camera: fx %.1f fy %.1f cx %.1f cy %.1f\n", c.fx, c.fy, c.cx, c.cy);
    std::printf("as the device reports it (saved with room scans):\n%s", kinect.CalibrationText().c_str());
}

// The depth pixel in the middle: its distance, its 3D point, and the colour pixel it lands on.
void PrintCentre(const spotter::RgbdFrame& frame)
{
    cv::Point centre(frame.depthMm.cols / 2, frame.depthMm.rows / 2);
    float mm = frame.depthMm.at<float>(centre);
    int index = frame.colourIndex[centre.y * frame.depthMm.cols + centre.x];
    if (mm <= 0 || index < 0) { std::printf("centre pixel: no reading\n"); return; }
    cv::Point3f p = frame.depthCamera.Unproject(cv::Point2f(centre), mm / 1000);
    std::printf("centre depth pixel: %.0f mm -> 3D (%+.3f, %+.3f, %.3f) m -> colour pixel (%d, %d)\n",
                mm, p.x, p.y, p.z, index % frame.colour.cols, index / frame.colour.cols);
}

cv::Mat SideBySide(const spotter::RgbdFrame& frame)
{
    cv::Mat colour, picture;
    cv::resize(frame.colour, colour, cv::Size(), 424.0 / frame.colour.rows, 424.0 / frame.colour.rows);
    cv::hconcat(std::vector<cv::Mat>{ colour, spotter::DepthPicture(frame.depthMm), frame.registered }, picture);
    return picture;
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    spotter::KinectSource kinect(spotter::KinectSource::Options{});   // colour + depth, the defaults
    if (!kinect.IsOpen()) return 1;

    cv::namedWindow(Window, cv::WINDOW_NORMAL);
    spotter::Every report(2000);
    spotter::RgbdFrame frame;
    bool first = true;
    while (kinect.Next(frame))
    {
        if (first) PrintFirstFrame(frame, kinect);
        if (first || report.Due()) PrintCentre(frame);
        first = false;
        cv::imshow(Window, spotter::WithKeys(SideBySide(frame), Keys));
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}
