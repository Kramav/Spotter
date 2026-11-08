// spotter/view.hpp: windows and drawing. A frame-rate meter, and a small 3D viewer for point
// clouds (no extra library): turn the points around a pivot, then project them onto a virtual
// pinhole camera, nearest point wins (example 08 explains the maths).
#pragma once

#include "spotter/frame.hpp"
#include "spotter/geometry.hpp"

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace spotter
{

// q or Esc.
inline bool IsQuitKey(int key) { return key == 'q' || key == 27; }

// A copy of `image` with the program's keys in a strip added below it (the picture itself isn't
// covered or changed). `keys` lists entries separated by 3 or more spaces, e.g.
// "SPACE  scan   Q or Esc  quit"; a long list wraps between entries. Grey or float pictures are
// shown as colour. Use it in every imshow: cv::imshow(window, spotter::WithKeys(image, Keys)).
cv::Mat WithKeys(const cv::Mat& image, const std::string& keys);

// True once every `milliseconds`: `if (report.Due()) Report(...)` prints about once a second.
class Every
{
public:
    explicit Every(int milliseconds);
    bool Due();   // true at most once per period; the first call is true after one period

private:
    double periodSeconds, nextSeconds;
};

// The live frame rate, smoothed so it doesn't flicker: each Tick() moves it (1 - smoothing) of
// the way toward the newest value. 0 = raw and jumpy, 0.9 = steady, 0.99 = slow to react.
class FpsMeter
{
public:
    explicit FpsMeter(double smoothing = 0.9) : smoothing(smoothing) {}
    double Tick();   // call once per shown frame; returns the smoothed fps
    double Fps() const { return fps; }
    void Draw(cv::Mat& image, const std::string& extra = "") const;   // "24.3 fps<extra>", bottom left, yellow

private:
    double smoothing, fps = 0, lastSeconds = -1;
};

// Where the 3D viewer looks from. Drag or W A S D turns it around the pivot, the wheel or Z X
// zooms, R resets. Points are in the camera's coordinates (metres; X right, Y down, Z ahead).
class OrbitView
{
public:
    struct Options
    {
        int width = 960, height = 720;   // the viewer picture
        float focalPixels = 700;         // 700 across 960 is about 69°, close to the depth camera's 70°
        float turnStepDegrees = 5;       // per W A S D press
        float zoomStepMeters = 0.25f;    // per Z X press or wheel notch
    };

    // The view starts where the camera is, looking at `pivot` from `distance` behind it.
    OrbitView(const Options& options, cv::Point3f pivot, float distance);

    void Attach(const std::string& window);   // the mouse turns and zooms this view in that window
    bool HandleKey(int key);                  // W A S D, Z X, R; true if it used the key
    void Reset();

    // A 3D point -> the viewer pixel it lands on, and how far in front of the viewer it is.
    // False if it's behind the viewer, off the picture, or NaN. Uses the angles as of the
    // last Update() (PointRenderer::Begin calls it).
    bool Project(const cv::Point3f& point, cv::Point& pixel, float& depth) const;
    void Update();   // after changing the angles by hand

    // A click (pressed and released within 4 pixels, not a drag) since the last call: true once, with where.
    bool TakeClick(cv::Point& where);

    Options options;
    cv::Point3f pivot;
    float yawDegrees = 0, pitchDegrees = 0, distance;

private:
    static void OnMouse(int event, int x, int y, int flags, void* view);
    float startDistance;
    float cosYaw = 1, sinYaw = 0, cosPitch = 1, sinPitch = 0;
    bool dragging = false, clicked = false;
    int lastX = 0, lastY = 0;
    cv::Point pressedAt, clickedAt;
};

// Draws point clouds into one picture through an OrbitView. Where two points land on the same
// pixel the nearer one wins (a z-buffer). Its buffers are kept between frames.
class PointRenderer
{
public:
    cv::Mat& Begin(OrbitView& view, cv::Scalar background = cv::Scalar(30, 30, 30));   // clears; returns the picture
    // Points with NaN are skipped. `colours` has one colour per point. every = 3 draws every 3rd point.
    void Draw(const std::vector<cv::Point3f>& points, const std::vector<cv::Vec3b>& colours, int sizePixels = 2, size_t every = 1);
    void Draw(const std::vector<cv::Point3f>& points, cv::Vec3b colour, int sizePixels = 2, size_t every = 1);
    cv::Mat& Image() { return image; }

private:
    void Splat(const cv::Point3f& point, const cv::Vec3b& colour, int sizePixels);
    OrbitView* view = nullptr;
    cv::Mat image, nearest;
};

// Where a camera stood, in a 3D view: a circle, four arrows to the corners of what it saw
// (arrowMeters ahead), and a label. pose = that camera -> the view's coordinates.
void DrawCamera(cv::Mat& image, const OrbitView& view, const Pose& pose, const PinholeCamera& camera, const cv::Scalar& colour,
                int thickness, const std::string& label, float arrowMeters = 0.6f);

}
