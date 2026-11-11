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

//bool IsQuitKey(int key){
//
//@param1 - key: what cv::waitKey returned
//
//@brief - true for q or Esc. O(1).
inline bool IsQuitKey(int key) { return key == 'q' || key == 27; }

//cv::Mat WithKeys(const cv::Mat& image, const std::string& keys){
//
//@param1 - image: the picture to show (any depth; grey, colour or BGRA)
//@param2 - keys: the program's keys, entries separated by 3 or more spaces, e.g. "SPACE  scan   Q or Esc  quit"
//
//@brief - a copy of `image` with the keys in a strip added below it (the picture itself isn't covered or
//         changed); a long list wraps between entries. Grey or float pictures are shown as colour, as
//         imshow would show them. Use it in every imshow: cv::imshow(window, spotter::WithKeys(image, Keys)).
//         O(pixels).
cv::Mat WithKeys(const cv::Mat& image, const std::string& keys);

//void OpenExhibitWindow(const std::string& window, cv::Size size, bool fullscreen){
//
//@param1 - window: the window's name
//@param2 - size: its starting size, before fullscreen
//@param3 - fullscreen: true = fullscreen at the start
//
//@brief - the demos' window: resizable, the picture scales to fit. O(1).
void OpenExhibitWindow(const std::string& window, cv::Size size, bool fullscreen);

//void ToggleFullscreen(const std::string& window){
//
//@param1 - window: the window's name
//
//@brief - fullscreen <-> a normal window (F in the demos). O(1).
void ToggleFullscreen(const std::string& window);

// True once every `milliseconds`: `if (report.Due()) Report(...)` prints about once a second.
class Every
{
public:
    //Every(int milliseconds){
    //
    //@param1 - milliseconds: the period
    //
    //@brief - the first period starts now. O(1).
    explicit Every(int milliseconds);

    //bool Due(){
    //
    //@brief - true at most once per period (the next period starts then); the first call is true after
    //         one period. O(1).
    bool Due();

private:
    double periodSeconds, nextSeconds;
};

// The live frame rate, smoothed so it doesn't flicker.
class FpsMeter
{
public:
    //FpsMeter(double smoothing){
    //
    //@param1 - smoothing: each Tick moves the rate (1 - smoothing) of the way toward the newest value;
    //          0 = raw and jumpy, 0.9 = steady (default), 0.99 = slow to react
    //
    //@brief - a meter showing 0 until the second Tick. O(1).
    explicit FpsMeter(double smoothing = 0.9) : smoothing(smoothing) {}

    //double Tick(){
    //
    //@brief - call once per shown frame: 1 / the time since the last Tick, smoothed (the first reading is
    //         taken as is). Returns the smoothed fps. O(1).
    double Tick();

    //double Fps(){
    //
    //@brief - the smoothed fps as of the last Tick. O(1).
    double Fps() const { return fps; }

    //void Draw(cv::Mat& image, const std::string& extra){
    //
    //@param1 - image: the picture to write on
    //@param2 - extra: text added after the number (default none)
    //
    //@brief - writes "24.3 fps<extra>" at the bottom left in yellow, sized to the picture. O(text pixels).
    void Draw(cv::Mat& image, const std::string& extra = "") const;

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

    //OrbitView(const Options& options, cv::Point3f pivot, float distance){
    //
    //@param1 - options: the picture size, focal length and step sizes
    //@param2 - pivot: the point it turns around and looks at
    //@param3 - distance: how far behind the pivot it starts (R goes back to this)
    //
    //@brief - the view starts where the camera is, looking at the pivot from `distance` behind it. O(1).
    OrbitView(const Options& options, cv::Point3f pivot, float distance);

    //void Attach(const std::string& window){
    //
    //@param1 - window: an open window's name
    //
    //@brief - the mouse turns (drag) and zooms (wheel) this view in that window, and clicks are kept for
    //         TakeClick. O(1).
    void Attach(const std::string& window);

    //bool HandleKey(int key){
    //
    //@param1 - key: what cv::waitKey returned
    //
    //@brief - W S tilt, A D turn, Z X zoom, R resets. True if it used the key. O(1).
    bool HandleKey(int key);

    //void Reset(){
    //
    //@brief - back to the start: no turn or tilt, the starting distance. O(1).
    void Reset();

    //bool Project(const cv::Point3f& point, cv::Point& pixel, float& depth){
    //
    //@param1 - point: a 3D point, camera coordinates
    //@param2 - pixel: set to the viewer pixel it lands on
    //@param3 - depth: set to how far in front of the viewer it is
    //
    //@brief - moves the pivot to the centre, turns about the vertical, tilts about the sideways axis,
    //         steps back `distance`, then the pinhole projection. False if it's behind the viewer, off the
    //         picture, or NaN. Uses the angles as of the last Update() (PointRenderer::Begin calls it). O(1).
    bool Project(const cv::Point3f& point, cv::Point& pixel, float& depth) const;

    //void Update(){
    //
    //@brief - after changing the angles by hand: clamps the tilt to +-89 degrees and the distance to at least
    //         0.1 m, and works out the sines and cosines Project uses. O(1).
    void Update();

    //bool TakeClick(cv::Point& where){
    //
    //@param1 - where: set to the clicked pixel
    //
    //@brief - a click (pressed and released within 4 pixels, not a drag) since the last call: true once,
    //         with where. O(1).
    bool TakeClick(cv::Point& where);

    Options options;
    cv::Point3f pivot;
    float yawDegrees = 0, pitchDegrees = 0, distance;

private:
    //static void OnMouse(int event, int x, int y, int flags, void* view){
    //
    //@param1 - event: OpenCV's mouse event (button down/up, move, wheel)
    //@param2 - x: the mouse's column in the window
    //@param3 - y: its row
    //@param4 - flags: OpenCV's flags (the wheel's direction)
    //@param5 - view: the OrbitView that Attach registered
    //
    //@brief - OpenCV's mouse callback: a drag turns and tilts, the wheel zooms, a press and release in
    //         nearly one place is kept as a click. O(1).
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
    //cv::Mat& Begin(OrbitView& view, cv::Scalar background){
    //
    //@param1 - view: the view to draw through; Update() is called on it
    //@param2 - background: the picture's colour (default dark grey)
    //
    //@brief - clears the picture and the z-buffer to the view's size; returns the picture. O(pixels).
    cv::Mat& Begin(OrbitView& view, cv::Scalar background = cv::Scalar(30, 30, 30));

    //void Draw(const std::vector<cv::Point3f>& points, const std::vector<cv::Vec3b>& colours, int sizePixels, size_t every){
    //
    //@param1 - points: the points; NaN ones are skipped
    //@param2 - colours: one BGR colour per point
    //@param3 - sizePixels: each point's square size (default 2)
    //@param4 - every: draw every n-th point (default 1 = all)
    //
    //@brief - each point projected and drawn where it's nearer than what's there. O(n / every x sizePixels^2).
    void Draw(const std::vector<cv::Point3f>& points, const std::vector<cv::Vec3b>& colours, int sizePixels = 2, size_t every = 1);

    //void Draw(const std::vector<cv::Point3f>& points, cv::Vec3b colour, int sizePixels, size_t every){
    //
    //@param1 - points: the points; NaN ones are skipped
    //@param2 - colour: one BGR colour for all of them
    //@param3 - sizePixels: each point's square size (default 2)
    //@param4 - every: draw every n-th point (default 1 = all)
    //
    //@brief - the same in one colour. O(n / every x sizePixels^2).
    void Draw(const std::vector<cv::Point3f>& points, cv::Vec3b colour, int sizePixels = 2, size_t every = 1);

    //cv::Mat& Image(){
    //
    //@brief - the picture drawn so far. O(1).
    cv::Mat& Image() { return image; }

private:
    //void Splat(const cv::Point3f& point, const cv::Vec3b& colour, int sizePixels){
    //
    //@param1 - point: one 3D point
    //@param2 - colour: its BGR colour
    //@param3 - sizePixels: the square's size
    //
    //@brief - projects the point and fills a square of pixels where it's nearer than the z-buffer. O(sizePixels^2).
    void Splat(const cv::Point3f& point, const cv::Vec3b& colour, int sizePixels);
    OrbitView* view = nullptr;
    cv::Mat image, nearest;
};

//void DrawCamera(cv::Mat& image, const OrbitView& view, const Pose& pose, const PinholeCamera& camera, const cv::Scalar& colour, int thickness, const std::string& label, float arrowMeters){
//
//@param1 - image: the 3D view's picture
//@param2 - view: the view it was drawn through
//@param3 - pose: that camera -> the view's coordinates
//@param4 - camera: its pinhole model, for the corners of what it saw
//@param5 - colour: BGR
//@param6 - thickness: line thickness in pixels
//@param7 - label: text beside it
//@param8 - arrowMeters: how far ahead the arrows reach (default 0.6)
//
//@brief - where a camera stood: a circle, four arrows to the corners of what it saw, and a label. Nothing
//         if the camera itself is out of view. O(1).
void DrawCamera(cv::Mat& image, const OrbitView& view, const Pose& pose, const PinholeCamera& camera, const cv::Scalar& colour,
                int thickness, const std::string& label, float arrowMeters = 0.6f);

}
