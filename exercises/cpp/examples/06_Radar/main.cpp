// =====================================================================
//  06 · Radar
//
//  Examples 02-05c say HOW FAR away each object is. This one adds WHERE:
//  every tracked object becomes a position on the floor, in metres, drawn
//  on a bird's-eye "radar" map with the camera at the bottom. It's the
//  first map in the project, and the first step toward a 3D map of the room.
//
//    1-5.  Detect (in the background), track, filter   as in 05c, from the library
//    6.    Deproject   filtered box centre + filtered distance -> X, Z in metres,
//                      with the colour camera's factory intrinsics (§5.3):
//                          X = (u + 0.5 - cx) * Z / fx   (left/right of the camera)
//                          Z = distance                  (straight ahead)
//    7.    Map         draw X, Z from above, plus a trail of where it has been.
//
//  Windows: the camera view (as in 05c), and the radar: camera at the bottom
//           centre, a ring every metre, the edges of the camera's view, one
//           dot per object (hollow = predicted only), trails behind them.
//  Console: each object's X, Z and straight-line distance.
//  Keys:   C  clear the trails      Q or Esc  quit
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/track.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "C  clear the trails      Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

const double RadarRangeMeters = 6.0;   // the Kinect's depth reaches about 6 m
const int RadarSizePixels = 700;
const int TrailLength = 150;           // past positions per object; at 30 fps, 5 s

// Only show these classes on the radar. Empty = show everything. Try { "person" }.
const std::vector<std::string> RadarClasses = {};

const int ReportEveryMs = 1000;
const std::string CameraWindow = "06 camera  (C clears trails, Q quits)", RadarWindow = "06 radar";

//spotter::YoloDetector::Options YoloSettings(){
//
//@brief - the detector's options: yolo26n.onnx and its names file from the repo, minimum confidence 0.15 (the
//         tracker's weak threshold, 05b). O(1).
spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = 0.15f;   // the tracker's weak threshold (05b)
    return options;
}

// ---------------------------------------------------------------------
//  The new idea: from a tracked box to a spot on the floor, drawn from above
// ---------------------------------------------------------------------

using Trails = std::map<int, std::deque<cv::Point2d>>;   // per ID: X, Z in metres, oldest first

//cv::Point2d FloorPosition(const spotter::Track& track, const spotter::PinholeCamera& colourCamera){
//
//@param1 - track: a tracked object with a distance
//@param2 - colourCamera: the colour camera's model
//
//@brief - step 6: the pinhole model on the filtered estimate: the box centre's distance from the picture's centre,
//         times distance / focal length, is the metres left/right of the camera. Returns (X right, Z ahead).
//         ponytail: the colour camera is the origin (it sits a few cm beside the depth camera), small next to box
//         jitter; example 09 adds the floor. O(1).
cv::Point2d FloorPosition(const spotter::Track& track, const spotter::PinholeCamera& colourCamera)
{
    cv::Point3f p = colourCamera.Unproject(cv::Point2f(track.filter.Centre()), (float)track.filter.Meters());
    return { p.x, p.z };
}

//void UpdateTrails(Trails& trails, const spotter::Tracker& tracker, const spotter::PinholeCamera& colourCamera){
//
//@param1 - trails: each ID's floor positions, oldest first; updated
//@param2 - tracker: the tracked objects
//@param3 - colourCamera: the colour camera's model
//
//@brief - adds each tracked object's floor position to its trail (once it has a distance), keeps the last
//         TrailLength, and forgets the IDs the tracker forgot. O(tracks log tracks).
void UpdateTrails(Trails& trails, const spotter::Tracker& tracker, const spotter::PinholeCamera& colourCamera)
{
    Trails kept;   // only the IDs still tracked
    for (const spotter::Track& track : tracker.Tracks())
    {
        std::deque<cv::Point2d>& trail = kept[track.id] = std::move(trails[track.id]);
        if (track.filter.Meters() <= 0) continue;   // no distance yet, so no position
        trail.push_back(FloorPosition(track, colourCamera));
        if ((int)trail.size() > TrailLength) trail.pop_front();
    }
    trails.swap(kept);
}

//bool ShownOnRadar(int classId, const spotter::ClassList& classes){
//
//@param1 - classId: a track's class number
//@param2 - classes: the class names
//
//@brief - true if the RadarClasses list is empty or holds this class's name. O(RadarClasses).
bool ShownOnRadar(int classId, const spotter::ClassList& classes)
{
    return RadarClasses.empty() || std::find(RadarClasses.begin(), RadarClasses.end(), classes.Name(classId)) != RadarClasses.end();
}

// The radar's layout: the camera at the bottom centre, looking up the screen.
struct Radar
{
    double pixelsPerMeter = (RadarSizePixels - 40) / RadarRangeMeters;
    cv::Point camera{ RadarSizePixels / 2, RadarSizePixels - 20 };
    //cv::Point ToPixel(cv::Point2d floor){
    //
    //@param1 - floor: a spot on the floor, (X right, Z ahead) in metres
    //
    //@brief - the radar pixel for that spot: the camera is at the bottom centre and Z runs up the map. O(1).
    cv::Point ToPixel(cv::Point2d floor) const
    {
        return { (int)std::lround(camera.x + floor.x * pixelsPerMeter), (int)std::lround(camera.y - floor.y * pixelsPerMeter) };
    }
};

//void DrawRadar(const Trails& trails, const spotter::Tracker& tracker, const spotter::ClassList& classes, const spotter::PinholeCamera& colourCamera){
//
//@param1 - trails: each ID's floor positions
//@param2 - tracker: the tracked objects
//@param3 - classes: the class names
//@param4 - colourCamera: the colour camera's model, for its field of view
//
//@brief - step 7: a ring every metre, the camera's view edges (half the field of view from atan(width / 2 / fx)),
//         the 0.5 m minimum range, then each shown object's trail with a dot (hollow while predicted) and label;
//         shown with the key strip. O(pixels + trail points).
void DrawRadar(const Trails& trails, const spotter::Tracker& tracker, const spotter::ClassList& classes, const spotter::PinholeCamera& colourCamera)
{
    // Half the colour camera's horizontal field of view: the picture's edge is width/2 pixels from
    // the centre, at a focal length of fx pixels.
    double halfFovRadians = std::atan((colourCamera.width / 2.0) / colourCamera.fx);
    Radar radar;
    cv::Mat map(RadarSizePixels, RadarSizePixels, CV_8UC3, cv::Scalar(25, 25, 25));
    const cv::Scalar grey(80, 80, 80);
    for (int meters = 1; meters <= (int)RadarRangeMeters; meters++)
    {
        int radius = (int)(meters * radar.pixelsPerMeter);
        cv::ellipse(map, radar.camera, cv::Size(radius, radius), 0, 180, 360, grey, 1);
        cv::putText(map, std::to_string(meters) + " m", radar.camera + cv::Point(4, -radius - 4), cv::FONT_HERSHEY_SIMPLEX, 0.4, grey, 1);
    }
    for (int side : { -1, 1 })
        cv::line(map, radar.camera, radar.ToPixel({ side * RadarRangeMeters * std::sin(halfFovRadians), RadarRangeMeters * std::cos(halfFovRadians) }),
                 cv::Scalar(0, 120, 160), 1);
    int half = (int)(0.5 * radar.pixelsPerMeter);
    cv::ellipse(map, radar.camera, cv::Size(half, half), 0, 180, 360, cv::Scalar(0, 0, 140), 1);
    cv::drawMarker(map, radar.camera, cv::Scalar(0, 255, 255), cv::MARKER_TRIANGLE_UP, 14, 2);

    for (const spotter::Track& track : tracker.Tracks())
    {
        const std::deque<cv::Point2d>& trail = trails.at(track.id);
        if (trail.empty() || !ShownOnRadar(track.classId, classes)) continue;
        cv::Scalar colour = spotter::ColourForId(track.id);
        for (size_t i = 1; i < trail.size(); i++) cv::line(map, radar.ToPixel(trail[i - 1]), radar.ToPixel(trail[i]), colour, 1);
        cv::Point now = radar.ToPixel(trail.back());
        cv::circle(map, now, 7, colour, track.Coasting() ? 1 : cv::FILLED);   // hollow = predicted only
        cv::putText(map, "#" + std::to_string(track.id) + " " + classes.Name(track.classId), now + cv::Point(10, 4), cv::FONT_HERSHEY_SIMPLEX, 0.45, colour, 1);
    }
    cv::imshow(RadarWindow, spotter::WithKeys(map, Keys));
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

//void ShowCamera(cv::Mat& picture, const spotter::Tracker& tracker, const spotter::ClassList& classes, const spotter::FpsMeter& fps){
//
//@param1 - picture: the colour picture; drawn on
//@param2 - tracker: the tracked objects
//@param3 - classes: the class names
//@param4 - fps: the frame-rate meter
//
//@brief - a box per track in its ID's colour (thin while coasting), labelled with ID, class and distance, and the
//         fps; shown with the key strip. O(tracks).
void ShowCamera(cv::Mat& picture, const spotter::Tracker& tracker, const spotter::ClassList& classes, const spotter::FpsMeter& fps)
{
    for (const spotter::Track& track : tracker.Tracks())
    {
        cv::Scalar colour = spotter::ColourForId(track.id);
        std::string label = "#" + std::to_string(track.id) + " " + classes.Name(track.classId)
                          + (track.filter.Meters() > 0 ? cv::format(" %.2fm", track.filter.Meters()) : "");
        cv::rectangle(picture, track.Box(), colour, track.Coasting() ? 1 : 3);
        cv::putText(picture, label, cv::Point((int)track.Box().x, std::max((int)track.Box().y - 10, 30)), cv::FONT_HERSHEY_SIMPLEX, 1.0, colour, 2);
    }
    fps.Draw(picture);
    cv::imshow(CameraWindow, spotter::WithKeys(picture, Keys));
}

//void Report(const Trails& trails, const spotter::Tracker& tracker, const spotter::ClassList& classes, const spotter::FpsMeter& fps){
//
//@param1 - trails: each ID's floor positions
//@param2 - tracker: the tracked objects
//@param3 - classes: the class names
//@param4 - fps: the frame-rate meter
//
//@brief - console lines: fps and the object count, then each radar object's X (left or right), Z ahead and
//         straight-line distance. O(tracks).
void Report(const Trails& trails, const spotter::Tracker& tracker, const spotter::ClassList& classes, const spotter::FpsMeter& fps)
{
    std::printf("%.1f fps | %zu objects tracked\n", fps.Fps(), tracker.Tracks().size());
    for (const spotter::Track& track : tracker.Tracks())
    {
        const std::deque<cv::Point2d>& trail = trails.at(track.id);
        if (trail.empty() || !ShownOnRadar(track.classId, classes)) continue;
        cv::Point2d p = trail.back();
        std::printf("  #%-3d %-12s X %+.2f m (%s), Z %.2f m ahead | straight-line %.2f m%s\n", track.id, classes.Name(track.classId).c_str(),
                    p.x, p.x < 0 ? "left" : "right", p.y, std::hypot(p.x, p.y), track.Coasting() ? "  (predicted)" : "");
    }
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

//int main(){
//
//@brief - the program: 06 · Radar. Examples 02-05c say HOW FAR away each object is. This one adds WHERE: every
//         tracked object becomes a position on the floor, in metres, drawn on a bird's-eye "radar" map with the
//         camera at the bottom. Setup, main loop and clean-up are laid out in the file's sections. Returns 0 on a
//         normal exit, 1 if it couldn't start.
int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    spotter::AsyncDetector detector(yolo);   // 05c's background thread
    spotter::Tracker tracker(spotter::Tracker::Options{});
    spotter::BoxDepthSampler boxDepth;
    Trails trails;
    cv::namedWindow(CameraWindow, cv::WINDOW_NORMAL);
    cv::resizeWindow(CameraWindow, 960, 540);
    spotter::FpsMeter fps;
    spotter::Every report(ReportEveryMs);
    spotter::RgbdFrame frame;
    spotter::FrameTimer timer;

    while (kinect.Next(frame))
    {
        boxDepth.Build(frame);
        tracker.Predict(timer.Step(frame));
        if (auto batch = detector.TryTake()) tracker.Match(*batch, boxDepth);
        detector.Offer(frame.colour);
        tracker.Remeasure(boxDepth);
        UpdateTrails(trails, tracker, frame.colourCamera);                 // step 6
        fps.Tick();
        ShowCamera(frame.colour, tracker, yolo.Classes(), fps);
        DrawRadar(trails, tracker, yolo.Classes(), frame.colourCamera);   // step 7
        if (report.Due()) Report(trails, tracker, yolo.Classes(), fps);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 'c') for (auto& [id, trail] : trails) trail.clear();
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Walk to your left. Your dot moves left on the radar, as your box does in the camera window
//     (the picture is un-mirrored by KinectSource; the legacy 06 had them opposite).
//  2. Walk a square on the floor, about 1 m per side, and press C first. Does the trail come
//     out square? Where is it worst? (Far away, one pixel of box jitter is many cm.)
//  3. Stand at the edge of the view: your dot sits on the orange field-of-view line. Step past
//     it and YOLO loses you; the hollow "predicted" dot keeps going for a moment, slowing down.
//  4. Put two chairs 1 m apart (or anything of known size). How far apart are their dots?
//  5. Set RadarClasses = { "person" } to declutter the map.
//  6. Z is the distance straight ahead, not the straight-line distance (the console prints both).
//     The depth camera measures along its axis, not along the ray.
