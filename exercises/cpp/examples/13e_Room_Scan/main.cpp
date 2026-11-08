// =====================================================================
//  13e · Room Scan
//
//  Our most advanced room scanner: use this one for real rooms. Scan the
//  room from many spots; each scan is placed by itself. The scans stay in
//  memory (nothing is written while you scan) and the folder is written in
//  one go on K, on Q, when the Kinect is lost, or on any error (D39).
//
//    1. Folder first  asked in the console before the windows open (Enter =
//                     room_scans, then Enter = room_<date>_<time>).
//    2. SPACE scans   hold still: 60 frames averaged (07), then its colour
//                     features (11b, near and far: 13d) and depth make a Scan.
//    3. Placed        in the background, against the earlier scans: features
//                     give a rough pose, ICP polishes it, a second ICP against
//                     the rest of the room checks it (spotter::PlaceScan).
//    4. Undo, fix     U takes the last scan back, X excludes a scan, P places
//                     it again: all in memory.
//    5. Live anchors  about twice a second: what the next scan would be placed
//                     by (green near, magenta far), and the next shot to take.
//    6. E / K         E merges everything into room_scan_<time>.obj (TSDF +
//                     surface net + the clean model: spotter::FuseScans and
//                     BuildRoom). K saves the folder (every scan, pose.txt
//                     last, then points.ply); Q, a lost Kinect and any error
//                     do it too.
//
//  The scanning session (what it holds, one function per key) is in
//  scan_state.hpp, next to this file.
//
//  Changed from legacy 13e: left and right are right (room folder format 2);
//  an error mid-scan saves the folder instead of losing it (the review's
//  bug 6); placing a scan again leaves out its own landmarks (bug 14); the
//  folder is flushed to disk before each pose.txt (bug 4); no saving as you
//  go (the user, 2026-11-07: autosave was a stopgap; D39).
//
//  Window 1 (viewer): every scan placed in the room. Circles + 4 arrows =
//            where each camera stood: yellow = fine, red = suspicious, grey =
//            excluded, white = selected, green = the next shot. Magenta dots =
//            far landmarks. C colours each scan differently.
//  Window 2 (camera): what the Kinect sees, the live anchors, what to do next.
//  Keys:     SPACE scan here   U undo the last scan   E export   K save the folder
//            click a camera circle to select it: X exclude / include   P place again
//            Esc deselect   C colour by scan
//            mouse drag / W A S D turn   wheel / Z + - zoom   R reset view   Q quit
//
//  Privacy: the scans are written to the folder you name at the start, with
//  their colour pictures. E writes room_scan_<time>.obj. Both are captures of
//  the room in colour: *.obj, *.ply and room_scans/ are gitignored, and the
//  folder gets its own .gitignore. Never commit them. Scan with nobody in view.
// =====================================================================

#include <cstdio>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/source_freenect2.hpp"
#include "spotter/track.hpp"   // ColourForId
#include "spotter/view.hpp"
#include "scan_state.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "SPACE scan here   U undo the last scan   E export   K save the folder   click a camera circle to select it: X exclude / include   P place again   Esc deselect   C colour by scan   mouse drag / W A S D turn   wheel / Z + - zoom   R reset view   Q quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

const int AverageFrames = 60;          // per scan: noise falls as 1 / sqrt(frames) (07)
const float MaxDepthMeters = 6.0f;
const std::string SaveFolder = "room_scans";   // Enter at the first question; relative = where it runs
const int LiveEveryMs = 500;           // the live anchors, this often
const int ShowStep = 2;                // the viewer draws every 2nd pixel of each scan...
const size_t MaxShownPoints = 600000;  // ...and at most this many points in all
const float PivotMeters = 2.5f;

// Placing, merging and the clean model: spotter::ScanOptions and RoomOptions hold 13e's values
// (room_scan.hpp, room.hpp). Change one here, for example:
spotter::ScanOptions ScanSettings() { spotter::ScanOptions o; o.maxDepthMeters = MaxDepthMeters; return o; }
spotter::RoomOptions RoomSettings() { return spotter::RoomOptions(); }

const std::string Viewer = "13e room scan - the room so far  (drag/WASD turn, wheel zoom, click a camera to select)";
const std::string CameraWindow = "13e room scan - camera  (SPACE scan, U undo, X exclude, P place again, E export, K save, Q quit)";

// ---------------------------------------------------------------------
//  Keys
// ---------------------------------------------------------------------

void HandleKey(Session& s, int key, spotter::OrbitView& view)
{
    cv::Point click;
    if (view.TakeClick(click)) OnClick(s, view, click);
    switch (key)
    {
    case ' ': if (Idle(s)) s.averager.Start(); break;
    case 'u': OnUndo(s); break;
    case 'x': OnExclude(s); break;   // not the viewer's zoom here
    case 'p': OnPlaceAgain(s); break;
    case 'e': OnExport(s); break;
    case 'k': OnSave(s); break;
    case 'c': s.colourByScan = !s.colourByScan; break;
    case 27: s.selected = -1; break;
    case '+': case '=': view.distance -= view.options.zoomStepMeters; break;
    case '-': view.distance += view.options.zoomStepMeters; break;
    default: view.HandleKey(key);
    }
}

// ---------------------------------------------------------------------
//  Output: the room so far, and the camera with instructions
// ---------------------------------------------------------------------

void ShowRoom(const Session& s, spotter::OrbitView& view, spotter::PointRenderer& renderer)
{
    cv::Mat& image = renderer.Begin(view);
    size_t total = 0;
    for (const auto& points : s.shownPoints) total += points.size();
    const size_t every = total / MaxShownPoints + 1;
    int excluded = 0;
    for (size_t i = 0; i < s.scans.size(); i++)
    {
        const spotter::Scan& scan = s.scans[i];
        excluded += scan.excluded;
        if (scan.excluded) renderer.Draw(s.shownPoints[i], cv::Vec3b(90, 90, 90), 2, every);
        else if (s.colourByScan) { cv::Scalar c = spotter::ColourForId((int)i + 1); renderer.Draw(s.shownPoints[i], cv::Vec3b((uchar)c[0], (uchar)c[1], (uchar)c[2]), 2, every); }
        else renderer.Draw(s.shownPoints[i], s.shownColours[i], 2, every);
    }
    for (size_t i = 0; i < s.scans.size(); i++)
    {
        const spotter::Scan& scan = s.scans[i];
        const bool chosen = (int)i == s.selected;
        cv::Scalar colour = chosen ? cv::Scalar(255, 255, 255) : scan.excluded ? cv::Scalar(110, 110, 110)
                          : scan.suspicious ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 255);
        spotter::DrawCamera(image, view, scan.pose, s.camera, colour, chosen ? 2 : 1, std::to_string(i + 1));
    }
    if (s.next.found && !Busy(s)) spotter::DrawCamera(image, view, s.next.pose, s.camera, cv::Scalar(0, 255, 0), 2, "next");
    for (const spotter::Landmark& l : s.landmarks)
    {
        cv::Point pixel;
        float depth;
        if (view.Project(l.point, pixel, depth)) cv::circle(image, pixel, 3, cv::Scalar(255, 0, 255), cv::FILLED);
    }
    cv::putText(image, cv::format("%zu scans (%d excluded), %zu far landmarks", s.scans.size(), excluded, s.landmarks.size()),
                cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
    cv::imshow(Viewer, spotter::WithKeys(image, Keys));
}

void ShowCamera(const Session& s, const spotter::RgbdFrame& frame, const std::string& warning = "")
{
    cv::Mat display;
    cv::resize(frame.colour, display, frame.colour.size() / 2);
    const int need = s.scanOptions.minFeatureMatches;
    if (Idle(s) && !s.scans.empty())   // the live anchors: the next scan is placed by the green ones
    {
        for (const cv::Point2f& p : s.live.otherPixels) cv::circle(display, p * 0.5f, 2, cv::Scalar(140, 140, 140), cv::FILLED);
        for (const cv::Point2f& p : s.live.anchorPixels) cv::circle(display, p * 0.5f, 4, cv::Scalar(0, 255, 0), cv::FILLED);
        for (const cv::Point2f& p : s.live.farAnchorPixels) cv::circle(display, p * 0.5f, 4, cv::Scalar(255, 0, 255), cv::FILLED);
        cv::Scalar countColour = s.live.anchors < need ? cv::Scalar(0, 0, 255) : s.live.anchors < 2 * need ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0);
        std::string count = s.live.scan >= 0 ? cv::format("anchors: %d (need %d; %d far) - matches scan %d", s.live.anchors, need, s.live.farAnchors, s.live.scan + 1)
                                             : cv::format("anchors: 0 (need %d) - aim at something already scanned", need);
        cv::putText(display, count, cv::Point(10, 65), cv::FONT_HERSHEY_SIMPLEX, 0.7, countColour, 2);
        cv::putText(display, s.next.text, cv::Point(10, 95), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 1);
    }
    if (s.selected >= 0)
        cv::putText(display, cv::format("Selected scan %d%s: X %s   P place again   Esc deselect", s.selected + 1, s.scans[s.selected].excluded ? " (excluded)" : "",
                    s.scans[s.selected].excluded ? "include" : "exclude"), cv::Point(10, 125), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
    std::string top = Busy(s) ? std::string("PLACING... (the windows keep working)")
                    : s.averager.Running() ? cv::format("HOLD STILL  %d / %d", s.averager.Added(), s.averager.Frames())
                                           : cv::format("%zu scans%s   SPACE scan here   U undo   E export   K save   Q quit", s.scans.size(), s.saved || s.scans.empty() ? "" : " (not saved yet)");
    cv::putText(display, top, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    cv::putText(display, s.message, cv::Point(10, display.rows - 15), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 1);
    if (!warning.empty()) cv::putText(display, warning, cv::Point(10, 160), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
    cv::imshow(CameraWindow, spotter::WithKeys(display, Keys));
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

// The folder (asked first), then the first frame for the camera's numbers.
bool Start(Session& s, spotter::KinectSource& kinect, spotter::RgbdFrame& frame)
{
    s.folder = spotter::AskRoomFolder(SaveFolder);
    if (s.folder.empty() || !kinect.Next(frame)) return false;
    s.camera = s.info.depthCamera = frame.depthCamera;
    s.info.colourSize = frame.colour.size();
    s.info.program = "13e_Room_Scan";
    s.info.calibration = kinect.CalibrationText();
    std::printf("The room will be saved to %s (on K, Q, a lost Kinect or an error).\n%s\n", s.folder.string().c_str(), s.message.c_str());
    return true;
}

// Bigdepth (depth per colour pixel, for the far features) only on the frames that need it.
bool NeedBigdepth(const Session& s) { return s.averager.Added() == s.averager.Frames() - 1 || LiveCheckDue(s); }

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    spotter::KinectSource::Options kinectOptions;
    kinectOptions.maxDepthMeters = MaxDepthMeters;
    spotter::KinectSource kinect(kinectOptions);
    Session s(AverageFrames);
    s.scanOptions = ScanSettings();
    s.roomOptions = RoomSettings();
    s.liveEveryMs = LiveEveryMs;
    s.showStep = ShowStep;
    spotter::RgbdFrame frame;
    if (!kinect.IsOpen() || !Start(s, kinect, frame)) { std::printf("Nothing scanned.\n"); return 1; }

    cv::namedWindow(Viewer);
    cv::namedWindow(CameraWindow);
    spotter::OrbitView view(spotter::OrbitView::Options(), cv::Point3f(0, 0, PivotMeters), PivotMeters);
    view.Attach(Viewer);
    spotter::PointRenderer renderer;
    std::string ending = "Quitting";
    try
    {
        while (true)
        {
            int key = cv::waitKey(1);
            if (key == 'q') break;
            kinect.SetBigdepth(NeedBigdepth(s));
            if (!kinect.Next(frame)) { ending = "Kinect lost (no frame for 10 s)"; ShowCamera(s, frame, "KINECT LOST - saving the folder"); cv::waitKey(1); break; }
            CollectJobs(s);
            HandleKey(s, key, view);
            if (s.averager.Running() && s.averager.Add(frame.depthMm)) OnScanReady(s, frame, s.averager.Mean());
            StartLiveCheck(s, frame);
            ShowRoom(s, view, renderer);
            ShowCamera(s, frame);
        }
    }
    catch (const std::exception& e) { ending = std::string("An error (") + e.what() + ")"; }   // bug 6: the folder is saved anyway
    EndSession(s, ending);   // before the Kinect stops: stopping one that is gone may hang
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Scan a room in 10-15 spots, following the Next line. Then press C: does any scan show as a
//     shifted copy? Click its camera, X to exclude it, E again.
//  2. Unplug the Kinect mid-session (not mid-scan). The console says "Kinect lost" and saves the
//     folder; open it with 13z.
//  3. Turn the lights down: the near anchors fade first, the far (magenta) ones need light too.
//  4. In a hallway, aim at the far end: 13d's far landmarks (magenta dots in the viewer) appear
//     once two spots have seen the same far detail at a clear angle.
//  5. ScanSettings: minFeatureMatches 20 -> 40. Fewer scans placed, but fewer misplaced?
