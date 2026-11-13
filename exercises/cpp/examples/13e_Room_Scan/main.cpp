// =====================================================================
//  13e · Room Scan
//
//  Our most advanced room scanner: use this one for real rooms. Scan the
//  room from many spots; each scan is placed by itself. The scans stay in
//  memory (nothing is written while you scan) and the folder is written in
//  one go on K, on Q, when the Kinect is lost, or on any error (D39).
//
//    1. Folder first  a Save As window before the windows open (it starts in the
//                     repo's room_scans/, the name room_<date>_<time>). The folder
//                     is checked: not inside build/, writable, room on the disk
//                     (D48). Its full path stays on the camera window, and is
//                     checked again before every scan.
//    2. SPACE scans   hold still: 60 frames averaged (07), then its colour
//                     features (11b, near and far: 13d) and depth make a Scan.
//    3. Placed        in the background, against the earlier scans: features
//                     give a rough pose, ICP polishes it, a second ICP against
//                     the rest of the room checks it (spotter::PlaceScan).
//    4. Undo, fix     U takes the last scan back, X excludes a scan, P places
//                     it again: all in memory.
//    5. Live anchors  about twice a second: what the next scan would be placed
//                     by (green near, magenta far), and the next shot to take.
//    6. E / K         E asks where (Save As, in the room's folder) and merges
//                     everything into room_scan_<date>_<time>.obj (TSDF +
//                     surface net + the clean model: spotter::FuseScans and
//                     BuildRoom). K asks where (Save As: keep the folder or
//                     choose another) and saves it (every scan, pose.txt last,
//                     then points.ply). A green banner says where each went.
//                     Q, a lost Kinect and any error save with no questions; if
//                     the folder fails, the rescue folders are tried in turn.
//
//  The scanning session (what it holds, one function per key) is in
//  scan_state.hpp, next to this file.
//
//  Changed from legacy 13e: left and right are right (room folder format 2);
//  an error mid-scan saves the folder instead of losing it (the review's
//  bug 6); placing a scan again leaves out its own landmarks (bug 14); the
//  folder is flushed to disk before each pose.txt (bug 4); no saving as you
//  go (the user, 2026-11-07: autosave was a stopgap; D39). Since D48: Save As windows
//  instead of console questions, the folder checks, and E saves where you choose
//  (it used to write into the folder the program ran from).
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
//  their colour pictures. E writes the .obj you name. Both are captures of
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
const std::string Keys = "SPACE scan here   U undo the last scan   E export (Save As)   K save the folder (Save As)   click a camera circle to select it: X exclude / include   P place again   Esc deselect   C colour by scan   mouse drag / W A S D turn   wheel / Z + - zoom   R reset view   Q quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

const int AverageFrames = 60;          // per scan: noise falls as 1 / sqrt(frames) (07)
const float MaxDepthMeters = 6.0f;
const std::string SaveFolder = SPOTTER_REPO_ROOT "/room_scans";   // where the Save As window starts: the repo's room_scans/ (gitignored, outside build/)
const int LiveEveryMs = 500;           // the live anchors, this often
const int ShowStep = 2;                // the viewer draws every 2nd pixel of each scan...
const size_t MaxShownPoints = 600000;  // ...and at most this many points in all
const float PivotMeters = 2.5f;

//spotter::ScanOptions ScanSettings(){
//
//@brief - 13e's placing settings (ScanOptions' defaults) with MaxDepthMeters set. O(1).
spotter::ScanOptions ScanSettings() { spotter::ScanOptions o; o.maxDepthMeters = MaxDepthMeters; return o; }
//spotter::RoomOptions RoomSettings(){
//
//@brief - 13e's merging and clean-model settings (RoomOptions' defaults). O(1).
spotter::RoomOptions RoomSettings() { return spotter::RoomOptions(); }

const std::string Viewer = "13e room scan - the room so far  (drag/WASD turn, wheel zoom, click a camera to select)";
const std::string CameraWindow = "13e room scan - camera  (SPACE scan, U undo, X exclude, P place again, E export, K save, Q quit)";

// ---------------------------------------------------------------------
//  Keys
// ---------------------------------------------------------------------

//void HandleKey(Session& s, int key, spotter::OrbitView& view){
//
//@param1 - s: the session
//@param2 - key: what cv::waitKey returned
//@param3 - view: the viewer, for clicks, zoom and its own keys
//
//@brief - a click in the viewer selects a camera; SPACE checks the folder and scans (when idle), U undoes, X excludes, P places again, E
//         exports, K saves, C colours by scan, Esc deselects, + - zoom, anything else goes to the viewer (W A S D,
//         Z, R). O(1).
void HandleKey(Session& s, int key, spotter::OrbitView& view)
{
    cv::Point click;
    if (view.TakeClick(click)) OnClick(s, view, click);
    switch (key)
    {
    case ' ': OnSpace(s); break;
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

//void ShowRoom(const Session& s, spotter::OrbitView& view, spotter::PointRenderer& renderer){
//
//@param1 - s: the session
//@param2 - view: the viewer's angles
//@param3 - renderer: draws the points
//
//@brief - the room so far: every scan's points (every n-th so no more than MaxShownPoints; excluded grey, or one
//         colour per scan with C), each scan's camera (selected white, excluded grey, suspicious red), the
//         recommended next camera in green, the far landmarks in magenta, and a count line; shown with the key
//         strip. O(shown points).
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

//std::string Tail(const std::string& text, size_t most){
//
//@param1 - text: a line, e.g. a long path
//@param2 - most: the most characters that fit
//
//@brief - the line, or "..." and its end if it's too long (a path's end is the part that tells folders apart). O(length).
std::string Tail(const std::string& text, size_t most)
{
    return text.size() <= most ? text : "..." + text.substr(text.size() - (most - 3));
}

//void ShowWhereItGoes(const Session& s, cv::Mat& display){
//
//@param1 - s: the session
//@param2 - display: the camera picture, drawn on
//
//@brief - D48: the room folder's full path and its room in scans (always), a red warning while the last check
//         says the folder can't take the scans, and K's or E's banner (green SAVED / EXPORTED, red FAILED) for 8
//         seconds after. O(1).
void ShowWhereItGoes(const Session& s, cv::Mat& display)
{
    cv::putText(display, cv::format("Room folder: %s   (room for ~%d scans)", Tail(s.folder.string(), 80).c_str(), s.scansThatFit),
                cv::Point(10, display.rows - 42), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    if (!s.folderProblem.empty())
    {
        cv::putText(display, "CAN'T SAVE TO THE ROOM FOLDER - press K to choose another", cv::Point(10, 195), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
        cv::putText(display, Tail(s.folderProblem, 110), cv::Point(10, 220), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255), 1);
    }
    if (s.banner.empty() || std::chrono::steady_clock::now() > s.bannerUntil) return;
    cv::rectangle(display, cv::Rect(0, 240, display.cols, 50), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(display, Tail(s.banner, 68), cv::Point(10, 274), cv::FONT_HERSHEY_SIMPLEX, 0.75,
                s.bannerOk ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
}

//void ShowCamera(const Session& s, const spotter::RgbdFrame& frame, const std::string& warning){
//
//@param1 - s: the session
//@param2 - frame: this frame
//@param3 - warning: a red message across the picture (default none)
//
//@brief - the half-size colour picture with the live anchors (green near, magenta far, grey the rest) and their
//         count in red / yellow / green against what's needed, the next-shot advice, the selected scan's keys, a
//         status line (PLACING, HOLD STILL n / N or the scan count), the message, the warning and where the room
//         goes (ShowWhereItGoes); shown with the key strip. O(pixels + features).
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
    ShowWhereItGoes(s, display);
    cv::imshow(CameraWindow, spotter::WithKeys(display, Keys));
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

//bool Start(Session& s, spotter::KinectSource& kinect, spotter::RgbdFrame& frame){
//
//@param1 - s: the session; its folder and camera info are filled
//@param2 - kinect: the open Kinect
//@param3 - frame: set to the first frame
//
//@brief - the folder (asked first in a Save As window and checked, AskRoomFolder), then the first frame for the
//         cameras' numbers and the device's calibration text. False, with the reason printed, if the answer was
//         cancel or no frame came. O(1).
bool Start(Session& s, spotter::KinectSource& kinect, spotter::RgbdFrame& frame)
{
    s.roomsFolder = SaveFolder;
    s.folder = spotter::AskRoomFolder(SaveFolder);
    if (s.folder.empty()) { std::printf("No folder chosen, so nothing can be saved: not starting.\n"); return false; }
    if (!kinect.Next(frame)) { std::printf("The Kinect gave no first frame: not starting.\n"); return false; }
    CheckFolderNow(s);
    s.camera = s.info.depthCamera = frame.depthCamera;
    s.info.colourSize = frame.colour.size();
    s.info.program = "13e_Room_Scan";
    s.info.calibration = kinect.CalibrationText();
    std::printf("\n=================================================================\nThe room will be saved to\n  %s\n"
                "(on K, Q, a lost Kinect or an error; room for about %d scans there)\n=================================================================\n%s\n",
                s.folder.string().c_str(), s.scansThatFit, s.message.c_str());
    return true;
}

//bool NeedBigdepth(const Session& s){
//
//@param1 - s: the session
//
//@brief - true on the last frame of a scan's average (the full colour picture's depth is needed to find far
//         features) and whenever a live check is due. O(1).
bool NeedBigdepth(const Session& s) { return s.averager.Added() == s.averager.Frames() - 1 || LiveCheckDue(s); }

//int main(){
//
//@brief - the program: 13e · Room Scan. Our most advanced room scanner: use this one for real rooms. Scan the room
//         from many spots; each scan is placed by itself. Setup, main loop and clean-up are laid out in the file's
//         sections. Returns 0 on a normal exit, 1 if it couldn't start.
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
