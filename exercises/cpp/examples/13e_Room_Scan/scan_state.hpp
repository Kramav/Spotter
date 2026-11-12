// 13e's scanning session: what it holds, and one function per thing that happens to it (a key, a
// finished scan, a finished background job, the end). main.cpp has the settings, the drawing and
// the loop. Kept apart because it's long; it's still this example's own code, not the library's.
//
// Threads: three jobs run in the background so the windows never freeze: placing a scan, the live
// check and the next-shot advice. They only READ `scans` and `landmarks`; the functions here change
// them only when no job is reading (WaitForReaders), and never while a scan is being placed.
//
// Nothing is written while scanning (D39): the scans live in memory, and the whole folder is
// written once, on K, Q, a lost Kinect or an error (the rescue).
#pragma once

#include <chrono>
#include <ctime>
#include <thread>
#include <cstdio>
#include <filesystem>
#include <future>
#include <random>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/room.hpp"
#include "spotter/room_folder.hpp"
#include "spotter/room_scan.hpp"
#include "spotter/view.hpp"

struct Session
{
    // Settings (from main's)
    spotter::ScanOptions scanOptions;
    spotter::RoomOptions roomOptions;
    spotter::PinholeCamera camera;     // the depth camera the clouds are made with
    int liveEveryMs = 500, showStep = 2;

    // The room
    std::filesystem::path folder;      // written once: on K, Q, a lost Kinect or an error
    spotter::RoomInfo info;
    std::vector<spotter::Scan> scans;
    std::vector<std::vector<cv::Point3f>> shownPoints;   // what the viewer draws of each scan, in the room
    std::vector<std::vector<cv::Vec3b>> shownColours;
    std::vector<spotter::Landmark> landmarks;
    spotter::ColourModel colourModel;  // the colour camera as seen from the depth camera (13d), from the latest scan
    bool saved = false;                // the folder holds every change
    std::vector<bool> onDisk;          // per scan: its files are in the folder (only pose.txt may be older)

    // Scanning and the background jobs
    spotter::DepthAverager averager;
    struct Job { spotter::Placement placement; spotter::Scan scan; int again = -1; };   // again = P's scan
    std::future<Job> placing;
    std::future<spotter::LiveCheck> checking;
    std::future<spotter::Recommendation> recommending;
    spotter::LiveCheck live;
    spotter::Recommendation next;
    std::chrono::steady_clock::time_point lastCheck;
    std::mt19937 random{ 1 };

    // On screen
    int selected = -1;                 // the scan clicked in the viewer
    bool colourByScan = false;
    std::string message = "Set the Kinect down where you'll start, then press SPACE (in a window) to scan.";

    //explicit Session(int averageFrames){
    //
    //@param1 - averageFrames: how many depth frames one scan averages
    //
    //@brief - a session whose averager takes that many frames per scan. O(1).
    explicit Session(int averageFrames) : averager(averageFrames) {}
};

//template <typename T> bool Running(const std::future<T>& job){
//
//@param1 - job: a background job
//
//@brief - true while the job has started and isn't finished (checked without waiting). O(1).
template <typename T> bool Running(const std::future<T>& job)
{
    return job.valid() && job.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
}
//bool Busy(const Session& s){
//
//@param1 - s: the session
//
//@brief - true while a scan is being placed. O(1).
inline bool Busy(const Session& s) { return Running(s.placing); }
//bool Idle(const Session& s){
//
//@param1 - s: the session
//
//@brief - true when nothing is being placed or averaged: SPACE, U, X, P, E and K only act then. O(1).
inline bool Idle(const Session& s) { return !Busy(s) && !s.averager.Running(); }

//void Say(Session& s, const std::string& message){
//
//@param1 - s: the session
//@param2 - message: the text
//
//@brief - sets the on-screen message and prints it to the console. O(1).
inline void Say(Session& s, const std::string& message)
{
    s.message = message;
    std::printf("%s\n", message.c_str());
}

//void WaitForReaders(Session& s){
//
//@param1 - s: the session
//
//@brief - waits for the live check and the next-shot advice, which may be reading `scans`, so the list can be
//         changed safely. O(a job's time).
inline void WaitForReaders(Session& s)
{
    if (s.checking.valid()) s.live = s.checking.get();
    if (s.recommending.valid()) s.next = s.recommending.get();
}

//void Recommend(Session& s){
//
//@param1 - s: the session
//
//@brief - takes the last advice if it's ready and starts working out the next shot (RecommendNext) in the
//         background. O(1) here.
inline void Recommend(Session& s)
{
    if (s.recommending.valid()) s.next = s.recommending.get();
    s.recommending = std::async(std::launch::async, spotter::RecommendNext, std::cref(s.scans), s.camera, s.scanOptions);
}

//void Shown(const Session& s, const spotter::Scan& scan, std::vector<cv::Point3f>& points, std::vector<cv::Vec3b>& colours){
//
//@param1 - s: the session, for showStep
//@param2 - scan: a scan
//@param3 - points: set to its points in the room
//@param4 - colours: set to their colours
//
//@brief - the points the viewer draws for a scan: every showStep-th pixel each way with a reading, moved into the
//         room by its pose. O(pixels / showStep^2).
inline void Shown(const Session& s, const spotter::Scan& scan, std::vector<cv::Point3f>& points, std::vector<cv::Vec3b>& colours)
{
    points.clear();
    colours.clear();
    for (int row = 0; row < scan.picture.rows; row += s.showStep)
        for (int column = 0; column < scan.picture.cols; column += s.showStep)
        {
            const cv::Point3f& p = scan.cloud[row * scan.picture.cols + column];
            if (std::isnan(p.z)) continue;
            points.push_back(scan.pose(p));
            colours.push_back(scan.picture.at<cv::Vec3b>(row, column));
        }
}

// --- Saving: once, on K, Q, a lost Kinect or an error ---

//bool SaveAll(Session& s){
//
//@param1 - s: the session
//
//@brief - the whole folder: StartRoomFolder, removes scan folders left from scans that U took back, each scan
//         not yet on disk (WriteScan: pose.txt last, swapped in whole) and the others' pose.txt (X and P change
//         only that), landmarks.txt, then points.ply and room.txt with finished 1. A save cut off part-way leaves
//         the earlier save whole. Prints the result and the time; on failure the scans stay in memory and K tries
//         again. True if it all saved. O(new scans x pixels) for the scans, plus FinishRoomFolder.
inline bool SaveAll(Session& s)
{
    auto start = std::chrono::steady_clock::now();
    s.info.scans = (int)s.scans.size();
    bool ok = spotter::StartRoomFolder(s.folder, s.info) && spotter::RemoveScansFrom(s.folder, s.scans.size());
    for (size_t i = 0; i < s.scans.size() && ok; i++)
    {
        ok = s.onDisk[i] ? spotter::WritePose(s.folder, i, s.scans[i]) : spotter::WriteScan(s.folder, i, s.scans[i]);
        s.onDisk[i] = s.onDisk[i] || ok;
    }
    ok = ok && spotter::WriteLandmarks(s.folder, s.landmarks);
    size_t points = 0;
    ok = ok && spotter::FinishRoomFolder(s.folder, s.info, s.scans, s.roomOptions, points);
    s.saved = ok;
    std::printf("%s %s: %zu scans, points.ply %zu points, %.1f s.\n", ok ? "Saved" : "SAVING FAILED (the disk full, or the folder gone?) in",
                s.folder.string().c_str(), s.scans.size(), points, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    if (!ok) s.message = "SAVING FAILED - see the console. The scans are still in memory: K tries again.";
    return ok;
}

//void AddScan(Session& s, spotter::Scan scan){
//
//@param1 - s: the session
//@param2 - scan: a placed scan
//
//@brief - waits for readers, adds the scan with its viewer points and colours, and marks the folder as not saved.
//         O(pixels).
inline void AddScan(Session& s, spotter::Scan scan)
{
    WaitForReaders(s);   // the live check or the advice may be reading `scans` (13e didn't wait for the first scan)
    s.scans.push_back(std::move(scan));
    s.onDisk.push_back(false);
    s.shownPoints.emplace_back();
    s.shownColours.emplace_back();
    Shown(s, s.scans.back(), s.shownPoints.back(), s.shownColours.back());
    s.saved = false;
}

// --- Placing in the background ---

//void StartPlacing(Session& s, spotter::Scan scan, int again){
//
//@param1 - s: the session
//@param2 - scan: the scan to place
//@param3 - again: the index of the scan being placed again, or -1 for a new one
//
//@brief - starts PlaceScan on a background thread (so the windows keep working); `again` leaves that scan and its
//         landmarks out of the matching. O(1) here.
inline void StartPlacing(Session& s, spotter::Scan scan, int again)
{
    WaitForReaders(s);
    s.placing = std::async(std::launch::async, [&s, again](spotter::Scan scan) {
        Session::Job job;
        std::mt19937 random((unsigned)s.scans.size() + 11);
        job.placement = spotter::PlaceScan(scan, s.scans, s.camera, random, again, s.landmarks, s.scanOptions);
        job.scan = std::move(scan);
        job.again = again;
        return job;
    }, std::move(scan));
}

//void TakePlacement(Session& s, Session::Job done){
//
//@param1 - s: the session
//@param2 - done: a finished placement
//
//@brief - a finished placement: a refused one says why; P's scan takes its new pose, flags and stats (and its
//         landmarks are re-made); a new scan joins the room with its far landmarks (AddLandmarks), and a
//         suspicious one is flagged red. O(AddLandmarks).
inline void TakePlacement(Session& s, Session::Job done)
{
    WaitForReaders(s);
    const spotter::Placement& p = done.placement;
    spotter::Scan& scan = done.scan;
    if (p.placed) { scan.pose = p.pose; scan.suspicious = p.suspicious; scan.stats = p.stats; }
    if (done.again >= 0)
    {
        std::printf("Scan %d again: %s.\n", done.again + 1, p.report.c_str());
        if (!p.placed) { Say(s, cv::format("Couldn't place scan %d again (see the console). X excludes it.", done.again + 1)); return; }
        spotter::Scan& old = s.scans[done.again];
        old.pose = scan.pose;
        old.suspicious = scan.suspicious;
        old.stats = scan.stats;
        Shown(s, old, s.shownPoints[done.again], s.shownColours[done.again]);
        spotter::AddLandmarks(old, done.again, s.scans[p.matched], (int)p.matched, p.farPairs, s.landmarks, s.scanOptions);
        s.saved = false;
        Say(s, cv::format("Scan %d placed again.%s", done.again + 1, old.suspicious ? " Still suspicious: X excludes it." : ""));
        return;
    }
    if (!p.placed)
    {
        std::printf("Scan refused: %s.\n", p.report.c_str());
        Say(s, "Couldn't place that scan (see the console). Aim back at scanned parts until the anchors are green.");
        return;
    }
    std::printf("Scan %zu: %zu near + %zu far features; %s.\n", s.scans.size() + 1, scan.near.keypoints.size(),
                scan.far.keypoints.size(), p.report.c_str());
    AddScan(s, std::move(scan));
    int added = spotter::AddLandmarks(s.scans.back(), (int)s.scans.size() - 1, s.scans[p.matched], (int)p.matched, p.farPairs,
                                      s.landmarks, s.scanOptions);
    if (added > 0) std::printf("  %d new far landmarks (%zu in all)\n", added, s.landmarks.size());
    Say(s, cv::format("Scan %zu placed%s. Keep the green anchors in view as you move, or follow the Next line.", s.scans.size(),
                      s.scans.back().suspicious ? " - SUSPICIOUS (red): check it" : ""));
}

//void CollectJobs(Session& s){
//
//@param1 - s: the session
//
//@brief - background jobs that finished since the last frame: a placement is taken (and all cores are freed and
//         the next shot worked out), a live check's anchors and the advice are picked up. O(1) unless a job
//         finished.
inline void CollectJobs(Session& s)
{
    if (s.placing.valid() && !Running(s.placing))
    {
        TakePlacement(s, s.placing.get());
        cv::setNumThreads(-1);   // all cores again
        Recommend(s);
    }
    if (s.checking.valid() && !Running(s.checking)) s.live = s.checking.get();
    if (s.recommending.valid() && !Running(s.recommending))
    {
        s.next = s.recommending.get();
        std::printf("%s\n", s.next.text.c_str());
    }
}

// --- A new scan: averaged frames -> a Scan -> placed ---

//void OnScanReady(Session& s, const spotter::RgbdFrame& frame, const cv::Mat& averageMm){
//
//@param1 - s: the session
//@param2 - frame: the last frame, for colour and the colour index
//@param3 - averageMm: the averaged depth, mm
//
//@brief - a scan's averaging finished: the cloud, the scan (MakeScan), its colour model and far features; refused
//         if there are too few features (a dark or plain view); the first one becomes the room's starting point;
//         later ones go to StartPlacing (leaving a core or two for the Kinect). O(pixels).
inline void OnScanReady(Session& s, const spotter::RgbdFrame& frame, const cv::Mat& averageMm)
{
    std::vector<cv::Point3f> cloud = frame.depthCamera.Cloud(averageMm);
    spotter::Scan scan = spotter::MakeScan(cloud, frame.registered, s.scanOptions);
    scan.fullPicture = frame.colour.clone();
    scan.colourIndex = frame.colourIndex;
    spotter::ColourModel fitted = spotter::FitColourModel(cloud, frame.colourIndex, frame.colour.cols);
    if (fitted.ok) s.colourModel = fitted;
    cv::Mat grey;
    cv::cvtColor(frame.colour, grey, cv::COLOR_BGR2GRAY);
    spotter::AddFarFeatures(scan, grey, frame.bigdepth <= 0, s.colourModel, frame.depthCamera, s.scanOptions);

    const size_t features = scan.near.keypoints.size() + scan.far.keypoints.size();
    if ((int)features < s.scanOptions.minScanFeatures)
    {
        std::printf("Scan refused: only %zu near + %zu far features (need %d): the colour picture is too dark or too plain.\n",
                    scan.near.keypoints.size(), scan.far.keypoints.size(), s.scanOptions.minScanFeatures);
        s.message = "Scan refused: too few features. Turn the lights on, or aim at something with more pattern.";
    }
    else if (s.scans.empty())
    {
        scan.stats = "the room's starting point";
        std::printf("Scan 1: %zu near + %zu far features; the room's starting point.\n", scan.near.keypoints.size(), scan.far.keypoints.size());
        AddScan(s, std::move(scan));
        Say(s, "Scan 1: the room's starting point. Move or turn, keeping the green anchors in view, SPACE again.");
        Recommend(s);
    }
    else
    {
        cv::setNumThreads(std::max(1, (int)std::thread::hardware_concurrency() - 2));   // leave the Kinect's threads a core or two
        StartPlacing(s, std::move(scan), -1);
        s.message = cv::format("Placing scan %zu...", s.scans.size() + 1);
    }
}

//bool LiveCheckDue(const Session& s){
//
//@param1 - s: the session
//
//@brief - true when idle with scans placed, no check running, and liveEveryMs (about half a second) has passed
//         since the last. O(1).
inline bool LiveCheckDue(const Session& s)
{
    return Idle(s) && !s.scans.empty() && !Running(s.checking) &&
           std::chrono::steady_clock::now() - s.lastCheck > std::chrono::milliseconds(s.liveEveryMs);
}

//void StartLiveCheck(Session& s, const spotter::RgbdFrame& frame){
//
//@param1 - s: the session
//@param2 - frame: this frame (needs its bigdepth)
//
//@brief - the live anchors, about twice a second, in the background: this frame made into a scan with far
//         features, then CheckLive against the placed scans and landmarks. Does nothing if it isn't due.
//         O(MakeScan) here.
inline void StartLiveCheck(Session& s, const spotter::RgbdFrame& frame)
{
    if (!LiveCheckDue(s) || frame.bigdepth.empty()) return;
    std::vector<cv::Point3f> cloud = frame.depthCamera.Cloud(frame.depthMm);
    if (!s.colourModel.ok) { spotter::ColourModel fitted = spotter::FitColourModel(cloud, frame.colourIndex, frame.colour.cols); if (fitted.ok) s.colourModel = fitted; }
    cv::Mat grey, noDepth = frame.bigdepth <= 0, picture = frame.registered.clone();
    cv::cvtColor(frame.colour, grey, cv::COLOR_BGR2GRAY);
    s.checking = std::async(std::launch::async, [&s, cloud = std::move(cloud), picture, grey, noDepth, index = frame.colourIndex,
                                                 model = s.colourModel, camera = frame.depthCamera, width = frame.colour.cols]() {
        spotter::Scan live = spotter::MakeScan(cloud, picture, s.scanOptions);
        live.colourIndex = index;
        spotter::AddFarFeatures(live, grey, noDepth, model, camera, s.scanOptions);
        return spotter::CheckLive(live, s.scans, s.landmarks, width, s.scanOptions);
    });
    s.lastCheck = std::chrono::steady_clock::now();
}

// --- The keys ---

//void OnUndo(Session& s){
//
//@param1 - s: the session
//
//@brief - U: removes the last scan (when idle). ponytail: landmarks made with it stay, as in 13e; rebuild them if
//         that matters. Works out the next shot again. O(1).
inline void OnUndo(Session& s)
{
    if (!Idle(s) || s.scans.empty()) return;
    WaitForReaders(s);
    s.scans.pop_back();
    s.onDisk.pop_back();
    s.shownPoints.pop_back();
    s.shownColours.pop_back();
    if (s.selected >= (int)s.scans.size()) s.selected = -1;
    // ponytail: landmarks made with the removed scan stay (as in 13e); rebuild them if that matters.
    s.saved = false;
    Say(s, cv::format("Removed scan %zu. %zu left.", s.scans.size() + 1, s.scans.size()));
    Recommend(s);
}

//void OnExclude(Session& s){
//
//@param1 - s: the session
//
//@brief - X: toggles the selected scan out of placing, the advice and E (when idle), then works out the next shot
//         again. O(1).
inline void OnExclude(Session& s)
{
    if (!Idle(s) || s.selected < 0) return;
    WaitForReaders(s);
    spotter::Scan& scan = s.scans[s.selected];
    scan.excluded = !scan.excluded;
    s.saved = false;
    Say(s, cv::format("Scan %d %s.", s.selected + 1, scan.excluded ? "excluded: left out of placing, advice and E" : "included again"));
    Recommend(s);
}

//void OnPlaceAgain(Session& s){
//
//@param1 - s: the session
//
//@brief - P: places the selected scan again (StartPlacing with its index); an excluded one must be included first.
//         O(1) here.
inline void OnPlaceAgain(Session& s)
{
    if (!Idle(s) || s.selected < 0) return;
    if (s.scans[s.selected].excluded) { s.message = "Include it first (X), then P."; return; }
    StartPlacing(s, s.scans[s.selected], s.selected);
    s.message = cv::format("Placing scan %d again...", s.selected + 1);
}

//std::vector<const spotter::Scan*> Included(const Session& s){
//
//@param1 - s: the session
//
//@brief - pointers to the scans that aren't excluded. O(scans).
inline std::vector<const spotter::Scan*> Included(const Session& s)
{
    std::vector<const spotter::Scan*> included;
    for (const spotter::Scan& scan : s.scans) if (!scan.excluded) included.push_back(&scan);
    return included;
}

//void OnExport(Session& s){
//
//@param1 - s: the session
//
//@brief - E: merges every included scan (FuseScans), builds the clean model (BuildRoom), saves
//         room_scan_<time>.obj where it runs with the far landmarks, and prints the size, time, up source, floor
//         and ceiling, walls and boxes. O(FuseScans).
inline void OnExport(Session& s)
{
    if (!Idle(s)) return;
    WaitForReaders(s);
    std::vector<const spotter::Scan*> included = Included(s);
    if (included.empty()) { Say(s, "Nothing to export yet: SPACE scans (excluded scans don't count)."); return; }
    std::printf("Merging %zu scans...\n", included.size());
    auto start = std::chrono::steady_clock::now();
    float voxel;
    spotter::Room room = spotter::BuildRoom(included, spotter::FuseScans(included, s.camera, s.roomOptions, voxel), s.random, s.roomOptions);
    std::vector<cv::Point3f> landmarks;
    for (const spotter::Landmark& l : s.landmarks) landmarks.push_back(l.point);
    std::string path = (std::filesystem::current_path() / cv::format("room_scan_%lld.obj", (long long)std::time(nullptr))).string();
    if (!spotter::SaveRoomObj(path, room, landmarks, "Spotter 13e: seen_room = every scan merged, clean_* and furniture_blocks = the tidy (partly guessed) model"))
    {
        Say(s, "Could not write " + path);
        return;
    }
    std::printf("Saved %s (%.1f MB, %.1f s). A capture of the room: never commit it.\n", path.c_str(), std::filesystem::file_size(path) / 1e6,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    std::printf("  merged mesh: %zu triangles, %.1f cm cubes; %zu far landmarks\n", room.mesh.groups["seen_room"].size(), voxel * 100, landmarks.size());
    std::printf("  up from %s; floor %.2f m below the first camera (%s), ceiling %.2f m above it (%s): %.2f m high\n", room.upFrom.c_str(),
                -room.floorY, room.floorSeen ? "seen" : "guessed", room.ceilingY, room.ceilingSeen ? "seen" : "guessed", room.ceilingY - room.floorY);
    for (const auto& [a, b] : room.walls)
        std::printf("  wall  %.2f m long, from (%+.2f, %.2f) to (%+.2f, %.2f) m\n", cv::norm(b - a), a.x, -a.y, b.x, -b.y);
    for (const auto& [lo, hi] : room.blocks)
        std::printf("  block %.2f wide x %.2f deep x %.2f high, at (%+.2f, %.2f) m\n", hi.x - lo.x, hi.z - lo.z, hi.y - lo.y, (lo.x + hi.x) / 2, -(lo.z + hi.z) / 2);
    std::printf("  (positions: X right, Z ahead, along the biggest wall, from the first camera)\n");
    s.message = "Exported (see the console). Scan more and export again, or Q.";
}

//void OnSave(Session& s){
//
//@param1 - s: the session
//
//@brief - K: saves the folder now (SaveAll); Q, a lost Kinect and an error do it too. O(scans x pixels).
inline void OnSave(Session& s)
{
    if (!Idle(s)) return;
    WaitForReaders(s);
    if (s.scans.empty()) { s.message = "Nothing scanned yet."; return; }
    if (SaveAll(s)) s.message = "Folder saved (see the console). Scan more, or Q.";
}

//void OnClick(Session& s, const spotter::OrbitView& view, cv::Point at){
//
//@param1 - s: the session
//@param2 - view: the viewer, to project each camera
//@param3 - at: the clicked pixel
//
//@brief - a click in the viewer selects the scan whose camera circle is nearest (within 15 pixels), or none;
//         prints its stats. O(scans).
inline void OnClick(Session& s, const spotter::OrbitView& view, cv::Point at)
{
    s.selected = -1;
    double nearest = 15;
    for (size_t i = 0; i < s.scans.size(); i++)
    {
        cv::Point pixel;
        float depth;
        if (!view.Project(s.scans[i].pose.shift, pixel, depth)) continue;
        if (cv::norm(pixel - at) < nearest) { nearest = cv::norm(pixel - at); s.selected = (int)i; }
    }
    if (s.selected >= 0)
        std::printf("Selected scan %d%s: %s\n", s.selected + 1, s.scans[s.selected].excluded ? " (excluded)" : "", s.scans[s.selected].stats.c_str());
}

// --- The end: Q, a lost Kinect, or an error. Everything is saved BEFORE the Kinect is stopped. ---

//void EndSession(Session& s, const std::string& why){
//
//@param1 - s: the session
//@param2 - why: why it ended (Quitting, Kinect lost, an error), for the console
//
//@brief - the rescue, run on Q, a lost Kinect or an exception: a scan still being placed is
//         waited for and kept, then the folder is saved with no questions if it isn't already. Called before the
//         Kinect stops. Errors while waiting are printed, not thrown. O(SaveAll).
inline void EndSession(Session& s, const std::string& why)
{
    try
    {
        if (s.placing.valid())
        {
            Session::Job done = s.placing.get();   // a scan still being placed is saved too; a P still running is dropped
            if (done.again < 0 && done.placement.placed)
            {
                std::printf("Scan %zu was still being placed: placed (%s).\n", s.scans.size() + 1, done.placement.report.c_str());
                TakePlacement(s, std::move(done));
            }
        }
    }
    catch (const std::exception& e) { std::printf("The scan being placed is lost: %s\n", e.what()); }
    try { WaitForReaders(s); } catch (...) {}
    if (s.scans.empty()) { std::printf("No scans, so nothing was saved.\n"); return; }
    if (!s.saved)
    {
        std::printf("\n=== %s: saving the folder ===\n", why.c_str());
        SaveAll(s);
    }
    std::printf("Every scan is in %s. 13z rebuilds the room from it.\n", s.folder.string().c_str());
}
