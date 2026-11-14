// Self-check for 13e's scanning session (examples/13e_Room_Scan/scan_state.hpp), with no Kinect: check_room's
// patterned box room, photographed from 4 spots, fed through the session's own functions (review 2026-11-11,
// initiative 6: this replaces the temporary harnesses behind D38 and D48).
//   1. Placing: scan 1 starts the room, scans 2-4 are placed in the background (StartPlacing + CollectJobs),
//      each within 5 mm of the truth.
//   2. K (SaveAll): the folder loads back with every pose within 0.5 mm, finished.
//   3. X then K: only pose.txt is rewritten (cloud.bin untouched, D45); the scan loads back excluded.
//   4. P: the scan is placed again, still within 5 mm.
//   5. U then K: the last scan's folder is gone from disk.
//   6. The rescue (D30, D48): a scan still being placed and the room folder blocked by a file. EndSession keeps
//      the scan, saves to the first RescueFolders entry with no questions, and every scan loads back.
//   7. Every folder blocked and no one to answer: EndSession reports the scans not saved (13e exits non-zero).
// Run: build\check_scan_session.exe. Exit code 0 = passed. Writes only in the temp folder (the rescue too, through
// SPOTTER_RESCUE_FOLDER), and removes it again.

#include "../../examples/13e_Room_Scan/scan_state.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>

namespace fs = std::filesystem;

int failures = 0;
void Check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAILED: %s\n", what); failures++; }
}

const float Low[3] = { -1.5f, -1.3f, -1.0f }, High[3] = { 2.0f, 1.2f, 3.0f };            // the room (Y down), as check_room
const float CouchLow[3] = { 0.2f, 0.6f, 2.0f }, CouchHigh[3] = { 1.0f, 1.2f, 2.5f };
const spotter::PinholeCamera Camera{ 365.5f, 365.5f, 256.0f, 212.0f, 512, 424 };

float Raycast(const cv::Point3f& from, const cv::Point3f& ray, cv::Vec3b& colour)
{
    const float o[3] = { from.x, from.y, from.z }, d[3] = { ray.x, ray.y, ray.z };
    float best = 1e9f;
    int face = 0;
    for (int a = 0; a < 3; a++)
        if (d[a] != 0)
        {
            float t = ((d[a] > 0 ? High[a] : Low[a]) - o[a]) / d[a];
            if (t < best) { best = t; face = 2 * a + (d[a] > 0); }
        }
    float enter = 0, leave = 1e9f;
    int enterFace = 0;
    for (int a = 0; a < 3; a++)
    {
        float t1 = (CouchLow[a] - o[a]) / d[a], t2 = (CouchHigh[a] - o[a]) / d[a];
        if (std::min(t1, t2) > enter) { enter = std::min(t1, t2); enterFace = 6 + a; }
        leave = std::min(leave, std::max(t1, t2));
    }
    if (enter < leave && enter > 0 && enter < best) { best = enter; face = enterFace; }
    cv::Point3f p = from + ray * best;
    int i = (int)std::floor(p.x / 0.1f), j = (int)std::floor(p.y / 0.1f), k = (int)std::floor(p.z / 0.1f);
    unsigned h = (unsigned)(face * 73856093) ^ (unsigned)(i * 19349663) ^ (unsigned)(j * 83492791) ^ (unsigned)(k * 2654435761u);
    h ^= h >> 13; h *= 0x5bd1e995; h ^= h >> 15;
    colour = cv::Vec3b((uchar)(h & 255), (uchar)((h >> 8) & 255), (uchar)((h >> 16) & 255));
    return best;
}

spotter::Scan Photograph(const spotter::Pose& pose, int seed)
{
    cv::RNG noise(seed);
    std::vector<cv::Point3f> cloud(512 * 424);
    cv::Mat picture(424, 512, CV_8UC3);
    for (int row = 0; row < 424; row++)
        for (int column = 0; column < 512; column++)
        {
            cv::Point3f ray = Camera.Unproject(cv::Point2f((float)column, (float)row), 1.0f);
            cv::Vec3b colour;
            float t = Raycast(pose.shift, pose.turn * ray, colour);
            cloud[row * 512 + column] = ray * (t * (1 + (float)noise.gaussian(0.0005)));
            picture.at<cv::Vec3b>(row, column) = colour;
        }
    spotter::Scan scan = spotter::MakeScan(cloud, picture, spotter::ScanOptions());
    scan.colourIndex.assign(512 * 424, -1);
    return scan;
}

spotter::Pose Spot(float right, float forward, float leftDegrees)
{
    return { spotter::TurnAbout(cv::Vec3d(0, -leftDegrees * CV_PI / 180, 0)), cv::Point3f(right, 0, forward) };
}

// The background placement finished and taken in, as 13e's loop does once per frame.
void Collect(Session& s)
{
    if (s.placing.valid()) s.placing.wait();
    CollectJobs(s);
}

// Every scan folder loads back, with its pose within 0.5 mm of the session's.
bool LoadsBack(const Session& s, std::vector<spotter::Scan>* out = nullptr)
{
    spotter::RoomInfo info;
    std::string problem;
    if (!spotter::ReadRoomInfo(s.folder, info, problem) || !info.finished) return false;
    std::vector<fs::path> folders = spotter::ScanFolders(s.folder);
    if (folders.size() != s.scans.size()) return false;
    for (size_t k = 0; k < folders.size(); k++)
    {
        spotter::Scan back;
        int number;
        if (!spotter::LoadScan(folders[k], info, back, number, problem) || cv::norm(back.pose.shift - s.scans[k].pose.shift) > 0.0005f) return false;
        if (out) out->push_back(std::move(back));
    }
    return true;
}

double MmOff(const spotter::Scan& scan, const spotter::Pose& truth) { return cv::norm(scan.pose.shift - truth.shift) * 1000; }

int main()
{
    // A broken rescue must fail here, not wait on a message box: every question goes to the console, which has no answers.
#ifdef _WIN32
    _putenv_s("SPOTTER_CONSOLE_DIALOGS", "1");
#else
    setenv("SPOTTER_CONSOLE_DIALOGS", "1", 1);
#endif
    std::cin.setstate(std::ios::eofbit);

    const fs::path base = fs::temp_directory_path() / "spotter_check_session";
    std::error_code error;
    fs::remove_all(base, error);
    const fs::path rescues = base / "rescues";   // the rescue's first folder, never the repo's room_scans
#ifdef _WIN32
    _putenv_s("SPOTTER_RESCUE_FOLDER", rescues.string().c_str());
#else
    setenv("SPOTTER_RESCUE_FOLDER", rescues.string().c_str(), 1);
#endif

    Session s(60);
    s.camera = Camera;
    s.folder = base / "room";
    s.roomsFolder = base.string();
    s.info.program = "check_scan_session";
    s.info.depthCamera = Camera;
    s.info.calibration = "ir_params 365.5 365.5 256 212 0 0 0 0 0\n";

    // 1. Placing.
    const std::vector<spotter::Pose> truth = { Spot(0, 0, 0), Spot(0.25f, 0.1f, 15), Spot(0.5f, 0.2f, 30), Spot(0, 0.3f, -30) };
    AddScan(s, Photograph(truth[0], 1));   // what OnScanReady does with the first scan
    for (size_t k = 1; k < truth.size(); k++)
    {
        StartPlacing(s, Photograph(truth[k], (int)k + 1), -1);
        Collect(s);
    }
    bool placed = s.scans.size() == truth.size();
    for (size_t k = 0; placed && k < truth.size(); k++) placed = MmOff(s.scans[k], truth[k]) < 5;
    std::printf("1. %zu scans placed; scan 4 %.1f mm off\n", s.scans.size(), placed ? MmOff(s.scans[3], truth[3]) : -1.0);
    Check(placed, "scans 2-4 are placed in the background, each within 5 mm");
    Check(!s.saved && s.onDisk.size() == s.scans.size(), "nothing is saved while scanning (D39)");

    // 2. K.
    std::string problem;
    Check(SaveAll(s, problem) && s.saved, "K saves the folder");
    Check(LoadsBack(s), "and it loads back: every scan, poses within 0.5 mm, finished");

    // 3. X then K: only pose.txt changes.
    const fs::path cloud2 = spotter::ScanFolders(s.folder)[1] / "cloud.bin";
    const fs::file_time_type cloudTime = fs::last_write_time(cloud2);
    s.selected = 1;
    OnExclude(s);
    Check(s.scans[1].excluded && !s.saved, "X excludes the selected scan and marks the folder unsaved");
    std::vector<spotter::Scan> back;
    Check(SaveAll(s, problem) && LoadsBack(s, &back) && back[1].excluded, "after K the scan loads back excluded");
    Check(fs::last_write_time(cloud2) == cloudTime, "and its cloud.bin wasn't rewritten (only pose.txt, D45)");
    OnExclude(s);   // included again

    // 4. P.
    s.selected = 2;
    OnPlaceAgain(s);
    Collect(s);
    std::printf("4. scan 3 placed again: %.1f mm off\n", MmOff(s.scans[2], truth[2]));
    Check(MmOff(s.scans[2], truth[2]) < 5 && !s.saved, "P places the scan again, within 5 mm");

    // 5. U then K.
    OnUndo(s);
    Check(s.scans.size() == 3 && SaveAll(s, problem) && spotter::ScanFolders(s.folder).size() == 3 && LoadsBack(s),
          "U takes the last scan back, and K removes its folder from disk");

    // 6. The rescue: a scan still being placed, the room folder blocked by a file. The first rescue folder is this
    // check's own (SPOTTER_RESCUE_FOLDER above), named to the second, so it's recognised by its parent and name.
    Check(spotter::RescueFolders()[0].parent_path() == rescues, "SPOTTER_RESCUE_FOLDER replaces the repo's rescue folder");
    std::ofstream(base / "a_file") << "in the way\n";
    s.folder = base / "a_file" / "room";
    s.saved = false;
    StartPlacing(s, Photograph(truth[3], 4), -1);
    const bool saved = EndSession(s, "check_scan_session");   // what Q, a lost Kinect or an error does
    const bool inRescues = s.folder.parent_path() == rescues && s.folder.filename().string().rfind("rescue_", 0) == 0;
    std::printf("6. rescued to %s\n", s.folder.string().c_str());
    Check(s.scans.size() == 4, "the scan still being placed is kept");
    Check(saved && inRescues && s.saved && LoadsBack(s), "the blocked folder fails over to the first rescue folder, with no questions, and every scan loads back");
    WaitForReaders(s);

    // 7. Every folder blocked (a file where each rescue folder's parent should be), no one to answer: EndSession
    // returns false, so 13e exits non-zero, and says the scans are not saved.
    s.folder = base / "a_file" / "room";
    s.saved = false;
    Check(!EndSession(s, "check_scan_session", { base / "a_file" / "r1", base / "a_file" / "r2" }) && !s.saved,
          "every folder failing: EndSession reports not saved");
    WaitForReaders(s);
    fs::remove_all(base, error);

    std::printf(failures ? "check_scan_session: %d FAILED\n" : "check_scan_session: all passed\n", failures);
    return failures ? 1 : 0;
}
