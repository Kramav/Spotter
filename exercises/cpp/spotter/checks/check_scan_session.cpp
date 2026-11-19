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
//   8. Continuing a saved room (ContinueRoom): loaded whole, a new scan placed against it, all saved to a new folder,
//      the saved room untouched; another depth camera refused.
// Run: build\check_scan_session.exe. Exit code 0 = passed. Writes only in the temp folder (the rescue too, through
// SPOTTER_RESCUE_FOLDER), and removes it again.

#include "../../examples/13e_Room_Scan/scan_state.hpp"
#include "test_room.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>

namespace fs = std::filesystem;


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
    const std::vector<spotter::Pose> truth = { Spot(0, 0, 0, 0, 0), Spot(0.25f, 0, 0.1f, 15, 0), Spot(0.5f, 0, 0.2f, 30, 0), Spot(0, 0, 0.3f, -30, 0) };
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

    // 8. Continuing a saved room (ContinueRoom): 3 whole scans saved, loaded into a new session, a 4th placed against
    // them, all 4 saved to a NEW folder; the saved room is left as it was. Another depth camera is refused.
    auto Whole = [](spotter::Scan scan) {   // as 13e's OnScanReady makes them: the full picture and the colour index too
        scan.fullPicture = scan.picture.clone();
        scan.colourIndex.resize(scan.cloud.size());
        for (size_t i = 0; i < scan.cloud.size(); i++) scan.colourIndex[i] = std::isnan(scan.cloud[i].z) ? -1 : (int)i;
        return scan;
    };
    auto NewSession = [&](Session& n, const fs::path& folder) {
        n.camera = Camera;
        n.folder = folder;
        n.roomsFolder = base.string();
        n.info.program = "check_scan_session";
        n.info.depthCamera = Camera;
        n.info.calibration = s.info.calibration;
    };
    Session first(60);
    NewSession(first, base / "to_continue");
    AddScan(first, Whole(Photograph(truth[0], 1)));
    for (size_t k = 1; k < 3; k++) { StartPlacing(first, Whole(Photograph(truth[k], (int)k + 1)), -1); Collect(first); }
    Check(first.scans.size() == 3 && SaveAll(first, problem), "8. a room of 3 whole scans is saved");
    const fs::file_time_type savedAt = fs::last_write_time(first.folder / "room.txt");

    Session next(60);
    NewSession(next, base / "continued");
    const bool continued = ContinueRoom(next, first.folder, problem);
    bool same = continued && next.scans.size() == 3 && next.landmarks.size() == first.landmarks.size();
    for (size_t k = 0; same && k < 3; k++)
    {
        const spotter::Scan &a = first.scans[k], &b = next.scans[k];
        same = cv::norm(a.pose.shift - b.pose.shift) < 0.0005f && a.near.keypoints.size() == b.near.keypoints.size() &&
               cv::norm(a.near.descriptors, b.near.descriptors, cv::NORM_HAMMING) == 0 && a.far.keypoints.size() == b.far.keypoints.size() &&
               b.fullPicture.size() == a.fullPicture.size() && b.colourIndex == a.colourIndex && !next.onDisk[k];
    }
    std::printf("8. continued: %s; %zu scans, %zu landmarks\n", continued ? "loaded" : problem.c_str(), next.scans.size(), next.landmarks.size());
    Check(same, "ContinueRoom loads every scan whole: poses within 0.5 mm, the same features, pictures and index, none counted as on disk");
    StartPlacing(next, Whole(Photograph(truth[3], 4)), -1);
    Collect(next);
    Check(next.scans.size() == 4 && MmOff(next.scans[3], truth[3]) < 5, "a new scan is placed against the loaded ones, within 5 mm");
    Check(SaveAll(next, problem) && LoadsBack(next) && spotter::ScanFolders(next.folder).size() == 4,
          "K writes all 4 scans to the new folder, and they load back");
    Check(spotter::ScanFolders(first.folder).size() == 3 && fs::last_write_time(first.folder / "room.txt") == savedAt,
          "the room that was continued is left as it was");
    Session other(60);
    NewSession(other, base / "other");
    other.camera.fx += 5;
    Check(!ContinueRoom(other, first.folder, problem) && other.scans.empty(), "a room from another depth camera is refused, nothing loaded");
    std::printf("   (refused: %s)\n", problem.c_str());
    WaitForReaders(first);
    WaitForReaders(next);
    fs::remove_all(base, error);

    std::printf(failures ? "check_scan_session: %d FAILED\n" : "check_scan_session: all passed\n", failures);
    return failures ? 1 : 0;
}
