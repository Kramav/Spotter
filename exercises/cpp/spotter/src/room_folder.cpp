#include "spotter/room_folder.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <opencv2/imgcodecs.hpp>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace spotter
{

//bool SyncToDisk(const fs::path& path){
//
//@param1 - path: a file just written
//
//@brief - pushes the file from the system's cache onto the disk (_commit on Windows, fsync elsewhere). Bug 4: after a blue
//         screen, a file "written" but still in the cache can come back the right size and full of zeros. False if it fails. O(1).
static bool SyncToDisk(const fs::path& path)
{
#ifdef _WIN32
    FILE* file = _wfopen(path.c_str(), L"r+b");
    if (!file) return false;
    bool ok = std::fflush(file) == 0 && _commit(_fileno(file)) == 0;
#else
    FILE* file = std::fopen(path.c_str(), "r+b");
    if (!file) return false;
    bool ok = std::fflush(file) == 0 && fsync(fileno(file)) == 0;
#endif
    std::fclose(file);
    return ok;
}

//bool ReplaceText(const fs::path& path, const std::string& text){
//
//@param1 - path: the file to replace
//@param2 - text: its new contents
//
//@brief - replaces a text file in one go: written beside it as path.new, flushed to disk (SyncToDisk), then renamed over it.
//         A crash part-way leaves the old file, never half of the new one. False if it fails. O(text length).
static bool ReplaceText(const fs::path& path, const std::string& text)
{
    fs::path fresh = path;
    fresh += ".new";
    {
        std::ofstream file(fresh);
        file << text;
        if (!file) return false;
    }
    if (!SyncToDisk(fresh)) return false;
    std::error_code error;
    fs::rename(fresh, path, error);
    return !error;
}

//std::map<std::string, std::string> ReadKeys(const fs::path& path){
//
//@param1 - path: a text file of "key value" lines (room.txt, pose.txt)
//
//@brief - reads each non-empty line that doesn't start with # as key (up to the first space) -> the rest. A missing file
//         gives an empty map. O(lines).
static std::map<std::string, std::string> ReadKeys(const fs::path& path)
{
    std::map<std::string, std::string> keys;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        size_t space = line.find(' ');
        keys[line.substr(0, space)] = space == std::string::npos ? "" : line.substr(space + 1);
    }
    return keys;
}

//std::vector<float> Numbers(const std::string& text){
//
//@param1 - text: numbers separated by spaces, e.g. a ReadKeys value
//
//@brief - every number in the text, in order; stops at the first thing that isn't one. O(length).
static std::vector<float> Numbers(const std::string& text)
{
    std::vector<float> numbers;
    std::istringstream in(text);
    for (float v; in >> v; ) numbers.push_back(v);
    return numbers;
}

//std::string Now(const char* layout){
//
//@param1 - layout: a strftime layout, e.g. "%Y-%m-%d %H:%M"
//
//@brief - the local time as text in that layout (up to 31 characters). O(1).
static std::string Now(const char* layout)
{
    char text[32];
    std::time_t now = std::time(nullptr);
    std::strftime(text, sizeof text, layout, std::localtime(&now));
    return text;
}

//std::string TurnText(const cv::Matx33f& t, const char* number){
//
//@param1 - t: a 3 x 3 turn
//@param2 - number: the printf layout of one number, e.g. "%.6f"
//
//@brief - the nine numbers row by row, separated by spaces (as room.txt and pose.txt hold a turn). O(1).
static std::string TurnText(const cv::Matx33f& t, const char* number)
{
    std::string text;
    for (int i = 0; i < 9; i++) text += cv::format(number, t(i / 3, i % 3)) + (i < 8 ? " " : "");
    return text;
}

// --- Writing ---

fs::path ScanFolder(const fs::path& room, size_t index) { return room / cv::format("scan_%03zu", index + 1); }

bool WriteRoomInfo(const fs::path& room, const RoomInfo& info)
{
    const PinholeCamera& c = info.depthCamera;
    std::string text = "# A Spotter room scan folder, written as it scans. 13z rebuilds the room from it. A capture of a room: never commit it.\n";
    text += cv::format("format 2\nprogram %s\nsaved %s\nscans %d\n", info.program.c_str(), Now("%Y-%m-%d %H:%M").c_str(), info.scans);
    text += cv::format("depth_size %d %d\ncolour_size %d %d\n", c.width, c.height, info.colourSize.width, info.colourSize.height);
    text += "# the depth camera the clouds were made with (left and right as in the room): fx fy cx cy\n";
    text += cv::format("depth_camera %.9g %.9g %.9g %.9g\n", c.fx, c.fy, c.cx, c.cy);
    text += "# the device's factory numbers, as it reports them (mirrored)\n" + info.calibration;
    if (!info.calibration.empty() && info.calibration.back() != '\n') text += "\n";
    if (!info.finished)
        text += "# finished 0: points.ply and first_camera_turn are missing or older than the scans (K or Q writes them)\nfinished 0\n";
    else
        text += "first_camera_turn " + TurnText(info.firstCamera, "%.6f") + "\nup_from " + info.upFrom + "\nfinished 1\n";
    return ReplaceText(room / "room.txt", text);
}

bool StartRoomFolder(const fs::path& room, const RoomInfo& info)
{
    std::error_code error;
    fs::create_directories(room, error);
    if (error) return false;
    std::ofstream(room / ".gitignore") << "# A capture of a room (Spotter): never commit it.\n*\n";
    RoomInfo empty = info;
    empty.scans = 0;
    empty.finished = false;
    return WriteRoomInfo(room, empty);
}

//std::string PoseText(size_t index, const Scan& scan){
//
//@param1 - index: the scan's index, from 0
//@param2 - scan: the scan
//
//@brief - pose.txt's contents: turn, shift, flags, the cloud's point count and the stats. O(pixels).
static std::string PoseText(size_t index, const Scan& scan)
{
    size_t points = 0;
    for (const cv::Point3f& p : scan.cloud) points += !std::isnan(p.z);
    return cv::format("# scan %zu's camera -> the room (scan 1's camera): room point = turn * point + shift\n", index + 1)
           + "turn " + TurnText(scan.pose.turn, "%.7f") + "\n"
           + cv::format("shift %.6f %.6f %.6f\n", scan.pose.shift.x, scan.pose.shift.y, scan.pose.shift.z)
           + cv::format("excluded %d\nsuspicious %d\n", (int)scan.excluded, (int)scan.suspicious)
           + cv::format("# how many readings cloud.bin holds: a cloud that doesn't match is refused\npoints %zu\n", points)
           + "stats " + scan.stats + "\n";
}

bool WritePose(const fs::path& room, size_t index, const Scan& scan)
{
    return ReplaceText(ScanFolder(room, index) / "pose.txt", PoseText(index, scan));
}

bool WriteScan(const fs::path& room, size_t index, const Scan& scan)
{
    // Written beside the scan's folder as scan_NNN.new, then swapped in: an older save of this number stays
    // whole until the new one is complete, so a crash mid-save never loses it. ScanFolders reads a swap cut off half-way.
    const fs::path folder = ScanFolder(room, index);
    fs::path dir = folder, old = folder;
    dir += ".new";
    old += ".old";
    std::error_code error;
    fs::remove_all(dir, error);   // left by an earlier save that was cut off
    fs::create_directories(dir, error);
    if (error) return false;
    bool ok = (bool)std::ofstream(dir / "cloud.bin", std::ios::binary).write((const char*)scan.cloud.data(), scan.cloud.size() * sizeof(cv::Point3f));
    ok = cv::imwrite((dir / "colour.bmp").string(), scan.picture) && ok;
    if (!scan.fullPicture.empty()) ok = cv::imwrite((dir / "colour_full.bmp").string(), scan.fullPicture) && ok;
    ok = (bool)std::ofstream(dir / "colour_index.bin", std::ios::binary).write((const char*)scan.colourIndex.data(), scan.colourIndex.size() * sizeof(int)) && ok;
    {
        cv::FileStorage features((dir / "features.yml").string(), cv::FileStorage::WRITE);
        features << "keypoints" << scan.near.keypoints << "descriptors" << scan.near.descriptors;
        features << "far_keypoints" << scan.far.keypoints << "far_descriptors" << scan.far.descriptors
                 << "far_rays" << scan.farRays << "far_origin" << scan.farOrigin;
    }
    for (const auto& entry : fs::directory_iterator(dir, error))   // bug 4: on the disk before pose.txt says "complete"
        ok = SyncToDisk(entry.path()) && ok;
    if (!ok || !ReplaceText(dir / "pose.txt", PoseText(index, scan))) return false;
    // The swap: folder -> .old, .new -> folder, then .old goes. Cut off between the two renames, both are whole.
    fs::remove_all(old, error);
    if (fs::exists(folder, error)) fs::rename(folder, old, error);
    if (error) return false;
    fs::rename(dir, folder, error);
    if (error) return false;
    fs::remove_all(old, error);
    return true;
}

//int ScanNumber(const fs::path& folder){
//
//@param1 - folder: a folder path
//
//@brief - the number in a folder named scan_NNN (5 for scan_005), also for WriteScan's scan_NNN.new and scan_NNN.old; 0 for
//         any other name. Sorting by it puts scan_1000 after scan_999 (bug 5). O(name length).
static int ScanNumber(const fs::path& folder)
{
    const std::string name = folder.stem().string(), swap = folder.extension().string();
    if (name.rfind("scan_", 0) != 0 || name.size() == 5 || name.find_first_not_of("0123456789", 5) != std::string::npos) return 0;
    return swap.empty() || swap == ".new" || swap == ".old" ? std::atoi(name.c_str() + 5) : 0;
}

bool RemoveScansFrom(const fs::path& room, size_t index)
{
    std::vector<fs::path> doomed;
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(room, error))
        if (ScanNumber(entry.path()) > (int)index) doomed.push_back(entry.path());
    bool ok = true;
    for (const fs::path& path : doomed) { fs::remove_all(path, error); ok = ok && !error; }
    return ok;
}

bool WriteLandmarks(const fs::path& room, const std::vector<Landmark>& landmarks)
{
    std::string text = "# far landmarks: x y z (metres, scan 1's camera), times seen, ORB descriptor (hex)\n";
    for (const Landmark& l : landmarks)
    {
        text += cv::format("%.4f %.4f %.4f %d ", l.point.x, l.point.y, l.point.z, l.seen);
        for (int b = 0; b < l.descriptor.cols; b++) text += cv::format("%02x", l.descriptor.at<uchar>(0, b));
        text += "\n";
    }
    return ReplaceText(room / "landmarks.txt", text);
}

bool FinishRoomFolder(const fs::path& room, RoomInfo& info, const std::vector<Scan>& scans, const RoomOptions& options, size_t& count)
{
    std::vector<const Scan*> included;
    for (const Scan& s : scans) if (!s.excluded) included.push_back(&s);
    std::mt19937 random(1);
    Room levelled = BuildRoom(included, Mesh(), random, options);   // only the planes decide the levelling
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colours;
    RoomPoints(included, levelled.firstCamera, points, colours);
    count = points.size();
    std::error_code error;
    bool ok = SavePly((room / "points.ply.new").string(), points, colours) && SyncToDisk(room / "points.ply.new");
    if (ok) fs::rename(room / "points.ply.new", room / "points.ply", error);
    info.scans = (int)scans.size();
    info.finished = ok && !error;
    info.firstCamera = levelled.firstCamera;
    info.upFrom = levelled.upFrom;
    return WriteRoomInfo(room, info) && info.finished;   // last: "finished 1"
}

// --- Reading ---

std::vector<fs::path> ScanFolders(const fs::path& room)
{
    // Per number: scan_NNN; after a swap cut off between its renames (WriteScan), the finished .new, else the .old.
    auto rank = [](const fs::path& p) { return p.extension() == ".new" ? 1 : p.extension() == ".old" ? 2 : 0; };
    std::map<int, fs::path> chosen;   // sorted by number
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(room, error))
    {
        const fs::path& p = entry.path();
        const int number = ScanNumber(p);
        if (number <= 0 || !entry.is_directory(error)) continue;
        if (rank(p) == 1 && !fs::exists(p / "pose.txt", error)) continue;   // a .new cut off before it was finished
        auto it = chosen.find(number);
        if (it == chosen.end() || rank(p) < rank(it->second)) chosen[number] = p;
    }
    std::vector<fs::path> folders;
    for (const auto& [number, p] : chosen) folders.push_back(p);
    return folders;
}

std::vector<SavedRoom> FindRooms(const fs::path& root, int maxDepth)
{
    std::vector<fs::path> found;
    std::error_code error;
    if (fs::exists(root / "room.txt", error)) found.push_back(root);
    else if (fs::is_directory(root, error))
        for (auto it = fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied, error);
             it != fs::recursive_directory_iterator(); it.increment(error))
        {
            if (error) break;
            if (!it->is_directory(error)) continue;
            if (fs::exists(it->path() / "room.txt", error)) { found.push_back(it->path()); it.disable_recursion_pending(); }
            else if (it.depth() + 1 >= maxDepth) it.disable_recursion_pending();
        }
    std::sort(found.begin(), found.end());
    std::vector<SavedRoom> rooms;
    for (const fs::path& folder : found)
    {
        auto keys = ReadKeys(folder / "room.txt");
        SavedRoom room{ folder, keys["program"], keys["saved"], std::atoi(keys["format"].c_str()) };
        uintmax_t bytes = 0;
        for (const auto& entry : fs::recursive_directory_iterator(folder, error))
            if (entry.is_regular_file(error)) bytes += entry.file_size(error);
        for (const fs::path& scan : ScanFolders(folder))
        {
            room.scans++;
            room.excluded += ReadKeys(scan / "pose.txt")["excluded"] == "1";
        }
        room.megabytes = bytes / 1e6;
        rooms.push_back(room);
    }
    return rooms;
}

bool ReadRoomInfo(const fs::path& room, RoomInfo& info, std::string& problem)
{
    auto keys = ReadKeys(room / "room.txt");
    info = RoomInfo();
    info.format = std::atoi(keys["format"].c_str());
    info.program = keys["program"];
    info.saved = keys["saved"];
    info.scans = std::atoi(keys["scans"].c_str());
    info.finished = keys["finished"] == "1";
    info.upFrom = keys["up_from"];
    std::vector<float> size = Numbers(keys["depth_size"]), colour = Numbers(keys["colour_size"]), turn = Numbers(keys["first_camera_turn"]);
    if (size.size() != 2) size = { 512, 424 };
    if (colour.size() == 2) info.colourSize = cv::Size((int)colour[0], (int)colour[1]);
    if (turn.size() == 9) for (int i = 0; i < 9; i++) info.firstCamera(i / 3, i % 3) = turn[i];
    std::vector<float> camera = Numbers(keys[info.format == 2 ? "depth_camera" : "ir_params"]);
    if ((info.format != 1 && info.format != 2) || camera.size() < 4)
    {
        problem = keys.empty() ? "no room.txt" : "room.txt is from an unknown format, or has no camera";
        return false;
    }
    if (!(size[0] >= 1 && size[1] >= 1 && size[0] <= 4096 && size[1] <= 4096) || !(camera[0] > 0 && camera[1] > 0))   // NaN fails too
    {
        problem = "room.txt's depth_size or camera numbers are impossible (a damaged file?)";
        return false;
    }
    info.depthCamera = { camera[0], camera[1], camera[2], camera[3], (int)size[0], (int)size[1] };
    if (info.format == 1)   // the device's own (mirrored) numbers: unmirror them as Unmirror does
    {
        info.depthCamera.cx = info.depthCamera.width - info.depthCamera.cx;
        info.firstCamera = cv::Matx33f::eye();   // format 1's turn levelled mirrored points; FinishRoomFolder redoes it
        info.finished = false;
    }
    return true;
}

bool LoadScan(const fs::path& folder, const RoomInfo& info, Scan& scan, int& number, std::string& problem, bool picture)
{
    const int width = info.depthCamera.width, height = info.depthCamera.height;
    number = ScanNumber(folder);
    scan = Scan();
    auto keys = ReadKeys(folder / "pose.txt");
    std::vector<float> turn = Numbers(keys["turn"]), shift = Numbers(keys["shift"]);
    if (turn.size() != 9 || shift.size() != 3) { problem = "pose.txt missing or unreadable (cut off before it was finished?)"; return false; }
    scan.cloud.resize((size_t)width * height);
    std::ifstream cloud(folder / "cloud.bin", std::ios::binary);
    cloud.read((char*)scan.cloud.data(), scan.cloud.size() * sizeof(cv::Point3f));
    if (!cloud || cloud.gcount() != (std::streamsize)(scan.cloud.size() * sizeof(cv::Point3f))) { problem = "cloud.bin missing or short"; return false; }
    if (cloud.peek() != std::char_traits<char>::eof()) { problem = "cloud.bin is longer than room.txt's depth_size says"; return false; }
    if (picture)
    {
        scan.picture = cv::imread((folder / "colour.bmp").string(), cv::IMREAD_COLOR);
        if (scan.picture.size() != cv::Size(width, height)) { problem = "colour.bmp missing or the wrong size"; return false; }
    }
    size_t readings = 0;
    for (cv::Point3f& p : scan.cloud)
    {
        if (!(p.z > 0)) p = cv::Point3f(NAN, NAN, NAN);   // bug 4: z = 0 is no reading, never a point at the camera
        else readings++;
    }
    if (info.format == 2 && (keys["points"].empty() || std::strtoull(keys["points"].c_str(), nullptr, 10) != readings))
    {
        problem = cv::format("cloud.bin holds %zu readings, pose.txt says %s (not all on the disk before a crash?)", readings,
                             keys["points"].empty() ? "nothing" : keys["points"].c_str());
        return false;
    }
    for (int i = 0; i < 9; i++) scan.pose.turn(i / 3, i % 3) = turn[i];
    scan.pose.shift = cv::Point3f(shift[0], shift[1], shift[2]);
    if (info.format == 1)   // mirrored: flip each row of the cloud and picture, x -> -x, the turn S R S
    {
        for (int row = 0; row < height; row++)
        {
            std::reverse(scan.cloud.begin() + (size_t)row * width, scan.cloud.begin() + (size_t)(row + 1) * width);
            for (int column = 0; column < width; column++) scan.cloud[(size_t)row * width + column].x *= -1;
        }
        if (picture) cv::flip(scan.picture, scan.picture, 1);
        for (int i = 0; i < 9; i++) if ((i / 3 == 0) != (i % 3 == 0)) scan.pose.turn(i / 3, i % 3) *= -1;
        scan.pose.shift.x *= -1;
    }
    scan.excluded = keys["excluded"] == "1";
    scan.suspicious = keys["suspicious"] == "1";
    scan.stats = keys["stats"];
    scan.surface = SampleSurface(scan.cloud, width, ScanOptions().icpStep);   // bug 3: the same rule as when scanned
    return true;
}

bool SetExcluded(const fs::path& scanFolder, bool excluded)
{
    std::ifstream in(scanFolder / "pose.txt");
    if (!in) return false;
    std::string text, line;
    while (std::getline(in, line))
        text += (line.rfind("excluded ", 0) == 0 ? "excluded " + std::to_string((int)excluded) : line) + "\n";
    in.close();
    return ReplaceText(scanFolder / "pose.txt", text);   // bug 5: atomic
}

std::vector<Landmark> LoadLandmarks(const fs::path& room, const RoomInfo& info)
{
    std::vector<Landmark> landmarks;
    std::ifstream file(room / "landmarks.txt");
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        Landmark l;
        std::string hex;
        if (!(in >> l.point.x >> l.point.y >> l.point.z >> l.seen)) continue;
        if (info.format == 1) l.point.x *= -1;
        if (in >> hex && hex.size() % 2 == 0 && hex.find_first_not_of("0123456789abcdefABCDEF") == std::string::npos)   // else no descriptor
        {
            l.descriptor = cv::Mat(1, (int)hex.size() / 2, CV_8U);
            for (int b = 0; b < l.descriptor.cols; b++) l.descriptor.at<uchar>(0, b) = (uchar)std::stoi(hex.substr(2 * b, 2), nullptr, 16);
        }
        landmarks.push_back(l);
    }
    return landmarks;
}

// --- Where a room goes (D48) ---

//std::string Comparable(const fs::path& path){
//
//@param1 - path: a path
//
//@brief - the path made full and tidy (as far as it exists), with / separators, lower case on Windows (its file
//         names ignore case), for comparing paths. O(depth).
static std::string Comparable(const fs::path& path)
{
    std::error_code error;
    fs::path full = fs::weakly_canonical(fs::absolute(path, error), error);
    if (error) full = fs::absolute(path, error);
    std::string text = full.lexically_normal().generic_string();
    while (text.size() > 1 && text.back() == '/') text.pop_back();
#ifdef _WIN32
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return (char)std::tolower(c); });
#endif
    return text;
}

//bool Inside(const fs::path& path, const fs::path& folder){
//
//@param1 - path: a path
//@param2 - folder: a folder
//
//@brief - true if the path is the folder or anywhere inside it. O(depth).
static bool Inside(const fs::path& path, const fs::path& folder)
{
    const std::string p = Comparable(path), f = Comparable(folder);
    return p == f || (p.size() > f.size() && p.compare(0, f.size(), f) == 0 && p[f.size()] == '/');
}

//FolderCheck Refused(FolderCheck check, const std::string& problem){
//
//@brief - the check, refused for that reason. O(1).
static FolderCheck Refused(FolderCheck check, const std::string& problem)
{
    check.ok = false;
    check.problem = problem;
    return check;
}

//std::string WriteTestFile(const fs::path& file, uintmax_t bytes){
//
//@param1 - file: the test file to create
//@param2 - bytes: how big
//
//@brief - writes that many bytes, pushes them to the disk and reads the size back. Empty if all went well, else
//         the system's reason. The file is left for the caller to remove. O(bytes).
static std::string WriteTestFile(const fs::path& file, uintmax_t bytes)
{
#ifdef _WIN32
    FILE* out = _wfopen(file.c_str(), L"wb");
#else
    FILE* out = std::fopen(file.c_str(), "wb");
#endif
    if (!out) return std::string("a file can't be created there (") + std::strerror(errno) + ")";
    std::vector<char> block(1 << 20, 0);
    std::string problem;
    for (uintmax_t left = bytes; left > 0 && problem.empty(); )
    {
        const size_t n = (size_t)std::min<uintmax_t>(left, block.size());
        if (std::fwrite(block.data(), 1, n, out) != n) problem = std::string("writing failed (") + std::strerror(errno) + ")";
        left -= n;
    }
#ifdef _WIN32
    if (problem.empty() && (std::fflush(out) != 0 || _commit(_fileno(out)) != 0)) problem = std::string("it can't be pushed to the disk (") + std::strerror(errno) + ")";
#else
    if (problem.empty() && (std::fflush(out) != 0 || fsync(fileno(out)) != 0)) problem = std::string("it can't be pushed to the disk (") + std::strerror(errno) + ")";
#endif
    if (std::fclose(out) != 0 && problem.empty()) problem = std::string("closing the file failed (") + std::strerror(errno) + ")";
    std::error_code error;
    if (problem.empty() && fs::file_size(file, error) != bytes) problem = "the test file came back the wrong size";
    return problem;
}

FolderCheck CheckRoomFolder(const fs::path& room, int scansToWrite, bool quick)
{
    FolderCheck check;
    std::error_code error;
    if (room.empty() || !room.is_absolute())
        return Refused(check, "\"" + room.string() + "\" isn't a full path, so it would land in whatever folder the program runs from. Choose a full path, e.g. in the repo's room_scans folder.");
    const fs::path full = room.lexically_normal();
#ifdef SPOTTER_BUILD_DIR
    if (Inside(full, fs::u8path(SPOTTER_BUILD_DIR)))
        return Refused(check, full.string() + " is inside the build folder (" + fs::u8path(SPOTTER_BUILD_DIR).make_preferred().string() +
                              "), which a clean rebuild deletes. Choose a folder outside it, e.g. the repo's room_scans folder.");
#endif
    for (fs::path above = full.parent_path(); !above.empty(); above = above.parent_path())
    {
        if (fs::exists(above / "room.txt", error))
            return Refused(check, full.string() + " is inside the saved room " + above.string() + ", where 13z, 13w and the demos wouldn't find it. Choose a folder outside it.");
        if (above == above.parent_path()) break;
    }
    if (fs::exists(full, error) && !fs::is_directory(full, error))
        return Refused(check, "There's a file called " + full.string() + " already. Choose another name.");
#ifdef _WIN32
    const size_t longest = full.native().size() + std::wstring(L"\\scan_1000.new\\colour_index.bin").size();
    if (longest >= 260)
        return Refused(check, cv::format("The path is too long for Windows: its files would need %zu characters (the limit is 259). Choose a shorter folder or name.", longest));
#endif
    std::vector<fs::path> created;   // deepest first, removed again at the end
    for (fs::path p = full; !p.empty() && !fs::exists(p, error); p = p.parent_path())
    {
        created.push_back(p);
        if (p == p.parent_path()) break;
    }
    fs::create_directories(full, error);
    if (error) return Refused(check, "The folder " + full.string() + " can't be created (" + error.message() + "). Is the drive there, and may you write to it?");
    const fs::path test = full / ".spotter_write_test";
    const std::string problem = WriteTestFile(test, quick ? 64 * 1024 : RoomBytesPerScan);
    fs::remove(test, error);
    const fs::space_info space = fs::space(full, error);
    check.freeBytes = error ? 0 : space.available;
    for (const fs::path& p : created) fs::remove(p, error);   // only removes empty folders
    if (!problem.empty())
        return Refused(check, "A test save into " + full.string() + " failed: " + problem + ". Choose another folder (or drive).");
    check.scansThatFit = (int)std::min<uintmax_t>(check.freeBytes / RoomBytesPerScan, 1000000);
    const double megabytes = check.freeBytes / 1e6;
    if (check.scansThatFit < scansToWrite + 5)
        return Refused(check, cv::format("Only %.0f MB free on that drive: room for %d scans, and this needs %d plus some to spare (about %.0f MB a scan). Free some space or choose another drive.",
                                         megabytes, check.scansThatFit, scansToWrite, RoomBytesPerScan / 1e6));
    if (check.scansThatFit < 60)
        check.warning = cv::format("Only %.0f MB free on that drive: room for about %d scans.", megabytes, check.scansThatFit);
    check.ok = true;
    return check;
}

std::vector<fs::path> RescueFolders()
{
    const std::string stamp = Now("%Y-%m-%d_%H%M%S");
    std::vector<fs::path> folders;
#ifdef SPOTTER_REPO_ROOT
    folders.push_back((fs::u8path(SPOTTER_REPO_ROOT) / "room_scans" / ("rescue_" + stamp)).make_preferred());
#endif
#ifdef _WIN32
    const char* home = std::getenv("USERPROFILE");
#else
    const char* home = std::getenv("HOME");
#endif
    if (home && *home) folders.push_back(fs::u8path(home) / "Documents" / "Spotter rescue" / ("rescue_" + stamp));
    std::error_code error;
    const fs::path temp = fs::temp_directory_path(error);
    if (!error) folders.push_back(temp / ("spotter_rescue_" + stamp));
    return folders;
}

fs::path AskRoomFolder(const std::string& defaultFolder, const fs::path& current, int scansToWrite)
{
    fs::path folder = current.empty() ? fs::absolute(fs::u8path(defaultFolder)).lexically_normal() : current.parent_path();
    std::string name = current.empty() ? Now("room_%Y-%m-%d_%H%M") : current.filename().string();
    const std::string title = current.empty() ? "Where should this room be saved?" : "Save the room as";
    while (true)
    {
        fs::path where = AskSaveAs(title, folder, name, "Spotter room folder", "");
        if (where.empty()) return {};
        where = where.lexically_normal();
        folder = where.parent_path();   // a second try opens where the first one was
        name = where.filename().string();
        std::error_code error;
        const bool same = !current.empty() && Comparable(where) == Comparable(current);
        if (!same && fs::exists(where, error))
        {
            if (!fs::is_directory(where, error)) { Tell("Can't save the room there", "There's a file called " + where.string() + " already. Choose another name."); continue; }
            if (fs::exists(where / "room.txt", error))
            {
                if (!Confirm("Replace a saved room?", where.string() + " is a saved room already.\n\nDelete it and save this room there instead?")) continue;
                fs::remove_all(where, error);
                if (error) { Tell("Couldn't delete the old room", where.string() + ": " + error.message() + ". Choose another name."); continue; }
            }
            else if (!fs::is_empty(where, error))
            {
                Tell("Can't save the room there", where.string() + " is there already and holds other files (it isn't a saved room), so it's left alone. Choose a new name.");
                continue;
            }
        }
        const FolderCheck check = CheckRoomFolder(where, scansToWrite);
        if (!check.ok) { Tell("Can't save the room there", check.problem); continue; }
        if (!check.warning.empty() && !Confirm("Little free space", check.warning + "\n\nSave there anyway?")) continue;
        std::printf("The room goes to %s (room for about %d scans there).\n", where.string().c_str(), check.scansThatFit);
        return where;
    }
}

}
