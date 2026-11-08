// =====================================================================
//  13z · Room Rebuild                                  (no Kinect needed)
//
//  Rebuilds rooms from the folders 13e writes as it scans (and legacy
//  13c/13d/13e's), any time later. No scanning and no placing: each scan's
//  points, picture and placement are read back (spotter::LoadScan). A scan
//  cut off by a crash (no pose.txt, or a cloud that doesn't match its point
//  count) is skipped.
//
//    1. Find     every saved room under SearchFolder (or a folder you give),
//                with its scans, date and size.
//    2. Choose   which rooms, and in each which scans to load (1-12,15).
//    3. Do       view one (X excludes a scan, saved to its pose.txt), or
//                build files, or both:
//                  .obj  the merged mesh + the clean model (spotter::FuseScans,
//                        BuildRoom), plus far landmarks if it has any
//                  .ply  every included scan's own points and colours
//
//  The files land in the room's own folder: room_scan_<time>.obj and
//  points_<time>.ply. Legacy rooms (format 1) were mirrored left-right; they
//  are read back the right way round, and what's built from them is too.
//
//  Run:     13z_Room_Rebuild [folder]   (default: SearchFolder), then answer in the console.
//  Viewer:  mouse drag / W A S D turn   wheel / Z + - zoom   R reset view
//           C colour by scan   click a camera circle to select it
//           X exclude / include   Esc deselect   E build .obj   K build .ply   N next room   Q quit
//
//  Privacy: the folders and what this builds are captures of a room in
//  colour. room_scans/, *.obj and *.ply are gitignored: never commit them.
// =====================================================================

#include <chrono>
#include <cstdio>
#include <set>
#include <sstream>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/room.hpp"
#include "spotter/room_folder.hpp"
#include "spotter/track.hpp"   // ColourForId
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "mouse drag / W A S D  turn   wheel / Z + -  zoom   R  reset view   C  colour by scan   click a camera  select it   X  exclude / include   Esc  deselect   E  build .obj   K  build .ply   N  next room   Q  quit";

namespace fs = std::filesystem;

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

const std::string SearchFolder = "room_scans";   // where 13e saves by default; relative = where it runs
const int SearchDepth = 3;                       // how many folders deep to look for saved rooms
const int ShowStep = 2;                          // the viewer draws every 2nd pixel of each scan
const size_t MaxShownPoints = 600000;
const float PivotMeters = 2.5f;

// The merge can be finer than live: time doesn't matter offline. 0.01 = 1 cm cubes.
spotter::RoomOptions RoomSettings() { spotter::RoomOptions o; o.voxelMeters = 0.02f; return o; }

// ---------------------------------------------------------------------
//  The new idea: choosing rooms and scans in the console
// ---------------------------------------------------------------------

// "1-12,15" (or "all", or nothing = all) as numbers from 1 to count. False if it can't be read.
bool ParseList(const std::string& text, int count, std::vector<int>& numbers)
{
    std::set<int> chosen;
    std::string list = text;
    std::replace(list.begin(), list.end(), ',', ' ');
    std::istringstream in(list);
    std::string token;
    bool any = false;
    while (in >> token)
    {
        any = true;
        if (token == "all") { for (int n = 1; n <= count; n++) chosen.insert(n); continue; }
        size_t dash = token.find('-');
        int first = std::atoi(token.substr(0, dash).c_str()), last = dash == std::string::npos ? first : std::atoi(token.substr(dash + 1).c_str());
        if (first < 1 || last > count || first > last) return false;
        for (int n = first; n <= last; n++) chosen.insert(n);
    }
    if (!any) for (int n = 1; n <= count; n++) chosen.insert(n);
    numbers.assign(chosen.begin(), chosen.end());
    return !numbers.empty();
}

// Step 1: the rooms to work on (empty = quit).
std::vector<spotter::SavedRoom> PickRooms(fs::path searchIn)
{
    while (true)
    {
        std::vector<spotter::SavedRoom> rooms = spotter::FindRooms(searchIn, SearchDepth);
        std::printf("\nSaved rooms in %s:\n", searchIn.string().c_str());
        if (rooms.empty()) std::printf("  none (13e saves them; give another folder below)\n");
        for (size_t r = 0; r < rooms.size(); r++)
            std::printf("  %2zu  %-28s %-24s %-16s  format %d  %3d scans (%d excluded)  %5.0f MB\n", r + 1, rooms[r].folder.filename().string().c_str(),
                        rooms[r].program.c_str(), rooms[r].saved.c_str(), rooms[r].format, rooms[r].scans, rooms[r].excluded, rooms[r].megabytes);
        std::string answer = spotter::Ask(rooms.empty() ? "Folder to look in (q quits): " : "Which rooms? (2   1,3   1-4   Enter = all; or a folder to look in; q quits): ");
        if (answer == "q" || answer == "cancel") return {};
        std::error_code error;
        if (!answer.empty() && fs::is_directory(answer, error)) { searchIn = fs::absolute(answer); continue; }
        std::vector<int> picks;
        if (!rooms.empty() && ParseList(answer, (int)rooms.size(), picks))
        {
            std::vector<spotter::SavedRoom> chosen;
            for (int n : picks) chosen.push_back(rooms[n - 1]);
            return chosen;
        }
        std::printf("Didn't understand \"%s\".\n", answer.c_str());
    }
}

// Step 2: one room's chosen scans, read back. False = skip this room.
struct Loaded
{
    spotter::RoomInfo info;
    std::vector<spotter::Scan> scans;
    std::vector<fs::path> folders;   // each loaded scan's folder (X writes there)
    std::vector<cv::Point3f> landmarks;
};

bool LoadRoom(const spotter::SavedRoom& room, Loaded& loaded)
{
    std::string problem;
    if (!spotter::ReadRoomInfo(room.folder, loaded.info, problem)) { std::printf("\n%s: %s - skipped.\n", room.folder.string().c_str(), problem.c_str()); return false; }
    std::vector<fs::path> folders = spotter::ScanFolders(room.folder);
    std::printf("\n=== %s (%s, %s, format %d%s): %zu scans ===\n", room.folder.filename().string().c_str(), loaded.info.program.c_str(),
                loaded.info.saved.c_str(), loaded.info.format, loaded.info.format == 1 ? ", mirrored: read back the right way round" : "", folders.size());
    std::vector<int> chosen;
    while (true)
    {
        std::string answer = spotter::Ask("Which scans to load? (1-12,15  Enter = all; skip = next room): ");
        if (answer == "skip" || answer == "cancel") return false;
        if (ParseList(answer, (int)folders.size(), chosen)) break;
        std::printf("Numbers from 1 to %zu, please.\n", folders.size());
    }
    for (int n : chosen)
    {
        spotter::Scan scan;
        int number;
        if (spotter::LoadScan(folders[n - 1], loaded.info, scan, number, problem)) { loaded.scans.push_back(std::move(scan)); loaded.folders.push_back(folders[n - 1]); }
        else std::printf("  %s skipped: %s\n", folders[n - 1].filename().string().c_str(), problem.c_str());
    }
    for (const spotter::Landmark& l : spotter::LoadLandmarks(room.folder, loaded.info)) loaded.landmarks.push_back(l.point);
    int excluded = 0;
    for (const spotter::Scan& s : loaded.scans) excluded += s.excluded;
    std::printf("  loaded %zu scans (%d excluded), %zu far landmarks\n", loaded.scans.size(), excluded, loaded.landmarks.size());
    return !loaded.scans.empty();
}

// ---------------------------------------------------------------------
//  Output: files
// ---------------------------------------------------------------------

std::string Stamp()
{
    char stamp[32];
    std::time_t now = std::time(nullptr);
    std::strftime(stamp, sizeof stamp, "%Y%m%d_%H%M%S", std::localtime(&now));
    return stamp;
}

std::vector<const spotter::Scan*> Included(const Loaded& loaded)
{
    std::vector<const spotter::Scan*> included;
    for (const spotter::Scan& s : loaded.scans) if (!s.excluded) included.push_back(&s);
    return included;
}

void BuildObj(const fs::path& room, const Loaded& loaded)
{
    std::vector<const spotter::Scan*> included = Included(loaded);
    if (included.empty()) { std::printf("  .obj: every loaded scan is excluded - nothing to merge.\n"); return; }
    std::printf("  Merging %zu scans...\n", included.size());
    auto start = std::chrono::steady_clock::now();
    std::mt19937 random(1);
    float voxel;
    const spotter::RoomOptions options = RoomSettings();
    spotter::Room built = spotter::BuildRoom(included, spotter::FuseScans(included, loaded.info.depthCamera, options, voxel), random, options);
    std::string path = (room / ("room_scan_" + Stamp() + ".obj")).string();
    if (!spotter::SaveRoomObj(path, built, loaded.landmarks, "Spotter 13z: seen_room = every scan merged, clean_* and furniture_blocks = the tidy (partly guessed) model"))
    {
        std::printf("  Could not write %s\n", path.c_str());
        return;
    }
    std::printf("  Saved %s (%.1f MB, %.1f s). A capture of the room: never commit it.\n", path.c_str(), fs::file_size(path) / 1e6,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    std::printf("    merged mesh: %zu triangles, %.1f cm cubes; up from %s; %.2f m high (floor %s, ceiling %s); %zu walls, %zu furniture blocks\n",
                built.mesh.groups["seen_room"].size(), voxel * 100, built.upFrom.c_str(), built.ceilingY - built.floorY,
                built.floorSeen ? "seen" : "guessed", built.ceilingSeen ? "seen" : "guessed", built.walls.size(), built.blocks.size());
}

void BuildPly(const fs::path& room, const Loaded& loaded)
{
    std::vector<const spotter::Scan*> included = Included(loaded);
    if (included.empty()) { std::printf("  .ply: every loaded scan is excluded - nothing to write.\n"); return; }
    std::mt19937 random(1);
    const cv::Matx33f turn = spotter::BuildRoom(included, spotter::Mesh(), random, RoomSettings()).firstCamera;   // levelled as the .obj
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colours;
    spotter::RoomPoints(included, turn, points, colours);
    std::string path = (room / ("points_" + Stamp() + ".ply")).string();
    if (!spotter::SavePly(path, points, colours)) { std::printf("  Could not write %s\n", path.c_str()); return; }
    std::printf("  Saved %s: %zu points from %zu scans (%.0f MB). A capture of the room: never commit it.\n", path.c_str(), points.size(),
                included.size(), fs::file_size(path) / 1e6);
}

// ---------------------------------------------------------------------
//  Output: the viewer
// ---------------------------------------------------------------------

void Draw(const Loaded& loaded, const std::vector<std::vector<cv::Point3f>>& shown, const std::vector<std::vector<cv::Vec3b>>& colours,
          int selected, bool byScan, spotter::OrbitView& view, spotter::PointRenderer& renderer, const std::string& window)
{
    cv::Mat& image = renderer.Begin(view);
    size_t total = 0;
    for (const auto& points : shown) total += points.size();
    for (size_t i = 0; i < loaded.scans.size(); i++)
    {
        cv::Scalar c = spotter::ColourForId((int)i + 1);
        if (loaded.scans[i].excluded) renderer.Draw(shown[i], cv::Vec3b(90, 90, 90), 2, total / MaxShownPoints + 1);
        else if (byScan) renderer.Draw(shown[i], cv::Vec3b((uchar)c[0], (uchar)c[1], (uchar)c[2]), 2, total / MaxShownPoints + 1);
        else renderer.Draw(shown[i], colours[i], 2, total / MaxShownPoints + 1);
    }
    for (size_t i = 0; i < loaded.scans.size(); i++)
    {
        const spotter::Scan& scan = loaded.scans[i];
        cv::Scalar colour = (int)i == selected ? cv::Scalar(255, 255, 255) : scan.excluded ? cv::Scalar(110, 110, 110)
                          : scan.suspicious ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 255);
        spotter::DrawCamera(image, view, scan.pose, loaded.info.depthCamera, colour, (int)i == selected ? 2 : 1, loaded.folders[i].filename().string().substr(5));
    }
    for (const cv::Point3f& l : loaded.landmarks)
    {
        cv::Point pixel;
        float depth;
        if (view.Project(l, pixel, depth)) cv::circle(image, pixel, 3, cv::Scalar(255, 0, 255), cv::FILLED);
    }
    cv::putText(image, selected >= 0 ? cv::format("selected %s: X %s, Esc deselect", loaded.folders[selected].filename().string().c_str(),
                loaded.scans[selected].excluded ? "include" : "exclude") : std::string("E build .obj   K build .ply   N next room   Q quit"),
                cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
    cv::imshow(window, spotter::WithKeys(image, Keys));
}

// One room in the viewer. False when Q was pressed (stop everything), true for N or the window closed.
bool View(const fs::path& room, Loaded& loaded)
{
    std::vector<std::vector<cv::Point3f>> shown(loaded.scans.size());
    std::vector<std::vector<cv::Vec3b>> colours(loaded.scans.size());
    for (size_t s = 0; s < loaded.scans.size(); s++)
        for (int row = 0; row < loaded.info.depthCamera.height; row += ShowStep)
            for (int column = 0; column < loaded.info.depthCamera.width; column += ShowStep)
            {
                const cv::Point3f& p = loaded.scans[s].cloud[row * loaded.info.depthCamera.width + column];
                if (std::isnan(p.z)) continue;
                shown[s].push_back(loaded.scans[s].pose(p));
                colours[s].push_back(loaded.scans[s].picture.at<cv::Vec3b>(row, column));
            }
    const std::string window = "13z " + room.filename().string() + "  (drag/WASD turn, click a camera, X exclude, E .obj, K .ply, N next, Q quit)";
    cv::namedWindow(window);
    spotter::OrbitView view(spotter::OrbitView::Options(), cv::Point3f(0, 0, PivotMeters), PivotMeters);
    view.Attach(window);
    spotter::PointRenderer renderer;
    int selected = -1;
    bool byScan = false;
    while (true)
    {
        Draw(loaded, shown, colours, selected, byScan, view, renderer, window);
        int key = cv::waitKey(15);
        cv::Point click;
        if (view.TakeClick(click))
        {
            selected = -1;
            double nearest = 15;
            for (size_t i = 0; i < loaded.scans.size(); i++)
            {
                cv::Point pixel;
                float depth;
                if (view.Project(loaded.scans[i].pose.shift, pixel, depth) && cv::norm(pixel - click) < nearest) { nearest = cv::norm(pixel - click); selected = (int)i; }
            }
            if (selected >= 0) std::printf("Selected %s: %s\n", loaded.folders[selected].filename().string().c_str(), loaded.scans[selected].stats.c_str());
        }
        if (key == 'q') { cv::destroyWindow(window); return false; }
        if (key == 'n' || cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1) break;
        if (key == 'x' && selected >= 0)
        {
            spotter::Scan& scan = loaded.scans[selected];
            scan.excluded = !scan.excluded;
            bool saved = spotter::SetExcluded(loaded.folders[selected], scan.excluded);
            std::printf("%s %s%s\n", loaded.folders[selected].filename().string().c_str(), scan.excluded ? "excluded" : "included again", saved ? " (saved)" : " (could NOT be saved)");
        }
        else if (key == 'e') BuildObj(room, loaded);
        else if (key == 'k') BuildPly(room, loaded);
        else if (key == 'c') byScan = !byScan;
        else if (key == 27) selected = -1;
        else if (key == '+' || key == '=') view.distance -= view.options.zoomStepMeters;
        else if (key == '-') view.distance += view.options.zoomStepMeters;
        else view.HandleKey(key);
    }
    cv::destroyWindow(window);
    return true;
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<spotter::SavedRoom> rooms = PickRooms(fs::absolute(argc > 1 ? fs::path(argv[1]) : fs::path(SearchFolder)));
    if (rooms.empty()) return 0;
    std::string what;
    while (what != "1" && what != "2" && what != "3" && what != "4")
    {
        what = spotter::Ask("What to do? 1 view (build from the viewer)  2 build .obj  3 build .ply  4 build both  [Enter = 1]: ");
        if (what.empty()) what = "1";
        if (what == "q" || what == "cancel") return 0;
    }
    for (const spotter::SavedRoom& room : rooms)
    {
        Loaded loaded;
        if (!LoadRoom(room, loaded)) continue;
        if (what == "1" && !View(room.folder, loaded)) break;
        if (what == "2" || what == "4") BuildObj(room.folder, loaded);
        if (what == "3" || what == "4") BuildPly(room.folder, loaded);
    }
    std::printf("\nDone.\n");
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Build the same room at voxelMeters 0.02 and 0.01 (RoomSettings). How much sharper is the .obj,
//     how much bigger, how much slower? A big room at 1 cm may hit maxVoxels and coarsen by itself.
//  2. Load only the first half of a room's scans, then the second half, and build both. Where does
//     each have holes?
//  3. In the viewer, press C and look for a scan that shows as a shifted copy. Exclude it (X), build
//     again (E): are the double walls gone?
//  4. Open a legacy (format 1) room in a legacy 13z and in this one: the two .obj files are mirror
//     images. Which one matches the real room?
