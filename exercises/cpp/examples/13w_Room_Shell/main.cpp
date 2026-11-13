// =====================================================================
//  13w · Room Shell                         (no Kinect; after 13e)
//
//  Just the room itself: every wall from the floor to the ceiling, the
//  floor and the ceiling, each a flat surface with its own picture
//  (a texture), as an .obj you can open in Blender or MeshLab. No
//  furniture: what stood in front of a surface is left out, and the
//  surface behind it is filled in with that surface's own pattern.
//
//    1. Load      a 13e room folder: every included scan, with its full
//                 1920x1080 colour picture.
//    2. Surfaces  (new) the walls are BuildRoom's (09c across every scan),
//                 now full height. The floor is FindTwinFloor's (D33). The
//                 ceiling is the HIGHEST height where level readings cover
//                 half a square metre: a ceiling fan or a lamp hangs lower
//                 and covers less, so it never wins. The floor and ceiling
//                 follow the walls' outline.
//    3. Texture   (new) every 1 cm of every surface asks each scan: "did you
//                 see this spot?" Yes if the scan's depth there agrees with
//                 the surface (not a couch in front of the wall, not the fan
//                 under the ceiling), the colour camera saw the same spot,
//                 and the view wasn't too slanted. Of the scans that saw it,
//                 the one looking most head-on and nearest gives the colour,
//                 from its full colour picture (about 4x sharper than the
//                 depth grid). A colour unlike the others' (a fan blade)
//                 is dropped first.
//    4. Fill      (new) what no scan saw (behind the couch, above the fan,
//                 under the table) gets two things: the slow changes of
//                 colour (light and shadow) inpainted from around, and the
//                 surface's own pattern (wood grain, carpet, plaster)
//                 copied in blocks from its plainest seen square, each
//                 block picked to continue its neighbours (image quilting).
//    5. Look      walls you'd see from outside are left out, as in a doll's
//                 house; G tints what was filled in.
//    6. Save      K writes shell_<date>_<time>\ into the room's folder:
//                 room_shell.obj, room_shell.mtl and a .bmp per surface.
//
//  Simple rooms only: walls upright, floor and ceiling level (D32). Doors
//  and windows get filled shut: "full walls".
//
//  Keys:   G  filled parts tinted / true colour   K  save the .obj
//          mouse drag / W A S D  turn   wheel / Z X  zoom   R reset   Q or Esc  quit
//  Run:    13w_Room_Shell [room folder]   (none: an Open window starting in RoomsFolder, the newest room selected)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "room_shell.hpp"   // steps 2-4 (new), also used by check_room_shell
#include "spotter/room.hpp"
#include "spotter/room_folder.hpp"
#include "spotter/room_twin.hpp"
#include "spotter/view.hpp"

namespace fs = std::filesystem;

// The keys, shown in a strip below the window (spotter::WithKeys).
const std::string Keys = "G  filled parts tinted / true colour   K  save the .obj   drag / W A S D  turn   wheel / Z X  zoom   R  reset   Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

const std::string RoomsFolder = SPOTTER_REPO_ROOT "/room_scans";   // 13e's default (gitignored, outside build/)

// --- The surfaces, texturing and filling (new): their settings are in room_shell.hpp. ---

// --- Output ---
const float QuadMeters = 0.10f;                 // the floor and ceiling as squares this big, so any outline works
const int ViewStep = 2;                         // the viewer draws every 2nd texel each way
const float ViewPitchDegrees = 35;
const cv::Vec3b MarkColour(255, 128, 0);        // B, G, R: filled-in parts tinted blue (G)
const std::string Window = "13w - room shell";

// ---------------------------------------------------------------------
//  Step 1: load the scans
// ---------------------------------------------------------------------

//bool LoadShell(int argc, char** argv, Shell& shell){
//
//@param1 - argc: how many command-line words there are
//@param2 - argv: the words: an optional room folder
//@param3 - shell: filled with the folder, info and shots
//
//@brief - step 1: the room folder given, or one chosen in an Open window (ChooseRoom). Refuses an old (format 1, mirrored)
//         room. Loads every included scan (LoadScan) with its full colour picture (colour_full.bmp), its colour
//         index and the colour model fitted from them; scans without the full picture fall back to the depth
//         grid's colour. False if nothing loaded. O(scans x pixels).
bool LoadShell(int argc, char** argv, Shell& shell)
{
    shell.folder = spotter::ChooseRoom(argc, argv, RoomsFolder, "13w");
    std::string problem;
    if (shell.folder.empty())
    {
        std::printf("No room to build. Scan one with 13e_Room_Scan, or name its folder: 13w_Room_Shell <folder>\n");
        return false;
    }
    if (!spotter::ReadRoomInfo(shell.folder, shell.info, problem)) { std::printf("Can't open %s: %s\n", shell.folder.string().c_str(), problem.c_str()); return false; }
    if (shell.info.format != 2) { std::printf("An old (format 1, mirrored) room: scan it again with 13e_Room_Scan.\n"); return false; }
    int sharp = 0;
    for (const fs::path& dir : spotter::ScanFolders(shell.folder))
    {
        Shot shot;
        int number = 0;
        if (!spotter::LoadScan(dir, shell.info, shot.scan, number, problem)) { std::printf("  %s skipped: %s\n", dir.filename().string().c_str(), problem.c_str()); continue; }
        if (shot.scan.excluded) continue;
        shot.full = cv::imread((dir / "colour_full.bmp").string(), cv::IMREAD_COLOR);
        shot.colourIndex.resize(shot.scan.cloud.size());
        std::ifstream index(dir / "colour_index.bin", std::ios::binary);
        if (!index.read((char*)shot.colourIndex.data(), shot.colourIndex.size() * sizeof(int))) shot.colourIndex.clear();
        if (!shot.full.empty() && !shot.colourIndex.empty()) shot.model = spotter::FitColourModel(shot.scan.cloud, shot.colourIndex, shot.full.cols);
        sharp += shot.model.ok;
        shell.shots.push_back(std::move(shot));
    }
    std::printf("%s: %zu scans, %d with their full colour picture (the rest use the depth grid's colour)\n", shell.folder.string().c_str(),
                shell.shots.size(), sharp);
    if (shell.shots.empty()) std::printf("No scan in it could be used (each one's reason is above; excluded scans are left out).\n");
    return !shell.shots.empty();
}

// ---------------------------------------------------------------------
//  Steps 2-4 (new): the surfaces, their texture and the fill are in room_shell.hpp, beside this file
//  (FindSurfaces, Texture, Fill), so check_room_shell runs the same code.
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
//  Steps 5-6: look, save
// ---------------------------------------------------------------------

//cv::Point3f ForView(const cv::Point3f& p){
//
//@param1 - p: a point in the room (Y up, Z towards the first camera)
//
//@brief - the room -> the viewer's coordinates (Y down, Z ahead): turned 180 degrees about X, (x, y, z) -> (x, -y,
//         -z), as 09b's flip, so the room is seen from in front. It is its own inverse. O(1).
cv::Point3f ForView(const cv::Point3f& p) { return { p.x, -p.y, -p.z }; }

//void ReadyToView(Surface& s){
//
//@param1 - s: a filled surface
//
//@brief - the points the viewer draws: every ViewStep-th texel inside the surface, with its point (turned with
//         ForView) and colour, and a second colour for the marked view (a filled texel blended 40% with
//         MarkColour). O(texels / ViewStep^2).
void ReadyToView(Surface& s)
{
    for (int row = 0; row < s.inside.rows; row += ViewStep)
        for (int column = 0; column < s.inside.cols; column += ViewStep)
        {
            if (!s.inside.at<uchar>(row, column)) continue;
            const cv::Vec3b colour = s.colour.at<cv::Vec3b>(row, column);
            s.viewPoints.push_back(ForView(Centre(s, row, column)));
            s.viewColours.push_back(colour);
            s.viewMarked.push_back(s.filled.at<uchar>(row, column) ? cv::Vec3b(cv::Vec3f(colour) * 0.6f + cv::Vec3f(MarkColour) * 0.4f) : colour);
        }
}

//cv::Point3f Eye(const spotter::OrbitView& view){
//
//@param1 - view: the viewer
//
//@brief - where the viewer's eye is, in the room (OrbitView::Project, backwards): pivot + distance x (cos pitch
//         sin yaw, -sin pitch, -cos pitch cos yaw), turned back with ForView. O(1).
cv::Point3f Eye(const spotter::OrbitView& view)
{
    const float r = (float)CV_PI / 180, d = view.distance;
    const float cy = std::cos(view.yawDegrees * r), sy = std::sin(view.yawDegrees * r), cp = std::cos(view.pitchDegrees * r), sp = std::sin(view.pitchDegrees * r);
    return ForView(view.pivot + cv::Point3f(d * cp * sy, -d * sp, -d * cp * cy));
}

//void Show(const Shell& shell, bool marked, spotter::OrbitView& view, spotter::PointRenderer& renderer){
//
//@param1 - shell: the surfaces
//@param2 - marked: true = show the filled texels marked
//@param3 - view: the viewer
//@param4 - renderer: draws the points
//
//@brief - the surfaces as points, leaving out any surface seen from outside (its back, towards the eye); shown
//         with the key strip. O(shown points).
void Show(const Shell& shell, bool marked, spotter::OrbitView& view, spotter::PointRenderer& renderer)
{
    const cv::Point3f eye = Eye(view);
    cv::Mat& image = renderer.Begin(view, cv::Scalar(20, 20, 20));
    for (const Surface& s : shell.surfaces)
        if ((eye - s.origin).dot(s.inward) > 0)   // a surface seen from outside is left out: a doll's house
            renderer.Draw(s.viewPoints, marked ? s.viewMarked : s.viewColours, 2);
    cv::putText(image, marked ? "blue tint = filled in (no scan saw it)" : "true colour", cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                marked ? cv::Scalar(MarkColour) : cv::Scalar(200, 200, 200), 1);
    cv::imshow(Window, spotter::WithKeys(image, Keys));
}

//fs::path Save(const Shell& shell){
//
//@param1 - shell: the finished surfaces
//
//@brief - shell_<date>_<time>\ in the room's folder: the .obj (levelled, Y up, metres: the floor and ceiling as
//         QuadMeters quads, a wall as one quad, counter-clockwise from inside), its .mtl, and a .bmp per surface.
//         Returns the folder. O(texels + quads).
fs::path Save(const Shell& shell)
{
    char stamp[32];
    std::time_t now = std::time(nullptr);
    std::strftime(stamp, sizeof stamp, "%Y-%m-%d_%H%M%S", std::localtime(&now));
    const fs::path dir = shell.folder / (std::string("shell_") + stamp);
    fs::create_directories(dir);
    std::ofstream obj(dir / "room_shell.obj"), mtl(dir / "room_shell.mtl");
    obj << "# 13w room shell: walls, floor and ceiling. Levelled: Y up, metres; spotter format 2 (not mirrored)\nmtllib room_shell.mtl\n";
    bool ok = true;
    int vertices = 0;
    for (const Surface& s : shell.surfaces)
    {
        ok = cv::imwrite((dir / (s.name + ".bmp")).string(), s.colour) && ok;
        mtl << "newmtl " << s.name << "\nKa 1 1 1\nKd 1 1 1\nmap_Kd " << s.name << ".bmp\n\n";
        obj << "o " << s.name << "\nusemtl " << s.name << "\n";
        const int rows = s.inside.rows, columns = s.inside.cols;
        const int q = s.wall ? std::max(rows, columns) : std::max(1, (int)std::lround(QuadMeters / s.texel));   // a wall is one quad
        for (int r = 0; r < rows; r += q)
            for (int c = 0; c < columns; c += q)
            {
                const int r1 = std::min(r + q, rows), c1 = std::min(c + q, columns);
                if (!s.inside.at<uchar>((r + r1) / 2, (c + c1) / 2)) continue;
                const int corners[4][2] = { { c, r }, { c, r1 }, { c1, r1 }, { c1, r } };   // counter-clockwise from inside
                for (const auto& k : corners)
                {
                    const cv::Point3f p = s.origin + s.across * (k[0] * s.texel) + s.down * (k[1] * s.texel);
                    obj << cv::format("v %.4f %.4f %.4f\nvt %.5f %.5f\n", p.x, p.y, p.z, (double)k[0] / columns, 1.0 - (double)k[1] / rows);
                }
                obj << cv::format("f %d/%d %d/%d %d/%d %d/%d\n", vertices + 1, vertices + 1, vertices + 2, vertices + 2, vertices + 3, vertices + 3,
                                  vertices + 4, vertices + 4);
                vertices += 4;
            }
    }
    ok = ok && (bool)obj && (bool)mtl;
    std::printf(ok ? "Saved %s\n" : "Could not write everything in %s\n", dir.string().c_str());
    return ok ? dir : fs::path();
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

//int main(int argc, char** argv){
//
//@param1 - argc: how many command-line words there are
//@param2 - argv: the words; the file's header says what it accepts
//
//@brief - the program: 13w · Room Shell (no Kinect; after 13e). Just the room itself: every wall from the floor to
//         the ceiling, the floor and the ceiling, each a flat surface with its own picture (a texture), as an .obj
//         you can open in Blender or MeshLab. Setup, main loop and clean-up are laid out in the file's sections.
//         Returns 0 on a normal exit, 1 if it couldn't start.
int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Shell shell;
    if (!LoadShell(argc, argv, shell) || !FindSurfaces(shell)) return 1;                // steps 1-2
    auto start = std::chrono::steady_clock::now();
    std::mt19937 random(1);
    for (Surface& s : shell.surfaces)                                                   // steps 3-4 (new)
    {
        Texture(s, shell.shots, shell.info.depthCamera);
        Fill(s, random);
        ReadyToView(s);
        const double area = cv::countNonZero(s.inside) * s.texel * s.texel;
        std::printf("  %-8s %5.1f m^2 at %.0f mm texels: %3.0f%% seen, the rest filled from a %d cm square of its own pattern\n", s.name.c_str(),
                    area, s.texel * 1000, 100.0 * cv::countNonZero(s.seen) / std::max(1, cv::countNonZero(s.inside)), (int)std::lround(s.exemplar * s.texel * 100));
    }
    std::printf("Textured in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    shell.shots.clear();   // the pictures aren't needed any more

    cv::namedWindow(Window);
    spotter::OrbitView view(spotter::OrbitView::Options(), ForView(shell.middle), 3 * shell.radius);
    view.pitchDegrees = ViewPitchDegrees;
    view.Attach(Window);
    spotter::PointRenderer renderer;
    bool marked = true;
    while (true)
    {
        Show(shell, marked, view, renderer);                                            // step 5
        int key = cv::waitKey(15);
        if (spotter::IsQuitKey(key) || cv::getWindowProperty(Window, cv::WND_PROP_VISIBLE) < 1) break;
        if (key == 'g') marked = !marked;
        if (key == 'k') Save(shell);                                                    // step 6
        if (view.HandleKey(key) && key == 'r') view.pitchDegrees = ViewPitchDegrees;
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Press G: how much of each wall was really seen? Where would one more scan help most?
//  2. Set MinCeilingAreaMeters2 = 0.05: with a fan, the console's ceiling height may drop to the fan's.
//     That's why the ceiling is the highest BIG level surface, not just the highest.
//  3. Set BaseCellMeters = 0.05 on a wooden floor: the planks under the couch smear into stripes of
//     colour, because the planks are now "slow changes". Over a plank's width, they're pattern.
//  4. Set BlockCandidates = 1: the blocks no longer match their neighbours. Compare the floor under
//     the couch in Blender.
//  5. Set OutlierColour = 255 (nothing dropped) on a room scanned while the fan was spinning.
//  6. A door left open: the wall is filled shut. Where would a door cut-out need to come from
//     (13x's "points far behind a wall")?
//  7. Set TexelMeters = 0.005: sharper? The full colour picture has about 4 mm per pixel at 2 m.
