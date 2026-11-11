// =====================================================================
//  13e · step 2 of 4 · One mesh and a clean model from every scan
//
//  Step 1's scans overlap: the same wall seen five times is five slightly
//  different layers of points. E merges them: a 3D grid of 2 cm cubes, each
//  holding its distance to the nearest surface (+ in front, - just behind),
//  averaged over every scan that saw it; the surface is where that changes
//  sign (spotter::FuseScans, a TSDF and a surface net). Then 09c's clean
//  model across all the scans: floor, ceiling, walls joined at corners,
//  furniture as boxes, the room stood up straight (spotter::BuildRoom).
//  Saved as room_scan_<time>.obj where it runs.
//
//  New in this step: merging scans (TSDF), the clean model from many scans.
//  Keys:  SPACE scan here   E merge and save   drag / W A S D turn   wheel / Z X zoom   R reset   Q quit
//  Next:  13e_s3_Save_The_Folder (the scans written to a folder when you quit)
//  Privacy: an .obj is a capture of the room: never commit one.
// =====================================================================

#include <cstdio>
#include <ctime>
#include <opencv2/highgui.hpp>

#include "spotter/depth.hpp"
#include "spotter/room.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "SPACE scan here   E merge and save   drag / W A S D turn   wheel / Z X zoom   R reset   Q quit";

struct Room
{
    std::vector<spotter::Scan> scans;
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colours;
};

//void AddScan(Room& room, spotter::Scan scan, const spotter::PinholeCamera& camera, std::mt19937& random, const spotter::ScanOptions& options){
//
//@param1 - room: the scans so far; one is added
//@param2 - scan: the new scan
//@param3 - camera: the depth camera
//@param4 - random: the random number generator
//@param5 - options: the scan settings
//
//@brief - the first scan is the origin; every later one is placed among the earlier ones (PlaceScan) and refused
//         if it can't be. A placed scan's points (every 2nd pixel) join the room's cloud. O(PlaceScan).
void AddScan(Room& room, spotter::Scan scan, const spotter::PinholeCamera& camera, std::mt19937& random, const spotter::ScanOptions& options)
{
    if (!room.scans.empty())
    {
        spotter::Placement placed = spotter::PlaceScan(scan, room.scans, camera, random, -1, {}, options);
        std::printf("Scan %zu %s: %s\n", room.scans.size() + 1, placed.placed ? "placed" : "REFUSED", placed.report.c_str());
        if (!placed.placed) return;
        scan.pose = placed.pose;
    }
    for (size_t i = 0; i < scan.cloud.size(); i += 2)
        if (!std::isnan(scan.cloud[i].z)) { room.points.push_back(scan.pose(scan.cloud[i])); room.colours.push_back(scan.picture.at<cv::Vec3b>((int)i / 512, (int)i % 512)); }
    room.scans.push_back(std::move(scan));
}

//void Merge(const Room& room, const spotter::PinholeCamera& camera, std::mt19937& random){
//
//@param1 - room: the placed scans
//@param2 - camera: the depth camera
//@param3 - random: the random number generator
//
//@brief - new: merges every scan into one mesh (FuseScans), builds the clean model (BuildRoom), writes
//         room_scan_<time>.obj and prints the triangle, floor, ceiling, wall and box counts. O(FuseScans).
void Merge(const Room& room, const spotter::PinholeCamera& camera, std::mt19937& random)
{
    std::vector<const spotter::Scan*> all;
    for (const spotter::Scan& s : room.scans) all.push_back(&s);
    spotter::RoomOptions options;
    float voxel;
    spotter::Mesh seen = spotter::FuseScans(all, camera, options, voxel);
    spotter::Room model = spotter::BuildRoom(all, seen, random, options);
    std::string path = cv::format("room_scan_%lld.obj", (long long)std::time(nullptr));
    bool saved = spotter::SaveRoomObj(path, model, {}, "Spotter 13e step 2");
    std::printf("%s %s: %zu triangles merged from %zu scans; floor %.2f m below, ceiling %.2f m above the first camera; %zu walls, %zu boxes\n",
                saved ? "Saved" : "Could NOT save", path.c_str(), seen.groups["seen_room"].size(), all.size(), -model.floorY, model.ceilingY,
                model.walls.size(), model.blocks.size());
}

//int main(){
//
//@brief - the program: 13e · step 2 of 4 · One mesh and a clean model from every scan. Step 1's scans overlap: the
//         same wall seen five times is five slightly different layers of points. Setup, main loop and clean-up are
//         laid out in the file's sections. Returns 0 on a normal exit, 1 if it couldn't start.
int main()
{
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!kinect.IsOpen()) return 1;
    const std::string window = "13e step 2: SPACE scans, E merges and saves";
    cv::namedWindow(window);
    spotter::OrbitView view(spotter::OrbitView::Options(), cv::Point3f(0, 0, 2.5f), 2.5f);
    view.Attach(window);
    spotter::PointRenderer renderer;
    spotter::DepthAverager averager(30);
    spotter::ScanOptions options;
    std::mt19937 random(1);
    Room room;
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        if (averager.Running() && averager.Add(frame.depthMm))
            AddScan(room, spotter::MakeScan(frame.depthCamera.Cloud(averager.Mean()), frame.registered, options), frame.depthCamera, random, options);
        cv::Mat& image = renderer.Begin(view);
        renderer.Draw(room.points, room.colours);
        for (size_t s = 0; s < room.scans.size(); s++) spotter::DrawCamera(image, view, room.scans[s].pose, frame.depthCamera, cv::Scalar(0, 255, 255), 1, std::to_string(s + 1));
        cv::imshow(window, spotter::WithKeys(image, Keys));
        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == ' ' && !averager.Running()) averager.Start();
        if (key == 'e' && !room.scans.empty()) Merge(room, frame.depthCamera, random);
        view.HandleKey(key);
    }
    return 0;
}

// Try: open the .obj in Blender or MeshLab and hide groups: seen_room is the merged scans,
// clean_* and furniture_blocks the tidy (partly guessed) model.
