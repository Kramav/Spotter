// =====================================================================
//  13e · step 3 of 4 · Save the scans as a room folder
//
//  In steps 1 and 2 the scans are gone when the program ends. Now a folder
//  is asked for first, and when the program ends (Q, or the Kinect lost)
//  every scan is written there in one go (spotter::WriteScan): its cloud,
//  colour pictures, features, and last its pose.txt, after everything else
//  is on the disk. A scan cut off half-way has no pose.txt, so 13z skips it.
//  Then points.ply and how to stand the room up straight
//  (spotter::FinishRoomFolder). Nothing is written while you scan.
//
//  New in this step: the room folder.
//  Keys:  SPACE scan here   drag / W A S D turn   wheel / Z X zoom   R reset   Q quit (and save the folder)
//  Next:  13e_s4_Live_Anchors (what the next scan would be placed by, while you aim)
//  Privacy: the folder holds colour pictures of the room. room_scans/ is gitignored: never commit it.
// =====================================================================

#include <cstdio>
#include <opencv2/highgui.hpp>

#include "spotter/depth.hpp"
#include "spotter/room_folder.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "SPACE  scan here   drag / W A S D  turn   wheel / Z X  zoom   R  reset   Q or Esc  quit (and save the folder)";

struct Room
{
    std::filesystem::path folder;
    spotter::RoomInfo info;
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
//         if it can't be. A placed scan's points join the room's cloud. O(PlaceScan).
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

//void SaveFolder(Room& room){
//
//@param1 - room: the placed scans and the folder to write
//
//@brief - new: StartRoomFolder, every scan (WriteScan, pose.txt last), then FinishRoomFolder: points.ply and
//         room.txt with finished 1. Prints whether it saved, the scan count and the point count (13z rebuilds the
//         room from it). O(scans x pixels).
void SaveFolder(Room& room)
{
    room.info.scans = (int)room.scans.size();
    bool ok = spotter::StartRoomFolder(room.folder, room.info);
    for (size_t i = 0; i < room.scans.size() && ok; i++) ok = spotter::WriteScan(room.folder, i, room.scans[i]);
    size_t points = 0;
    ok = ok && spotter::FinishRoomFolder(room.folder, room.info, room.scans, spotter::RoomOptions(), points);
    std::printf("%s %s: %zu scans, points.ply %zu points. 13z rebuilds the room from it.\n", ok ? "Saved" : "COULD NOT SAVE",
                room.folder.string().c_str(), room.scans.size(), points);
}

//int main(){
//
//@brief - the program: 13e · step 3 of 4 · Save the scans as a room folder. In steps 1 and 2 the scans are gone
//         when the program ends. Setup, main loop and clean-up are laid out in the file's sections. Returns 0 on a
//         normal exit, 1 if it couldn't start.
int main()
{
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    Room room;
    room.folder = spotter::AskRoomFolder("room_scans");   // new: where, before anything is scanned
    if (!kinect.IsOpen() || room.folder.empty()) return 1;
    room.info.program = "13e_s3_Save_The_Folder";
    room.info.calibration = kinect.CalibrationText();

    const std::string window = "13e step 3: SPACE scans, Q saves the folder";
    cv::namedWindow(window);
    spotter::OrbitView view(spotter::OrbitView::Options(), cv::Point3f(0, 0, 2.5f), 2.5f);
    view.Attach(window);
    spotter::PointRenderer renderer;
    spotter::DepthAverager averager(30);
    spotter::ScanOptions options;
    std::mt19937 random(1);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))   // ends on Q, or after 10 s without a frame (the Kinect lost)
    {
        room.info.depthCamera = frame.depthCamera;
        if (averager.Running() && averager.Add(frame.depthMm))
        {
            spotter::Scan scan = spotter::MakeScan(frame.depthCamera.Cloud(averager.Mean()), frame.registered, options);
            scan.fullPicture = frame.colour.clone();
            scan.colourIndex = frame.colourIndex;
            AddScan(room, std::move(scan), frame.depthCamera, random, options);
        }
        renderer.Begin(view);
        renderer.Draw(room.points, room.colours);
        cv::imshow(window, spotter::WithKeys(renderer.Image(), Keys));
        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == ' ' && !averager.Running()) averager.Start();
        view.HandleKey(key);
    }
    if (!room.scans.empty()) SaveFolder(room);   // new: once, at the end
    return 0;
}

// Try: scan three times, then pull the Kinect's cable. After 10 seconds the loop ends and the three
// scans are saved anyway. Open the folder with 13z.
