// =====================================================================
//  13e · step 1 of 4 · Place each scan among the earlier ones
//
//  SPACE: hold still while 30 frames are averaged (07), then the averaged
//  cloud and its colour become a Scan (spotter::MakeScan: the points, the
//  ICP points with their facing, the colour features). The first scan is
//  the room's origin. Each later one is placed by spotter::PlaceScan:
//  features against the earlier scans give a rough pose (11b), ICP against
//  everything those scans saw polishes it (11), all in one call. 11b lined
//  up two views; this lines up as many as you like.
//
//  The window freezes for a few seconds while a scan is placed (13e does it
//  in the background).
//
//  New in this step: many scans, each placed against all the earlier ones.
//  Keys:  SPACE scan here   mouse drag / W A S D turn   wheel / Z X zoom   R reset   Q quit
//  Next:  13e_s2_Merge (one mesh and a clean model from all of them)
// =====================================================================

#include <cstdio>
#include <opencv2/highgui.hpp>

#include "spotter/depth.hpp"
#include "spotter/room_scan.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "SPACE scan here   mouse drag / W A S D turn   wheel / Z X zoom   R reset   Q quit";

struct Room
{
    std::vector<spotter::Scan> scans;
    std::vector<cv::Point3f> points;   // every 2nd pixel of every placed scan, in the room, for the viewer
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
//@brief - new: the first scan is the origin; every later one is placed among the earlier ones (PlaceScan:
//         features, then ICP) and refused if it can't be. A placed scan's points (every 2nd pixel) join the room's
//         cloud with their colours. O(PlaceScan).
void AddScan(Room& room, spotter::Scan scan, const spotter::PinholeCamera& camera, std::mt19937& random, const spotter::ScanOptions& options)
{
    if (!room.scans.empty())
    {
        spotter::Placement placed = spotter::PlaceScan(scan, room.scans, camera, random, -1, {}, options);
        std::printf("Scan %zu %s: %s\n", room.scans.size() + 1, placed.placed ? "placed" : "REFUSED", placed.report.c_str());
        if (!placed.placed) return;
        scan.pose = placed.pose;
    }
    else std::printf("Scan 1: the room's origin (%zu features). Move a little, keep part of it in view, SPACE again.\n", scan.near.keypoints.size());
    for (size_t i = 0; i < scan.cloud.size(); i += 2)
        if (!std::isnan(scan.cloud[i].z)) { room.points.push_back(scan.pose(scan.cloud[i])); room.colours.push_back(scan.picture.at<cv::Vec3b>((int)i / 512, (int)i % 512)); }
    room.scans.push_back(std::move(scan));
}

//int main(){
//
//@brief - the program: 13e · step 1 of 4 · Place each scan among the earlier ones. SPACE: hold still while 30
//         frames are averaged (07), then the averaged cloud and its colour become a Scan (spotter::MakeScan: the
//         points, the ICP points with their facing, the colour features). The first scan is the room's origin.
//         Setup, main loop and clean-up are laid out in the file's sections. Returns 0 on a normal exit, 1 if it
//         couldn't start.
int main()
{
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!kinect.IsOpen()) return 1;
    const std::string window = "13e step 1: every scan, placed (SPACE scans)";
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
        if (key == ' ' && !averager.Running()) { averager.Start(); std::printf("Hold still...\n"); }
        view.HandleKey(key);
    }
    return 0;
}

// Try: scan, slide the Kinect 30 cm, scan, then turn it 30 degrees, scan. Turn the view side on:
// the three scans' walls should lie on top of each other. Then aim at a bare wall: refused.
