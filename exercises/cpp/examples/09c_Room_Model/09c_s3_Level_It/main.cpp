// =====================================================================
//  09c · step 3 of 3 · Stand the room up straight
//
//  The floor plane's normal is the room's UP, whatever the camera's tilt.
//  Build new axes from it:
//    up       the floor's normal
//    towards  the camera's -Z (straight back), with its up/down part dropped
//    right    up x towards
//  and turn every point into them: (right . p, up . p, towards . p). The
//  raw mesh (09b, spotter::MeshFromGrid) is saved turned, so it stands
//  straight in Blender even if the camera was tilted. Tilt and roll printed.
//
//  New in this step: new axes from a plane; turning points into them.
//  Keys:  S  scan and save     Q or Esc  quit
//  Next:  09c_Room_Model (the full example adds clean walls, floor, ceiling and furniture boxes)
//  Privacy: an .obj is a 3D capture of the room: never commit one.
// =====================================================================

#include <cmath>
#include <cstdio>
#include <ctime>
#include <random>
#include <opencv2/highgui.hpp>

#include "spotter/depth.hpp"
#include "spotter/geometry.hpp"
#include "spotter/mesh.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "S  scan and save     Q or Esc  quit";

//bool FindFloor(const std::vector<cv::Point3f>& cloud, std::mt19937& random, spotter::Plane& floor){
//
//@param1 - cloud: a depth cloud, 512 wide
//@param2 - random: the random number generator
//@param3 - floor: set to the floor plane
//
//@brief - the floor: the level plane with the most points (FindPlanes, one level plane), or nothing; its normal
//         must point up (-Y), not down, which would be a ceiling. O(FindPlanes).
bool FindFloor(const std::vector<cv::Point3f>& cloud, std::mt19937& random, spotter::Plane& floor)
{
    spotter::PlaneSearch level;
    level.facing = spotter::PlaneSearch::Facing::Level;
    level.maxPlanes = 1;
    std::vector<spotter::PlanePiece> pieces = spotter::FindPlanes(spotter::ThinCloud(cloud, 512, 4), level, random);
    if (pieces.empty() || pieces[0].plane.normal.y > 0) return false;   // its normal must point up (-Y), not down (a ceiling)
    floor = pieces[0].plane;
    return true;
}

//void SaveLevelled(const std::vector<cv::Point3f>& cloud, const cv::Mat& registered, const spotter::Plane& floor){
//
//@param1 - cloud: a depth cloud, 512 wide
//@param2 - registered: colour per depth pixel
//@param3 - floor: the floor plane, which says which way is up
//
//@brief - new: the room axes from the floor (up = its normal, "towards" = straight back with the up part dropped,
//         right = up x towards); the mesh (MeshFromGrid) is expressed in them, so it stands straight with Y up.
//         Prints the camera's height, tilt and roll and writes room_level_<time>.obj. O(pixels).
void SaveLevelled(const std::vector<cv::Point3f>& cloud, const cv::Mat& registered, const spotter::Plane& floor)
{
    cv::Point3f up = floor.normal;
    cv::Point3f towards = cv::Point3f(0, 0, -1) - up * up.dot(cv::Point3f(0, 0, -1));   // straight back, up/down part dropped
    towards /= (float)cv::norm(towards);
    cv::Point3f right = up.cross(towards);
    spotter::Mesh mesh;
    spotter::MeshFromGrid(cloud, registered.reshape(0, 1), 512, 2, 0.03f, mesh, "seen_room");
    for (cv::Point3f& p : mesh.points) p = cv::Point3f(right.dot(p), up.dot(p), towards.dot(p));   // new: room axes (Y up)
    std::printf("camera %.2f m above the floor, tilted %+.1f deg (+ = looking up), rolled %+.1f deg\n",
                floor.distance, std::asin(up.z) * 180 / CV_PI, std::asin(up.x) * 180 / CV_PI);
    std::string path = cv::format("room_level_%lld.obj", (long long)std::time(nullptr));
    if (spotter::SaveObj(path, mesh, "Spotter 09c step 3: levelled by the floor")) std::printf("Saved %s (never commit it)\n", path.c_str());
}

//int main(){
//
//@brief - the program: 09c · step 3 of 3 · Stand the room up straight. The floor plane's normal is the room's UP,
//         whatever the camera's tilt. Build new axes from it: up the floor's normal towards the camera's -Z
//         (straight back), with its up/down part dropped right up x towards and turn every point into them. Setup,
//         main loop and clean-up are laid out in the file's sections. Returns 0 on a normal exit, 1 if it couldn't
//         start.
int main()
{
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!kinect.IsOpen()) return 1;

    spotter::DepthAverager averager(30);
    std::mt19937 random(1);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        if (averager.Running() && averager.Add(frame.depthMm))
        {
            std::vector<cv::Point3f> cloud = frame.depthCamera.Cloud(averager.Mean());
            spotter::Plane floor;
            if (FindFloor(cloud, random, floor)) SaveLevelled(cloud, frame.registered, floor);
            else std::printf("No floor in view: tilt the Kinect down a little.\n");
        }
        cv::imshow("09c step 3: S scans, levels and saves", spotter::WithKeys(frame.colour, Keys));
        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 's' && !averager.Running()) { averager.Start(); std::printf("Scanning: hold still...\n"); }
    }
    return 0;
}

// Try: tilt the Kinect 15 degrees down, scan, and open the .obj: the floor is level anyway. Compare
// with 09b's file from the same spot, which is tilted.
