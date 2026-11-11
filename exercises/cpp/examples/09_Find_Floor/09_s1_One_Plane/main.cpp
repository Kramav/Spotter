// =====================================================================
//  09 · step 1 of 3 · Three points make a plane
//
//  Pick one random depth point and two more near it in the picture. They
//  define a plane. Paint green every point within 2 cm of it. A new random
//  pick every frame, so the green jumps around: sometimes a wall, sometimes
//  the floor, sometimes a sliver across a chair.
//
//  New in this step: the plane through three points (spotter::PlaneThrough),
//  a point's distance to a plane.
//  Keys:  Q or Esc  quit
//  Next:  09_s2_Ransac (try many, keep the best)
// =====================================================================

#include <cmath>
#include <random>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/geometry.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "Q or Esc  quit";

const float ThicknessMeters = 0.02f;   // "on the plane" = within this distance
const int NeighbourPixels = 40;        // the other two points come from this close in the picture

//bool PickPlane(const std::vector<cv::Point3f>& cloud, std::mt19937& random, spotter::Plane& plane){
//
//@param1 - cloud: a depth cloud, 512 wide (NaN = no reading)
//@param2 - random: the random number generator
//@param3 - plane: set to the plane through the three points
//
//@brief - three valid points near each other (one random pixel and two within NeighbourPixels of it, kept inside
//         the picture) and the plane through them. False if this try didn't find three readings or they lie in a
//         line. O(1).
bool PickPlane(const std::vector<cv::Point3f>& cloud, std::mt19937& random, spotter::Plane& plane)
{
    std::uniform_int_distribution<int> anyPixel(0, (int)cloud.size() - 1), offset(-NeighbourPixels, NeighbourPixels);
    int a = anyPixel(random), row = a / 512, column = a % 512;
    int b = std::clamp(row + offset(random), 0, 423) * 512 + std::clamp(column + offset(random), 0, 511);
    int c = std::clamp(row + offset(random), 0, 423) * 512 + std::clamp(column + offset(random), 0, 511);
    if (std::isnan(cloud[a].z) || std::isnan(cloud[b].z) || std::isnan(cloud[c].z)) return false;
    return spotter::PlaneThrough(cloud[a], cloud[b], cloud[c], plane);
}

//int main(){
//
//@brief - the program: 09 · step 1 of 3 · Three points make a plane. Pick one random depth point and two more near
//         it in the picture. They define a plane. Paint green every point within 2 cm of it. Setup, main loop and
//         clean-up are laid out in the file's sections. Returns 0 on a normal exit, 1 if it couldn't start.
int main()
{
    spotter::KinectSource::Options options;
    options.colour = false;
    spotter::KinectSource kinect(options);
    if (!kinect.IsOpen()) return 1;

    std::mt19937 random(1);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        std::vector<cv::Point3f> cloud = frame.depthCamera.Cloud(frame.depthMm);
        cv::Mat picture = spotter::DepthPicture(frame.depthMm);
        spotter::Plane plane;
        int on = 0;
        if (PickPlane(cloud, random, plane))
            for (int i = 0; i < (int)cloud.size(); i++)
                if (!std::isnan(cloud[i].z) && std::abs(plane.DistanceTo(cloud[i])) < ThicknessMeters) { picture.at<cv::Vec3b>(i) = cv::Vec3b(0, 255, 0); on++; }
        cv::putText(picture, cv::format("%d points on this random plane", on), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        cv::imshow("09 step 1: one random plane per frame", spotter::WithKeys(picture, Keys));
        if (spotter::IsQuitKey(cv::waitKey(300))) break;   // slowed down so you can see each try
    }
    return 0;
}

// Try: count how many tries land on the floor. A plane runs on for ever, so even a good one also
// paints thin strips where it slices through other things.
