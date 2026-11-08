// =====================================================================
//  00b · Tour: PinholeCamera and Unmirror           (no Kinect needed)
//
//  What it's for: turning a depth pixel into a 3D point (Unproject), a 3D
//  point into a pixel (Project), and a whole depth picture into a cloud
//  (Cloud). Every 3D example uses these three.
//
//  How it works: a pinhole camera has four numbers, in pixels: the focal
//  lengths fx, fy and the principal point cx, cy (where the lens's axis
//  hits the picture). A pixel's centre sits at +0.5.
//     X = (column + 0.5 - cx) * Z / fx      Y = (row + 0.5 - cy) * Z / fy
//  and back:  column = fx * X / Z + cx - 0.5.
//  Coordinates: metres from the camera, X right, Y down, Z straight out.
//
//  The Kinect has TWO cameras with different numbers, a few cm apart. The
//  same spot lands on different pixels in each; libfreenect2's registration
//  (KinectSource) does that mapping with its factory model. And it sends
//  mirror images: Unmirror flips the pictures and moves cx to width - cx.
//
//  This program prints worked examples on made-up numbers shaped like a Kinect's.
// =====================================================================

#include <cmath>
#include <cstdio>

#include "spotter/frame.hpp"

const spotter::PinholeCamera DepthCamera{ 365.5f, 365.5f, 255.3f, 205.7f, 512, 424 };
const spotter::PinholeCamera ColourCamera{ 1081.4f, 1081.4f, 959.5f, 539.5f, 1920, 1080 };

void UnprojectAndBack()
{
    std::printf("1. Depth pixel -> 3D point -> pixel, at 2 m:\n");
    for (cv::Point2f pixel : { cv::Point2f(255, 205), cv::Point2f(0, 0), cv::Point2f(511, 212) })
    {
        cv::Point3f p = DepthCamera.Unproject(pixel, 2.0f);
        cv::Point2f back = *DepthCamera.Project(p);
        std::printf("   pixel (%3.0f, %3.0f) -> (%+.3f, %+.3f, %.3f) m -> pixel (%.2f, %.2f)\n",
                    pixel.x, pixel.y, p.x, p.y, p.z, back.x, back.y);
    }
    std::printf("   The corner pixel at 2 m is %.2f m left of centre: the depth camera sees about %.0f deg across.\n",
                -DepthCamera.Unproject({ 0, 205 }, 2).x, 2 * std::atan(256 / DepthCamera.fx) * 180 / CV_PI);
}

void TwoCameras()
{
    std::printf("2. One spot, two cameras (here pretending they sit at the same place):\n");
    cv::Point3f spot(0.30f, -0.20f, 2.0f);
    cv::Point2f inDepth = *DepthCamera.Project(spot), inColour = *ColourCamera.Project(spot);
    std::printf("   (%.2f, %.2f, %.2f) m lands on depth pixel (%.0f, %.0f) and colour pixel (%.0f, %.0f):\n"
                "   about %.1f colour pixels per depth pixel. For real the cameras are ~5 cm apart, so the\n"
                "   shift also depends on distance; that's why KinectSource gives colourIndex per frame.\n",
                spot.x, spot.y, spot.z, inDepth.x, inDepth.y, inColour.x, inColour.y, ColourCamera.fx / DepthCamera.fx);
}

void WholeCloud()
{
    std::printf("3. Cloud(): a flat wall 3 m ahead, every pixel at once:\n");
    cv::Mat depthMm(424, 512, CV_32F, cv::Scalar(3000));
    depthMm(cv::Rect(0, 0, 50, 50)).setTo(0);   // a patch with no reading
    std::vector<cv::Point3f> cloud = DepthCamera.Cloud(depthMm);
    const cv::Point3f& corner = cloud.back();
    std::printf("   %zu points; the top-left patch is NaN (%s); the bottom-right corner is (%+.2f, %+.2f, %.2f) m\n",
                cloud.size(), std::isnan(cloud[0].z) ? "yes" : "no", corner.x, corner.y, corner.z);
}

void Mirror()
{
    std::printf("4. Unmirror(): the Kinect's pictures are mirrored, like a selfie camera.\n");
    spotter::RgbdFrame frame;
    frame.depthMm = cv::Mat(424, 512, CV_32F, cv::Scalar(0));
    frame.depthMm.at<float>(212, 400) = 2000;   // something on the picture's RIGHT, as the Kinect sends it
    frame.depthCamera = DepthCamera;
    frame.colourCamera = ColourCamera;
    cv::Point3f before = frame.depthCamera.Unproject({ 400, 212 }, 2);
    spotter::Unmirror(frame);
    cv::Point onlyReading;
    cv::minMaxLoc(frame.depthMm, nullptr, nullptr, nullptr, &onlyReading);
    cv::Point3f after = frame.depthCamera.Unproject(cv::Point2f(onlyReading), 2);
    std::printf("   as sent: column 400, X %+.2f m. After Unmirror: column %d, cx %.1f -> %.1f, X %+.2f m.\n"
                "   The thing really was on the camera's left; now the picture and the 3D point agree.\n",
                before.x, onlyReading.x, DepthCamera.cx, frame.depthCamera.cx, after.x);
}

int main()
{
    UnprojectAndBack();
    TwoCameras();
    WholeCloud();
    Mirror();
    // A console program with no window: started by double-clicking, its console would close
    // before you could read it.
    std::printf("\nPress Enter to close.\n");
    std::getchar();
    return 0;
}
