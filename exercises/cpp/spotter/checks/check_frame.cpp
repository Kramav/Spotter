// Self-check for spotter::PinholeCamera and Unmirror (frame.hpp), with no Kinect: libfreenect2's
// own Registration runs on made-up factory settings and a made-up frame, and our unmirrored
// frame must give the same 3D points with X negated (the mirror undone), at every pixel.
// Run: build\check_frame.exe. Exit code 0 = passed.

#include "spotter/frame.hpp"

#include <cmath>
#include <cstdio>
#include <random>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/registration.h>
#include <opencv2/imgproc.hpp>

int failures = 0;
void Check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAILED: %s\n", what); failures++; }
}

int main()
{
    // Factory settings shaped like a real Kinect's: no lens distortion, and a colour mapping of
    // about 2.9 colour pixels per depth pixel (libfreenect2's polynomial, linear terms only).
    libfreenect2::Freenect2Device::IrCameraParams ir = { 365.5f, 365.5f, 255.3f, 205.7f, 0, 0, 0, 0, 0 };
    libfreenect2::Freenect2Device::ColorCameraParams colour = {};
    colour.fx = colour.fy = 1081.37f;
    colour.cx = 959.5f;
    colour.cy = 539.5f;
    colour.shift_d = 863;
    colour.shift_m = 52;
    colour.mx_x1y0 = colour.my_x0y1 = 2.9f * 0.2199f;   // libfreenect2 scales by depth_q 0.01 / color_q 0.002199
    libfreenect2::Registration registration(ir, colour);

    // A made-up depth frame (0.5-4.5 m, 5% holes) and a noisy colour picture.
    std::mt19937 random(1);
    std::uniform_real_distribution<float> distance(500, 4500), chance(0, 1);
    libfreenect2::Frame depth(512, 424, 4), colourFrame(1920, 1080, 4), undistorted(512, 424, 4), registered(512, 424, 4);
    float* mm = (float*)depth.data;
    for (int i = 0; i < 512 * 424; i++) mm[i] = chance(random) < 0.05f ? 0 : distance(random);
    cv::Mat colourBgrx(1080, 1920, CV_8UC4, colourFrame.data);
    cv::randu(colourBgrx, 0, 256);
    std::vector<int> colourIndex(512 * 424);
    registration.apply(&colourFrame, &depth, &undistorted, &registered, false, nullptr, colourIndex.data());

    // What KinectSource::Next builds from the same buffers, then the unmirror.
    spotter::RgbdFrame frame;
    frame.depthMm = cv::Mat(424, 512, CV_32F, undistorted.data).clone();
    frame.depthMm.setTo(0, frame.depthMm != frame.depthMm);   // NaN -> 0
    cv::cvtColor(colourBgrx, frame.colour, cv::COLOR_BGRA2BGR);
    cv::cvtColor(cv::Mat(424, 512, CV_8UC4, registered.data), frame.registered, cv::COLOR_BGRA2BGR);
    frame.colourIndex = colourIndex;
    frame.depthCamera = { ir.fx, ir.fy, ir.cx, ir.cy, 512, 424 };
    frame.colourCamera = { colour.fx, colour.fy, colour.cx, colour.cy, 1920, 1080 };
    spotter::Unmirror(frame);

    // 1. Every pixel's 3D point = libfreenect2's getPointXYZ at the mirrored pixel, X negated.
    int compared = 0, mapped = 0;
    double worst = 0;
    std::vector<cv::Point3f> cloud = frame.depthCamera.Cloud(frame.depthMm);
    for (int row = 0; row < 424; row++)
        for (int column = 0; column < 512; column++)
        {
            float x, y, z;
            registration.getPointXYZ(&undistorted, row, 511 - column, x, y, z);
            const cv::Point3f& p = cloud[row * 512 + column];
            if (std::isnan(z)) { Check(std::isnan(p.z), "no reading stays no reading"); continue; }
            worst = std::max(worst, (double)cv::norm(p - cv::Point3f(-x, y, z)));
            compared++;

            // 2. The colour index still points at the colour the registered picture shows.
            int index = frame.colourIndex[row * 512 + column];
            if (index < 0) continue;
            mapped++;
            Check(frame.registered.at<cv::Vec3b>(row, column) == frame.colour.at<cv::Vec3b>(index / 1920, index % 1920),
                  "colourIndex points at the registered colour");
        }
    Check(compared > 100000 && worst < 1e-5, "Unproject = getPointXYZ with X negated");
    Check(mapped > 50000, "most depth pixels land in the colour picture");
    std::printf("%d points compared with getPointXYZ (worst gap %.2g m), %d colour lookups\n", compared, worst, mapped);

    // 3. Project undoes Unproject, and right of centre in the picture is +X (to the right).
    spotter::PinholeCamera camera = frame.depthCamera;
    cv::Point2f pixel(400.25f, 100.5f);
    std::optional<cv::Point2f> back = camera.Project(camera.Unproject(pixel, 2.0f));
    Check(back && cv::norm(*back - pixel) < 1e-3, "Project(Unproject(pixel)) = pixel");
    Check(!camera.Project(cv::Point3f(0, 0, -1)), "a point behind the camera doesn't project");
    Check(camera.Unproject(cv::Point2f(500, 212), 1).x > 0, "the right of the picture is +X");

    std::printf(failures == 0 ? "PASSED\n" : "%d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
