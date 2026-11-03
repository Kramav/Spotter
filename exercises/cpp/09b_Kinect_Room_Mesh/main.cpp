// =====================================================================
//  09b · Kinect Room Mesh
//
//  Example 08b turned ONE object into a 3D model. This one does the
//  same for EVERYTHING the Kinect sees: the first recreation of the room,
//  from a single view.
//
//    1. Press S. Keep the camera still.
//    2. Average 30 depth frames (07): less noise, flatter walls.
//    3. Every depth pixel becomes a 3D point (08), coloured from the
//       colour camera.
//    4. Mesh: the depth image is already a grid. Each square of points
//       becomes two triangles (08b), unless its corners jump in depth:
//       that's an edge (a chair in front of a wall), not a surface.
//    5. Save room_<time>.obj in the folder you ran from. Open it in
//       Blender or MeshLab and walk around it.
//
//  You'll see the room from where the camera stood, and HOLES behind
//  everything: the camera can't see behind a chair. Those holes are why
//  the next steps (11-13) take several views and merge them.
//
//  Window: colour (half size). After a scan, the pixels that went into
//          the mesh are tinted green: untinted = no depth there.
//  Keys:   S  scan     C  clear the tint     Q or Esc  quit
//
//  Privacy: an .obj is a 3D capture of the room, in colour. *.obj is
//  gitignored: never commit one. Scan with nobody in view.
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/registration.h>
#include <opencv2/opencv.hpp>

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// How many depth frames to average (07). 30 = one second.
const int AverageFrames = 30;

// Depth cut-off (07b): rooms are bigger than libfreenect2's 4.5 m default. Past 6 m almost
// nothing reads.
const float MaxDepthMeters = 6.0f;

// Use every Nth depth pixel in each direction. 1 = all of them (about 400k triangles, a 25 MB
// file); 2 = a quarter of that, which still looks the same from across the room.
const int MeshStep = 2;

// A triangle is skipped when its corners differ in depth by more than this share of their
// distance. A share, not a fixed number of mm as in 08b, because far away neighbouring pixels
// are farther apart: a wall seen at a slant 4 m away steps about 3 cm per pixel, but would still
// be torn by 08b's 20 mm.
const float MaxJumpShare = 0.03f;

// 05c: keeps 30 fps in a dim room, at the cost of a darker picture (and darker mesh colours).
const bool CapColorExposure = false;

const int Width = 512, Height = 424, ColorWidth = 1920, ColorHeight = 1080;

// A triangle mesh: points in the Kinect's coordinates (metres; X right, Y down, Z forward).
struct Mesh
{
    std::vector<cv::Vec3f> points;
    std::vector<cv::Vec3b> colours;   // blue, green, red
    std::vector<cv::Vec3i> faces;     // three point indices each
};

// Step 4: every MeshStep-th point of the grid, joined into triangles where the depth doesn't jump.
// cloud: one point per depth pixel (NaN = no reading); colours: one per depth pixel.
Mesh MeshFromCloud(const std::vector<cv::Vec3f>& cloud, const std::vector<cv::Vec3b>& colours)
{
    Mesh mesh;
    const int columns = (Width + MeshStep - 1) / MeshStep, rows = (Height + MeshStep - 1) / MeshStep;
    std::vector<int> pointOf(columns * rows, -1);   // grid cell -> point number, -1 = no reading
    std::vector<float> depthOf(columns * rows);
    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            int pixel = row * MeshStep * Width + column * MeshStep;
            if (std::isnan(cloud[pixel][2])) continue;
            pointOf[row * columns + column] = (int)mesh.points.size();
            depthOf[row * columns + column] = cloud[pixel][2];
            mesh.points.push_back(cloud[pixel]);
            mesh.colours.push_back(colours[pixel]);
        }
    }
    auto addTriangle = [&](int a, int b, int c)
    {
        if (pointOf[a] < 0 || pointOf[b] < 0 || pointOf[c] < 0) return;
        float nearest = std::min({ depthOf[a], depthOf[b], depthOf[c] });
        float farthest = std::max({ depthOf[a], depthOf[b], depthOf[c] });
        if (farthest - nearest > MaxJumpShare * nearest) return;   // an edge, not a surface
        mesh.faces.emplace_back(pointOf[a], pointOf[b], pointOf[c]);   // this order faces the camera
    };
    for (int row = 0; row + 1 < rows; row++)
    {
        for (int column = 0; column + 1 < columns; column++)
        {
            int topLeft = row * columns + column, topRight = topLeft + 1;
            int bottomLeft = topLeft + columns, bottomRight = bottomLeft + 1;
            addTriangle(topLeft, bottomLeft, topRight);
            addTriangle(topRight, bottomLeft, bottomRight);
        }
    }
    return mesh;
}

// Step 5, as in 08b: "v x y z red green blue" per point (colours 0-1), "f a b c" per triangle
// (counting from 1). Saved Y up, Z towards the viewer, as 08's .ply.
bool SaveObj(const std::string& path, const Mesh& mesh)
{
    std::ofstream file(path);
    file << "# Spotter 09b: the room from one view\n";
    for (size_t i = 0; i < mesh.points.size(); i++)
    {
        const cv::Vec3f& p = mesh.points[i];
        const cv::Vec3b& c = mesh.colours[i];
        file << cv::format("v %.4f %.4f %.4f %.3f %.3f %.3f\n", p[0], -p[1], -p[2], c[2] / 255.0, c[1] / 255.0, c[0] / 255.0);
    }
    file << "g seen_room\n";
    for (const cv::Vec3i& f : mesh.faces) file << cv::format("f %d %d %d\n", f[0] + 1, f[1] + 1, f[2] + 1);
    return (bool)file;
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // -----------------------------------------------------------------
    //  Setup: Kinect and registration (as in 08b)
    // -----------------------------------------------------------------

    libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Warning));
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device* device = freenect2.enumerateDevices() > 0 ? freenect2.openDefaultDevice() : nullptr;
    if (device == nullptr)
    {
        std::printf("No Kinect v2 found, or it is in use. See the main README's troubleshooting table.\n");
        return 1;
    }
    libfreenect2::Freenect2Device::Config config;
    config.MaxDepth = MaxDepthMeters;
    device->setConfiguration(config);
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Depth);
    device->setColorFrameListener(&listener);
    device->setIrAndDepthFrameListener(&listener);
    device->startStreams(true, true);
    if (CapColorExposure) device->setColorSemiAutoExposure(16.0f);

    libfreenect2::Registration registration(device->getIrCameraParams(), device->getColorCameraParams());
    libfreenect2::Frame undistorted(Width, Height, 4), registered(Width, Height, 4), averaged(Width, Height, 4);
    std::vector<int> colorIndexOfDepthPixel(Width * Height);

    int framesCaptured = -1;   // -1 = not scanning
    std::vector<double> sumMm(Width * Height);
    std::vector<int> readings(Width * Height);
    std::vector<cv::Vec3f> cloud(Width * Height);
    std::vector<cv::Vec3b> colours(Width * Height);
    std::vector<cv::Point> tintPixels;   // colour pixels that went into the last mesh
    double fps = 0;
    auto lastFrameTime = std::chrono::steady_clock::now();

    const std::string window = "09b room mesh  (S scan, C clear, Q quit)";
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    std::printf("Press S to scan the room (in the window). Keep the camera still for %d frames.\n", AverageFrames);

    // -----------------------------------------------------------------
    //  Main operation
    // -----------------------------------------------------------------

    libfreenect2::FrameMap frames;
    cv::Mat frame;
    while (true)
    {
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == 'c') tintPixels.clear();

        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }
        libfreenect2::Frame* colorFrame = frames[libfreenect2::Frame::Color];
        cv::cvtColor(cv::Mat(ColorHeight, ColorWidth, CV_8UC4, colorFrame->data), frame, cv::COLOR_BGRA2BGR);
        registration.apply(colorFrame, frames[libfreenect2::Frame::Depth], &undistorted, &registered,
                           false, nullptr, colorIndexOfDepthPixel.data());
        listener.release(frames);

        auto now = std::chrono::steady_clock::now();
        fps = 0.9 * fps + 0.1 / std::max(1e-3, std::chrono::duration<double>(now - lastFrameTime).count());
        lastFrameTime = now;

        // --- Step 1: start a scan ----------------------------------------------------
        if (key == 's' && framesCaptured < 0)
        {
            framesCaptured = 0;
            std::fill(sumMm.begin(), sumMm.end(), 0.0);
            std::fill(readings.begin(), readings.end(), 0);
            std::printf("Scanning: hold still...\n");
        }

        // --- Step 2: average the depth (07) -----------------------------------------
        if (framesCaptured >= 0)
        {
            const float* depthMm = (const float*)undistorted.data;
            for (int i = 0; i < Width * Height; i++)
            {
                if (!(depthMm[i] > 0) || std::isinf(depthMm[i])) continue;   // holes don't count
                sumMm[i] += depthMm[i];
                readings[i]++;
            }
            framesCaptured++;
        }

        // --- Steps 3-5: points, mesh, save -----------------------------------------
        if (framesCaptured == AverageFrames)
        {
            framesCaptured = -1;
            auto start = std::chrono::steady_clock::now();
            float* averageMm = (float*)averaged.data;
            int holes = 0;
            for (int i = 0; i < Width * Height; i++)
            {
                averageMm[i] = readings[i] >= AverageFrames / 2 ? (float)(sumMm[i] / readings[i]) : 0;   // too often a hole: a hole
                if (averageMm[i] == 0) holes++;
            }
            // Step 3: as in 08. The colour comes from the last frame (black where the colour
            // camera doesn't reach: it sees a slightly different part of the room).
            for (int row = 0; row < Height; row++)
            {
                for (int column = 0; column < Width; column++)
                {
                    float x, y, z;
                    registration.getPointXYZ(&averaged, row, column, x, y, z);   // NaN = no reading
                    int i = row * Width + column;
                    cloud[i] = cv::Vec3f(x, y, z);
                    const unsigned char* bgrx = registered.data + 4 * i;
                    colours[i] = cv::Vec3b(bgrx[0], bgrx[1], bgrx[2]);
                }
            }

            Mesh mesh = MeshFromCloud(cloud, colours);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

            tintPixels.clear();
            for (int i = 0; i < Width * Height; i++)
                if (averageMm[i] > 0 && colorIndexOfDepthPixel[i] >= 0)
                    tintPixels.emplace_back(colorIndexOfDepthPixel[i] % ColorWidth, colorIndexOfDepthPixel[i] / ColorWidth);

            std::string path = (std::filesystem::current_path() / cv::format("room_%lld.obj", (long long)std::time(nullptr))).string();
            if (mesh.faces.empty())
                std::printf("Nothing to mesh: no depth readings.\n");
            else if (SaveObj(path, mesh))
                std::printf("Saved %zu points and %zu triangles (%.0f ms), %.1f MB, to %s\n"
                            "  %.0f%% of the view had no depth. (A capture of the room: never commit it.)\n",
                            mesh.points.size(), mesh.faces.size(), ms, std::filesystem::file_size(path) / 1e6, path.c_str(),
                            100.0 * holes / (Width * Height));
            else
                std::printf("Could not write %s\n", path.c_str());
        }

        // -----------------------------------------------------------------
        //  Output: half-size colour, the last scan tinted green
        // -----------------------------------------------------------------

        cv::Mat display;
        cv::resize(frame, display, cv::Size(ColorWidth / 2, ColorHeight / 2));
        for (const cv::Point& p : tintPixels)
        {
            cv::Vec3b& pixel = display.at<cv::Vec3b>(p.y / 2, p.x / 2);
            pixel = cv::Vec3b(pixel[0] / 2, (pixel[1] + 255) / 2, pixel[2] / 2);
        }
        std::string status = framesCaptured >= 0 ? cv::format("HOLD STILL  %d / %d", framesCaptured, AverageFrames)
                                                 : cv::format("%.1f fps   S to scan the room", fps);
        cv::putText(display, status, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        cv::imshow(window, display);
    }

    device->stop();
    device->close();
    cv::destroyAllWindows();
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Open the .obj in Blender (File > Import > Wavefront; set the viewport to show vertex colours)
//     or MeshLab. Turn it round: where are the holes, and what made each one?
//  2. Look at the room from above in Blender. Every hole is a "shadow" pointing away from where
//     the camera stood. Which spot would you scan from next to fill the biggest one?
//  3. MaxJumpShare = 0.005 versus 0.2: a torn room versus webs stretched from every chair to the
//     wall behind it.
//  4. MeshStep = 1 versus 4: file size and look. When does the room stop looking like the room?
//  5. AverageFrames = 1: look at a flat wall edge-on in Blender. How bumpy is one frame?
//  6. A window or a mirror: what does the Kinect make of glass?
//  7. Scan, move the Kinect 1 m, scan again, and import both files. They overlap in the wrong
//     place: each was saved from where its own camera stood. Lining them up is example 11's job.
