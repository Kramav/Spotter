// =====================================================================
//  08b · step 3 of 3 · Join the pixels into triangles and save them
//
//  Step 2's cut-out, then:
//    - keep its biggest connected blob (stray bits of other things go);
//    - the depth picture is already a grid: each 2x2 square of kept pixels
//      becomes two triangles, unless the corners jump more than 20 mm in
//      depth (that's an edge, not a surface);
//    - save as an .obj (spotter::SaveObj), turned Y up for 3D programs.
//  Only the FRONT: the camera never saw the back.
//
//  New in this step: a triangle mesh from a mask; an OBJ file.
//  Keys:  click a box  capture and save     Q or Esc  quit
//  Next:  08b_Object_Mesh (the full example drops the table, guesses the back,
//         and lets you click objects YOLO has no box for)
//  Privacy: an .obj is a capture of part of the room: never commit one. Not of people.
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/mesh.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/stats.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "click a box  capture and save     Q or Esc  quit";

const float DepthBandMeters = 0.3f, MaxJumpMm = 20;

void OnMouse(int event, int x, int y, int, void* click)
{
    if (event == cv::EVENT_LBUTTONDOWN) *(cv::Point2d*)click = cv::Point2d(x * 2, y * 2);
}

spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = 0.4f;
    return options;
}

// Step 2's cut-out as a mask over the depth picture, then its biggest blob.
cv::Mat CutOut(const cv::Rect2d& box, const cv::Mat& depthMm, const spotter::RgbdFrame& frame)
{
    cv::Rect2d centre(box.x + box.width / 4, box.y + box.height / 4, box.width / 2, box.height / 2);
    cv::Mat mask = cv::Mat::zeros(depthMm.size(), CV_8U);
    std::vector<float> centreMm;
    for (int i = 0; i < (int)frame.colourIndex.size(); i++)
    {
        int c = frame.colourIndex[i];
        cv::Point2d pixel(c % frame.colour.cols, c / frame.colour.cols);
        if (c < 0 || depthMm.at<float>(i) <= 0 || !box.contains(pixel)) continue;
        mask.data[i] = 255;
        if (centre.contains(pixel)) centreMm.push_back(depthMm.at<float>(i));
    }
    float middle = spotter::Median(centreMm);
    for (int i = 0; i < (int)mask.total(); i++) if (mask.data[i] && std::abs(depthMm.at<float>(i) - middle) > DepthBandMeters * 1000) mask.data[i] = 0;
    cv::Mat labels, stats, centroids;
    int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    int biggest = 0;
    for (int label = 1; label < count; label++)
        if (biggest == 0 || stats.at<int>(label, cv::CC_STAT_AREA) > stats.at<int>(biggest, cv::CC_STAT_AREA)) biggest = label;
    return biggest > 0 ? cv::Mat(labels == biggest) : mask;
}

// New: two triangles per 2x2 square of kept pixels, unless the corners jump in depth.
spotter::Mesh MeshFromMask(const cv::Mat& mask, const std::vector<cv::Point3f>& cloud, const cv::Mat& registered)
{
    spotter::Mesh mesh;
    std::vector<int> pointOf(mask.total(), -1);
    for (int i = 0; i < (int)mask.total(); i++)
        if (mask.data[i]) { pointOf[i] = (int)mesh.points.size(); mesh.points.push_back(cloud[i]); mesh.colours.push_back(registered.at<cv::Vec3b>(i)); }
    std::vector<cv::Vec3i>& faces = mesh.groups["seen_front"];
    auto Triangle = [&](int a, int b, int c)
    {
        if (pointOf[a] < 0 || pointOf[b] < 0 || pointOf[c] < 0) return;
        float nearest = std::min({ cloud[a].z, cloud[b].z, cloud[c].z }), farthest = std::max({ cloud[a].z, cloud[b].z, cloud[c].z });
        if ((farthest - nearest) * 1000 <= MaxJumpMm) faces.emplace_back(pointOf[a], pointOf[b], pointOf[c]);
    };
    for (int row = 0; row + 1 < mask.rows; row++)
        for (int column = 0; column + 1 < mask.cols; column++)
        {
            int topLeft = row * mask.cols + column;
            Triangle(topLeft, topLeft + mask.cols, topLeft + 1);
            Triangle(topLeft + 1, topLeft + mask.cols, topLeft + mask.cols + 1);
        }
    return mesh;
}

void Save(spotter::Mesh& mesh)
{
    std::string path = cv::format("object_%lld.obj", (long long)std::time(nullptr));
    spotter::TurnToYUp(mesh);
    if (spotter::SaveObj(path, mesh, "Spotter 08b step 3: the front only")) std::printf("Saved %zu triangles to %s (never commit it)\n", mesh.groups["seen_front"].size(), path.c_str());
}

int main()
{
    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;

    const char* window = "08b step 3: click a box to save its front as .obj";
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    cv::Point2d click(-1, -1);
    cv::setMouseCallback(window, OnMouse, &click);
    spotter::DepthAverager averager(30);
    cv::Rect2d chosen;
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        if (averager.Running() && averager.Add(frame.depthMm))
        {
            cv::Mat averageMm = averager.Mean();
            spotter::Mesh mesh = MeshFromMask(CutOut(chosen, averageMm, frame), frame.depthCamera.Cloud(averageMm), frame.registered);
            Save(mesh);
        }
        else if (!averager.Running())
            for (const spotter::Detection& d : yolo.Detect(frame.colour))
                if (d.box.contains(click) && d.classId != yolo.Classes().Find("person")) { chosen = d.box; averager.Start(); }
        click = cv::Point2d(-1, -1);

        cv::Mat display;
        cv::resize(frame.colour, display, frame.colour.size() / 2);
        if (averager.Running()) cv::putText(display, cv::format("HOLD STILL %d / 30", averager.Added()), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        cv::imshow(window, spotter::WithKeys(display, Keys));
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: open the .obj in MeshLab or Blender and turn it round: a thin shell, hollow behind.
// MaxJumpMm = 5: torn; 100: webs stretched to whatever was behind.
