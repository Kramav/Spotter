// =====================================================================
//  07d · step 3 of 3 · Paint the person out
//
//  Steps 1 and 2 together. A foreground blob (something in front of the
//  room) that lies mostly inside person boxes is a person: replace its
//  pixels with the colour memory. Whole blobs, not boxes, so the wall
//  inside the box stays live.
//
//  New in this step: blobs + person boxes -> who to remove.
//  Keys:  SPACE  forget and start over     Q or Esc  quit
//  Next:  07d_Remove_Person (the full example keeps carried objects, hides the edges,
//         and runs YOLO on its own thread)
// =====================================================================

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "SPACE  forget and start over     Q or Esc  quit";

const double PersonShare = 0.3;   // a blob is a person if this share of it is in person boxes
const int MinBlobPixels = 500;
const float BlendRate = 0.1f;

//spotter::YoloDetector::Options YoloSettings(){
//
//@brief - the detector's options: yolo26n.onnx and its names file from the repo, minimum confidence 0.4. O(1).
spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = 0.4f;
    return options;
}

cv::Mat InsideBoxes(const std::vector<spotter::Detection>& detections, int classId, const spotter::RgbdFrame& frame)   // step 1
{
    cv::Mat mask = cv::Mat::zeros(frame.depthMm.size(), CV_8U);
    for (size_t i = 0; i < frame.colourIndex.size(); i++)
    {
        int c = frame.colourIndex[i];
        if (c < 0) continue;
        for (const spotter::Detection& d : detections)
            if (d.classId == classId && d.box.contains(cv::Point2d(c % frame.colour.cols, c / frame.colour.cols))) mask.data[i] = 1;
    }
    return mask;
}

void Learn(cv::Mat& memory, const cv::Mat& registered, const cv::Mat& foreground, const cv::Mat& inPerson)   // step 2
{
    for (int i = 0; i < (int)registered.total(); i++)
    {
        cv::Vec3b live = registered.at<cv::Vec3b>(i);
        if (foreground.data[i] || inPerson.data[i] || live == cv::Vec3b(0, 0, 0)) continue;
        cv::Vec3f& remembered = memory.at<cv::Vec3f>(i);
        if (remembered == cv::Vec3f(0, 0, 0)) remembered = live;
        else remembered += BlendRate * (cv::Vec3f(live) - remembered);
    }
}

//cv::Mat PersonBlobs(const cv::Mat& foreground, const cv::Mat& inPerson){
//
//@param1 - foreground: the foreground mask
//@param2 - inPerson: 1 where a pixel is inside a person box
//
//@brief - new: labels the foreground blobs; a blob of at least MinBlobPixels with PersonShare of it inside person
//         boxes is a person and goes in the returned mask (255). O(pixels).
cv::Mat PersonBlobs(const cv::Mat& foreground, const cv::Mat& inPerson)
{
    cv::Mat labels, stats, centroids;
    int count = cv::connectedComponentsWithStats(foreground, labels, stats, centroids);
    std::vector<int> inBox(count, 0);
    for (int i = 0; i < (int)foreground.total(); i++) if (labels.at<int>(i) > 0 && inPerson.data[i]) inBox[labels.at<int>(i)]++;
    cv::Mat remove = cv::Mat::zeros(foreground.size(), CV_8U);
    for (int i = 0; i < (int)foreground.total(); i++)
    {
        int label = labels.at<int>(i), area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (label > 0 && area >= MinBlobPixels && inBox[label] >= PersonShare * area) remove.data[i] = 255;
    }
    return remove;
}

//int main(){
//
//@brief - the program: 07d · step 3 of 3 · Paint the person out. Steps 1 and 2 together. A foreground blob
//         (something in front of the room) that lies mostly inside person boxes is a person: replace its pixels
//         with the colour memory. Whole blobs, not boxes, so the wall inside the box stays live. Setup, main loop
//         and clean-up are laid out in the file's sections. Returns 0 on a normal exit, 1 if it couldn't start.
int main()
{
    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;
    const int person = yolo.Classes().Find("person");

    spotter::AdaptiveBackground background(spotter::AdaptiveBackground::Options{});
    cv::Mat memory = cv::Mat::zeros(424, 512, CV_32FC3);
    const char* window = "07d step 3: people painted out";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, 1024, 848);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        cv::Mat inPerson = InsideBoxes(yolo.Detect(frame.colour), person, frame);
        cv::Mat foreground = background.Update(frame.depthMm, inPerson);   // a person standing still is never absorbed
        Learn(memory, frame.registered, foreground, inPerson);
        cv::Mat remembered, picture = frame.registered.clone();
        memory.convertTo(remembered, CV_8UC3);
        remembered.copyTo(picture, PersonBlobs(foreground, inPerson));      // paint them out
        cv::imshow(window, spotter::WithKeys(picture, Keys));

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == ' ') { background.Forget(); memory.setTo(0); }
    }
    return 0;
}

// Try: look at your outline: a thin halo of you remains (colour and depth edges don't line up
// exactly). Pick up a cup: it disappears with you. The full example fixes both.
