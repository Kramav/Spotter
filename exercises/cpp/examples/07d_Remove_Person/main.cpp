// =====================================================================
//  07d · Remove Person
//
//  07c finds everything in front of the remembered room. This example
//  decides which of it is a PERSON, and paints them out with what the
//  room looked like behind them. Anything they carry stays visible.
//
//    1. Depth background (07c, spotter::AdaptiveBackground) -> foreground.
//    2. YOLO on its own thread (05c, spotter::AsyncDetector) -> person boxes,
//       object boxes.
//    3. A blob that lies mostly inside person boxes is a person.
//       Whole blobs, not boxes: the boxes lag behind (YOLO runs ~6x a
//       second), but the blob is fresh every frame.
//    4. Paint the person's pixels with the remembered COLOUR of the room,
//       EXCEPT inside another object's box: a carried cup stays.
//    5. The colour memory only learns where there's no foreground and no
//       person, so it keeps what was behind them.
//
//  Never-seen pixels (behind someone since the start) show magenta.
//  The picture is the colour camera's view mapped onto the depth pixels
//  (512 x 424, as in 08), so colour and depth line up pixel for pixel.
//
//  Keys:   V  explain view (red = removed, yellow = kept object, green = other foreground)
//          SPACE  forget both memories and start over      Q or Esc  quit
// =====================================================================

#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/detector_yolo_opencvdnn.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

const float PersonConfidence = 0.40f;   // a "person" box must be at least this sure
const double PersonShare = 0.3;         // a blob is a person if this share of it lies in person boxes
const float ObjectConfidence = 0.25f;   // any other class at least this sure is kept visible
const double ObjectBoxGrow = 0.2;       // grow object boxes by this share: YOLO's boxes lag a moving hand
const int GrowPixels = 4;               // widen the removed area: colour and depth edges don't line up exactly
const float ColourBlendRate = 0.1f;     // per frame, how fast the colour memory follows the room
const int MinBlobPixels = 500;          // as in 07c
const std::string Window = "07d remove person  (V explain view, SPACE forget, Q quit)";
const cv::Vec3b Magenta(255, 0, 255);

spotter::YoloDetector::Options YoloSettings()
{
    spotter::YoloDetector::Options options;
    options.modelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";
    options.namesPath = SPOTTER_REPO_ROOT "/yolo26n.names";
    options.minConfidence = std::min(PersonConfidence, ObjectConfidence);
    return options;
}

spotter::KinectSource::Options KinectSettings()
{
    spotter::KinectSource::Options options;
    options.capExposure = true;   // a steady 30 fps in dim light (05c)
    return options;
}

// ---------------------------------------------------------------------
//  The new idea: which foreground is a person, and what was behind them
// ---------------------------------------------------------------------

// Step 2: for every depth pixel, does it land inside a person box, or another object's box?
struct BoxMasks
{
    cv::Mat inPerson, inObject;   // CV_8U, the depth picture's size, 1 = inside
    int people = 0, objects = 0;
};

BoxMasks MaskBoxes(const std::vector<spotter::Detection>& detections, const spotter::RgbdFrame& frame, int personClass)
{
    std::vector<cv::Rect2d> personBoxes, objectBoxes;
    for (const spotter::Detection& d : detections)
    {
        if (d.classId == personClass && d.confidence >= PersonConfidence) personBoxes.push_back(d.box);
        if (d.classId != personClass && d.confidence >= ObjectConfidence)
            objectBoxes.push_back(cv::Rect2d(d.box.x - d.box.width * ObjectBoxGrow / 2, d.box.y - d.box.height * ObjectBoxGrow / 2,
                                             d.box.width * (1 + ObjectBoxGrow), d.box.height * (1 + ObjectBoxGrow)));
    }
    BoxMasks masks{ cv::Mat::zeros(frame.depthMm.size(), CV_8U), cv::Mat::zeros(frame.depthMm.size(), CV_8U),
                    (int)personBoxes.size(), (int)objectBoxes.size() };
    for (size_t i = 0; i < frame.colourIndex.size(); i++)
    {
        int c = frame.colourIndex[i];
        if (c < 0) continue;   // the colour camera doesn't see this depth pixel
        cv::Point2d spot(c % frame.colour.cols, c / frame.colour.cols);
        for (const cv::Rect2d& box : personBoxes) if (box.contains(spot)) masks.inPerson.data[i] = 1;
        for (const cv::Rect2d& box : objectBoxes) if (box.contains(spot)) masks.inObject.data[i] = 1;
    }
    return masks;
}

// Steps 3-4: a big blob with at least PersonShare of it inside person boxes is a person. Widen
// it by GrowPixels (the edge halo), then give back whatever lies in an object's box (the cup).
cv::Mat WhatToRemove(const cv::Mat& foreground, const BoxMasks& masks)
{
    cv::Mat labels, stats, centroids;
    int count = cv::connectedComponentsWithStats(foreground, labels, stats, centroids);
    const int* label = labels.ptr<int>();
    std::vector<int> personPixels(count, 0);
    for (size_t i = 0; i < foreground.total(); i++) if (label[i] > 0 && masks.inPerson.data[i]) personPixels[label[i]]++;
    cv::Mat remove = cv::Mat::zeros(foreground.size(), CV_8U);
    for (size_t i = 0; i < foreground.total(); i++)
    {
        int area = stats.at<int>(label[i], cv::CC_STAT_AREA);
        if (label[i] > 0 && area >= MinBlobPixels && personPixels[label[i]] >= PersonShare * area) remove.data[i] = 255;
    }
    cv::dilate(remove, remove, cv::Mat(), cv::Point(-1, -1), GrowPixels);
    remove.setTo(0, masks.inObject);   // the cup stays
    return remove;
}

// Step 5: the room's colour per depth pixel, learnt only where nothing and nobody is in the way.
struct ColourMemory
{
    cv::Mat colour;   // CV_32FC3
    cv::Mat seen;     // CV_8U, 1 = learnt at least once

    void Learn(const spotter::RgbdFrame& frame, const cv::Mat& foreground, const BoxMasks& masks)
    {
        if (colour.empty()) Forget(frame.registered.size());
        for (int i = 0; i < (int)frame.registered.total(); i++)
        {
            cv::Vec3b live = frame.registered.at<cv::Vec3b>(i);
            if (live == cv::Vec3b(0, 0, 0) || foreground.data[i] || masks.inPerson.data[i]) continue;   // black = the colour camera can't see it
            cv::Vec3f& remembered = colour.at<cv::Vec3f>(i);
            if (!seen.data[i]) { remembered = live; seen.data[i] = 1; }
            else remembered += ColourBlendRate * (cv::Vec3f(live) - remembered);
        }
    }
    void Forget(cv::Size size)
    {
        colour = cv::Mat::zeros(size, CV_32FC3);
        seen = cv::Mat::zeros(size, CV_8U);
    }
};

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

void Show(const spotter::RgbdFrame& frame, const cv::Mat& foreground, const cv::Mat& remove, const BoxMasks& masks,
          const ColourMemory& memory, bool explain, spotter::Every& report)
{
    cv::Mat picture(frame.registered.size(), CV_8UC3);
    int removed = 0, neverSeen = 0;
    for (int i = 0; i < (int)picture.total(); i++)
    {
        cv::Vec3b live = frame.registered.at<cv::Vec3b>(i);
        cv::Vec3b& pixel = picture.at<cv::Vec3b>(i);
        if (remove.data[i]) { removed++; neverSeen += !memory.seen.data[i]; }
        if (explain)
        {
            cv::Vec3b tint = remove.data[i] ? cv::Vec3b(0, 0, 255) : (masks.inObject.data[i] && foreground.data[i]) ? cv::Vec3b(0, 255, 255)
                           : foreground.data[i] ? cv::Vec3b(0, 255, 0) : live;
            pixel = live / 2 + tint / 2;
        }
        else if (remove.data[i]) pixel = memory.seen.data[i] ? cv::Vec3b(memory.colour.at<cv::Vec3f>(i)) : Magenta;
        else pixel = live;
    }
    std::string status = cv::format("%d person box(es), %d object box(es), %d px removed (%d never seen)", masks.people, masks.objects, removed, neverSeen);
    cv::putText(picture, status, cv::Point(8, 16), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
    cv::imshow(Window, picture);
    if (report.Due()) std::printf("%s\n", status.c_str());
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(KinectSettings());
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;
    const int personClass = yolo.Classes().Find("person");

    spotter::AsyncDetector detector(yolo);
    spotter::AdaptiveBackground background(spotter::AdaptiveBackground::Options{});   // 07c's settings
    ColourMemory memory;
    std::vector<spotter::Detection> detections;   // the latest batch, used until the next one (stale on purpose)
    cv::namedWindow(Window, cv::WINDOW_NORMAL);
    cv::resizeWindow(Window, 1024, 848);
    bool explain = false;
    spotter::Every report(1000);
    spotter::RgbdFrame frame;

    while (kinect.Next(frame))
    {
        if (auto batch = detector.TryTake()) detections = *batch;                     // step 2
        detector.Offer(frame.colour);
        BoxMasks masks = MaskBoxes(detections, frame, personClass);
        cv::Mat foreground = background.Update(frame.depthMm, masks.inPerson);        // step 1 (people are never absorbed)
        cv::Mat remove = WhatToRemove(foreground, masks);                              // steps 3-4
        memory.Learn(frame, foreground, masks);                                        // step 5
        Show(frame, foreground, remove, masks, memory, explain, report);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 'v') explain = !explain;
        if (key == ' ') { background.Forget(); memory.Forget(frame.registered.size()); std::printf("Both memories forgotten.\n"); }
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Start with the room empty, then walk in: you're painted out with the room behind you.
//     Start WITH you in view instead: where you stood is magenta (never seen) until you move.
//  2. Pick up a cup and carry it: the cup floats across the room, with the fingers holding it
//     (they're inside its box). Press V to see the yellow "kept" area follow it, a little late.
//  3. Put the cup down somewhere new and step away. The spot it came from shows bare table
//     within a third of a second (depth "farther" for 07c's leaveFrames).
//  4. Walk fast: in the explain view, does the red cover you, or does a ghost trail behind?
//     PersonShare needs enough of you inside the old box. Try PersonShare = 0.1.
//  5. Wear black, or look at your hair: no depth reading there, so it's never foreground and stays
//     visible. A depth-only silhouette can't remove what depth can't see.
//  6. Stand still for longer than 10 s: you stay removed (person boxes block absorbing).
//  7. Set GrowPixels = 0 and look at your outline: a halo, from colour and depth not lining up.
