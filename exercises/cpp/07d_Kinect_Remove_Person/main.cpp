// =====================================================================
//  07d · Kinect Remove Person
//
//  07c finds everything in front of the remembered room. This example
//  decides which of it is a PERSON, and paints them out with what the
//  room looked like behind them. Anything they carry stays visible.
//
//    1. Depth background (07c) -> foreground blobs.
//    2. YOLO (on its own thread, as in 05c) -> person boxes, object boxes.
//    3. A blob that lies mostly inside person boxes is a person.
//       Whole blobs, not boxes: the boxes lag behind (YOLO runs ~6x a
//       second), but the blob is fresh every frame.
//    4. Paint the person's pixels with the remembered COLOUR of the room,
//       EXCEPT inside another object's box: a carried cup stays.
//    5. The colour memory only learns where there's no foreground and no
//       person, so it keeps what was behind them.
//
//  Never-seen pixels (behind someone since the start) show magenta:
//  guessing those (inpainting) is a later step.
//
//  The picture is the colour camera's view mapped onto the depth pixels
//  (512 x 424, as in 08), so colour and depth line up pixel for pixel.
//
//  Window: the room with people removed.
//  Keys:   V  toggle the explain view (red = removed, yellow = kept object, green = other foreground)
//          SPACE  forget both memories and start over      Q or Esc  quit
// =====================================================================

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/registration.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// --- Who is a person (new) ---
const float PersonConfidence = 0.40f;   // a "person" box must be at least this sure
const double PersonShare = 0.3;         // a blob is a person if this share of it lies in person boxes

// --- What they carry stays (new) ---
const float ObjectConfidence = 0.25f;   // any other class at least this sure is kept visible
const double ObjectBoxGrow = 0.2;       // grow object boxes by this share: YOLO's boxes lag a moving hand

// --- Hiding the edges (new) ---
const int GrowPixels = 4;               // widen the removed area: colour and depth edges don't line up exactly

// --- Colour memory (new) ---
const float ColourBlendRate = 0.1f;     // per frame, 0..1

// --- Depth background, as in 07c ---
const float NearerMeters = 0.10f;
const int AbsorbFrames = 300;           // people are never absorbed: only what isn't in a person box
const int LeaveFrames = 10;
const float BlendRate = 0.05f;
const int MinBlobPixels = 500;
const float MaxDepthMeters = 6.0f;

// --- Detector, as in 05c ---
const int ModelInputSize = 640;
const std::string ModelPath = std::string(SPOTTER_REPO_ROOT) + (ModelInputSize == 320 ? "/yolo26n_320.onnx" : "/yolo26n.onnx");
const bool CapColorExposure = true;
const int PersonClass = 0;              // "person" is class 0 of the 80 COCO classes

const int Width = 512, Height = 424, ColourWidth = 1920, ColourHeight = 1080;
const cv::Vec3b Magenta(255, 0, 255);

struct Detection
{
    int classId;
    float confidence;
    cv::Rect2d box;   // in colour-camera pixels
};

// YOLO on one colour frame (letterbox, forward, undo letterbox: as in 02 and 05c).
std::vector<Detection> Detect(cv::dnn::Net& detector, const cv::Mat& frame)
{
    double scale = std::min((double)ModelInputSize / frame.cols, (double)ModelInputSize / frame.rows);
    int scaledWidth = (int)std::round(frame.cols * scale), scaledHeight = (int)std::round(frame.rows * scale);
    int padLeft = (ModelInputSize - scaledWidth) / 2, padTop = (ModelInputSize - scaledHeight) / 2;
    cv::Mat scaled, letterboxed;
    cv::resize(frame, scaled, cv::Size(scaledWidth, scaledHeight));
    cv::copyMakeBorder(scaled, letterboxed, padTop, ModelInputSize - scaledHeight - padTop,
                       padLeft, ModelInputSize - scaledWidth - padLeft, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    detector.setInput(cv::dnn::blobFromImage(letterboxed, 1.0 / 255.0, cv::Size(ModelInputSize, ModelInputSize),
                                             cv::Scalar(0, 0, 0), true, false));
    cv::Mat output = detector.forward();
    cv::Mat rows(output.size[1], output.size[2], CV_32F, output.ptr<float>());

    std::vector<Detection> detections;
    for (int row = 0; row < rows.rows; row++)
    {
        float confidence = rows.at<float>(row, 4);
        if (confidence < std::min(PersonConfidence, ObjectConfidence)) continue;
        double left = (rows.at<float>(row, 0) - padLeft) / scale, top = (rows.at<float>(row, 1) - padTop) / scale;
        double right = (rows.at<float>(row, 2) - padLeft) / scale, bottom = (rows.at<float>(row, 3) - padTop) / scale;
        detections.push_back({ (int)rows.at<float>(row, 5), confidence, cv::Rect2d(left, top, right - left, bottom - top) });
    }
    return detections;
}

// As in 07c, plus: pixels in a person box are never absorbed (a person standing still stays a person).
cv::Mat UpdateBackground(const std::vector<float>& liveMm, std::vector<float>& backgroundMm,
                         std::vector<int>& nearerFrames, std::vector<int>& fartherFrames,
                         const std::vector<uchar>& inPerson)
{
    cv::Mat foreground(Height, Width, CV_8U, cv::Scalar(0));
    for (int i = 0; i < Width * Height; i++)
    {
        float live = liveMm[i], remembered = backgroundMm[i];
        if (live <= 0) continue;
        if (remembered <= 0) { backgroundMm[i] = live; continue; }
        if (live < remembered - NearerMeters * 1000)
        {
            fartherFrames[i] = 0;
            if (++nearerFrames[i] < AbsorbFrames || inPerson[i]) { foreground.data[i] = 255; continue; }
            backgroundMm[i] = live;
        }
        else if (live > remembered + NearerMeters * 1000)
        {
            nearerFrames[i] = 0;
            if (++fartherFrames[i] < LeaveFrames) continue;
            backgroundMm[i] = live;
        }
        else backgroundMm[i] += BlendRate * (live - remembered);
        nearerFrames[i] = fartherFrames[i] = 0;
    }
    return foreground;
}

int main()
{
    // -----------------------------------------------------------------
    //  Setup: detector, Kinect with depth + colour + registration
    // -----------------------------------------------------------------

    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (!std::filesystem::exists(ModelPath))
    {
        std::printf("Model not found: %s\n", ModelPath.c_str());
        return 1;
    }
    cv::dnn::Net detector = cv::dnn::readNetFromONNX(ModelPath);

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
    libfreenect2::Frame undistorted(Width, Height, 4), registered(Width, Height, 4);
    std::vector<int> colourIndexOfDepthPixel(Width * Height);

    // The detector thread, trimmed from 05c: the main loop hands it a frame, it hands back boxes.
    std::mutex shared;
    std::condition_variable frameHandedOver;
    cv::Mat frameForDetector;
    std::vector<Detection> detectorResult;
    bool frameWaiting = false, resultWaiting = false, quit = false;
    std::thread detectorThread([&]
    {
        while (true)
        {
            cv::Mat image;
            {
                std::unique_lock<std::mutex> lock(shared);
                frameHandedOver.wait(lock, [&] { return frameWaiting || quit; });
                if (quit) return;
                image = frameForDetector;
                frameWaiting = false;
            }
            std::vector<Detection> result = Detect(detector, image);
            std::lock_guard<std::mutex> lock(shared);
            detectorResult = std::move(result);
            resultWaiting = true;
        }
    });
    bool detectorBusy = false;
    std::vector<Detection> detections;   // the latest batch, used until the next one arrives

    // The two memories: depth (07c) and colour (new), per depth pixel.
    std::vector<float> liveMm(Width * Height), backgroundMm(Width * Height, 0);
    std::vector<int> nearerFrames(Width * Height, 0), fartherFrames(Width * Height, 0);
    std::vector<cv::Vec3f> backgroundColour(Width * Height);
    std::vector<uchar> colourSeen(Width * Height, 0);

    const std::string window = "07d remove person  (V explain view, SPACE forget, Q quit)";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, Width * 2, Height * 2);
    bool explain = false;
    int frameCount = 0;

    // -----------------------------------------------------------------
    //  Main operation
    // -----------------------------------------------------------------

    libfreenect2::FrameMap frames;
    while (true)
    {
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == 'v') explain = !explain;
        if (key == ' ')
        {
            std::fill(backgroundMm.begin(), backgroundMm.end(), 0.0f);
            std::fill(nearerFrames.begin(), nearerFrames.end(), 0);
            std::fill(fartherFrames.begin(), fartherFrames.end(), 0);
            std::fill(colourSeen.begin(), colourSeen.end(), 0);
            std::printf("Both memories forgotten.\n");
        }

        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }
        libfreenect2::Frame* colourFrame = frames[libfreenect2::Frame::Color];
        registration.apply(colourFrame, frames[libfreenect2::Frame::Depth], &undistorted, &registered,
                           false, nullptr, colourIndexOfDepthPixel.data());
        const float* depthMm = (const float*)undistorted.data;
        for (int i = 0; i < Width * Height; i++)
            liveMm[i] = (depthMm[i] > 0 && !std::isinf(depthMm[i])) ? depthMm[i] : 0;

        // --- Step 1: YOLO in the background (as in 05c) ----------------------
        {
            std::lock_guard<std::mutex> lock(shared);
            if (resultWaiting)
            {
                detections = std::move(detectorResult);
                resultWaiting = false;
                detectorBusy = false;
            }
            if (!detectorBusy)
            {
                cv::cvtColor(cv::Mat(ColourHeight, ColourWidth, CV_8UC4, colourFrame->data), frameForDetector, cv::COLOR_BGRA2BGR);
                frameWaiting = true;
                detectorBusy = true;
                frameHandedOver.notify_one();
            }
        }
        listener.release(frames);

        // --- Step 2: which depth pixels lie in a person box, or another object's box --
        std::vector<cv::Rect2d> personBoxes, objectBoxes;
        for (const Detection& d : detections)
        {
            if (d.classId == PersonClass && d.confidence >= PersonConfidence) personBoxes.push_back(d.box);
            if (d.classId != PersonClass && d.confidence >= ObjectConfidence)
                objectBoxes.push_back(cv::Rect2d(d.box.x - d.box.width * ObjectBoxGrow / 2, d.box.y - d.box.height * ObjectBoxGrow / 2,
                                                 d.box.width * (1 + ObjectBoxGrow), d.box.height * (1 + ObjectBoxGrow)));
        }
        std::vector<uchar> inPerson(Width * Height, 0), inObject(Width * Height, 0);
        for (int i = 0; i < Width * Height; i++)
        {
            int c = colourIndexOfDepthPixel[i];
            if (c < 0) continue;   // the colour camera doesn't see this depth pixel
            cv::Point2d spot(c % ColourWidth, c / ColourWidth);
            for (const cv::Rect2d& box : personBoxes) if (box.contains(spot)) inPerson[i] = 1;
            for (const cv::Rect2d& box : objectBoxes) if (box.contains(spot)) inObject[i] = 1;
        }

        // --- Step 3: depth background -> foreground (07c) --------------------
        cv::Mat foreground = UpdateBackground(liveMm, backgroundMm, nearerFrames, fartherFrames, inPerson);

        // --- Step 4: a blob mostly inside person boxes is a person: remove it, keep carried objects --
        cv::Mat labels, stats, centroids;
        int count = cv::connectedComponentsWithStats(foreground, labels, stats, centroids);
        const int* label = labels.ptr<int>();
        std::vector<int> personPixels(count, 0);
        for (int i = 0; i < Width * Height; i++) if (label[i] > 0 && inPerson[i]) personPixels[label[i]]++;
        cv::Mat remove(Height, Width, CV_8U, cv::Scalar(0));
        for (int i = 0; i < Width * Height; i++)
        {
            int area = stats.at<int>(label[i], cv::CC_STAT_AREA);
            if (label[i] > 0 && area >= MinBlobPixels && personPixels[label[i]] >= PersonShare * area) remove.data[i] = 255;
        }
        cv::dilate(remove, remove, cv::Mat(), cv::Point(-1, -1), GrowPixels);   // hide the edge halo
        for (int i = 0; i < Width * Height; i++) if (inObject[i]) remove.data[i] = 0;   // the cup stays

        // --- Step 5: colour memory: learn only where nobody is in the way ----
        const uchar* bgrx = registered.data;
        for (int i = 0; i < Width * Height; i++)
        {
            if (colourIndexOfDepthPixel[i] < 0 || foreground.data[i] || inPerson[i]) continue;
            cv::Vec3f live(bgrx[4 * i], bgrx[4 * i + 1], bgrx[4 * i + 2]);
            if (!colourSeen[i]) { backgroundColour[i] = live; colourSeen[i] = 1; }
            else backgroundColour[i] += ColourBlendRate * (live - backgroundColour[i]);
        }

        // -----------------------------------------------------------------
        //  Output
        // -----------------------------------------------------------------

        cv::Mat picture(Height, Width, CV_8UC3);
        int removed = 0, neverSeen = 0;
        for (int i = 0; i < Width * Height; i++)
        {
            cv::Vec3b live(bgrx[4 * i], bgrx[4 * i + 1], bgrx[4 * i + 2]);
            cv::Vec3b& pixel = picture.at<cv::Vec3b>(i / Width, i % Width);
            if (remove.data[i]) { removed++; if (!colourSeen[i]) neverSeen++; }
            if (explain)
            {
                cv::Vec3b tint = remove.data[i] ? cv::Vec3b(0, 0, 255) : (inObject[i] && foreground.data[i]) ? cv::Vec3b(0, 255, 255)
                               : foreground.data[i] ? cv::Vec3b(0, 255, 0) : live;
                pixel = live / 2 + tint / 2;
            }
            else if (remove.data[i]) pixel = colourSeen[i] ? cv::Vec3b(backgroundColour[i]) : Magenta;
            else pixel = live;
        }
        cv::putText(picture, cv::format("%zu person box(es), %zu object box(es), %d px removed (%d never seen)",
                    personBoxes.size(), objectBoxes.size(), removed, neverSeen),
                    cv::Point(8, 16), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
        cv::imshow(window, picture);

        if (++frameCount % 30 == 0)
            std::printf("%zu person box(es), %zu object box(es), %d px removed, %d never seen\n",
                        personBoxes.size(), objectBoxes.size(), removed, neverSeen);
    }

    {
        std::lock_guard<std::mutex> lock(shared);
        quit = true;
    }
    frameHandedOver.notify_one();
    detectorThread.join();
    device->stop();
    device->close();
    cv::destroyAllWindows();
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Start with the room empty, then walk in: you're painted out with the room behind you.
//     Start WITH you in view instead: where you stood is magenta (never seen) until you move.
//  2. Pick up a cup and carry it: the cup floats across the room, with the fingers holding it
//     (they're inside its box). Press V to see the yellow "kept" area follow it, a little late.
//  3. Put the cup down somewhere new and step away. The cup is shown live; the spot it came from
//     shows bare table within a third of a second (depth "farther" for LeaveFrames).
//  4. Walk fast: in the explain view, does the red cover you, or does a ghost trail behind?
//     Blobs are fresh every frame, but PersonShare needs enough of you inside the old box.
//     Try PersonShare = 0.1.
//  5. Wear black, or look at your hair: no depth reading there, so it's never foreground and stays
//     visible. A depth-only silhouette can't remove what depth can't see (07c idea 5).
//  6. Stand still for longer than AbsorbFrames: you stay removed (person boxes block absorbing).
//     Set PersonConfidence = 0.9 so YOLO loses you sometimes, and watch yourself fade into the room.
//  7. Set GrowPixels = 0 and look at your outline: a halo, from colour and depth not lining up.
