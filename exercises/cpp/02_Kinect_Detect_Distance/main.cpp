// =====================================================================
//  02 · Kinect Detect + Distance
//
//  Run YOLO on the Kinect's colour camera, then use the depth camera to
//  measure every box:   person 1.84m x-0.31 y+0.12
//
//  This is the guide's "late fusion" (§1, §5): the detector never sees depth.
//  Depth is only used AFTER detection, in four visible steps:
//
//    1. Detect      YOLO finds boxes in the 1920 x 1080 colour image.
//    2. Register    libfreenect2 tells us where every depth pixel lands in
//                   the colour image (the two cameras don't line up, §5.1).
//    3. Sample      Collect the depth pixels that land in the CENTRE of each
//                   box and take their median (§5.2).
//    4. Deproject   Pixel + depth -> X, Y, Z in metres, using the depth
//                   camera's factory intrinsics (§5.3, §5.4).
//
//  Window: thick box = detection, thin box = the centre area that was sampled.
//  Console: timing per step, and each detection's numbers next to libfreenect2's
//           own getPointXYZ answer, so you can check the hand-done math.
//
//  Keys:   Q or Esc  quit
//
//  Needs:  yolo26n.onnx in the repo root (see README for the one-line export).
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
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

// SPOTTER_REPO_ROOT is the repo's folder, filled in by CMake, so this works from any folder.
const std::string ModelPath = SPOTTER_REPO_ROOT "/yolo26n.onnx";

// The square image size the model was exported with (Ultralytics' default imgsz=640).
const int ModelInputSize = 640;

// Detections scoring below this are ignored. Lower = more boxes, more mistakes.
const float MinConfidence = 0.40f;

// Fraction of the box (in each direction) to sample depth from, around its centre.
// 0.5 = the middle half. 1.0 = the whole box, which usually measures the wall behind
// the object, not the object (the guide's second self-check assertion).
const double BoxShrink = 0.5;

// Fewer depth pixels than this inside the sampled area -> "?m" instead of a guess.
// Black, shiny and too-close objects end up here.
const int MinDepthSamples = 20;

// Only draw detections closer than this. Set 1.5 for a "within arm's reach" filter (guide §10).
const double MaxDistanceMeters = 100.0;

// How often the console report prints, in milliseconds.
const int ReportEveryMs = 1000;

// The 80 COCO classes yolo26n.pt was trained on, in classId order.
const char* ClassNames[] =
{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
};

// Middle value of a list. For an even count, the average of the two middle values (like numpy's median).
float Median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) return values[middle];
    return (values[middle - 1] + values[middle]) / 2;
}

double MillisecondsSince(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // print immediately, even when piped (see example 01)

    // -----------------------------------------------------------------
    //  Setup: the detector
    // -----------------------------------------------------------------

    if (!std::filesystem::exists(ModelPath))
    {
        std::printf("Model not found: %s\n", std::filesystem::absolute(ModelPath).string().c_str());
        std::printf("Export it once from the repo root:\n");
        std::printf("  yolo export model=yolo26n.pt format=onnx nms=False\n");
        return 1;
    }

    // OpenCV's DNN module runs the ONNX file directly: no Python, no extra runtime.
    // The model was exported with nms=False, which for YOLO26 means "end-to-end": the network
    // itself removes duplicate boxes and outputs at most 300 final detections, one per row:
    //     x1, y1, x2, y2, confidence, classId        (box corners in 640 x 640 model-input pixels)
    cv::dnn::Net detector = cv::dnn::readNetFromONNX(ModelPath);
    std::printf("Loaded %s\n", ModelPath.c_str());

    // -----------------------------------------------------------------
    //  Setup: the Kinect
    // -----------------------------------------------------------------

    libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Warning));
    libfreenect2::Freenect2 freenect2;
    if (freenect2.enumerateDevices() == 0)
    {
        std::printf("No Kinect v2 found. Check the adapter's power light, use a USB 3.0 port,\n");
        std::printf("and check the USB driver is UsbDk or libusbK (Windows) - see README.\n");
        return 1;
    }
    libfreenect2::Freenect2Device* device = freenect2.openDefaultDevice();
    if (device == nullptr)
    {
        std::printf("Found a Kinect but could not open it. Is another program using it?\n");
        return 1;
    }

    // One listener for both cameras: waitForNewFrame returns once it has a colour AND a depth frame.
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Depth);
    device->setColorFrameListener(&listener);
    device->setIrAndDepthFrameListener(&listener);
    device->startStreams(true, true);   // (colour, depth)

    const int colorWidth = 1920, colorHeight = 1080, depthWidth = 512, depthHeight = 424;
    std::printf("Colour %d x %d, depth %d x %d\n", colorWidth, colorHeight, depthWidth, depthHeight);

    // The depth camera's intrinsics (focal lengths and principal point, guide §5.4) and the
    // colour camera's parameters come from the device's factory calibration.
    libfreenect2::Freenect2Device::IrCameraParams depthIntrinsics = device->getIrCameraParams();
    std::printf("Depth intrinsics: fx %.1f  fy %.1f  cx %.1f  cy %.1f  (pixels)\n",
                depthIntrinsics.fx, depthIntrinsics.fy, depthIntrinsics.cx, depthIntrinsics.cy);

    // Registration does step 2. It also removes the depth lens's distortion: its "undistorted"
    // depth image is a perfect pinhole camera, so step 4's simple formula is exact on it.
    libfreenect2::Registration registration(depthIntrinsics, device->getColorCameraParams());
    libfreenect2::Frame undistorted(depthWidth, depthHeight, 4);   // depth in mm, lens distortion removed
    libfreenect2::Frame registered(depthWidth, depthHeight, 4);    // colour per depth pixel (unused here)

    // For every undistorted depth pixel: its position in the colour image, as  y * 1920 + x.
    // -1 means "no depth reading" or "lands outside the colour image".
    std::vector<int> colorIndexOfDepthPixel(depthWidth * depthHeight);

    cv::Mat frame;   // the colour picture as 3-channel BGR, which the model and the drawing use

    // Letterbox: the model wants a 640 x 640 square. Shrink the 16:9 image to fit, keeping its
    // shape, and fill the rest with grey, just as Ultralytics does in Python. These numbers
    // never change, so work them out once.
    double modelScale = std::min((double)ModelInputSize / colorWidth, (double)ModelInputSize / colorHeight);
    int scaledWidth = (int)std::round(colorWidth * modelScale);
    int scaledHeight = (int)std::round(colorHeight * modelScale);
    int padLeft = (ModelInputSize - scaledWidth) / 2;
    int padTop = (ModelInputSize - scaledHeight) / 2;
    std::printf("Letterbox: scale %.4f, image %d x %d, padding %d px left, %d px top\n",
                modelScale, scaledWidth, scaledHeight, padLeft, padTop);

    const std::string windowName = "02 Kinect detect + distance  (Q quits)";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, colorWidth / 2, colorHeight / 2);

    auto reportStart = std::chrono::steady_clock::now();
    int framesSinceReport = 0;

    // -----------------------------------------------------------------
    //  Main operation
    // -----------------------------------------------------------------

    libfreenect2::FrameMap frames;
    while (true)
    {
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;   // 27 = Esc

        // --- Capture -------------------------------------------------------
        auto stepStart = std::chrono::steady_clock::now();
        if (!listener.waitForNewFrame(frames, 10 * 1000))
        {
            std::printf("No frame for 10 seconds - giving up.\n");
            break;
        }
        libfreenect2::Frame* colorFrame = frames[libfreenect2::Frame::Color];
        libfreenect2::Frame* depthFrame = frames[libfreenect2::Frame::Depth];
        // The camera sends JPEG. libfreenect2 decodes it to BGRX: 4 bytes per pixel, the 4th unused.
        cv::Mat colorBgrx(colorHeight, colorWidth, CV_8UC4, colorFrame->data);
        cv::cvtColor(colorBgrx, frame, cv::COLOR_BGRA2BGR);
        double captureMs = MillisecondsSince(stepStart);
        framesSinceReport++;

        // --- Step 2: Register ----------------------------------------------
        // (Done before step 1 because it needs this frame's data, which is handed back right after.)
        // false = skip libfreenect2's occlusion filter, so this matches the Microsoft SDK's mapping.
        stepStart = std::chrono::steady_clock::now();
        registration.apply(colorFrame, depthFrame, &undistorted, &registered, false, nullptr, colorIndexOfDepthPixel.data());
        const float* depthMm = (const float*)undistorted.data;
        double registerMs = MillisecondsSince(stepStart);
        listener.release(frames);

        // --- Step 1: Detect ------------------------------------------------
        stepStart = std::chrono::steady_clock::now();
        cv::Mat scaled, letterboxed;
        cv::resize(frame, scaled, cv::Size(scaledWidth, scaledHeight));
        cv::copyMakeBorder(scaled, letterboxed,
                           padTop, ModelInputSize - scaledHeight - padTop,
                           padLeft, ModelInputSize - scaledWidth - padLeft,
                           cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        // blobFromImage turns the picture into what the network expects:
        // pixel values 0..1 (scale 1/255), RGB order (swapRB), laid out as 1 x 3 x 640 x 640.
        cv::Mat inputBlob = cv::dnn::blobFromImage(letterboxed, 1.0 / 255.0, cv::Size(ModelInputSize, ModelInputSize),
                                                   cv::Scalar(0, 0, 0), true, false);
        detector.setInput(inputBlob);
        cv::Mat output = detector.forward();   // shape 1 x 300 x 6
        cv::Mat detections(output.size[1], output.size[2], CV_32F, output.ptr<float>());   // view it as 300 rows x 6
        double detectMs = MillisecondsSince(stepStart);

        // --- Steps 3 + 4: Sample and deproject, one detection at a time -----
        stepStart = std::chrono::steady_clock::now();
        std::vector<std::string> reportLines;

        for (int row = 0; row < detections.rows; row++)
        {
            float confidence = detections.at<float>(row, 4);
            if (confidence < MinConfidence) continue;
            int classId = (int)detections.at<float>(row, 5);
            std::string label = ClassNames[classId];

            // Undo the letterbox: model-input pixels -> colour-image pixels.
            double boxLeft = (detections.at<float>(row, 0) - padLeft) / modelScale;
            double boxTop = (detections.at<float>(row, 1) - padTop) / modelScale;
            double boxRight = (detections.at<float>(row, 2) - padLeft) / modelScale;
            double boxBottom = (detections.at<float>(row, 3) - padTop) / modelScale;

            // The centre part of the box that we sample depth from.
            double boxCenterX = (boxLeft + boxRight) / 2;
            double boxCenterY = (boxTop + boxBottom) / 2;
            double sampleHalfWidth = (boxRight - boxLeft) * BoxShrink / 2;
            double sampleHalfHeight = (boxBottom - boxTop) * BoxShrink / 2;
            double sampleLeft = boxCenterX - sampleHalfWidth;
            double sampleRight = boxCenterX + sampleHalfWidth;
            double sampleTop = boxCenterY - sampleHalfHeight;
            double sampleBottom = boxCenterY + sampleHalfHeight;

            // Step 3: every depth pixel whose colour position falls inside the sample area.
            std::vector<float> sampleDepthsMm, sampleDepthColumns, sampleDepthRows;
            for (int depthY = 0; depthY < depthHeight; depthY++)
            {
                for (int depthX = 0; depthX < depthWidth; depthX++)
                {
                    int i = depthY * depthWidth + depthX;
                    int colorIndex = colorIndexOfDepthPixel[i];
                    if (colorIndex < 0) continue;
                    int colorX = colorIndex % colorWidth;
                    int colorY = colorIndex / colorWidth;
                    bool insideSampleArea = colorX >= sampleLeft && colorX < sampleRight
                                         && colorY >= sampleTop && colorY < sampleBottom;
                    if (insideSampleArea && depthMm[i] > 0 && !std::isinf(depthMm[i]))
                    {
                        sampleDepthsMm.push_back(depthMm[i]);
                        sampleDepthColumns.push_back((float)depthX);
                        sampleDepthRows.push_back((float)depthY);
                    }
                }
            }

            char text[96], line[200];
            if ((int)sampleDepthsMm.size() < MinDepthSamples)
            {
                std::snprintf(text, sizeof text, "%s ?m", label.c_str());
                std::snprintf(line, sizeof line, "  %-12s %.2f  ?m  (only %d depth samples)",
                              label.c_str(), confidence, (int)sampleDepthsMm.size());
            }
            else
            {
                // The median ignores the few background and edge pixels that are left (§5.2).
                float zMm = Median(sampleDepthsMm);
                // Where the object's centre is in the DEPTH image: the middle of the sampled pixels.
                float depthU = Median(sampleDepthColumns);
                float depthV = Median(sampleDepthRows);

                // Step 4: pinhole camera model (§5.3).  X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fy.
                // x is to the right in the picture, y is down, z is straight out of the sensor.
                double zMeters = zMm / 1000.0;
                if (zMeters > MaxDistanceMeters) continue;
                double xMeters = (depthU - depthIntrinsics.cx) * zMeters / depthIntrinsics.fx;
                double yMeters = (depthV - depthIntrinsics.cy) * zMeters / depthIntrinsics.fy;

                // libfreenect2 can do step 4 for us, for one whole pixel (row, column).
                // It uses that pixel's own depth, and measures from the pixel's centre (+0.5).
                float libX, libY, libZ;
                registration.getPointXYZ(&undistorted, (int)std::lround(depthV), (int)std::lround(depthU), libX, libY, libZ);

                std::snprintf(text, sizeof text, "%s %.2fm x%+.2f y%+.2f", label.c_str(), zMeters, xMeters, yMeters);
                std::snprintf(line, sizeof line, "  %-12s %.2f  %-34s  samples %5d  | lib: X%+.2f Y%+.2f Z%.2f",
                              label.c_str(), confidence, text, (int)sampleDepthsMm.size(), libX, libY, libZ);
            }
            reportLines.push_back(line);

            // --- Output: draw ----------------------------------------------
            cv::Scalar green(0, 255, 0), yellow(0, 255, 255);
            cv::rectangle(frame, cv::Point((int)boxLeft, (int)boxTop), cv::Point((int)boxRight, (int)boxBottom), green, 3);
            cv::rectangle(frame, cv::Point((int)sampleLeft, (int)sampleTop), cv::Point((int)sampleRight, (int)sampleBottom), yellow, 1);
            cv::putText(frame, text, cv::Point((int)boxLeft, std::max((int)boxTop - 10, 30)),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, green, 2);
        }
        double fuseMs = MillisecondsSince(stepStart);

        // -----------------------------------------------------------------
        //  Output
        // -----------------------------------------------------------------

        cv::imshow(windowName, frame);

        double reportElapsedMs = MillisecondsSince(reportStart);
        if (reportElapsedMs >= ReportEveryMs)
        {
            std::printf("%.1f fps | capture %.0f ms, detect %.0f ms, register %.0f ms, sample+deproject %.0f ms | %d detections\n",
                        framesSinceReport * 1000.0 / reportElapsedMs, captureMs, detectMs, registerMs, fuseMs, (int)reportLines.size());
            for (const std::string& l : reportLines) std::printf("%s\n", l.c_str());
            reportStart = std::chrono::steady_clock::now();
            framesSinceReport = 0;
        }
    }

    device->stop();
    device->close();
    cv::destroyAllWindows();
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Stand 1 m in front of a far wall and set BoxShrink = 1.0. Your distance jumps toward the
//     wall's, because most pixels in a loose box are background. That's why the centre is used.
//  2. Walk slowly sideways across the view at a fixed distance. The number should stay steady
//     even at the edges. If it jumps to the wall at the edges, registration is off (guide §7 test 3).
//  3. Check against a tape measure at 1, 2 and 3 m (guide §7 test 2). Expect a few cm.
//  4. Compare the "x y" from the pinhole math with libfreenect2's X Y in the console. They agree
//     everywhere, even in the corners, because the depth was undistorted first. (The C# version
//     used the raw depth, and drifted from the SDK toward the corners.) Small differences come
//     from the +0.5 pixel-centre convention and the single pixel vs the median.
//  5. Set MaxDistanceMeters = 1.5 and walk toward the sensor: boxes appear only within reach.
//  6. Hold up a phone showing a photo of a person. YOLO says "person". The distance says 0.4 m.
//     How would you reject it? (Hint: real height = box height in pixels * Z / focal length; guide §10.)
