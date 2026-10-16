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
//    2. Register    The SDK tells us where every depth pixel lands in the
//                   colour image (the two cameras don't line up, §5.1).
//    3. Sample      Collect the depth pixels that land in the CENTRE of each
//                   box and take their median (§5.2).
//    4. Deproject   Pixel + depth -> X, Y, Z in metres, using the depth
//                   camera's factory intrinsics (§5.3, §5.4).
//
//  Window: thick box = detection, thin box = the centre area that was sampled.
//  Console: timing per step, and each detection's numbers next to the SDK's
//           own CoordinateMapper answer, so you can check the hand-done math.
//
//  Keys:   Q or Esc  quit
//
//  Needs:  yolo26n.onnx in the repo root (see README for the one-line export).
// =====================================================================

using System.Diagnostics;
using Microsoft.Kinect;
using OpenCvSharp;
using OpenCvSharp.Dnn;

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// Relative to this folder, where `dotnet run` starts.
const string ModelPath = @"..\..\..\yolo26n.onnx";

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

// ---------------------------------------------------------------------
//  Setup: the detector
// ---------------------------------------------------------------------

if (!File.Exists(ModelPath))
{
    Console.WriteLine($"Model not found: {Path.GetFullPath(ModelPath)}");
    Console.WriteLine("Export it once from the repo root:  yolo export model=yolo26n.pt format=onnx nms=False");
    return;
}

// OpenCV's DNN module runs the ONNX file directly: no Python, no extra runtime.
// The model was exported with nms=False, which for YOLO26 means "end-to-end": the network
// itself removes duplicate boxes and outputs at most 300 final detections, one per row:
//     x1, y1, x2, y2, confidence, classId        (box corners in 640 x 640 model-input pixels)
Net detector = CvDnn.ReadNetFromOnnx(ModelPath);
Console.WriteLine($"Loaded {Path.GetFileName(ModelPath)}");

// The 80 COCO classes yolo26n.pt was trained on, in classId order.
string[] classNames =
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

// ---------------------------------------------------------------------
//  Setup: the Kinect
// ---------------------------------------------------------------------

KinectSensor sensor = KinectSensor.GetDefault();
sensor.Open();

Console.WriteLine("Waiting for the Kinect...");
var waitTimer = Stopwatch.StartNew();
while (!sensor.IsAvailable && waitTimer.ElapsedMilliseconds < 10_000)
{
    Thread.Sleep(100);
}
if (!sensor.IsAvailable)
{
    Console.WriteLine("No Kinect v2 found. Check the adapter's power light, use a USB 3.0 port,");
    Console.WriteLine("and run Tools\\ConfigurationVerifier in the Kinect SDK folder.");
    sensor.Close();
    return;
}

int colorWidth = sensor.ColorFrameSource.FrameDescription.Width;     // 1920
int colorHeight = sensor.ColorFrameSource.FrameDescription.Height;   // 1080
int depthWidth = sensor.DepthFrameSource.FrameDescription.Width;     // 512
int depthHeight = sensor.DepthFrameSource.FrameDescription.Height;   // 424
Console.WriteLine($"Colour {colorWidth} x {colorHeight}, depth {depthWidth} x {depthHeight}");

// A multi-source reader hands us a colour frame and a depth frame captured at the same moment.
MultiSourceFrameReader reader = sensor.OpenMultiSourceFrameReader(FrameSourceTypes.Color | FrameSourceTypes.Depth);

// Buffers, allocated once and refilled every frame.
byte[] colorBgra = new byte[colorWidth * colorHeight * 4];       // 4 bytes per pixel: blue, green, red, unused
ushort[] depthMm = new ushort[depthWidth * depthHeight];         // millimetres, 0 = no reading
ColorSpacePoint[] colorPointOfDepthPixel = new ColorSpacePoint[depthWidth * depthHeight];

// This Mat shares memory with colorBgra: copying into the array updates the image.
Mat colorBgraImage = Mat.FromPixelData(colorHeight, colorWidth, MatType.CV_8UC4, colorBgra);
Mat frame = new Mat();   // the same picture as 3-channel BGR, which the model and the drawing use

// Letterbox: the model wants a 640 x 640 square. Shrink the 16:9 image to fit, keeping its
// shape, and fill the rest with grey, just as Ultralytics does in Python. These numbers
// never change, so work them out once.
double modelScale = Math.Min((double)ModelInputSize / colorWidth, (double)ModelInputSize / colorHeight);
int scaledWidth = (int)Math.Round(colorWidth * modelScale);
int scaledHeight = (int)Math.Round(colorHeight * modelScale);
int padLeft = (ModelInputSize - scaledWidth) / 2;
int padTop = (ModelInputSize - scaledHeight) / 2;
Console.WriteLine($"Letterbox: scale {modelScale:F4}, image {scaledWidth} x {scaledHeight}, "
                + $"padding {padLeft} px left, {padTop} px top");

// The depth camera's intrinsics (focal lengths and principal point, guide §5.4) come from the
// device's factory calibration. The SDK returns zeros until the first frames have arrived,
// so they are read inside the loop.
CameraIntrinsics depthIntrinsics = new CameraIntrinsics();

const string WindowName = "02 Kinect detect + distance  (Q quits)";
Cv2.NamedWindow(WindowName, WindowFlags.Normal);
Cv2.ResizeWindow(WindowName, colorWidth / 2, colorHeight / 2);

var reportTimer = Stopwatch.StartNew();
var stepTimer = new Stopwatch();
int framesSinceReport = 0;

// ---------------------------------------------------------------------
//  Main operation
// ---------------------------------------------------------------------

while (true)
{
    int key = Cv2.WaitKey(1);
    if (key == 'q' || key == 27) break;   // 27 = Esc

    // --- Capture -------------------------------------------------------
    // Either frame can be null if it expired before we asked for it. Then try again.
    stepTimer.Restart();
    MultiSourceFrame multiFrame = reader.AcquireLatestFrame();
    if (multiFrame == null) continue;
    using (ColorFrame colorFrame = multiFrame.ColorFrameReference.AcquireFrame())
    using (DepthFrame depthFrame = multiFrame.DepthFrameReference.AcquireFrame())
    {
        if (colorFrame == null || depthFrame == null) continue;
        // The camera sends YUY2 internally. The SDK converts it to BGRA for us.
        colorFrame.CopyConvertedFrameDataToArray(colorBgra, ColorImageFormat.Bgra);
        depthFrame.CopyFrameDataToArray(depthMm);
    }
    Cv2.CvtColor(colorBgraImage, frame, ColorConversionCodes.BGRA2BGR);
    double captureMs = stepTimer.Elapsed.TotalMilliseconds;
    framesSinceReport++;

    if (depthIntrinsics.FocalLengthX == 0)
    {
        depthIntrinsics = sensor.CoordinateMapper.GetDepthCameraIntrinsics();
        if (depthIntrinsics.FocalLengthX != 0)
        {
            Console.WriteLine($"Depth intrinsics: fx {depthIntrinsics.FocalLengthX:F1}  fy {depthIntrinsics.FocalLengthY:F1}"
                            + $"  cx {depthIntrinsics.PrincipalPointX:F1}  cy {depthIntrinsics.PrincipalPointY:F1}  (pixels)");
        }
    }

    // --- Step 1: Detect ------------------------------------------------
    // 'using' frees each Mat's native memory at the end of this loop pass. Without it, a few MB
    // per frame pile up until the garbage collector gets around to them.
    stepTimer.Restart();
    using Mat scaled = new Mat();
    Cv2.Resize(frame, scaled, new Size(scaledWidth, scaledHeight));
    using Mat letterboxed = new Mat();
    Cv2.CopyMakeBorder(scaled, letterboxed,
                       padTop, ModelInputSize - scaledHeight - padTop,
                       padLeft, ModelInputSize - scaledWidth - padLeft,
                       BorderTypes.Constant, new Scalar(114, 114, 114));

    // BlobFromImage turns the picture into what the network expects:
    // pixel values 0..1 (scale 1/255), RGB order (swapRB), laid out as 1 x 3 x 640 x 640.
    using Mat inputBlob = CvDnn.BlobFromImage(letterboxed, 1.0 / 255.0, new Size(ModelInputSize, ModelInputSize),
                                        new Scalar(0, 0, 0), swapRB: true, crop: false);
    detector.SetInput(inputBlob);
    using Mat detections = detector.Forward();   // shape 1 x 300 x 6
    double detectMs = stepTimer.Elapsed.TotalMilliseconds;

    // --- Step 2: Register ----------------------------------------------
    // For every depth pixel, where does it land in the colour image? Pixels with no depth
    // reading get -infinity, which fails every "is it inside the box" test below.
    stepTimer.Restart();
    sensor.CoordinateMapper.MapDepthFrameToColorSpace(depthMm, colorPointOfDepthPixel);
    double registerMs = stepTimer.Elapsed.TotalMilliseconds;

    // --- Steps 3 + 4: Sample and deproject, one detection at a time -----
    stepTimer.Restart();
    var reportLines = new List<string>();

    for (int row = 0; row < detections.Size(1); row++)
    {
        float confidence = detections.At<float>(0, row, 4);
        if (confidence < MinConfidence) continue;
        int classId = (int)detections.At<float>(0, row, 5);
        string label = classNames[classId];

        // Undo the letterbox: model-input pixels -> colour-image pixels.
        double boxLeft = (detections.At<float>(0, row, 0) - padLeft) / modelScale;
        double boxTop = (detections.At<float>(0, row, 1) - padTop) / modelScale;
        double boxRight = (detections.At<float>(0, row, 2) - padLeft) / modelScale;
        double boxBottom = (detections.At<float>(0, row, 3) - padTop) / modelScale;

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
        var sampleDepthsMm = new List<float>();
        var sampleDepthColumns = new List<float>();
        var sampleDepthRows = new List<float>();
        for (int depthY = 0; depthY < depthHeight; depthY++)
        {
            for (int depthX = 0; depthX < depthWidth; depthX++)
            {
                int i = depthY * depthWidth + depthX;
                ColorSpacePoint colorPoint = colorPointOfDepthPixel[i];
                bool insideSampleArea = colorPoint.X >= sampleLeft && colorPoint.X < sampleRight
                                     && colorPoint.Y >= sampleTop && colorPoint.Y < sampleBottom;
                if (insideSampleArea && depthMm[i] > 0)
                {
                    sampleDepthsMm.Add(depthMm[i]);
                    sampleDepthColumns.Add(depthX);
                    sampleDepthRows.Add(depthY);
                }
            }
        }

        string text;
        if (sampleDepthsMm.Count < MinDepthSamples || depthIntrinsics.FocalLengthX == 0)
        {
            text = $"{label} ?m";
            reportLines.Add($"  {label,-12} {confidence:F2}  ?m  (only {sampleDepthsMm.Count} depth samples)");
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
            double xMeters = (depthU - depthIntrinsics.PrincipalPointX) * zMeters / depthIntrinsics.FocalLengthX;
            double yMeters = (depthV - depthIntrinsics.PrincipalPointY) * zMeters / depthIntrinsics.FocalLengthY;

            // The SDK can do step 4 for us, and it also corrects for lens distortion, which the
            // plain pinhole model ignores. Its Y axis points UP, so its Y has the opposite sign.
            CameraSpacePoint sdkPoint = sensor.CoordinateMapper.MapDepthPointToCameraSpace(
                new DepthSpacePoint { X = depthU, Y = depthV }, (ushort)zMm);

            text = $"{label} {zMeters:F2}m x{xMeters:+0.00;-0.00} y{yMeters:+0.00;-0.00}";
            reportLines.Add($"  {label,-12} {confidence:F2}  {text,-34}  samples {sampleDepthsMm.Count,5}"
                          + $"  | SDK: X{sdkPoint.X:+0.00;-0.00} Y{sdkPoint.Y:+0.00;-0.00} Z{sdkPoint.Z:F2}");
        }

        // --- Output: draw ----------------------------------------------
        Cv2.Rectangle(frame, new Point(boxLeft, boxTop), new Point(boxRight, boxBottom), Scalar.LimeGreen, 3);
        Cv2.Rectangle(frame, new Point(sampleLeft, sampleTop), new Point(sampleRight, sampleBottom), Scalar.Yellow, 1);
        Cv2.PutText(frame, text, new Point(boxLeft, Math.Max(boxTop - 10, 30)),
                    HersheyFonts.HersheySimplex, 1.0, Scalar.LimeGreen, 2);
    }
    double fuseMs = stepTimer.Elapsed.TotalMilliseconds;

    // ---------------------------------------------------------------------
    //  Output
    // ---------------------------------------------------------------------

    Cv2.ImShow(WindowName, frame);

    if (reportTimer.ElapsedMilliseconds >= ReportEveryMs)
    {
        double framesPerSecond = framesSinceReport * 1000.0 / reportTimer.ElapsedMilliseconds;
        Console.WriteLine($"{framesPerSecond:F1} fps | capture {captureMs:F0} ms, detect {detectMs:F0} ms, "
                        + $"register {registerMs:F0} ms, sample+deproject {fuseMs:F0} ms | {reportLines.Count} detections");
        foreach (string line in reportLines) Console.WriteLine(line);
        reportTimer.Restart();
        framesSinceReport = 0;
    }
}

reader.Dispose();
sensor.Close();
Cv2.DestroyAllWindows();

// Middle value of a list. For an even count, the average of the two middle values (like numpy's median).
static float Median(List<float> values)
{
    var sorted = new List<float>(values);
    sorted.Sort();
    int middle = sorted.Count / 2;
    if (sorted.Count % 2 == 1) return sorted[middle];
    return (sorted[middle - 1] + sorted[middle]) / 2;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Stand 1 m in front of a far wall and set BoxShrink = 1.0. Your distance jumps toward the
//     wall's, because most pixels in a loose box are background. That's why the centre is used.
//  2. Walk slowly sideways across the view at a fixed distance. The number should stay steady
//     even at the edges. If it jumps to the wall at the edges, registration is off (guide §7 test 3).
//  3. Check against a tape measure at 1, 2 and 3 m (guide §7 test 2). Expect a few cm.
//  4. Compare the "x y" from the pinhole math with the SDK's X Y in the console. They agree in
//     the middle of the view and drift apart toward the corners: that's the lens distortion the
//     SDK corrects and the pinhole model doesn't.
//  5. Set MaxDistanceMeters = 1.5 and walk toward the sensor: boxes appear only within reach.
//  6. Hold up a phone showing a photo of a person. YOLO says "person". The distance says 0.4 m.
//     How would you reject it? (Hint: real height = box height in pixels * Z / focal length; guide §10.)
