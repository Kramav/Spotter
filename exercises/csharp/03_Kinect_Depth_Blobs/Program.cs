// =====================================================================
//  03 · Kinect Depth Blobs  (no AI)
//
//  Find "something new in the room" from depth alone, and measure its
//  distance and real size in metres. This is the guide's no-AI rung (§5.6),
//  the classic Kinect people-counter:
//
//    1. Remember the empty room's depth once (the "background").
//    2. Every pixel now noticeably CLOSER than the background is foreground.
//    3. Clean up the speckles, then group touching foreground pixels into blobs.
//    4. Each blob: median depth = distance; pixel size * depth / focal length
//       = real width and height (§5.3).
//
//  It doesn't know WHAT anything is, but it ignores lighting and shadows and
//  works in a completely dark room, unlike example 02.
//
//  Window: grey depth, foreground in green, a box and label per blob.
//  Keys:   B  re-record the background (step out of view first)
//          Q or Esc  quit
// =====================================================================

using System.Diagnostics;
using Microsoft.Kinect;
using OpenCvSharp;

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// A pixel is foreground when it is at least this much closer than the background.
// Too small: sensor noise shows up as blobs. Too big: a hand flat on a table disappears.
const int ForegroundThresholdMm = 100;

// Blobs with fewer pixels than this are ignored (a person at 3 m covers a few thousand).
const int MinBlobPixels = 400;

// Size of the "opening" filter that removes isolated foreground speckles, in pixels.
// 1 = no cleanup. Try 1, then 5, then 11 and watch the green.
const int CleanupSizePixels = 5;

// Seconds to step out of view before the background is recorded at startup.
const int SecondsBeforeBackground = 3;

// The grey scale for the picture, as in example 01.
const int NearMm = 500;
const int FarMm = 4500;

// How often the console report prints, in milliseconds.
const int ReportEveryMs = 1000;

// ---------------------------------------------------------------------
//  Setup
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

int depthWidth = sensor.DepthFrameSource.FrameDescription.Width;     // 512
int depthHeight = sensor.DepthFrameSource.FrameDescription.Height;   // 424
DepthFrameReader depthReader = sensor.DepthFrameSource.OpenReader();

ushort[] depthMm = new ushort[depthWidth * depthHeight];        // this frame, millimetres, 0 = no reading
ushort[] backgroundMm = new ushort[depthWidth * depthHeight];   // the remembered empty room
byte[] foreground = new byte[depthWidth * depthHeight];         // 255 = foreground, 0 = not

// These Mats share memory with the arrays above: change the array, the image changes.
Mat foregroundMask = Mat.FromPixelData(depthHeight, depthWidth, MatType.CV_8UC1, foreground);
Mat cleanedMask = new Mat();
Mat picture = new Mat(depthHeight, depthWidth, MatType.CV_8UC3);
Mat cleanupShape = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(CleanupSizePixels, CleanupSizePixels));

// Depth intrinsics for the real-size math. Zero until the first frames arrive (see example 02).
CameraIntrinsics depthIntrinsics = new CameraIntrinsics();

bool backgroundRecorded = false;
var backgroundCountdown = Stopwatch.StartNew();
Console.WriteLine($"Recording the background in {SecondsBeforeBackground} s: step out of view.");

const string WindowName = "03 Kinect depth blobs  (B = new background, Q quits)";
Cv2.NamedWindow(WindowName, WindowFlags.Normal);
Cv2.ResizeWindow(WindowName, depthWidth * 2, depthHeight * 2);

var reportTimer = Stopwatch.StartNew();

// ---------------------------------------------------------------------
//  Main operation
// ---------------------------------------------------------------------

while (true)
{
    int key = Cv2.WaitKey(1);
    if (key == 'q' || key == 27) break;   // 27 = Esc
    if (key == 'b')
    {
        backgroundRecorded = false;
        backgroundCountdown.Restart();
        Console.WriteLine($"Recording the background in {SecondsBeforeBackground} s: step out of view.");
    }

    using (DepthFrame frame = depthReader.AcquireLatestFrame())
    {
        if (frame == null) continue;
        frame.CopyFrameDataToArray(depthMm);
    }

    if (depthIntrinsics.FocalLengthX == 0)
    {
        depthIntrinsics = sensor.CoordinateMapper.GetDepthCameraIntrinsics();
    }

    // --- Step 1: remember the empty room --------------------------------
    if (!backgroundRecorded && backgroundCountdown.ElapsedMilliseconds >= SecondsBeforeBackground * 1000)
    {
        Array.Copy(depthMm, backgroundMm, depthMm.Length);
        backgroundRecorded = true;
        Console.WriteLine("Background recorded. Walk in!");
    }

    // --- Step 2: foreground = closer than the background ----------------
    // Same rule as the guide:  (background - depth) > threshold  AND  depth has a reading.
    // Where the background had no reading (0), background - depth is negative, so those
    // pixels never count as foreground. Experiment idea 3 is about that.
    for (int y = 0; y < depthHeight; y++)
    {
        for (int x = 0; x < depthWidth; x++)
        {
            int i = y * depthWidth + x;
            int closerByMm = backgroundMm[i] - depthMm[i];
            bool isForeground = backgroundRecorded && depthMm[i] > 0 && closerByMm > ForegroundThresholdMm;
            foreground[i] = isForeground ? (byte)255 : (byte)0;

            // Draw this pixel: green if foreground, otherwise grey by distance, black for no reading.
            byte brightness = 0;
            if (depthMm[i] > 0)
            {
                double fraction = Math.Clamp((double)(depthMm[i] - NearMm) / (FarMm - NearMm), 0.0, 1.0);
                brightness = (byte)(255 * (1.0 - fraction));
            }
            if (isForeground)
                picture.Set(y, x, new Vec3b(0, brightness, 0));   // B, G, R: green, still brighter when nearer
            else
                picture.Set(y, x, new Vec3b(brightness, brightness, brightness));
        }
    }

    // --- Step 3: clean up and group into blobs ---------------------------
    // "Opening" = shrink every foreground region a little, then grow it back. Specks smaller
    // than the shape vanish in the shrink and never come back. Big regions keep their outline.
    Cv2.MorphologyEx(foregroundMask, cleanedMask, MorphTypes.Open, cleanupShape);

    // Connected components: label every group of touching foreground pixels 1, 2, 3...
    // Blob 0 is always the background (all the zero pixels), so it is skipped below.
    ConnectedComponents components = Cv2.ConnectedComponentsEx(cleanedMask, PixelConnectivity.Connectivity8);

    // --- Step 4: measure each blob ---------------------------------------
    var reportLines = new List<string>();
    foreach (ConnectedComponents.Blob blob in components.Blobs)
    {
        if (blob.Label == 0) continue;
        if (blob.Area < MinBlobPixels) continue;

        // Median depth of the pixels that belong to THIS blob (not the whole rectangle around it).
        var blobDepthsMm = new List<float>();
        for (int y = blob.Top; y < blob.Top + blob.Height; y++)
        {
            for (int x = blob.Left; x < blob.Left + blob.Width; x++)
            {
                if (components.Labels[y, x] == blob.Label && depthMm[y * depthWidth + x] > 0)
                {
                    blobDepthsMm.Add(depthMm[y * depthWidth + x]);
                }
            }
        }
        if (blobDepthsMm.Count == 0 || depthIntrinsics.FocalLengthX == 0) continue;
        double zMeters = Median(blobDepthsMm) / 1000.0;

        // Real size (§5.3): an object w pixels wide at depth Z is about  w * Z / fx  wide.
        double widthMeters = blob.Width * zMeters / depthIntrinsics.FocalLengthX;
        double heightMeters = blob.Height * zMeters / depthIntrinsics.FocalLengthY;

        // Where the blob's centre is: pinhole model on its centroid (x right, y down).
        double xMeters = (blob.Centroid.X - depthIntrinsics.PrincipalPointX) * zMeters / depthIntrinsics.FocalLengthX;
        double yMeters = (blob.Centroid.Y - depthIntrinsics.PrincipalPointY) * zMeters / depthIntrinsics.FocalLengthY;

        string text = $"{zMeters:F2}m  {widthMeters:F2} x {heightMeters:F2}m";
        reportLines.Add($"  blob {blob.Label,2}: {blob.Area,6} px  {text,-22}  centre x{xMeters:+0.00;-0.00} y{yMeters:+0.00;-0.00}");

        // ---------------------------------------------------------------------
        //  Output: draw
        // ---------------------------------------------------------------------
        Cv2.Rectangle(picture, blob.Rect, Scalar.Yellow, 1);
        Cv2.PutText(picture, text, new Point(blob.Left, Math.Max(blob.Top - 4, 12)),
                    HersheyFonts.HersheySimplex, 0.4, Scalar.Yellow, 1);
    }

    // ---------------------------------------------------------------------
    //  Output
    // ---------------------------------------------------------------------
    if (!backgroundRecorded)
    {
        double secondsLeft = SecondsBeforeBackground - backgroundCountdown.Elapsed.TotalSeconds;
        Cv2.PutText(picture, $"background in {secondsLeft:F0} s", new Point(10, 20),
                    HersheyFonts.HersheySimplex, 0.6, Scalar.Red, 2);
    }
    Cv2.ImShow(WindowName, picture);

    if (backgroundRecorded && reportTimer.ElapsedMilliseconds >= ReportEveryMs)
    {
        int rawForegroundPixels = Cv2.CountNonZero(foregroundMask);
        int cleanedForegroundPixels = Cv2.CountNonZero(cleanedMask);
        Console.WriteLine($"foreground pixels: {rawForegroundPixels} raw, {cleanedForegroundPixels} after cleanup"
                        + $" | {components.Blobs.Count - 1} groups, {reportLines.Count} big enough");
        foreach (string line in reportLines) Console.WriteLine(line);
        reportTimer.Restart();
    }
}

depthReader.Dispose();
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
//  1. Stand in view and read your height off the label. Expect roughly your real height
//     (guide §7 test 5). Now crouch, raise an arm, hold a chair: the box measures the outline.
//  2. Turn the lights off. Everything still works. Then try example 02 in the dark.
//  3. Point the Kinect at an open doorway or a window, record the background, and walk into
//     the region that was black (no reading). You don't show up there. Why? (Step 2's rule.)
//     Try treating background 0 as "very far" (8000 mm) and see what changes.
//  4. Set ForegroundThresholdMm = 20. The floor and walls start to flicker green: that's
//     the sensor's noise. Find the smallest value that stays clean at your distances.
//  5. Set CleanupSizePixels = 1 and compare the "raw" and "after cleanup" pixel counts.
//  6. Put a box on the floor, press B, then remove it. Nothing appears where it was: only
//     "closer than before" counts. What would detect objects that were taken AWAY?
