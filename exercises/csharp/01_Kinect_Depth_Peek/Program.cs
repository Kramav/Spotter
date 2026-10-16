// =====================================================================
//  01 · Kinect Depth Peek
//
//  Read the Kinect v2 depth camera and look at what it actually measures.
//
//  What you'll see:
//    - a grey picture of the room: near = bright, far = dark
//    - RED pixels where the sensor got NO reading
//      (closer than ~0.5 m, too far, black / shiny / glass surfaces)
//    - a green crosshair, with the distance under it in millimetres
//
//  Keys:   W A S D   move the crosshair
//          Q or Esc  quit
//
//  Guide sections: §2 (Kinect v2 specs), §5.5 (what depth gets wrong).
// =====================================================================

using System.Diagnostics;
using Microsoft.Kinect;
using OpenCvSharp;

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// The grey scale. Depth at NearMm is drawn white, depth at FarMm (or beyond) is drawn black.
// Try 1000 / 2000 to stretch the contrast over one metre of the room.
const int NearMm = 500;
const int FarMm = 4500;

// How far one key press moves the crosshair, in depth pixels.
const int CrosshairStepPixels = 10;

// How often the console report prints, in milliseconds.
const int ReportEveryMs = 1000;

// ---------------------------------------------------------------------
//  Setup
// ---------------------------------------------------------------------

// GetDefault() always returns an object, even with no Kinect plugged in.
// The real "is it there?" check is IsAvailable, which turns true a second or two after Open().
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
Console.WriteLine($"Kinect found after {waitTimer.ElapsedMilliseconds} ms.");

// The depth camera's fixed properties. These never change, so read them once.
FrameDescription depthDescription = sensor.DepthFrameSource.FrameDescription;
int depthWidth = depthDescription.Width;     // 512
int depthHeight = depthDescription.Height;   // 424
Console.WriteLine($"Depth image:          {depthWidth} x {depthHeight} pixels");
Console.WriteLine($"SDK's reliable range: {sensor.DepthFrameSource.DepthMinReliableDistance}"
                + $" .. {sensor.DepthFrameSource.DepthMaxReliableDistance} mm");

// A reader hands us depth frames. One reader per stream we care about.
DepthFrameReader depthReader = sensor.DepthFrameSource.OpenReader();

// One ushort per pixel = distance in millimetres. 0 means "no reading".
// Row by row: pixel (x, y) lives at index  y * depthWidth + x.
ushort[] depthMm = new ushort[depthWidth * depthHeight];

// The picture we draw into: 3 bytes (blue, green, red) per pixel.
Mat picture = new Mat(depthHeight, depthWidth, MatType.CV_8UC3);

const string WindowName = "01 Kinect depth peek  (WASD moves, Q quits)";
Cv2.NamedWindow(WindowName, WindowFlags.Normal);   // Normal = you can drag the window bigger
Cv2.ResizeWindow(WindowName, depthWidth * 2, depthHeight * 2);

int crosshairX = depthWidth / 2;
int crosshairY = depthHeight / 2;

var reportTimer = Stopwatch.StartNew();
int framesSinceReport = 0;

// ---------------------------------------------------------------------
//  Main operation
// ---------------------------------------------------------------------

while (true)
{
    // AcquireLatestFrame() returns null when no NEW frame has arrived since the last call.
    // The depth camera runs at 30 frames per second and this loop runs faster, so null is normal.
    // The frame must be disposed quickly (the 'using'), or the Kinect stops sending new ones.
    bool gotNewFrame = false;
    using (DepthFrame frame = depthReader.AcquireLatestFrame())
    {
        if (frame != null)
        {
            frame.CopyFrameDataToArray(depthMm);
            gotNewFrame = true;
            framesSinceReport++;
        }
    }

    if (gotNewFrame)
    {
        // Turn every depth value into a colour, and collect a few statistics on the way.
        int noReadingCount = 0;
        int nearestMm = int.MaxValue;
        int farthestMm = 0;

        for (int y = 0; y < depthHeight; y++)
        {
            for (int x = 0; x < depthWidth; x++)
            {
                ushort mm = depthMm[y * depthWidth + x];

                if (mm == 0)
                {
                    noReadingCount++;
                    picture.Set(y, x, new Vec3b(0, 0, 255));   // OpenCV order is B, G, R: this is red
                    continue;
                }

                nearestMm = Math.Min(nearestMm, mm);
                farthestMm = Math.Max(farthestMm, mm);

                // 0.0 at NearMm, 1.0 at FarMm, clamped so values outside the range don't wrap around.
                double fraction = (double)(mm - NearMm) / (FarMm - NearMm);
                fraction = Math.Clamp(fraction, 0.0, 1.0);
                byte brightness = (byte)(255 * (1.0 - fraction));
                picture.Set(y, x, new Vec3b(brightness, brightness, brightness));
            }
        }

        // -----------------------------------------------------------------
        //  Output
        // -----------------------------------------------------------------

        ushort crosshairMm = depthMm[crosshairY * depthWidth + crosshairX];
        string crosshairText = crosshairMm == 0 ? "no reading" : $"{crosshairMm} mm";

        Cv2.DrawMarker(picture, new Point(crosshairX, crosshairY), Scalar.LimeGreen, MarkerTypes.Cross, 20, 1);
        Cv2.PutText(picture, crosshairText, new Point(crosshairX + 8, crosshairY - 8),
                    HersheyFonts.HersheySimplex, 0.5, Scalar.LimeGreen, 1);
        Cv2.ImShow(WindowName, picture);

        if (reportTimer.ElapsedMilliseconds >= ReportEveryMs)
        {
            double framesPerSecond = framesSinceReport * 1000.0 / reportTimer.ElapsedMilliseconds;
            double noReadingPercent = 100.0 * noReadingCount / depthMm.Length;
            Console.WriteLine($"{framesPerSecond,4:F1} fps | crosshair ({crosshairX},{crosshairY}) = {crosshairText,-10}"
                            + $" | nearest {nearestMm} mm, farthest {farthestMm} mm | no reading: {noReadingPercent:F1}%");
            reportTimer.Restart();
            framesSinceReport = 0;
        }
    }

    // WaitKey(1) also lets OpenCV redraw the window, so it must run every pass of the loop.
    int key = Cv2.WaitKey(1);
    if (key == 'q' || key == 27) break;   // 27 = Esc
    if (key == 'w') crosshairY -= CrosshairStepPixels;
    if (key == 's') crosshairY += CrosshairStepPixels;
    if (key == 'a') crosshairX -= CrosshairStepPixels;
    if (key == 'd') crosshairX += CrosshairStepPixels;
    crosshairX = Math.Clamp(crosshairX, 0, depthWidth - 1);
    crosshairY = Math.Clamp(crosshairY, 0, depthHeight - 1);
}

depthReader.Dispose();
sensor.Close();
Cv2.DestroyAllWindows();

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Put your hand closer and closer to the sensor. Below ~50 cm it turns red, not brighter:
//     inside the minimum range you get holes, not small numbers (guide §5.5).
//  2. Aim the crosshair at a black monitor, a window or a mirror. Watch "no reading" climb.
//  3. Turn the lights off. Depth keeps working: the Kinect lights the room with its own infrared.
//  4. Set NearMm = 1000 and FarMm = 2000 to see fine depth steps (and the noise) on one object.
//  5. Hold the crosshair on a wall and watch the console: how much does one pixel jitter
//     from frame to frame at 1 m? At 3 m?
//  6. Replace the grey with false colour: after the loop, try
//     Cv2.ApplyColorMap(picture, picture, ColormapTypes.Jet) before ImShow.
