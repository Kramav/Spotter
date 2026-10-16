# Kinect v2 examples (C#)

Three small programs that walk through [kinect_rgbd_fusion_guide.md](../../kinect_rgbd_fusion_guide.md): what the depth camera measures, YOLO boxes labelled with distance, and depth-only detection.

Each folder holds one `Program.cs` that you read from top to bottom, plus a project file. Nothing is shared between them. Settings sit at the top of each file and experiment ideas at the bottom.

## Why C# and not the guide's Python

The guide's scripts are all Python (`freenect`, `pylibfreenect2`, Ultralytics). These examples use the other Kinect v2 route the guide lists in §4, **Microsoft's Kinect for Windows SDK 2.0**, which is C#-first. YOLO runs from an ONNX export of the same `yolo26n.pt` through OpenCV's DNN module.

What carries over from the guide unchanged is the fusion method in §5: register depth to colour, take the median depth of the box centre, then deproject with the camera's intrinsics. What changes:

| Guide (Python, libfreenect2) | Here (C#, Microsoft SDK) |
|---|---|
| `Registration.apply(..., bigdepth)` | `CoordinateMapper.MapDepthFrameToColorSpace` |
| `getColorCameraParams()` intrinsics | `CoordinateMapper.GetDepthCameraIntrinsics()` (math done in the depth image) |
| `model(frame)` (Ultralytics) | letterbox → `CvDnn.BlobFromImage` → `Net.Forward()` |
| Zadig / UsbDk driver swap | not needed: uses the stock Microsoft driver |

## Prerequisites

1. **Kinect v2** (Xbox One or Kinect for Windows v2) with the **Kinect Adapter for Windows**, on a **USB 3.0** port (guide §2).
2. **Kinect for Windows SDK 2.0** ([download](https://www.microsoft.com/en-us/download/details.aspx?id=44561)). It installs the driver and the real `Microsoft.Kinect.dll` into the 64-bit Global Assembly Cache (GAC), which the project files point at.
   Check it with `Tools\ConfigurationVerifier` in the SDK folder, or with `bin\DepthBasics-WPF.exe`.
3. **.NET 10 SDK** (`dotnet --list-sdks`).
4. **Example 02 only:** `yolo26n.onnx` in the repo root. It has already been exported on this machine. To redo it, run this once from the repo root using the existing `.venv`:
   ```
   .venv\Scripts\yolo export model=yolo26n.pt format=onnx nms=False
   ```
   `nms=False` makes YOLO26 export its end-to-end head. The output is `1 x 300 x 6` and already free of duplicate boxes, so no NMS code is needed on the C# side.

NuGet restores OpenCvSharp4 (4.13.0.20260627) automatically on the first build.

## Running

```
cd exercises\csharp\01_Kinect_Depth_Peek
dotnet run
```

Do the same with `02_Kinect_Detect_Distance` and `03_Kinect_Depth_Blobs`. Press **Q** or **Esc** in the window to quit. Every example prints a status line to the console once per second.

---

## 01 · Kinect Depth Peek

**Teaches:** opening the sensor, reading depth frames, and what the numbers mean. Depth is a 512 × 424 grid of `ushort` millimetres, and `0` means no reading. You also see where the sensor fails: too close, black, shiny or glass surfaces.

**Observe:** a grey depth picture (near is bright), with **red** wherever there's no reading. **W A S D** moves a crosshair, and the console shows the distance under it.

**Change first:** `NearMm` / `FarMm` (try 1000 / 2000), then move your hand closer than 50 cm.

Real output (this machine, 2026-10-15, nobody moving):
```
Kinect found after 438 ms.
Depth image:          512 x 424 pixels
SDK's reliable range: 500 .. 4500 mm
30.5 fps | crosshair (256,212) = 2768 mm    | nearest 500 mm, farthest 7992 mm | no reading: 9.0%
29.8 fps | crosshair (256,212) = 2768 mm    | nearest 500 mm, farthest 7999 mm | no reading: 9.1%
```
Note that "farthest" reaches about **8 m** even though the SDK calls 4.5 m its reliable limit. Values past 4.5 m arrive, but treat them as noisy.

## 02 · Kinect Detect + Distance

**Teaches:** the guide's main build (§1, §5), as four visible steps: **detect** (YOLO on 1920 × 1080 colour), **register** (where each depth pixel lands in colour), **sample** (median depth of the box's central half), and **deproject** (pinhole math). It also shows the letterbox pre-processing that Ultralytics normally hides. Next to each result, the console prints the SDK's own `MapDepthPointToCameraSpace` answer as a check on the hand-done math.

**Observe:** a thick green box per detection labelled like `person 0.71m x-0.11 y+0.09`. A thin yellow box inside it shows the area that depth was sampled from.

**Change first:** `BoxShrink` (set it to 1.0 and stand in front of a far wall to see why the centre is used), then `MaxDistanceMeters = 1.5`, then `MinConfidence`.

Real output (this machine, 2026-10-15):
```
Letterbox: scale 0.3333, image 640 x 360, padding 0 px left, 140 px top
Depth intrinsics: fx 365.1  fy 365.1  cx 256.9  cy 209.4  (pixels)
5.0 fps | capture 4 ms, detect 154 ms, register 0 ms, sample+deproject 1 ms | 2 detections
  keyboard     0.52  keyboard 0.64m x-0.44 y+0.24        samples    32  | SDK: X-0.46 Y-0.25 Z0.64
  person       0.46  person 0.53m x-0.07 y+0.17          samples  5424  | SDK: X-0.07 Y-0.17 Z0.53
```
- The pinhole X agrees with the SDK's X to within 1–2 cm toward the edge. The difference is lens distortion, which the SDK corrects and the plain pinhole model doesn't.
- The SDK's Y has the opposite sign because its Y axis points up.
- About **5 fps**. YOLO on the CPU through OpenCV DNN is the slow step (about 155 ms). That's expected, since these examples aren't optimised.

## 03 · Kinect Depth Blobs (no AI)

**Teaches:** the guide's no-AI rung (§5.6). It remembers the empty room, marks anything more than 10 cm closer as foreground, cleans that up with a morphological opening, groups it with connected components, and gives each blob a distance and a **real width × height** (`pixels × Z / focal length`, §5.3). It works in the dark.

**Observe:** step out of view during the 3-second countdown, then walk in. Foreground turns green, and each blob gets a label like `2.10m  0.55 x 1.72m`. Press **B** to record a new background.

**Change first:** `ForegroundThresholdMm` (try 20 and watch the noise appear), `CleanupSizePixels` (1 vs 11), and `MinBlobPixels`.

Real output (this machine, 2026-10-15, hands and arms moving about 0.5–1 m from the sensor; sizes not checked against a tape):
```
Background recorded. Walk in!
foreground pixels: 0 raw, 0 after cleanup | 0 groups, 0 big enough
foreground pixels: 7959 raw, 6824 after cleanup | 6 groups, 3 big enough
  blob  1:   2117 px  0.51m  0.10 x 0.08m     centre x+0.17 y-0.12
  blob  2:   2343 px  0.69m  0.12 x 0.11m     centre x+0.20 y+0.18
  blob  3:   2072 px  0.86m  0.08 x 0.27m     centre x-0.29 y+0.36
```

---

## What isn't here, and why

The guide's Build C add-ons rely on Python-only tooling, and porting them would mean writing them from scratch, so they were left out:
- **mask depth** needs YOLO segmentation output decoding;
- **tracking and speed** needs Ultralytics' `model.track`, which has no C# counterpart;
- **point cloud** needs Open3D.

Microsoft's SDK has its own **body tracking** (25 joints, 6 people; guide §4). It's the shortest path if people's 3-D positions are all you need. Ask if you want it as example 04.

## Known limits

- Tested only on this machine: Windows 10, Kinect v2 on the stock driver, SDK 2.0 (v2.0_1409), .NET 10.0.401, OpenCvSharp 4.13. Not tested on Windows 11 or with only the Kinect Runtime installed (no SDK).
- Distances were **not** checked against a tape measure (guide §7 tests 2–4). The only accuracy check is that the pinhole math matches the SDK's mapper.
- The detector was compared with Ultralytics on one still image (`bus.jpg`): the same 5 objects and boxes within a few pixels. Confidences were within 0.05, except for one half-hidden person (0.53 vs 0.66). Ultralytics pads that image less, so the model sees a slightly different input.
- The windows themselves weren't inspected (no screenshots). Only the console output was read.
