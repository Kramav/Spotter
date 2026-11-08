# Spotter examples, on the `spotter` library

This folder is the hand-off point for anyone who picks up the code. Each program is one
`main.cpp`. Each one teaches a single idea and builds on the ones before it, and all of them use
the shared library in [`../spotter/`](../spotter/). The originals they were re-written from are in
[`../legacy/`](../legacy/). Those are kept unchanged, each one written out in full.

The target system: a Kinect v2 (RGB-D) on a Raspberry Pi (Linux, CPU only). It detects objects in
colour, measures them with depth, and builds a 3D map of a room. Development is on Windows for now,
and everything must also build on Linux.

## Build, run, test

From `exercises/cpp/`. On Windows, use the "x64 Native Tools" prompt, or call `vcvars64.bat` first:

```
cmake --preset default
cmake --build --preset default
ctest --test-dir build                 # the library's self-checks: synthetic data, no Kinect
build\examples\00b_Pinhole_Camera.exe  # a tour program (no Kinect)
build\examples\02_Detect_Distance.exe  # an example (Kinect + yolo26n.onnx)
```

vcpkg installs libfreenect2 (our port in `ports/`) and OpenCV the first time.

## How a program reads

Every `main.cpp` has the same parts:

1. **Header comment:** what it does, the keys, and what you'll see.
2. **Settings:** constants to change and re-run.
3. **The new idea:** the functions this example exists to teach.
4. **Output:** drawing and the console report.
5. **`main()`:** 15–40 lines. It creates the library objects, then loops over the named steps.
6. **Experiment ideas:** things to try.

```cpp
int main()
{
    spotter::YoloDetector yolo(YoloSettings());
    spotter::KinectSource kinect(KinectSettings());
    if (!yolo.IsReady() || !kinect.IsOpen()) return 1;
    ...
    while (kinect.Next(frame))
    {
        std::vector<spotter::Detection> detections = yolo.Detect(frame.colour);
        ...
    }
}
```

There are no globals. A program's state lives in plain structs declared in that file.

## The library

| Header | What it gives you | Tour |
|---|---|---|
| `frame.hpp` | `RgbdFrame` (colour, depth in mm, registered colour, colourIndex, cameras); `PinholeCamera` (`Unproject`, `Project`, `Cloud`); `Unmirror` | 00b |
| `source_freenect2.hpp` | `KinectSource`: opens the Kinect v2 and fills an `RgbdFrame` per `Next()`; it stops the device when destroyed | 00a |
| `depth.hpp` | `BoxDepthSampler` (a box's median distance, fast); `DepthPicture` (depth as grey) | 00c |
| `detection.hpp` | `Detection`; `ClassList` (class names from a text file) | 00d |
| `detector_yolo_opencvdnn.hpp` | `YoloDetector`; `DecodeYolo` (both output layouts) | 00d |
| `stats.hpp` | `Median`, `Percentile` | |
| `view.hpp` | `FpsMeter`, `Every` (a periodic timer), `IsQuitKey`; `OrbitView` + `PointRenderer` (a small 3D viewer) | |
| `mesh.hpp` | `SavePly` | |

Still to come, each with its own check and tour:
- `track` (Kalman tracker), `geometry` (Pose, planes) and `walls`;
- `register` (ICP, feature pose) and `features_orb`;
- `room` / `room_folder` / `room_scan` (TSDF, the clean model, scan folders).

**External dependencies show in the file name.** Core headers use only OpenCV core types. Anything
tied to a device, a model or a runtime is an *adapter* named `<role>_<dependency>`. To swap one,
add a sibling file:
- **Another camera:** write `source_<camera>.cpp` with the same `Options` / `IsOpen` / `Next` and
  fill the same `RgbdFrame`. Nothing downstream changes.
- **Another YOLO model, or your own trained one:** change only `YoloDetector::Options`:
  - `modelPath`;
  - `inputSize`;
  - `namesPath` (one class name per line, in training order);
  - `layout`: `EndToEnd` for YOLO26 exported with `nms=False`, or `Classic` for older exports.

  `check_yolo_decode` tests both layouts.
- **Another runtime** (NCNN on the Pi): write `detector_yolo_ncnn.cpp` with the same `Detect()`.
  There is no virtual base class yet; add one only when a program has to choose at run time.

## Conventions

- Units:
  - depth pictures are in **millimetres** (`depthMm`, 0 = no reading);
  - everything in 3D is in **metres**;
  - camera coordinates: X right, Y down, Z straight out of the camera.
- **The picture is the right way round.** The Kinect sends mirror images; `KinectSource` flips
  every frame (`Unmirror`), so the person's right hand appears on the right. The old programs in
  `legacy/` are still mirrored.
- Style: PascalCase functions, Allman braces, 4-space indents. Comments explain *why*. A
  `ponytail:` comment marks a deliberate shortcut and names its limit.
- Privacy: never save or commit camera frames or pictures of people. `*.jpg`, `*.ply` and room
  scan folders are gitignored. Checks use synthetic data only.

## Ladders: small steps up to each example

Each example's folder starts with a few small programs, `<example>_s1_…`, `_s2_…`, and so on.
Each one is 30–60 lines and adds exactly one new thing to the step before it. The header says
what is new and which program comes next. The last step leads into the example itself (the
folder's own `main.cpp`). Read a folder top to bottom as a lesson:

| Example | Steps |
|---|---|
| `01_Depth_Peek` | s1 show the depth picture → s2 read one pixel → s3 a crosshair you can move → **01** |
| `02_Detect_Distance` | s1 show the colour camera → s2 find objects with YOLO → s3 where each depth pixel lands in colour → s4 how far away each box is → **02** |
| `03_Depth_Blobs` | s1 remember the room → s2 which pixels are new → s3 group them into blobs → **03** |

More ladders are added as each example is ported.

## The programs

| Program | Idea | Kinect | Status |
|---|---|---|---|
| `00_tour/00a_Kinect_Source` | the camera and the frame; the mirror check | yes | built, not run on the sensor |
| `00_tour/00b_Pinhole_Camera` | pixel ↔ 3D, two cameras, unmirror | no | runs (synthetic) |
| `00_tour/00c_Box_Depth` | a box's distance, slow vs fast | no | runs (synthetic) |
| `00_tour/00d_Yolo_Detector` | detection, class names, swapping the model | yes / a BMP | built, not run |
| `01_Depth_Peek` | what a depth picture holds | yes | built, not run |
| `02_Detect_Distance` | YOLO + depth: the four steps of late fusion | yes | built, not run |
| `03_Depth_Blobs` | foreground from depth alone, real sizes | yes | built, not run |
| `07_Capture_Average` | averaging frames, measured noise, far range | yes | built, not run |
| `07c_Depth_Background` | a self-healing background memory | yes | built, not run |
| `08_Point_Cloud` | every pixel as a coloured 3D point | yes | built, not run |

Still to port: 04, 05, 05b, 05c, 06, 06b, 06c, 07d, 08b, 09, 09b, 09c, 10, 11, 11b, 13e, 13z and
the six exhibition demos. 13–13d are left out: 13e contains them. The flowcharts in
[`../EXAMPLES.md`](../EXAMPLES.md) still describe each idea.

## What is tested, and how

- `ctest` runs one self-check per module, on synthetic data:
  - `check_frame`: `Unmirror` and `PinholeCamera` against libfreenect2's own `Registration`, with
    no device. Every 3D point equals `getPointXYZ` with X negated.
  - `check_yolo_decode`: both model output layouts, the letterbox, and the class-name lookup.
  - `check_box_depth`: `BoxDepthSampler` gives exactly the same answers as the old per-box loop, on
    4,000 random boxes, about 2.5× faster.
- The tour programs without a Kinect run to completion and print what they show.
- **Not yet confirmed on the sensor:** the programs above that need a Kinect. The first thing to
  check: run `00a_Kinect_Source` and raise your right hand. It must show on the right.
