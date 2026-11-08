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
| `frame.hpp` | `RgbdFrame` (colour, depth in mm, registered colour, colourIndex, cameras); `PinholeCamera` (`Unproject`, `Project`, `Cloud`); `Unmirror`; `FrameTimer` | 00b |
| `source_freenect2.hpp` | `KinectSource`: opens the Kinect v2 and fills an `RgbdFrame` per `Next()`; it stops the device when destroyed | 00a |
| `depth.hpp` | `BoxDepthSampler` (a box's median distance, fast); `DepthPicture` (depth as grey); `AdaptiveBackground` (a self-healing depth memory, 07c); `KeepBigBlobs`; `DepthAverager` (07) | 00c |
| `detection.hpp` | `Detection`; `ClassList` (class names from a text file) | 00d |
| `detector_yolo_opencvdnn.hpp` | `YoloDetector`; `DecodeYolo` (both output layouts); `AsyncDetector` (YOLO on its own thread, 05c) | 00d |
| `track.hpp` | `Tracker` (IDs, two confidence thresholds, coasting, fresh distances every frame); `BoxFilter` (a Kalman filter per box); `Overlap`, `ColourForId` | 00e |
| `stats.hpp` | `Median`, `Percentile` | |
| `view.hpp` | `FpsMeter`, `Every` (a periodic timer), `IsQuitKey`; `OrbitView` + `PointRenderer` (a small 3D viewer) | 00g |
| `geometry.hpp` | `Pose` (turn + shift, `Then`, `Degrees`), `Kabsch` (best pose for known pairs), `TurnAbout`; `Plane`, `PlaneThrough`, `FitPlane`; `PointGrid` + `ThinCloud`; `FindPlanes` (repeated RANSAC, biggest connected piece; upright / level / any) | 00f |
| `register.hpp` | `SampleSurface` (sparse points + normals: one rule for every scan); `Icp` (13e's point-to-plane ICP; directions the shape can't pin keep the start's value) → `IcpResult` | 00f |
| `walls.hpp` | `Wall`, `FindWalls` (06b: walls as lines on the floor plan) | |
| `radar.hpp` | `RadarView` (06's bird's-eye map), `TrackTrails` (where each tracked object has been) | |
| `mesh.hpp` | `Mesh` (named groups), `MeshFromGrid`, `AddQuad`, `AddBox`, `TurnToYUp`, `SaveObj`, `SavePly` | 00g |

Still to come, each with its own check and tour:
- the wall filter (Demo 3);
- `features_orb` (ORB built once, matching) and `register`'s feature pose and descriptor index,
  when 13e needs them (11b keeps its own copy inline, as its lesson);
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
| `04_Track_Smooth` | s1 how much two boxes overlap → s2 keep an ID → s3 remember an object YOLO missed → **04** |
| `05_Kalman` | s1 a Kalman filter follows a noisy dot (no Kinect) → s2 coasting when measurements stop (no Kinect) → s3 a filter per box → **05** |
| `05b_Two_Thresholds` | s1 sure and unsure detections → s2 a coasting object slows down (no Kinect) → **05b** |
| `05c_Background_Detect` | s1 what YOLO costs → s2 YOLO on its own thread → **05c** |
| `06_Radar` | s1 where on the floor each object is → s2 draw them from above → **06** |
| `07d_Remove_Person` | s1 which depth pixels are inside a person box → s2 remember what the room looks like → s3 paint the person out → **07d** |
| `06b_Radar_Walls` | s1 the biggest upright plane → s2 several planes, each as one piece → s3 walls seen from above → **06b** |
| `06c_Wall_Filter` | s1 walls wobble → s2 an extended Kalman filter for one wall (no Kinect) → **06c** |
| `08b_Object_Mesh` | s1 click a box → s2 cut the object out of the depth → s3 its front as triangles, saved → **08b** |
| `09_Find_Floor` | s1 three points make a plane → s2 RANSAC: try many, keep the best → s3 only level planes → **09** |
| `09b_Room_Mesh` | s1 a still, averaged scan → s2 every square becomes two triangles → **09b** |
| `09c_Room_Model` | s1 every big flat surface → s2 floor, ceiling, walls → s3 stand the room up straight → **09c** |
| `10_Two_Views` | s1 two clouds at once → s2 move B into A's coordinates → **10** |
| `11_ICP` | s1 the nearest point, fast: a k-d tree (no Kinect) → s2 the best move for known pairs (no Kinect) → s3 pair up, fit, repeat (no Kinect) → **11** |
| `11b_Feature_Pose` | s1 ORB features → s2 match them between two views → s3 a camera move from the matches (RANSAC) → **11b** |

More ladders are added as each example is ported.

## The programs

| Program | Idea | Kinect | Status |
|---|---|---|---|
| `00_tour/00a_Kinect_Source` | the camera and the frame; the mirror check | yes | built, not run on the sensor |
| `00_tour/00b_Pinhole_Camera` | pixel ↔ 3D, two cameras, unmirror | no | runs (synthetic) |
| `00_tour/00c_Box_Depth` | a box's distance, slow vs fast | no | runs (synthetic) |
| `00_tour/00d_Yolo_Detector` | detection, class names, swapping the model | yes / a BMP | built, not run |
| `00_tour/00e_Tracker` | IDs, coasting, two thresholds on made-up boxes | no | runs (synthetic) |
| `00_tour/00f_Pose_Planes_Icp` | a pose, Kabsch, the planes of a raycast room, ICP between two views, a bare wall's weak directions | no | runs (synthetic) |
| `00_tour/00g_Mesh_And_Viewer` | a made-up depth picture → mesh → `tour_00g.obj`; the viewer's maths by hand | no (a window) | runs (synthetic) |
| `01_Depth_Peek` | what a depth picture holds | yes | built, not run |
| `02_Detect_Distance` | YOLO + depth: the four steps of late fusion | yes | built, not run |
| `03_Depth_Blobs` | foreground from depth alone, real sizes | yes | built, not run |
| `04_Track_Smooth` | IDs by box overlap, a smoothed distance | yes | built, not run |
| `05_Kalman` | a Kalman filter per object: prediction, coasting, speed | yes | built, not run |
| `05b_Two_Thresholds` | sure detections start IDs, unsure ones only keep them | yes | built, not run |
| `05c_Background_Detect` | YOLO on its own thread; tracking at camera speed | yes | built, not run |
| `06_Radar` | tracked objects on a bird's-eye floor map | yes | built, not run |
| `06b_Radar_Walls` | walls from depth alone, on the radar | yes | built, not run |
| `06c_Wall_Filter` | an extended Kalman filter per wall | yes | built, not run |
| `07_Capture_Average` | averaging frames, measured noise, far range | yes | built, not run |
| `07c_Depth_Background` | a self-healing background memory | yes | built, not run |
| `07d_Remove_Person` | people painted out with the remembered room; carried things stay | yes | built, not run |
| `08_Point_Cloud` | every pixel as a coloured 3D point | yes | built, not run |
| `08b_Object_Mesh` | one object as a 3D model, with a guessed back | yes | built, not run |
| `09_Find_Floor` | RANSAC for the floor: camera height, tilt, roll | yes | built, not run |
| `09b_Room_Mesh` | the whole view as one triangle mesh | yes | built, not run |
| `09c_Room_Model` | a clean model: walls, floor, ceiling, furniture boxes | yes | built, not run |
| `10_Two_Views` | two captures put together by a pose measured by hand | yes | built, not run |
| `11_ICP` | ICP lines the two up by itself (point-to-point, written out) | yes | built, not run |
| `11b_Feature_Pose` | a rough pose from colour features, polished by the library's ICP | yes | built, not run |

Still to port: 13e, 13z and
the six exhibition demos. 13–13d are left out: 13e contains them. The flowcharts in
[`../EXAMPLES.md`](../EXAMPLES.md) still describe each idea.

## What is tested, and how

- `ctest` runs one self-check per module, on synthetic data:
  - `check_frame`: `Unmirror` and `PinholeCamera` against libfreenect2's own `Registration`, with
    no device. Every 3D point equals `getPointXYZ` with X negated.
  - `check_yolo_decode`: both model output layouts, the letterbox, and the class-name lookup.
  - `check_track`: `Tracker` agrees with example 05c's own tracking code, every frame, on a made-up
    sequence (IDs, boxes, distances, coasting, a dropped and a new object).
  - `check_planes`: a raycast synthetic room. `FindPlanes` finds its floor, ceiling and walls, and
    `FindWalls` its two walls but not the couch. It also checks that a scattered plane no longer
    hides the real ones (the review's bug 8; the check fails with the old code).
  - `check_icp`: `Kabsch` on exact pairs; `Icp` between two raycast views 12 cm and 5° apart
    (0.4 mm off); a NaN normal no longer spoils the pose (the review's bug 2); a small corner 0.8 m
    away still pins all six directions (bug 13). Both bug tests fail with 13e's code.
  - `check_mesh`: grid triangles (none across a depth jump), an outward-facing box, the OBJ writer,
    and `DepthAverager`'s hole rule.
  - `check_box_depth`: `BoxDepthSampler` gives exactly the same answers as the old per-box loop, on
    4,000 random boxes, about 2.5× faster.
- The tour programs without a Kinect run to completion and print what they show.
- **Not yet confirmed on the sensor:** the programs above that need a Kinect. The first thing to
  check: run `00a_Kinect_Source` and raise your right hand. It must show on the right.
