# Spotter examples, on the `spotter` library

This folder is the hand-off point for anyone who picks up the code. Each program is one
`main.cpp`. Each one teaches a single idea and builds on the ones before it, and all of them use
the shared library in [`../spotter/`](../spotter/). The originals they were re-written from are in
[`../legacy/`](../legacy/). Those are kept unchanged, each one written out in full.

The target system: a Kinect v2 (RGB-D) on a Raspberry Pi (Linux, CPU only). It detects objects in
colour, measures them with depth, and builds a 3D map of a room. Development is on Windows for now,
and everything must also build on Linux.

## Build, run, test

From `exercises/cpp/`. On Windows, use the "x64 Native Tools" prompt, or call `vcvars64.bat` first
from `cmd` or PowerShell (not Git Bash: `vcvars64.bat` fails there with "\Microsoft was unexpected at
this time"):

```
cmake --preset default
cmake --build --preset default
ctest --test-dir build                 # the library's self-checks: synthetic data, no Kinect
build\examples\00b_Pinhole_Camera.exe  # a tour program (no Kinect)
build\examples\02_Detect_Distance.exe  # an example (Kinect + yolo26n.onnx)
build\examples\13e_Room_Scan.exe       # the room scanner (Kinect)
```

vcpkg installs libfreenect2 (our port in `ports/`) and OpenCV the first time.

- **The programs are in `build\examples\`.** Old `.exe`s sitting straight in `build\` (such as
  `13e_Kinect_Room_Scan_Autosave.exe`) are leftovers of the legacy archive from before D47. Nothing
  rebuilds them any more, so after a `git pull` they're out of date. Run the ones in
  `build\examples\`, or delete `build\` and build again.
- **Linux:** the same commands, with `/` in the paths, after the packages in
  [`../README.md`](../README.md)'s Linux section. Not tried yet (D7); expect a long first OpenCV build.
- **The legacy archive** (`../legacy/`, 34 programs) builds only when asked:
  `cmake --preset default -DSPOTTER_BUILD_LEGACY=ON`, into `build\legacy\` (D47).
- **Room scans** go to the repo's `room_scans/` (gitignored, outside `build`), never into git.

### Building offline

vcpkg needs the internet only for things it hasn't downloaded yet. Everything it gets once stays on
the PC:

| Where | What |
|---|---|
| `build\vcpkg_installed\` | the libraries this build folder uses |
| `%LOCALAPPDATA%\vcpkg\archives\` (Linux: `~/.cache/vcpkg/archives/`) | vcpkg's binary cache: every library it has compiled, zipped |
| `%VCPKG_ROOT%\downloads\` | source archives, plus the tools vcpkg fetched (CMake, Ninja, 7-Zip, PowerShell) |

**Once, while online:** run `cmake --preset default` and `cmake --build --preset default` through to
the end. After that, offline:

| Situation | What to do |
|---|---|
| `build\` is there (after a `git pull` or an edit) | The usual two commands. vcpkg finds `build\vcpkg_installed\` up to date and downloads nothing. |
| `build\` was deleted | The usual two commands. vcpkg unzips every library from its binary cache in a few minutes, without compiling. |
| vcpkg still tries to download (the error names a URL), and `build\vcpkg_installed\` is there | Skip vcpkg: `cmake --preset default -DVCPKG_MANIFEST_INSTALL=OFF`, then build. CMake remembers this setting. Once you're back online, run `cmake --preset default -DVCPKG_MANIFEST_INSTALL=ON` to switch vcpkg back on, or a later change to `vcpkg.json` is silently ignored. |

You do need the internet when `vcpkg.json`, its `builtin-baseline` or `ports/` changed in a pull
(new library versions are compiled from new sources), or on a PC where vcpkg has never built this
project.

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
| `view.hpp` | `FpsMeter`, `Every` (a periodic timer), `IsQuitKey`; `WithKeys` (every window shows its keys in a strip below the picture); `OrbitView` (clicks too) + `PointRenderer` (a small 3D viewer); `DrawCamera` | 00g |
| `geometry.hpp` | `Pose` (turn + shift, `Then`, `Degrees`), `Kabsch` (best pose for known pairs), `TurnAbout`; `Plane`, `PlaneThrough`, `FitPlane`; `PointGrid` + `ThinCloud`; `FindPlanes` (repeated RANSAC, biggest connected piece; upright / level / any) | 00f |
| `register.hpp` | `SampleSurface` (sparse points + normals: one rule for every scan); `Icp` (13e's point-to-plane ICP; directions the shape can't pin keep the start's value) → `IcpResult` | 00f |
| `walls.hpp` | `Wall`, `FindWalls` (06b: walls as lines on the floor plan) | |
| `radar.hpp` | `RadarView` (06's bird's-eye map), `TrackTrails` (where each tracked object has been) | |
| `mesh.hpp` | `Mesh` (named groups), `MeshFromGrid`, `AddQuad`, `AddBox`, `TurnToYUp`, `SaveObj`, `SavePly` | 00g |
| `features_orb.hpp` | `FindFeatures` (ORB, optional contrast evening), `MatchFeatures` (ratio test), `NearestFeatures` | |
| `room_scan.hpp` | `Scan` + `MakeScan`; `PlaceScan` (features against the earlier scans, ICP, a second opinion) → `Placement`; far features, `Landmark`, `AddLandmarks`; `CheckLive` (live anchors); `RecommendNext` (the next shot); `ScanOptions` = 13e's settings | 00h |
| `room.hpp` | `FuseScans` (TSDF + surface net), `BuildRoom` (floor, ceiling, walls, furniture boxes, levelled) → `Room`; `SaveRoomObj`, `RoomPoints` | 00h |
| `structure.hpp` | a prototype for the structure-localisation study: `ExtractStructure` (planes sorted into floor, ceiling, walls; area and density), `FindEdges`, `MergeStructures`, `Relocate` (a pose from structure alone, ranked by `ViewAgreement`, ambiguity flagged), `ColourVotes` / `OffPlaneShare` (tie-breakers), `PlaceAfter` (Continuous mode) | 13y |
| `room_folder.hpp` | the room folder, format 2: `StartRoomFolder`, `WriteScan` (pose.txt last), `WritePose`, `RemoveScansFrom`, `WriteLandmarks`, `FinishRoomFolder`; `FindRooms`, `ScanFolders`, `ReadRoomInfo`, `LoadScan` (format 1 mirrored back), `SetExcluded`, `LoadLandmarks`; `Ask`, `AskRoomFolder` | 00h |
| `room_twin.hpp` | for the room twin (Demos 4, 6): `LoadTwinRoom` (a room folder merged into one levelled surface), `NewestRoomFolder`, `OneViewRoom` (no room: the Kinect's own view); `FindTwinFloor` (D33's floor rule, with the review's bug 7 fixed); `LocateKinect` (ICP from several starts around scan 1's spot) | |

Added with the demos: `StillBackground` in `depth.hpp` (03's empty view after a countdown, for Demos 4 and 5)
and `OpenExhibitWindow` / `ToggleFullscreen` in `view.hpp`. The wall filter stays inline in 06c, and
Demo 3 copies it.

**The room folder (format 2).** 13e writes it on K, Q, a lost Kinect or an error (no autosave, D39); 13z reads it. `room.txt` (format,
cameras, scan count, `finished`), one `scan_NNN` folder per scan (`cloud.bin`, `colour.bmp`,
`colour_full.bmp`, `colour_index.bin`, `features.yml`, then `pose.txt` last, holding the cloud's
point count), `landmarks.txt`, and `points.ply` once finished. Format 2 is the right way round;
format 1 (legacy 13c–13e) was mirrored and is read back unmirrored. Legacy 13z refuses format 2,
and legacy Demos 4 and 6 read 13's mirrored `.obj` files: use the new programs with new rooms. The
new Demos 4 and 6 read room folders directly.

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
| `13e_Room_Scan` | s1 place each scan among the earlier ones → s2 one mesh and a clean model from all → s3 save the scans as a room folder → s4 live anchors, checked in the background → **13e** |
| `13z_Room_Rebuild` | s1 find the saved rooms (no Kinect) → s2 one room's scans in the viewer (no Kinect) → **13z** |

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
| `00_tour/00h_Room_Folder` | 3 made-up scans written as a room folder, read back, a zeroed cloud refused, the room rebuilt | no | runs (synthetic) |
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
| `13e_Room_Scan` | the room scanner: many spots, placed in the background, live anchors, next shot; saved on K / Q / a lost Kinect / an error | yes | run on the sensor by the user (a 21-scan room, with the autosave version); the save-once version run on synthetic frames |
| `13w_Room_Shell` | just the walls (full height), floor and ceiling, each textured from the scans' full colour pictures, unseen parts filled with the surface's own pattern; K saves `.obj` + `.mtl` + `.bmp`s | no | runs on a synthetic room with a fan, a couch and a plank floor (a temporary harness: all checks passed); the view and a real room not checked |
| `13x_Room_Fill` | a 13e room with the holes in its flat surfaces filled and marked (legacy Demo 6's filling); K saves a `.ply` | no | runs on a synthetic room folder (floor -1.20 m, couch top 0.60 m, 2.9 m^2 filled); K and the view not checked by eye |
| `13y_Structure_Survey` | measurements for the structure-localisation study: coverage, edges, Relocate vs baselines, Continuous | no | runs (on the real room, numbers only) |
| `13z_Room_Rebuild` | rooms rebuilt from their folders: view, exclude, `.obj`, `.ply` | no | runs (on a synthetic room) |
| `demos/Demo1_You_In_3D` | yourself as a live, swinging 3D point cloud (08) | yes | built, not run |
| `demos/Demo2_Invisibility_Cloak` | camera and "cloaked" view side by side: people painted out (07d) | yes | built, not run |
| `demos/Demo3_Booth_Radar` | camera with boxes and distances, plus the radar with trails and filtered walls (06c) | yes | built, not run |
| `demos/Demo4_Room_Twin` | a 13e room as a turning doll's house, the visitors live inside it | yes | built; its library part passes `check_room_twin`, and its house was built from a synthetic room folder by a temporary harness |
| `demos/Demo5_How_Tall` | your height over your head, from depth alone (03) | yes | built, not run |
| `demos/Demo6_Room_Filled` | Demo 4 with the holes in flat surfaces filled and marked | yes | built; a temporary harness filled a synthetic room folder (floor at −1.20 m, couch top 0.60 m) |

13–13d are left out: 13e contains them. The flowcharts in [`../EXAMPLES.md`](../EXAMPLES.md) still
describe each idea.

**The demos** are fullscreen remixes for an exhibition: F toggles fullscreen, Q or Esc quits, and the
keys are shown along the bottom. Nothing is saved. The stand set-up is in
[`../legacy/presentation/README.md`](../legacy/presentation/README.md). Differences from legacy:
- Demos 4 and 6 load a 13e room folder: the one named on the command line, or the newest under
  the repo's `room_scans/` (legacy read 13's `room_scan_*.obj`).
- The floor rule (D33) is one library function for both, and it now also refuses a level patch
  well under the walls' foot (the review's bug 7).
- Demo 5 refuses a height when the top or bottom row has too few readings (the review's bug 12).
- The demos keep their legacy settings and logic. 07d's and 06c's code is copied into Demos 2
  and 3, and legacy Demo 6's fill is copied into Demo 6 unchanged.
- Demos have no ladders: the examples they're built from have them.

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
  - `check_room`: a patterned room raycast from 5 spots. `PlaceScan` places scans 2–5 within
    0.2 mm and 0.03°; the merged mesh lies 0.4 mm (median) off the true surfaces; floor, ceiling,
    walls and the couch come out right; the room folder reads back unchanged, a zeroed `cloud.bin` is
    refused (the review's bug 4), folders sort by number (bug 5), a format-1 scan loads unmirrored.
  - `check_structure`: rooms from a floor plan (L-shape, rectangle, square, one with a couch and a
    doorway). `Relocate` never gives a wrong answer as unique; symmetric rooms are flagged ambiguous
    and colour votes pick the truth; Continuous recovers moves up to 30 cm / 20 deg from a corner view.
  - `check_room_twin`: check_room's room written as a 3-scan folder. `LoadTwinRoom` levels it; the
    floor is found at −1.20 m, also with a level "reflection" 0.4 m under it (the review's bug 7);
    `LocateKinect` finds the Kinect put back 0.28 m / 25° and 0.64 m / 33° off, within 0.1 cm / 0.1°;
    `OneViewRoom` stands level.
  - `check_mesh`: grid triangles (none across a depth jump), an outward-facing box, the OBJ writer,
    and `DepthAverager`'s hole rule.
  - `check_box_depth`: `BoxDepthSampler` gives exactly the same answers as the old per-box loop, on
    4,000 random boxes, about 2.5× faster.
- The tour programs without a Kinect run to completion and print what they show.
- **Not yet confirmed on the sensor:** the programs above that need a Kinect. The first thing to
  check: run `00a_Kinect_Source` and raise your right hand. It must show on the right.
