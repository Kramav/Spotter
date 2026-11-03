# Kinect v2 examples: C++ and libfreenect2

Reference notes for the C++ examples. To get started, see the [main README](../../README.md).

These are the open-source versions of `exercises/csharp`. They use no Microsoft SDK and no
Microsoft driver, and the same code builds on Windows and Linux. For the reasons behind this, see
[docs/decisions.md](../../docs/decisions.md), entries D5 to D15.

Each example's pipeline, and what it adds to the one before, is drawn in
[EXAMPLES.md](EXAMPLES.md).

| Example | Teaches | Status |
|---|---|---|
| `01_Kinect_Depth_Peek` | Reading depth, what "no reading" looks like, depth noise | Runs on the sensor at 30 fps; window matches the C# version |
| `02_Kinect_Detect_Distance` | YOLO detection plus depth registration: how far away is it? | Runs on the sensor at 4–6 fps (detection 130–156 ms on the CPU, same as C#) |
| `03_Kinect_Depth_Blobs` | Background subtraction, blob size from depth | Runs on the sensor; blobs not yet tested with a person |
| `04_Kinect_Track_Smooth` | Tracking by box overlap (IDs), and a running average of each object's distance | Runs on the sensor at about 6 fps. Two still chairs kept their IDs; jitter fell from 2–4 mm to 0.4 mm per frame |
| `05_Kinect_Kalman` | Kalman filter per object (predict, then correct): matching against predicted boxes, coasting through missed frames, speed | Runs on the sensor at about 6 fps. A seated person kept ID #1 and read about 0 m/s. Not yet tested walking |
| `05b_Kinect_Two_Thresholds` | ByteTrack-style two-pass matching (strong detections start IDs, weak ones only continue them); 5 s memory with the speed fading out | A flickering TV: 7 IDs in 60 s with 05, 1 ID with 05b |
| `05c_Kinect_Background_Detect` | YOLO on a background thread; the main loop predicts boxes and re-measures depth every frame. Settings: `DetectInBackground`, `ModelInputSize` 640/320 | Display: 4 fps when waiting for YOLO; 24–27 fps in the background with the exposure capped (main loop 3–5 ms per frame). YOLO about 200 ms at 640, 45–60 ms at 320 |
| `06_Kinect_Radar` | Deproject tracked objects to X/Z in metres (colour intrinsics); bird's-eye radar with range rings, field of view, trails | Runs on the sensor. Colour FOV 83° (spec 84°); still objects steady to about 1 cm. The left/right direction is not yet checked by eye |
| `06b_Kinect_Radar_Walls` | 06 plus walls from depth alone: 09's neighbour RANSAC on a sparse grid (`SampleStep` 4), keeping only upright planes (within `MaxWallTiltDegrees` 20 of vertical); the plane's biggest connected piece, refit by PCA, taken away, then search again (up to `MaxWalls` 4); pieces under `MinWallShare` (5%) or `MinWallHeightMeters` (0.8) skipped; drawn as a line from end to end (2–98% percentiles along the wall). `MaxDepthMeters` 6 | Runs on the sensor at 5.6–5.7 fps, the same as 06 (the wall search takes 1–6 ms). A wall at 3.14–3.15 m, turned +6°, steady to 1 cm; one to three nearby surfaces at 0.26–0.88 m also counted (not yet identified). A synthetic room passed (scratchpad, not committed): back wall 4.000 m and side wall 1.500 m found, a 0.45 m-tall couch front skipped; at 10° tilt the same; at 30° the wall ahead drops out, the side wall stays. The first try took the couch: its plane also sliced a strip off the ceiling, hence the biggest-piece rule. Not yet checked by eye or with a tape |
| `06c_Kinect_Wall_Filter` | 06b plus an extended Kalman filter per wall. State: the foot (nearest point) as X, Z plus each end's distance along the wall; measurement: the plane's distance, the angle to the foot, the two ends (the hypot/atan2 Jacobian). Walls stand still (`WallDriftMeters` per frame); matched by `MatchMeters`/`MatchDegrees` (as 05); drawn after `ConfirmFrames` (3), forgotten after `ForgetFrames` (30). Radar: raw walls thin, filtered thick | Runs on the sensor at the same fps as 06b (3.6–4.2 fps that day for both; the filter takes 0.006 ms). Spread of the once-a-second readings over a minute: wall ahead raw 1.7 mm → filtered 0.3 mm, near wall 1.6 → 0.6 mm, two side surfaces 50–63 → 14–16 mm. The camera was tilted about 14°: 06b's radar line put the wall ahead at 3.255 m, the plane's own distance (used now) 3.16 m. A synthetic check passed (scratchpad, not committed): Jacobian, noise 19.6 → 3.8 mm, angle wrap, confirm/forget, the 06b room level and tilted 10° (4.000 and 1.499 m, IDs kept 60 frames). Radar not yet checked by eye |
| `07_Kinect_Capture_Average` | Per-pixel average over `CaptureFrames` still frames: one-frame noise by distance, expected noise of the average (÷√N), and its real noise measured from two captures | Runs on the sensor: one frame ±1.2 mm at 0.5–1 m; average expected ±0.2, measured ±0.3 mm. Tested with a temporary auto-capture build (no key presses sent) |
| `07b_Kinect_Far_Range` | `Config::MaxDepth` raised from libfreenect2's default 4.5 m to `MaxDepthMeters` (8); per 1 m band: share of view, flaky pixels, one-frame noise | Runs on the sensor: never-read pixels 33.8% (4.5 m) → 2.3% (8 m); 4–6 m noise ±5 mm, about the same as 3–4 m; past 6 m almost nothing reads. Same auto-capture test |
| `07c_Kinect_Depth_Background` | Per-pixel depth memory: blend when the same, take after `LeaveFrames` (10) when farther, foreground when nearer (absorbed after `AbsorbFrames`, 300); blobs under `MinBlobPixels` dropped | Runs on the sensor: 0 blobs with nobody in view. The first version took a farther reading at once and showed 2–6 blobs on a spinning ceiling fan (blades 3 m, ceiling 4.7 m); with `LeaveFrames` 10 the fan blends into the background. Not yet tried with someone walking in |
| `07d_Kinect_Remove_Person` | 07c plus YOLO on a thread (05c): a blob mostly (`PersonShare`) inside person boxes is painted with the remembered colour, except inside other objects' boxes (a carried cup stays); colour memory learns only away from foreground and person boxes; never-seen = magenta; V = explain view | Runs on the sensor at 28 fps. Nobody was in view and YOLO found nothing (dim room, capped exposure), so the removal itself is untested |
| `08_Kinect_Point_Cloud` | `getPointXYZ` for every pixel plus registered colour; its own z-buffered viewer (drag/wheel or WASD/ZX); P saves ASCII `.ply` (Y up), cut-off 6 m | Runs on the sensor: about 193k points, 14 loops/s; test `.ply` had z 0.5–5.7 m, 11.6% black points (outside the colour camera's view). Tested with a temporary auto-save build; the file was deleted. Viewer not yet checked by eye |
| `08b_Kinect_Object_Mesh` | Click a YOLO box (02, synchronous; person refused), or click an object with no box: snap to the nearest depth pixel, points within `ClickReachMeters` (0.5), support plane removed, then `floodFill` from the click with `MaxJumpMm` as the step limit (refused if over `PersonShare` 0.3 lies in person boxes). Box mode: 30-frame average (07); cut-out = box + `DepthBandMeters` (0.3) around the centre median + support plane by RANSAC with few points below it (09) + biggest blob; grid triangles skipped over `MaxJumpMm` (20); `BackGuess` Mirror (as deep as wide) / Flat / None, rim stitched on border edges, grey in group `guessed_back`; `.obj` with vertex colours, Y up | Runs on the sensor (auto-click test build, files deleted): a TV at 4.5 m came out 1.55–1.57 m wide, 0.85–0.91 m high, three captures within 2 cm, about 7,200 points and 13k triangles, 15 ms to build; two people in view were skipped. Needs `MaxDepthMeters` 6 (the 4.5 m default cut the angled TV's far half: 0.9 m wide). Rim edges at pinch points are used twice (closed, not strictly manifold). A synthetic ball-on-table check (scratchpad, not committed) passed: table and wall removed, mesh closed (every edge used once each way), scanned faces face the camera, mirrored back 1.077 vs true 1.080 m, OBJ parses. Click mode, same synthetic scene: a click on the ball keeps the same 2,658 pixels as the box; a click on the table is refused; a click on the wall keeps only wall. Click mode not yet run on the sensor |
| `09_Kinect_Find_Floor` | RANSAC plane fit (3 neighbouring points, level planes only, then a PCA refit on the inliers): camera height, tilt, roll; crosshair height above the floor | Runs on the sensor, but no floor was in view: it reported a different level-ish surface each second (6–8% of points). Not yet tested with a floor in view |
| `09b_Kinect_Room_Mesh` | S scans: 30-frame average (07), a point per pixel with colour (08), grid triangles every `MeshStep` (2) pixels, skipped where the corners differ by more than `MaxJumpShare` (3%) of their distance (08b's fixed mm tears slanted far walls); `room_<time>.obj`, group `seen_room` | Runs on the sensor (temporary auto-scan build; file counted, then deleted): 46k points, 89k triangles, 3.8 MB, 7 ms; 15% of the view had no depth. Everything in view was 0.5–1.5 m away that time. Not yet opened in Blender/MeshLab |
| `09c_Kinect_Room_Model` | 09b plus a clean model: 06b's repeated RANSAC for planes facing any way (up to `MaxPlanes` 8, each its biggest connected piece); sorted by the camera's up within `MaxTiltDegrees` (25): floor = lowest level plane below, ceiling = highest level plane at least `MinCeilingHeightMeters` (1.8) above the lowest point, wall = upright and `MinWallHeightMeters` (0.8) tall; up from the floor, else the ceiling, else two walls (cross product), else the camera; "ahead" from the biggest wall; walls as floor-to-ceiling rectangles, joined where both ends come within `CornerSnapMeters` (0.5) of the crossing; left-over points between floor and ceiling split at depth jumps into boxes standing on the floor (`MinBlockCells` 30); each piece painted its average colour; one `.obj` with groups `seen_room`, `clean_walls`, `clean_floor`, `clean_ceiling`, `furniture_blocks` | A synthetic room passed (scratchpad, not committed), level, looking down 10° and up 14°: up found (from the ceiling when looking up), tilt right to 0.1°, floor −1.000 and ceiling 1.400 m, both walls joined exactly at the corner, the couch one box (0.52–1.45 m wide, top 0.45 m above the floor), raw mesh levelled, OBJ valid (5 groups). On the sensor (auto-scan build, file deleted): 19 ms; the view held only things 0.5–1.5 m away, so up fell back to the camera, two "walls" about 1.1 m ahead and five small boxes. Not yet tried on a real room view, nor opened in Blender |
| `10_Kinect_Two_Views` | Capture A, move the Kinect, live/frozen B moved into A's coordinates by a `Pose` (shift + turn left); keys nudge the pose; median gap = B's depth vs A's along A's line of sight | Runs on the sensor (temporary auto-capture build, Kinect not moved): gap 0.5 cm at the true pose, 0.8 cm at 1 cm off, 1.2 cm at 5 cm sideways, 1.8 cm with 1° turn added. The gap is blind to sliding along a wall. Viewer and a real move not yet tried by the user |
| `11_Kinect_ICP` | 10 plus ICP (P): every `IcpStep` (4) pixel of B, nearest A point by a FLANN k-d tree, pairs beyond a gate shrinking 0.20 → 0.02 m ignored, Kabsch fit (SVD of the cross-covariance), repeated up to 30 rounds; the pose is now a full 3D turn + shift | Runs on the sensor (auto-key test build): Kinect not moved, ICP stayed at 0.000 m / 0.00°, 11.6k pairs, 3.4 mm rms, 0.5 s. Not yet tried with a real move |
| `11b_Kinect_Feature_Pose` | 11 with colour: ORB (`FeatureCount` 1000) on the registered colour picture where depth exists, Hamming matching with the ratio test (`MatchRatio` 0.8), features lifted to 3D, RANSAC over 3-pair Kabsch fits (`FeatureInlierMeters` 0.05, `MinFeatureMatches` 15), then 11's ICP; a second window draws the kept matches | Builds. On the sensor it found 0 features: the colour picture was black (brightness 5/255, no variation; the room was dark or the lens covered). The feature path is verified only synthetically (in 13's test) |
| `13_Kinect_Room_Scan` | The full room scan. SPACE: 30-frame average, sparse points with normals, ORB features (scans with under `MinScanFeatures` 100 refused: too dark). Placing: features against every earlier scan (best wins), then ICP against all earlier scans, point-to-plane, linearised about the pairs' centre, solved per eigen-direction (directions pinned by under 2% of the pairs keep the features' value: bare walls), gate 0.05 → 0.02 m; refused under `MinOverlapShare` (25%). U undoes, C colours by scan. E: TSDF fusion (`VoxelMeters` 0.03, truncation 3 voxels, grows if over `MaxVoxels` 30M) and a surface-net mesh, then 09c's clean model from all scans (planes merged across scans within `MergeDegrees` 10 / `MergeMeters` 0.05, points on a wall plane never furniture, boxes that touch joined). Scans live in memory only | A synthetic textured room from 4 poses passed (scratchpad, not committed): features alone 0.1–0.7 cm off, placed 0.2–2.1 mm / ≤ 0.08°, about 0.3–0.4 s each; an unrelated view refused; merged mesh within 0.6 mm (median) / 2.3 mm (95%) of the true surfaces, 87k triangles in 0.1 s; floor −1.001, ceiling 1.399 m, 3 walls, 2 corners joined, the couch one box. Point-to-point ICP (11's) slid 3.4 cm along a wall there, plain point-to-plane 21 cm on a view of one wall; hence the two changes. On the sensor (auto-key build, dark room): scans refused for 0 features, as designed; export of one scan worked. Not yet tried in a lit room |

## Reproducible by design (D14)

Every dependency is pinned in this folder, so a fresh `git clone` builds the same thing:

| File | Pins |
|---|---|
| `vcpkg.json` | OpenCV 4.14 with its features (including `intrinsics` and `thread`, D13). The `builtin-baseline` fixes the exact version of every library. |
| `ports/libfreenect2/` | Our vcpkg recipe for libfreenect2: upstream commit `fd64c5d`, checked by SHA512, plus our OpenMP patch (D11). |
| `CMakePresets.json` | The configure options: Ninja, Release, vcpkg toolchain. |

The first build compiles everything from source, which takes about 30 minutes. After that, vcpkg
reuses its cache (`%LOCALAPPDATA%\vcpkg\archives`), and a rebuild from a clean folder takes about
3 minutes.

## Setup and build: Windows

**New here? Follow the step-by-step guide in the [main README](../../README.md#getting-started-the-kinect-examples-windows)**:
what to install, the USB driver swap, and building and running.

In short, from an **"x64 Native Tools Command Prompt"** in `exercises\cpp`:

    cmake --preset default
    cmake --build --preset default
    build\01_Kinect_Depth_Peek

## Setup and build: Linux / Raspberry Pi

**Not yet tested.** These are the expected steps:

    sudo apt install build-essential cmake ninja-build pkg-config git curl zip unzip tar libudev-dev
    git clone https://github.com/microsoft/vcpkg ~/vcpkg && ~/vcpkg/bootstrap-vcpkg.sh
    export VCPKG_ROOT=~/vcpkg
    cmake --preset default && cmake --build --preset default
    # USB access without root: libfreenect2's udev rule
    sudo curl -L -o /etc/udev/rules.d/90-kinect2.rules https://raw.githubusercontent.com/OpenKinect/libfreenect2/fd64c5d9b214df6f6a55b4419357e51083f15d93/platform/linux/udev/90-kinect2.rules

OpenCV's window support on Linux (`gtk`) makes vcpkg build GTK too. Expect a long first build on
a Pi (hours), and possibly a few more `apt` packages that vcpkg asks for.

## Running

Run the examples from a terminal rather than by double-clicking, so error messages stay visible.
Example 02 finds `yolo26n.onnx` in the repo root by itself. On Linux, run
`./build/01_Kinect_Depth_Peek`.

Differences from the C# versions:

- **02** registers depth to colour with libfreenect2 `Registration`, in place of the SDK's
  `CoordinateMapper`. It works on the *undistorted* depth image, so the pinhole maths is exact
  everywhere. Its cross-check column is libfreenect2's `getPointXYZ`.
- **03** reads the same factory intrinsics as the SDK (fx 365.1, cx 256.9, cy 209.4 on our
  Kinect).

**Before the driver swap** you get this, which is expected:

    [Error] [Freenect2Impl] failed to open Kinect v2: @1:4 LIBUSB_ERROR_NOT_FOUND Entity not found
    No Kinect v2 found. ...

libfreenect2 can see the Kinect on the USB bus, but the Microsoft driver still owns it.

If the program exits silently with code C0000135, a DLL is missing next to the .exe. vcpkg
normally copies them all. Run `dumpbin /dependents` on the program to find which one.

## Known limits

- **The colour camera drops to about 15 fps in a dim room** (auto exposure needs more than one
  frame's time). 05c's `CapColorExposure` keeps it at about 29 fps, with a darker, grainier
  picture. In a dim room that combined with the 320 model found nothing (D15).

- Depth is decoded on the CPU across all cores (D11), because the target Raspberry Pi has no GPU
  path. Measured on a Ryzen 5 3600: 30 fps with the filters on. The Pi is not yet measured.
  `UseDepthFilters = false` roughly doubles CPU headroom.
- libfreenect2 decodes depth itself, so its holes and noise differ slightly from the Microsoft
  SDK's.
- No body tracking, because no open-source driver provides it (D5).
