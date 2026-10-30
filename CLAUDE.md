# Spotter

Learning object detection, recognition and tracking. The current focus is Kinect v2 RGB-D fusion
(detect in colour, measure distance with depth), following `kinect_rgbd_fusion_guide.md`.

**Deployment target: Raspberry Pi (Linux, CPU only, no GPU compute).** Develop on Windows for now
(D7). Anything that needs a GPU is out (D11).

**Long-term goal: a 3D map of the space.** Staged plan: fixed camera → several fixed stations
merged offline → moving camera (see [docs/ideas.md](docs/ideas.md)). The camera stays fixed for now.

## Living documents — keep these current

- **This file**: the Status and Roadmap sections. Update them at the end of any work session that
  changes either one.
- **[docs/decisions.md](docs/decisions.md)**: the design decision log. Append an entry whenever a
  choice is made that someone would later ask "why?" about. Never rewrite old entries; mark them
  superseded instead.
- **[docs/ideas.md](docs/ideas.md)**: techniques researched but not scheduled, with when each would
  pay off and how to build it. Add to it when a "maybe later" topic comes up. Move an item to the
  Roadmap once it's scheduled.

## Layout

| Path | What |
|---|---|
| `exercises/python/` | Webcam/OpenCV/YOLO exercises (no Kinect). Complete. |
| `exercises/csharp/` | Kinect v2 examples 01–03 on the Microsoft Kinect SDK 2.0. Windows only, frozen (see D5). |
| `exercises/cpp/` | Kinect v2 examples on libfreenect2 + OpenCV, in C++ (D8). The active track. Builds with two commands from its README (D14). `ports/` = our vcpkg recipe for libfreenect2. |
| `docs/` | Living documents: `decisions.md` and `ideas.md`. |
| `yolo26n.onnx` / `.pt` | Detector model. The ONNX file was exported with `nms=False` and outputs 1x300x6. |
| `kinect_rgbd_fusion_guide.md` | Background reading: the method and the accuracy tests (§7). |

## Conventions

- **The goal is understanding the system through small examples, not one-shot builds.** Each
  example teaches one idea and builds on the previous one. Long-term plans (Roadmap, ideas.md)
  get broken into a ladder of such examples; never jump straight to the full system.
- Examples are written like Arduino sketches: one file each, settings at the top, then the
  sections Settings / Setup / Main operation / Output / Experiment ideas. Priorities, in order:
  readability, simplicity, mutability, learnability, capability.
- Privacy: never save or commit camera frames or images of people. `*.jpg` is gitignored for this
  reason.
- Lean towards our own software and open source over vendor SDKs (D5). New code must build on
  Linux as well as Windows (D7).

## Status (2026-10-25)

- Python exercises: done.
- C# + Kinect SDK examples 01–03: done, frozen (commit cea5c21).
- C++ / libfreenect2 examples 01–05: built, and all run on the sensor. Reproducible build (D14).
- C++ examples through 06 (stage A) built, plus 05b/05c. Pi bring-up paused (2026-10-30). Stage B: 07, 07b, 08 and 09 built; 09 still needs a test with the floor in view. Stage C: 10 built. Side branch: 07c/07d (remove people from the picture) built.

## Roadmap

The ladder from here to the long-term goal (a 3D map of the space, on the Pi). One idea per
example, each building on the one before. Numbers are the planned example folders in
`exercises/cpp/`.

### Done

- [x] C++ track set up: language C++ (D8), libfreenect2 with the OpenMP patch (D9, D11), libusbK
      driver (D10), OpenCV with `intrinsics`/`thread` (D13), reproducible build (D14).
- [x] 01 Depth Peek: 30 fps on the sensor; the user confirmed the window matches the C# version.
- [x] 02 Detect + Distance: runs on the sensor, about 6 fps, detection 130–180 ms.
- [x] 03 Depth Blobs: runs on the sensor; the intrinsics match the SDK's.
- [x] 04 Track + Smooth: IDs by box overlap, running average (D12). Still chairs: jitter fell from
      2–4 mm to 0.4 mm per frame.
- [x] 05 Kalman: predict then correct, matching against predicted boxes, coasting, speed.
- [x] 05b Two thresholds: strong detections start IDs, weak ones (0.15+) only continue them;
      MaxMissedFrames 30, with the speed fading while unseen. Same TV in 60 s: 7 IDs with 05, 1 with
      05b. 05's logs showed the cause: detection gaps over 2 s, not matching.
- [x] 05c Background detect: YOLO on its own thread; the main loop predicts and re-measures depth
      every frame. Display 4 → 15 fps (later 24–27, D15). `ModelInputSize` 320 setting
      (`yolo26n_320.onnx`): 175 → 55–80 ms per YOLO run, but it misses objects cut off at the image
      edge. A dynamic-size ONNX does NOT load in OpenCV 4.14 DNN (Concat shape error), hence two
      files.
- [x] Camera frame rate fixed (D15): OpenMP `schedule(dynamic, 8)` (depth with colour on
      14 → 27 fps), plus 05c's `CapColorExposure` setting. 05c display: 15 → 24–27 fps. Capped
      exposure plus the 320 model found nothing in a dim room.
- [x] 06 Radar (stage A, below).

### Open checks (the user, by eye)

- [ ] 02: do the boxes sit on the objects? 03: walk in; is the height on the blob roughly right?
- [ ] fps overlay visible in each window.
- [ ] 04 vs 05 while walking: does 05 hold IDs better? Is its speed plausible (walking is about
      1–1.5 m/s)? Both print `new ID` lines explaining each ID change.

### Stage A — objects in 3D (fixed camera)

- [x] **06 Top-down radar:** built 2026-10-25, runs on the sensor (colour FOV 83°, still objects
      steady to about 1 cm). The user still needs to check that left/right matches the camera
      view. each tracked object (from 05) becomes an X/Z position, drawn on a
      bird's-eye floor plan. The first "map".

### Milestone — Pi bring-up (PAUSED by the user 2026-10-30; stage B goes ahead on the PC meanwhile)

- [ ] Build and run on Linux, then on the Raspberry Pi. Measure the depth fps with the filters on
      and off, check the Kinect is stable on the Pi's USB 3 port, and time YOLO. Expect over 1 s
      per frame on a Pi 5, versus about 170 ms on the Ryzen. The results decide how heavy stages B–D
      can be. Already available: background detection (05c) and the 320 model.
- [ ] **Lever 3, NCNN** (BSD, the usual YOLO engine on a Pi): export `yolo26n.pt` to NCNN and
      compare its speed with OpenCV DNN on the Pi at 640 and 320. Adding it means a vcpkg
      dependency (`ncnn`) and a second detector backend. Decide from the measurements.
- [ ] **Lever 4, INT8 quantisation:** an 8-bit model (NCNN int8, needs a calibration image set).
      Measure the speed gain against the accuracy loss on the same test scene.
- [ ] If the CPU still isn't enough: the Raspberry Pi AI HAT (Hailo) is the only non-CPU option.
      It would mean revisiting the CPU-only rule (D11).

### Stage B — the room in 3D (fixed camera)

- [x] **07 Capture and average:** built 2026-10-30. Runs on the sensor: one frame ±1.2 mm (0.5–1 m),
      average measured ±0.3 mm (expected ±0.2). Space capture not yet tried by the user.
- [x] **07b Far range:** built 2026-10-30. `Config::MaxDepth` 4.5 → 8 m: never-read pixels
      33.8% → 2.3% in the same scene; a 4–6 m wall read at ±5 mm one-frame noise (no worse than
      3–4 m); past 6 m almost nothing reads. The user confirmed the blue (past 4.5 m) covers the right parts of the room. Accuracy (bias) at range is unmeasured: tape measure.
      Later examples still use the 4.5 m default; decide when 08 is built.
- [x] **08 Depth to point cloud:** built 2026-10-30. A coloured 3D point per pixel, a window you
      can rotate (our own z-buffered viewer, no extra library), P saves `.ply`. Cut-off 6 m (07b).
      Runs on the sensor: about 193k points, 14 loops/s. The user still needs to check the viewer.
      Point clouds are captures of the room: `*.ply`/`*.pcd` are gitignored; never commit them.
- [x] **07c Depth background:** built 2026-10-30. Remembers the room's depth per pixel; anything
      clearly nearer is foreground. Something must read farther for `LeaveFrames` (10) before it
      counts as gone: taking one reading at once let ceiling noise make 2–6 false blobs. Still
      scene: 0 blobs.
- [x] **07d Remove person:** built 2026-10-30. Blobs mostly inside YOLO person boxes are painted
      with the remembered colour; other objects' boxes stay live, so a carried cup stays visible
      (the user's requirement). 28 fps. Removal untested: nobody in view, YOLO found nothing.
- [ ] **07e Fill never-seen pixels** (needed later, the user's call): what nobody has seen yet
      (magenta in 07d) gets a guess: `cv::inpaint` for colour, plane fill for depth (09).
- [x] **09 Find the floor:** built 2026-10-30. A RANSAC plane fit gives the camera's height, tilt
      and roll, and any point's height above the floor. Three random points from anywhere almost
      never all hit a small floor, so it picks three neighbours. Run with no floor in view, it
      reported a different level-ish surface every second (6–8% of points), so `MinFloorFraction`
      (5%) is too low to say "no floor". Needs a test with the floor in view, and a tape check of
      the height.

### Milestone — accuracy (before stage C; merging views exaggerates any distance error)

- [ ] Tape-measure accuracy, guide §7 tests 2–4. If it comes out poor, revisit D6 (our own
      calibration).

### Stage C — several views merged (details: docs/ideas.md, "Multi-view registration")

- [x] **10 Two views, known pose:** built 2026-10-30. Capture A, move the Kinect, B is placed by
      the pose from the settings (shift + turn left), keys nudge it, plus a median gap number.
      Runs on the sensor (Kinect not moved): gap 0.5 cm at the true pose, only 1.2 cm at 5 cm
      sideways, because the gap is measured along A's line of sight (blind to sliding along a
      wall). The user still needs to try a real measured move and look at the double walls.
- [ ] **11 ICP:** fix that misalignment. Write a simple version ourselves first (nearest neighbours
      + SVD); Open3D only if we outgrow it.
- [ ] **12 ArUco pose:** a printed marker gives the camera's pose. Needs OpenCV's `calib3d` and
      `objdetect` added to `vcpkg.json`.
- [ ] **13 Merge many views:** 5–10 captures into one room model; downsampled points first, then a
      TSDF mesh.

### Stage D — the goal

- [ ] **14 Objects in the map:** the tracked objects (05/06) placed in the room model from 13.

### Stage E — moving camera (later, optional; docs/ideas.md, "Motion compensation")

- [ ] **15 Visual odometry:** the camera's own motion, frame to frame (11's ICP on consecutive
      frames).
- [ ] **16 Live mapping:** a coarse CPU KinectFusion, or decide to adopt RTAB-Map instead.

### Housekeeping

- [ ] Decide whether to retire `exercises/csharp/`. Then the Microsoft Kinect SDK (KinectService)
      can be uninstalled from the dev PC.

Toolchain on the dev machine: VS Build Tools 2026 with the C++ workload, and .NET 10. Building C++
needs `vcvars64.bat`, which puts `cmake`, `ninja` and `vcpkg` on PATH. In Git Bash, run the built
.exe through cmd or PowerShell, because Git Bash's loader fails on `api-ms-win-crt-*`.
