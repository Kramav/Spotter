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
- **README "How it fits together"**: the Mermaid system map. When a planned step is built, drop
  it from the `class … planned` line; add new steps as nodes.
- **[exercises/cpp/EXAMPLES.md](exercises/cpp/EXAMPLES.md)**: one Mermaid flowchart per example.
  Every new example gets a section there (its pipeline, with the steps it adds in the thick-border
  `new` class) and a node in the family tree.
- **[docs/visualizer.html](docs/visualizer.html)**: one page that animates every example's steps on a
  simulated Kinect (raycast room, walking person, depth noise) using the examples' own settings and
  logic. Every new example gets an `EX.push({...})` entry there too. Open it in a browser.

## Layout

| Path | What |
|---|---|
| `exercises/python/` | Webcam/OpenCV/YOLO exercises (no Kinect). Complete. |
| `exercises/csharp/` | Kinect v2 examples 01–03 on the Microsoft Kinect SDK 2.0. Windows only, frozen (see D5). |
| `exercises/cpp/` | Kinect v2 examples on libfreenect2 + OpenCV, in C++ (D8). The active track. Builds with two commands from its README (D14). `ports/` = our vcpkg recipe for libfreenect2. `EXAMPLES.md` = a flowchart per example. `presentation/` = five fullscreen exhibition demos remixed from the examples (D22). |
| `docs/` | Living documents: `decisions.md` and `ideas.md`, plus `visualizer.html` (animated examples). |
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

## Status (2026-11-03)

- Python exercises: done.
- C# + Kinect SDK examples 01–03: done, frozen (commit cea5c21).
- C++ / libfreenect2 examples 01–05: built, and all run on the sensor. Reproducible build (D14).
- C++ examples through 06 (stage A) built, plus 05b/05c, 06b (walls on the radar) and 06c (an EKF per wall). Pi bring-up paused (2026-10-30). Stage B: 07, 07b, 08 and 09 built; 09 still needs a test with the floor in view. Room recreation from one view: 09b (raw mesh) and 09c (clean model) built. Stage C: 11 (ICP), 11b (feature pose) and 13 (room scan from many spots, the full tool) built; not yet run in a lit room. 13b (13 for quality: CLAHE + FAST 7 for features in dim light, D24) built: the current dim view gives 13 about 60 features (refused), 13b about 1250. 13c (scan assist: live anchors, next shot, exclude/re-place, suspicious flag, no slowdown, D26) built and synthetic-tested; it also fixes an ICP drift that 13 and 13b share. 13d (far features for hallways: rays, landmarks, D27) built and synthetic-tested. Stage C: 10 built. Side branches: 07c/07d (remove people from the picture) built; 08b (a detected object as an `.obj`) built and run on the sensor. Exhibition: `exercises/cpp/presentation/` Demos 1–5 built (D22); Demo 4 (room twin) passed a synthetic test; none run on the sensor yet. Planned 2026-11-04 (not started): K in 13c/13d saves a scan folder (raw points `.ply` + every scan's data), 13z rebuilds rooms from those folders, 14 = textured room; the later examples shift up one (15 objects in the map, 16 visual odometry, 17 live mapping).

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
- [x] **06b Radar + walls:** built 2026-11-02 (D18). Walls from depth alone: 09's RANSAC for
      upright planes, biggest connected piece, repeated (up to 4), under 0.8 m tall = furniture;
      drawn as lines on 06's radar. Synthetic room passed (4.000 m and 1.500 m walls, couch
      skipped). On the sensor: 5.6–5.7 fps (same as 06), a wall at 3.15 m steady to 1 cm, plus
      some near surfaces (0.26–0.88 m) the user should identify. Not yet checked by eye or tape.
- [x] **06c Wall filter:** built 2026-11-02 (D19). An extended Kalman filter per wall (state: the
      foot as X, Z plus the ends; measurement: distance and angle, hence the Jacobian), matched as
      in 05, drawn after 3 sightings. On the sensor the wobble fell 3–6× (wall ahead 1.7 → 0.3 mm).
      It also takes the plane's own distance, fixing 06b's tilt error (3.255 → 3.16 m at about
      14°). Radar not yet checked by eye.

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
- [x] **08b Object mesh:** built 2026-11-01 (D17). Click a YOLO box: 30-frame average, cut-out
      (box + depth band + support plane + biggest blob), grid triangles, a guessed grey back
      (mirror, as deep as wide) in its own group, saved `.obj` (gitignored). A synthetic
      ball-on-table check passed (closed mesh, back 1.077 vs 1.080 m). On the sensor: a TV at
      4.5 m measured 1.55–1.57 m wide, repeatable to 2 cm; needed `MaxDepthMeters` 6 (at 4.5 m
      its far half was cut off). Mirror makes the TV 1.5 m deep: Flat suits it. A paper cup on the
      floor, seen side-on, was "toilet"/"fire hydrant", never "cup" (ideas.md); scanned by its
      box it measured 8–9 cm × 13.5 cm, right.
      Click-anywhere added 2026-11-02: with no box under the click, grow the object from the
      clicked point (flood fill stopping at depth jumps). Synthetic check passed; not yet run on
      the sensor. The user still needs to open a file in Blender/MeshLab.
- [ ] **08c Object mask:** 08b with a YOLO-seg mask instead of box + depth band. First test
      whether OpenCV DNN loads `yolo26n-seg`. Better back guesses (shape by class): ideas.md.
- [x] **07c Depth background:** built 2026-10-30. Remembers the room's depth per pixel; anything
      clearly nearer is foreground. Something must read farther for `LeaveFrames` (10) before it
      counts as gone: taking one reading at once made 2–6 blobs on a spinning ceiling fan (the
      user confirmed it was the fan, not noise). With 10, the fan blends into the background: 0 blobs.
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
      the height. The Kinect sits ON the floor (2026-11-02), where the floor is seen edge-on and
      gives no depth: raise it (table or tripod, tilted slightly down) before this test.

- [x] **09b Room mesh:** built 2026-11-02 (D20). The user's ask: recreate the room from a scan,
      one view first, then several spots like Meshroom, raw and clean. S scans: 30-frame average,
      the whole view as a coloured grid mesh (jumps over 3% of the distance skipped), `.obj`.
      Runs on the sensor (46k points, 89k triangles, 3.8 MB). Not yet opened in Blender.
- [x] **09c Room model:** built 2026-11-02 (D20). 09b plus a clean model in its own OBJ groups:
      planes facing any way (06b's search), sorted into floor/ceiling/walls, levelled (up from the
      floor, ceiling or two walls), walls joined at corners, left-overs as boxes on the floor.
      Synthetic room passed at 0°, 10° down and 14° up. On the sensor the view held only things
      0.5–1.5 m away (the Kinect seems to have moved), so it hasn't modelled a real room view yet.

### Milestone — accuracy (before stage C; merging views exaggerates any distance error)

- [ ] Tape-measure accuracy, guide §7 tests 2–4. If it comes out poor, revisit D6 (our own
      calibration). The user has no tape measure (2026-11-02): use things of known size instead
      (the TV's spec-sheet width against 08b's 1.55–1.57 m, A4 paper 297 × 210 mm, a door) or a
      phone measuring app.

### Stage C — several views merged (details: docs/ideas.md, "Multi-view registration")

- [x] **10 Two views, known pose:** built 2026-10-30. Capture A, move the Kinect, B is placed by
      the pose from the settings (shift + turn left), keys nudge it, plus a median gap number.
      Runs on the sensor (Kinect not moved): gap 0.5 cm at the true pose, only 1.2 cm at 5 cm
      sideways, because the gap is measured along A's line of sight (blind to sliding along a
      wall). The user still needs to try a real measured move and look at the double walls.
- [x] **11 ICP:** built 2026-11-02 (D21). Point-to-point, k-d tree (FLANN) + Kabsch/SVD, P runs
      it. On the sensor (Kinect not moved) it held 0.000 m / 0.00°. Not yet tried with a real move.
- [x] **11b Feature pose (Meshroom-style, D20):** built 2026-11-02. ORB on the registered colour,
      ratio-test matches lifted to 3D, RANSAC + Kabsch, then 11's ICP. On the sensor it found 0
      features: the colour picture was black (dark room or covered lens). Needs a lit room.
- [ ] **12 ArUco pose:** a printed marker gives the camera's pose. Needs OpenCV's `calib3d` and
      `objdetect` added to `vcpkg.json`. Now a fallback after 11b (blank walls, nothing to match).
- [x] **13 Room scan:** built 2026-11-02 (D21). The full tool; the user's guide is "Recreate a
      room" in the README. SPACE per spot (scans refused under 100 features, i.e. in the dark),
      placed by features + point-to-plane ICP against all earlier scans (bare-wall directions keep
      the features' value), U undo, E = TSDF + surface nets + 09c's clean model across scans.
      Synthetic textured room from 4 poses: placed within 2.1 mm / 0.08°, mesh 0.6 mm median off,
      clean model right. On the sensor only a dark-room smoke test (scans refused, as designed).
      The user still needs a real scan in a lit room. Later: loop closure for long loops.
- [x] **13b Room Scan HQ:** built 2026-11-03 (D24). 13 with CLAHE before ORB and a FAST
      threshold of 7 (the user prefers quality over time for room scans), plus more frames,
      features, RANSAC tries, ICP points and rounds, and 2 cm cubes. Same dim view on the sensor:
      13 found 48–73 features (refused), 13b 1225–1271, with a second scan placed 0.5 mm off.
      Used for a real 34-scan room: slowed down from scan 28, scan 34 came out misplaced, a
      hallway was dropped, and the PC blue-screened once (D25). Hence 13c and 13d.
- [x] **13c Room Scan Assist:** built 2026-11-03 (D26). Live green anchors + counter (replaces the
      0.5–1 m rule), next-shot advice, click to select, X exclude, P re-place, red suspicious flag
      (incl. a second-opinion ICP against all but the matched scan), shortlist + cropped/thinned
      ICP on a background thread. Synthetic 30-scan walk: worst 2.9 mm / 0.18°, about 3 s per scan
      at any count (13b: 3–45 s and 12+ cm drift, from ICP's 2% weak cut-off; 13c uses 5%).
      Live anchors not yet run on the sensor. The user's choice for real scans.
      Export fixed 2026-11-03 (D28): up from the biggest level plane, boxes capped at 3 m (13b's
      flat export came out 28° tilted inside one flat-sized box). 13, 13b and 13d have the same fix.
- [x] **13d Room Scan Far:** built 2026-11-03 (D27). 13c plus features with no depth (the full
      colour picture, masked by bigdepth) as rays: they vote in RANSAC, give the turn when near
      matches are few (then ICP with a 0.5 m gate), and triangulate into far landmarks (exported as
      `far_landmarks`), which pin later scans' position and turn. Synthetic plain hallway with a
      12 m end wall: 13c refuses every scan; 13d places a sideways step 34 mm off and a 2 m walk in
      33 mm off (via 95 landmarks, median 12 cm depth error). Not yet run on the sensor.
- [ ] **13c/13d: K keeps the scans** (planned 2026-11-04, the user's ask; not started). The `.obj`
      loses colour and detail: one averaged colour per 2 cm vertex, mixed across every scan that saw
      the spot, and no texture. And the scans that could rebuild it are gone on quit. K (when idle)
      asks in the console for a folder (Enter = the `SaveFolder` setting, default `room_scans\`
      where it runs) and a name (Enter = `room_<date>_<time>`). An existing name asks before
      overwriting. Then it writes one new folder:
      - `room.txt`: format version, program (13c/13d), date, scan count, the settings that matter,
        both cameras' libfreenect2 parameters, the first camera turn (Demo 4).
      - `points.ply` (**A**): every placed scan's own points and colours, no averaging, in the same
        levelled coordinates as E's `.obj`. Binary PLY (about 5M points for 34 scans).
      - `scan_001\` … (**B**): `cloud.bin` (the 60-frame average as float x, y, z per depth pixel),
        `colour.png` (registered, 512×424), `colour_full.png` (1920×1080, for 14),
        `features.yml` (keypoints + descriptors; 13d adds its far ones), and `pose.txt` (turn,
        shift, excluded, suspicious, stats). 13d also writes `landmarks.txt` at the room level.
      - A `.gitignore` containing `*`, so the folder can never be committed, wherever it is.
      - About 8 MB per scan (34 scans ≈ 270 MB). The windows freeze while you type in the console.
      - **Privacy, needs the user's OK before building:** this saves colour frames, which the
        Conventions rule forbids. Proposed: allowed for empty-room scans only, in self-gitignored
        folders, never uploaded, never read into Claude's context (scripts print summaries only),
        deleted when no longer needed. Record as D29 and amend the Conventions line.
- [ ] **13z Room Rebuild** (planned 2026-11-04; not started). No Kinect needed. It rebuilds rooms
      from K's folders.
      1. Finds every folder holding a `room.txt` under `SearchFolder` (or a path given as an
         argument or at the prompt), and lists them: name, program, date, scans (excluded), size.
      2. You pick rooms (`2`, `1,3`, `all`), then for each one which scans to load (Enter = all,
         or `1-12,15`). Only those are read, which keeps memory low.
      3. For each room, choose what to do: **view** (13c's "room so far" viewer: C colour by scan,
         click to select, X exclude or include), **build `.obj`** (13c's E: TSDF + surface net +
         clean model, plus `far_landmarks` for a 13d room), **build `.ply`**, or both builds.
         Builds run without a window, so several rooms can be done in one go.
      4. Outputs go into the room's own folder, with the `# first camera turn:` line, so Demo 4
         works. Settings at the top (`VoxelMeters`, `MaxVoxels`) can be finer than live, since
         time doesn't matter offline.
      - A copy of 13c's export and viewer code, minus the Kinect and placing (one file per example).
        Each room keeps its own coordinates.
      - Open questions for the user: (a) should 13z also merge folders from separate sessions of
        the same space (placing one session against another, with 13c's features + ICP)? (b) should
        13c/13d auto-save after every placed scan (the blue screen, D25), or K only? (c) should 13z
        offer P (re-place a scan) using the saved features?
- [ ] **14 Textured room** (planned 2026-11-04, after 13z; the user said not to start it yet).
      Option C. A 13z folder in; `.obj` + `.mtl` + a texture atlas `.png` out. Each triangle takes
      its colour from the scan that saw it most head-on and nearest (normal vs view direction,
      distance), from the full 1920×1080 pictures via 13d's colour camera model, checked against
      that scan's depth so hidden surfaces aren't painted. Then seams blended and exposure evened
      between scans. About 4× sharper colour than the registered 512×424 picture.

### Stage D — the goal

- [ ] **15 Objects in the map:** the tracked objects (05/06) placed in the room model from 13.
      A first cut for people exists: presentation Demo 4 (D22) puts depth blobs into 13's room,
      located from the camera turn 13 saves. Still to do: YOLO names and 05's tracks instead of blobs.
      (Was 14; renumbered 2026-11-04 when the textured room took 14.)

### Stage E — moving camera (later, optional; docs/ideas.md, "Motion compensation")

- [ ] **16 Visual odometry:** the camera's own motion, frame to frame (11's ICP on consecutive
      frames). (Was 15.)
- [ ] **17 Live mapping:** a coarse CPU KinectFusion, or decide to adopt RTAB-Map instead. (Was 16.)

### Housekeeping

- [ ] **libusbK blue screen (D25, the user chose: document only for now).** If it happens again:
      read `C:\Windows\MEMORY.DMP` with WinDbg (`!analyze -v`), stop or uninstall Microsoft's
      Kinect runtime (it runs beside libusbK), then consider UsbDk (D10's alternative).
- [ ] Decide whether to retire `exercises/csharp/`. Then the Microsoft Kinect SDK (KinectService)
      can be uninstalled from the dev PC.

Toolchain on the dev machine: VS Build Tools 2026 with the C++ workload, and .NET 10. Building C++
needs `vcvars64.bat`, which puts `cmake`, `ninja` and `vcpkg` on PATH. In Git Bash, run the built
.exe through cmd or PowerShell, because Git Bash's loader fails on `api-ms-win-crt-*`.
