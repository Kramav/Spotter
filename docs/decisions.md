# Design decisions

Append-only. Each entry records the date, the decision, why, and its consequences. When a
decision changes, add a new entry and mark the old one superseded.

---

### D1 — 2026-10-15 — First Kinect examples in C# on the Microsoft Kinect SDK 2.0
**Status:** superseded by D5 for the long-term track. The examples are kept as a reference.
**Why:** the guide is Python-only, but the task excluded Python. The SDK's first-class languages
are C# and C++, and C# reads closest to the guide's Python. The SDK also handles depth-to-colour
registration (`CoordinateMapper`). Independence from Microsoft was not a goal at the time.

### D2 — 2026-10-15 — OpenCvSharp4 + OpenCvSharp4.runtime.win, not OpenCvSharp4.Windows
**Why:** OpenCvSharp4.Windows pulls in WpfExtensions, which raises the NU1701 warning.

### D3 — 2026-10-15 — YOLO26n via ONNX in OpenCV DNN, exported with `nms=False`
**Why:** no Python or Ultralytics at runtime; OpenCV is already a dependency. With `nms=False` the
model outputs 1x300x6 boxes that are already final (x1, y1, x2, y2, conf, cls). The default export
produced [1,84,8400], which needs NMS written by hand. About 155 ms per frame on CPU.

### D4 — 2026-10-15 — Reference Microsoft.Kinect.dll from GAC_64
**Why:** the copies in NuGet and the SDK's Assemblies folder are reference-only and crash at
runtime. Applies to the C# examples only.

### D5 — 2026-10-25 — Move off the Microsoft Kinect SDK to libfreenect2
**Why:** Linux is the long-term target, and the user does not want the Microsoft driver.
libfreenect2 is open source (Apache-2.0 / GPL-2.0), runs on Linux, Windows and macOS, and
provides depth, colour, IR, registration and the factory calibration read from the device.
**Consequences:**
- No body tracking. Skeleton tracking is Microsoft-only, so that idea is dropped.
- On Windows, the USB driver is replaced with UsbDk or libusbK, and the C# examples stop working
  while it is installed.
- Upstream has been frozen since March 2020 (v0.2.1, 169 open issues). This is accepted because
  the hardware is frozen too.
- The existing C# wrappers (freenect2.net, libfreenect2.Net) were abandoned around 2018 and are not
  used.

### D6 — 2026-10-25 — No checkerboard calibration of our own for now
**Why:** libfreenect2 reads the factory intrinsics and the depth-to-colour calibration from the
device, and its `Registration` class does the alignment.
**Revisit if:** the tape-measure tests (guide §7) show errors larger than the guide expects.

### D7 — 2026-10-25 — Develop on Windows now; the code must also build on Linux
**Consequences:** CMake (or `dotnet`) only, and no Windows-only APIs in the new track. Each example
is checked to build on both before it counts as done.

### D8 — 2026-10-25 — C++ for the libfreenect2 track
**Why:** libfreenect2 and OpenCV are native C++, so there's no glue code and a single CMake build
works on Windows and Linux. A C# version would need our own C++ shim plus P/Invoke, which means
two languages and two builds. The new code lives in `exercises/cpp/`.

### D9 — 2026-10-25 — Dependencies: vcpkg manifest for libraries, libfreenect2 built from source
**Why:** libfreenect2 has no vcpkg port, so it is cloned into `exercises/cpp/third_party/`
(gitignored) and built with CPU depth decoding only, which avoids OpenGL, OpenCL and CUDA. Its
dependencies (libusb, libjpeg-turbo) and OpenCV (dnn and highgui only) come from
`exercises/cpp/vcpkg.json`. vcpkg ships with VS Build Tools and also runs on Linux, so one manifest
covers both platforms. On Linux, apt packages work too.
**Pinned:** libfreenect2 at commit fd64c5d (2020-03-01, the last commit on master, newer than the
v0.2.1 tag). The vcpkg baseline is fixed in `vcpkg.json`.
**Revisit if:** CPU depth decoding is too slow. libfreenect2 can use OpenCL or OpenGL instead.

### D10 — 2026-10-25 — Windows USB driver: libusbK via Zadig, not UsbDk
**Why:** libusbK replaces the Microsoft driver for the Kinect, which matches D5. UsbDk leaves the
Microsoft driver installed and takes over the device while UsbDk itself is installed. Both work
with libfreenect2 (it requests the UsbDk backend first). Install only one of the two.
**Undo:** in Device Manager, uninstall "Xbox NUI Sensor (Composite Parent)" and tick "Delete the
driver software".

### D11 — 2026-10-25 — Parallelize libfreenect2's CPU depth decoder with OpenMP; no GPU decoding
**Why:** the target is a Raspberry Pi, which has no usable OpenCL or CUDA for this, so libfreenect2's
GPU pipelines are out. The C# version reached 30 fps only because the Microsoft SDK decodes depth
on the GPU (it requires DirectX 11). libfreenect2's CPU decoder is single-threaded, but every pixel
is independent, so our patch (`exercises/cpp/libfreenect2-openmp.patch`, 24 lines) adds
`#pragma omp parallel for` to its five per-pixel passes. OpenMP is built into MSVC and GCC, so no
new dependency.
**Measured, Ryzen 5 3600 (6 cores / 12 threads), Release build:**

| | Filters on | Filters off |
|---|---|---|
| Single-threaded | 7.5 fps | 15 fps |
| OpenMP | 30 fps | 30 fps |

Depth readings were unchanged by the patch (crosshair about 2,930–2,950 mm in both builds).
**Caveat:** with the filters off, libfreenect2 also skips its 0.5–4.5 m clipping, so junk values
come through.
**Revisit if:** the Pi can't keep up even with the filters off. Options then: decode every other
frame, or simplify the decoder further.

### D12 — 2026-10-25 — Smooth distance per tracked object, not per pixel
**Why:** averaging each pixel over time only works when the camera and the scene are still.
Spotter cares about each object's distance, so we track each detected object and filter its
measured distance over time instead. That still works with a moving camera, and a Kalman filter
also gives the object's speed. Plain Gaussian blur on depth was rejected: it smears objects into
the background at edges (more flying pixels) and averages in the 0 values used for "no reading".
**Plan:** example 04 (see Roadmap). A thresholded per-pixel average (reset when the change is
large) stays a possible small setting for example 01, useful only for a fixed camera.

### D13 — 2026-10-25 — OpenCV must keep vcpkg's `intrinsics` and `thread` features
**Why:** D9 turned off vcpkg's default OpenCV features to keep the build small, and that dropped
CPU vector instructions (`CV_ENABLE_INTRINSICS`) and multithreading. YOLO detection in example 02
took about 650 ms. With both turned back on it takes 130–156 ms, the same as the C# version's
OpenCvSharp (about 155 ms). On the Pi, `intrinsics` is what enables ARM NEON, so it matters even
more there.
**Rule:** when trimming OpenCV features, keep `intrinsics` and `thread`.

### D14 — 2026-10-25 — Reproducible build: libfreenect2 as a vcpkg overlay port, plus CMake presets
**Supersedes** D9's hand-built `third_party/` approach. D9's choices (pinned commit, CPU only)
still stand.
**Why:** the repo must build the same thing from a fresh clone. Before this, OpenCV was pinned but
libfreenect2 took eight hand-typed commands. Now:
- `exercises/cpp/ports/libfreenect2/` holds our vcpkg recipe: upstream commit fd64c5d, checked by
  SHA512, with `openmp.patch` applied. It disables pkg-config so the libusb paths we pass are used,
  and installs its own relocatable `freenect2Config.cmake` (upstream's has absolute paths baked in).
- `vcpkg.json` lists `libfreenect2` and `opencv4`, and its `vcpkg-configuration.overlay-ports`
  points at `./ports`.
- `CMakePresets.json` holds the configure options. A fresh clone needs only `cmake --preset default`
  and `cmake --build --preset default`.
- vcpkg copies every DLL next to the .exe, so the hand-maintained copy list in CMakeLists is gone.

**Verified 2026-10-25:** only the files git would commit were copied into an empty folder, then
the two commands were run. It built, all DLLs were copied, and 01 and 02 ran on the sensor.
**Not pinned:** compiler and CMake versions (any VS 2026 Build Tools / CMake 3.20+), and the vcpkg
tool version. The port versions are pinned by the baseline.
**Gotchas:**
- On Windows, clone to a short path: vcpkg's nested build folders hit the 260-character limit.
- Linux is untested: it needs `libudev-dev`, and on a Pi the first OpenCV + GTK build takes hours.
- The first build anywhere compiles from source (about 30 min on the PC). A shared vcpkg binary
  cache would avoid that, but isn't needed yet.

### D15 — 2026-10-25 — OpenMP `schedule(dynamic, 8)` in the depth decoder patch; colour exposure cap as a setting
**Why (scheduling):** with colour on, depth fell to about 15 fps. libfreenect2 drops ("loses") any
depth packet that arrives while the decoder is still busy, so decoding must take under 33 ms. With
OpenMP's default static split (equal fixed slices of rows), one thread sharing its core with the
JPEG decoder or YOLO held up all the others. Dynamic scheduling hands out 8-row chunks to whichever
thread is free.
**Measured (colour on, no other work):** depth 14.4 → 27.4 fps; colour 27–29 fps.
**Why (exposure):** in a dim room, auto exposure holds the colour camera at 12–15 fps.
`setColorSemiAutoExposure(16)` gives about 29 fps, at the cost of a darker, grainier picture.
05c has a `CapColorExposure` setting (default true): its display went from 15 to 24–27 fps.
**Trade-off found:** capped exposure plus the 320 model detected nothing in a dim room (640 still
found 4 objects). Detection quality, not just speed, has to be checked on the Pi.

### D16 — 2026-10-30 — Removing people: depth blobs decide, YOLO boxes label, carried objects stay
**What:** 07d removes a person by painting their pixels with a remembered picture of the room.
Which pixels: a depth-foreground blob (07c) that lies mostly inside YOLO "person" boxes, minus
the pixels inside any other detected object's box.
**Why blobs, not boxes:** YOLO runs about 6 times a second on its own thread, so its boxes lag a
moving person. The depth blob is fresh every frame and follows the outline; the box only labels it.
**Why carried objects stay:** the user's requirement: a cup picked up and carried must not vanish
with the person. The cost: the fingers holding it stay visible too, inside its box.
**Why the colour memory learns only away from person boxes:** so it keeps what was behind a person
who stands still, and a person is never absorbed into the room.
**Rejected for now:** YOLO-seg masks (cleaner edges, too heavy for every frame on the Pi);
deep inpainting such as LaMa (too heavy). Never-seen pixels show magenta until 07e.

### D17 — 2026-11-01 — Object scans: mesh before segmentation, OBJ with vertex colours, guessed back kept apart
**What:** 08b turns a clicked YOLO box into an `.obj`: a 30-frame average, a box + depth band +
support plane cut-out, grid triangles, and a guessed back.
**Why the mesh before YOLO-seg:** the seg model is the risky part (OpenCV 4.14 DNN failed to load
our dynamic-size export before), while the box cut works today. The mask can replace the cut later
(08c) without touching the meshing.
**Why OBJ with vertex colours:** plain text, opens in Blender, MeshLab and CloudCompare, and needs
no texture image. A texture would mean saving a colour picture, which the privacy rule avoids.
**Why the person class is refused:** a coloured 3D scan of a person is an image of a person.
**Why the back is guessed but kept apart:** the user asked for a guess at the hidden side. It's grey
and in its own group (`guessed_back`), so seen and made-up surfaces never mix, the same rule as
07d's magenta. Mirror ("as deep as it is wide") was the simplest guess that closes the model.
**Support plane:** RANSAC (09) over the object's own pixels, keeping only planes with few points
below them, so a chair seat (legs below) isn't taken for the floor.

### D18 — 2026-11-02 — Walls on the radar: repeated RANSAC for upright planes, one connected piece each
**What:** 06b finds walls as the biggest upright planes in a sparse depth grid: RANSAC (09), take
the plane's biggest connected piece away, search again (up to 4). Too short (under 0.8 m of height)
counts as furniture. Each wall is drawn from above as one line.
**Why planes, not lines in a top-down picture (Hough):** 09's RANSAC is already in the project,
and a plane also gives the wall's height, which tells a wall from a couch back.
**Why the biggest connected piece:** a plane slices thin strips off the floor, the ceiling and
other walls. On the first try a couch front's plane picked up a ceiling strip and passed as 2.4 m
tall.
**Why "upright" is judged by the camera, not the floor:** like 06, it assumes the camera is
roughly level. Taking 09's floor would fix tilt, at the cost of one more idea per example; worth
it when the camera is tilted on purpose.

### D19 — 2026-11-02 — Wall filter: an extended Kalman filter per wall, state as a point on the floor
**What:** 06c gives each wall from 06b its own EKF. The state is the wall's foot (its nearest
point to the camera) as X, Z in metres, plus how far along the wall each end is. The measurement is
the plane's distance, the angle to the foot, and the two ends. Walls are matched frame to frame by
distance and angle (as 05 matched boxes), drawn after 3 sightings, forgotten after 30 misses.
**Why an EKF (the user asked for one), and why the state is a point:** with a fixed camera, a plain
Kalman filter on (distance, angle) would do almost the same. The foot as X, Z is what can be moved
when the camera moves (stage E: every foot shifts by the camera's own step). The measurement then
stays a distance and an angle, which is the nonlinear part an EKF exists for.
**Why the plane's distance, not the radar line's:** 06b's line sits at distance ÷ cos(tilt), so a
camera tilted 14° read 3.255 m against the plane's 3.16 m. The filter takes the plane's, so a
tilted camera no longer pushes the walls out.
**Why a fixed match gate in metres and degrees, not the filter's own (Mahalanobis) gate:** easier
to read and tune; revisit if walls swap IDs.

### D20 — 2026-11-02 — Room recreation: one view first (raw + clean), then automatic poses like Meshroom
**What:** the user wants a room recreated from a scan: one view first, then several spots aligned
automatically, "like Meshroom", and both a raw and a clean result. 09b meshes the whole view; 09c
adds a clean model (planes and boxes) in separate OBJ groups. Several spots: 11 ICP refines a pose,
11b finds the rough pose itself from colour features, 13 merges.
**Why both outputs in one file, in groups:** the same rule as 08b's guessed back: what was seen and
what was made up stay apart, and Blender/MeshLab hide a group with one click.
**Why the clean model is planes + boxes:** walls, floor and ceiling come straight from 06b's and
09's RANSAC; boxes are the simplest furniture that stands on a floor. Boxes line up with the
biggest wall (most furniture does) and are partly guessed (only the front is seen).
**Why up can come from the ceiling or two walls:** the user's Kinect sits on the floor looking up,
where the floor is a sliver or not seen at all.
**Why a jump threshold as a share of the distance (09b), not 08b's fixed mm:** neighbouring pixels
are farther apart far away, so a slanted wall at 4 m steps about 3 cm per pixel.
**Why 11b (features) before 12 (ArUco) for several spots:** Meshroom finds poses from matching
image features, with no markers; with depth it's simpler (3D matches, then RANSAC + SVD).
`features2d` is already in our OpenCV build; ArUco would add `calib3d` and `objdetect`.
12 stays as a fallback for blank walls with nothing to match.

### D21 — 2026-11-02 — Room scan (13): point-to-plane ICP that leaves bare-wall directions alone, TSDF + surface nets, scans in memory only
**What:** the user asked for the room recreation "built out fully". 11 (ICP), 11b (feature pose)
and 13 (any number of scans, placed automatically, merged and saved) were built in one go, still
as a ladder of single-idea examples.
**Why 13's ICP differs from 11's:** the synthetic test showed it. Point-to-point ICP on a sparse
sample (every 4th pixel, about 3 cm apart at 3 m) slid 3.4 cm along a wall, because no point can
get closer to its neighbour than the sample spacing. Point-to-plane (the gap along the surface
normal) fixed that, but on a view of one wall plus floor and ceiling it slid 21 cm, because
nothing pins where along the wall the camera is. So the 6x6 equations are linearised about the
pairs' centre (turns and shifts stay apart) and solved per eigen-direction; a direction pinned by
under 2% of the pairs keeps the features' value. With features within 1 cm, ICP also starts at a
5 cm gate, not 11's 20 cm. Result: 0.2–2.1 mm, under 0.1°. 11 keeps point-to-point: it's the
textbook version, and its experiment 3 shows the weakness.
**Why TSDF + surface nets, not marching cubes:** the same TSDF as KinectFusion (ideas.md), and
surface nets need no 256-case lookup table: one point per cube where the sign changes, a square
per edge that crosses. About 80 readable lines.
**Why ICP against all earlier scans, not just the previous one:** errors don't pile up scan by
scan; it costs one bigger k-d tree. Loop closure (a pose graph) is left for later.
**Why the scans live in memory only:** colour pictures are images of the room and maybe of people
(privacy rule). Only the final `.obj` is written, as with 08b and 09b. *(Superseded by D29, 2026-11-04: K now saves the scans on request.)*
**Why refuse scans with under 100 features:** a dark room gives a black colour picture (seen on the
sensor: brightness 5/255) and 0 features; no later scan could ever be matched to such a scan.

### D22 — 2026-11-03 — Presentation demos: a separate folder of remixes; Demo 4 locates itself from a turn that 13 saves
**What:** the user asked for "the most visually impressive things" for a small exhibition, then
for all of them in a `presentation` folder. Five demos: 08 → You in 3D, 07d → Invisibility Cloak,
06c → Booth Radar, 13 → Room Twin (new), 03 → How Tall.
**Why a separate folder, not more ladder examples:** the demos teach nothing new, except Demo 4. They're
fullscreen copies of examples with the save keys removed, and the ladder stays one idea per step.
Each is still a single file with the usual sections, so a visitor who asks "how" can be shown it.
**Why no save keys, and "live only" on screen:** at a stand, every frame has strangers in it
(privacy rule: never save images of people).
**Why Demo 4 starts from a turn saved by 13:** two other ways were tried first, in a synthetic bare room:
- *54 starting guesses* (every 20° of turn × 3 tilts) with 11's point-to-point ICP: 4–6 s, and wrong
  in most cases. A box-shaped room looks alike turned 90°, and a view of one wall can slide along it.
- *A pull towards scan 1's spot* (extra ICP pairs): it also dragged directions that the room does
  pin down (5 cm off where the truth was 5 cm away).

13 already knows how scan 1's camera sat in the room (its levelling turn). It now writes that as
an OBJ comment (`# first camera turn:`, ignored by Blender and MeshLab), and Demo 4 starts there
with 13's point-to-plane ICP (gate 0.50 m). Result: found within 0.2 mm from up to 15° and 58 cm
off, in 0.3 s. Past that it's lost and says so (38–40% matched, warning under 60%). 11's
point-to-point ICP from the same start failed at 5° and 11 cm: again D21's lesson.
**Why Demo 4's visitors are depth blobs (03), not YOLO tracks:** it works in the dark, and needs no
second thread. Naming what's in the room (YOLO, 05's tracks) is roadmap step 14 proper.
**Why Demo 5 measures height from each end's own depth:** 03's box height × one depth is wrong
once the Kinect is tilted (the feet are nearer than the head). Measuring between the top and bottom
rows as 3D points holds at any tilt.

### D23 — 2026-11-03 — Demo 4's fallback lives in its own file
**What:** with no room file from 13, Demo 4 used to stop. It now calls `OneViewRoom.h` next to its
`main.cpp`: the Kinect's empty view, averaged over 15 frames, levelled by the Kinect's tilt
(`OneView::LevelDegrees`, 14 for the Kinect on the floor) and given normals, becomes the room.
**Why a second file, against the one-file rule:** the user asked for it to be a separate file, so the
fallback reads as its own idea and Demo 4's main file stays about the room twin. It's a header, so
CMake needs no change.
**Why it's only for rehearsing:** one view is 08's shell. Visitors appear inside the front of what
the Kinect sees, not inside a room seen from all sides.

### D24 — 2026-11-03 — 13b: CLAHE and a gentler FAST test for features; room scans trade time for quality
**What:** a copy of 13 that evens out the grey picture's contrast tile by tile (CLAHE) before ORB,
and lowers ORB's FAST corner threshold from 20 to 7. It also raises the settings that buy quality
with time.
**Why:** a probe of the real view, through 13's own code, found 48–73 features in a dim room
(brightness 57/255). 13 needs 100, so it refused the scan. Measured on that view: CLAHE alone gave
about 125, FAST 7 alone about 170, both together about 1250 (with up to 3000 allowed). A second
scan from the same spot was then placed 0.5 mm off, with 285 matches agreeing.
**Why only the features see the evened-out picture:** CLAHE changes the colours. The mesh keeps
the camera's true colours.
**Why quality over time:** the user's call ("we prefer high quality over quick time"). 60 frames
per scan (13: 30), 3000 features, 5000 RANSAC tries, ICP on every 2nd pixel with 50 rounds, 20
agreeing matches needed (fewer wrong placements with so many features), and 2 cm cubes with room
for 80M of them (about 1.9 GB: fine on the PC, not on a Pi). In the synthetic room: placements
1.4–1.9 s instead of 0.3–0.4 s, and the same accuracy. Its file has the same name as 13's, so
Demo 4 reads it too.
**Why a new example, not a change to 13:** the user asked for an iteration, and 13 stays the
quicker one to learn from.

### D25 — 2026-11-03 — A blue screen while scanning: written up, driver unchanged (the user's call)
**What happened:** at 16:16 during a long 13b scan (30+ scans), the PC blue-screened with bugcheck
**0xB8 ATTEMPTED_SWITCH_FROM_DPC**: a kernel driver tried to wait while running at a raised
interrupt level. Only a driver can do that; no program of ours can. The event log has it as
Kernel-Power 41 and WER 1001. A full dump is at `C:\Windows\MEMORY.DMP` (4.8 GB). No kernel
debugger is installed, so the driver isn't confirmed.
**What's on this PC (read from Windows, 2026-11-03):**
- `libusbK.sys` **3.0.7.0**, installed by Zadig on 2026-10-25 (D10). That's libusbK's long-standing
  build.
- The USB controllers are **AMD USB 3.10 eXtensible Host Controllers**. libfreenect2's README warns
  that it's picky about controllers.
- **Microsoft's Kinect runtime is still running** (`KinectService.exe`, `KinectMonitor.exe`), and
  Windows still lists a Microsoft "WDF KinectSensor Interface 0" under the Kinect, whose parent is
  on libusbK. So two driver stacks are on one device.

**Options, in the order to try them, if it happens again:**
1. Install WinDbg and run `!analyze -v` on `MEMORY.DMP`. It names the faulting driver.
2. Stop, then uninstall, Microsoft's Kinect runtime (already a Housekeeping item; the C# track is
   frozen).
3. Switch to **UsbDk**, D10's alternative. libfreenect2 tries it first. Roll libusbK back in
   Device Manager first: never both.
4. Turn off USB selective suspend (Power Options), and use a rear USB port (straight to the chipset).

**Decided:** the user chose to document only, for now. D10 stands.
**Related, but not the blue screen:** heavy CPU work can starve libfreenect2's USB and depth
threads, which shows up as lost packets and libusb errors, not a blue screen. 13c places scans on a
background thread with `cv::setNumThreads(cores - 2)`. Scans live only in memory (D21), so a crash
loses them: export (E) now and then during a long scan.

### D26 — 2026-11-03 — 13c: scan assist, constant speed, and an ICP drift fix that 13 and 13b need too
**What:** after a real 34-scan room with 13b, the user asked for a recommended next shot, a view of
which features matter for the next scan, a way to remove bad scans (scan 34 was misplaced), and a
fix for the slowdown from about scan 28.
**Why a shortlist vote, a cropped and thinned ICP target, and a background thread:** 13b matched
every new scan against every earlier one in full, and ran ICP against all their points (about
35k each), on the main loop. Now a single Hamming match against all descriptors stacked picks 4
candidates, and ICP gets only what the rough pose can see, one point per 1 cm cube. The work
follows the room, not the number of scans.
**The slowdown was mostly an ICP bug, found while timing:** on a synthetic 30-scan walk, scans
7–11 took 12–18 s and ended 12–21 cm off. Traced round by round: the features' pose was 2 mm off,
then each ICP round slid it 2.3 mm further along a direction pinned by only 2.4–2.7% of the pairs
(the bent normals where wall meets floor). That's just above the 2% cut-off from D21, so the tiny,
constant bias was amplified, and ICP never settled (50 rounds). Raised to 5% (`WeakShare`): worst
2.9 mm / 0.18° over all 30, about 3 s each. 13 and 13b keep 2% (their files are unchanged), and
are open to the same drift.
**Why the live anchors, not a distance rule:** a scan is placed when 20+ features match an earlier
scan and agree, plus 25% overlap for ICP. Distance doesn't matter; what's shared does. ORB stops
matching beyond about 30–45° of viewing angle or 3× distance. The green dots and counter show the
real condition while aiming.
**Why the second opinion:** a scan that matches a misplaced earlier scan inherits its error, and
ICP agrees, because that scan's points are right there. A quick ICP against every other scan,
with a 30 cm gate, catches it (synthetic: "the rest of the room disagrees by 15.0 cm"). It costs
about 1 s per scan, with no false alarms on 30 good scans.
**Why exclude, not delete:** it's reversible, and excluding scan 34 doesn't move scan 35 (poses
are in room coordinates). P places a scan again, against all the others.
**Why the next shot only turns from existing spots:** a suggested new position could be inside
furniture. It needs a free-space check first (`ponytail:` note in the code).

### D27 — 2026-11-03 — 13d: far features as rays, and landmarks for distance
**What:** the user's hallway kept getting dropped. Its walls are plain, and the doorway at the end is
past the depth range, so 13c had nothing to match. The user asked whether features outside the
point cloud could still help place scans, "as an out-of-bounds metric".
**Answer, built:** yes. A detail without depth still gives a direction.
- **The mask and model:** 13d runs ORB on the full colour picture where `bigdepth` reads nothing,
  inside the depth camera's field of view. Each feature becomes a ray. The colour camera's model
  (and its 5 cm offset) is fitted from each scan's own pixels that do have depth, so no hand
  calibration and no extra library are needed (calib3d is still not built).
- **Voting:** far matches vote in RANSAC. Two rays agree if they meet far away (over 4 m, within 0.5°).
- **Landmarks:** seen from two spots at least 2° apart, they're triangulated into landmarks, and
  later scans' rays to those pin position and turn together.
**The limit, measured:** rays carry no distance. And a far point isn't at infinity: stepping 0.6 m
sideways turns a 12 m doorway by 2.9°, which rays alone can't tell from a 2.9° turn. The first
version trusted the rays' turn and ended 2.9° and 0.6 m off. So, after a far-ray placement:
- **ICP starts with a 0.5 m gate.** In a hallway the walls, floor and ceiling pin sideways, height
  and every turn.
- **Landmarks decide the distance along it.** Synthetic hallway: 34 mm after a sideways step, and
  33 mm after a 2 m walk in, both of which 13c refuses.
- **With no landmarks yet, the distance stays unknown,** and the scan is flagged red. The guide
  says to step 0.5 m sideways at a hallway's start.

**Why landmarks in the .obj:** they place far things (the doorway) in the room model, about ±12 cm
in depth at 12 m from a 0.6 m step. That's good enough to place, not to model.
**Why not essential matrices or PnP:** they need `calib3d`, which isn't built. The two 3 × 3 least
squares (Wahba, and lines meeting at a point) cover this case in about 40 readable lines.

### D28 — 2026-11-03 — Room export: "up" from the biggest level plane, and furniture boxes capped
**What:** the user's 34-scan flat (13b) exported a model nobody could inspect. Read headlessly, its
merged mesh was fine (the ceiling flat to ±5 cm over 8 m), but the clean model was wrong:
- **"Up" was 28° off.** Up came from the *farthest* up-facing plane within 25° of the first
  camera's up: a slanted piece, 2.55 m away. The 29 m² ceiling was ignored, the room came out 4.4 m
  high, the walls were split into 21 fragments, and the tilted walls became "furniture".
- **One box covered the whole flat (8.3 × 9 × 4.4 m)**, because boxes that touch were joined, so
  they chained through the flat. Every viewer showed only that box.

**Fix (13, 13b, 13c, 13d):**
- Up comes from the level plane with the most points, floor or ceiling.
- The pieces are then sorted again against that up, within `LevelDegrees` (10°), so a ramp is
  neither floor nor wall.
- A box wider than `MaxBlockMeters` (3 m) is dropped, and touching boxes join only while they stay
  under that size.
- Synthetic test: the room plus an 18° ramp. Before the fix, the floor came out at −1.415 m (the
  ramp) and the room tilted. After it, the floor is at −0.998 m, the walls are exact and the ramp
  has its own box.

The flat's existing file can't be rebuilt, because scans live only in memory. A levelled copy with only the mesh was made from it instead.

### D29 — 2026-11-04 — Keeping the scans (K in 13c/13d), rebuilding later (13z), and 14 = textured room
**What:** the user found E's `.obj` loses colour and detail, and asked for the scans to be kept and
rebuilt later. Built: K in 13c and 13d writes a folder per room; 13z reads such folders back,
without the Kinect, and views or rebuilds them. Option C (a textured mesh) is planned as 14, so the
later examples shifted up one: 15 objects in the map, 16 visual odometry, 17 live mapping.
**Why the `.obj` looks blurred:** the merge keeps one colour per 2 cm corner (`VoxelMeters`),
averaged over every scan that saw it, and the triangles just blend between corners. At 2 m one depth
pixel is about 0.5 cm wide, so roughly 16 pixels' colours become one. K's `points.ply` skips the
merge: every scan's own points and colours, as the viewer shows them.
**Privacy (the user's decision):** saving colour frames was forbidden (Conventions, D21). Now allowed
for room scans only: the user said saving is fine, the folders go in `.gitignore`, and Claude never
opens or views them unless the user asks. So `room_scans/` is in the root `.gitignore`, and K also
writes a `.gitignore` of `*` into every folder, for folders saved elsewhere. Scan with nobody in view,
as before. Tests use a synthetic room.
**Where K saves:** the user asked to choose the place and the name. A console question (Enter = the
default) works the same on Windows and Linux. A file dialog would be Windows-only code (D7). The
windows wait while you type. An existing saved room is overwritten only after a yes, and a folder
that isn't a saved room is never touched.
**Folder format (format 1):** plain files, so other tools and 14 can read them. Per scan:
- `cloud.bin`: the averaged points as raw floats, so no camera maths is needed to read them back.
- `colour.bmp` and `colour_full.bmp`: the 512 × 424 and 1920 × 1080 pictures.
- `colour_index.bin`: for 14's colour camera model.
- `features.yml`: for re-placing later.
- `pose.txt`.

Per room: `room.txt` (both cameras' factory settings, the first camera turn), `points.ply`, and 13d's
`landmarks.txt`.
**Why BMP:** the first test failed, because `cv::imwrite` has no PNG writer here. `vcpkg.json` builds
OpenCV without its `png` and `jpeg` features, and the `libspng` and `libjpeg-turbo` that are
installed belong to libfreenect2. BMP is built into OpenCV and lossless, at about 11 MB per scan
instead of about 8. Adding `png` would mean changing the pinned build (D14) and recompiling OpenCV
on both PCs.
**Why 13z copies 13c's code:** one file per example (Conventions). 13z holds 13c's viewer and merge
code unchanged (assembled from 13c's own text), so a rebuilt `.obj` matches a live E.
**Tested (synthetic, scratchpad, not committed):** 5 raycast scans of a 4 × 2.5 × 5 m textured box
room with a couch, one excluded and one suspicious, were fed into the real 13c and 13d `KeepScans`,
then rebuilt by 13z:
- Floor and ceiling at −1.2996 / 1.2002 m (true −1.3 / 1.2), 4 walls, the couch one box.
- The mesh is 0.4 mm (median) and 1.8 mm (95%) from the true surfaces.
- 13d's landmark landed at the right place to the millimetre.
- K's `points.ply` and 13z's rebuilt one were identical.
- Loading a subset, a wrong answer, overwrite, refusal and cancel all behaved.
- The viewer was smoke-tested. Nothing has run on the sensor yet.

**Left for later (docs/ideas.md):** merging separate sessions of one space, P (re-place) in 13z, and
auto-save after each placed scan. *(Auto-save: built as 13e, D30, 2026-11-04.)*

### D30 — 2026-11-04 — The scans survive a lost Kinect (13c, 13d) and a crash (13e); every multi-scan example does both from now on
**What:** the user asked what happens if the Kinect is unplugged during a multi-scan session. Before
this, the main loop waited 10 s for a frame, printed "giving up" and quit, and every scan not yet
kept with K was lost. The user chose two of the fixes offered:
1. **The rescue, in 13c and 13d:** after the 10 s timeout, everything is written to
   `room_scans\rescue_<date>_<time>` with K's own code and no questions asked (nobody may be at the
   console), *before* `device->stop()`. A scan still being placed is waited for and kept too.
2. **Auto-save, in a new 13e (13d plus saving as you go):** the folder is asked once at the start,
   and every scan is written the moment it is placed.

The user also called 13e our most advanced implementation, and set a rule: **every future
multi-scan example has both** (Conventions in CLAUDE.md).
**Why stop the Kinect last:** stopping a device that has gone away might hang in libfreenect2; this
hasn't been tested. With the save done first, a hang costs nothing but closing the window.
**Why the folder comes first in 13e:** auto-save needs to know where to write before scan 1. A
question at the first SPACE would interrupt the scanning. A question at start-up comes before the
windows open, so nothing waits on it.
**Why 13e writes the same files as K:** 13z, and 14 later, then read any of the three without
changes (format 1). Each event updates the folder:
- a placed scan writes its `scan_NNN` folder;
- X and P rewrite that scan's `pose.txt`;
- U deletes the scan's folder;
- `room.txt` and `landmarks.txt` are rewritten after every change.

`points.ply` and the levelled `first_camera_turn` need the clean model's planes and about 75 MB of
writing for 30 scans, so only K, Q and a lost Kinect write them. `room.txt` says `finished 0/1`.
13z never needed either: it works out both again.
**Crash safety:** a scan's `pose.txt` is written last, and 13z skips a scan without one. Files that
get rewritten (`room.txt`, `pose.txt`, `landmarks.txt`, `points.ply`) are written beside the old
copy, then renamed over it, so a crash leaves the old copy, never half of the new one. What a crash
can still lose: the scan being held still for, or the one being placed. A blue screen might also
lose the last second or two that Windows hadn't written to disk yet.
**Writes on the main thread:** a scan's 11 MB took 0.027 s in the synthetic test. A background
writer would need its own copy of the scan, and rules about which job writes first. That isn't worth
it for a pause that short, once per scan.
**Q quits with an empty room:** if nothing was scanned (or all of it was undone), 13e removes only
the files it wrote, then the folder only if it is empty. Anything else in the folder stays.
**Tested (synthetic, scratchpad):** the same 5-scan box room as D29.
- The rescue: 13c's and 13d's real `KeepScans` with `RescueFolder()` asked nothing, wrote
  `rescue_<time>` folders, and 13z listed and rebuilt them.
- 13e: its real `StartRoom`, `WriteScan`, `WritePose` (X), U's folder removal and `FinishRoom`, in
  the order the main loop calls them. 0.027 s per scan, no `.new` files left.
- 13e's `points.ply` was byte-identical to 13d's rescue copy.
- 13z's rebuild matched D29's: floor and ceiling −1.2996 / 1.2002 m, 4 walls, the couch one box,
  mesh 0.4 / 1.8 mm.
- A room "crashed" during scan 3 (its folder holding a short `cloud.bin` and no `pose.txt`): 13z
  loaded scans 1–2 and skipped scan 3.
- Not tested: the timeout path itself (that needs the Kinect unplugged for real), and 13e on the
  sensor.

### D31 — 2026-11-04 — Demo 6: Demo 4 with the holes in flat surfaces filled by software, and marked
*(Its wall finding is superseded by D32, after the first real run.)*
**What:** the user wanted Demo 4 without "massive holes" (walls and tables), as a new Demo 6 that
shows which areas the software reconstructed. Demo 6 is a copy of Demo 4 with one new step, run
once at start-up:
- find the flat surfaces in the room's points: upright planes by RANSAC, where a point and its
  normal propose the plane; level ones as crowded heights of up-facing points;
- lay a 3 cm grid on each one, and fill the cells no room point is near;
- colour the filled cells with `cv::inpaint`, from the seen cells around.

The floor fills to its outline, a wall from the floor to the doll's-house cut, and a table top to
its rectangle when the piece covers at least half of it (otherwise only its enclosed holes).
G cycles marked (blue stripes) / unmarked / off. The keys are listed along the bottom of the
screen, tersely (the user's ask; now the rule for demos).

**Why mark the fill:** every filled cell is a guess. The point of the demo is that visitors can
see real and guessed side by side, and G's "off" is Demo 4 itself.

**Why only flat surfaces:** a plane is the one shape a hole can be filled from honestly: its
edges pin it down. Chairs, plants, legs and what's under a table would be invented. Those are
ideas.md material (shape by class).

**Why points seen behind a wall keep it open:** without that, every doorway and window would be
filled shut. Where the first camera's line of sight to a point more than 10 cm behind the wall
crosses the wall, there's an opening. It uses scan 1's camera only, because the `.obj` doesn't
keep the other scans' spots. A doorway nothing was seen through still gets filled.

**Why the floor stops at walls:** the floor seen through a doorway is the next room's. Its convex
outline would otherwise spill out past the wall.

**Why a demo and not 07e:** 07e (fill never-seen pixels in one picture) stays on the ladder. This
does its plane-fill half on the room instead, for the exhibition. It reuses Demo 4's fallback by
including `../Demo4_Room_Twin/OneViewRoom.h` (one copy, D23).

**Tests (synthetic, through the real `FillHoles`):** a 4 × 5 m room, 155k points:
- the couch shadow on the wall was filled 1.27 m² (hole 1.26);
- the never-seen floor patch 0.41 of 0.42 m²;
- the table top under a box 0.035 of 0.040 m²;
- nothing was filled between the floor and the table top, and nothing in the doorway;
- every filled point was within 0.4 mm of a true plane;
- 0.07 s.

Not run on the sensor. Like Demo 4, it inherits the left-right mirror (Housekeeping).

### D32 — 2026-11-04 — Demo 6, round 2: walls traced from above, a slab for "seen", legs and dark things from the readings
**What went wrong on the first real room** (a 13e `.obj`, the user's screenshots):
- blue was painted over a wall that had clearly been seen;
- a blue sheet floated in front of the table, from the floor to the cut;
- the table got no fill;
- the unseen floor stayed black.

The causes:
- the plane's tilt came from one point's noisy normal and was never refitted;
- a merged wall is slightly bent or doubled by scan drift, and one flat plane can't follow that;
- "seen" was a 5 cm slab, so where the plane stood more than 5 cm in front of the real wall,
  the wall counted as unseen and the fill covered it;
- anything upright with 1.5 m of height range counted as a wall (a chair back, a table edge and
  a shelf in a line);
- a 2 cm height histogram cut a table top tilted 1–2° into strips.

**The user's calls:**
- a simple layout for now: walls exactly upright, surfaces level;
- but walls may curve along the floor;
- table tops filled, and legs only from what the depth camera read of them;
- the "negative space" on walls used to guess at things that sent no reflection back.

**What changed:**
- **Walls are paths seen from above.**
  - Upright points that stack at least 1.2 m high in one 3 cm cell (and at 15 points per metre
    or more, so stray points don't count) are wall cells.
  - These are thinned to one cell across the wall's thickness, placed at their points' mean.
  - Gaps up to 1.2 m between two ends pointing at each other are bridged (doorways).
  - The paths are split where three meet, then ordered by the shortest walk along them and
    unrolled into a flat picture (along × height), which handles straight, bent and curved walls
    alike.
  - A wall must be at least 1 m long and 40% seen, and not seen only through another wall.
- **"Seen" is a slab:** a real point from 5 cm in front of the wall to 50 cm behind it (10 cm for a
  cabinet front or a table top). A wall drawn a little in front of the real one can't hide it.
  Points further behind are still openings.
- **Cabinet fronts** are RANSAC planes refitted by a 2D least-squares line (PCA, as 13's
  `FitPlane`).
- **Level surfaces:**
  - gathered ±4 cm around each crowded height;
  - their height fitted per 50 cm tile;
  - pieces up to 10 cm apart are joined (a top cut by a shadow).
- **The floor nobody saw** is guessed at the walls' foot (their lowest readings), inside the walls
  (seen from the first camera, no wall in the way).
- **Legs:** under a top, a blob of cells whose own readings stack up (a flat seat doesn't), at
  most 12 cm across, spanning 15% of the floor-to-top gap and crossing its middle height, with no
  seat on it. It becomes a post from the floor to the top. Only unread heights are filled.
  - *Why "crossing the middle", not "reaching high up":* the table top hides the far legs' upper
    part from a camera above it, a chair hides the near legs' lower part, and the sides of a box
    standing on the table don't reach down that far.
- **Dark things (step 1c):** a wall cell that is unseen, inside the first camera's view and within
  4.5 m, with no reading on that camera's line of sight in front of it, is a shadow nothing seen
  explains: something black, glossy or see-through sent no depth back.
  - Pieces of at least 0.03 m² become purple cut-outs.
  - Each is placed at the median floor distance of the readings just below it and to its sides
    (8° down, 2° sideways, not wall or floor): a dark chair's legs, a TV's stand. With none, it
    goes on the wall.
  - *Why mostly downward:* a dark object's readable parts are usually what it stands on, and a wide
    cone picked up a normal chair next to the TV.
  - *Why floor distance, not straight-line distance:* the legs are lower than the seat, so their
    straight-line distance from the camera is longer.

**Tests (synthetic, scratchpad, through the real `FillHoles`; every point hidden from the first
camera by a box dropped):**

| Case | Result |
|---|---|
| Bent, doubled and curved walls | no filled point within 1.5 cm of a real one; the curve followed to 3.1 mm; the doubled wall's hole one layer (117 points, about 106 expected) |
| Table top tilted 1.5° | found, the hole under its box 0.032 of 0.040 m² filled, within 7 mm of the top |
| 4 legs seen over only 13–22 cm each | all 4 rebuilt from the floor to the top |
| A chair back, a table edge and a shelf in a line | no sheet |
| The floor never seen | guessed at −1.20 m (true −1.20), nothing outside the walls |
| A black chair with only its legs read | ghost 11 cm from the legs' distance |
| A black TV | ghost on the wall |
| A normal chair | no ghost |

- Round 1's checks still pass.
- Each room took 0.14–0.19 s.
- Noise no longer makes walls.
- Known leftovers:
  - a chair pushed under a table, whose seat the table top hides, can have its legs taken for
    table legs;
  - a short post can appear at the corner of a box under a table.
- Not yet run on the real room again.
