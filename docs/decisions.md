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
*(How it finds the floor is superseded by D33, after it ran on the sensor.)*
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

### D33 — 2026-11-04 — Demo 6's floor from its own readings; Demos 4 and 6 locate from several starts
**What the user saw on the sensor** (Demo 6, the real 13e room, round 2):
- the filled floor sat too low, under the floor patches the scans had seen;
- the visitor showed up in the wrong spot. The Kinect was "roughly" where scan 1 was, not exactly
  there.

**The mirror is not the cause (investigated).** 13e builds the room with libfreenect2's
`getPointXYZ`, and Demos 4 and 6 place visitors with it too. The room and the live view are mirrored
the same way, so a visitor's place relative to the furniture is unaffected. The user's call: the
mirror fix stays in Housekeeping.

**The floor: why it was too low.** Every height in Demo 6 came from one number, the lowest 2% of
all room points.
- A Kinect at chest height sees the floor at a grazing angle, so the floor is a small share of a
  scan.
- Readings under it (flying pixels, multipath in the corners, a slightly misplaced scan) are enough
  to drag that mark below the real floor.
- The real floor was then looked for only within 15 cm of the mark. Missing that, the floor counted
  as never seen and was laid at the walls' lowest 0.2%, which is lower still.
- The walls were filled down to the mark too.

**What changed in the floor** (`FindFloor`, Demo 6):
- **The floor is the lowest level surface with 0.25 m² in one connected piece,** seen from above
  in 5 cm cells.
  - It's measured by footprint, not point count. Grazing-angle readings are sparse but wide; stray
    readings never make one big connected piece.
  - Its height is the median of that piece's points.
- **The walls' foot gates it.** The foot is the lowest 2 cm step whose upright readings number at
  least a third of a typical step up the walls; stray readings under the floor are far sparser. A
  level surface more than 25 cm above the foot is a table top or the ceiling, never the floor.
  - *Why:* the first try of the rule, in the test, took the ceiling for an unseen floor. A seen
    table would have gone the same way.
- **The floor never seen:** laid at the walls' foot.
- **Readings more than 10 cm under the floor** are noise: hidden in the doll's house and left out of
  the filling. Locating still uses every point.
- **The console says how it chose:** the floor line, with the old 2% mark beside it for
  comparison, and each level surface it passed over with why.

**Locating: why the visitor was in the wrong spot.** Locating ran one ICP from one start: scan 1's
turn, at the room's origin. From about 15° or 30 cm off, it settled on a wrong fit. In the test,
that fit matched 46–64% of the view, sometimes over the 60% warning line, so nothing warned.

**What changed in locating** (`Locate`, Demos 4 and 6, the same code):
- **ICP is tried from up to 63 starts:** scan 1's turn and 0, ±10, ±20 or ±30° about the room's up,
  each at scan 1's spot or 0.4 m away (the 8 neighbours).
- **The search:** each start runs on a quarter of the view, and the best is refined on all of it.
- **Nearest start first.** The search stops at a start that matches 95%, so a Kinect put back on the
  spot takes one try. In the test, wrong fits reached at most 87%.
- **The console lists the best three distinct fits** (match, turn, shift from scan 1's spot). It
  warns when two fits far apart match almost alike: the room looks the same from both.

**Tests (synthetic, scratchpad, through the real `FindFloor`, `FillHoles`, `Icp` and `Locate`):**

| Case | Result |
|---|---|
| A far floor strip (about 3% of points), 4% stray readings 15–40 cm under it, wall readings dribbling 20 cm below | old mark 27 cm too low (outside the old 15 cm search); found at −1.200 m (true −1.20), no fill under the floor |
| As above, plus a 1.9 m² table top 0.45 m up (the seen floor: 1.4 m²) | the floor still found at −1.200 m |
| The floor never seen, with stray readings and dribble | guessed at −1.209 m; the ceiling rejected (2.61 m above the walls' foot) |
| The floor never seen, plus the table | guessed at −1.209 m; the table rejected (0.46 m above the foot) |
| Kinect put back 0°/0 m, 15°/0.36 m, 25°/0.50 m, −30°/0.64 m, 20°/0.50 m, −20°/0.60 m off, tilted 3° | old single start: from 15° on, 0.29–0.91 m and 15–31° off. New: all within 1 mm and 0.04°, in 0.4–5.4 s (1 start when on the spot) |

- Demo 4 gives the same locating results.
- D31's and D32's checks still pass, with the same numbers.
- Not yet run on the sensor.

### D34 — 2026-11-05 — The examples move to `legacy/`; new ones share a `spotter` CMake library
**What:** after an adversarial review ([review-2026-11-05.md](review-2026-11-05.md)), the user
chose to re-implement the examples and demos themselves on shared classes:
- **Moved, not deleted.** The 28 example folders and `presentation/` are now under
  `exercises/cpp/legacy/` (`git mv`, history kept). `CMakeLists.txt` stays where it was, so
  `SPOTTER_REPO_ROOT` and every `.exe` name are unchanged, and all 34 targets still build. 13e is
  still the scanner for real rooms. Older entries keep their old paths.
- **A real CMake library** (the user's choice over header-only files or keeping one file each):
  `exercises/cpp/spotter/`, with `include/spotter/*.hpp` and `src/*.cpp`, built by
  `add_library(spotter STATIC …)`. Each example still keeps its own new idea inline, in the usual
  sections, and calls the library for the rest.
- **External dependencies show in the file name.** Core modules (`geometry`, `depth`, `track`,
  `register`, `room`, `mesh`, `view`, `detection.hpp`) use only OpenCV core types. Anything tied
  to a sensor, a model or a backend is an adapter named `<role>_<dependency>`:
  `source_freenect2`, `detector_yolo_opencvdnn`, `features_orb`. Swapping one means adding a
  sibling file (e.g. `detector_yolo_ncnn`).
- **A different model is a settings change, not a code change** (the user may train their own):
  `YoloDetector::Options {modelPath, inputSize, namesPath, layout}`. The class names come from a
  text file next to the model (one per line, `yolo26n.names`), because OpenCV DNN can't read
  Ultralytics' ONNX metadata. `layout` is `EndToEnd` (today's 1x300x6) or `Classic`
  (1x(4+classes)x8400 + NMS). Person is the class named "person", not id 0.

**Why:** the copies drift, and the drift causes bugs. Two examples:
- The ICP weak cut-off fix (D26) never reached 13 and 13b.
- Demo 6's floor fix (D33) never reached Demo 4.

The same Kinect setup is written 34 times, with three different depth cut-offs. And D29's promise
that a 13z rebuild matches a live E is broken by two different `sparse` loops.

**Not done yet:** no virtual `Detector` base class. A second backend copies the same `detect()`
signature, and the base class waits until an example needs to choose a backend at run time.

**Supersedes:** the one-file-per-example rule (Conventions, D29) for shared code only. Examples
are still one file each, plus the library.

### D35 — 2026-11-06 — The examples are re-written in `examples/`, on the library, with the mirror fixed

The user asked for every example re-written on the `spotter` classes in a new folder, with tour
programs showing what each class does, and a short, readable `main()`: the hand-off point for a
software team.
- **Where:** `exercises/cpp/examples/`. Folders drop `Kinect_` (the camera sits behind
  `source_freenect2`). Targets are `ex_<folder>`, built as `build/examples/<folder>.exe`, so the
  names never clash with `legacy/`. `00_tour/` holds one program per module, most with no Kinect.
- **Scope (the user's choice):** every example and demo, except 13, 13b, 13c and 13d (13e contains
  them). 07b is a setting of 07 (`MaxDepthMeters`).
- **main() (the user's choice):** a "story main" of 15–40 lines. It sets up the library objects,
  then loops over named steps. Each step is a function above `main` in the same file, and the
  example's own idea lives in those functions. No globals.
- **The mirror is fixed (the user's choice)**, in one place: `KinectSource` calls `Unmirror` on
  every frame. It flips the pictures, remaps `colourIndex`, and sets cx to width − cx for both
  cameras. `check_frame` compares this against libfreenect2's own `Registration` with no device:
  every 3D point equals `getPointXYZ` with X negated. `KinectSource::Options::unmirror` stays as a
  knob, in case the sensor check says otherwise. Room folders written by the new 13e will be
  format 2, and old rooms are mirrored on load.
- **The rule for duplicates:** the example that introduces an idea keeps it inline; later examples
  call the library's copy.
- **Changed on purpose:**
  - The occlusion filter is on by default.
  - Depth is always the undistorted picture (01 and 03 used the raw one).
  - There is one median (it averages the two middle values).
  - Class names come from `yolo26n.names`.

**Why:** the user wants a launch point to hand to a software team. The originals stay untouched
in `legacy/` for comparison.

### D36 — 2026-11-07 — The library's default depth range is 6 m; the mirror fix is confirmed

- **Mirror:** the user ran the new programs on the sensor and confirmed left and right are now
  correct (D35's `Unmirror`). `legacy/` is still mirrored.
- **6 m by default:** the new programs still stopped at 4.5 m, because the library had kept
  libfreenect2's own defaults in two places. `KinectSource::Options::maxDepthMeters` was 4.5 (the
  cut-off), and `DepthPicture`'s grey scale ended at 4500 mm. Both are now 6 m, the range 07b
  found usable and that 07c, 08 and later already used. The examples' own grey scales (01, 03, the
  01/03 steps) and 07's `MaxDepthMeters` follow. 07b's 8 m stays a setting in 07.


### D37 — 2026-11-07 — The library's ICP is 13e's, with the turn measured in metres; 11b uses it

- **`spotter::Icp` is 13e's point-to-plane ICP** (directions the shape can't pin keep the start's
  value), not 11's point-to-point one. 11 keeps its point-to-point ICP inline, as its lesson.
  11b's last step now calls the library's, so its result is 13e's quality, and it reports the
  paired share, the gap and the weak directions.
- **Bug 13 fixed by scaling:** the turn's three columns are divided by the pairs' rms distance
  from their centre, so a turn is scored by how far it moves points (metres) like a shift. 13e
  scored turns in metres squared, so a small or near patch (radius 0.1 m) had all three turns
  called weak and left at the start: 19 mm off in `check_icp`, now 0.00 mm. On room-sized views
  (radius 1–2 m) turns now count a little less than before; the 13e port's synthetic tests will
  show whether `weakShare` (5%) needs to change.
- **Bug 2 fixed twice:** `SampleSurface` skips a point whose neighbours give no facing, and `Icp`
  writes its gate as `!(|gap| <= max)`, so a NaN gap is left out instead of let in.
- **`SampleSurface` is the one sparse rule** (bug 3): 13e's (every `step` pixels, edges dropped).
  The 13e and 13z ports will both call it.
- **Not built yet:** `features_orb`, the library's feature pose and the descriptor index wait for
  the 13e port, their first user (11b keeps its own inline). `Pose::Inverse` / text form wait for
  the room folder.

**Why:** one ICP for everything from 11b on, with two of the review's bugs closed and a check
that fails without each fix.

### D38 — 2026-11-07 — Room folder format 2; 13e and 13z on the library

- **Format 2** (`spotter::room_folder`): the same files as format 1, the right way round. Also new:
  - `room.txt` has a `depth_camera fx fy cx cy` line, the camera the clouds were made with.
  - `pose.txt` holds the cloud's point count.
  - Every file is flushed to disk (`_commit` / `fsync`) before `pose.txt` is written. A `cloud.bin`
    that doesn't match the count is refused, and z = 0 counts as no reading (the review's bug 4).
  - `pose.txt` is rewritten atomically by X, and scan folders sort by number (bug 5).
- **Format 1 is read mirrored back:** each cloud row is reversed with x → −x, each picture flipped,
  turns become S R S, shifts get x → −x, and the camera's cx becomes width − cx. Features aren't
  loaded (13z doesn't need them, and ORB descriptors don't survive a mirror). Legacy 13z refuses
  format 2. New `.obj`s carry a `# spotter format 2` line: legacy Demos 4 and 6 expect mirrored
  rooms, so they shouldn't be used with new rooms. Their ports (phase 6) will mirror format-1 rooms
  on load.
- **One sparse rule** (bug 3): `MakeScan` and `LoadScan` both use `SampleSurface`, so 13z's merge
  box is 13e's. `check_room` and a synthetic 13e session both rebuild `points.ply` identically.
- **13e:** its session (state plus one function per key) is `scan_state.hpp`, beside `main.cpp`.
  It's the example's own code, not the library's.
  - Bug 6: a try/catch around the loop finishes the folder on any error, as on Q or a lost Kinect,
    before the Kinect is stopped.
  - Bug 14: each landmark records its two scans, and placing a scan again leaves them out.
  - Every change to `scans` now waits for the background readers, including the first scan.
- **Library choices:**
  - `FindFeatures` makes its ORB and CLAHE per call, so the live check and a new scan can run at
    once without sharing one detector.
  - `Shortlist` still stacks every scan's descriptors per call (fine for tens of scans). The
    persistent descriptor index waits until sessions get that long.
  - `BuildRoom` uses the library's `FindPlanes` (bug 8 fixed), and every projection uses
    `PinholeCamera`'s −0.5 rule.
  - Demo 6's `FindFloor` (bug 7) and locating the visitor wait for the demos.

**Why:** D35's plan for phase 5. A crash or blue screen can't pass off a half-written scan as real,
and old rooms stay usable, the right way round.

### D39 — 2026-11-08 — No autosave: scans are kept in memory and saved once (supersedes half of D30)

- **The user's call (2026-11-07):** autosave was a stopgap, chosen under time pressure and for the
  blue-screen risk (D25), and it isn't needed any more. The design goal is now no heavy disk writes
  while scanning: 13e wrote about 11 MB per scan.
- **`examples/13e`, renamed `13e_Room_Scan` (it was `13e_Room_Scan_Autosave`), and the coming 13f:**
  - the scans stay in memory, and U, X and P change only memory;
  - the folder is asked for at the start, but written only on K, on Q, when the Kinect is lost (10 s
    without a frame), or on any exception, in that order of likelihood;
  - each save writes every scan (pose.txt last, flushed to disk; D38), landmarks, `points.ply` and
    `room.txt` with `finished 1`, and removes scan folders left over from scans that U took back.
- **The rescue half of D30 stays:** a lost Kinect or an error still saves everything, before the
  Kinect is stopped.
- **What it costs:** a crash that isn't an exception (a blue screen, killing the process) now loses
  the whole session, not just the scan in progress.
- **Step 3 of the ladder:** `13e_s3_Save_As_You_Go` became `13e_s3_Save_The_Folder` (saves once at
  the end).
- `legacy/13e` keeps its autosave.

**Why:** the user's decision. A temporary synthetic harness confirmed nothing is written while
scanning, that K and Q each write one complete folder, that a second save removes a scan taken back
by U, and that 13z loads the result.

### D40 — 2026-11-08 — The six demos on the library; Demos 4 and 6 read room folders

The user asked for the demos to be rebuilt (phase 6 of D35's plan), in `examples/demos/`.
- **Demos 1, 2, 3, 5:** the legacy demos on the library, with the same settings and behaviour.
  Demo 2 copies 07d's code and Demo 3 copies 06c's: a demo stands on its own, and the example that
  introduces an idea keeps it inline (D35). Demo 5 refuses a height when its top or bottom row has
  too few readings (the review's bug 12; legacy measured to the camera).
- **Demos 4 and 6 read a 13e room folder,** not 13's `room_scan_*.obj`. Every included scan is
  merged by `FuseScans` at 3 cm (13's export size) and levelled by `room.txt`'s first camera turn.
  With no folder they take the Kinect's own empty view, as legacy's `OneViewRoom.h` did. The folder
  is the one 13e writes on K anyway, so there's no separate export step, and its rooms are the right
  way round (format 2).
- **New library module `room_twin`** (both demos use it): `LoadTwinRoom`, `NewestRoomFolder`,
  `OneViewRoom`, `FindTwinFloor` and `LocateKinect`.
  - `FindTwinFloor` is Demo 6's D33 floor rule. It now also refuses a level patch more than 25 cm
    *under* the walls' foot (the review's bug 7). Demo 4 used the lowest 2% of points; it uses this too.
  - `LocateKinect` is D33's multi-start ICP on the library's `Icp`. Each start rebuilds the k-d tree,
    which is fine at 3 cm.
- **Also added to the library:** `StillBackground` in `depth`, 03's empty view after a countdown (for
  Demos 4 and 5), and `OpenExhibitWindow` / `ToggleFullscreen` in `view`.
- **Kept from legacy:** Demo 6's fill (about 950 lines, D31–D33) is copied unchanged, apart from using
  the library's `Percentile` and `FindTwinFloor`. Splitting it into free functions (the review's
  suggestion) is left for when it changes.
- **Not adopted:** Relocate from the structure study. It stays out of the demos until the findings
  report is reviewed.
- **Ladders:** none for the demos; the examples they're built from have them.
- **Point sizes:** legacy drew each point as big as the patch it stands for. The ports use
  `PointRenderer`'s fixed sizes (3 px for the room). Change them if the room looks gappy on the big
  screen.

**Why:** the user's ask. Room folders remove the extra E-export step and the mirrored `.obj`. One
floor rule fixes bug 7 in both demos. `check_room_twin` covers the shared part on a synthetic room
folder. A temporary harness ran Demo 6's fill on one (floor −1.20 m, couch top 0.60 m) and then
was deleted. None of the six has run on the sensor.

## D41: the docs move to `examples/`; legacy becomes an archive (2026-11-08)

The user's ask, after phase 6 and the build-folder move.

- **Build output:** the `legacy/` programs and presentation demos now build into `build\legacy\`
  (`CMakeLists.txt`); the library's programs stay in `build\examples\`. `build\` itself holds only
  the `check_*` programs.
- **Docs:**
  - The main README, `exercises/cpp/README.md` and `EXAMPLES.md` point to `examples/` and the new
    names (`01_Depth_Peek`, `13e_Room_Scan`, `13z_Room_Rebuild`, `examples/demos/`).
  - 13–13d and 07b are no longer listed there: 13e contains 13–13d, and 07b is a setting of 07.
  - The old per-example Status table in `exercises/cpp/README.md` stays, labelled as the archive of
    what the legacy programs measured on the sensor.
  - 13e's guide no longer says "autosave" (D39).
- **Still built:** `legacy/` is not deleted and keeps building (D34), so old results can be reproduced.

## D42: 13w Room Shell, textured walls, floor and ceiling (2026-11-09)

The user's ask: an example that makes only the room's walls (full height, not cut off), floor and
ceiling, textured as they should be: the floor's texture taken from the floor pixels that were
found, and the ceiling right despite a ceiling fan.

- **Number 13w, not 14.** 14 (textured room, the whole seen mesh) stays planned and unstarted. 13w
  textures only the flat shell, and its texturing (per texel, from the full colour pictures) is the
  first half of 14's plan.
- **Geometry:** walls from `BuildRoom` (09c across every scan) at full height; the floor from
  `FindTwinFloor` (D33); the floor and ceiling follow the walls' outline (a closed chain of wall
  ends, else their convex hull), as squares of 10 cm in the `.obj`, so any outline works.
- **The ceiling: the highest height where level readings cover 0.5 m².** A fan, a lamp or a beam
  hangs lower and covers less, so it never wins. "The most crowded level height" could pick a big
  fan; "the highest reading" picks noise.
- **Texture: pulled, not pushed.** For each 1 cm texel and each scan: the scan's depth at that spot
  must agree with the surface (within 4 cm + 1%), so a couch in front of a wall or the fan under
  the ceiling never paints it. The colour camera must have seen the same spot (the scan's colour
  index), so nothing nearer it does either. The view must not be too slanted (cos 0.25). The colour
  comes from `colour_full.bmp` through `FitColourModel`, about 4x sharper than the depth grid's
  colour. With 3+ scans, a colour more than 60 from their median (a fan blade) is dropped. Then the
  most head-on, nearest scan wins (one scan per texel, not an average, so no blur from small
  misplacements). Exposure isn't evened between scans; that's 14's job.
- **Filling unseen texels:** the slow changes of colour (light, shadow) on 25 cm cells, inpainted
  (Telea), plus the surface's own pattern: what's left of the seen colour after the slow changes,
  copied in 32-texel blocks from its plainest fully seen square (colour nearest the surface's usual
  colour, least busy, so not a poster). Each block is the best of 80 tries at matching what's already
  there where it overlaps by 8 texels (Efros & Freeman 2001's image quilting, without the seam
  cut). The cells are 25 cm so a plank's width counts as pattern, not as slow change.
- **Doors and windows are filled shut** ("full walls"). Cut-outs would need 13x's "points far behind
  a wall".
- **Format 1 (mirrored) rooms are refused:** their full pictures and colour indexes would need
  flipping too.

**Why:** the user's ask. A temporary harness made a synthetic room (a plank floor, a couch, a
poster, a fan 35 cm under a plain ceiling, 10 scans with depth and full colour pictures). Results:
- the ceiling came out 2.500 m above the floor, where the fan would be 2.15 m;
- the seen texels came out 0–3 per channel off the true colours;
- the floor under the couch (never seen) had the floor's colour, within 14 per channel, and 0.78 of
  its pattern's spread. With the quilting switched off, the spread was 0.21 and the check failed;
- the ceiling above the fan came out white, at 235;
- the walls were full height, and K saved an `.obj` with 2956 faces.

It has not run on a real room.

## D43: the floor when the walls' foot is measured too high (2026-11-09)

The user ran 13w on their room: the floor came out well above where it should be.

- **Likely cause:** `FindTwinFloor` refuses a level patch more than 25 cm under the walls' foot (the
  review's bug 7, against a reflection under the floor). If the foot is measured too high, the real
  floor is refused, and a seat or table top wins, or the floor is guessed at the foot. 13w also fed
  the rule raw ICP points, where Demos 4 and 6 give it a room merged at 3 cm: with raw points, walls
  seen up close or by many scans (the scans tilted up for the ceiling) outweigh the walls' lower part.
- **Fixes:**
  - `FindTwinFloor` accepts a patch under the foot after all when it's the biggest level piece that
    isn't too high. A reflection is never bigger than the floor it mirrors, so bug 7's test still
    passes. A new case in `check_room_twin` has the walls' lower 0.6 m hidden and a couch seat; the
    old rule took the seat.
  - 13w thins its points to one per 3 cm cube before the floor and ceiling searches.
  - The `Floor:` console line now prints the walls' foot too.
- **Not confirmed on the user's room.** A synthetic room scanned mostly tilted up, with cabinets
  along the walls, found the floor right with or without the thinning, so the density cause is
  unproven. The user's console lines (`Floor:`, `level readings at …`) will tell which case it was.

## D44: the Pi is the scanner; processing happens on a PC (2026-11-11)

The user's decision, after the 2026-11-11 review ([review-2026-11-11-architecture.md](review-2026-11-11-architecture.md)).

- **No detection on the Pi for now.** The user judged on-device YOLO (OpenCV DNN, NCNN, INT8)
  unviable at this point. The Pi milestone's levers 3 and 4 and the AI HAT are dropped, not deferred
  to a measurement.
- **The model:** collect scans on the Pi; process them on the user's own PC at first (the cloud is a
  later option). Placing, merging, meshing, texturing and detection run on the PC.
- **What "works on the Pi" means:** the scanner builds and runs on the Pi, captures each spot's colour
  and depth, and the result can be exported to a PC, where it loads. Nothing else has to run on the
  Pi. 13e's live placement (ORB + ICP, the anchors, the next-shot advice) is welcome there if it's
  fast enough, but isn't required.
- **Changes:** "Deployment target" in CLAUDE.md now means the scanner. The Pi milestone measures
  capture (depth fps, USB stability, time per spot, saving to the SD card) instead of YOLO.
  `FuseScans`' memory (about 2 GB) is no longer a Pi problem.
- **Open, not decided here:**
  - the capture format: today's room folder, or raw depth as 16-bit PNG + colour + calibration
    (needs OpenCV's `png`; lets RTAB-Map and Open3D read the same capture);
  - how the scanner is controlled without a desktop (screen, VNC, button, phone page);
  - whether the PC side may use a GPU (D11 was written for the Pi);
  - saving as you go on the Pi (power cuts, SD card) against D39's save-once;
  - sending room scans to a cloud, a privacy question under D29.

## D45: room scans outside build/, and saves that can't destroy an earlier save (2026-11-11)

The review's initiative 1 ([review-2026-11-11-architecture.md](review-2026-11-11-architecture.md)); the
user chose the folder and kept D39's save-once.

- **Where:** every program's default is now the repo's `room_scans/` (`SPOTTER_REPO_ROOT "/room_scans"`,
  gitignored): 13e and its step 3, 13w, 13x, 13y, 13z and its steps, Demos 4 and 6. Before, it was
  relative to where the program ran, so 13e's real room landed in `build/examples/room_scans/`, which
  the README's troubleshooting told you to delete. That room was moved, unopened.
- **How (`WriteScan`):** a scan is written as `scan_NNN.new` (every file flushed, `pose.txt` last), then
  swapped in: `scan_NNN` -> `.old`, `.new` -> `scan_NNN`, `.old` deleted. Before, it deleted the scan's
  folder first, so a blue screen (D25) during a second K lost scans the first K had saved.
  `ScanFolders` reads a swap cut off between its renames (the finished `.new`, else the `.old`), and
  ignores an unfinished `.new`. `ScanNumber` now accepts only `scan_` + digits (+ `.new`/`.old`):
  before, `scan_005.new` would have been read as a second scan 5.
- **Only what changed:** 13e remembers which scans' files are on disk; K writes those that aren't, and
  only `pose.txt` (already atomic) for the rest, since X and P change nothing else. `RemoveScan`
  became `RemoveScansFrom` (U's scans and any `.new`/`.old` a cut-off save left).
- `AskRoomFolder` no longer says "every scan goes there the moment it is placed" (false since D39).
- **Checked:** `check_room` has seven new cases: a save swapped in whole; cut off while writing; a
  first save cut off; cut off between the renames; only the `.old` left; the next save tidying up;
  `RemoveScansFrom`. Breaking the unfinished-`.new` rule makes it fail. `ctest` 10/10. Not run on the
  sensor: a real K is the user's check.

## D46: errors reach someone who can handle them (2026-11-11)

The review's initiative 2 ([review-2026-11-11-architecture.md](review-2026-11-11-architecture.md)).

- **`AsyncDetector`:** an error inside the detector (an OpenCV DNN error) used to escape its thread,
  which ends the program on the spot (`std::terminate`): no rescue, no destructors. The thread now keeps
  it, and `TryTake` throws it again on the caller's thread, as `std::future::get` does. Examples that
  catch (13e's way) handle it; the others stop with the message, as any other error.
- **Room folders refuse damage with a reason instead of throwing:** `ReadRoomInfo` refuses an impossible
  `depth_size` or camera (a negative size used to throw in `resize`); `LoadScan` refuses a `cloud.bin`
  that is too long as well as too short; `LoadLandmarks` leaves a non-hex descriptor out (it threw from
  `std::stoi`, and 13z doesn't catch). `LoadTwinRoom` prints each skipped scan's reason (Demos 4 and 6,
  13w, 13x said nothing).
- **13e's `SaveAll` catches errors itself:** the rescue calls it outside main's `try`, so an error there
  ended the program mid-rescue. Scans written before the error are whole (D45).
- **Checks:** `check_yolo_decode` now loads the repo's model on `AsyncDetector`'s thread (no check ran
  the real model before), and a wrong input size (320 on the 640 export) throws through `TryTake`.
  `check_room`: a damaged `room.txt`, a long `cloud.bin`, a broken landmark line; and the
  `RecommendNext` check, which always passed (it only tested that the text starts with "Next:"), now
  needs real advice from a scan's spot, and none with no scans.
- **Not changed:** 13e quits on `q` only, not through `IsQuitKey`, because Esc deselects there. The
  review counted it as a slip; it's deliberate.

## D47: the legacy archive is opt-in; docs and repo tidied (2026-11-11)

The review's initiative 9 ([review-2026-11-11-architecture.md](review-2026-11-11-architecture.md)); the
user's choices on legacy and the living documents.

- **Legacy (the user's choice):** `legacy/` (34 programs, 26k lines) builds only with
  `cmake --preset default -DSPOTTER_BUILD_LEGACY=ON`, still into `build\legacy\`. Git tag
  `legacy-archive` marks it as last built by default. Nothing in `legacy/` changed (D34). Both settings
  were configured and built.
- **The living documents stay as they are (the user's choice):** all six are kept up to date per
  example, the visualizer included; the review had suggested freezing it. 13y, which was missing, now
  has its EXAMPLES.md flowchart and family-tree node, and a visualizer entry with the real room's
  numbers from the findings report (its script parses; the page not looked at).
- **Docs that contradicted the code:** ideas.md said ArUco is already in our OpenCV (it isn't: `objdetect`
  isn't built); CLAUDE.md still called the mirror "fix soon" (fixed by D35/D36) and claimed one check per
  module (10 checks, none for `source_freenect2`, `radar`, `view`); the hand-off README's two tables were
  split by blank lines (the `room_twin` row and the demos fell out), it still named `RemoveScan`
  (D45 renamed it), and it had Windows steps only (now a Linux note, the legacy option and where room
  scans go).
- **Repo:** `*.bmp` is gitignored (the only image format our OpenCV writes, so room pictures and test
  renders); `zone_log.csv` (listed in `.gitignore` but tracked) and the unused `synthetic_seen.bmp` /
  `synthetic_filled.bmp` are untracked, the local copies kept.
- **Licence note:** the README now says the MIT licence covers our code, not the Ultralytics weights
  (AGPL-3.0 or a paid licence), to check before publishing or shipping them.
- **Left for later:** 49 header comments repeat a signature that no longer matches the code (the review's
  count). `.vscode/settings.json` now finds the CMake folder from the workspace, not an old `C:` path.
