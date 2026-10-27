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
