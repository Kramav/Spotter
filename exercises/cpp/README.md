# Kinect v2 examples: C++ and libfreenect2

Reference notes for the C++ examples. To get started, see the [main README](../../README.md).

These are the open-source versions of `exercises/csharp`. They use no Microsoft SDK and no
Microsoft driver, and the same code builds on Windows and Linux. For the reasons behind this, see
[docs/decisions.md](../../docs/decisions.md), entries D5 to D15.

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
| `07_Kinect_Capture_Average` | Per-pixel average over `CaptureFrames` still frames: one-frame noise by distance, expected noise of the average (÷√N), and its real noise measured from two captures | Runs on the sensor: one frame ±1.2 mm at 0.5–1 m; average expected ±0.2, measured ±0.3 mm. Tested with a temporary auto-capture build (no key presses sent) |
| `07b_Kinect_Far_Range` | `Config::MaxDepth` raised from libfreenect2's default 4.5 m to `MaxDepthMeters` (8); per 1 m band: share of view, flaky pixels, one-frame noise | Runs on the sensor: never-read pixels 33.8% (4.5 m) → 2.3% (8 m); 4–6 m noise ±5 mm, about the same as 3–4 m; past 6 m almost nothing reads. Same auto-capture test |
| `08_Kinect_Point_Cloud` | `getPointXYZ` for every pixel plus registered colour; its own z-buffered viewer (drag/wheel or WASD/ZX); P saves ASCII `.ply` (Y up), cut-off 6 m | Runs on the sensor: about 193k points, 14 loops/s; test `.ply` had z 0.5–5.7 m, 11.6% black points (outside the colour camera's view). Tested with a temporary auto-save build; the file was deleted. Viewer not yet checked by eye |
| `09_Kinect_Find_Floor` | RANSAC plane fit (3 neighbouring points, level planes only, then a PCA refit on the inliers): camera height, tilt, roll; crosshair height above the floor | Runs on the sensor, but no floor was in view: it reported a different level-ish surface each second (6–8% of points). Not yet tested with a floor in view |

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
