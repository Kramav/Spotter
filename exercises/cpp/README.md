# Kinect v2 examples: C++ and libfreenect2

Reference notes for the C++ examples. To get started, see the [main README](../../README.md).

These are the open-source versions of `exercises/csharp`. They use no Microsoft SDK and no
Microsoft driver, and the same code builds on Windows and Linux. For the reasons behind this, see
[docs/decisions.md](../../docs/decisions.md), entries D5 to D14.

| Example | Teaches | Status |
|---|---|---|
| `01_Kinect_Depth_Peek` | Reading depth, what "no reading" looks like, depth noise | Runs on the sensor at 30 fps; window matches the C# version |
| `02_Kinect_Detect_Distance` | YOLO detection plus depth registration: how far away is it? | Runs on the sensor at 4–6 fps (detection 130–156 ms on the CPU, same as C#) |
| `03_Kinect_Depth_Blobs` | Background subtraction, blob size from depth | Runs on the sensor; blobs not yet tested with a person |
| `04_Kinect_Track_Smooth` | Tracking by box overlap (IDs), and a running average of each object's distance | Runs on the sensor at about 6 fps. Two still chairs kept their IDs; jitter fell from 2–4 mm to 0.4 mm per frame |

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

- Depth is decoded on the CPU across all cores (D11), because the target Raspberry Pi has no GPU
  path. Measured on a Ryzen 5 3600: 30 fps with the filters on. The Pi is not yet measured.
  `UseDepthFilters = false` roughly doubles CPU headroom.
- libfreenect2 decodes depth itself, so its holes and noise differ slightly from the Microsoft
  SDK's.
- No body tracking, because no open-source driver provides it (D5).
