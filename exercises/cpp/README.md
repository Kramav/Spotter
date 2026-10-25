# Kinect v2 examples: C++ and libfreenect2

These are the open-source versions of `exercises/csharp`. They use no Microsoft SDK and no
Microsoft driver, and the same code builds on Windows and Linux. For the reasons behind this, see
[docs/decisions.md](../../docs/decisions.md), entries D5 to D11.

| Example | Teaches | Status |
|---|---|---|
| `01_Kinect_Depth_Peek` | Reading depth, what "no reading" looks like, depth noise | Runs on the sensor at 30 fps; window matches the C# version |
| `03_Kinect_Depth_Blobs` | Background subtraction, blob size from depth | To do |
| `02_Kinect_Detect_Distance` | YOLO detection plus depth registration: how far away is it? | To do |

## One-time setup: Windows

You need **VS Build Tools 2026 with "Desktop development with C++"**. That gives you `cl`,
`cmake`, `ninja` and `vcpkg`. Run every command below from an
**"x64 Native Tools Command Prompt"**, in `exercises\cpp`.

**1. Libraries (libusb, libjpeg-turbo, OpenCV).** This step takes about 30 minutes the first time.

    vcpkg install --triplet x64-windows

**2. libfreenect2, built from source and pinned (D9).** Our patch spreads its CPU depth decoder
across all CPU cores. On this PC that takes it from 7.5 to 30 fps (D11).

    git clone https://github.com/OpenKinect/libfreenect2.git third_party\libfreenect2
    git -C third_party\libfreenect2 checkout fd64c5d9b214df6f6a55b4419357e51083f15d93
    git -C third_party\libfreenect2 apply ..\..\libfreenect2-openmp.patch
    set V=%CD%\vcpkg_installed\x64-windows
    cmake -S third_party\libfreenect2 -B third_party\libfreenect2\build -G Ninja ^
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
      -DCMAKE_INSTALL_PREFIX=%CD:\=/%/third_party/install ^
      -DBUILD_EXAMPLES=OFF -DBUILD_OPENNI2_DRIVER=OFF -DENABLE_CXX11=ON ^
      -DENABLE_OPENCL=OFF -DENABLE_CUDA=OFF -DENABLE_OPENGL=OFF -DENABLE_VAAPI=OFF -DENABLE_TEGRAJPEG=OFF ^
      -DLibUSB_INCLUDE_DIRS=%V%\include\libusb-1.0 -DLibUSB_LIBRARIES=%V%\lib\libusb-1.0.lib ^
      -DTurboJPEG_INCLUDE_DIRS=%V%\include -DTurboJPEG_LIBRARIES=%V%\lib\turbojpeg.lib
    cmake --build third_party\libfreenect2\build
    cmake --install third_party\libfreenect2\build

**3. USB driver.** You do this step yourself. Replace the Microsoft driver with libusbK, using
the [libfreenect2 README](https://github.com/OpenKinect/libfreenect2#windows--visual-studio)
instructions:

1. Run [Zadig](https://zadig.akeo.ie/). Under Options, tick "List All Devices" and untick "Ignore Hubs or Composite Parents".
2. Choose **"Xbox NUI Sensor (composite parent)"** (VID 045E). Do not choose "NuiSensor Adaptor".
3. Choose libusbK as the replacement driver, then click **Replace Driver**.

After this, the C# examples and anything else that uses the Microsoft SDK stop working.

To undo it: in Device Manager, open "libusbK USB Devices", right-click "Xbox NUI Sensor
(Composite Parent)", choose Uninstall and tick "Delete the driver software". UsbDk is an
alternative to libusbK (see the libfreenect2 README); install only one of the two.

## One-time setup: Linux (Ubuntu/Debian)

Not yet tested on Linux. These are the expected steps:

    sudo apt install build-essential cmake ninja-build pkg-config libusb-1.0-0-dev libturbojpeg0-dev libopencv-dev
    # Step 2 above, without the LibUSB_ / TurboJPEG_ lines (pkg-config finds them),
    # and with -DCMAKE_INSTALL_PREFIX=$PWD/third_party/install
    sudo cp third_party/libfreenect2/platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/   # USB access without root

With apt's OpenCV, leave out the `CMAKE_TOOLCHAIN_FILE` option below.

## Build and run

    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"
    cmake --build build
    build\01_Kinect_Depth_Peek

On Linux, run `./build/01_Kinect_Depth_Peek`.

**Before the driver swap** you get this, which is expected:

    [Error] [Freenect2Impl] failed to open Kinect v2: @1:4 LIBUSB_ERROR_NOT_FOUND Entity not found
    No Kinect v2 found. ...

libfreenect2 can see the Kinect on the USB bus, but the Microsoft driver still owns it.

If the program exits silently with code C0000135, a DLL is missing. Run
`dumpbin /dependents` on the program to find which one, then add it to the copy list in
`CMakeLists.txt`.

## Known limits

- Depth is decoded on the CPU across all cores (D11), because the target Raspberry Pi has no GPU
  path. Measured on a Ryzen 5 3600: 30 fps with the filters on. The Pi is not yet measured.
  `UseDepthFilters = false` roughly doubles CPU headroom.
- libfreenect2 decodes depth itself, so its holes and noise differ slightly from the Microsoft
  SDK's.
- No body tracking, because no open-source driver provides it (D5).
