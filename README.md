# Spotter

Spotter is a hands-on project for learning object detection, recognition and tracking. The current
work uses a **Kinect v2**, a camera that sees both colour and *depth* (how far away each pixel
is). The examples find objects in the colour image and use depth to measure how far away they
are.

- **How we work:** small, runnable examples. Each one teaches one idea and builds on the
  previous one.
- **Long-term goal:** a 3D map of a space.
- **Final hardware:** a Raspberry Pi.

---

## What's in this repository

| Folder | What it is | Needs |
|---|---|---|
| `exercises/python/` | Beginner exercises with a normal webcam: drawing, motion, YOLO detection, tracking | Any webcam, Python |
| `exercises/cpp/` | **The current work.** Kinect v2 examples in C++ | Kinect v2, see below |
| `exercises/csharp/` | An earlier version of the Kinect examples, on Microsoft's Kinect SDK. Kept as a reference, no longer developed | Windows + Microsoft Kinect SDK |
| `docs/` | `decisions.md` records why things are built the way they are. `ideas.md` holds researched ideas for later | — |
| `kinect_rgbd_fusion_guide.md` | Background reading: the method behind the Kinect examples | — |
| `yolo26n.pt`, `yolo26n.onnx`, `yolo26n_320.onnx` | The object detector (YOLO), already trained. Python uses `.pt`; C++ uses `.onnx` (640 px input, or the faster 320 px one) | — |

### Words you'll see

| Term | Meaning |
|---|---|
| **Kinect v2** | A Microsoft camera from 2014 with a colour camera (1920×1080) and a depth camera (512×424, about 0.5–4.5 m). |
| **Depth image** | An image where each pixel holds a distance in millimetres instead of a colour. |
| **YOLO** | A neural network that draws boxes around objects (people, chairs, cups…) and labels them. |
| **OpenCV** | The standard open-source library for images, windows and running YOLO. |
| **libfreenect2** | An open-source driver that talks to the Kinect. We use it instead of Microsoft's SDK. |
| **CMake** | Turns our build description (`CMakeLists.txt`) into a compiled program. |
| **vcpkg** | Downloads and builds the libraries we depend on, at the exact versions pinned in `vcpkg.json`. |

---

## Getting started: the Kinect examples (Windows)

### What you need

**Hardware**
- A **Kinect v2** (the Xbox One Kinect) plus the **Kinect Adapter for Windows**, which provides
  power and the USB connection.
- A **USB 3.0** port. These are often blue, or marked "SS". USB 2.0 will not work.

**Software** (all free)
- [Git](https://git-scm.com/download/win), to download this repository.
- [Build Tools for Visual Studio 2026](https://visualstudio.microsoft.com/downloads/), the C++
  compiler. Under "Tools for Visual Studio", download *Build Tools*. In the installer, tick the
  workload **"Desktop development with C++"**. That also installs CMake and vcpkg; you don't
  install them separately.

### Step 1: download the repository

Open a terminal (Command Prompt or PowerShell) and clone to a **short path**. vcpkg's build
folders nest deeply, and Windows can't handle paths over 260 characters.

    cd E:\
    git clone <this repository's URL> Spotter

### Step 2: give the Kinect an open-source USB driver (once per PC)

Windows attaches Microsoft's driver to the Kinect. libfreenect2 needs a generic USB driver
called **libusbK** instead. You swap it once with a small tool called Zadig:

1. Plug in the Kinect (adapter powered, USB in a 3.0 port).
2. Download and run [Zadig](https://zadig.akeo.ie/).
3. Open the **Options** menu. Tick **"List All Devices"** and untick **"Ignore Hubs or Composite Parents"**.
4. In the drop-down, choose **"Xbox NUI Sensor (composite parent)"**. Don't choose "NuiSensor
   Adaptor", which is the adapter, not the Kinect.
5. Choose **libusbK** as the new driver and click **Replace Driver**. Say yes to the warning.

After this, programs that use Microsoft's Kinect SDK (including `exercises/csharp`) can't see the
Kinect.

**To undo it:** open Device Manager, open "libusbK USB Devices", right-click "Xbox NUI Sensor
(Composite Parent)", choose **Uninstall device**, tick **"Delete the driver software"**, and replug
the Kinect.

### Step 3: build

1. Open the Start menu and search for **"x64 Native Tools Command Prompt for VS"**. Use this
   prompt, not a normal one; it has the compiler and tools set up.
2. Run:

       cd /d E:\Spotter\exercises\cpp
       cmake --preset default
       cmake --build --preset default

The first `cmake --preset default` takes **about 30 minutes**, because vcpkg downloads and
compiles OpenCV and libfreenect2 from source. That's normal. Later builds take seconds, because
vcpkg keeps a cache.

### Step 4: run

Still in `exercises\cpp`:

    build\01_Kinect_Depth_Peek.exe

You should see a grey picture of the room. Near things are bright, far things dark, and red
marks places with no depth reading. A green crosshair (moved with **W A S D**) shows the distance
under it. The console prints the frame rate, about 30 fps. Press **Q** to quit.

Then work through the other examples in this order:

| Order | Program | What it teaches |
|---|---|---|
| 1 | `01_Kinect_Depth_Peek` | What the depth camera measures, and where it fails |
| 2 | `03_Kinect_Depth_Blobs` | Find people or objects from depth alone, and measure their size (no AI) |
| 3 | `02_Kinect_Detect_Distance` | YOLO finds objects in colour, then depth tells you how far away each one is |
| 4 | `04_Kinect_Track_Smooth` | Follow each object over time (ID numbers), and smooth its distance |
| 5 | `05_Kinect_Kalman` | A Kalman filter: predict where each object will be, then correct. Keeps IDs better, and gives speed |
| 5b | `05b_Kinect_Two_Thresholds` | Two confidence levels plus a longer memory: objects that flicker in and out keep their ID |
| 5c | `05c_Kinect_Background_Detect` | Run YOLO on a second thread so the picture, tracking and distances keep the camera's pace |
| 6 | `06_Kinect_Radar` | Turn each tracked object into a floor position in metres, on a bird's-eye "radar" map with trails |
| 7 | `07_Kinect_Capture_Average` | Average many still frames to cut depth noise, and measure how much it helps (first step toward a 3D map) |
| 7b | `07b_Kinect_Far_Range` | Raise the depth cut-off from 4.5 m to 8 m, and see how much noisier the far readings are |
| 8 | `08_Kinect_Point_Cloud` | Turn every depth pixel into a coloured 3D point, and turn the room around with the mouse. P saves a `.ply` (never commit it) |

Each example is **one file** (`main.cpp` in its folder), meant to be read from top to bottom:
settings at the top, then setup, the main loop, output, and **experiment ideas** at the bottom.
Change a setting and rebuild with `cmake --build --preset default` to see what it does.

### If something goes wrong

| Problem | Fix |
|---|---|
| `No Kinect v2 found` or `LIBUSB_ERROR_NOT_FOUND` | Step 2 isn't done, or the Kinect isn't powered or not on a USB 3.0 port. |
| `'cmake' is not recognized` | You're in a normal terminal. Use the **x64 Native Tools Command Prompt** (step 3). |
| Build fails with very long paths | Clone to a short path such as `E:\Spotter` (step 1). |
| The window opens and closes immediately | Run it from the command prompt instead of double-clicking, so you can read the error. |
| Program exits silently, code `C0000135` | A DLL is missing next to the `.exe`. Delete the `build` folder and rebuild. |

More detail (Linux and Raspberry Pi setup, how the build stays reproducible, known limits) is in
[exercises/cpp/README.md](exercises/cpp/README.md).

---

## Getting started: the Python webcam exercises

These need only a webcam and Python 3. From the repository root:

    python -m venv .venv
    .venv\Scripts\activate
    pip install opencv-python ultralytics supervision numpy
    python exercises\python\01_CV_Draw.py

Run them from the **repository root**, because the YOLO exercises load `yolo26n.pt` from there.
Press **Q** to quit.

---

## Working on the project

- **Read first:** [CLAUDE.md](CLAUDE.md) has the project's status, roadmap and conventions.
  [docs/decisions.md](docs/decisions.md) explains *why* things are the way they are.
- **Example style:** one idea per example, one file, settings at the top, experiment ideas at the
  bottom. Readability comes before cleverness.
- **Privacy:** never commit camera images, recordings or point clouds. They capture people and
  rooms. Image files are gitignored for this reason.
- **New code** must build on both Windows and Linux.

## License

MIT. See [LICENSE](LICENSE).
