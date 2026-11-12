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

## How it fits together

```mermaid
flowchart TB
    K["Kinect v2<br/>colour 1920×1080 · depth 512×424"]

    subgraph DRV["Driver layer: open source, CPU only"]
        U["libusbK USB driver"] --> F["libfreenect2<br/>decodes depth on the CPU"]
        F --> R["Registration<br/>matches depth pixels to colour pixels"]
    end
    K -->|USB 3| U

    subgraph OBJ["Objects lane: what is where (02–06)"]
        Y["YOLO detector<br/>OpenCV DNN · yolo26n.onnx<br/>on its own thread (05c)"]
        D["Distance to each box (02)"]
        T["Tracker: ID numbers + Kalman filter (04–05b)"]
        X["Floor position in metres<br/>bird's-eye radar (06), walls (06b)<br/>steadied by a filter (06c)"]
        Y --> D --> T --> X
    end

    subgraph ROOM["Room lane: the space in 3D (07–13)"]
        AV["Depth noise and range<br/>(07, 07b)"]
        BG["Remember the empty room<br/>paint people out (07c, 07d)"]
        PC["Point cloud (08)"]
        FL["Find the floor (09)"]
        RM["The room from one view<br/>raw mesh + clean model (09b, 09c)"]
        V11b["Automatic pose from<br/>colour features (11b)"]
        V10["Two views, known pose (10)"]
        V11["ICP alignment (11)"]
        V12["ArUco marker pose (12)"]
        V13["Room scan: many spots merged (13)<br/>TSDF mesh + clean model"]
        SAFE["Scans saved on K / Q (13e)<br/>+ a rescue if the Kinect is lost"]
        KEEP["Kept scans, rebuilt later<br/>K folders + 13z"]
        TEX["Textured room (14)"]
        AV -.->|6 m cut-off| PC
        PC --> FL
        PC --> RM
        FL -.->|up| RM
        PC --> V10 --> V11 --> V13
        V11b -->|rough pose| V11
        V11b --> V13
        V12 -.->|fallback| V13
        RM -.->|clean model of the merged room| V13
        V13 -->|K| KEEP --> TEX
        V13 --> SAFE -->|same folders| KEEP
    end

    R -->|colour| Y
    R -->|depth| D
    F -->|depth| AV
    F -->|depth| BG
    R -->|depth + colour| PC
    Y -.->|person boxes| BG

    MAP["3D map of the space<br/>with the tracked objects in it (15)"]
    X --> MAP
    V13 --> MAP

    classDef planned stroke-dasharray: 5 5
    class V12,TEX,MAP planned
```

The Kinect's colour and depth come in through an open-source driver, then split into two lanes.
The **objects lane** finds things in the colour image and uses depth to place them; the **room
lane** turns depth into 3D points of the room itself. The goal is where they meet: a 3D map with
the tracked objects in it. Solid boxes are built, dashed boxes are planned, and the numbers are
the example folders in `exercises/cpp/`. Everything runs on the CPU, so the same code can move
from the Windows PC to the Raspberry Pi.

How each example works, and what it adds to the one before, is drawn in
[exercises/cpp/EXAMPLES.md](exercises/cpp/EXAMPLES.md).

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
- For room scans (example 13e): a **camera tripod with a ball head**. Without one the floor is
  barely scanned. See [A stand for room scans](#a-stand-for-room-scans).

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

    build\examples\01_Depth_Peek.exe

You should see a grey picture of the room. Near things are bright, far things dark, and red
marks places with no depth reading. A green crosshair (moved with **W A S D**) shows the distance
under it. The console prints the frame rate, about 30 fps. Press **Q** to quit.

Every example is built as `build\examples\<folder name>.exe`. Then work through the others in this order:

| Order | Program | What it teaches |
|---|---|---|
| 1 | `01_Depth_Peek` | What the depth camera measures, and where it fails |
| 2 | `03_Depth_Blobs` | Find people or objects from depth alone, and measure their size (no AI) |
| 3 | `02_Detect_Distance` | YOLO finds objects in colour, then depth tells you how far away each one is |
| 4 | `04_Track_Smooth` | Follow each object over time (ID numbers), and smooth its distance |
| 5 | `05_Kalman` | A Kalman filter: predict where each object will be, then correct. Keeps IDs better, and gives speed |
| 5b | `05b_Two_Thresholds` | Two confidence levels plus a longer memory: objects that flicker in and out keep their ID |
| 5c | `05c_Background_Detect` | Run YOLO on a second thread so the picture, tracking and distances keep the camera's pace |
| 6 | `06_Radar` | Turn each tracked object into a floor position in metres, on a bird's-eye "radar" map with trails |
| 6b | `06b_Radar_Walls` | 06's radar plus the walls, found in the depth alone: the biggest upright flat surfaces, drawn as lines |
| 6c | `06c_Wall_Filter` | Steady the walls: each one gets an extended Kalman filter, because walls don't move but each frame's measurement wobbles |
| 7 | `07_Capture_Average` | Average many still frames to cut depth noise, and measure how much it helps (first step toward a 3D map). Set `MaxDepthMeters` to 8 to see how much noisier the far readings are |
| 7c | `07c_Depth_Background` | Remember the room's depth; anything clearly nearer is new (a person, a moved cup). No neural network |
| 7d | `07d_Remove_Person` | Paint people out with the remembered room behind them; anything they carry stays visible |
| 8 | `08_Point_Cloud` | Turn every depth pixel into a coloured 3D point, and turn the room around with the mouse. P saves a `.ply` (never commit it) |
| 8b | `08b_Object_Mesh` | Click a detected object: its depth pixels become a triangle mesh with a guessed (grey) back, saved as `.obj` for Blender or MeshLab (never commit it) |
| 9 | `09_Find_Floor` | Find the floor with RANSAC: the camera's height, tilt and roll, and any point's height above the floor |
| 9b | `09b_Room_Mesh` | Press S: the whole view becomes a coloured mesh, saved as `.obj`. The first recreation of the room, holes and all (never commit it) |
| 9c | `09c_Room_Model` | The same scan, plus a clean model next to it: flat walls, floor and ceiling, furniture as boxes, stood straight. Both in one `.obj`, in separate groups |
| 10 | `10_Two_Views` | Two captures merged by a hand-measured pose: see the walls show double, nudge the pose until they meet |
| 11 | `11_ICP` | ICP lines the two captures up by itself: pair each point with its nearest neighbour, fit, repeat. Needs a roughly right start |
| 11b | `11b_Feature_Pose` | The rough start found automatically, like Meshroom: matching features in the two colour pictures, then ICP |
| 13e | `13e_Room_Scan` | **Scan a whole room** from several spots and save one model. Each scan is placed by colour details plus ICP, with live "anchors", a suggested next shot, flags for suspicious scans, and far details for hallways. The scans stay in memory and are written to a folder when you press **K** or **Q**, or if the Kinect is lost (a rescue). Guide: [below](#recreate-a-room-example-13e) |
| 13w | `13w_Room_Shell` | No Kinect: just the room itself, as a textured `.obj`: every wall full height, the floor and the ceiling, each with a picture taken from the scans' full colour pictures. Behind the couch, under the table and above a ceiling fan, the surface's own pattern is continued. K saves it |
| 13x | `13x_Room_Fill` | No Kinect: a 13e room with the holes in its flat surfaces (walls, floor, table tops) filled in and marked with blue stripes, so guessed and seen stay apart. K saves a `.ply` with the fill |
| 13y | `13y_Structure_Survey` | No Kinect: a study of placing scans by the room's structure (floor, walls, ceiling) instead of colour. Prints numbers about a saved room. Findings: [docs/structure-localization-findings.md](docs/structure-localization-findings.md) |
| 13z | `13z_Room_Rebuild` | No Kinect: rebuild rooms later from the folders that 13e saves. Pick rooms and scans, look at them, exclude bad ones, build the `.obj` and the `.ply` again (finer if you like). Step by step: [below](#keep-the-scans-and-rebuild-later-k-and-13z) |

**Showing it at an exhibition:** six fullscreen demos (you in 3D, an invisibility cloak, a radar
of the stand, the scanned room with visitors live inside it, "how tall are you", and the scanned
room with its holes filled in by software, marked as guesses) are in
[exercises/cpp/examples/demos/](exercises/cpp/examples/README.md) (built as `build\examples\Demo1_You_In_3D.exe` and so on).
Demos 4 and 6 read a room folder saved by 13e.

Each example's pipeline is drawn as a flowchart, with what it adds highlighted, in
[exercises/cpp/EXAMPLES.md](exercises/cpp/EXAMPLES.md).

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
| Program exits silently, code `C0000135` | A DLL is missing next to the `.exe`. Delete the `build` folder and rebuild (room scans are safe: they live in `room_scans\` at the repo root, not in `build`). |
| **The whole PC blue-screens** while a Kinect program runs (seen once: `ATTEMPTED_SWITCH_FROM_DPC`, 0xB8) | A kernel driver crashed, not our program. Suspects and fixes are in [D25](docs/decisions.md): first quit Microsoft's `KinectService`/`KinectMonitor`, which still run beside libusbK. 13e keeps the scans in memory until K or Q (or a lost Kinect), so press K now and then in a long scan. |

More detail (Linux and Raspberry Pi setup, how the build stays reproducible, known limits) is in
[exercises/cpp/README.md](exercises/cpp/README.md).

---

## Recreate a room (example 13e)

Scan the room from several spots; each scan is placed automatically, and E saves one 3D model of
the whole room. It takes about 10 minutes for a first try. Use `13e_Room_Scan`; it is our room
scanner.

- **Before the windows open,** it asks in the console where to save: `Folder` (Enter =
  `room_scans\` at the repo root, outside `build`) and `Name` (Enter = `room_<date>_<time>`).
- **Nothing is written while you scan.** The scans stay in memory. **K** (or **Q** on the way out)
  writes the folder. If the Kinect is lost or the program fails, it writes the folder by itself first,
  with no questions, so a lost Kinect or an error costs nothing already scanned. A crash or a blue
  screen does cost the unsaved scans, so press **K** now and then in a long scan.
- **Live anchors.** Green dots in the camera window are the details the next scan will be placed
  by: things already scanned. The counter reads `anchors: N (need 20)`. Distance was never the
  real limit; sharing enough of what's been seen is. Move as far as you like while the counter
  stays green. It drops when you face a scanned surface from more than about 30-45 degrees to
  the side, or from 3x further away, because the details stop looking alike.
- **Next shot.** A line like `Next: at spot 12, turn 60 deg right (about 45% new)`, and a green
  field of view in "the room so far".
- **Bad scans.** A scan that looks misplaced gets a **red** circle. Click any camera circle to
  select it (Esc deselects): **X** excludes it from placing, the advice and E (X again puts it
  back), **P** places it again. **U** removes the last scan.
- **Dim rooms.** It evens out the colour picture's contrast before looking for details, and finds
  far more of them than a plain picture would (about 1250 against about 50 in one dim test room).
  Hold still about **4 s** per scan.
- **Long hallways or big rooms.** Plain colour details too far for the depth camera (a hallway's far
  doorway past about 6 m) are used too: magenta dots in the camera window. They pin which way the
  Kinect faces, but not how far down the hallway it is, because a direction has no distance. For that:
  1. At the hallway's start, scan, then **step about 0.5 m sideways** and scan again. The far details
     seen from both spots become **far landmarks** (magenta dots in "the room so far").
  2. Then walk down the hallway as usual. Each scan's directions to the landmarks pin its position.

  A scan placed by far details with nothing to pin its distance is flagged **red**: its position
  along the hallway was kept from the scan it matched. Step sideways for landmarks, then press P.
- **13z** rebuilds a saved folder later, without the Kinect ([below](#keep-the-scans-and-rebuild-later-13z)).

### Before you start

1. **Turn the lights on.** Each scan is placed by matching details in the colour picture. In the
   dark the colour camera sees black, and every scan is refused with "too dark or too plain".
   (Depth still works in the dark, but depth alone can't place the scans.)
2. **Nobody in view.** The model is a colour capture of the room (privacy), and people moving
   between scans smear it.
3. **Pick 4–8 spots** around the room. The Kinect stays plugged into the PC, so the cable decides
   how far you can go. At each spot it must stand **still** for about 2 seconds, best on a
   tripod (see [A stand for room scans](#a-stand-for-room-scans)), otherwise on a chair, a box or a shelf.
   Take some scans **tilted down** for the floor and some **tilted up** for the ceiling. Level
   scans miss both.
4. **Build and start it** from the x64 Native Tools Command Prompt:

       cd /d E:\Spotter\exercises\cpp
       cmake --build --preset default --target 13e_Room_Scan
       build\examples\13e_Room_Scan.exe

   The scans and the model go into the folder you name at the start (Enter = `room_scans\` at the repo root).

### Scan

5. Two windows open: **camera** (what the Kinect sees, and what to do next) and **the room so far**.
   The keys work in either window.
6. **Spot 1:** aim at the busiest part of the room (furniture, shelves, posters, a corner), not
   a bare wall. Press **SPACE**. The camera window shows `HOLD STILL 1/30 ... 30/30`; don't touch
   the Kinect until it's done. The console prints `Scan 1: ... the room's starting point`.
7. **Spot 2:** move the Kinect **0.5–1 m**, or turn it **about 30°**, so that at least half of
   what it sees now was also in an earlier scan. Set it down still, press **SPACE**.
8. **Read the console line for each scan:**

   | It says | Meaning | Do |
   |---|---|---|
   | `matched scan 1 (54 features agree); ICP: 60% of points overlap, 3.0 mm apart` | Placed. Under about 5 mm apart is good | Go on to the next spot |
   | `... (1 direction not pinned by the shape ...)` | Placed, but the view was mostly flat walls; the colour details decided that direction | Fine. Check it in the viewer (step 9) |
   | `Scan refused: only 8 feature matches agree` or `too little overlap` | Too far from every earlier spot | Go back halfway towards an earlier spot and scan again |
   | `Scan refused: only 12 features ... too dark or too plain` | The colour picture has too little detail | Lights on, or aim at something with more pattern |

9. **Check it in the viewer:** drag with the mouse to turn the room, wheel to zoom. Press **C**:
   each scan gets its own colour. A good scan's colour lies on top of the others' surfaces. A
   scan placed wrong shows as a shifted copy (a second wall beside the first): press **U** to
   remove the last scan, then scan that spot again with more overlap.
10. **Carry on round the room.** Corners and furniture help; overlapping too much is fine,
    too little isn't. Keep the Kinect roughly upright. Scan every place you want in the model:
    anything no scan saw stays a hole.

### Save and look at it

11. Press **E**. After a few seconds the console prints the file name (`room_scan_<number>.obj`),
    the room's height, its walls and furniture boxes. You can keep scanning and press E again:
    each export is a new file. **K** saves the scans (below); **Q** saves them too, then quits.
12. **Open it in Blender** (free): File > Import > Wavefront (.obj). In the import options tick
    **Split by Group**. To see the colours: in the viewport's Solid shading menu, set Color to
    **Attribute**. Or in **MeshLab**: File > Import Mesh; colours show by themselves.
13. **The parts**, one object each (hide them in Blender's Outliner):

    | Group | What it is |
    |---|---|
    | `seen_room` | Every scan merged into one mesh: what was really seen. Holes = nobody saw there |
    | `clean_walls`, `clean_floor`, `clean_ceiling` | The tidy room: flat walls from floor to ceiling, joined at the corners. Grey floor or ceiling = guessed (not seen) |
    | `furniture_blocks` | Everything else, as boxes standing on the floor, in each thing's average colour. Only the front of each was seen, so their depth is a guess |

14. **Never commit the `.obj`** (it's a capture of your room; `*.obj` is gitignored). Delete it
    when you're done if it shows anything private.

### Keep the scans and rebuild later (13z)

E's `.obj` keeps one colour per 2 cm corner, averaged over every scan, so posters and small details
blur. The folder 13e writes (on **K**, on **Q**, or by itself when the Kinect is lost) keeps every
scan, so you can rebuild with 13z later.

1. **At the start,** 13e asks in its console window for the folder (Enter = `room_scans\` at the repo
   root, or type or paste any folder) and the name (Enter = `room_<date>_<time>`). A room folder
   of the same name is overwritten only if you answer `y`; type `cancel` at either question to stop.
2. The folder holds every scan's points, pictures and placement, plus **`points.ply`**. That's
   every scan's own points with their own colours, what the viewer shows, with no averaging. Open
   it in MeshLab (File > Import Mesh) or Blender (File > Import > Stanford PLY). It's about 11 MB
   per scan.
3. **Rebuild any time, without the Kinect:**

       build\examples\13z_Room_Rebuild.exe

   It lists every saved room in the repo's `room_scans\` (or give a folder: `13z_Room_Rebuild D:\scans`).
   Then you choose:
   - **Which rooms:** `2`, `1,3`, or Enter for all.
   - **What to do:** view, build `.obj`, build `.ply`, or both.
   - **For each room, which scans to load:** `1-12,15`, or Enter for all.

   In the viewer, click a camera circle and press **X** to exclude a bad scan (13z remembers it).
   Then **E** builds the `.obj` and **K** the `.ply`, into the room's own folder. For a finer mesh,
   set `voxelMeters` to 0.01 in `RoomSettings` at the top of `exercises/cpp/examples/13z_Room_Rebuild/main.cpp` and rebuild it.
4. **Privacy:** the folder holds colour pictures of your room. `room_scans/` is gitignored, and 13e
   puts a `.gitignore` into every folder it writes, wherever that is. Never commit one, and scan
   with nobody in view.

### If something goes wrong

| Problem | Fix |
|---|---|
| Every scan is refused as "too dark" | Lights on. The colour camera needs light; check nothing covers the Kinect's lenses |
| Double walls in the saved model | One scan was placed slightly wrong. Rescan: watch the viewer with C on and press U as soon as a scan shows as a copy |
| "nothing level found" in the export | No scan saw the floor or ceiling well. Tilt the Kinect so some scans include them |
| No floor in `seen_room` (holes underfoot) | The scans were level or tilted up. Rescan with the Kinect 30–45° down, ideally on a tripod: see [A stand for room scans](#a-stand-for-room-scans). `clean_floor` is still drawn, as a guess |
| Export is slow or the file is huge | Set `voxelMeters` to 0.04 in `RoomSettings` at the top of `exercises/cpp/examples/13e_Room_Scan/main.cpp` and rebuild |
| The room drifts: the last scan doesn't meet the first | Small errors add up over many scans. Scan in a loop with plenty of overlap; see experiment 5 in the file |

### A stand for room scans

**Why the floor goes missing.** The Kinect sees 30° above and 30° below where it points. Where
it first sees the floor, held 1 m up:

| Kinect pointing | Floor first seen | Light hits the floor at |
|---|---|---|
| 14° up (the 2026-11-03 flat scan) | 3.5 m away | 16°, a weak return and few readings |
| Level | 1.7 m away | 30° |
| 30° down | 0.6 m away | 60°, a strong return |
| 45° down | at its feet | 75° |

Straight down isn't needed. From 1–1.5 m high, **30–45° down** is enough, and you keep the
walls in view for matching. Software can tidy a floor it saw badly, but it can't measure a floor
it never saw.

**What to buy:** a camera tripod with a **ball head**. The Kinect v2 (Xbox One Kinect) has a
standard **1/4"-20 tripod socket** under its base; check yours before buying. Look for:
- a ball head with a **drop notch**: the slot that lets the camera tip 90°, so steep tilts hold
  firmly;
- a load rating of **3 kg or more**. The Kinect weighs 1–1.4 kg (sources differ), and tipping it
  forward strains the head more than its weight suggests;
- working height up to about **1.5 m**, with a **centre-column hook** to hang a bag on for
  steadiness.

Examples (prices change, so check before buying):
- [K&F Concept 64"/163 cm aluminium travel tripod](https://www.amazon.com/Lightweight-Capacity-Cellphone-Smartphone-Streaming/dp/B0B1HYVVTV):
  36 mm ball head rated 8 kg. Plenty for the Kinect.
- [K&F Concept 90"/230 cm overhead tripod](https://www.amazon.com/CONCEPT-Horizontal-Overhead-Camera-Tripod/dp/B0C7BX47VW):
  its centre column swings horizontal, for true straight-down shots. Only needed for floor detail
  (rugs, tiles). Keep the legs out of view.

**Mounting:**
- Screw the head's quick-release plate into the Kinect's base.
- Point a leg forward, under the direction the Kinect faces, so the weight sits over a leg when
  it's tilted.
- Route the cable down a leg, so a tug doesn't pull the tripod over.

**Until it arrives:**
- **Tilt the Kinect down on its own hinge.** Its body tilts on its base by hand.
- **For more tilt,** set it on a table or shelf edge and wedge a book under the back of the base.
- **Each scan must stand still for 2 s,** so don't hand-hold it.

The examples that build up to this one, one idea each: 09b (one view as a mesh), 09c (the clean
model), 10 (two views by hand), 11 (ICP), 11b (features). Their flowcharts are in
[exercises/cpp/EXAMPLES.md](exercises/cpp/EXAMPLES.md).

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
