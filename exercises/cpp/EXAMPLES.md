# The examples, as flowcharts

Each C++ example teaches one idea on top of an earlier one. This page draws every example's
pipeline, from the Kinect to what you see. **Boxes with a thick border are what that example adds**;
the thin ones it takes over from the example it starts from. To build and run them, see the
[main README](../../README.md); the measurements behind the "What to compare" lines are in
[README.md](README.md).

## Family tree

Which example each one starts from. A solid arrow means "starts from that example's code"; a
dotted arrow means "borrows one idea from it". Dashed boxes are planned.

```mermaid
flowchart LR
    E01["01 Depth Peek"] --> E02["02 Detect + Distance"] --> E04["04 Track + Smooth"] --> E05["05 Kalman"]
    E05 --> E05b["05b Two Thresholds"] --> E05c["05c Background Detect"]
    E05 --> E06["06 Radar"]
    E01 --> E03["03 Depth Blobs"] --> E07c["07c Depth Background"] --> E07d["07d Remove Person"]
    E01 --> E07["07 Capture + Average"] --> E07b["07b Far Range"]
    E03 --> E08["08 Point Cloud"]
    E08 --> E09["09 Find the Floor"]
    E08 --> E10["10 Two Views"] --> E11["11 ICP"]
    E07 -.->|running average| E07c
    E05c -.->|YOLO thread| E07d
    E08 -.->|colour on depth pixels| E07d
    E07b -.->|6 m cut-off| E08

    classDef planned stroke-dasharray: 5 5
    class E11 planned
```

---

## 01 Depth Peek

The starting point: read the depth camera and look at what it actually measures, including where
it fails.

```mermaid
flowchart LR
    K["Kinect"] --> DEP["Depth frame<br/>512×424, mm per pixel"]
    DEP --> G["Grey picture<br/>near = bright, far = dark"]
    DEP --> NR["No reading = red<br/>too near, too far, shiny, glass"]
    DEP --> CH["Crosshair (WASD)<br/>distance in mm"]

    classDef new stroke-width:3px
    class DEP,G,NR,CH new
```

**What to compare:** about 30 fps. Point it at a window, a mirror or something black and watch the
red appear.

## 02 Detect + Distance

Starts from 01. Adds the colour camera and YOLO: the detector finds boxes in colour, and depth then
says how far away each box is ("late fusion").

```mermaid
flowchart LR
    K["Kinect"] --> COL["Colour frame<br/>1920×1080"]
    K --> DEP["Depth frame"]
    COL --> Y["1. Detect<br/>YOLO boxes"]
    DEP --> REG["2. Register<br/>where each depth pixel lands in colour"]
    Y --> S["3. Sample<br/>median depth in the box centre"]
    REG --> S
    S --> DP["4. Deproject<br/>pixel + depth = X, Y, Z in metres"]
    DP --> O["Label: person 1.84 m"]

    classDef new stroke-width:3px
    class COL,Y,REG,S,DP,O new
```

**What to compare:** the frame rate drops from 30 to about 6 fps, because YOLO takes 130–180 ms.

## 03 Depth Blobs

Starts from 01, and uses no AI. Remember the empty room; anything nearer than it is something new.
It doesn't know what anything is, but it works in the dark.

```mermaid
flowchart LR
    K["Kinect"] --> DEP["Depth frame"]
    DEP -->|B key| BG["1. Remember the empty room"]
    DEP --> FG["2. Nearer than the room<br/>= foreground"]
    BG --> FG
    FG --> BL["3. Clean speckles<br/>group into blobs"]
    BL --> SZ["4. Median depth = distance<br/>pixels × depth ÷ focal = size in m"]
    SZ --> O["Box + label per blob"]

    classDef new stroke-width:3px
    class BG,FG,BL,SZ,O new
```

**What to compare:** turn the lights off. 03 keeps working, 02 finds nothing.

## 04 Track + Smooth

Starts from 02. 02 measures every box from scratch, so distances jitter and nothing links an
object in one frame to the next. 04 follows each object over time.

```mermaid
flowchart LR
    K["Kinect"] --> D["Detect, register, sample<br/>(as in 02)"]
    D --> TR["4. Track<br/>match boxes to last frame's by overlap<br/>each object keeps an ID"]
    TR --> SM["5. Smooth<br/>running average of each object's distance"]
    SM --> O["Label: ID, smoothed distance<br/>raw value in brackets"]

    classDef new stroke-width:3px
    class TR,SM new
```

**What to compare:** two still chairs: jitter fell from 2–4 mm to 0.4 mm per frame.

## 05 Kalman

Starts from 04. At 6 fps a walking person barely overlaps their last box, and averaging lags
behind anything that moves. A Kalman filter per object fixes both: predict, then correct.

```mermaid
flowchart LR
    K["Kinect"] --> D["Detect, register, sample<br/>(as in 02)"]
    P["Predict<br/>each object's box and distance<br/>moved on by its speed"] --> M["Match boxes to the<br/>PREDICTED boxes"]
    D --> M
    M -->|matched| C["Correct<br/>blend in the measurement"]
    M -.->|missed| CO["Coast<br/>keep the prediction"]
    C --> O["Label: ID, distance, speed"]
    CO --> O
    C -->|next frame| P
    CO -->|next frame| P

    classDef new stroke-width:3px
    class P,M,C,CO new
```

**What to compare:** walk across the room with 04 and then 05. Each prints a `new ID` line when an
ID changes. Walking speed should read about 1–1.5 m/s.

## 05b Two Thresholds

Starts from 05. IDs reset because YOLO's confidence wobbles around the 0.40 cut-off. Two levels
fix it (the core of ByteTrack): strong boxes may start an ID, weak ones only continue one.

```mermaid
flowchart LR
    K["Kinect"] --> Y["YOLO boxes"]
    P["Predict (05)"] --> M1
    Y --> M1["Pass 1: strong boxes, 0.40+<br/>match, or start a new ID"]
    M1 --> M2["Pass 2: weak boxes, 0.15+<br/>only continue a leftover ID"]
    M2 --> C["Correct (05)"]
    M2 -.->|still unseen| F["Speed fades out<br/>kept for 30 frames"]
    C --> O["Label: strong, weak or predicted"]
    F --> O

    classDef new stroke-width:3px
    class M1,M2,F new
```

**What to compare:** the same flickering TV for 60 s: 7 IDs with 05, 1 with 05b.

## 05c Background Detect

Starts from 05b. Everything so far waits about 150 ms for YOLO every frame. 05c stops waiting:
YOLO runs on its own thread, and the main loop keeps the camera's pace.

```mermaid
flowchart LR
    K["Kinect frame<br/>30 fps"] --> P["Predict every box<br/>(Kalman, 05)"]
    K -.->|newest colour frame| YT["YOLO on its own thread<br/>a few batches a second"]
    P --> NB{"New YOLO<br/>batch?"}
    YT -.->|boxes| NB
    NB -->|yes| M["Two-pass match (05b)"]
    NB -->|no| RM
    M --> RM["Re-measure each distance<br/>from FRESH depth"]
    RM --> O["Draw at camera speed"]

    classDef new stroke-width:3px
    class YT,NB,RM new
```

**What to compare:** set `DetectInBackground = false` to get 05b's behaviour: display 4 fps
waiting for YOLO, 24–27 fps in the background. `ModelInputSize = 320` makes YOLO about 3× faster
but misses objects cut off at the image edge.

## 06 Radar

Starts from 05. 02–05 say how far away each object is; 06 adds where: a position on the floor in
metres, drawn from above. The project's first map.

```mermaid
flowchart LR
    K["Kinect"] --> T["Detect, sample, track, Kalman<br/>(as in 05)"]
    T --> DP["7. Deproject<br/>X = (u − cx) · Z ÷ fx<br/>Z = filtered distance"]
    DP --> MAP["8. Radar from above<br/>ring every metre, field of view, trails"]
    T --> CAM["Camera view (as in 05)"]

    classDef new stroke-width:3px
    class DP,MAP new
```

**What to compare:** still objects stay steady to about 1 cm. Walk left: does your dot move left?

## 07 Capture + Average

Starts from 01. The first room-lane example: a map is built from still captures, so noise matters
more than speed. With the camera still, average each pixel over many frames.

```mermaid
flowchart LR
    K["Kinect"] --> DEP["Depth frame"]
    DEP --> LIVE["Live depth (as in 01)"]
    DEP -->|SPACE| CAP["Capture 30 still frames"]
    CAP --> AVG["Average each pixel<br/>flaky pixels filled from frames that read"]
    CAP --> N1["One frame's noise<br/>by distance"]
    AVG --> N2["Next capture: difference of two averages<br/>= the average's real noise"]

    classDef new stroke-width:3px
    class CAP,AVG,N1,N2 new
```

**What to compare:** one frame ±1.2 mm at 0.5–1 m; the average ±0.3 mm (theory says ±0.2).

## 07b Far Range

Starts from 07. Everything so far stops at 4.5 m, but that's libfreenect2's default, not the
sensor. Raise the cut-off and measure how much worse far readings are.

```mermaid
flowchart LR
    CFG["MaxDepth setting<br/>4.5 m → 8 m"] --> K["Kinect"]
    K --> DEP["Depth frame"]
    DEP --> LIVE["Live view<br/>grey up to 4.5 m, blue beyond"]
    DEP -->|SPACE| CAP["Capture frames (07)"]
    CAP --> BANDS["Per 1 m band:<br/>share of view, one-frame noise, flaky pixels"]

    classDef new stroke-width:3px
    class CFG,LIVE,BANDS new
```

**What to compare:** never-read pixels 33.8% → 2.3% in the same scene. 4–6 m reads ±5 mm, as good as
3–4 m; past 6 m almost nothing reads, hence the 6 m cut-off in later examples.

## 07c Depth Background

Starts from 03. 03 needed an empty room at the start (the B key). 07c's memory keeps itself up to
date, per pixel, so it heals by itself when something moves away.

```mermaid
flowchart LR
    K["Kinect"] --> DEP["Depth frame"]
    DEP --> CMP{"Compare with<br/>remembered depth"}
    MEM["Background memory<br/>per pixel"] --> CMP
    CMP -->|about the same| BL["Blend in<br/>slow running average (07)"]
    CMP -->|farther for 10 frames| TK["Something left<br/>take the new value"]
    CMP -->|nearer| FG["Foreground"]
    FG --> BLOB["Blobs, boxed (03)"]
    FG -.->|nearer for 300 frames| AB["Something stayed<br/>joins the room"]
    BL --> MEM
    TK --> MEM
    AB --> MEM

    classDef new stroke-width:3px
    class CMP,MEM,BL,TK,AB new
```

**What to compare:** a spinning ceiling fan made 2–6 blobs when a farther reading was taken at
once; waiting 10 frames (`LeaveFrames`) makes it 0.

## 07d Remove Person

Starts from 07c, with YOLO on its own thread (05c). Decides which foreground is a person and
paints them out with the room behind them. Anything they carry stays visible.

```mermaid
flowchart LR
    K["Kinect"] --> DEP["Depth frame"]
    K --> COL["Colour frame"]
    DEP --> BLOB["1. Foreground blobs (07c)"]
    COL -.-> Y["2. YOLO on its own thread (05c)<br/>person boxes, object boxes"]
    COL --> MAPC["Colour mapped onto<br/>depth pixels (as in 08)"]
    MAPC --> CM["5. Colour memory<br/>learns only where nobody stands"]
    BLOB --> WHO{"3. Blob mostly inside<br/>person boxes?"}
    Y --> WHO
    WHO -->|yes| PAINT["4. Paint with the remembered colour<br/>except inside other objects' boxes"]
    CM --> PAINT
    WHO -->|no| KEEP["Stays visible"]
    PAINT --> O["Room with people removed<br/>never seen = magenta"]
    KEEP --> O

    classDef new stroke-width:3px
    class CM,WHO,PAINT new
```

**What to compare:** walk in carrying a cup: you should vanish and the cup should float. Press V
to see why each pixel was removed or kept.

## 08 Point Cloud

Starts from 03's maths (pixel + depth = X, Y, Z), done for every pixel at once. The room becomes
3D dots, in a small viewer we wrote ourselves.

```mermaid
flowchart LR
    K["Kinect"] --> DEP["Depth frame"]
    K --> COL["Colour frame"]
    DEP --> XYZ["X, Y, Z for every pixel<br/>(03's maths, cut-off 6 m from 07b)"]
    COL --> REG["Registration<br/>a colour for each depth pixel"]
    XYZ --> PTS["Coloured point cloud<br/>about 193,000 points"]
    REG --> PTS
    PTS --> VIEW["Own 3D viewer<br/>turn, project, nearest point wins"]
    PTS -->|P key| PLY[".ply file<br/>never commit it"]

    classDef new stroke-width:3px
    class XYZ,PTS,VIEW,PLY new
```

**What to compare:** turn the view sideways: the room is a thin shell, because the camera only sees
the front of everything.

## 09 Find the Floor

Starts from 08. Its points are in the camera's coordinates, so tilting the Kinect tilts the room.
RANSAC finds the floor plane, which gives the room's own "down".

```mermaid
flowchart LR
    PTS["Points (08)"] --> PICK["1. Pick 3 neighbouring points<br/>skip planes that aren't level"]
    PICK --> CNT["2. Count points<br/>within a few cm of the plane"]
    CNT -->|repeat a few hundred times| PICK
    CNT --> BEST["3. Keep the plane with the most points<br/>refit it to all of them"]
    BEST --> OUT["Camera height, tilt, roll<br/>height of any point above the floor"]

    classDef new stroke-width:3px
    class PICK,CNT,BEST,OUT new
```

**What to compare:** tilt the Kinect down so some floor shows, then check its reported height with a
tape measure.

## 10 Two Views

Starts from 08. One position only sees the front of things, so a map needs captures from several
places, placed by each capture's pose. Here the pose is measured by hand.

```mermaid
flowchart LR
    A["View A<br/>captured with SPACE (08)"] --> VIEW["One viewer<br/>A orange, B blue"]
    B["View B<br/>live, then frozen (08)"] --> POSE["Pose from the settings<br/>turn + shift into A's coordinates"]
    NUDGE["Keys nudge the pose"] -.-> POSE
    POSE --> VIEW
    POSE --> GAP["Median gap<br/>B's depth vs A's, along A's line of sight"]
    A --> GAP

    classDef new stroke-width:3px
    class POSE,NUDGE,GAP new
```

**What to compare:** move the Kinect a measured 30 cm and type it in: do the walls meet, or show
double? The gap can't see sliding along a wall: 5 cm sideways only raised it from 0.5 to 1.2 cm.

---

**Next (planned):** 11 ICP finds the pose by itself, 12 gets it from a printed marker, 13 merges
many views, and 14 places the tracked objects in that map. See the Roadmap in
[CLAUDE.md](../../CLAUDE.md).
