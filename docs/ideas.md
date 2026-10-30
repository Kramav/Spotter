# Ideas and techniques — not scheduled

A living document. It holds techniques we've looked into but haven't committed to: what each is,
when it would pay off, and how to build it. When one gets scheduled, move it to the Roadmap in
`CLAUDE.md`, and log the choice in `decisions.md`.

---

## Long-term goal: a 3D map of the space, in three stages

*Added 2026-10-25. The user's long-term goal is a 3D map; a fixed camera is the agreed starting
point.*

1. **Fixed camera (now).** Examples 01–06.
2. **Several fixed stations, merged offline (next step towards the map).** See below.
3. **Moving camera, continuous.** Motion compensation / KinectFusion, further down this file.

Stage 2 uses the same building blocks as stage 3 (deprojection, ICP, TSDF), just offline and over
a few views. Very little of it is wasted later.

## Multi-view registration: merging point clouds from several fixed positions

*Added 2026-10-25. **Scheduled 2026-10-25:** now Roadmap stages B–C (examples 07–13) in
`CLAUDE.md`. This entry keeps the background and how-to.*

**Idea:** capture the room from several known positions, convert each capture into a point cloud,
and merge them into one model.

**As a ladder of small examples** (one idea each):
1. Capture and average: still camera, average 30 frames, measure how much the noise drops.
2. Depth to point cloud: deproject one averaged frame and view it in 3D.
3. Two clouds, known pose: place a second view by hand-measured pose, and see the doubled walls.
4. ICP refine: fix exactly that misalignment.
5. ArUco pose: get the pose from a printed marker instead of a tape measure.
6. Merge several views (TSDF or downsampled points).

Saved point clouds are 3D captures of the room, so treat them like camera frames: keep people
out, never commit them, and gitignore `*.ply` / `*.pcd`.

**Pipeline:**
1. **Capture at each station** with the camera still. Average about 30 depth frames per pixel:
   temporal averaging *is* valid here because nothing moves. Take one colour frame, register it to
   depth, and build a coloured point cloud (deproject as in 02/03).
2. **Initial placement:** transform each cloud into room coordinates using its known pose.
3. **Refine pairwise with ICP**, starting from the known pose. This step is needed: a 1° pose
   error puts things about 5 cm out at 3 m, so walls show double. Use point-to-plane or *coloured*
   ICP, which helps on flat walls. Overlapping views only.
4. **Pose-graph optimisation** if the views go all the way around the room, so the loop closes
   without a seam.
5. **Merge:** TSDF integration, then extract a mesh. Or combine the points and voxel-downsample.

**Getting the poses:**
- Marked tripod spots plus measured angles. Simple, but expect cm/degree errors; ICP fixes them.
- **Printed ArUco markers in the room (recommended).** The colour camera sees a marker and
  OpenCV computes the camera pose automatically (`cv::aruco`, in the core objdetect module since
  OpenCV 4.7). It's accurate to about a centimetre, needs no tape measure, and the module is
  already in our OpenCV.
- No poses at all: global registration (FPFH features + RANSAC) works, but less reliably.

**Tools:** Open3D's reconstruction system already does fragments → registration → refinement →
integration (MIT, CPU, ARM64). Offline merging of 5–10 views is fine on a Pi, or capture on the Pi
and merge on the PC.

**Capture tips:**
- Overlap neighbouring views by 30–50%.
- Include furniture, not just bare walls, so ICP has shapes to lock onto.
- Stay within about 3 m for the best depth.
- Keep people out of the scene, or mask them with Spotter's detector before merging.
- Kinect v2 depth bulges slightly in room corners (multipath). Expect a few mm to cm there.

## Motion compensation / KinectFusion (3D mapping with a moving camera)

*Added 2026-10-25.*

**What it is:** work out how the camera moved between frames, line each new depth frame up with
the earlier ones in room coordinates, and average there. That gives smoothing over time that
survives camera motion, a clean, hole-free 3D model of the room, and the camera's path.
KinectFusion (Newcombe et al., Microsoft Research, 2011) is the classic version.

**When it pays off:** it's built for a *moving camera in a still room*. Moving objects are its
weakness, not its use case. What decides it is whether you need positions in room coordinates or
only relative to the camera:

| Situation | Need it? |
|---|---|
| Fixed camera, moving objects | **No.** The room already lines up frame to frame; use tracking (example 04, D12). |
| Moving camera, "how far is it from me?" | **No.** Distance is measured fresh each frame; example 04 covers it. |
| Moving camera, "where is it in the room?" or remember it after turning away | **Odometry only** (step 2 below, no grid). |
| Moving camera, want a 3D map of the space | **Yes, the full pipeline.** Mask detected people out before fusing, or they leave ghosts. |

**Why not now:**
- KinectFusion is built for a GPU. A 512³ grid is about 512 MB and updates at 30 fps; the Pi has
  neither the memory nor the compute.
- It assumes a still scene. Moving people smear into the model or leave ghosts.
- Tracking fails on flat, featureless views (a single wall) and on fast motion.

### How to build it

Each frame goes through four steps:

1. **Depth → 3D points.** Deproject every valid pixel with the pinhole model and the intrinsics
   from libfreenect2 `getIrCameraParams()` (the same maths as example 02). Also compute a surface
   normal per point from its neighbours (cross product of the right and down neighbours).
2. **Track the camera with ICP (Iterative Closest Point).** Find the 6-DoF motion (3 rotations,
   3 shifts) that best lines the new points up with the model's predicted surface. Use
   point-to-plane ICP, starting from the previous pose, with a coarse-to-fine image pyramid.
   Reject large jumps; they mean tracking was lost.
3. **Fuse into a TSDF grid.** For each voxel inside the camera's view, compute its signed
   distance to the measured surface along the camera ray. Truncate it to about ±3 voxels, then
   update a weighted running average: `tsdf = (tsdf*w + d) / (w+1)`, `w = min(w+1, maxW)`.
   `maxW` controls how fast the model forgets, which lets it cope with slowly changing scenes.
4. **Raycast the grid** from the new pose. The surface is where the TSDF crosses zero. That
   rendering is what step 2 matches the next frame against.

### Don't write it from scratch: Open3D

[Open3D](https://www.open3d.org/) is MIT-licensed, has C++ and Python APIs, and runs on a CPU.
It also builds on ARM64.

| Step | Open3D C++ building block (check against current docs) |
|---|---|
| Tracking | `pipelines::odometry::ComputeRGBDOdometry`, or `pipelines::registration::RegistrationICP` |
| Fusion | `pipelines::integration::ScalableTSDFVolume` (voxel hashing: memory only where surfaces are) |
| Output | `ExtractTriangleMesh()` / `ExtractPointCloud()` |

### A version scaled for the Pi

- Use coarse voxels (2–4 cm instead of about 5 mm) and a small volume, a few metres across.
- Track at reduced resolution (256×212), and fuse every 2nd–3rd frame.
- Expect a few fps, which is enough for a slow robot that maps as it goes.
- Cheaper middle ground: **odometry only** (step 2, no grid). That gives the camera's path,
  which is enough to put tracked objects into room coordinates.

### Alternatives to know about

- **Kintinuous**, **ElasticFusion**: larger spaces, with loop closure.
- **BundleFusion**: best quality, needs a strong GPU.
- **RTAB-Map**: SLAM (mapping while tracking position), BSD licence, ROS-friendly, runs on a Pi.
  The most realistic option if mapping on a moving robot becomes a real requirement.

## Rejecting phantom detections (a "person" on a screen, in a mirror, a poster)

*Added 2026-10-25.* With nobody in view, YOLO still reported a "person" now and then, with no
depth reading. That suggests a screen, a mirror, a window, or something beyond 4.5 m. It creates
ID churn.

**Idea:** the depth camera can check what colour can't:
- **No depth reading in the box:** probably a screen, mirror or window, so reject it, or at least
  don't start an ID.
- **Implausible real size:** box height in pixels × Z / fy should be about 1–2 m for a person. A
  photo of a person at 0.5 m comes out around 0.2 m.

This is example 02's experiment idea 6 and guide §10. A small example after 06, or a setting in a
later one.

## Distance beyond the depth camera's range, from the colour image

*Added 2026-10-30.* The depth camera reads 0.5–4.5 m, and the colour camera sees further. Ways to
estimate distance from colour alone, cheapest first:

0. **Raise the cap first.** 4.5 m is libfreenect2's default `Config::MaxDepth`, not the sensor's
   limit. Every example uses the default. Set `MaxDepth` to 8 m and measure the noise with 07.
   Readings past 4.5 m are reported to be noisier but usable.
   *Measured with 07b (2026-10-30):* a wall at 4–6 m read at ±5 mm, and almost nothing read past
   6 m. So this covers up to about 6 m, and colour is only needed beyond that.
1. **Known size:** Z = fy × real height / box height in pixels. Take the real height from depth
   while the object is in range, then keep using it after it leaves. Accuracy is about ±10–20%,
   because people's heights vary. Cheap.
2. **Floor contact:** if the floor plane is known (09), a ray through the bottom of the box hits
   the floor at the object's distance. Pixel error is small (about 6 cm at 8 m with the camera
   1 m high), but tilt error grows with Z²: a 1° tilt error is about 1 m at 8 m. So it needs 09.
3. **Monocular depth network** (Depth Anything V2 Small, Apache-2.0; MiDaS, MIT): depth for every
   colour pixel, relative only, scaled with the in-range depth pixels. Gives the whole room, not
   just objects, but it is several times heavier than YOLO and probably seconds per frame on the
   Pi.

Not usable: stereo or structure-from-motion, which need a second camera or a moving camera.

**When:** 1 and 2 fit after 09, as one small example ("distance past the depth range"). 3 only if
the room map needs far walls that the depth camera can't reach.

## Object scans: cut a detected object out and mesh it

*Added 2026-10-30.* Pull a detected object out of the depth data and turn it into a mesh you can
view. One camera position only sees the front of an object, so a closed mesh needs views from
around it.

**Plan: cheap cut-out live, segmentation once at export.**
- **Live, every frame (cheap):** the YOLO box (02/05c), plus the depth pixels inside it. Keep the
  points near the box's median depth (03's blob idea) and remove the table or floor plane (09).
  Good enough to show and track the object, and light enough for the Pi.
- **At export (one key press):**
  1. Freeze that frame: the full-size colour image and depth mapped onto it ("bigdepth", from
     `Registration::apply`).
  2. Run YOLO-seg once. If it finds several objects, pick the mask that overlaps the live box
     best.
  3. Mask × bigdepth gives the object's points with clean edges. Shrinking the mask by a pixel or
     two trims most of the flying pixels. The cheap cut-out was only the preview.
  4. Mesh and save `.ply`/`.obj`.
- **Why at export, not later from a saved file:** segmentation needs the colour image. Running it
  at export keeps the image in memory and writes only the object's points or mesh, so the privacy
  rule holds. Re-running it later means saving the colour image. That's the user's call; if so,
  crop it to the box, never save it for the "person" class, gitignore it and delete it after
  processing.

**Costs and risks:**
- **Speed:** one segmentation run is about 200 ms on the PC and probably a few seconds on the Pi.
  Run it in the background (05c) so the live view keeps going.
- **Model:** export `yolo26n-seg` like the detector (`nms=False`, fixed size). Whether OpenCV
  4.14 DNN loads it is untested; the dynamic-size detector export failed before, so test this
  first.
- **Mask decoding:** the model outputs mask coefficients plus prototype masks. Turning those into
  a mask is a matrix multiply, a sigmoid, a crop and a resize: about 30 lines.
- **Accuracy:** the colour and depth cameras don't line up perfectly, so mask edges can be off by
  a few mm at depth edges.
- **What the Kinect v2 can scan:** at 1 m one depth pixel covers about 2.7 mm, with about 1 mm of
  noise, so a chair or a box comes out well. A cup at 2 m gets only a few hundred points and will
  be blobby; averaging frames (07) helps. Black, shiny and see-through objects read badly, and
  nothing closer than 0.5 m reads.

**Meshing:**
- **One view:** connect neighbouring depth pixels into triangles and drop any triangle that spans
  a depth jump. That gives a 2.5D shell: the front surface only, but cheap to build.
- **All sides:** turn the object (a lazy Susan in front of the fixed camera), align the views
  with ICP (11) or a marker on the turntable (12), and merge them with a TSDF (13) into a closed
  mesh.
- **Viewing:** 08's viewer draws points, not triangles. Save `.ply`/`.obj` and open it in
  MeshLab, CloudCompare or Blender (all free and open source).

**Existing tools:** Open3D (MIT: TSDF, Poisson and ball-pivoting meshing, a viewer; runs on the
CPU, including the Pi). PCL (BSD: fast triangulation). Meshroom (MPL: photogrammetry from colour
photos alone; slow, and a different approach).

**As a ladder:**
- A. Cheap cut-out: live, the object's points in 08's viewer.
- B. Export with segmentation: one frame, YOLO-seg mask, clean points, saved `.ply`.
- C. Single-view mesh from B.
- D. Turntable scan: after 11–13.

**When:** A–C need only 02, 03, 08 and 09, so they could start any time. D comes after stage C.
Object scans are captures of the room: never commit them (`*.ply` is already gitignored; add
`*.obj` when C is built).

## Removing people: later refinements

*Added 2026-10-30.* 07c/07d remove people using a depth background plus YOLO person boxes, and
07e (on the Roadmap) will fill never-seen pixels. Not scheduled:
- **Moved objects via the tracker:** when the tracker (05) sees an object's ID move, refresh the
  memory at both spots at once, instead of waiting for `LeaveFrames` and `AbsorbFrames`.
- **YOLO-seg masks** instead of boxes: cleaner person outlines, and cup-only masks (no fingers).
  Heavy for every frame on the Pi; see "Object scans" for running it once instead.
- **Deep inpainting** (LaMa, Apache-2.0) for large never-seen areas: convincing, but far too heavy
  for the Pi live.
- **People-free room map:** feed the "removed" depth into stage C, so people don't end up as ghosts
  in the merged model.
