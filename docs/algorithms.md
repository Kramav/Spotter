# Algorithms used in Spotter

Every algorithm the code in `exercises/cpp` uses: the `spotter` library (`spotter/src`), and the examples' own
ideas (`examples/`). For each: where it lives, why it was chosen, and a source that explains how it works.
`legacy/` (the archive) uses the same ideas in older copies and isn't listed separately.

- **Source** is the original paper or the standard textbook treatment. Where we call OpenCV, the OpenCV
  documentation for that function (https://docs.opencv.org/4.x/) describes its exact variant.
- **Ours** means a heuristic written for this project with no outside paper; the decision (D-number in
  [decisions.md](decisions.md)) or the code comment is the explanation.
- The citations were written from memory (2026-11-14), not checked online: confirm the details before quoting
  them in anything published.

## 1. The camera: depth, colour, and 3D points

| Algorithm | Where | Why | Source |
|---|---|---|---|
| Time-of-flight depth (the Kinect v2 measures the phase of reflected infrared light) | The sensor itself, decoded by libfreenect2 (`source_freenect2`) | It's the depth camera we have; libfreenect2 is open source and runs on Linux (D5, D8) | Sell & O'Connor, "The Xbox One System on a Chip and Kinect Sensor", IEEE Micro 34(2), 2014. libfreenect2: https://github.com/OpenKinect/libfreenect2 |
| Pinhole camera model: unprojecting a pixel and its depth to a 3D point, and projecting back | `frame` (`PinholeCamera::Unproject`), everywhere | The standard model; the Kinect reports its own focal lengths and centre | Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., 2004, ch. 6 |
| Depth-to-colour registration (where each depth pixel lands in the colour picture), with the occlusion filter | libfreenect2's `Registration::apply`, used by `source_freenect2` | Detection runs on colour and distance comes from depth, so every box needs the depth pixels under it (the guide's RGB-D fusion) | libfreenect2 `registration.cpp`; Hartley & Zisserman ch. 9 (two calibrated views) |
| Undoing the Kinect's mirror image (flip the pictures, remap the index, `cx' = width - cx`) | `frame` (`Unmirror`) | The Kinect sends mirror images; left and right were swapped in every scan (D35, D36) | Ours (D35) |
| Averaging depth over N frames | `depth` (`DepthAverager`), 07, every scan | Random noise falls with the square root of N (±1.2 mm → ±0.3 mm measured, 07); pixels that only sometimes read get filled | Standard error of the mean, any statistics text; e.g. Taylor, *An Introduction to Error Analysis*, 1997, ch. 4 |
| A colour camera model fitted from the registration (six numbers, linear least squares by SVD) | `room_scan` (`FitColourModel`) | Projects room points into the 1920 × 1080 colour picture for far features and 13w's textures; libfreenect2 doesn't hand over the full model | Hartley & Zisserman ch. 7 (camera resection by linear least squares, the DLT) |

## 2. Detection and tracking (examples 02–07d)

| Algorithm | Where | Why | Source |
|---|---|---|---|
| YOLO object detector (Ultralytics YOLO26 nano, ONNX, run by OpenCV DNN) | `detector_yolo_opencvdnn` | A small, fast single-pass detector that runs on a CPU (D11); OpenCV DNN avoids another dependency | Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection", CVPR 2016; the model: https://docs.ultralytics.com |
| End-to-end output with no NMS (300 boxes out), or classic output plus non-maximum suppression per class | `DecodeYolo` (`cv::dnn::NMSBoxesBatched` for the classic layout) | The ONNX was exported with `nms=False` and outputs 1×300×6; the classic layout is kept for other models | NMS-free heads: Wang et al., "YOLOv10: Real-Time End-to-End Object Detection", NeurIPS 2024. NMS: Neubeck & Van Gool, "Efficient Non-Maximum Suppression", ICPR 2006 |
| Letterbox resize (scale to fit, pad, map boxes back) | `detector_yolo_opencvdnn` | The model wants a square input; stretching would distort objects | Ultralytics documentation (preprocessing) |
| Median depth in a box, from readings filed by a counting sort into 32-column cells | `depth` (`BoxDepthSampler`), `stats` (`Median` by `std::nth_element`) | The median ignores the background and edge pixels a box always includes; the cells make it 2.0 ms instead of 5.1 ms per frame with 5 boxes | Median as a robust estimator: Huber, *Robust Statistics*, 1981. Selection: Hoare, "Algorithm 65: Find", CACM 1961. Counting sort: Cormen et al., *Introduction to Algorithms* (CLRS), ch. 8 |
| Intersection over union (box overlap) | `track` (`Overlap`) | The usual measure of whether two boxes are the same object | Everingham et al., "The PASCAL Visual Object Classes (VOC) Challenge", IJCV 2010 |
| Greedy matching by best overlap, then a Kalman filter per box (predict, match against predictions, correct, coast) | `track` (`Tracker`, `BoxFilter`), 04–05c | Keeps IDs through fast moves and short misses; greedy is enough for a handful of objects (D12) | Bewley et al., "Simple Online and Realtime Tracking" (SORT), ICIP 2016; Bochinski et al., "High-Speed Tracking-by-Detection Without Using Image Information", AVSS 2017 |
| Kalman filter (constant velocity) | `track`, 05 and its ladder | Smooths jitter and predicts where a box will be (jitter 2–4 mm → 0.4 mm, 04 → 05) | Kalman, "A New Approach to Linear Filtering and Prediction Problems", J. Basic Engineering 82(1), 1960; Welch & Bishop, "An Introduction to the Kalman Filter", UNC TR 95-041 |
| Two confidence thresholds (strong detections start tracks, weak ones only continue them) | `track`, 05b | YOLO's confidence drops for a moment; one threshold made 7 IDs for one TV in 60 s, two made 1 | Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022 |
| Detection on a background thread while the main loop predicts and re-measures every frame | `AsyncDetector`, 05c | YOLO takes 130–180 ms; the display went from 4 to 24–27 fps (D15) | Ours (a producer/consumer thread); see Williams, *C++ Concurrency in Action*, 2nd ed., 2019 |
| Per-pixel depth background with "leave" and "absorb" frame counts | `depth` (`AdaptiveBackground`), 07c, 07d | Finds what's in front of the remembered room; the counts stop a spinning fan from making blobs (07c) | Ours; the family it belongs to: Piccardi, "Background subtraction techniques: a review", IEEE SMC 2004 |
| Connected components (blobs), with sizes | `depth` (`KeepBigBlobs`), 03, 07c, `FindPlanes` | Group foreground pixels into objects and drop specks | Rosenfeld & Pfaltz, "Sequential Operations in Digital Picture Processing", JACM 1966; OpenCV `connectedComponentsWithStats` |
| Morphology (erode, dilate, close) | 03, 07c, `room_scan` (feature masks), 13x | Clean masks; keep features clear of depth holes' edges | Serra, *Image Analysis and Mathematical Morphology*, 1982; OpenCV `morphologyEx` |
| Flood fill from a clicked point, stopping at depth jumps | 08b (click anywhere), 13x (`PieceDomains`) | Grow the object under a click when no box is there; find a surface's enclosed area | Heckbert, "A Seed Fill Algorithm", *Graphics Gems*, 1990; OpenCV `floodFill` |

## 3. Planes, walls and floors (examples 06b–09c)

| Algorithm | Where | Why | Source |
|---|---|---|---|
| RANSAC: a plane through three random points, keep the one most points agree with | `geometry` (`FindPlanes`), 09 | Fits a plane through noise and clutter; three neighbouring cells, because three random points rarely all hit a small floor (09) | Fischler & Bolles, "Random Sample Consensus", CACM 24(6), 1981 |
| Least-squares plane by principal component analysis (the direction of least spread is the normal) | `geometry` (`FitPlane`) | Refits RANSAC's plane to all its points | Hoppe et al., "Surface Reconstruction from Unorganized Points", SIGGRAPH 1992 (normals by PCA) |
| Walls as upright planes turned into floor lines (2nd–98th percentile ends) | `walls` (`FindWalls`), 06b | Walls from depth alone; percentiles so stray points don't stretch a wall (D18) | Ours, on RANSAC planes |
| Extended Kalman filter per wall (state: the foot's X, Z and the ends; measurement: distance and angle; Joseph-form update) | 06c, Demo 3 | Walls don't move, so filtering removes the wobble (3–6× less); the measurement is non-linear (hypot, atan2), hence the EKF; the Joseph form stays symmetric under rounding (D19, bug 9) | Thrun, Burgard & Fox, *Probabilistic Robotics*, 2005, ch. 3 (EKF) and ch. 7 (range-bearing). Joseph form: Bucy & Joseph, *Filtering for Stochastic Processes with Applications to Guidance*, 1968 |
| Top-down radar: objects and walls on a floor plan, with trails | `radar`, 06 | The first "map" (stage A) | Ours |
| Room model: planes sorted into floor, ceiling and walls, levelled, walls joined at corners, the rest as boxes | `room` (`BuildRoom`), 09c, 13e's E | A clean, editable model of a room from noisy scans (D20, D28) | Ours, on RANSAC planes |
| The floor: walls' foot from upright point density, level pieces in 2 cm bins, the lowest big piece near the foot | `room_twin` (`FindTwinFloor`) | A reflection or a low patch was taken as the floor (bug 7, D33); the walls' foot was measured too high (D43) | Ours (D33, D43) |
| The ceiling: the highest 2 cm step where level readings cover 0.5 m² | 13w (`FindCeiling`) | A fan or a lamp hangs lower and covers less, so it never wins (D42) | Ours (D42) |
| "Up" from structure: the biggest level plane, else two walls' cross product, else one wall | `room`, `structure` (`ExtractStructure`) | Level the room before sorting surfaces; checked side by side with the floor rules (D54) | Ours |

## 4. Placing scans and merging a room (examples 10–13z)

| Algorithm | Where | Why | Source |
|---|---|---|---|
| Rotation from an axis and angle (Rodrigues' formula) | `geometry` (`TurnAbout`) | The small turn ICP solves for is an axis-angle vector | Murray, Li & Sastry, *A Mathematical Introduction to Robotic Manipulation*, 1994, ch. 2 |
| Kabsch / SVD: the best turn and shift between paired points, a reflection made a turn | `geometry` (`Kabsch`), 11, 11b, `FeaturePose` | Exact, closed form, no starting guess | Kabsch, Acta Crystallographica A32, 1976; Arun, Huang & Blostein, "Least-Squares Fitting of Two 3-D Point Sets", IEEE PAMI 9(5), 1987; Umeyama, IEEE PAMI 13(4), 1991 (the reflection case) |
| k-d tree nearest neighbours (FLANN) | `register` (`Icp`), 11 | ICP pairs thousands of points every round; 11_s1 measured it 4× faster than checking every point | Bentley, "Multidimensional Binary Search Trees", CACM 1975; Muja & Lowe, "Fast Approximate Nearest Neighbors with Automatic Algorithm Configuration", VISAPP 2009 |
| ICP, point-to-point | 11 (inline, as the lesson) | The simplest registration; shows why a good start matters (stalls past ~15 cm / 8°, 11_s3) | Besl & McKay, "A Method for Registration of 3-D Shapes", IEEE PAMI 14(2), 1992 |
| ICP, point-to-plane, linearised 6 × 6 least squares | `register` (`Icp`), 13e, Demos 4/6 | Converges in far fewer rounds on flat rooms; distance along the normal lets surfaces slide into place (D21) | Chen & Medioni, "Object Modelling by Registration of Multiple Range Images", Image and Vision Computing 10(3), 1992; Low, "Linear Least-Squares Optimization for Point-to-Plane ICP", UNC TR04-004, 2004; Rusinkiewicz & Levoy, "Efficient Variants of the ICP Algorithm", 3DIM 2001 |
| Weak directions: a direction the shape doesn't pin (a bare wall) keeps the starting value | `register` (`Icp`), with the turn scaled by the pairs' radius (bug 13) | A bare wall lets ICP slide anywhere along it; drift on noise placed scans wrong (D26) | Gelfand et al., "Geometrically Stable Sampling for the ICP Algorithm", 3DIM 2003; Zhang, Kaess & Singh, "On Degeneracy of Optimization-based State Estimation Problems", ICRA 2016 |
| Surface sampling with normals from neighbours' cross product | `register` (`SampleSurface`) | One rule for every scan, live or loaded, so a rebuild equals a live merge (bug 3) | Standard for organised (grid) point clouds; Holz et al., "Real-Time Plane Segmentation Using RGB-D Cameras", RoboCup 2011 (normals from neighbours) |
| ORB features: FAST corners + rotated BRIEF descriptors | `features_orb`, 11b, `room_scan` | Free (unlike SIFT/SURF until recently), fast on a CPU, rotation-aware (D20) | Rublee et al., "ORB: An Efficient Alternative to SIFT or SURF", ICCV 2011; Rosten & Drummond, ECCV 2006 (FAST); Calonder et al., ECCV 2010 (BRIEF) |
| CLAHE (contrast-limited adaptive histogram equalisation) before ORB | `features_orb` (`evenContrast`), 13b onwards | A dim room gave about 60 features (refused); with CLAHE, about 1,250 (D24) | Zuiderveld, "Contrast Limited Adaptive Histogram Equalization", *Graphics Gems IV*, 1994 |
| Brute-force Hamming matching with Lowe's ratio test (0.8) | `features_orb` (`MatchFeatures`) | Drops ambiguous matches (repeated textures) before RANSAC | Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", IJCV 60(2), 2004 |
| RANSAC over mixed matches (3D–3D, ray–ray, ray–point) for a first pose | `room_scan` (`FeaturePose`) | Features give ICP the start it needs; rays add far, depth-less features in hallways (D27) | Fischler & Bolles 1981; Hartley & Zisserman ch. 4 (robust estimation) |
| Far landmarks: two rays' closest points, midpoint, with a minimum parallax | `room_scan` (`AddLandmarks`) | Pins later scans in hallways where depth reads nothing (D27) | Hartley & Sturm, "Triangulation", CVIU 68(2), 1997 (the midpoint method and its limits) |
| Shortlist by descriptor votes (each feature votes for the scan holding its nearest twin) | `room_scan` (`Shortlist`) | Matching against every scan grew slow (13b: 3–45 s); voting picks the few worth matching (D26) | The idea of image retrieval by feature votes: Sivic & Zisserman, "Video Google", ICCV 2003 |
| Second-opinion ICP and a "suspicious" flag | `room_scan` (`PlaceScan`) | A scan placed against one wrong match is caught by checking against the others (D26) | Ours |
| Next shot: 7 turns × 3 tilts from every spot, scored by new view share minus distance | `room_scan` (`RecommendNext`) | Tells the person scanning where to point next (D26) | Connolly, "The Determination of Next Best Views", ICRA 1985 (the next-best-view problem) |
| TSDF fusion: a grid of signed distances, averaged over scans | `room` (`FuseScans`), 13e's E, 13z, `room_twin` | Merges many noisy scans into one surface and averages the noise out (D21) | Curless & Levoy, "A Volumetric Method for Building Complex Models from Range Images", SIGGRAPH 1996; Newcombe et al., "KinectFusion", ISMAR 2011 |
| Surface nets: one point per cube where the sign changes, joined into triangles | `room` (`FuseScans`) | Simpler than marching cubes, no lookup tables, smooth output | Gibson, "Constrained Elastic Surface Nets", MICCAI 1998 |
| Grid triangles from a depth picture, skipping depth jumps over 3% | `mesh` (`MeshFromGrid`), 08b, 09b | A mesh straight from one view; a fixed mm limit tore slanted far walls (D20) | Standard for organised point clouds; ours for the 3% rule |
| Thinning to one point per cube (voxel-grid downsampling; the cube's three indices packed into one 64-bit key, 21 bits each) | `room_scan` (`CubeKey`), 13w, `structure` | Even spread, so a wall seen up close doesn't outweigh one seen once (D43) | Rusu & Cousins, "3D is here: Point Cloud Library (PCL)", ICRA 2011 (its VoxelGrid filter is the same idea) |
| The test room's colour blocks (a spatial hash of the block's indices) | `checks/test_room.hpp` | Random but repeatable colours, so ORB finds features on every synthetic wall | Teschner et al., "Optimized Spatial Hashing for Collision Detection of Deformable Objects", VMV 2003 (the three primes) |

## 5. Finding the Kinect in a known room (Demos 4/6, 13y)

| Algorithm | Where | Why | Source |
|---|---|---|---|
| Multi-start ICP (up to 63 starts around scan 1's spot) | `room_twin` (`LocateKinect`) | ICP finds only the nearest fit; many starts find the right one (D33) | Besl & McKay 1992 (local minima); a standard remedy |
| Relocating from structure: floor turns up onto up, wall pairs give yaw and position in closed form, ICP polishes | `structure` (`Relocate`), 13y | The localisation study: can planes alone place a scan? (right and unique on 8 of 21 real scans, confidently wrong on 6) | Ours; the study: [structure-localization-findings.md](structure-localization-findings.md) |
| View agreement: penalise seeing through a mapped surface | `structure` (`ViewAgreement`) | ICP's share can't tell one corner from another; seeing through a wall can | Free-space (visibility) consistency, as in Huber & Hebert, "Fully Automatic Registration of Multiple 3D Data Sets", Image and Vision Computing 21(7), 2003 |
| Plane-pair edges (the line where two planes meet, 2nd–98th percentile stretch) | `structure` (`FindEdges`) | Corners as features for the study | Ours |

## 6. Textures and filling what nobody saw (13w, 13x, Demo 6)

| Algorithm | Where | Why | Source |
|---|---|---|---|
| Per-texel colour from the best scan (depth agreement, colour-camera check, odd colours dropped) with bilinear sampling | 13w (`Texture`, `Sight`, `Bilinear`) | About 4× sharper colour than the merged mesh's one colour per corner (D42) | Bilinear interpolation: Press et al., *Numerical Recipes*, 3rd ed., 2007, §3.6. Picking views for texturing: Waechter, Moehrle & Goesele, "Let There Be Color!", ECCV 2014 |
| Inpainting (Telea's fast-marching method) | 13w (`Fill`, the slow colour changes), 13x (`PaintIn`) | Fills holes with smoothly varying colour from the edges | Telea, "An Image Inpainting Technique Based on the Fast Marching Method", J. Graphics Tools 9(1), 2004; OpenCV `inpaint` with `INPAINT_TELEA` |
| Image quilting (blocks of the surface's own pattern, each the best at matching its overlap; no seam cut) | 13w (`Fill`, `Exemplar`) | Inpainting alone smears; a floor needs its planks back (D42) | Efros & Freeman, "Image Quilting for Texture Synthesis and Transfer", SIGGRAPH 2001 |
| Plane fill: each flat surface's enclosed cells that no point is near, marked as guessed | 13x (`FillSurface`, `FillWalls`), Demo 6 | Shows a room without holes and says which parts are made up (D31, D32) | Ours (D31, D32) |

## 7. Rendering and files

| Algorithm | Where | Why | Source |
|---|---|---|---|
| Z-buffered point rendering with an orbit camera | `view` (`PointRenderer`, `OrbitView`) | A 3D viewer with no extra library (08) | Catmull, *A Subdivision Algorithm for Computer Display of Curved Surfaces*, PhD thesis, Utah, 1974 (the z-buffer) |
| OBJ and binary PLY writers | `mesh` (`SaveObj`, `SavePly`) | Opens in Blender and MeshLab; binary PLY because 5 million points would be ~150 MB of text | Turk, "The PLY Polygon File Format", 1994; Wavefront OBJ (Appendix B1 of the Advanced Visualizer manual) |
| Safe saves: write beside, push to disk (fsync / `_commit`), then rename over; the completion marker last | `room_folder` (`WriteScan`, `ReplaceText`) | A crash or blue screen mid-save must never lose an earlier save (D45, bug 4) | Pillai et al., "All File Systems Are Not Created Equal: On the Complexity of Crafting Crash-Consistent Applications", OSDI 2014 |

## Keeping this current

When an example or module adds an algorithm, add a row here with its source. An idea with no outside source
is marked "Ours" and points to its D-entry.
