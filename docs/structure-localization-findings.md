# Structure-based localisation: findings (2026-11-08)

**The question:** can room structure (floor, ceiling and walls, and the edges where they meet)
place a new scan in the room's map? Continuous mode places a scan against the previous one;
Relocate places it from an unknown spot. Should ceilings be used?

**The short answer:**
- **Not yet as the main source of truth.**
  - Synthetic rooms: Relocate never gave a wrong unique answer.
  - The real room: it was right and unique on 8 of 21 scans, and **confidently wrong on 6**.
  - It needs a verification step; colour features are the one that worked in the tests.
- **Continuous (ICP from the last scan's pose) failed on all 20 real pairs.** The Kinect was turned
  about 40° between shots, twice ICP's capture range. It needs small steps, live tracking, or
  features first (section 8).
- **Edges add nothing beyond their planes,** and on real data they disagreed by 8–15 cm between scans.
- **Ceilings:** use them for "up" and height (they gave up on 10 of 21 real scans, more than the
  floor), but don't build on ceiling edges.

Labels: **(a)** a published method; **(b)** our own inference or design. Nothing here has been
tested beyond what each section says.

## 1. What exists already (survey of the code)

- **Storage:** room folder format 2 (`spotter/room_folder.hpp`, D38). Per scan:
  - `cloud.bin`, a 3D point per depth pixel;
  - `pose.txt`, the scan's camera → the room (scan 1's camera);
  - pictures and features.
- **Poses:** `spotter::Pose {turn, shift}`, in `geometry.hpp`.
- **Planes:**
  - `FindPlanes`: repeated RANSAC (a) on neighbouring cells, keeping each plane's biggest connected
    piece (b).
  - `BuildRoom`: merges planes across scans, sorts them into floor/ceiling/walls, and joins walls at
    corners. These are 2D wall-wall crossings; there was no 3D edge code.
- **Registration:**
  - `Icp`: point-to-plane (a, Chen & Medioni 1992; Low 2004). Directions the view can't pin are left
    alone (a, the degeneracy analysis of Gelfand et al. 2003 and Zhang et al. 2016).
  - 13e places scans by colour features + RANSAC + ICP.
- **An existing "relocate":** legacy Demo 4/6 `Locate`, multi-start ICP. It only searches ±30° and
  ±0.4 m around scan 1's spot, so it is not a true unknown-location search.
- **New for this study** (`spotter/structure.hpp`, a prototype):
  - per-scan structure and edges;
  - `Relocate`, ranked by `ViewAgreement`;
  - two tie-breakers (`ColourVotes`, `OffPlaneShare`);
  - `PlaceAfter` (Continuous).
- **The measurements:** `check_structure` (synthetic rooms) and `13y_Structure_Survey` (real
  rooms, numbers only).

## 2. Coverage on the real room

One room was available: `room_2026-11-07_2338`, 21 scans from 13e on the sensor. Only `cloud.bin`,
`pose.txt` and `room.txt` were read. Surfaces were classified using the room's up from `room.txt`.

| Surface | Scans that see it | Area per scan (mean) | Readings per m² |
|---|---|---|---|
| Walls | 20 of 21 (95%) | 5.07 m² | 19,445 |
| Floor | 9 of 21 (43%) | 0.51 m² | 30,150 |
| Ceiling | 8 of 21 (38%) | 2.00 m² | 11,519 |

- 2+ wall directions in view: 12 of 21 scans (57%).
- A corner with floor, which pins all 6 directions: 4 of 21 (19%).
- A ceiling-wall edge in view: 8 of 21 (38%).
- **Ceilings come all or nothing.** The 13 scans without ceiling see none of it. The 8 with it see
  a lot: up to 12 m² in one scan. In this room the ceiling was a bigger level plane than the floor
  (`room.txt`: "up from the ceiling"), and its density varies 4,500–31,000 readings/m² with
  distance.
- **The floor is seen less often and in small patches** (0.3–3 m²).
- **On synthetic rooms** a level camera 1.2 m up sees almost no floor, and tilting 15° down
  brings it into most views. So how a person holds the Kinect decides floor coverage. (b)

## 3. Edges between consecutive scans (real room)

Edges are plane intersections (b). Each edge of a scan was matched to the nearest edge of the same
kind in the next scan, within 30 cm and 10°:

| Edge | Found | Seen again next scan | Offset (median) | Angle (median) |
|---|---|---|---|---|
| floor-wall | 34 | 3 | 109 mm | 6.4° |
| ceiling-wall | 15 | 3 | 77 mm | 3.7° |
| wall-wall | 16 | 5 | 153 mm | 3.6° |

**Edges were unstable on this data.** Few were seen again by the next scan (consecutive scans
mostly look different ways), and those that were disagree by 8–15 cm.
- On synthetic scans the same measure gives 0.3 mm and 0.02°, so the measure itself works.
- **Unexplained so far.** Either the stored poses (13e's own estimates, not ground truth) disagree
  by that much between neighbours, or a scan's small floor patch gives a poor plane.
- **To verify:** compare the same wall's plane offset between scan pairs directly. Then re-place
  the pairs by ICP and see whether the offsets drop.

**An edge adds no constraint beyond its two planes:** the line is fixed by them (b). Edges only add
information through their *endpoints* (where a wall stops at a doorway or corner). Those endpoints
depend on occlusion, and the numbers above suggest they're not yet reliable here.

## 4. Can structure pin all six directions? (b)

- **Floor (or ceiling) alone:** tilt, roll and height.
- **Each wall direction:** the turn about up and one sideways position.
- **Two non-parallel wall directions plus floor/ceiling:** all 6.
- **One wall direction:** sliding along it is free.
- **The ceiling adds no direction the floor doesn't:** it's parallel to the floor. It helps only
  when the floor isn't seen. That happened in 6 of the 8 real scans that saw the ceiling.
- **On real data:** 57% of scans had 2+ wall directions, and only 19% a full corner with floor.

## 5. Relocate: candidates, scoring and symmetry (synthetic)

**The method:**
1. The scan's floor (or ceiling, or walls) gives up and height.
2. Each pair of scan walls is matched to each pair of map walls at the same angle, which gives the
   turn and position in closed form (b). This is the plane-correspondence idea of Pathak et al. 2010
   and Khoshelham 2016 (a).
3. Each candidate is polished by ICP.
4. **Ranking is by view agreement:** the map, seen from the candidate, against the scan's depth.
   Pixels where the scan sees *through* a map wall count against it. This is projective data
   association, as in KinectFusion (Newcombe et al. 2011) (a).
5. ICP's own paired share is useless for ranking. A view of one corner pairs perfectly with every
   same-angle corner, because every scan point lies on *some* map wall.

**Results** (`check_structure`; rooms built from a floor plan, scanned 15° down):

| Room / view | Result |
|---|---|
| L-shaped room, 4 views | 2 right and unique; 1 ambiguous (a corner, alike to the room's others), colour picks the truth; 1 ambiguous with no true candidate (one wall in view, no floor: up guessed 7.6° off); **0 wrong-and-unique** |
| 4 × 3 m rectangle | **ambiguous** (2-fold symmetry; all 4 rivals agree +1.000); colour votes pick the truth (18 vs ≤ 2) |
| 3.5 m square | **ambiguous** (4-fold); colour votes pick the truth (222 vs ≤ 2) |
| rectangle + couch + doorway | unique and right: the couch and the doorway break the tie |
| floor + one wall, close | under-constrained, says so, ambiguous |
| a corner view in the L | ambiguous among the L's corners; colour picks the truth |

**Disambiguation:**
- **(i) Colour features** (count matches that land within 10 cm under each candidate) resolved
  every synthetic tie where the truth was among the rivals. Caveat: the synthetic walls are
  patterned, which flatters colour. Real blank walls won't be.
- **(ii) Off-plane points** (furniture): no signal on these synthetic views. That's untested on
  furniture-rich scenes.
- **(iii) Ask the user,** when there's no true candidate.

**Time:** 1–6 s per Relocate, and 10–26 s for one-wall views (many slid candidates).

## 6. Relocate on the real room, and the baselines

Leave-one-out: each of the 21 scans is relocated in a map built from the other 20. "Agrees" means
within 5 cm and 2° of 13e's stored pose.

| Method | Agrees, unique | Ambiguous, truth among rivals | No answer / ambiguous without truth | **Wrong, unique** |
|---|---|---|---|---|
| Structure Relocate | 8 | 1 | 6 | **6** |
| Demo 4 multi-start (63 starts around scan 1's spot) | 4 | – | – | 17 off |
| Wide multi-start (240 starts: every other scan's spot × 12 turns) | 8 | – | – | 13 off |

- **The synthetic safety property does not hold on real data.** Relocate gave 6 wrong answers
  presented as unique (scans 5, 8, 12, 14, 16, 18):
  - In 4 of them (5, 8, 12, 14) the wide baseline agrees with the stored pose, so the stored pose is
    consistent and Relocate is the one that's wrong.
  - For 16 and 18 nothing agrees, so either method could be wrong, or the stored pose could be.
  - **Likely causes (b), not yet checked:**
    - a real room has many more wall-like planes (cabinets, furniture faces), giving 19–391
      candidates per scan;
    - `ViewAgreement` can't see past furniture the map lacks;
    - the 0.05 ambiguity margin was tuned on clean synthetic rooms.
  - `13y_Structure_Survey room_scans view N` shows any of them in 3D, against where 13e placed it.
- **No answer at all on 6 scans:** 3 saw no wall (scans 6, 9, 15), 1 saw ceiling only (11). Scans
  10 and 13 were ambiguous without the truth.
- **The baselines are worse:**
  - Demo 4's search (±30°, ±0.4 m) finds only 4 scans: real shots are rarely near scan 1's spot.
  - The wide search finds 8, at several times the cost.
  - Structure or the wide search agrees on 14 of 21 between them.
- **Time per scan** (measured while other builds and tests ran, so only rough): structure about
  30 s, Demo 4 about 85 s, wide about 300 s, for scans 11–21.
- **The stored poses aren't ground truth:** each was placed by 13e with colour features, which the
  colour-free methods here don't use.

## 7. Continuous mode

`PlaceAfter` runs ICP against the previous scan only, from the previous pose. In the synthetic L
(scanned 15° down), "recovered" means within 1 cm and 0.5°:

- **A view with a corner** recovers every move up to 30 cm and 20°, and most up to 45 cm and 10°.
  At 30° turns it fails.
- **A view of one wall** fails almost everywhere, by the along-wall part of the move. ICP flags those
  directions as weak, so the failure can be *detected* and turned into "this scan needs Relocate".

**On the real room** (20 consecutive pairs; "agrees" = within 5 cm and 2° of 13e's stored pose):

| First guess | Agrees |
|---|---|
| The previous scan's pose | **0 of 20** |
| The recommended next shot (`RecommendNext`, as it stood before that scan) | 5 of 20 |

- **Between shots the Kinect was mostly turned, not moved:**
  - median move about 5 cm;
  - turns of 16–176°, with a median of about 40°.

  The synthetic capture range is about 20°, so every pair started outside it. 13e never needed
  small steps, because colour features place a scan from any turn.
- **Following the advice helps when it was followed:**
  - the 5 that agreed started 1–8 cm and 7–15° from the advice;
  - where the user turned 30–50° away from the advice, it failed;
  - so did some pairs that started within 15° of it (scans 2, 4–7, 18).
- **Weak directions:** ICP flagged 1–3 weak directions on most failed pairs, so Continuous can say
  "lost" rather than return a wrong pose (b). This needs a threshold test before relying on it.
- **Conclusion (b):** Continuous as specified (ICP from the last pose) does not fit how this user
  scans. It needs one of three things:
  - steps under about 20° and 30 cm, enforced by the UI;
  - live frame-to-frame tracking between shots (ideas.md), so the guess is always close;
  - 13e's colour features as the first guess (what 13e already does), with structure ICP as the polish.

## 8. Recommendations

All (b), from the numbers above.

1. **Don't make structure the main source of truth yet.**
   - On the real room it was right and unique on 8 of 21 scans, and confidently wrong on 6.
   - Keep colour features (13e's placement) as the main method.
   - Use structure for **checking** a placement (does the view agree with the map's planes?) and
     for **up and height**.
2. **Relocate needs a verification step before it can be trusted:**
   - colour votes (`ColourVotes`), which picked the truth in every synthetic tie;
   - or ICP's weak directions plus `ViewAgreement` thresholds tuned on real scans.

   Next experiment: re-run section 3 with colour votes as a tie-breaker on every answer, not only on
   ambiguous ones (this reads `features.yml`, which needs your OK). Also look at the 6 wrong scans
   in 3D.
3. **Continuous (13f) as specified won't work for how the room was scanned:**
   - 0 of 20 pairs from the last pose, because turns between shots were typically 40°;
   - choose one: (a) the UI asks for small steps (under 20° and 30 cm) and refuses bigger ones,
     (b) live frame-to-frame tracking between shots (ideas.md; roadmap 16), or (c) features first,
     ICP second, which is 13e today;
   - my recommendation is (c) now, and (b) when 16 is built.
   - The recommended-shot guess helped (5 of 20). It's worth keeping as an option, but not on its own.
4. **Ceilings: use them for up and height, not for edges.**
   - In this room the ceiling gave up on 10 of the 21 Relocate attempts, more than the floor
     (9), because the Kinect sees it in big pieces when tilted up.
   - It adds no new direction beyond the floor (section 4).
   - Ceiling-wall edges were re-seen in only 3 of 15 cases, 8 cm apart: don't build on them.
5. **Don't build on edges in general** until the 8–15 cm disagreement between consecutive scans
   (section 3) is explained.
6. **Before building 13f:** decide 3 (which Continuous), then 2 (which verification).

## 9. Doubtful assumptions, and what's untested

- **No ground truth on real data:** agreement with 13e's stored poses means consistent, not
  accurate.
- **One real room:** coverage numbers will differ with room size, how the Kinect is held, and
  ceiling height.
- **Synthetic walls are patterned**, which favours colour; real walls may be blank.
- **The sensor is a Kinect v2 time-of-flight camera, not LiDAR:** about 0.5–6 m range, a roughly
  70° × 60° view.
- **No IMU:** up comes from the floor, the ceiling or the walls. A single wall with no floor gives
  up only roughly (assumes no camera roll).
- **Untested:** `OffPlaneShare` on furniture, edge endpoints as landmarks, and anything on the
  Raspberry Pi.

## Sources

- (a) Fischler & Bolles 1981, *Random sample consensus*, Communications of the ACM 24(6) —
  [doi:10.1145/358669.358692](https://doi.org/10.1145/358669.358692)
- (a) Schnabel, Wahl & Klein 2007, *Efficient RANSAC for point-cloud shape detection*, Computer
  Graphics Forum 26(2) — [Eurographics library](https://diglib.eg.org/handle/10.2312/CGF.v26i2pp214-226)
- (a) Pathak, Birk, Vaškevičius & Poppinga 2010, *Fast registration based on noisy planes with
  unknown correspondences for 3-D mapping*, IEEE T-RO 26(3) —
  [doi:10.1109/TRO.2010.2042989](https://doi.org/10.1109/TRO.2010.2042989)
- (a) Khoshelham 2016, *Closed-form solutions for estimating a rigid motion from plane
  correspondences extracted from point clouds*, ISPRS J. Photogramm. Remote Sens. 114 —
  [ISPRS archives version](https://isprs-archives.copernicus.org/articles/XLI-B3/403/2016/)
- (a) Chen & Medioni 1992, *Object modelling by registration of multiple range images*, Image and
  Vision Computing 10(3) — [doi:10.1016/0262-8856(92)90066-C](https://doi.org/10.1016/0262-8856(92)90066-C)
- (a) Low 2004, *Linear least-squares optimization for point-to-plane ICP surface registration*,
  UNC TR04-004 — [PDF](https://comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf)
- (a) Gelfand, Ikemoto, Rusinkiewicz & Levoy 2003, *Geometrically stable sampling for the ICP
  algorithm*, 3DIM — [project page](https://www.cs.princeton.edu/~smr/papers/stabicp)
- (a) Zhang, Kaess & Singh 2016, *On degeneracy of optimization-based state estimation problems*,
  ICRA — [PDF](https://frc.ri.cmu.edu/~zhangji/publications/ICRA_2016.pdf)
- (a) Coughlan & Yuille 1999, *Manhattan World*, ICCV —
  [PDF](https://www.cs.jhu.edu/~ayuille/courses/Stat238-Winter12/ManhattanWorld.pdf)
- (a) Newcombe et al. 2011, *KinectFusion: real-time dense surface mapping and tracking*, ISMAR —
  [PDF](https://www.doc.ic.ac.uk/%7Eajd/Publications/newcombe_etal_ismar2011.pdf)
- (a) Yang, Li, Campbell & Jia 2016, *Go-ICP*, IEEE TPAMI 38(11) — [arXiv:1605.03344](https://arxiv.org/abs/1605.03344)
- (a) Yang, Shi & Carlone 2020, *TEASER: fast and certifiable point cloud registration*, IEEE T-RO —
  [arXiv:2001.07715](https://arxiv.org/abs/2001.07715)
- (a) Rusu, Blodow & Beetz 2009, *Fast Point Feature Histograms (FPFH) for 3D registration*, ICRA —
  [PDF](https://www.cvl.iis.u-tokyo.ac.jp/class2016/2016w/papers/6.3DdataProcessing/Rusu_FPFH_ICRA2009.pdf)
