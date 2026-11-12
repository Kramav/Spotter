# Adversarial review: architecture, engineering and roadmap (2026-11-11)

This reviews the whole repo against where it is meant to go (CLAUDE.md's Roadmap, `docs/ideas.md`,
the decision log). It asks four questions: is the architecture sound, does it support the roadmap,
what should be adopted instead of built, and what to fix first. The 2026-11-05 review
([review-2026-11-05.md](review-2026-11-05.md)) covered bugs in the legacy examples; this one is about
structure, process and direction.

**How it was made:** three independent reviewers, each covering two roles:
- Architect + Adversarial;
- Maintainability + Test/Reliability/Security;
- Roadmap + Open-source strategist.

A fourth reviewer then challenged all three. Items marked ✔ were checked again by reading the code.

**How to read the references:** paths without a folder are relative to `exercises/cpp/` (for
example `room_folder.cpp` = `spotter/src/room_folder.cpp`), except `README.md:235`, the root README,
and `CLAUDE.md`. Line numbers are from commit be87325.

**Labels:** **[Doc]** = written down in the repo; **[Inf]** = strong inference; **[Rec]** = recommendation.

**Nothing below has been decided.** The decisions in §7 are open until they are logged as D-entries.

**Since then, D44 (same day):** the Pi is the scanner only. It captures colour and depth and exports
them to a PC, and there is no detection on the Pi. This changes several items below:
- the NCNN rows (§5, §6, §9) no longer apply;
- initiative 11 measures capture and export, not YOLO;
- the dense merge's memory (finding 7) is no longer a Pi problem.

## 1. Ground truth

| | Fact |
|---|---|
| Size | ~50k lines of C++ written in **one month**: first commit 2026-10-11, 44 commits.<br>• `spotter` library: 19 modules, ~6.5k lines.<br>• `examples/`: ~20k lines in 96 files.<br>• `legacy/`: ~26k lines, frozen, but all 34 targets are still built.<br>• Python (235 lines) and C# (797 lines) are done/frozen. |
| Entry points | ~140 executables, each a `cv::waitKey`/`imshow` loop. `KinectSource` is constructed directly in ~77 files. |
| Library graph | One-directional with no cycles:<br>frame → depth → track → radar<br>geometry → register → room_scan → room → room_folder<br>`view` is the only highgui user; `source_freenect2` is the only libfreenect2 user.<br>This part is sound. |
| Data flow | KinectSource → RgbdFrame (unmirrored)<br>→ YOLO/AsyncDetector → BoxDepthSampler → Tracker → Radar<br>or → DepthAverager → MakeScan → PlaceScan (ORB + RANSAC + ICP against every earlier scan) → in-memory Session (D39) → K → room folder (format 2) → FuseScans (dense TSDF) → BuildRoom → .obj/.ply → room_twin / 13w / 13x |
| Tests | 10 synthetic `check_*` programs; they really fail `ctest`. 9 of 19 modules have no direct check. The 13e Session (543 lines) and all 95 example programs are outside ctest. Many claims rest on "temporary harnesses" that were later deleted. |
| Build | vcpkg baseline pinned; libfreenect2 overlay port pinned by SHA512. No CI. **Never built on Linux.** No warning flags (MSVC effectively at /W1). |
| Docs | Thorough decision log (D1–D43). The docs claim more than the code: see §3. |

## 2. The central finding (all four reviewers converge)

**The bottleneck is verification, not architecture.**
- Code is generated far faster than it is run on hardware or understood.
- About 31 status notes read "not run on the sensor" or "synthetic only". Library phases 2–6 have never touched the sensor.
- **Every recorded contact with real data found a bug that synthetic tests had passed:**
  - Demo 6 round 1 painted over a seen wall;
  - round 2 put the floor too low and the visitor in the wrong spot;
  - 13w put the floor too high (D43, still unconfirmed).
- The roadmap's physical gates were skipped while code-only work continued:
  - accuracy milestone "before stage C" (CLAUDE.md:231) ✔;
  - Pi bring-up, paused with "stage B goes ahead" only (CLAUDE.md:163) ✔.
- This also threatens the stated purpose: "understanding through small examples" is at odds with 1,400-line "examples" and 50k lines in a month.

Most of the architectural findings below are symptoms of this one cause. For example, the three floor rules survived because nothing forced them to agree against real data.

## 3. Findings, ranked

1. **The only real room scan sits where the docs tell you to delete it. ✔**
   - The scan is in `build/examples/room_scans/room_2026-11-07_2338`. 13e's `SaveFolder` is relative to the working directory (`examples/13e_Room_Scan/main.cpp:71`).
   - The README troubleshooting row says "Delete the `build` folder and rebuild" (`README.md:235`).
   - `WriteScan` does `remove_all` and then rewrites (`room_folder.cpp:177`). `SaveAll` rewrites every scan on every K (`scan_state.hpp:164-170`). A blue screen during the second K (D25) can therefore destroy scans that were already safely saved.
   - The start-up prompt still says "Every scan goes there the moment it is placed" (`room_folder.cpp:419`), which has been false since D39.
2. **D7 is unmet: no Linux build has ever happened. ✔**
   - D7 says each example "is checked to build on both before it counts as done" (`decisions.md:46`). Yet D14 says "Linux is untested" (`:127`), and the README's Linux section says "Not yet tested" (`exercises/cpp/README.md:82`).
   - The source looks portable: the only platform branch is `_WIN32` in `room_folder.cpp`. The risk is in the dependency build.
   - The Pi is unmeasured, but stages C and D were sized anyway.
3. **No replay and no ground truth.**
   - `KinectSource` is concrete. There is no recorder or playback, and `frame.seconds` is the PC clock.
   - Live logic is testable only in front of the Kinect. D43 can't be diagnosed after the fact.
   - The 21-scan real room has no reference: the structure study can't tell method error from 13e pose error.
4. **D34's copy drift has returned inside `examples/`. ✔**
   - The ~950-line fill exists three times: 13x, the new Demo 6, and legacy Demo 6.
   - 06c and Demo 3 share ~155 lines.
   - There are three different floor/"up" rules: `BuildRoom` (`room.cpp:253-301`), `ExtractStructure` (`structure.cpp:75-94`), and `FindTwinFloor` plus 13w's own `FindCeiling`. D43 came from mixing two of them.
   - `Pose` has no `Inverse`, so rotations are transposed by hand in 17 places.
5. **The algorithms nearest the roadmap have no lasting tests.**
   - 13w (D42: "the first half of 14"), 13x and Demo 6 were checked only by deleted harnesses.
   - 13w also starts step 14, which you had said not to start yet (CLAUDE.md:318).
6. **Reliability gaps.**
   - `AsyncDetector::Run` has no try/catch, so a DNN error kills the process and skips D30's rescue ✔ (`detector_yolo_opencvdnn.cpp:137-157`).
   - Room-folder parsing doesn't validate its inputs: unchecked `depth_size`, `stoi` throwing in 13z without a catch, `LoadTwinRoom` swallowing the reason a load failed.
   - The `check_room.cpp:124` assertion always passes ✔.
   - 13e's threads are safe only because every key path calls `WaitForReaders` first.
7. **The batch room pipeline can't grow into live mapping.**
   - `FuseScans` is a dense grid (`maxVoxels` 80M, ~2.2 GB) that visits every voxel for every scan. `PlaceScan` matches against all earlier scans.
   - This is fine for "capture on the Pi, merge on the PC" [Doc, ideas.md]. It is a blocker only for 16–17, which are unscheduled.
8. **Docs and process cost.**
   - Every new example has to update six living documents, and they have already drifted:
     - 13y is missing from EXAMPLES.md and visualizer.html;
     - CLAUDE.md:108 still says the mirror is "fix soon";
     - "one check per module" is false: 10 checks for 19 modules;
     - ideas.md:56-57 says ArUco is already in our OpenCV; it isn't (`objdetect` isn't built);
     - 49 header doc-comments no longer match their real signatures.
   - `visualizer.html` (2,919 lines) re-implements the algorithms in JavaScript.
9. **Hygiene.**
   - `zone_log.csv` is tracked although `.gitignore` lists it ✔.
   - `*.bmp` isn't gitignored, though BMP is the only image format this OpenCV can write.
   - Two unused 2.7 MB `synthetic_*.bmp` files are committed.
   - ~25 MB of Ultralytics weights are committed without LFS. They are likely AGPL-3.0, in an MIT repo [Inf: check before publishing].
   - `SPOTTER_REPO_ROOT` is an absolute path compiled into the binaries, and `.vscode` still points at the old `C:` path.
10. **Legacy is built by default.** 34 targets and 26k lines compile on every clean build. Their reproducibility value (D34) is kept equally well by a git tag.

## 4. Cross-agent disagreements: how they were resolved

| Disagreement | Ruling |
|---|---|
| A/B: add an `RgbdSource` interface. Against: the single-file convention and no premature abstraction | **Build replay without an interface.** Add a playback mode inside `KinectSource` plus a recorder, so the ~77 call sites don't change. Room-folder programs already replay. Limits: bandwidth (record depth + registered colour, short clips) and privacy (colour only of empty rooms, per D29). |
| A: the `structure` module is premature. Against: it's a study | **Open, low priority.** Freeze it until you've reviewed the findings report. |
| A/B: move the fill into the library. Against: D40, "copy until it changes" | **Split.** The floor/"up" rule is single-sourced now, because it already caused D43. The fill follows D40's own trigger and is extracted the next time it's edited. |
| B: CI, sanitizers, a doc manifest. Against: overhead for one person | **Split.** A Linux + Windows CI build is your own D7 rule, not enterprise overhead. Warnings: yes. Sanitizers: optional, later. Doc lint: **rejected**; cut the docs instead. |
| C: decide RTAB-Map vs our own code now. Against: the learning goal | Log now that the room pipeline is **teaching code until evidence exists**. Run RTAB-Map and Open3D outside the build as reference answers. Decide product vs teaching after the Pi numbers. |
| A: dense TSDF rules out the Pi; one library linking GTK prevents a headless Pi | **Downgraded.** The first is a blocker only for 16–17. For the second, Pi OS ships GTK, so the cost is size, not feasibility; measure it at bring-up. |
| **Preserved disagreement:** "the architecture is fine" vs "structural rework is needed for 16–17" | Both are right on their own horizon. The review sides with "measure before redesigning", because 16–17 are unscheduled and the Pi numbers may change the answer, e.g. towards RTAB-Map. |

**Overengineered:** visualizer.html; six docs per example; 1,400-line examples; three copies of the fill; the 13c→13z branching ahead of the gates.
**Underengineered:** data safety; replay; ground truth; the Linux build; exception boundaries; keeping harnesses.

## 5. Gap analysis

| Area | Current | Intended (label) | Gap | Consequence |
|---|---|---|---|---|
| Architecture | One static library, 96 example files, legacy archive; a batch room pipeline | Small sketches on a shared library; fixed camera → multi-station → moving camera [Doc] | The algorithms nearest the roadmap live in examples; no incremental path | Fine for A–C; 16–17 need rework, but only after Pi numbers |
| Domain model | `Scan` holds everything; three floor/up rules; no depth→colour extrinsics; no `Pose::Inverse` | One meaning for floor/up and for camera geometry [Inf] | The rules disagree | Geometry bugs whenever modules are combined (D43) |
| Core abstractions | `KinectSource` used directly in ~77 files; no replay | Testable without the sensor [Rec] | No playback mode | Bugs are found only live, late |
| Data layer | Format 2, flushed; stored under `build/`; delete-then-rewrite; full rewrite on every K; parsing unvalidated | Scans survive a crash or a lost Kinect (D30/D39) [Doc] | Location, atomic writes, validation, a misleading prompt | **Irreversible loss of the only real scan** |
| Integrations | libfreenect2 port; OpenCV DNN on a thread with no exception guard | Pi CPU; NCNN if needed [Doc] | Never built on Linux or a Pi | The Pi plan rests on assumptions |
| Testing | 10 checks; deleted harnesses; one always-true assertion; loose tolerances | "Checked … before it counts as done" (D7) [Doc] | No replay, ground truth, Linux, or Session check | Regressions slip through unseen |
| Developer experience | Windows only; legacy always built; absolute repo root | Two-command build on both platforms (D14) [Doc] | The Linux path is untested | Hand-off fails |
| Observability | printf only | Enough to diagnose real runs [Inf] | No run log or replay | Field bugs can't be reproduced |
| Security/privacy | Privacy rule in place; a tracked csv; `*.bmp` not ignored; AGPL-licensed weights. Auth, APIs, secrets: **N/A** (no network surface) | No images of people; room scans only by D29 [Doc] | Ignore rules; a licence note | A committed frame, or a licence mismatch if published |
| Deployment (Pi) | Paused; never built | Pi 5, Linux, CPU only (D11) [Doc] | Everything | Stages C–D are sized without numbers |

## 6. Build vs adopt

| Capability | Call | Why |
|---|---|---|
| Point-to-plane ICP with a weak-direction solve | **Keep** | ~100 lines, checked, handles bare-wall degeneracy, which plain Open3D ICP doesn't |
| k-d tree / Kalman / inpaint / DNN | **Already adopted** (OpenCV FLANN, `cv::KalmanFilter`, `cv::inpaint`, OpenCV DNN) | Correct the docs that imply these are ours |
| RANSAC planes, OBJ/PLY writers, z-buffered viewer, tracker logic (05b ≈ ByteTrack's idea), ORB glue | **Keep** | Small, checked, teach the ideas; a dependency would cost more |
| Dense TSDF + surface nets | **Keep** as teaching code. **Defer** the product choice: a sparse/hashed TSDF rung modelled on Open3D, or replacement by RTAB-Map | Can't scale to live mapping; not needed until 16–17 |
| Room folder format | **Keep**, plus an *export* to TUM/Open3D RGB-D layout later (needs OpenCV `png`) | Opens every outside tool for comparison |
| Whole stages C–E | **Defer**, but adopt **RTAB-Map** (BSD-3, libfreenect2 driver, runs on a Pi, loop closure, multi-session) **and Open3D (MIT, Python) now as reference tools outside the build** | Gives the missing ground truth for 13e at no integration cost |
| ArUco (OpenCV `objdetect`) | **Adopt** (small vcpkg feature) | An independent pose ground truth; also roadmap item 12 |
| YOLO on the Pi | **Adopt NCNN** (BSD-3) after the Pi timings | Already planned as a sibling file (D34) |
| Step 14 seam blending | **Defer**: mvs-texturing (BSD-3, inactive since ~2020) | Decide before building the seam step |
| CI | **Adopt** GitHub Actions + vcpkg binary cache | Enforces D7 |
| ORB-SLAM3 (GPL-3), KinectFusion/ElasticFusion (GPU, research licences) | **Reject** | Licence, and D11 |

Library versions and maintenance status here come from general knowledge, not checked online.

## 7. Critical decisions (before more feature work)

1. **Where scans live and how saving is made safe.**
   - Options: as now / outside `build/` with temp-folder-and-rename and only changed scans rewritten / back to per-scan autosave (D30).
   - **[Rec]** The middle one.
   - Cost if wrong: permanent data loss. Reversible.
   - **Yours to judge:** whether D39's accepted risk (a blue screen loses the session) still stands.
2. **A time-boxed feature freeze (2–3 weeks)** until the backlog has run on real data.
   - **[Rec]** Yes. Cost if wrong: a few weeks. Reversible.
3. **How replay works.**
   - Options: a playback mode in `KinectSource` / a virtual interface / nothing.
   - **[Rec]** The playback mode.
   - Investigate first: bytes per frame and which streams are needed.
4. **Is the room pipeline product or teaching code?**
   - **[Rec]** Log "teaching until evidence" now; decide after the Pi numbers and the RTAB-Map comparison.
   - Cost if wrong: months spent on 14/16 code that gets discarded.
5. **When does code move from an example into the library?**
   - **[Rec]** Geometry that correctness depends on (floor/up, frames, poses) is always single-sourced. Anything else moves on its second diverging copy or its first bug fix. This refines D35/D40.
6. **Legacy.** **[Rec]** A git tag, plus `SPOTTER_BUILD_LEGACY` defaulting to OFF.
7. **Living-doc scope.** **[Rec]** Keep decisions.md and the CLAUDE.md Status. Freeze visualizer.html. Make EXAMPLES.md entries optional.

## 8. Remediation roadmap (F = foundational, O = optional)

| # | Phase | Initiative | Size | Depends on | Acceptance criteria | Blocks |
|---|---|---|---|---|---|---|
| 1 | Stabilize | **Data safety (F):** copy the existing scan out of `build/` (your action, today); default save folder outside `build/`; fix the prompt text; `WriteScan` writes to temp then renames; K writes only changed scans | S | — | A check that kills the save midway shows the previous save intact; a clean rebuild leaves scans untouched | All real-data work |
| 2 | Stabilize | **Exception boundaries (F):** `AsyncDetector` catches and reports errors; room-folder loads validate and give reasons; fix the always-true check | S | — | A corrupt-folder check and a throwing-detector check pass | 3 |
| 3 | Stabilize | **Real-data sprint (F):** run 13w (capture the D43 lines), 13x, 13z and Demos 4/6 on the real room; the accuracy milestone with known-size objects (TV, A4, door) | M | 1 | Every library phase has a logged real run or an explicit waiver | 8, 10 |
| 4 | Foundations | **Linux build + CI (F):** legacy option OFF; Ubuntu + Windows GitHub Actions building the library, checks and examples and running ctest; vcpkg cache; repo root found at run time | M–L | — | Both platforms green (D7 honoured) | 11 |
| 5 | Foundations | **Record/replay (F):** a recorder plus a playback mode in `KinectSource`; device timestamps; privacy-safe streams; replay checks for the 13e Session and the rescue rule | M–L | 2 | A 10 s clip replays the same tracker and session output in ctest | 6 |
| 6 | Foundations | **Keep the harnesses (F):** `check_*` for the 13e session, the Demo 6 fill, 13w's ceiling and texture, the rescue rule | M | 5 (partly) | Every "temporary harness" claim points to a check that exists | 8 |
| 7 | Foundations | **Warnings (O):** /W4 and -Wall -Wextra on the library and checks | S | 4 | Builds clean, or warnings suppressed with a reason | — |
| 8 | Align | **One floor/up rule (F)** + `Pose::Inverse`; extract the fill to `room_fill` the next time it's touched | M | 3, 6 | A grep finds no second rule; D43 and the real room agree | Step 14 |
| 9 | Align | **Doc diet + hygiene (O):** freeze the visualizer; fix ideas.md:56, CLAUDE.md:108 and the "one check per module" claim; gitignore `*.bmp`; untrack `zone_log.csv`; licence note for the weights; Linux steps in the hand-off README | S | — | No known doc contradicts the code | — |
| 10 | Align | **Reference reconstruction (F):** RTAB-Map and Open3D on the same room (ArUco optional); log the product-vs-teaching D-entry | M | 3 | Pose and mesh differences logged; decision written down | 12 |
| 11 | Optimize | **Resume Pi bring-up (F for C–E):** depth fps, USB stability, YOLO at 640 and 320, NCNN | L | 4 | Numbers logged; stage B–D weights decided | 16–17 |
| 12 | Optimize | **Sparse TSDF + incremental placement as a ladder rung (O)** | XL | 10 | Only if decision 4 picks our own code as the product | — |

**30 days:** 1, 2, 3, 9; the feature freeze; a first Linux build in WSL.
**60 days:** 4, 5, 6, 10.
**90 days:** 11, 8; decision 4; then step 14 or a TSDF rung.

## 9. Executive verdict

- **Is the architecture fundamentally sound?** Yes, for what is scheduled: teaching fixed-camera detection plus offline multi-station merging. The library graph, decision log, checks and pinned build are better than typical. For live mapping on a Pi it is unproven. Measure that; don't redesign in advance.
- **Does the foundation support the roadmap?** Stages A–C and 14–15: yes. 16–17: not as it stands, but that decision depends on Pi numbers you don't have yet.
- **Five biggest problems:**
  1. Verification lags code generation: gates skipped, harnesses deleted.
  2. The only real data is at risk: stored under `build/`, saved by delete-then-write, and the prompt misstates how saving works.
  3. No replay and no ground truth.
  4. D7 unmet: no Linux build, Pi unmeasured.
  5. Duplicated geometry rules and fill code, plus documentation drift.
- **Unnecessarily built:** the JavaScript re-implementations in visualizer.html; demo copies of large algorithms; more 13-family variants built ahead of the gates; six docs per example.
- **Libraries to take seriously:** RTAB-Map and Open3D (as references now, a possible foundation later); OpenCV ArUco; NCNN (after the Pi numbers); GitHub Actions.
- **Evolve or replace?** Evolve. Nothing justifies a rewrite.
- **Stop immediately:** new examples or demos until the backlog has run on real data; deleting harnesses; per-example visualizer entries; saving into `build/`.
- **Biggest risks:** losing the real scan; the Pi proving too slow after C–D were built; code volume outrunning your understanding, which defeats the learning goal.
- **First change for a 3-year owner:** in code, scans out of `build/` with an atomic save (hours of work, removes an irreversible risk). In process: **nothing counts as done until it has run on real or recorded data and its check stays in ctest.**

### Final recommendation: **Continue as-is with targeted improvements.**
Keep the design and change the process. The architecture blocks nothing that is scheduled. What fails is verification, and a refactor would not fix that. Revisit "adopt selected external foundations" only if the Pi numbers and the RTAB-Map comparison argue for it.

