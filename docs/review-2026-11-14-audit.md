# Audit of the 2026-11-11 architecture review and its remediation (2026-11-14)

An adversarial audit of [review-2026-11-11-architecture.md](review-2026-11-11-architecture.md) and of the work
done since (D44-D52). Six independent reviewers, each read-only and kept out of `room_scans/`:
implementation verification, architecture, code quality, testing and reliability, privacy and operational
risk, and an adversarial pass on the review itself. The lead checked their high-severity claims in the code
and reconciled where they disagreed. The fixes that needed no decision from the user were then made: D53, D54.

Everything here is a recommendation unless a D-entry records it as decided. The review's §7 decisions are
still the user's.

## Verdict

- **The review's diagnosis holds:** what's missing is verification, not architecture. The library's
  dependencies still run one way with no loops; nothing calls for a rewrite.
- **The remediation mostly holds up.** Initiatives 1, 2, 9, 7 and the `Pose::Inverse` half of 8 meet their
  acceptance criteria in the code. Initiative 6 is partly done.
- **The process repeated the pattern the review criticised.** Since the review: 13 commits of new code and
  synthetic checks, no planned run on the sensor. The real-data sprint (initiative 3) hasn't started. The one
  real run (the Ambulance scare, D48) happened by accident and found a bug in initiative 1, already marked done.
- **D44 made parts of the review stale.** It was decided the same day: the Pi only captures. The NCNN row,
  initiative 11's YOLO timings, initiative 12 and decision 4's "wait for Pi numbers" no longer apply. What D44
  does need isn't in the review: a capture format and a capture-only program.

## Verification matrix

| # | Requirement | Status at the audit | Since (D53, D54) |
|---|---|---|---|
| 1 | Safe saves, scans out of `build/` | Done for 13e only. 08, 08b, 09b, 09c, tour 00g and four ladder steps saved into the folder they ran from. The `build/` refusal used a path compiled in. No real fault injection. | Save As everywhere; any folder with `CMakeCache.txt` above it refused; the repo rescue folder only if the repo is there. Fault injection still to do. |
| 2 | Errors handled | Done. `features.yml` unchecked; save failures gave no reason. | Every write checked after closing; the failing file and reason printed. |
| 3 | Real-data sprint | Not started; gates 8 and 10. | Unchanged (the user's runs). |
| 4 | Linux build + CI | Not started; `_WIN32` branches 2 → 22, D48's zenity path and the guard's Linux handler never compiled. | Excluded by the user for now. |
| 5 | Record/replay | Not started. Revise (below). | Waits for the privacy rules. |
| 6 | Harnesses → checks | Partial. Solid 13e session/rescue check; 13w's colour-model texture path never runs; fill and texture limits fitted to measured output; far features, landmarks, live check, `OnScanReady`, E uncovered. | Placement limit tightened; one shared test room. The gaps remain. |
| 7 | Warnings | MSVC only; the examples (~17k lines) had none. | Examples at /W4 too: 26 found and fixed, 0 in the whole build. |
| 8 | One floor/up rule + `Pose::Inverse` | Half: three rules remain. | The rules compared in `check_room_shell` (they agree on the test room). Choosing one waits for D43. |
| 9 | Docs and hygiene | Done, one leftover ("one check per module"). | Fixed. |
| 10-12 | Reference, Pi bring-up, sparse TSDF | Not started. | Revisions below. |

## Findings, ranked

1. **Photos in the git history.** `alert_*.jpg` (four files, commit `057eb9b`) were removed from the files by
   `6e5bc02` but stay in history, and `main` is pushed. Purging means rewriting history and the
   `legacy-archive` tag and re-cloning on the other PC: the user's decision. The cost grows with each clone.
2. **A test wrote into the real `room_scans\`** (`check_scan_session`'s rescue, every `ctest`). Fixed (D53):
   `SPOTTER_RESCUE_FOLDER`.
3. **13e could lose scans silently:** it caught only `std::exception` and returned 0 when nothing was saved.
   Fixed (D53): any error reaches the rescue; exit code 2 when the scans aren't saved. A hard crash still loses
   the session: that follows from D39 (no autosave), the user's rule, and is now the largest loss path.
4. **"Colour only of empty rooms" is convention, not code** (a comment in 13e). Record/replay and the Pi export
   both move camera data: decide first which streams, depth-only by default or not, encryption, retention.
   `docs/standards.md` suggests these; no D-entry adopts them.
5. **D50 made a third home for code.** `fill_holes.hpp` (1,127 lines) is used by 13x, Demo 6 and a check;
   `scan_state.hpp` mixes session logic with dialogs; both are in the global namespace, and Demo 6 can't have
   its own fill settings. The library has no capture boundary: the console guard pulls in highgui, `vcpkg.json`
   always builds dnn and GTK, so a Pi capture program would compile both for nothing.
6. **The checks shared one scene** (five copies of one box room: closed, square, no holes, ORB-friendly).
   Now one copy (D54); a second scene is still needed. The real room showed why: the structure study's Relocate
   was confidently wrong on 6 of 21 real scans.
7. **Absolute build paths carried safety logic** (the `build/` refusal, the first rescue folder). Fixed in the
   library (D53); ~75 example uses (model path, default rooms folder) remain. On Linux the last rescue
   fallback is `/tmp`, wiped on reboot (now said in `RescueFolders`' comment).
8. **Leftover duplication:** 06c and Demo 3 share 242 lines, Demo 4 and Demo 6 196. Left: the wall filter is
   06c's own lesson; where shared code goes is decision 5.

Where the lead disagreed with a reviewer: a crash handler that saves the scans was rejected (a crashed process
can't save safely; catching any exception type is enough). The count of "not run on the sensor" notes
(30 → 35) is a rough text search: the trend is right, the number isn't precise. Splitting the dialogs out of
`room_folder` is tidy but low value; finding paths at run time mattered more.

## How the review's recommendations should change

| Item | Call | Why |
|---|---|---|
| Decision 1 (scan location, safe saves) | Expand | Every saver, and paths found at run time (done for the library, D53). |
| Decision 2 (feature freeze) | Keep, log it, widen it | An examples-only freeze let new library code through (D48, D49). Gate: no new modules until three real runs are logged. |
| Decision 3 + initiative 5 (replay) | Revise | One frame-file format as D44's capture format, the recording, and a `ReplaySource` file beside `source_freenect2`, instead of a playback mode inside `KinectSource`. Needs the privacy rules first. |
| Decision 4 (product vs teaching) | Revise | Not tied to Pi numbers any more (D44). Log "teaching" now. |
| Decision 5 (example → library) | Revise | Add: a header included from outside its own folder moves to the library (`fill_holes.hpp` → `room_fill`). |
| Initiative 4 (Linux + CI) | Revise | One WSL build first, CI later. |
| Initiative 8 (floor/up rule) | Split | Compare now (done, D54); choose after the real-data sprint. |
| Initiative 10 (reference) | Narrow | One RTAB-Map run of the same room; drop Open3D and ArUco until they show a gap. |
| Initiative 11 (Pi bring-up) | Rewrite | In D44's terms, with a capture-only program. |
| Initiative 12 (sparse TSDF) | Remove | Gated on Pi numbers that no longer apply. |
| Missing from the review | Add | Purge the photos; privacy rules for capture; paths at run time; a run log beside each room; a capture-only build target. |

## Decisions waiting on the user

1. Purge `alert_*.jpg` from history (rewrites history and the `legacy-archive` tag; force-push; re-clone).
2. The privacy rules for capture and recording.
3. Log the freeze (decision 2) and decisions 3, 4, 5 as revised above, or not.
4. Then: the real-data sprint (13e with K twice, then 13z; 13w's `Floor:` lines; one known-size measurement),
   one WSL build, the capture format, `ReplaySource`, and choosing the floor rule.
