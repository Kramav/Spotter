# Spotter

Learning object detection, recognition and tracking. The current focus is Kinect v2 RGB-D fusion
(detect in colour, measure distance with depth), following `kinect_rgbd_fusion_guide.md`.

**Deployment target: Raspberry Pi (Linux, CPU only, no GPU compute).** Develop on Windows for now
(D7). Anything that needs a GPU is out (D11).

**Long-term goal: a 3D map of the space.** Staged plan: fixed camera → several fixed stations
merged offline → moving camera (see [docs/ideas.md](docs/ideas.md)). The camera stays fixed for now.

## Living documents — keep these current

- **This file**: the Status and Roadmap sections. Update them at the end of any work session that
  changes either one.
- **[docs/decisions.md](docs/decisions.md)**: the design decision log. Append an entry whenever a
  choice is made that someone would later ask "why?" about. Never rewrite old entries; mark them
  superseded instead.
- **[docs/ideas.md](docs/ideas.md)**: techniques researched but not scheduled, with when each would
  pay off and how to build it. Add to it when a "maybe later" topic comes up. Move an item to the
  Roadmap once it's scheduled.

## Layout

| Path | What |
|---|---|
| `exercises/python/` | Webcam/OpenCV/YOLO exercises (no Kinect). Complete. |
| `exercises/csharp/` | Kinect v2 examples 01–03 on the Microsoft Kinect SDK 2.0. Windows only, frozen (see D5). |
| `exercises/cpp/` | Kinect v2 examples on libfreenect2 + OpenCV, in C++ (D8). The active track. Builds with two commands from its README (D14). `ports/` = our vcpkg recipe for libfreenect2. |
| `docs/` | Living documents: `decisions.md` and `ideas.md`. |
| `yolo26n.onnx` / `.pt` | Detector model. The ONNX file was exported with `nms=False` and outputs 1x300x6. |
| `kinect_rgbd_fusion_guide.md` | Background reading: the method and the accuracy tests (§7). |

## Conventions

- **The goal is understanding the system through small examples, not one-shot builds.** Each
  example teaches one idea and builds on the previous one. Long-term plans (Roadmap, ideas.md)
  get broken into a ladder of such examples; never jump straight to the full system.
- Examples are written like Arduino sketches: one file each, settings at the top, then the
  sections Settings / Setup / Main operation / Output / Experiment ideas. Priorities, in order:
  readability, simplicity, mutability, learnability, capability.
- Privacy: never save or commit camera frames or images of people. `*.jpg` is gitignored for this
  reason.
- Lean towards our own software and open source over vendor SDKs (D5). New code must build on
  Linux as well as Windows (D7).

## Status (2026-10-25)

- Python exercises: done.
- C# + Kinect SDK examples 01–03: done and running on hardware (commit cea5c21). They have not
  been checked against a tape measure.
- Moving to libfreenect2 in C++ (D5, D8): `exercises/cpp/`, in progress.

## Roadmap

- [x] Decide the language for the libfreenect2 track → C++ (D8).
- [x] Windows setup: vcpkg dependencies, libfreenect2 built and installed (D9). Done 2026-10-25.
- [x] User swapped the Kinect USB driver to libusbK (2026-10-25).
- [x] Port 01 Depth Peek. Runs on the sensor at 30 fps after the OpenMP patch (D11). The user
      checked the window by eye: identical to the C# version.
- [x] Port 03 Depth Blobs. Runs on the sensor; the intrinsics match the SDK's. The user still
      needs to walk in and check the blobs.
- [x] Port 02 Detect Distance with libfreenect2 `Registration`. Runs on the sensor: it detected a
      bed at 0.95 m, and the pinhole maths agrees with `getPointXYZ`. The user still needs to check
      the window.
- [x] Reproducible build: fresh clone → `cmake --preset default` + `cmake --build --preset default` (D14).
- [ ] **Example 04: Track and smooth (key).** Detect (as in 02), track each object across frames
      (match detections by box overlap, as the Python exercises 06/07 did), then smooth each object's
      distance over time with a running average or Kalman filter. A Kalman filter also gives the
      object's speed towards or away from the camera. This works with a moving camera, because the
      smoothing follows the object rather than a pixel. Do it after 02. See D12.
- [ ] Build and run on Linux, then on the Raspberry Pi: measure fps with the filters on and off.
      Check that the Pi's USB 3 port keeps up with the Kinect.
- [ ] Tape-measure accuracy, guide §7 tests 2–4. If it comes out poor, revisit D6 (own calibration).
- [ ] Decide whether to retire `exercises/csharp/` once the new track reaches parity.

Toolchain on the dev machine: VS Build Tools 2026 with the C++ workload, and .NET 10. Building C++
needs `vcvars64.bat`, which puts `cmake`, `ninja` and `vcpkg` on PATH. In Git Bash, run the built
.exe through cmd or PowerShell, because Git Bash's loader fails on `api-ms-win-crt-*`.
