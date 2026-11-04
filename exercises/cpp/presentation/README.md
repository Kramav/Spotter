# Presentation demos

Five fullscreen demos for a small exhibition. Each one is an example from `exercises/cpp/`,
remixed for a crowd: fullscreen, big text, no save keys, and a "live only: nothing is recorded" line.
Visitors see themselves, which is what makes people stop at a stand.

| Demo | What visitors see | Built from | Needs |
|---|---|---|---|
| **1 You in 3D** | Themselves as a live 3D point cloud, slowly swinging round: from the side they're a hollow shell | 08 | Kinect only |
| **2 Invisibility Cloak** | Camera and "cloaked" view side by side: people vanish, anything they hold floats on its own | 07d | Room empty at the start, good light |
| **3 Booth Radar** | Camera with boxes and distances, plus a top-down radar with a dot and trail for each visitor, and the stand's walls | 06c | Good light |
| **4 Room Twin** | The room scanned with 13, as a turning doll's house, with visitors live inside it | 13 + 08 + 03 | A room scan from 13 first |
| **5 How Tall** | Their height in metres over their head, plus "tallest today" (a number, no picture) | 03 | Head and feet in view |

All five build with the other examples. They're in the same `CMakeLists.txt`, so the README's
build command builds them too. They run from `exercises\cpp\build\`, like the examples:

```powershell
cd E:\Company\Github\Spotter\exercises\cpp\build
.\Demo1_You_In_3D.exe
```

Only one program can use the Kinect at a time: quit one demo (Q or Esc) before starting the next.

## The stand

- **Kinect at chest height** (about 1.2–1.4 m) on a tripod or a table, level or tilted slightly
  down. On the floor it sees legs, Demo 5 can't see heads, and Demo 3's walls tilt.
- **Mark a spot on the floor** 2–3 m from the Kinect with tape: "stand here". Closer than 0.5 m
  reads nothing; past about 4.5 m it gets thin.
- **No sunlight on the Kinect or the visitors.** Sunlight drowns the Kinect's infrared light, and
  depth fails. Hall lighting is fine. Demos 2 and 3 use YOLO on the colour camera, which needs
  light. Demos 1 (press C), 4 and 5 work in the dark.
- **Run on the PC**, not the Pi: YOLO takes over 1 s per frame on a Pi.
- **A sign:** "Live only: nothing is recorded." That's true: no demo writes a file (privacy rule:
  never save images of people). Demo 4 reads a room file that you made beforehand with nobody in it.
- **Keys in every demo:** F toggles fullscreen, Q or Esc quits. The rest are listed at the top of
  each `main.cpp`.

## Suggested set-up: two screens

- **Screen 1:** Demo 1, the eye-catcher: people walking past see themselves turn round.
- **Screen 2:** Demo 4 if your room scan worked, otherwise Demo 3. Switch to Demo 5 or 2 when a
  group stops and wants to play.

With one Kinect, run one demo at a time and switch.

## Each demo, before the doors open

**1 You in 3D.** Start it, stand on the mark, and check you're in the middle as the view swings.
If not, change `PivotMeters` to the mark's distance. If the room looks tilted, press `[` or `]`
until the floor and walls look level, then put the printed `LevelDegrees` in the file (14 suits
the Kinect sitting on the floor and aimed up; use 0 when it stands level on a tripod). To show
only people with nothing behind them, set `MaxDepthMeters` to 3.

**2 Invisibility Cloak.** Start it with **nobody in view** and wait about 10 s while it learns the
empty stand. Then walk through: you should vanish on the right. If the stand has changed (a bag
put down, a chair moved), clear the view and press SPACE to learn it again. Its limits: black
clothes and hair give no depth reading, so they stay visible. Fast walkers leave a short ghost.

**3 Booth Radar.** Start it; walk a circle on the floor, and your trail should draw a circle. On
a busy stand, set `RadarClasses = { "person" }` so chairs and bottles don't clutter the radar. C
clears the trails between visitors.

**4 Room Twin.** Without a room file it still runs. It then takes the Kinect's own empty view
as the room (`OneViewRoom.h`: clear the view when the console says so). That's one view only, but
it's good for rehearsing. Set `OneView::LevelDegrees` in that file to the Kinect's tilt (14 on
the floor, 0 level). The real thing needs the most preparation:
1. The day before, or in the morning, scan the stand with `13c_Kinect_Room_Scan_Assist` (13d for long halls; 13 and 13b work too), following
   "Recreate a room" in the [main README](../../../README.md#recreate-a-room-example-13), with
   nobody in view. Press E. It writes `room_scan_<time>.obj` into the folder you ran it from.
   The file has the line `# first camera turn:` at the end; Demo 4 needs it, so use a scan made
   after 2026-11-03.
2. Put the Kinect back where scan 1 was taken, aimed the same way. In the synthetic test, it found
   itself from up to 15° and 58 cm off. Mark the tripod's feet with tape before you scan.
3. Run `Demo4_Room_Twin.exe` from the same folder (it takes the newest `room_scan_*.obj`, or give
   the path), with nobody in view. After 3 s the console prints `Located: NN% of the view within
   2 cm of the room`. 80% or more is good. Under 60% it warns: move the Kinect closer to where
   scan 1 was, clear the view, and press L.
4. Walk around: an orange shell (you) should walk around inside the doll's house. H switches to
   true colour.

**5 How Tall.** Start it with nobody in view. It learns the empty stand in 3 s, and B learns it
again later. Check it on someone who knows their height: expect a few cm short (hair gives no
depth reading). "step back" means head or feet are cut off. T resets "tallest today".

## What's been checked, and what hasn't

| Demo | Checked | Not yet |
|---|---|---|
| 1 | Builds; it's 08 (runs on the sensor, 14 loops/s) plus the swing | Run on the sensor |
| 2 | Builds; it's 07d (28 fps) plus the side-by-side view | 07d has never removed a real person (nobody was in view): test with a person first |
| 3 | Builds; it's 06c (runs on the sensor) in one window | Run on the sensor |
| 4 | Fallback without a room file (`OneViewRoom.h`): a synthetic wall seen 14° tilted comes out upright to ±1.5 mm. Synthetic test (bare room, the worst case): located within 0.2 mm when put back up to 15° and 58 cm off, in 0.3 s; at 20° and 30° it's lost and warns (38–40% matched). 13's own test: the saved turn lands 94% of scan 1's points within 3 cm of the merged mesh | 13 has never run in a lit room, so there's no real room file yet |
| 5 | Builds; the height now uses each end's own depth | Run on the sensor; check against a known height |

Rehearse on the sensor before the day. For each demo, look at the points under "Each demo" and
note anything that looks wrong.
