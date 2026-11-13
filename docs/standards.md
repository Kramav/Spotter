# Standards for scanning spaces with a Kinect

Reference notes (2026-11-13). Spotter uses a Kinect v2 (time-of-flight depth plus a colour camera) on a
Raspberry Pi to capture rooms, then turns them into `.ply` point clouds and `.obj` room models on a PC.
No standard certifies a consumer Kinect. These standards say how to test and report its accuracy, how
to exchange the data, and what safety and privacy duties apply.

Each entry has a short report and what the scanner (the Pi, the Kinect and the PC software together)
should do to meet it. Groups are roughly in priority order; a consolidated checklist is at the end.

**Written from memory, no pages fetched.** Check each standard's current edition and clause numbers on
iso.org, iec.ch or astm.org before citing it outside the team.

## Contents

1. [Measurement accuracy and testing](#1-measurement-accuracy-and-testing): ISO 17123-9, ISO 10360-13,
   ISO 5725, ISO/IEC Guide 98-3 (GUM), ISO/IEC 17025
2. [Building and space data](#2-building-and-space-data): ISO 9836, ISO 16739 (IFC), ISO 19650
3. [Point cloud formats](#3-point-cloud-formats): ISO/IEC 23090-5 and 23090-9
4. [Safety](#4-safety): IEC 60825-1, IEC 62471, IEC 62368-1
5. [Privacy and security](#5-privacy-and-security): ISO/IEC 27001, ISO/IEC 27701, ISO/IEC 29100
6. [Non-ISO but widely used](#6-non-iso-but-widely-used): ASTM E2807 (E57), ASTM E2938, ASTM E3125,
   VDI/VDE 2634 Part 2
7. [Checklist](#7-checklist)

---

## 1. Measurement accuracy and testing

These matter most. They turn "the scan looks right" into a number with a stated uncertainty, and that
number is what anyone using the scans asks for first. They also feed the Roadmap's open accuracy
milestone (guide §7 tests 2–4).

### ISO 17123-9: field tests for terrestrial laser scanners

Part 9 of ISO 17123 (Optics and optical instruments: field procedures for testing geodetic and
surveying instruments) sets out how a user checks a terrestrial laser scanner in the field, with
targets set up in an ordinary space rather than a lab. It gives a simplified test (is the instrument
fine to use?) and a full test (estimate its precision). Both measure distances between targets from
several scanner positions and compare them with reference values. It is the closest ISO match for a
room scanner, though the Kinect isn't a laser scanner.

The device should:

- Support a repeatable test: scan a fixed set of targets (spheres or flat checkerboard targets) from at
  least two stations, as the standard lays out.
- Report target-to-target distances and their spread across repeated scans, so the simplified test
  runs without extra tools.
- Log the conditions with each test: temperature, time since power-on (the time-of-flight Kinect
  drifts while it warms up), range and lighting.

### ISO 10360-13: tests for optical 3D coordinate measuring systems

Part 13 of ISO 10360 (Geometrical product specifications: acceptance and reverification tests) covers
optical 3D CMS, including structured-light and area-scanning sensors. It defines probing errors (form
and size, measured on a sphere) and length-measurement errors (ball bars or step gauges). It is a lab
standard for object-scale metrology, much stricter than room scanning needs, but its error definitions
are the accepted vocabulary.

The device should:

- Use its error terms in internal tests: sphere form error, sphere size error, length error across the
  working volume.
- Fit a sphere to each sphere target in a scan and report its radius and residual. This is a cheap way
  to see depth noise and bias at range.
- Make no ISO 10360 claim for room-scale results: its working volumes are far smaller than a room.

### ISO 5725: accuracy (trueness and precision)

ISO 5725 (parts 1–6) defines trueness (how close the mean is to the true value), precision (spread of
repeated results), and how to estimate both from repeat experiments. It separates repeatability (same
setup, short time) from reproducibility (different setups, operators or days).

The device should:

- Report accuracy as two numbers, not one: bias (trueness) and standard deviation (precision), each
  with the range it applies to.
- Measure repeatability (same spot, 30 scans in a row) separately from reproducibility (different days,
  Kinect re-plugged, Pi rebooted).
- Use the standard's terms in docs and output: "±1.2 mm repeatability at 0.5–1 m", not "accurate to
  1 mm".

### ISO/IEC Guide 98-3 (GUM): uncertainty of measurement

The GUM is the international method for expressing measurement uncertainty. It combines Type A
(statistical, from repeat readings) and Type B (other knowledge: specs, calibration) components into a
combined standard uncertainty, then an expanded uncertainty (usually k = 2, about 95%).

The device should:

- Keep an uncertainty budget for a measured distance: depth noise, depth bias against range,
  calibration (intrinsics, depth-to-colour alignment), registration error between scans, temperature
  drift.
- Give per-measurement uncertainty that grows with range: time-of-flight noise rises with distance and
  on dark or slanted surfaces.
- State expanded uncertainty and its k whenever a dimension (wall length, room area) is exported.

### ISO/IEC 17025: competence of testing and calibration labs

ISO/IEC 17025 is what an accredited lab follows: documented methods, traceable equipment, uncertainty
estimates and records. It applies to whoever tests the device, not to the device itself, and only
matters once accuracy claims need formal backing (sales, contracts, legal measurements).

The device should:

- Keep test records good enough to hand to a 17025 lab: method, reference artifacts and their
  calibration certificates, raw data, analysis scripts.
- Measure against references with known traceability (a calibrated tape, a certified ball bar), not a
  door or A4 paper, once claims are made outside the team.

---

## 2. Building and space data

These apply once scans become floor plans, areas or building models for architects, facilities
managers or construction teams. Those users work in ISO 19650 and IFC, and expect areas to follow
ISO 9836.

### ISO 9836: area and space indicators

ISO 9836 (Performance standards in building) defines how to calculate building areas and volumes:
gross and net floor area, usable area, circulation area, volume. It says what counts (for example,
where a wall's thickness is measured) and what doesn't. Without it, two people measuring the same room
can report different "floor areas" and both be right.

The device should:

- Label every reported area with the definition used, e.g. net floor area at floor level, between
  finished wall faces.
- Measure walls at a stated height above the floor: walls are rarely plumb, and skirting boards get in
  the way.
- Report volume from the floor-to-ceiling height, and mark areas where the ceiling wasn't seen as
  estimated.
- Leave furniture out of floor area, with a separate "free floor area" if wanted.

### ISO 16739: Industry Foundation Classes (IFC)

IFC is the open data model for building information. It describes walls, slabs, spaces, doors and
openings as typed objects with geometry and properties, not just triangles. Revit, ArchiCAD,
BlenderBIM and most BIM viewers read it.

The device should:

- Export the clean room model as IFC: `IfcSpace` for the room, `IfcWall` and `IfcSlab` for walls,
  floor and ceiling, `IfcOpeningElement` for doors and windows where found.
- Write the measurement uncertainty and scan date into each element's property sets.
- Keep `.obj` for viewing; treat IFC as the hand-off format for building professionals.
- Use metres and a stated up axis, as IFC requires (Spotter already exports a levelled room).

### ISO 19650 series: managing BIM information

ISO 19650 sets out how information is managed over a building's life: information requirements,
naming, status codes, a common data environment, level of information need. It is about process, not
file format.

The device should:

- Name exported files by the client's ISO 19650 convention when asked (project, originator, zone,
  level, type, role, number).
- Record metadata with each room: who scanned it, when, which device and firmware, which processing
  version, so it can carry a status such as "work in progress" or "shared".
- Say what a scan is fit for (rough space planning, yes; fabrication, no): the "level of information
  need" idea.

---

## 3. Point cloud formats

### ISO/IEC 23090-5 and 23090-9: MPEG point cloud compression

Two parts of MPEG-I: V-PCC (23090-5, video-based, projects the cloud to 2D and uses video codecs; suits
dense, moving clouds) and G-PCC (23090-9, geometry-based, octree coding; suits static scans and sparse
clouds). Both have lossless and lossy modes. They matter when clouds must be sent or streamed (Pi to
PC, PC to cloud), not for archiving.

The device should:

- Keep the master copy uncompressed or losslessly compressed. This matches the user's "no compression
  for room scans" note and D44's open capture-format question.
- If transfer size becomes the bottleneck, use G-PCC (lossless, or lossy with the error stated)
  rather than a home-made format.
- Never let a lossy copy replace the master, and record in the room folder which one a file is.

---

## 4. Safety

These are IEC standards, not ISO, but they are the ones a regulator or a venue (e.g. an exhibition)
would ask about.

### IEC 60825-1: safety of laser products

IEC 60825-1 classifies lasers by the radiation they can emit (Class 1 is safe under all reasonably
foreseeable use) and sets labelling and manufacturer duties. The Kinect's infrared emitter is rated
Class 1 as sold. That rating holds only for the unmodified device.

The device should:

- Never modify the emitter, its diffuser or its drive current, or place optics (lenses, magnifying
  windows) in front of it. Any of these can void Class 1.
- If the Kinect goes inside an enclosure, use a flat, clear window and keep the class label visible
  or reproduced on the outside.
- State "contains a Class 1 laser product" in the device's documentation.

### IEC 62471: photobiological safety of lamps and LEDs

IEC 62471 rates lamps and LEDs (including infrared) into risk groups for eye and skin exposure. It
does not cover the Kinect's laser (that is 60825-1), but it covers any lamps or LEDs added to the
scanner.

The device should:

- Use only exempt or risk-group-1 parts if extra lighting is added (e.g. LED work lights to help
  feature matching in dim rooms, or an IR illuminator).
- Keep status LEDs and any added lighting out of the depth camera's view, so they don't add depth
  noise.

### IEC 62368-1: audio/video, IT and communication equipment

IEC 62368-1 is the hazard-based safety standard for electronic equipment: electrical energy sources,
heat, fire, mechanical hazards. It applies if the scanner is sold or deployed as a product.

The device should:

- Use certified power supplies for the Pi and the Kinect's 12 V adapter. No bare mains wiring in a
  home-made enclosure.
- Keep the Pi and the Kinect within their temperature limits inside any enclosure. Both get warm,
  and heat also shifts the Kinect's depth readings.
- Have stable mounting (tripod, weighted base) and strain relief on cables, so it can't topple or be
  pulled off a table.

---

## 5. Privacy and security

Room scans record people's private spaces. The colour frames can show faces, documents, screens and
photos, and even the depth alone shows a home's layout. The project's existing rule (never commit
camera frames, D29's exception for empty-room scans, scans gitignored) is the start of this.

### ISO/IEC 27001: information security management

ISO/IEC 27001 defines an information security management system: risk assessment, then controls
(Annex A) for access, encryption, logging, suppliers and incident response. It certifies an
organisation, not a device, but the device has to support the controls.

The device should:

- Encrypt scans at rest on the Pi's SD card and on the PC, and in transit between them (e.g. SSH/SFTP
  or TLS, never an open share).
- Require a login to reach the Pi (no default passwords, SSH keys only) and log who copied which room.
- Have a way to wipe the Pi after export, so a lost scanner doesn't carry old rooms.

### ISO/IEC 27701: privacy information management

ISO/IEC 27701 extends 27001 to personal data, with separate duties for controllers (who decide why
data is collected) and processors. It lines up closely with GDPR, so it is the practical route to
showing GDPR-style compliance.

The device should:

- Treat colour frames and point clouds of occupied spaces as personal data: record why each room was
  scanned and who asked.
- Support deleting a room on request, everywhere (Pi, PC, cloud), with a record that it was done.
- Keep a retention period per room (e.g. delete raw colour frames after the model is built, if they
  aren't needed).

### ISO/IEC 29100: privacy framework

ISO/IEC 29100 gives the privacy principles behind the other two: consent and choice, purpose
limitation, collection limitation, data minimisation, openness, accountability.

The device should:

- Collect the minimum: a depth-only mode when colour isn't needed, and remove people from frames
  before saving (07d already paints people out).
- Show that it is recording (a light or a screen message) so people in the space know.
- Let people opt out: pause scanning while someone is in view.

---

## 6. Non-ISO but widely used

### ASTM E2807: the E57 file format

E57 is the vendor-neutral exchange format for 3D imaging data: point clouds (with intensity and
colour), each scan's pose, and the images taken with them, in one file. CloudCompare, Recap, Cyclone,
Scene and most survey software read it. The open-source libE57Format implements it, which fits the
project's preference for open source (D5).

The device should:

- Export a room folder as `.e57`: one scan section per scan with its pose, colour per point, and
  optionally `colour_full` as the scan's image.
- Treat E57 as the hand-off format for survey and point-cloud users; keep `.ply` for quick viewing.

### ASTM E2938: relative-range performance of 3D imaging systems

ASTM E2938 is a test method for how well a 3D imaging system measures the distance between two points
along its line of sight, in the medium range, using targets at known spacing. It suits a fixed depth
camera better than ISO 17123-9 does.

The device should:

- Run this as its main range test: a target moved along the line of sight in steps, compared with a
  reference distance at each step.
- Report bias and spread per range band (e.g. 0.5–1, 1–2, 2–4.5, 4.5–6 m). Spotter's range past
  4.5 m is untested for bias.

### ASTM E3125: point-to-point distance performance

ASTM E3125 tests point-to-point distances between targets at many positions and orientations around a
spherical-coordinate scanner (one that rotates). A Kinect isn't one, so the method needs adapting:
take its target layouts, not its instrument assumptions.

The device should:

- Use its target layouts for whole-room checks: targets spread across the room, distances between them
  measured in the merged room and compared with references.
- Treat the result as the end-to-end number (registration errors included), separate from the
  single-view range test.

### VDI/VDE 2634 Part 2: optical area-scanning systems

The German VDI/VDE guideline for testing optical 3D systems that scan areas (Part 2: systems that
measure in single views). It defines probing error (on a sphere), sphere-spacing error (on a dumbbell
or ball bar) and flatness error (on a flat plane). It is often cited alongside ISO 10360-13.

The device should:

- Run the flatness test: scan a known-flat board, fit a plane, report the spread of points from it.
  Spotter already fits planes, so this is cheap.
- Run the sphere-spacing test with two spheres at a known spacing, at several ranges.

---

## 7. Checklist

| Priority | What the device should do | Standards | Spotter today |
|---|---|---|---|
| 1 | Range test with targets: bias and spread per range band | ASTM E2938, ISO 5725, ISO 17123-9 | Accuracy milestone open; only noise measured (07, 07b) |
| 2 | Uncertainty budget; per-measurement uncertainty in exports | ISO/IEC Guide 98-3 | Not started |
| 3 | Flatness and sphere tests | VDI/VDE 2634-2, ISO 10360-13 | Plane fits exist; tests not written |
| 4 | Whole-room target distances (merged room) | ASTM E3125 (adapted) | Synthetic checks only |
| 5 | Emitter left unmodified; class label visible | IEC 60825-1 | Unmodified Kinect |
| 6 | Certified power supplies; thermal limits; stable mount | IEC 62368-1 | Not assessed |
| 7 | Encryption, access control, wipe after export | ISO/IEC 27001 | Scans gitignored; no encryption |
| 8 | Minimise data, deletion on request, recording indicator | ISO/IEC 27701, 29100 | 07d removes people; D29 empty-room rule |
| 9 | E57 export | ASTM E2807 | `.ply`, `.obj` only |
| 10 | Areas labelled by definition | ISO 9836 | No areas reported |
| 11 | IFC export with metadata | ISO 16739, ISO 19650 | `.obj` only |
| 12 | Lossless masters; G-PCC only for transfer | ISO/IEC 23090-9 | Uncompressed; D44 open |
