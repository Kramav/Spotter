import csv
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

SOURCE = sys.argv[1] if len(sys.argv) > 1 else 0   # python 06_Real_Time_Tracking.py video.mp4
COOLDOWN = 10                                      # seconds between alerts
ZONE_FRAC = [(0.3, 0.3), (0.7, 0.3), (0.7, 0.9), (0.3, 0.9)]  # polygon as fractions of the frame
LINE_FRAC = ((0.05, 0.5), (0.95, 0.5))             # counting line, also fractions
LIVE = isinstance(SOURCE, int) or str(SOURCE).startswith("rtsp")  # files are read frame by frame

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise SystemExit(f"Unable to open {SOURCE}")


class Reader:
    """Live sources: a thread keeps only the newest frame, so slow inference never builds a backlog."""
    def __init__(self, cap):
        self.cap, self.item, self.new, self.done = cap, None, threading.Event(), False
        if LIVE:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.done:
            ok, frame = self.cap.read()
            self.item = (ok, frame, time.time())   # overwrite = drop stale frame
            self.new.set()
            if not ok:
                return

    def read(self):
        if not LIVE:
            ok, frame = self.cap.read()
            return ok, frame, time.time()
        self.new.wait()
        self.new.clear()
        return self.item


reader = Reader(cap)
h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# polygon is built in original-frame coordinates, same as the detections
zone = sv.PolygonZone(np.array([(x * w, y * h) for x, y in ZONE_FRAC], dtype=int))
zone_annotator = sv.PolygonZoneAnnotator(zone, color=sv.Color.RED)
line = sv.LineZone(sv.Point(LINE_FRAC[0][0] * w, LINE_FRAC[0][1] * h),
                   sv.Point(LINE_FRAC[1][0] * w, LINE_FRAC[1][1] * h))
line_annotator = sv.LineZoneAnnotator()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

log_file = open("zone_log.csv", "w", newline="")
log = csv.writer(log_file)
log.writerow(["track_id", "enter_time", "exit_time"])

inside = {}                  # track_id -> enter_time, for tracks currently in the zone
last_alert = -COOLDOWN
prev = time.time()

while True:
    ok, frame, captured = reader.read()
    if not ok:
        break
    now = time.time()

    # classes=[0] -> people only; persist=True keeps IDs between frames
    result = model.track(frame, persist=True, classes=[0], verbose=False)[0]
    det = sv.Detections.from_ultralytics(result)
    if det.tracker_id is None:                     # no tracks this frame
        det = sv.Detections.empty()
        det.tracker_id = np.array([], dtype=int)

    line.trigger(det)                              # counts unique track IDs crossing, not detections
    in_zone = zone.trigger(det)
    current = {int(t) for t in det.tracker_id[in_zone]}

    for tid in current - inside.keys():            # entered
        inside[tid] = datetime.now()
        if now - last_alert >= COOLDOWN:
            last_alert = now
            cv2.imwrite(f"alert_{inside[tid]:%H%M%S}_id{tid}.jpg", frame)
            print(f"ALERT: track {tid} entered zone")
    for tid in list(inside.keys() - current):      # exited
        log.writerow([tid, inside.pop(tid).isoformat(), datetime.now().isoformat()])
        log_file.flush()

    labels = [f"#{t}" for t in det.tracker_id]
    out = box_annotator.annotate(frame.copy(), det)
    out = label_annotator.annotate(out, det, labels=labels)
    out = zone_annotator.annotate(out)
    out = line_annotator.annotate(out, line_counter=line)
    latency = time.time() - captured               # if this keeps growing, you're buffering
    cv2.putText(out, f"{1 / (now - prev):.0f} FPS  lag {latency * 1000:.0f} ms", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    prev = now
    cv2.imshow("Tracking", out)
    if cv2.waitKey(1) in (ord("q"), 27):
        break

for tid, t in inside.items():                      # still inside when we quit
    log.writerow([tid, t.isoformat(), ""])
log_file.close()
reader.done = True
print(f"line crossings: in={line.in_count} out={line.out_count}")
cap.release()
cv2.destroyAllWindows()
