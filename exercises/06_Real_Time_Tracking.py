import csv
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

SOURCE = sys.argv[1] if len(sys.argv) > 1 else 0   # python 06_Real_Time_Tracking.py video.mp4
COOLDOWN = 10                                      # seconds between alerts
ZONE_FRAC = [(0.3, 0.3), (0.7, 0.3), (0.7, 0.9), (0.3, 0.9)]  # polygon as fractions of the frame

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise SystemExit(f"Unable to open {SOURCE}")

h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# polygon is built in original-frame coordinates, same as the detections
zone = sv.PolygonZone(np.array([(x * w, y * h) for x, y in ZONE_FRAC], dtype=int))
zone_annotator = sv.PolygonZoneAnnotator(zone, color=sv.Color.RED)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

log_file = open("zone_log.csv", "w", newline="")
log = csv.writer(log_file)
log.writerow(["track_id", "enter_time", "exit_time"])

inside = {}                  # track_id -> enter_time, for tracks currently in the zone
last_alert = -COOLDOWN
prev = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    now = time.time()

    # classes=[0] -> people only; persist=True keeps IDs between frames
    result = model.track(frame, persist=True, classes=[0], verbose=False)[0]
    det = sv.Detections.from_ultralytics(result)
    if det.tracker_id is None:                     # no tracks this frame
        det = sv.Detections.empty()
        det.tracker_id = np.array([], dtype=int)

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
    cv2.putText(out, f"{1 / (now - prev):.0f} FPS", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    prev = now
    cv2.imshow("Tracking", out)
    if cv2.waitKey(1) in (ord("q"), 27):
        break

for tid, t in inside.items():                      # still inside when we quit
    log.writerow([tid, t.isoformat(), ""])
log_file.close()
cap.release()
cv2.destroyAllWindows()
