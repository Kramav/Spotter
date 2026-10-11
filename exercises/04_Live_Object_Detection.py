from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(0)                     # 0 = default webcam; try 1 if it's the wrong camera
if not cap.isOpened():
    raise SystemExit("Unable to open webcam")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    result = model(frame, verbose=False)[0]
    cv2.imshow("Detections", result.plot())
    if cv2.waitKey(1) in (ord("q"), 27):      # q or Esc to quit
        break

cap.release()
cv2.destroyAllWindows()
