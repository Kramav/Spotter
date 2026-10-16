from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(0)                     # 0 = default webcam; try 1 if it's the wrong camera
if not cap.isOpened():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    raise SystemExit("Unable to open webcam")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)       # camera picks the closest resolution it supports
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detections", 1280, 720)

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
