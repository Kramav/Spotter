from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")
result = model("your_photo.jpg")[0]           # put any JPG next to the script
for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
    print(result.names[int(cls)], round(float(conf), 2), [round(v) for v in box.tolist()])
cv2.imwrite("detections.jpg", result.plot())
