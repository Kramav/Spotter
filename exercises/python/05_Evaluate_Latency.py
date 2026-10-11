import time, cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(0)
for _ in range(10):                 # warm-up: first calls are slow
    ok, f = cap.read(); model(f, verbose=False)
times = []
for _ in range(100):
    ok, f = cap.read()
    t0 = time.perf_counter()
    model(f, verbose=False)
    times.append((time.perf_counter() - t0) * 1000)
cap.release()
times.sort()
print(f"median {times[50]:.1f} ms  p95 {times[95]:.1f} ms  ~{1000/times[50]:.1f} FPS")
