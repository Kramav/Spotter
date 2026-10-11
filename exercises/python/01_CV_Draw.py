import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open camera 0 (try 1, or close other apps using it)")
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                                                                 # mirror function
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)                       # draws a rectangle // openCV uses BGR, not RGB.
        #cv2.circle(frame, (w//2, h//2), min(w, h)//4, (255, 0, 0), 2)                              # draws a circle
        #cv2.ellipse(frame, (w//2, h//2), (min(w, h)//4, min(w, h)//6), 0, 0, 360, (255, 255, 0), 2)# draws an ellipse
        #cv2.line(frame, (0, h//2), (w, h//2), (0, 0, 255), 2)                                      # draws a line
        cv2.putText(frame, f"{w}x{h}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # displays resolution
        cv2.imshow("webcam", frame)                                                                # displays the frame to the window
        if cv2.waitKey(1) & 0xFF == ord("q"):                                                      # if 'q' is pressed 
            break                                                                                  # quits the loop
finally:
    cap.release()
    cv2.destroyAllWindows()
