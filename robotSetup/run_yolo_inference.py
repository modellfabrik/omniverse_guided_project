import cv2
from ultralytics import YOLO

# This is a default script to run inference from a webcam stream.
# You can adapt it to your code from the isaac-sim stream

model = YOLO("/home/modellfabrik2/yolo_models/yolo11x.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # this part here...
    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
