from ultralytics import YOLO
import cv2

model = YOLO("yolo11x.pt") 

image_path = 'rgb_0000.png'
results = model(image_path)

image = cv2.imread(image_path)

result = results[0]

# Access boxes, scores, and class IDs
boxes = result.boxes.xyxy.cpu().numpy()        # [[x1, y1, x2, y2], ...]
confidences = result.boxes.conf.cpu().numpy()  # [0.85, 0.9, ...]
class_ids = result.boxes.cls.cpu().numpy().astype(int)  # [0, 1, ...]
class_names = model.names

for box, conf, class_id in zip(boxes, confidences, class_ids):
    x1, y1, x2, y2 = map(int, box)
    label = f"{class_names[class_id]}: {conf:.2f}"

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - 20), (x1 + text_width, y1), (0, 255, 0), -1)

    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.imshow("YOLO Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'output_{image_path}.jpg', image)
