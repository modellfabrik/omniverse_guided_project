from ultralytics import YOLO

# to train the model:
# Load a pretrained model (you can pick yolov8n, yolov8s, etc.)
model = YOLO("yolov8n.pt")
# Train on your dataset
model.train(data="data.yaml", epochs=50, imgsz=640)

# To run inference:

from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run inference on an image or video
results = model("image.jpg")   # for a single image
# results = model("video.mp4") # for a video

# Show or save results
results[0].show()      # display with boxes
results[0].save("out.jpg")  # save annotated image
