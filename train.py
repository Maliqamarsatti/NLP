# train.py
from ultralytics import YOLO

# Load the base YOLOv8 model
model = YOLO("yolov8n.pt")

# Train: save a checkpoint every 5 epochs
model.train(
    data="data.yaml",   # path to your data config
    epochs=30,          # total number of epochs
    imgsz=640,          # input image size
    batch=16,           # batch size
    name="coffee_model",# output folder runs/detect/coffee_model
    save_period=5       # save checkpoint every 5 epochs
)
