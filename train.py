from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="data.yaml", epochs=300, imgsz=640)
