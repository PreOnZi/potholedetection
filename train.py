from ultralytics import YOLO
model = YOLO ('yolov8n.pt')

results = model.train(
    data='pothole.yaml',
    imgsz=640,
    epochs=100,
    batch=8,
    name='yolov8n_v8_50e'
)