from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO('yolov8-fruit.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='fruit.yaml', epochs=25, imgsz=640, device='cuda', workers=2, batch=8)