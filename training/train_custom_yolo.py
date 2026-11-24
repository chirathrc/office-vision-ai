"""
Custom Office Activity AI Model Trainer (Ultralytics YOLOv8)
-----------------------------------------------------------
1. Capture labeled frames using the OfficeVision AI web app ("Train Your Own Model" panel)
   or annotate office camera videos with Label Studio / Roboflow in YOLO format.
2. Install dependencies:
      pip install ultralytics
3. Run training:
      python train_custom_yolo.py
4. Export to TensorFlow.js or ONNX to plug directly into the web app or edge camera server.
"""
from ultralytics import YOLO


def main():
    # Start from lightweight pretrained YOLOv8-nano for real-time 30+ FPS camera inference
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="office_activity_v1",
    )

    # Export trained model to ONNX & TensorFlow.js format
    model.export(format="onnx")
    print("Training complete! Best weights saved under runs/detect/office_activity_v1/weights/best.pt")


if __name__ == "__main__":
    main()
