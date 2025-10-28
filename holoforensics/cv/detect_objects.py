from ultralytics import YOLO
import cv2
import numpy as np
import json
from pathlib import Path

print("Loading YOLO...")
model = YOLO("yolov8n.pt")
print("YOLO loaded!")

Path("data/keyframes/scene_001/cam_001").mkdir(parents=True, exist_ok=True)
Path("data/detections/scene_001").mkdir(parents=True, exist_ok=True)

img = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
cv2.imwrite("data/keyframes/scene_001/cam_001/test.jpg", img)
print("Test image created")

results = model("data/keyframes/scene_001/cam_001/test.jpg")
print("Detection complete!")

with open("data/detections/scene_001/result.json", "w") as f:
    json.dump({"status": "success"}, f)

print("Results saved!")
