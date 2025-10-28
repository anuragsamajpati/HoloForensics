from ultralytics import YOLO
import cv2
import numpy as np
import json
from pathlib import Path
from byte_tracker import SimpleTracker

class MultiObjectTracker:
    def __init__(self):
        print("Loading YOLO model...")
        self.model = YOLO("yolov8n.pt")
        self.tracker = SimpleTracker()
        print("Tracker initialized!")

    def detect_and_track(self, frame):
        # Run detection
        results = self.model(frame, conf=0.3, verbose=False)
        detections = []

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "confidence": float(confidences[i]),
                    "class_id": int(classes[i]),
                    "class_name": self.model.names[classes[i]]
                })

        # Run tracking
        tracks = self.tracker.update(detections)
        return tracks

def main():
    tracker = MultiObjectTracker()

    # Create test data
    Path("data/keyframes/scene_001/cam_001").mkdir(parents=True, exist_ok=True)
    Path("data/tracks/scene_001").mkdir(parents=True, exist_ok=True)

    # Create test images with multiple objects
    for i in range(5):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 100
        # Object 1 moving right
        cv2.rectangle(img, (50+i*20, 100), (100+i*20, 200), (200, 150, 100), -1)
        # Object 2 moving down
        cv2.rectangle(img, (300, 50+i*15), (400, 150+i*15), (100, 200, 150), -1)
        cv2.imwrite(f"data/keyframes/scene_001/cam_001/frame_{i:03d}.jpg", img)

    print("Created test sequence with moving objects")

    # Process frames
    all_tracks = []
    frame_files = sorted(Path("data/keyframes/scene_001/cam_001").glob("*.jpg"))

    for frame_idx, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file))
        tracks = tracker.detect_and_track(frame)

        print(f"Frame {frame_idx}: {len(tracks)} tracks")
        for track in tracks:
            track_data = {
                "frame": frame_idx,
                "track_id": track.track_id,
                "bbox": track.tlwh.tolist(),
                "confidence": float(track.score),
                "class_id": track.cls_id
            }
            all_tracks.append(track_data)

    # Save tracks
    tracks_summary = {
        "scene_id": "scene_001",
        "camera_id": "cam_001",
        "total_tracks": len(all_tracks),
        "tracks": all_tracks
    }

    with open("data/tracks/scene_001/cam_001_tracks.json", "w") as f:
        json.dump(tracks_summary, f, indent=2)

    print(f"Saved {len(all_tracks)} tracks to data/tracks/scene_001/cam_001_tracks.json")

if __name__ == "__main__":
    main()
