from ultralytics import YOLO
import cv2
import json
from pathlib import Path

def main():
    model = YOLO("yolov8n.pt")
    frames_path = Path("data/keyframes/scene_001/cam_001")
    output_path = Path("data/tracks/scene_001")
    output_path.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frames_path.glob("*.jpg"))
    all_tracks = []
    track_id = 1

    for frame_idx, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file))
        if frame is None:
            continue

        results = model(frame, conf=0.3, verbose=False)
        detections = []

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                track_data = {
                    "frame": frame_idx,
                    "track_id": track_id + i,
                    "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "confidence": float(confidences[i]),
                    "class_name": model.names[classes[i]]
                }
                all_tracks.append(track_data)

        print(f"Frame {frame_idx}: {len(results[0].boxes) if results[0].boxes else 0} detections")
        track_id += 10  # Increment for next frame

    with open("data/tracks/scene_001/cam_001_tracks2d.json", "w") as f:
        json.dump({"total_tracks": len(all_tracks), "tracks": all_tracks}, f, indent=2)

    print(f"Saved {len(all_tracks)} tracks")

if __name__ == "__main__":
    main()
