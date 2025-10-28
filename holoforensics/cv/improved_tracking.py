from ultralytics import YOLO
import cv2
import json
from pathlib import Path
from consistent_tracker import ConsistentTracker
import sys

def process_tracking(scene_id, cam_id):
    print(f"Starting tracking for {scene_id}/{cam_id}")
    
    model = YOLO("yolov8n.pt")
    tracker = ConsistentTracker()
    
    frames_path = Path(f"data/keyframes/{scene_id}/{cam_id}")
    output_path = Path(f"data/tracks/{scene_id}")
    
    print(f"Output path: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    frame_files = sorted(frames_path.glob("seq_*.jpg"))
    print(f"Found {len(frame_files)} frames")
    
    all_tracks = []
    
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
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "confidence": float(confidences[i]),
                    "class_name": model.names[classes[i]]
                })
        
        tracks = tracker.update(detections)
        print(f"Frame {frame_idx}: {len(detections)} detections -> {len(tracks)} tracks")
        
        for track in tracks:
            all_tracks.append({
                "frame": frame_idx,
                "track_id": track["track_id"],
                "bbox": track["bbox"]
            })
    
    # Save results
    unique_tracks = len(set(t["track_id"] for t in all_tracks))
    tracks_summary = {
        "scene_id": scene_id,
        "camera_id": cam_id,
        "total_tracks": len(all_tracks),
        "unique_tracks": unique_tracks,
        "tracks": all_tracks
    }
    
    output_file = output_path / f"{cam_id}_consistent.json"
    print(f"Saving to: {output_file}")
    
    with open(output_file, "w") as f:
        json.dump(tracks_summary, f, indent=2)
    
    print(f"File saved! Exists: {output_file.exists()}")
    print(f"Completed: {len(all_tracks)} tracks ({unique_tracks} unique IDs)")
    return tracks_summary

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        scene_id = sys.argv[1]
        cam_id = sys.argv[2]
        process_tracking(scene_id, cam_id)
    else:
        print("Usage: python cv/improved_tracking.py scene_002 cam_001")
