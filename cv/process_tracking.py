from ultralytics import YOLO
import cv2
import json
from pathlib import Path
from advanced_tracker import AdvancedTracker

def process_scene_tracking(scene_id, cam_id):
    # Initialize
    model = YOLO('yolov8n.pt')
    tracker = AdvancedTracker()
    
    # Paths
    frames_path = Path(f'data/keyframes/{scene_id}/{cam_id}')
    output_path = Path(f'data/tracks/{scene_id}')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get frame files
    frame_files = sorted(frames_path.glob('*.jpg'))
    all_tracks = []
    
    print(f"Processing {len(frame_files)} frames for {scene_id}/{cam_id}")
    
    for frame_idx, frame_file in enumerate(frame_files):
        # Load frame
        frame = cv2.imread(str(frame_file))
        if frame is None:
            continue
            
        # Detect objects
        results = model(frame, conf=0.3, verbose=False)
        detections = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'confidence': float(confidences[i]),
                    'class_id': int(classes[i]),
                    'class_name': model.names[classes[i]]
                })
        
        # Update tracker
        tracks = tracker.update(detections)
        
        print(f"Frame {frame_idx}: {len(detections)} detections -> {len(tracks)} tracks")
        
        # Store track data
        for track in tracks:
            track_data = {
                'frame': frame_idx,
                'track_id': track.track_id,
                'bbox': track.bbox,
                'confidence': track.confidence,
                'class_id': track.class_id,
                'class_name': track.class_name,
                'age': track.age,
                'hits': track.hits
            }
            all_tracks.append(track_data)
    
    # Save results
    tracks_summary = {
        'scene_id': scene_id,
        'camera_id': cam_id,
        'total_frames': len(frame_files),
        'total_tracks': len(all_tracks),
        'unique_tracks': len(set(t['track_id'] for t in all_tracks)),
        'tracks': all_tracks
    }
    
    output_file = output_path / f'{cam_id}_tracks2d.json'
    with open(output_file, 'w') as f:
        json.dump(tracks_summary, f, indent=2)
    
    print(f"Saved {len(all_tracks)} tracks ({tracks_summary['unique_tracks']} unique) to {output_file}")
    return tracks_summary

if __name__ == "__main__":
    import sys
    scene_id = sys.argv[1] if len(sys.argv) > 1 else 'scene_001'
    cam_id = sys.argv[2] if len(sys.argv) > 2 else 'cam_001'
    process_scene_tracking(scene_id, cam_id)