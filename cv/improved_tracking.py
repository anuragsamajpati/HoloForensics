from ultralytics import YOLO
import cv2
import json
from pathlib import Path
from consistent_tracker import ConsistentTracker
import sys

def process_tracking(scene_id, cam_id):
    """Process tracking for any scene/camera combination"""
    print(f"Starting tracking for {scene_id}/{cam_id}")
    
    try:
        model = YOLO('yolov8n.pt')
        tracker = ConsistentTracker()
        
        frames_path = Path(f'data/keyframes/{scene_id}/{cam_id}')
        output_path = Path(f'data/tracks/{scene_id}')
        
        print(f"Frames path: {frames_path}")
        print(f"Output path: {output_path}")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created/verified: {output_path.exists()}")
        
        if not frames_path.exists():
            print(f"Error: Frames path not found: {frames_path}")
            return None
        
        # Try different frame patterns
        frame_files = []
        for pattern in ['seq_*.jpg', 'frame_*.jpg', '*.jpg', '*.png']:
            frame_files = sorted(frames_path.glob(pattern))
            if frame_files:
                print(f"Found {len(frame_files)} frames with pattern {pattern}")
                break
        
        if not frame_files:
            print(f"Error: No frames found in {frames_path}")
            return None
        
        all_tracks = []
        
        for frame_idx, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                print(f"Warning: Could not load frame {frame_file}")
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
                        'class_name': model.names[classes[i]]
                    })
            
            # Update tracker
            tracks = tracker.update(detections)
            
            print(f"Frame {frame_idx}: {len(detections)} detections -> {len(tracks)} tracks")
            
            # Store track data
            for track in tracks:
                track_data = {
                    'frame': frame_idx,
                    'track_id': track['track_id'],
                    'bbox': track['bbox']
                }
                all_tracks.append(track_data)
        
        # Save results
        unique_tracks = len(set(t['track_id'] for t in all_tracks))
        tracks_summary = {
            'scene_id': scene_id,
            'camera_id': cam_id,
            'total_tracks': len(all_tracks),
            'unique_tracks': unique_tracks,
            'tracks': all_tracks
        }
        
        output_file = output_path / f'{cam_id}_consistent.json'
        print(f"Attempting to save to: {output_file}")
        
        try:
            with open(output_file, 'w') as f:
                json.dump(tracks_summary, f, indent=2)
            print(f"File saved successfully: {output_file}")
            print(f"File exists after save: {output_file.exists()}")
        except Exception as save_error:
            print(f"Error saving file: {save_error}")
            return None
        
        print(f"Completed: {len(all_tracks)} tracks ({unique_tracks} unique IDs)")
        return tracks_summary
        
    except Exception as e:
        print(f"Error in process_tracking: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_all_scenes():
    """Process tracking for all available scenes"""
    keyframes_root = Path('data/keyframes')
    if not keyframes_root.exists():
        print("No keyframes directory found")
        return
    
    scenes_processed = 0
    
    for scene_dir in keyframes_root.iterdir():
        if not scene_dir.is_dir():
            continue
            
        scene_id = scene_dir.name
        print(f"\n=== Processing Scene: {scene_id} ===")
        
        for cam_dir in scene_dir.iterdir():
            if not cam_dir.is_dir():
                continue
                
            cam_id = cam_dir.name
            try:
                result = process_tracking(scene_id, cam_id)
                if result:
                    scenes_processed += 1
            except Exception as e:
                print(f"Error processing {scene_id}/{cam_id}: {e}")
    
    print(f"\nTotal scenes processed: {scenes_processed}")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        scene_id = sys.argv[1]
        cam_id = sys.argv[2]
        result = process_tracking(scene_id, cam_id)
        if result:
            print("SUCCESS: Tracking completed successfully")
        else:
            print("FAILED: Tracking did not complete")
    elif len(sys.argv) == 2:
        if sys.argv[1] == "all":
            process_all_scenes()
        else:
            print("Usage:")
            print("  python cv/improved_tracking.py scene_001 cam_001  # Specific scene/camera")
            print("  python cv/improved_tracking.py all               # All scenes")
    else:
        print("Usage:")
        print("  python cv/improved_tracking.py scene_001 cam_001  # Specific scene/camera")
        print("  python cv/improved_tracking.py all               # All scenes")