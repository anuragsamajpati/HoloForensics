from ultralytics import YOLO
import cv2
import numpy as np
import json
from pathlib import Path
import time
import sys

class ObjectDetector:
    def __init__(self):
        print("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')
        self.class_names = self.model.names
        print("Model loaded successfully!")
        
    def detect_frame(self, frame):
        results = self.model(frame, conf=0.3, verbose=False)
        detections = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                detection = {
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'confidence': float(confidences[i]),
                    'class_id': int(classes[i]),
                    'class_name': self.class_names[classes[i]]
                }
                detections.append(detection)
        
        return detections

def process_scene_detections(scene_id, cam_id=None):
    """Process detections for a scene. If cam_id is None, process all cameras."""
    detector = ObjectDetector()
    
    keyframes_path = Path(f'data/keyframes/{scene_id}')
    output_path = Path(f'data/detections/{scene_id}')
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not keyframes_path.exists():
        print(f"Keyframes path not found: {keyframes_path}")
        return
    
    # Determine which cameras to process
    if cam_id:
        cam_dirs = [keyframes_path / cam_id]
    else:
        cam_dirs = [d for d in keyframes_path.iterdir() if d.is_dir()]
    
    for cam_dir in cam_dirs:
        if not cam_dir.exists():
            print(f"Camera directory not found: {cam_dir}")
            continue
            
        cam_name = cam_dir.name
        print(f"Processing {scene_id}/{cam_name}...")
        
        # Try different frame patterns
        frame_files = []
        for pattern in ['seq_*.jpg', 'frame_*.jpg', '*.jpg', '*.png']:
            frame_files = sorted(cam_dir.glob(pattern))
            if frame_files:
                print(f"  Found {len(frame_files)} frames with pattern {pattern}")
                break
        
        if not frame_files:
            print(f"  No frames found in {cam_dir}")
            continue
        
        all_detections = []
        start_time = time.time()
        
        for frame_idx, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue
            
            detections = detector.detect_frame(frame)
            
            for det in detections:
                det['frame'] = frame_idx
                det['timestamp'] = frame_idx / 30.0
                det['frame_file'] = frame_file.name
            
            all_detections.extend(detections)
            
            if frame_idx % 20 == 0:
                print(f"    Frame {frame_idx}/{len(frame_files)}")
        
        # Save detections
        output_file = output_path / f'{cam_name}_detections.json'
        detection_summary = {
            'scene_id': scene_id,
            'camera_id': cam_name,
            'total_frames': len(frame_files),
            'total_detections': len(all_detections),
            'detections': all_detections
        }
        
        with open(output_file, 'w') as f:
            json.dump(detection_summary, f, indent=2)
        
        processing_time = time.time() - start_time
        print(f"  Completed {cam_name}: {len(all_detections)} detections in {processing_time:.1f}s")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python cv/detect_objects.py scene_001                # Process all cameras")
        print("  python cv/detect_objects.py scene_001 cam_001        # Process specific camera")
        sys.exit(1)
    
    scene_id = sys.argv[1]
    cam_id = sys.argv[2] if len(sys.argv) > 2 else None
    process_scene_detections(scene_id, cam_id)