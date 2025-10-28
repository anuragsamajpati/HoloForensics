# cv/visualize_detections.py
import cv2
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class DetectionVisualizer:
    def __init__(self):
        # Color palette for different classes
        self.colors = {
            'person': (0, 255, 0),
            'bicycle': (255, 0, 255),
            'car': (255, 0, 0),
            'motorcycle': (0, 255, 255),
            'airplane': (255, 255, 0),
            'bus': (255, 165, 0),
            'train': (128, 0, 128),
            'truck': (0, 0, 255),
            'boat': (0, 128, 255),
            'traffic light': (255, 192, 203),
            'backpack': (255, 20, 147),
            'handbag': (0, 191, 255),
            'suitcase': (139, 69, 19)
        }
        self.default_color = (128, 128, 128)
    
    def draw_detections_on_frame(self, frame, detections, frame_idx, conf_thresh=0.5):
        """Draw bounding boxes on frame"""
        frame_copy = frame.copy()
        
        frame_detections = [d for d in detections if d['frame'] == frame_idx and d['confidence'] >= conf_thresh]
        
        for det in frame_detections:
            x, y, w, h = det['bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
            
            color = self.colors.get(det['class_name'], self.default_color)
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Label background
            cv2.rectangle(frame_copy, (x1, y1-label_size[1]-10), 
                         (x1+label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(frame_copy, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add frame info
        info_text = f"Frame {frame_idx} | Detections: {len(frame_detections)}"
        cv2.putText(frame_copy, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame_copy
    
    def create_detection_video(self, scene_id, cam_id, output_fps=10):
        """Create video with detection overlays"""
        # Load detections
        det_file = Path(f'data/detections/{scene_id}/{cam_id}_detections.json')
        if not det_file.exists():
            print(f"Detection file not found: {det_file}")
            return None
        
        with open(det_file) as f:
            data = json.load(f)
        
        detections = data['detections']
        
        # Load frames
        frames_dir = Path(f'data/keyframes/{scene_id}/{cam_id}')
        frame_files = sorted(frames_dir.glob('*.jpg'))
        
        if not frame_files:
            print(f"No frames found in {frames_dir}")
            return None
        
        # Setup video writer
        first_frame = cv2.imread(str(frame_files[0]))
        height, width = first_frame.shape[:2]
        
        output_path = Path(f'data/detections/{scene_id}/{cam_id}_detections.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
        
        print(f"Creating detection video: {output_path}")
        
        for frame_idx, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue
            
            # Draw detections
            frame_with_detections = self.draw_detections_on_frame(frame, detections, frame_idx)
            out.write(frame_with_detections)
            
            if frame_idx % 50 == 0:
                print(f"  Processed {frame_idx}/{len(frame_files)} frames")
        
        out.release()
        print(f"Detection video saved: {output_path}")
        return output_path
    
    def analyze_detection_quality(self, scene_id):
        """Comprehensive detection quality analysis"""
        detections_dir = Path(f'data/detections/{scene_id}')
        
        if not detections_dir.exists():
            print(f"Detections directory not found: {detections_dir}")
            return None
        
        analysis = {
            'scene_id': scene_id,
            'cameras': {},
            'overall_stats': {
                'total_detections': 0,
                'total_frames': 0,
                'avg_detections_per_frame': 0,
                'class_distribution': defaultdict(int),
                'confidence_stats': []
            }
        }
        
        all_confidences = []
        
        # Analyze each camera
        for det_file in detections_dir.glob('*_detections.json'):
            with open(det_file) as f:
                data = json.load(f)
            
            cam_id = data['camera_id']
            detections = data['detections']
            
            # Per-camera analysis
            cam_analysis = {
                'total_detections': len(detections),
                'total_frames': data.get('total_frames', 0),
                'class_distribution': defaultdict(int),
                'confidence_distribution': [],
                'temporal_distribution': defaultdict(int)
            }
            
            for det in detections:
                # Class distribution
                cam_analysis['class_distribution'][det['class_name']] += 1
                analysis['overall_stats']['class_distribution'][det['class_name']] += 1
                
                # Confidence distribution
                conf = det['confidence']
                cam_analysis['confidence_distribution'].append(conf)
                all_confidences.append(conf)
                
                # Temporal distribution
                frame_bin = det['frame'] // 30  # Group by 30-frame bins
                cam_analysis['temporal_distribution'][frame_bin] += 1
            
            # Calculate statistics
            if cam_analysis['confidence_distribution']:
                cam_analysis['avg_confidence'] = np.mean(cam_analysis['confidence_distribution'])
                cam_analysis['min_confidence'] = np.min(cam_analysis['confidence_distribution'])
                cam_analysis['max_confidence'] = np.max(cam_analysis['confidence_distribution'])
            
            if cam_analysis['total_frames'] > 0:
                cam_analysis['avg_detections_per_frame'] = cam_analysis['total_detections'] / cam_analysis['total_frames']
            
            analysis['cameras'][cam_id] = cam_analysis
            analysis['overall_stats']['total_detections'] += cam_analysis['total_detections']
            analysis['overall_stats']['total_frames'] += cam_analysis['total_frames']
        
        # Overall statistics
        if analysis['overall_stats']['total_frames'] > 0:
            analysis['overall_stats']['avg_detections_per_frame'] = (
                analysis['overall_stats']['total_detections'] / analysis['overall_stats']['total_frames']
            )