import numpy as np
import json
from pathlib import Path
import cv2
from collections import defaultdict
import uuid

class SimpleTracker:
    """Simplified tracking for Mac compatibility"""
    
    def __init__(self, max_disappeared=10, max_distance=100):
        self.next_id = 1
        self.tracks = {}  # track_id -> track_data
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def update(self, detections, frame_idx, timestamp):
        """Update tracker with new detections"""
        if len(detections) == 0:
            # Mark all tracks as disappeared
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return []
        
        # If no existing tracks, create new ones
        if len(self.tracks) == 0:
            for detection in detections:
                self.tracks[self.next_id] = {
                    'bbox': detection['bbox'],
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'disappeared': 0,
                    'frames': [{'frame_idx': frame_idx, 'timestamp': timestamp, 'bbox': detection['bbox']}]
                }
                self.next_id += 1
            return list(range(1, len(detections) + 1))
        
        # Compute distances between existing tracks and new detections
        track_ids = list(self.tracks.keys())
        costs = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track_bbox = self.tracks[track_id]['bbox']
            track_center = self.bbox_center(track_bbox)
            
            for j, detection in enumerate(detections):
                det_center = self.bbox_center(detection['bbox'])
                distance = np.linalg.norm(np.array(track_center) - np.array(det_center))
                costs[i, j] = distance
        
        # Simple assignment (Hungarian algorithm would be better)
        assigned_tracks = []
        used_detections = set()
        
        for i, track_id in enumerate(track_ids):
            min_cost_idx = np.argmin(costs[i, :])
            min_cost = costs[i, min_cost_idx]
            
            if min_cost < self.max_distance and min_cost_idx not in used_detections:
                # Update existing track
                detection = detections[min_cost_idx]
                self.tracks[track_id]['bbox'] = detection['bbox']
                self.tracks[track_id]['confidence'] = detection['confidence']
                self.tracks[track_id]['disappeared'] = 0
                self.tracks[track_id]['frames'].append({
                    'frame_idx': frame_idx,
                    'timestamp': timestamp,
                    'bbox': detection['bbox']
                })
                assigned_tracks.append(track_id)
                used_detections.add(min_cost_idx)
            else:
                # Mark as disappeared
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
        
        # Create new tracks for unmatched detections
        for j, detection in enumerate(detections):
            if j not in used_detections:
                self.tracks[self.next_id] = {
                    'bbox': detection['bbox'],
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'disappeared': 0,
                    'frames': [{'frame_idx': frame_idx, 'timestamp': timestamp, 'bbox': detection['bbox']}]
                }
                assigned_tracks.append(self.next_id)
                self.next_id += 1
        
        return assigned_tracks
    
    def bbox_center(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]
    
    def get_active_tracks(self):
        """Get currently active tracks"""
        return {tid: track for tid, track in self.tracks.items() if track['disappeared'] < self.max_disappeared}

class MultiObjectTracker:
    def __init__(self, scene_id):
        self.scene_id = scene_id
        self.detections_dir = Path(f"data/detections/{scene_id}")
        self.tracks_dir = Path(f"data/tracks/{scene_id}")
        self.tracks_dir.mkdir(parents=True, exist_ok=True)
    
    def track_single_camera(self, cam_id):
        """Track objects for a single camera"""
        print(f"Tracking objects for camera {cam_id}")
        
        # Load detections
        detections_file = self.detections_dir / f"{cam_id}_detections.json"
        with open(detections_file, 'r') as f:
            cam_data = json.load(f)
        
        # Initialize tracker
        tracker = SimpleTracker()
        
        # Process each frame
        tracked_frames = []
        for frame_data in cam_data:
            frame_idx = frame_data['frame_idx']
            timestamp = frame_data['timestamp']
            detections = frame_data['detections']
            
            # Update tracker
            active_track_ids = tracker.update(detections, frame_idx, timestamp)
            
            # Store tracking results
            tracked_detections = []
            for i, detection in enumerate(detections):
                if i < len(active_track_ids):
                    tracked_detection = detection.copy()
                    tracked_detection['track_id'] = active_track_ids[i]
                    tracked_detections.append(tracked_detection)
            
            tracked_frame = {
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'frame_path': frame_data['frame_path'],
                'tracked_objects': tracked_detections
            }
            tracked_frames.append(tracked_frame)
        
        # Save tracking results
        tracks_output = self.tracks_dir / f"{cam_id}_tracks.json"
        tracking_data = {
            'camera_id': cam_id,
            'total_frames': len(tracked_frames),
            'frames': tracked_frames,
            'track_summary': self.summarize_tracks(tracker.tracks)
        }
        
        with open(tracks_output, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        print(f"Saved tracks for {cam_id} to {tracks_output}")
        return tracking_data
    
    def summarize_tracks(self, tracks):
        """Create summary of all tracks"""
        summary = {}
        for track_id, track_data in tracks.items():
            frames = track_data['frames']
            summary[track_id] = {
                'class_name': track_data['class_name'],
                'duration_frames': len(frames),
                'first_seen': frames[0]['timestamp'],
                'last_seen': frames[-1]['timestamp'],
                'total_duration': frames[-1]['timestamp'] - frames[0]['timestamp']
            }
        return summary
    
    def track_all_cameras(self):
        """Track objects for all cameras in scene"""
        print(f"Tracking objects for scene {self.scene_id}")
        
        scene_tracks = {}
        
        # Track each camera
        for detections_file in self.detections_dir.glob("*_detections.json"):
            cam_id = detections_file.stem.replace('_detections', '')
            if cam_id != 'scene':  # Skip combined scene file
                tracking_data = self.track_single_camera(cam_id)
                scene_tracks[cam_id] = tracking_data
        
        # Save combined scene tracks
        scene_output = self.tracks_dir / "scene_tracks.json"
        with open(scene_output, 'w') as f:
            json.dump(scene_tracks, f, indent=2)
        
        print(f"Scene tracking complete: {scene_output}")
        return scene_tracks

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python cv/track_objects.py scene_001")
        sys.exit(1)
    
    scene_id = sys.argv[1]
    
    # Check if detections exist
    detections_dir = Path(f"data/detections/{scene_id}")
    if not detections_dir.exists():
        print(f"No detections found for {scene_id}. Run detect_objects.py first.")
        sys.exit(1)
    
    # Initialize tracker
    tracker = MultiObjectTracker(scene_id)
    
    # Track all cameras
    scene_tracks = tracker.track_all_cameras()
    
    # Print summary
    total_tracks = 0
    for cam_id, cam_data in scene_tracks.items():
        track_count = len(cam_data['track_summary'])
        total_tracks += track_count
        print(f"{cam_id}: {track_count} tracks")
    
    print(f"Total tracks across all cameras: {total_tracks}")

if __name__ == "__main__":
    main()