import cv2
import numpy as np
import json
from pathlib import Path
import sys

class SimplePoseEstimator:
    def __init__(self):
        print("Initializing simple pose estimator...")
        
    def estimate_pose_simple(self, image, bbox):
        x, y, w, h = [int(v) for v in bbox]
        
        # Ensure valid crop coordinates
        h_img, w_img = image.shape[:2]
        x = max(0, min(x, w_img-1))
        y = max(0, min(y, h_img-1))
        x2 = max(x+1, min(x+w, w_img))
        y2 = max(y+1, min(y+h, h_img))
        
        person_crop = image[y:y2, x:x2]
        if person_crop.size == 0:
            return None
        
        # Simple pose features based on bounding box and basic CV
        pose_landmarks = {
            'center_x': (x + w/2) / image.shape[1],
            'center_y': (y + h/2) / image.shape[0],
            'bbox_width': w / image.shape[1],
            'bbox_height': h / image.shape[0],
            'aspect_ratio': h / w if w > 0 else 0,
            'area_ratio': (w * h) / (image.shape[0] * image.shape[1])
        }
        
        return pose_landmarks
    
    def extract_gait_features(self, pose_sequence):
        if len(pose_sequence) < 2:
            return None
        
        features = {
            'movement_pattern': [],
            'total_movement': 0,
            'pose_changes': 0
        }
        
        for i in range(1, len(pose_sequence)):
            prev_pose = pose_sequence[i-1]
            curr_pose = pose_sequence[i]
            
            if prev_pose and curr_pose:
                dx = curr_pose['center_x'] - prev_pose['center_x']
                dy = curr_pose['center_y'] - prev_pose['center_y']
                movement = np.sqrt(dx**2 + dy**2)
                
                features['movement_pattern'].append(movement)
                features['total_movement'] += movement
                
                if movement > 0.01:
                    features['pose_changes'] += 1
        
        if features['movement_pattern']:
            features['avg_movement'] = np.mean(features['movement_pattern'])
            features['movement_variance'] = np.var(features['movement_pattern'])
        
        return features

def process_simple_pose_estimation(scene_id, cam_id):
    print(f"Starting pose estimation for {scene_id}/{cam_id}")
    pose_estimator = SimplePoseEstimator()
    
    tracks_file = Path(f'data/tracks/{scene_id}/{cam_id}_consistent.json')
    if not tracks_file.exists():
        print(f"Tracking file not found: {tracks_file}")
        return None
    
    with open(tracks_file) as f:
        tracks_data = json.load(f)
    
    frames_path = Path(f'data/keyframes/{scene_id}/{cam_id}')
    frame_files = sorted(frames_path.glob('seq_*.jpg'))
    
    if not frame_files:
        print(f"No frames found in {frames_path}")
        return None
    
    tracks_by_frame = {}
    tracks_by_id = {}
    
    for track in tracks_data['tracks']:
        frame_idx = track['frame']
        track_id = track['track_id']
        
        if frame_idx not in tracks_by_frame:
            tracks_by_frame[frame_idx] = []
        tracks_by_frame[frame_idx].append(track)
        
        if track_id not in tracks_by_id:
            tracks_by_id[track_id] = []
        tracks_by_id[track_id].append(track)
    
    all_poses = []
    
    for frame_idx, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file))
        if frame is None:
            continue
        
        if frame_idx in tracks_by_frame:
            for track in tracks_by_frame[frame_idx]:
                track_id = track['track_id']
                bbox = track['bbox']
                
                pose_landmarks = pose_estimator.estimate_pose_simple(frame, bbox)
                
                if pose_landmarks:
                    pose_data = {
                        'frame': frame_idx,
                        'track_id': track_id,
                        'bbox': bbox,
                        'pose_landmarks': pose_landmarks,
                        'timestamp': frame_idx / 30.0
                    }
                    all_poses.append(pose_data)
        
        if frame_idx % 2 == 0:
            print(f"  Processed frame {frame_idx}/{len(frame_files)}")
    
    # Gait analysis
    gait_analysis = {}
    for track_id in tracks_by_id:
        track_poses = [p['pose_landmarks'] for p in all_poses if p['track_id'] == track_id]
        if track_poses:
            gait_features = pose_estimator.extract_gait_features(track_poses)
            gait_analysis[track_id] = gait_features
    
    output_path = Path(f'data/poses/{scene_id}')
    output_path.mkdir(parents=True, exist_ok=True)
    
    pose_summary = {
        'scene_id': scene_id,
        'camera_id': cam_id,
        'total_poses': len(all_poses),
        'unique_tracks': len(set(p['track_id'] for p in all_poses)),
        'gait_analysis': gait_analysis,
        'poses': all_poses
    }
    
    output_file = output_path / f'{cam_id}_poses.json'
    with open(output_file, 'w') as f:
        json.dump(pose_summary, f, indent=2)
    
    print(f"Pose estimation complete!")
    print(f"  Total poses: {len(all_poses)}")
    print(f"  Tracks with gait analysis: {len(gait_analysis)}")
    print(f"  Output saved: {output_file}")
    
    return pose_summary

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        scene_id = sys.argv[1]
        cam_id = sys.argv[2]
        process_simple_pose_estimation(scene_id, cam_id)
    else:
        print("Usage: python cv/simple_pose_estimator.py scene_001 cam_001")
