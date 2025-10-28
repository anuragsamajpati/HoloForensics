import cv2
import json
import numpy as np
from pathlib import Path
import sys

class PoseVisualizer:
    def __init__(self):
        self.behavior_colors = {
            'stationary': (0, 0, 255),      # Red
            'walking': (0, 255, 0),         # Green  
            'fast_walking': (0, 255, 255),  # Yellow
            'running': (255, 0, 255),       # Magenta
            'unknown': (128, 128, 128)      # Gray
        }
    
    def visualize_poses_and_behavior(self, scene_id, cam_id):
        """Create visualization combining tracking, poses, and behavior"""
        
        # Load all required data
        pose_file = Path(f'data/poses/{scene_id}/{cam_id}_poses.json')
        behavior_file = Path(f'data/behavior/{scene_id}/{cam_id}_behavior.json')
        
        if not pose_file.exists():
            print(f"Pose file not found: {pose_file}")
            return None
            
        if not behavior_file.exists():
            print(f"Behavior file not found: {behavior_file}")
            return None
        
        with open(pose_file) as f:
            pose_data = json.load(f)
        
        with open(behavior_file) as f:
            behavior_data = json.load(f)
        
        # Load frames
        frames_path = Path(f'data/keyframes/{scene_id}/{cam_id}')
        frame_files = sorted(frames_path.glob('seq_*.jpg'))
        
        if not frame_files:
            print(f"No frames found in {frames_path}")
            return None
        
        # Group poses by frame
        poses_by_frame = {}
        for pose in pose_data['poses']:
            frame_idx = pose['frame']
            if frame_idx not in poses_by_frame:
                poses_by_frame[frame_idx] = []
            poses_by_frame[frame_idx].append(pose)
        
        # Create output directory
        viz_dir = Path(f'data/poses/{scene_id}/visualization')
        viz_dir.mkdir(exist_ok=True)
        
        behavioral_profiles = behavior_data.get('behavioral_profiles', {})
        
        print(f"Creating pose + behavior visualization for {scene_id}/{cam_id}...")
        
        for frame_idx, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue
            
            if frame_idx in poses_by_frame:
                for pose in poses_by_frame[frame_idx]:
                    track_id = pose['track_id']
                    bbox = pose['bbox']
                    pose_landmarks = pose['pose_landmarks']
                    
                    # Get behavior classification
                    behavior_profile = behavioral_profiles.get(str(track_id), {})
                    movement_class = behavior_profile.get('movement_classification', {})
                    behavior_type = movement_class.get('type', 'unknown')
                    confidence = movement_class.get('confidence', 0.0)
                    
                    # Choose color based on behavior
                    color = self.behavior_colors.get(behavior_type, (128, 128, 128))
                    
                    # Draw bounding box
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                    
                    # Draw pose center
                    center_x = int(pose_landmarks['center_x'] * frame.shape[1])
                    center_y = int(pose_landmarks['center_y'] * frame.shape[0])
                    cv2.circle(frame, (center_x, center_y), 8, color, -1)
                    
                    # Add behavior label
                    label = f"ID:{track_id} {behavior_type}"
                    confidence_label = f"({confidence:.2f})"
                    
                    cv2.putText(frame, label, (x, y - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, confidence_label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add frame info and legend
            self.add_legend(frame, behavioral_profiles)
            
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Save visualization frame
            output_file = viz_dir / f'{cam_id}_pose_behavior_{frame_idx:03d}.jpg'
            cv2.imwrite(str(output_file), frame)
        
        print(f"Pose + behavior visualization complete!")
        print(f"  Output directory: {viz_dir}")
        
        return viz_dir
    
    def add_legend(self, frame, behavioral_profiles):
        """Add behavior legend to frame"""
        legend_y = frame.shape[0] - 150
        legend_x = 10
        
        cv2.rectangle(frame, (legend_x-5, legend_y-25), 
                     (legend_x+200, frame.shape[0]-10), (0, 0, 0), -1)
        
        cv2.putText(frame, "Behavior Legend:", (legend_x, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset = 20
        for behavior, color in self.behavior_colors.items():
            if behavior != 'unknown':
                cv2.rectangle(frame, (legend_x, legend_y + y_offset), 
                             (legend_x + 15, legend_y + y_offset + 15), color, -1)
                cv2.putText(frame, behavior, (legend_x + 20, legend_y + y_offset + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20

def create_scene_comparison(scene_ids):
    """Compare behavioral patterns across multiple scenes"""
    comparison_data = {
        'scenes': {},
        'cross_scene_analysis': {}
    }
    
    for scene_id in scene_ids:
        behavior_files = list(Path(f'data/behavior/{scene_id}').glob('*_behavior.json'))
        
        scene_behaviors = []
        for behavior_file in behavior_files:
            with open(behavior_file) as f:
                behavior_data = json.load(f)
                scene_behaviors.append(behavior_data)
        
        # Aggregate scene statistics
        all_profiles = {}
        for data in scene_behaviors:
            all_profiles.update(data.get('behavioral_profiles', {}))
        
        movement_types = [profile['movement_classification']['type'] 
                         for profile in all_profiles.values()]
        
        from collections import Counter
        type_distribution = Counter(movement_types)
        
        comparison_data['scenes'][scene_id] = {
            'total_persons': len(all_profiles),
            'movement_distribution': dict(type_distribution),
            'cameras': len(scene_behaviors)
        }
    
    # Cross-scene analysis
    all_movement_types = []
    for scene_data in comparison_data['scenes'].values():
        for movement_type, count in scene_data['movement_distribution'].items():
            all_movement_types.extend([movement_type] * count)
    
    from collections import Counter
    overall_distribution = Counter(all_movement_types)
    
    comparison_data['cross_scene_analysis'] = {
        'total_scenes': len(scene_ids),
        'overall_movement_distribution': dict(overall_distribution),
        'most_common_behavior': overall_distribution.most_common(1)[0][0] if overall_distribution else 'none'
    }
    
    # Save comparison
    output_file = Path('data/behavior/scene_comparison.json')
    with open(output_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print("Scene comparison analysis:")
    for scene_id, data in comparison_data['scenes'].items():
        print(f"  {scene_id}: {data['total_persons']} persons, {data['cameras']} cameras")
        print(f"    Behaviors: {data['movement_distribution']}")
    
    print(f"\nOverall analysis: {comparison_data['cross_scene_analysis']}")
    
    return comparison_data

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        scene_id = sys.argv[1]
        cam_id = sys.argv[2]
        
        visualizer = PoseVisualizer()
        visualizer.visualize_poses_and_behavior(scene_id, cam_id)
        
    elif len(sys.argv) == 2 and sys.argv[1] == "compare":
        # Compare all available scenes
        available_scenes = [d.name for d in Path('data/behavior').iterdir() if d.is_dir()]
        if available_scenes:
            create_scene_comparison(available_scenes)
        else:
            print("No scenes found for comparison")
    else:
        print("Usage:")
        print("  python cv/pose_visualization.py scene_001 cam_001  # Visualize specific scene")
        print("  python cv/pose_visualization.py compare            # Compare all scenes")
