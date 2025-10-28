import json
import numpy as np
from pathlib import Path
import sys

class BehavioralAnalyzer:
    def __init__(self):
        self.movement_thresholds = {
            'stationary': 0.005,
            'walking': 0.02,
            'running': 0.05
        }
    
    def analyze_movement_patterns(self, pose_data):
        """Analyze movement patterns for behavioral classification"""
        gait_analysis = pose_data.get('gait_analysis', {})
        poses = pose_data.get('poses', [])
        
        behavioral_profiles = {}
        
        for track_id, gait_features in gait_analysis.items():
            if not gait_features or 'movement_pattern' not in gait_features:
                continue
                
            movement_pattern = gait_features['movement_pattern']
            
            # Classify movement behavior
            behavior = self.classify_movement_behavior(movement_pattern)
            
            # Extract temporal patterns
            temporal_analysis = self.analyze_temporal_patterns(poses, int(track_id))
            
            # Calculate consistency metrics
            consistency = self.calculate_movement_consistency(movement_pattern)
            
            behavioral_profiles[track_id] = {
                'movement_classification': behavior,
                'temporal_patterns': temporal_analysis,
                'consistency_metrics': consistency,
                'summary': self.generate_behavior_summary(behavior, temporal_analysis, consistency)
            }
        
        return behavioral_profiles
    
    def classify_movement_behavior(self, movement_pattern):
        """Classify movement into behavioral categories"""
        if not movement_pattern:
            return {'type': 'unknown', 'confidence': 0.0}
        
        avg_movement = np.mean(movement_pattern)
        movement_variance = np.var(movement_pattern)
        
        # Classify based on movement magnitude
        if avg_movement < self.movement_thresholds['stationary']:
            movement_type = 'stationary'
            confidence = 1.0 - (avg_movement / self.movement_thresholds['stationary'])
        elif avg_movement < self.movement_thresholds['walking']:
            movement_type = 'walking'
            confidence = 0.8
        elif avg_movement < self.movement_thresholds['running']:
            movement_type = 'fast_walking'
            confidence = 0.7
        else:
            movement_type = 'running'
            confidence = 0.6
        
        # Adjust confidence based on consistency
        if movement_variance > avg_movement * 2:
            confidence *= 0.7  # Lower confidence for erratic movement
        
        return {
            'type': movement_type,
            'confidence': min(confidence, 1.0),
            'avg_speed': avg_movement,
            'speed_variance': movement_variance
        }
    
    def analyze_temporal_patterns(self, poses, track_id):
        """Analyze temporal movement patterns"""
        track_poses = [p for p in poses if p['track_id'] == track_id]
        
        if len(track_poses) < 3:
            return {'duration': 0, 'activity_periods': []}
        
        # Sort by frame
        track_poses.sort(key=lambda x: x['frame'])
        
        # Calculate activity periods
        activity_periods = []
        current_period = {'start': track_poses[0]['frame'], 'end': track_poses[0]['frame']}
        
        for i in range(1, len(track_poses)):
            frame_gap = track_poses[i]['frame'] - track_poses[i-1]['frame']
            
            if frame_gap <= 2:  # Continuous activity
                current_period['end'] = track_poses[i]['frame']
            else:  # Gap in activity
                activity_periods.append(current_period)
                current_period = {'start': track_poses[i]['frame'], 'end': track_poses[i]['frame']}
        
        activity_periods.append(current_period)
        
        return {
            'total_duration': track_poses[-1]['frame'] - track_poses[0]['frame'],
            'active_frames': len(track_poses),
            'activity_periods': activity_periods,
            'avg_period_length': np.mean([p['end'] - p['start'] + 1 for p in activity_periods])
        }
    
    def calculate_movement_consistency(self, movement_pattern):
        """Calculate consistency metrics for movement"""
        if len(movement_pattern) < 2:
            return {'consistency_score': 0, 'stability': 'unknown'}
        
        # Calculate coefficient of variation
        mean_movement = np.mean(movement_pattern)
        std_movement = np.std(movement_pattern)
        
        if mean_movement > 0:
            cv = std_movement / mean_movement
            consistency_score = max(0, 1 - cv)  # Higher score = more consistent
        else:
            consistency_score = 1.0 if std_movement == 0 else 0.0
        
        # Classify stability
        if consistency_score > 0.8:
            stability = 'very_stable'
        elif consistency_score > 0.6:
            stability = 'stable'
        elif consistency_score > 0.4:
            stability = 'moderate'
        elif consistency_score > 0.2:
            stability = 'unstable'
        else:
            stability = 'erratic'
        
        return {
            'consistency_score': consistency_score,
            'stability': stability,
            'coefficient_of_variation': cv if mean_movement > 0 else float('inf')
        }
    
    def generate_behavior_summary(self, behavior, temporal, consistency):
        """Generate human-readable behavior summary"""
        movement_type = behavior['type']
        confidence = behavior['confidence']
        stability = consistency['stability']
        duration = temporal.get('total_duration', 0)
        
        summary = f"Person exhibits {movement_type} behavior "
        summary += f"(confidence: {confidence:.2f}) "
        summary += f"with {stability} movement patterns "
        summary += f"over {duration} frames."
        
        if temporal.get('activity_periods'):
            num_periods = len(temporal['activity_periods'])
            avg_period = temporal.get('avg_period_length', 0)
            summary += f" Active in {num_periods} periods, "
            summary += f"average {avg_period:.1f} frames per period."
        
        return summary

def process_behavioral_analysis(scene_id, cam_id):
    """Process behavioral analysis for pose data"""
    print(f"Starting behavioral analysis for {scene_id}/{cam_id}")
    
    # Load pose data
    pose_file = Path(f'data/poses/{scene_id}/{cam_id}_poses.json')
    if not pose_file.exists():
        print(f"Pose file not found: {pose_file}")
        return None
    
    with open(pose_file) as f:
        pose_data = json.load(f)
    
    analyzer = BehavioralAnalyzer()
    behavioral_profiles = analyzer.analyze_movement_patterns(pose_data)
    
    # Create comprehensive analysis
    analysis_summary = {
        'scene_id': scene_id,
        'camera_id': cam_id,
        'analysis_type': 'behavioral_patterns',
        'total_tracks_analyzed': len(behavioral_profiles),
        'behavioral_profiles': behavioral_profiles,
        'scene_summary': generate_scene_summary(behavioral_profiles)
    }
    
    # Save behavioral analysis
    output_path = Path(f'data/behavior/{scene_id}')
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f'{cam_id}_behavior.json'
    with open(output_file, 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    
    print(f"Behavioral analysis complete!")
    print(f"  Tracks analyzed: {len(behavioral_profiles)}")
    print(f"  Output saved: {output_file}")
    
    # Print summary
    for track_id, profile in behavioral_profiles.items():
        print(f"  Track {track_id}: {profile['summary']}")
    
    return analysis_summary

def generate_scene_summary(behavioral_profiles):
    """Generate overall scene behavioral summary"""
    if not behavioral_profiles:
        return "No behavioral data available"
    
    movement_types = [profile['movement_classification']['type'] 
                     for profile in behavioral_profiles.values()]
    
    from collections import Counter
    type_counts = Counter(movement_types)
    
    summary = {
        'total_persons': len(behavioral_profiles),
        'movement_distribution': dict(type_counts),
        'dominant_behavior': type_counts.most_common(1)[0][0] if type_counts else 'unknown'
    }
    
    return summary

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        scene_id = sys.argv[1]
        cam_id = sys.argv[2]
        process_behavioral_analysis(scene_id, cam_id)
    else:
        print("Usage: python cv/behavioral_analysis.py scene_001 cam_001")
