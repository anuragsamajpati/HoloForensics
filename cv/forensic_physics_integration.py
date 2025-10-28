import numpy as np
import cv2
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
import time
from datetime import datetime
import pickle

from physics_prediction import (
    PhysicsInformedPredictor, ObjectTrajectory, TrajectoryPoint, 
    SocialForceParams, KalmanTrajectoryPredictor, SocialForceModel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ForensicPredictionConfig:
    """Configuration for forensic trajectory prediction"""
    prediction_horizon: int = 60  # frames to predict ahead
    confidence_threshold: float = 0.7  # minimum confidence for predictions
    kalman_weight: float = 0.6  # weight for Kalman vs social force fusion
    scene_calibration: bool = True  # use scene-specific calibration
    evidence_preservation: bool = True  # maintain chain of custody
    uncertainty_quantification: bool = True  # compute prediction uncertainty

@dataclass
class PredictionResult:
    """Result of trajectory prediction analysis"""
    object_id: int
    predicted_positions: List[Tuple[float, float]]
    confidence_scores: List[float]
    prediction_timestamps: List[float]
    method_used: str
    accuracy_metrics: Dict[str, float]
    uncertainty_bounds: List[Tuple[float, float, float, float]]  # x_min, x_max, y_min, y_max

@dataclass
class ForensicScenario:
    """Forensic scenario definition"""
    scenario_id: str
    scene_dimensions: Tuple[float, float]  # width, height in meters
    obstacles: List[Dict[str, Any]]
    entry_points: List[Tuple[float, float]]
    exit_points: List[Tuple[float, float]]
    high_interest_zones: List[Dict[str, Any]]
    timestamp: float

class ForensicTrajectoryAnalyzer:
    """Forensic-specific trajectory analysis and prediction"""
    
    def __init__(self, config: ForensicPredictionConfig = None):
        self.config = config or ForensicPredictionConfig()
        self.predictor = None
        self.scenario = None
        self.analysis_results = {}
        
    def load_forensic_scenario(self, scenario_path: str):
        """Load forensic scenario configuration"""
        scenario_path = Path(scenario_path)
        
        if scenario_path.suffix == '.json':
            with open(scenario_path, 'r') as f:
                scenario_data = json.load(f)
        else:
            raise ValueError(f"Unsupported scenario format: {scenario_path.suffix}")
        
        self.scenario = ForensicScenario(**scenario_data)
        
        # Initialize predictor with scenario parameters
        width, height = self.scenario.scene_dimensions
        self.predictor = PhysicsInformedPredictor(width, height)
        
        # Add obstacles from scenario
        for obstacle in self.scenario.obstacles:
            self.predictor.social_force_model.add_obstacle(
                tuple(obstacle['center']), obstacle['radius']
            )
        
        logger.info(f"Loaded forensic scenario: {self.scenario.scenario_id}")
    
    def calibrate_physics_model(self, training_trajectories: List[ObjectTrajectory]):
        """Calibrate physics model parameters using training data"""
        if not self.predictor:
            raise ValueError("Must load scenario before calibration")
        
        logger.info("🔧 Calibrating physics model parameters...")
        
        # Extract movement patterns from training data
        speeds = []
        accelerations = []
        interaction_distances = []
        
        for trajectory in training_trajectories:
            if len(trajectory.points) < 3:
                continue
            
            # Calculate speeds and accelerations
            for i in range(1, len(trajectory.points)):
                p1, p2 = trajectory.points[i-1], trajectory.points[i]
                dt = p2.timestamp - p1.timestamp
                
                if dt > 0:
                    dx = p2.x - p1.x
                    dy = p2.y - p1.y
                    speed = np.sqrt(dx**2 + dy**2) / dt
                    speeds.append(speed)
                    
                    if i < len(trajectory.points) - 1:
                        p3 = trajectory.points[i+1]
                        dt2 = p3.timestamp - p2.timestamp
                        if dt2 > 0:
                            dx2 = p3.x - p2.x
                            dy2 = p3.y - p2.y
                            speed2 = np.sqrt(dx2**2 + dy2**2) / dt2
                            acceleration = (speed2 - speed) / dt
                            accelerations.append(acceleration)
        
        # Update social force parameters based on observed data
        if speeds:
            avg_speed = np.mean(speeds)
            self.predictor.social_force_model.params.desired_speed = avg_speed
            logger.info(f"   Calibrated desired speed: {avg_speed:.2f} m/s")
        
        # Calibrate interaction parameters (simplified approach)
        if len(training_trajectories) > 1:
            # Analyze interaction patterns between trajectories
            self._calibrate_social_interactions(training_trajectories)
    
    def _calibrate_social_interactions(self, trajectories: List[ObjectTrajectory]):
        """Calibrate social interaction parameters"""
        interaction_events = []
        
        # Find temporal overlaps between trajectories
        for i, traj1 in enumerate(trajectories):
            for j, traj2 in enumerate(trajectories[i+1:], i+1):
                # Find overlapping time periods
                t1_start = traj1.points[0].timestamp if traj1.points else 0
                t1_end = traj1.points[-1].timestamp if traj1.points else 0
                t2_start = traj2.points[0].timestamp if traj2.points else 0
                t2_end = traj2.points[-1].timestamp if traj2.points else 0
                
                overlap_start = max(t1_start, t2_start)
                overlap_end = min(t1_end, t2_end)
                
                if overlap_start < overlap_end:
                    # Analyze interaction during overlap
                    min_distance = self._analyze_trajectory_interaction(
                        traj1, traj2, overlap_start, overlap_end
                    )
                    if min_distance is not None:
                        interaction_events.append(min_distance)
        
        if interaction_events:
            avg_min_distance = np.mean(interaction_events)
            # Update personal space parameter
            self.predictor.social_force_model.params.personal_space = max(0.5, avg_min_distance * 0.8)
            logger.info(f"   Calibrated personal space: {self.predictor.social_force_model.params.personal_space:.2f} m")
    
    def _analyze_trajectory_interaction(self, traj1: ObjectTrajectory, traj2: ObjectTrajectory,
                                      start_time: float, end_time: float) -> Optional[float]:
        """Analyze interaction between two trajectories in a time window"""
        min_distance = float('inf')
        
        # Sample points from both trajectories in the time window
        points1 = [p for p in traj1.points if start_time <= p.timestamp <= end_time]
        points2 = [p for p in traj2.points if start_time <= p.timestamp <= end_time]
        
        if not points1 or not points2:
            return None
        
        # Find minimum distance between trajectories
        for p1 in points1:
            for p2 in points2:
                if abs(p1.timestamp - p2.timestamp) < 0.5:  # Within 0.5 seconds
                    distance = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                    min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else None
    
    def predict_suspect_trajectory(self, partial_trajectory: ObjectTrajectory,
                                 investigation_context: Dict[str, Any]) -> PredictionResult:
        """Predict suspect trajectory for forensic investigation"""
        if not self.predictor:
            raise ValueError("Must load scenario before prediction")
        
        logger.info(f"🎯 Predicting trajectory for suspect {partial_trajectory.object_id}")
        
        # Add trajectory to predictor
        self.predictor.add_trajectory(partial_trajectory)
        
        # Determine likely goals based on investigation context
        goals = self._infer_suspect_goals(partial_trajectory, investigation_context)
        
        # Generate predictions for each possible goal
        all_predictions = []
        all_confidences = []
        
        for goal, goal_confidence in goals:
            self.predictor.set_goal(partial_trajectory.object_id, np.array(goal))
            
            predictions = self.predictor.predict_hybrid_trajectory(
                partial_trajectory.object_id,
                steps=self.config.prediction_horizon,
                fusion_weight=self.config.kalman_weight
            )
            
            if predictions:
                # Weight predictions by goal confidence
                weighted_predictions = [(p, goal_confidence) for p in predictions]
                all_predictions.extend(weighted_predictions)
                all_confidences.extend([goal_confidence] * len(predictions))
        
        if not all_predictions:
            logger.warning("No predictions generated")
            return self._create_empty_result(partial_trajectory.object_id)
        
        # Aggregate predictions (simplified - could use more sophisticated fusion)
        final_predictions = []
        final_confidences = []
        uncertainty_bounds = []
        
        for i in range(min(len(p[0]) for p in all_predictions)):
            positions = [p[0][i] for p in all_predictions if i < len(p[0])]
            confidences = [p[1] for p in all_predictions if i < len(p[0])]
            
            if positions:
                # Weighted average of positions
                weights = np.array(confidences)
                positions_array = np.array(positions)
                
                avg_position = np.average(positions_array, axis=0, weights=weights)
                avg_confidence = np.mean(confidences)
                
                final_predictions.append(tuple(avg_position))
                final_confidences.append(avg_confidence)
                
                # Compute uncertainty bounds
                if self.config.uncertainty_quantification:
                    std_x = np.std([p[0] for p in positions])
                    std_y = np.std([p[1] for p in positions])
                    
                    bounds = (
                        avg_position[0] - 2*std_x,  # x_min
                        avg_position[0] + 2*std_x,  # x_max
                        avg_position[1] - 2*std_y,  # y_min
                        avg_position[1] + 2*std_y   # y_max
                    )
                    uncertainty_bounds.append(bounds)
                else:
                    uncertainty_bounds.append((0, 0, 0, 0))
        
        # Generate timestamps
        start_time = partial_trajectory.points[-1].timestamp if partial_trajectory.points else 0
        timestamps = [start_time + i * (1/30.0) for i in range(len(final_predictions))]
        
        # Compute accuracy metrics (would need ground truth in real scenario)
        accuracy_metrics = self._compute_prediction_metrics(partial_trajectory, final_predictions)
        
        result = PredictionResult(
            object_id=partial_trajectory.object_id,
            predicted_positions=final_predictions,
            confidence_scores=final_confidences,
            prediction_timestamps=timestamps,
            method_used="hybrid_kalman_social_force",
            accuracy_metrics=accuracy_metrics,
            uncertainty_bounds=uncertainty_bounds
        )
        
        logger.info(f"✅ Generated {len(final_predictions)} predictions with avg confidence {np.mean(final_confidences):.3f}")
        
        return result
    
    def _infer_suspect_goals(self, trajectory: ObjectTrajectory, 
                           context: Dict[str, Any]) -> List[Tuple[Tuple[float, float], float]]:
        """Infer likely goals for suspect based on trajectory and context"""
        goals = []
        
        # Use exit points as potential goals
        for exit_point in self.scenario.exit_points:
            # Simple heuristic: closer exit points are more likely
            if trajectory.points:
                last_pos = (trajectory.points[-1].x, trajectory.points[-1].y)
                distance = np.sqrt((exit_point[0] - last_pos[0])**2 + (exit_point[1] - last_pos[1])**2)
                confidence = max(0.1, 1.0 - distance / 10.0)  # Inverse distance weighting
                goals.append((exit_point, confidence))
        
        # Consider high interest zones
        for zone in self.scenario.high_interest_zones:
            center = tuple(zone['center'])
            zone_type = zone.get('type', 'unknown')
            
            # Weight based on zone type and investigation context
            confidence = 0.5
            if zone_type in context.get('relevant_zones', []):
                confidence = 0.8
            
            goals.append((center, confidence))
        
        # If no specific goals, use scene boundaries
        if not goals:
            width, height = self.scenario.scene_dimensions
            boundary_goals = [
                ((width, height/2), 0.3),  # Right edge
                ((0, height/2), 0.3),      # Left edge
                ((width/2, height), 0.3),  # Top edge
                ((width/2, 0), 0.3)        # Bottom edge
            ]
            goals.extend(boundary_goals)
        
        return goals
    
    def _compute_prediction_metrics(self, trajectory: ObjectTrajectory, 
                                  predictions: List[Tuple[float, float]]) -> Dict[str, float]:
        """Compute prediction quality metrics"""
        # Simplified metrics - in real scenario would compare with ground truth
        metrics = {
            'prediction_length': len(predictions),
            'trajectory_consistency': 1.0,  # Placeholder
            'physics_compliance': 1.0,      # Placeholder
            'social_realism': 1.0           # Placeholder
        }
        
        if len(predictions) > 1:
            # Compute smoothness of predicted trajectory
            distances = []
            for i in range(1, len(predictions)):
                dist = np.sqrt((predictions[i][0] - predictions[i-1][0])**2 + 
                             (predictions[i][1] - predictions[i-1][1])**2)
                distances.append(dist)
            
            if distances:
                metrics['avg_step_size'] = np.mean(distances)
                metrics['trajectory_smoothness'] = 1.0 / (1.0 + np.std(distances))
        
        return metrics
    
    def _create_empty_result(self, object_id: int) -> PredictionResult:
        """Create empty prediction result"""
        return PredictionResult(
            object_id=object_id,
            predicted_positions=[],
            confidence_scores=[],
            prediction_timestamps=[],
            method_used="none",
            accuracy_metrics={},
            uncertainty_bounds=[]
        )
    
    def generate_forensic_report(self, results: List[PredictionResult], 
                               case_id: str, output_path: str):
        """Generate comprehensive forensic prediction report"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report = {
            'case_id': case_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'scenario_id': self.scenario.scenario_id if self.scenario else 'unknown',
            'configuration': asdict(self.config),
            'results': [],
            'summary': {}
        }
        
        # Process each prediction result
        total_predictions = 0
        avg_confidence = 0
        
        for result in results:
            result_dict = asdict(result)
            report['results'].append(result_dict)
            
            total_predictions += len(result.predicted_positions)
            if result.confidence_scores:
                avg_confidence += np.mean(result.confidence_scores)
        
        # Generate summary statistics
        report['summary'] = {
            'total_objects_analyzed': len(results),
            'total_predictions_generated': total_predictions,
            'average_confidence': avg_confidence / len(results) if results else 0,
            'prediction_horizon_frames': self.config.prediction_horizon,
            'analysis_method': 'physics_informed_hybrid'
        }
        
        # Save report
        report_file = output_path / f"forensic_prediction_report_{case_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save detailed results for each object
        for result in results:
            object_file = output_path / f"object_{result.object_id}_predictions.json"
            with open(object_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)
        
        logger.info(f"📋 Forensic report generated: {report_file}")
        return report_file

def create_sample_forensic_scenario() -> str:
    """Create sample forensic scenario for testing"""
    scenario = {
        'scenario_id': 'bank_robbery_001',
        'scene_dimensions': [20.0, 15.0],  # 20m x 15m
        'obstacles': [
            {'center': [5.0, 7.5], 'radius': 1.0, 'type': 'pillar'},
            {'center': [15.0, 7.5], 'radius': 1.0, 'type': 'pillar'},
            {'center': [10.0, 3.0], 'radius': 2.0, 'type': 'counter'}
        ],
        'entry_points': [
            [0.0, 7.5],   # Main entrance
            [20.0, 2.0],  # Side entrance
            [10.0, 0.0]   # Back entrance
        ],
        'exit_points': [
            [0.0, 7.5],   # Main exit
            [20.0, 2.0],  # Side exit
            [10.0, 15.0], # Emergency exit
            [20.0, 12.0]  # Staff exit
        ],
        'high_interest_zones': [
            {'center': [8.0, 3.0], 'radius': 3.0, 'type': 'teller_area'},
            {'center': [18.0, 12.0], 'radius': 2.0, 'type': 'vault_area'},
            {'center': [2.0, 2.0], 'radius': 1.5, 'type': 'atm_area'}
        ],
        'timestamp': time.time()
    }
    
    # Save scenario
    scenario_path = Path("test_scenario.json")
    with open(scenario_path, 'w') as f:
        json.dump(scenario, f, indent=2)
    
    return str(scenario_path)

def main():
    """Demo of forensic physics integration"""
    print("🏛️ Forensic Physics-Informed Prediction Demo")
    print("=" * 50)
    
    # Create sample scenario
    scenario_path = create_sample_forensic_scenario()
    print(f"📁 Created sample scenario: {scenario_path}")
    
    # Initialize analyzer
    config = ForensicPredictionConfig(
        prediction_horizon=45,
        confidence_threshold=0.6,
        kalman_weight=0.7
    )
    
    analyzer = ForensicTrajectoryAnalyzer(config)
    analyzer.load_forensic_scenario(scenario_path)
    
    # Create sample suspect trajectory (partial)
    from physics_prediction import create_sample_trajectory
    
    suspect_trajectory = create_sample_trajectory(
        object_id=999,  # Suspect ID
        start_pos=(1.0, 8.0),  # Near main entrance
        end_pos=(8.0, 4.0),    # Moving toward teller area
        num_points=20          # Partial trajectory
    )
    
    # Investigation context
    context = {
        'crime_type': 'bank_robbery',
        'relevant_zones': ['teller_area', 'vault_area'],
        'time_pressure': 'high',
        'suspect_behavior': 'evasive'
    }
    
    # Generate prediction
    print("🎯 Generating suspect trajectory prediction...")
    result = analyzer.predict_suspect_trajectory(suspect_trajectory, context)
    
    print(f"✅ Generated {len(result.predicted_positions)} predictions")
    print(f"📊 Average confidence: {np.mean(result.confidence_scores):.3f}")
    print(f"🔬 Method used: {result.method_used}")
    
    # Generate forensic report
    print("📋 Generating forensic report...")
    report_path = analyzer.generate_forensic_report(
        [result], 
        case_id="CASE_2024_001", 
        output_path="forensic_reports"
    )
    
    print(f"✅ Report saved to: {report_path}")
    print("\n🎉 Forensic physics integration demo completed!")
    
    # Cleanup
    Path(scenario_path).unlink()

if __name__ == "__main__":
    main()
