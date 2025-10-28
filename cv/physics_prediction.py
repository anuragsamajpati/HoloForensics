import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path
import logging
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrajectoryPoint:
    """Single point in a trajectory"""
    x: float
    y: float
    timestamp: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    confidence: float = 1.0

@dataclass
class ObjectTrajectory:
    """Complete trajectory for a tracked object"""
    object_id: int
    points: List[TrajectoryPoint]
    object_type: str = "person"
    start_frame: int = 0
    end_frame: int = 0

@dataclass
class SocialForceParams:
    """Parameters for social force model"""
    desired_speed: float = 1.3  # m/s typical walking speed
    relaxation_time: float = 0.5  # seconds
    personal_space: float = 0.8  # meters
    interaction_strength: float = 2000.0  # N
    wall_repulsion: float = 10.0  # N
    noise_variance: float = 0.1

class KalmanTrajectoryPredictor:
    """Kalman filter-based trajectory prediction"""
    
    def __init__(self, dt: float = 1/30.0):  # 30 FPS default
        self.dt = dt
        self.filters = {}  # Object ID -> KalmanFilter
        
    def create_filter(self, initial_state: np.ndarray) -> KalmanFilter:
        """Create Kalman filter for constant velocity model"""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement function (observe position only)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise
        kf.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=0.1, block_size=2)
        
        # Measurement noise
        kf.R = np.eye(2) * 0.5
        
        # Initial state covariance
        kf.P *= 100
        
        # Initial state [x, y, vx, vy]
        kf.x = initial_state
        
        return kf
    
    def update_trajectory(self, object_id: int, position: np.ndarray):
        """Update trajectory with new observation"""
        if object_id not in self.filters:
            # Initialize filter with zero velocity
            initial_state = np.array([position[0], position[1], 0.0, 0.0])
            self.filters[object_id] = self.create_filter(initial_state)
        
        kf = self.filters[object_id]
        kf.predict()
        kf.update(position)
    
    def predict_trajectory(self, object_id: int, steps: int = 30) -> List[np.ndarray]:
        """Predict future trajectory points"""
        if object_id not in self.filters:
            return []
        
        kf = self.filters[object_id]
        predictions = []
        
        # Create a copy for prediction without affecting the main filter
        temp_kf = KalmanFilter(dim_x=4, dim_z=2)
        temp_kf.F = kf.F.copy()
        temp_kf.H = kf.H.copy()
        temp_kf.Q = kf.Q.copy()
        temp_kf.R = kf.R.copy()
        temp_kf.P = kf.P.copy()
        temp_kf.x = kf.x.copy()
        
        for _ in range(steps):
            temp_kf.predict()
            predictions.append(temp_kf.x[:2].copy())  # Return position only
        
        return predictions

class SocialForceModel:
    """Social force model for pedestrian dynamics"""
    
    def __init__(self, params: SocialForceParams = None):
        self.params = params or SocialForceParams()
        self.scene_boundaries = None
        self.obstacles = []
    
    def set_scene_boundaries(self, width: float, height: float):
        """Set scene boundaries for wall repulsion"""
        self.scene_boundaries = (width, height)
    
    def add_obstacle(self, center: Tuple[float, float], radius: float):
        """Add circular obstacle"""
        self.obstacles.append({'center': center, 'radius': radius})
    
    def compute_desired_force(self, position: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Compute force toward desired goal"""
        direction = goal - position
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return np.zeros(2)
        
        desired_velocity = (direction / distance) * self.params.desired_speed
        current_velocity = np.zeros(2)  # Simplified - would use actual velocity
        
        force = (desired_velocity - current_velocity) / self.params.relaxation_time
        return force
    
    def compute_social_force(self, position: np.ndarray, other_positions: List[np.ndarray]) -> np.ndarray:
        """Compute repulsive force from other people"""
        total_force = np.zeros(2)
        
        for other_pos in other_positions:
            diff = position - other_pos
            distance = np.linalg.norm(diff)
            
            if distance < 1e-6 or distance > 5.0:  # Ignore very close or far people
                continue
            
            # Exponential repulsion
            force_magnitude = self.params.interaction_strength * np.exp(
                (self.params.personal_space - distance) / self.params.personal_space
            )
            
            force_direction = diff / distance
            total_force += force_magnitude * force_direction
        
        return total_force
    
    def compute_wall_force(self, position: np.ndarray) -> np.ndarray:
        """Compute repulsive force from walls"""
        if self.scene_boundaries is None:
            return np.zeros(2)
        
        width, height = self.scene_boundaries
        force = np.zeros(2)
        
        # Left wall
        if position[0] < self.params.personal_space:
            force[0] += self.params.wall_repulsion / (position[0] + 0.1)
        
        # Right wall
        if position[0] > width - self.params.personal_space:
            force[0] -= self.params.wall_repulsion / (width - position[0] + 0.1)
        
        # Top wall
        if position[1] < self.params.personal_space:
            force[1] += self.params.wall_repulsion / (position[1] + 0.1)
        
        # Bottom wall
        if position[1] > height - self.params.personal_space:
            force[1] -= self.params.wall_repulsion / (height - position[1] + 0.1)
        
        return force
    
    def compute_obstacle_force(self, position: np.ndarray) -> np.ndarray:
        """Compute repulsive force from obstacles"""
        total_force = np.zeros(2)
        
        for obstacle in self.obstacles:
            center = np.array(obstacle['center'])
            radius = obstacle['radius']
            
            diff = position - center
            distance = np.linalg.norm(diff)
            
            if distance < radius + self.params.personal_space:
                force_magnitude = self.params.wall_repulsion * np.exp(
                    (radius + self.params.personal_space - distance) / self.params.personal_space
                )
                
                if distance > 1e-6:
                    force_direction = diff / distance
                    total_force += force_magnitude * force_direction
        
        return total_force
    
    def predict_next_position(self, current_pos: np.ndarray, goal: np.ndarray, 
                            other_positions: List[np.ndarray], dt: float = 1/30.0) -> np.ndarray:
        """Predict next position using social force model"""
        # Compute all forces
        desired_force = self.compute_desired_force(current_pos, goal)
        social_force = self.compute_social_force(current_pos, other_positions)
        wall_force = self.compute_wall_force(current_pos)
        obstacle_force = self.compute_obstacle_force(current_pos)
        
        # Add noise
        noise = np.random.normal(0, self.params.noise_variance, 2)
        
        # Total force
        total_force = desired_force + social_force + wall_force + obstacle_force + noise
        
        # Simple Euler integration (could use more sophisticated methods)
        acceleration = total_force  # Assuming unit mass
        velocity = acceleration * dt  # Simplified - should integrate properly
        next_position = current_pos + velocity * dt
        
        return next_position

class PhysicsInformedPredictor:
    """Combined physics-informed prediction system"""
    
    def __init__(self, scene_width: float = 10.0, scene_height: float = 10.0):
        self.kalman_predictor = KalmanTrajectoryPredictor()
        self.social_force_model = SocialForceModel()
        self.social_force_model.set_scene_boundaries(scene_width, scene_height)
        
        self.trajectories = {}  # Object ID -> ObjectTrajectory
        self.goals = {}  # Object ID -> goal position
        
    def add_trajectory(self, trajectory: ObjectTrajectory):
        """Add trajectory for prediction"""
        self.trajectories[trajectory.object_id] = trajectory
        
        # Update Kalman filter with trajectory points
        for point in trajectory.points:
            position = np.array([point.x, point.y])
            self.kalman_predictor.update_trajectory(trajectory.object_id, position)
    
    def set_goal(self, object_id: int, goal: np.ndarray):
        """Set goal position for object"""
        self.goals[object_id] = goal
    
    def predict_hybrid_trajectory(self, object_id: int, steps: int = 30, 
                                fusion_weight: float = 0.5) -> List[np.ndarray]:
        """Predict trajectory using hybrid Kalman + Social Force approach"""
        if object_id not in self.trajectories:
            return []
        
        # Get Kalman predictions
        kalman_predictions = self.kalman_predictor.predict_trajectory(object_id, steps)
        
        if not kalman_predictions:
            return []
        
        # Get current position
        trajectory = self.trajectories[object_id]
        if not trajectory.points:
            return kalman_predictions
        
        current_pos = np.array([trajectory.points[-1].x, trajectory.points[-1].y])
        goal = self.goals.get(object_id, kalman_predictions[-1])  # Use final Kalman prediction as goal
        
        # Get other object positions for social forces
        other_positions = []
        for other_id, other_traj in self.trajectories.items():
            if other_id != object_id and other_traj.points:
                other_pos = np.array([other_traj.points[-1].x, other_traj.points[-1].y])
                other_positions.append(other_pos)
        
        # Generate social force predictions
        social_predictions = []
        current_social_pos = current_pos.copy()
        
        for step in range(steps):
            next_pos = self.social_force_model.predict_next_position(
                current_social_pos, goal, other_positions
            )
            social_predictions.append(next_pos)
            current_social_pos = next_pos
        
        # Fuse predictions
        hybrid_predictions = []
        for i in range(min(len(kalman_predictions), len(social_predictions))):
            fused_pos = (
                fusion_weight * kalman_predictions[i] + 
                (1 - fusion_weight) * social_predictions[i]
            )
            hybrid_predictions.append(fused_pos)
        
        return hybrid_predictions
    
    def evaluate_prediction_accuracy(self, object_id: int, ground_truth: List[np.ndarray], 
                                   predictions: List[np.ndarray]) -> Dict[str, float]:
        """Evaluate prediction accuracy"""
        if not predictions or not ground_truth:
            return {'mse': float('inf'), 'mae': float('inf')}
        
        min_length = min(len(predictions), len(ground_truth))
        pred_array = np.array(predictions[:min_length])
        truth_array = np.array(ground_truth[:min_length])
        
        # Mean Squared Error
        mse = np.mean(np.sum((pred_array - truth_array) ** 2, axis=1))
        
        # Mean Absolute Error
        mae = np.mean(np.sum(np.abs(pred_array - truth_array), axis=1))
        
        # Average Displacement Error
        ade = np.mean(np.sqrt(np.sum((pred_array - truth_array) ** 2, axis=1)))
        
        # Final Displacement Error
        fde = np.sqrt(np.sum((pred_array[-1] - truth_array[-1]) ** 2))
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'ade': float(ade),
            'fde': float(fde)
        }
    
    def visualize_predictions(self, object_id: int, predictions: List[np.ndarray], 
                            save_path: str = None):
        """Visualize trajectory predictions"""
        if object_id not in self.trajectories:
            return
        
        trajectory = self.trajectories[object_id]
        
        # Extract historical points
        hist_x = [p.x for p in trajectory.points]
        hist_y = [p.y for p in trajectory.points]
        
        # Extract predictions
        pred_x = [p[0] for p in predictions]
        pred_y = [p[1] for p in predictions]
        
        plt.figure(figsize=(12, 8))
        
        # Plot historical trajectory
        plt.plot(hist_x, hist_y, 'b-o', label='Historical Trajectory', linewidth=2, markersize=4)
        
        # Plot predictions
        if predictions:
            plt.plot([hist_x[-1]] + pred_x, [hist_y[-1]] + pred_y, 
                    'r--s', label='Predicted Trajectory', linewidth=2, markersize=3)
        
        # Plot goal if available
        if object_id in self.goals:
            goal = self.goals[object_id]
            plt.plot(goal[0], goal[1], 'g*', markersize=15, label='Goal')
        
        # Plot scene boundaries
        if self.social_force_model.scene_boundaries:
            width, height = self.social_force_model.scene_boundaries
            plt.xlim(0, width)
            plt.ylim(0, height)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axhline(y=height, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=width, color='k', linestyle='-', alpha=0.3)
        
        # Plot obstacles
        for obstacle in self.social_force_model.obstacles:
            circle = plt.Circle(obstacle['center'], obstacle['radius'], 
                              color='gray', alpha=0.5, label='Obstacle')
            plt.gca().add_patch(circle)
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title(f'Physics-Informed Trajectory Prediction - Object {object_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction visualization saved to {save_path}")
        
        plt.show()

def create_sample_trajectory(object_id: int, start_pos: Tuple[float, float], 
                           end_pos: Tuple[float, float], num_points: int = 50) -> ObjectTrajectory:
    """Create sample trajectory for testing"""
    points = []
    
    for i in range(num_points):
        t = i / (num_points - 1)
        x = start_pos[0] + t * (end_pos[0] - start_pos[0])
        y = start_pos[1] + t * (end_pos[1] - start_pos[1])
        
        # Add some noise
        x += np.random.normal(0, 0.1)
        y += np.random.normal(0, 0.1)
        
        point = TrajectoryPoint(
            x=x, y=y, timestamp=i * (1/30.0),  # 30 FPS
            velocity_x=(end_pos[0] - start_pos[0]) / (num_points * (1/30.0)),
            velocity_y=(end_pos[1] - start_pos[1]) / (num_points * (1/30.0))
        )
        points.append(point)
    
    return ObjectTrajectory(
        object_id=object_id,
        points=points,
        object_type="person",
        start_frame=0,
        end_frame=num_points-1
    )

def main():
    """Demo of physics-informed prediction system"""
    print("🔬 Physics-Informed Trajectory Prediction Demo")
    print("=" * 50)
    
    # Create predictor
    predictor = PhysicsInformedPredictor(scene_width=10.0, scene_height=8.0)
    
    # Add obstacles
    predictor.social_force_model.add_obstacle((5.0, 4.0), 1.0)
    predictor.social_force_model.add_obstacle((7.0, 2.0), 0.5)
    
    # Create sample trajectories
    traj1 = create_sample_trajectory(1, (1.0, 1.0), (8.0, 6.0), 30)
    traj2 = create_sample_trajectory(2, (2.0, 7.0), (9.0, 2.0), 25)
    
    # Add trajectories
    predictor.add_trajectory(traj1)
    predictor.add_trajectory(traj2)
    
    # Set goals
    predictor.set_goal(1, np.array([9.0, 7.0]))
    predictor.set_goal(2, np.array([9.0, 1.0]))
    
    # Generate predictions
    print("🎯 Generating trajectory predictions...")
    
    predictions1 = predictor.predict_hybrid_trajectory(1, steps=20, fusion_weight=0.6)
    predictions2 = predictor.predict_hybrid_trajectory(2, steps=20, fusion_weight=0.4)
    
    print(f"✅ Generated {len(predictions1)} predictions for Object 1")
    print(f"✅ Generated {len(predictions2)} predictions for Object 2")
    
    # Visualize predictions
    print("📊 Creating visualizations...")
    predictor.visualize_predictions(1, predictions1)
    predictor.visualize_predictions(2, predictions2)
    
    # Evaluate with synthetic ground truth
    ground_truth1 = [np.array([8.5 + i*0.1, 6.5 + i*0.05]) for i in range(20)]
    metrics1 = predictor.evaluate_prediction_accuracy(1, ground_truth1, predictions1)
    
    print(f"\n📈 Prediction Metrics for Object 1:")
    print(f"   ADE: {metrics1['ade']:.3f} meters")
    print(f"   FDE: {metrics1['fde']:.3f} meters")
    print(f"   MSE: {metrics1['mse']:.3f}")
    
    print("\n🎉 Physics-informed prediction demo completed!")

if __name__ == "__main__":
    main()
