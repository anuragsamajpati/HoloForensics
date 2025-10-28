"""
Event Detection System for Forensic Video Analysis
Detects significant events from scene graphs and object trajectories
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import networkx as nx

from scene_graph_generation import SceneGraph, SceneObject, SceneRelation, RelationType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of forensic events that can be detected"""
    # Movement Events
    PERSON_ENTERING = "person_entering"
    PERSON_EXITING = "person_exiting"
    VEHICLE_ARRIVING = "vehicle_arriving"
    VEHICLE_DEPARTING = "vehicle_departing"
    OBJECT_DROPPED = "object_dropped"
    OBJECT_PICKED_UP = "object_picked_up"
    
    # Interaction Events
    PERSON_MEETING = "person_meeting"
    PERSON_SEPARATION = "person_separation"
    HANDOFF_EVENT = "handoff_event"
    CONFRONTATION = "confrontation"
    CHASE_SEQUENCE = "chase_sequence"
    
    # Behavioral Events
    LOITERING = "loitering"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    RAPID_MOVEMENT = "rapid_movement"
    SUDDEN_STOP = "sudden_stop"
    DIRECTION_CHANGE = "direction_change"
    
    # Security Events
    PERIMETER_BREACH = "perimeter_breach"
    RESTRICTED_AREA_ACCESS = "restricted_area_access"
    UNAUTHORIZED_ENTRY = "unauthorized_entry"
    TAMPERING = "tampering"
    
    # Temporal Events
    ACTIVITY_START = "activity_start"
    ACTIVITY_END = "activity_end"
    PATTERN_ANOMALY = "pattern_anomaly"
    TIMELINE_GAP = "timeline_gap"

class EventSeverity(Enum):
    """Severity levels for detected events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DetectedEvent:
    """Represents a detected forensic event"""
    event_id: str
    event_type: EventType
    severity: EventSeverity
    confidence: float
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    involved_objects: List[str]
    location: Tuple[float, float]  # x, y coordinates
    description: str
    evidence: Dict[str, Any]
    metadata: Dict[str, Any] = None

@dataclass
class EventPattern:
    """Template for event detection patterns"""
    pattern_id: str
    event_type: EventType
    required_objects: List[str]
    spatial_constraints: Dict[str, Any]
    temporal_constraints: Dict[str, Any]
    confidence_threshold: float
    description: str

class MovementAnalyzer:
    """Analyzes object movement patterns for event detection"""
    
    def __init__(self, velocity_threshold: float = 5.0, acceleration_threshold: float = 2.0):
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.movement_history = defaultdict(deque)
        self.max_history_length = 30
    
    def analyze_movement(self, objects: List[SceneObject], frame_id: int) -> List[DetectedEvent]:
        """Analyze movement patterns and detect movement-based events"""
        events = []
        
        for obj in objects:
            # Update movement history
            self.movement_history[obj.object_id].append({
                'frame_id': frame_id,
                'timestamp': obj.timestamp,
                'center': obj.center,
                'bbox': obj.bbox
            })
            
            # Maintain history length
            if len(self.movement_history[obj.object_id]) > self.max_history_length:
                self.movement_history[obj.object_id].popleft()
            
            # Analyze movement patterns
            movement_events = self._analyze_object_movement(obj.object_id, frame_id)
            events.extend(movement_events)
        
        return events
    
    def _analyze_object_movement(self, object_id: str, frame_id: int) -> List[DetectedEvent]:
        """Analyze movement patterns for a specific object"""
        events = []
        history = self.movement_history[object_id]
        
        if len(history) < 3:
            return events
        
        # Calculate velocities and accelerations
        positions = [h['center'] for h in history]
        timestamps = [h['timestamp'] for h in history]
        
        velocities = []
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                velocity = np.sqrt(dx**2 + dy**2) / dt
                velocities.append(velocity)
        
        if len(velocities) < 2:
            return events
        
        # Detect rapid movement
        recent_velocity = np.mean(velocities[-3:])
        if recent_velocity > self.velocity_threshold * 3:
            events.append(DetectedEvent(
                event_id=f"rapid_movement_{object_id}_{frame_id}",
                event_type=EventType.RAPID_MOVEMENT,
                severity=EventSeverity.MEDIUM,
                confidence=min(1.0, recent_velocity / (self.velocity_threshold * 3)),
                start_time=history[-3]['timestamp'],
                end_time=history[-1]['timestamp'],
                start_frame=history[-3]['frame_id'],
                end_frame=frame_id,
                involved_objects=[object_id],
                location=history[-1]['center'],
                description=f"Object {object_id} moving rapidly at {recent_velocity:.2f} units/sec",
                evidence={'velocity': recent_velocity, 'threshold': self.velocity_threshold * 3}
            ))
        
        # Detect sudden stops
        if len(velocities) >= 5:
            prev_velocity = np.mean(velocities[-5:-2])
            current_velocity = np.mean(velocities[-2:])
            
            if prev_velocity > self.velocity_threshold and current_velocity < self.velocity_threshold * 0.3:
                events.append(DetectedEvent(
                    event_id=f"sudden_stop_{object_id}_{frame_id}",
                    event_type=EventType.SUDDEN_STOP,
                    severity=EventSeverity.MEDIUM,
                    confidence=0.8,
                    start_time=history[-2]['timestamp'],
                    end_time=history[-1]['timestamp'],
                    start_frame=history[-2]['frame_id'],
                    end_frame=frame_id,
                    involved_objects=[object_id],
                    location=history[-1]['center'],
                    description=f"Object {object_id} stopped suddenly",
                    evidence={'prev_velocity': prev_velocity, 'current_velocity': current_velocity}
                ))
        
        # Detect direction changes
        if len(positions) >= 4:
            direction_change = self._detect_direction_change(positions[-4:])
            if direction_change > 90:  # Significant direction change
                events.append(DetectedEvent(
                    event_id=f"direction_change_{object_id}_{frame_id}",
                    event_type=EventType.DIRECTION_CHANGE,
                    severity=EventSeverity.LOW,
                    confidence=min(1.0, direction_change / 180.0),
                    start_time=history[-2]['timestamp'],
                    end_time=history[-1]['timestamp'],
                    start_frame=history[-2]['frame_id'],
                    end_frame=frame_id,
                    involved_objects=[object_id],
                    location=history[-1]['center'],
                    description=f"Object {object_id} changed direction by {direction_change:.1f} degrees",
                    evidence={'direction_change': direction_change}
                ))
        
        return events
    
    def _detect_direction_change(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate the angle of direction change"""
        if len(positions) < 3:
            return 0.0
        
        # Calculate vectors
        v1 = np.array(positions[1]) - np.array(positions[0])
        v2 = np.array(positions[-1]) - np.array(positions[-2])
        
        # Calculate angle between vectors
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle

class InteractionAnalyzer:
    """Analyzes interactions between objects for event detection"""
    
    def __init__(self, proximity_threshold: float = 100.0):
        self.proximity_threshold = proximity_threshold
        self.interaction_history = defaultdict(list)
    
    def analyze_interactions(self, scene_graph: SceneGraph) -> List[DetectedEvent]:
        """Analyze interactions in scene graph and detect interaction events"""
        events = []
        
        # Analyze proximity-based interactions
        proximity_events = self._analyze_proximity_interactions(scene_graph)
        events.extend(proximity_events)
        
        # Analyze meeting and separation events
        meeting_events = self._analyze_meeting_separation(scene_graph)
        events.extend(meeting_events)
        
        # Analyze handoff events
        handoff_events = self._analyze_handoff_events(scene_graph)
        events.extend(handoff_events)
        
        return events
    
    def _analyze_proximity_interactions(self, scene_graph: SceneGraph) -> List[DetectedEvent]:
        """Analyze proximity-based interactions"""
        events = []
        
        # Find close proximity relations
        close_relations = [
            rel for rel in scene_graph.relations
            if rel.relation_type == RelationType.SPATIAL_NEAR and 
               rel.confidence > 0.7
        ]
        
        for relation in close_relations:
            # Check if this is a new interaction
            interaction_key = f"{relation.subject_id}_{relation.object_id}"
            
            if interaction_key not in self.interaction_history:
                self.interaction_history[interaction_key] = []
            
            self.interaction_history[interaction_key].append({
                'frame_id': scene_graph.frame_id,
                'timestamp': scene_graph.timestamp,
                'confidence': relation.confidence
            })
            
            # Detect start of interaction
            if len(self.interaction_history[interaction_key]) == 1:
                events.append(DetectedEvent(
                    event_id=f"interaction_start_{interaction_key}_{scene_graph.frame_id}",
                    event_type=EventType.PERSON_MEETING,
                    severity=EventSeverity.MEDIUM,
                    confidence=relation.confidence,
                    start_time=scene_graph.timestamp,
                    end_time=scene_graph.timestamp,
                    start_frame=scene_graph.frame_id,
                    end_frame=scene_graph.frame_id,
                    involved_objects=[relation.subject_id, relation.object_id],
                    location=self._get_interaction_location(scene_graph, relation),
                    description=f"Interaction started between {relation.subject_id} and {relation.object_id}",
                    evidence={'proximity_confidence': relation.confidence}
                ))
        
        return events
    
    def _analyze_meeting_separation(self, scene_graph: SceneGraph) -> List[DetectedEvent]:
        """Analyze meeting and separation patterns"""
        events = []
        
        # Check for ongoing interactions that have ended
        current_interactions = set()
        for relation in scene_graph.relations:
            if relation.relation_type == RelationType.SPATIAL_NEAR:
                interaction_key = f"{relation.subject_id}_{relation.object_id}"
                current_interactions.add(interaction_key)
        
        # Find interactions that have ended
        for interaction_key, history in self.interaction_history.items():
            if (interaction_key not in current_interactions and 
                len(history) > 0 and 
                scene_graph.frame_id - history[-1]['frame_id'] == 1):
                
                # Interaction just ended
                duration = history[-1]['timestamp'] - history[0]['timestamp']
                subject_id, object_id = interaction_key.split('_', 1)
                
                events.append(DetectedEvent(
                    event_id=f"interaction_end_{interaction_key}_{scene_graph.frame_id}",
                    event_type=EventType.PERSON_SEPARATION,
                    severity=EventSeverity.LOW,
                    confidence=0.8,
                    start_time=history[0]['timestamp'],
                    end_time=history[-1]['timestamp'],
                    start_frame=history[0]['frame_id'],
                    end_frame=history[-1]['frame_id'],
                    involved_objects=[subject_id, object_id],
                    location=(0, 0),  # Would need to calculate from last known positions
                    description=f"Interaction ended between {subject_id} and {object_id} after {duration:.2f} seconds",
                    evidence={'duration': duration, 'interaction_frames': len(history)}
                ))
        
        return events
    
    def _analyze_handoff_events(self, scene_graph: SceneGraph) -> List[DetectedEvent]:
        """Analyze potential handoff events"""
        events = []
        
        # Look for brief close interactions between people
        person_relations = [
            rel for rel in scene_graph.relations
            if rel.relation_type == RelationType.SPATIAL_NEAR and
               rel.confidence > 0.8
        ]
        
        for relation in person_relations:
            # Check if both objects are persons (simplified check)
            subject_obj = next((obj for obj in scene_graph.objects if obj.object_id == relation.subject_id), None)
            object_obj = next((obj for obj in scene_graph.objects if obj.object_id == relation.object_id), None)
            
            if (subject_obj and object_obj and 
                subject_obj.object_type == 'person' and 
                object_obj.object_type == 'person'):
                
                # Check interaction history for brief contact pattern
                interaction_key = f"{relation.subject_id}_{relation.object_id}"
                history = self.interaction_history.get(interaction_key, [])
                
                if len(history) >= 3 and len(history) <= 10:  # Brief interaction
                    events.append(DetectedEvent(
                        event_id=f"handoff_{interaction_key}_{scene_graph.frame_id}",
                        event_type=EventType.HANDOFF_EVENT,
                        severity=EventSeverity.HIGH,
                        confidence=0.7,
                        start_time=history[0]['timestamp'],
                        end_time=scene_graph.timestamp,
                        start_frame=history[0]['frame_id'],
                        end_frame=scene_graph.frame_id,
                        involved_objects=[relation.subject_id, relation.object_id],
                        location=self._get_interaction_location(scene_graph, relation),
                        description=f"Potential handoff event between {relation.subject_id} and {relation.object_id}",
                        evidence={'interaction_duration': len(history), 'proximity_confidence': relation.confidence}
                    ))
        
        return events
    
    def _get_interaction_location(self, scene_graph: SceneGraph, relation: SceneRelation) -> Tuple[float, float]:
        """Get the location of an interaction"""
        subject_obj = next((obj for obj in scene_graph.objects if obj.object_id == relation.subject_id), None)
        object_obj = next((obj for obj in scene_graph.objects if obj.object_id == relation.object_id), None)
        
        if subject_obj and object_obj:
            # Return midpoint between objects
            return (
                (subject_obj.center[0] + object_obj.center[0]) / 2,
                (subject_obj.center[1] + object_obj.center[1]) / 2
            )
        
        return (0, 0)

class BehavioralAnalyzer:
    """Analyzes behavioral patterns for suspicious activity detection"""
    
    def __init__(self, loitering_threshold: float = 30.0):
        self.loitering_threshold = loitering_threshold  # seconds
        self.object_positions = defaultdict(deque)
        self.max_position_history = 100
    
    def analyze_behavior(self, objects: List[SceneObject], frame_id: int) -> List[DetectedEvent]:
        """Analyze behavioral patterns and detect suspicious activities"""
        events = []
        
        for obj in objects:
            # Update position history
            self.object_positions[obj.object_id].append({
                'frame_id': frame_id,
                'timestamp': obj.timestamp,
                'center': obj.center
            })
            
            # Maintain history length
            if len(self.object_positions[obj.object_id]) > self.max_position_history:
                self.object_positions[obj.object_id].popleft()
            
            # Analyze loitering behavior
            loitering_events = self._analyze_loitering(obj.object_id, frame_id)
            events.extend(loitering_events)
            
            # Analyze suspicious movement patterns
            suspicious_events = self._analyze_suspicious_patterns(obj.object_id, frame_id)
            events.extend(suspicious_events)
        
        return events
    
    def _analyze_loitering(self, object_id: str, frame_id: int) -> List[DetectedEvent]:
        """Detect loitering behavior"""
        events = []
        positions = self.object_positions[object_id]
        
        if len(positions) < 10:
            return events
        
        # Calculate area covered in recent positions
        recent_positions = list(positions)[-20:]  # Last 20 positions
        
        if len(recent_positions) < 10:
            return events
        
        # Calculate bounding box of recent positions
        x_coords = [pos['center'][0] for pos in recent_positions]
        y_coords = [pos['center'][1] for pos in recent_positions]
        
        area_covered = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
        time_span = recent_positions[-1]['timestamp'] - recent_positions[0]['timestamp']
        
        # If object stayed in small area for long time, it's loitering
        if area_covered < 2500 and time_span > self.loitering_threshold:  # 50x50 pixel area
            events.append(DetectedEvent(
                event_id=f"loitering_{object_id}_{frame_id}",
                event_type=EventType.LOITERING,
                severity=EventSeverity.MEDIUM,
                confidence=min(1.0, time_span / self.loitering_threshold),
                start_time=recent_positions[0]['timestamp'],
                end_time=recent_positions[-1]['timestamp'],
                start_frame=recent_positions[0]['frame_id'],
                end_frame=frame_id,
                involved_objects=[object_id],
                location=recent_positions[-1]['center'],
                description=f"Object {object_id} loitering for {time_span:.2f} seconds",
                evidence={'area_covered': area_covered, 'time_span': time_span}
            ))
        
        return events
    
    def _analyze_suspicious_patterns(self, object_id: str, frame_id: int) -> List[DetectedEvent]:
        """Analyze movement patterns for suspicious behavior"""
        events = []
        positions = self.object_positions[object_id]
        
        if len(positions) < 15:
            return events
        
        # Analyze for back-and-forth movement (pacing)
        recent_positions = list(positions)[-15:]
        
        # Calculate direction changes
        direction_changes = 0
        for i in range(2, len(recent_positions)):
            v1 = np.array(recent_positions[i-1]['center']) - np.array(recent_positions[i-2]['center'])
            v2 = np.array(recent_positions[i]['center']) - np.array(recent_positions[i-1]['center'])
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle) * 180 / np.pi
                
                if angle > 120:  # Significant direction change
                    direction_changes += 1
        
        # If many direction changes in short time, it's suspicious
        if direction_changes >= 5:
            events.append(DetectedEvent(
                event_id=f"suspicious_behavior_{object_id}_{frame_id}",
                event_type=EventType.SUSPICIOUS_BEHAVIOR,
                severity=EventSeverity.HIGH,
                confidence=min(1.0, direction_changes / 8.0),
                start_time=recent_positions[0]['timestamp'],
                end_time=recent_positions[-1]['timestamp'],
                start_frame=recent_positions[0]['frame_id'],
                end_frame=frame_id,
                involved_objects=[object_id],
                location=recent_positions[-1]['center'],
                description=f"Object {object_id} showing suspicious pacing behavior",
                evidence={'direction_changes': direction_changes, 'analysis_window': len(recent_positions)}
            ))
        
        return events

class EventDetectionSystem:
    """Main event detection system that coordinates all analyzers"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Initialize analyzers
        self.movement_analyzer = MovementAnalyzer(
            velocity_threshold=self.config['velocity_threshold'],
            acceleration_threshold=self.config['acceleration_threshold']
        )
        self.interaction_analyzer = InteractionAnalyzer(
            proximity_threshold=self.config['proximity_threshold']
        )
        self.behavioral_analyzer = BehavioralAnalyzer(
            loitering_threshold=self.config['loitering_threshold']
        )
        
        # Event storage
        self.detected_events = []
        self.event_timeline = []
        
        # Event patterns
        self.event_patterns = self._load_event_patterns()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for event detection"""
        return {
            'velocity_threshold': 5.0,
            'acceleration_threshold': 2.0,
            'proximity_threshold': 100.0,
            'loitering_threshold': 30.0,
            'confidence_threshold': 0.6,
            'enable_movement_analysis': True,
            'enable_interaction_analysis': True,
            'enable_behavioral_analysis': True,
            'enable_pattern_matching': True
        }
    
    def _load_event_patterns(self) -> List[EventPattern]:
        """Load predefined event patterns"""
        patterns = [
            EventPattern(
                pattern_id="person_vehicle_entry",
                event_type=EventType.PERSON_ENTERING,
                required_objects=["person", "vehicle"],
                spatial_constraints={'max_distance': 50.0},
                temporal_constraints={'max_duration': 10.0},
                confidence_threshold=0.7,
                description="Person entering vehicle"
            ),
            EventPattern(
                pattern_id="unauthorized_access",
                event_type=EventType.UNAUTHORIZED_ENTRY,
                required_objects=["person"],
                spatial_constraints={'restricted_area': True},
                temporal_constraints={'time_restriction': True},
                confidence_threshold=0.8,
                description="Person in restricted area during restricted time"
            )
        ]
        return patterns
    
    def detect_events(self, scene_graph: SceneGraph) -> List[DetectedEvent]:
        """Main event detection method"""
        all_events = []
        
        # Movement-based event detection
        if self.config['enable_movement_analysis']:
            movement_events = self.movement_analyzer.analyze_movement(
                scene_graph.objects, scene_graph.frame_id
            )
            all_events.extend(movement_events)
        
        # Interaction-based event detection
        if self.config['enable_interaction_analysis']:
            interaction_events = self.interaction_analyzer.analyze_interactions(scene_graph)
            all_events.extend(interaction_events)
        
        # Behavioral event detection
        if self.config['enable_behavioral_analysis']:
            behavioral_events = self.behavioral_analyzer.analyze_behavior(
                scene_graph.objects, scene_graph.frame_id
            )
            all_events.extend(behavioral_events)
        
        # Pattern-based event detection
        if self.config['enable_pattern_matching']:
            pattern_events = self._detect_pattern_events(scene_graph)
            all_events.extend(pattern_events)
        
        # Filter events by confidence threshold
        filtered_events = [
            event for event in all_events
            if event.confidence >= self.config['confidence_threshold']
        ]
        
        # Store events
        self.detected_events.extend(filtered_events)
        
        # Update timeline
        for event in filtered_events:
            self.event_timeline.append({
                'timestamp': event.start_time,
                'event': event
            })
        
        # Sort timeline
        self.event_timeline.sort(key=lambda x: x['timestamp'])
        
        return filtered_events
    
    def _detect_pattern_events(self, scene_graph: SceneGraph) -> List[DetectedEvent]:
        """Detect events based on predefined patterns"""
        events = []
        
        for pattern in self.event_patterns:
            pattern_events = self._match_pattern(scene_graph, pattern)
            events.extend(pattern_events)
        
        return events
    
    def _match_pattern(self, scene_graph: SceneGraph, pattern: EventPattern) -> List[DetectedEvent]:
        """Match a specific pattern against scene graph"""
        events = []
        
        # Find objects matching pattern requirements
        matching_objects = []
        for required_type in pattern.required_objects:
            type_objects = [obj for obj in scene_graph.objects if obj.object_type == required_type]
            if type_objects:
                matching_objects.extend(type_objects)
        
        if len(matching_objects) < len(pattern.required_objects):
            return events
        
        # Check spatial constraints
        spatial_match = self._check_spatial_constraints(matching_objects, pattern.spatial_constraints)
        
        # Check temporal constraints (simplified)
        temporal_match = True  # Would implement based on specific constraints
        
        if spatial_match and temporal_match:
            events.append(DetectedEvent(
                event_id=f"pattern_{pattern.pattern_id}_{scene_graph.frame_id}",
                event_type=pattern.event_type,
                severity=EventSeverity.MEDIUM,
                confidence=pattern.confidence_threshold,
                start_time=scene_graph.timestamp,
                end_time=scene_graph.timestamp,
                start_frame=scene_graph.frame_id,
                end_frame=scene_graph.frame_id,
                involved_objects=[obj.object_id for obj in matching_objects],
                location=matching_objects[0].center if matching_objects else (0, 0),
                description=pattern.description,
                evidence={'pattern_id': pattern.pattern_id, 'matching_objects': len(matching_objects)}
            ))
        
        return events
    
    def _check_spatial_constraints(self, objects: List[SceneObject], 
                                 constraints: Dict[str, Any]) -> bool:
        """Check if objects satisfy spatial constraints"""
        if 'max_distance' in constraints and len(objects) >= 2:
            distance = np.sqrt((objects[0].center[0] - objects[1].center[0])**2 + 
                             (objects[0].center[1] - objects[1].center[1])**2)
            return distance <= constraints['max_distance']
        
        return True
    
    def get_event_summary(self) -> Dict[str, Any]:
        """Get summary of detected events"""
        if not self.detected_events:
            return {}
        
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for event in self.detected_events:
            event_counts[event.event_type.value] += 1
            severity_counts[event.severity.value] += 1
        
        return {
            'total_events': len(self.detected_events),
            'event_type_distribution': dict(event_counts),
            'severity_distribution': dict(severity_counts),
            'timeline_span': {
                'start': min(event.start_time for event in self.detected_events),
                'end': max(event.end_time for event in self.detected_events)
            },
            'high_severity_events': len([e for e in self.detected_events if e.severity == EventSeverity.HIGH]),
            'critical_events': len([e for e in self.detected_events if e.severity == EventSeverity.CRITICAL])
        }
    
    def export_events(self, output_path: str, format: str = 'json'):
        """Export detected events to file"""
        output_path = Path(output_path)
        
        if format == 'json':
            self._export_events_json(output_path)
        elif format == 'timeline':
            self._export_timeline(output_path)
        elif format == 'report':
            self._export_forensic_report(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_events_json(self, output_path: Path):
        """Export events as JSON"""
        export_data = {
            'events': [
                {
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'severity': event.severity.value,
                    'confidence': event.confidence,
                    'start_time': event.start_time,
                    'end_time': event.end_time,
                    'start_frame': event.start_frame,
                    'end_frame': event.end_frame,
                    'involved_objects': event.involved_objects,
                    'location': event.location,
                    'description': event.description,
                    'evidence': event.evidence,
                    'metadata': event.metadata
                }
                for event in self.detected_events
            ],
            'summary': self.get_event_summary(),
            'export_metadata': {
                'export_time': datetime.now().isoformat(),
                'total_events': len(self.detected_events)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Events exported to {output_path}")
    
    def _export_timeline(self, output_path: Path):
        """Export event timeline visualization"""
        if not self.event_timeline:
            return
        
        plt.figure(figsize=(15, 8))
        
        # Prepare data for timeline
        timestamps = [item['timestamp'] for item in self.event_timeline]
        event_types = [item['event'].event_type.value for item in self.event_timeline]
        severities = [item['event'].severity.value for item in self.event_timeline]
        
        # Color mapping for severities
        severity_colors = {
            'low': 'green',
            'medium': 'orange',
            'high': 'red',
            'critical': 'darkred'
        }
        
        colors = [severity_colors.get(sev, 'blue') for sev in severities]
        
        # Create timeline plot
        plt.scatter(timestamps, range(len(timestamps)), c=colors, alpha=0.7, s=100)
        
        # Add labels
        for i, (ts, event_type) in enumerate(zip(timestamps, event_types)):
            plt.annotate(event_type, (ts, i), xytext=(5, 0), 
                        textcoords='offset points', fontsize=8, rotation=45)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Event Sequence')
        plt.title('Forensic Event Timeline')
        plt.grid(True, alpha=0.3)
        
        # Add legend
        for severity, color in severity_colors.items():
            plt.scatter([], [], c=color, label=severity.capitalize())
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Event timeline exported to {output_path}")
    
    def _export_forensic_report(self, output_path: Path):
        """Export comprehensive forensic report"""
        report = {
            'forensic_event_analysis_report': {
                'case_id': 'AUTO_GENERATED',
                'analysis_date': datetime.now().isoformat(),
                'summary': self.get_event_summary(),
                'critical_findings': [
                    {
                        'event_id': event.event_id,
                        'event_type': event.event_type.value,
                        'severity': event.severity.value,
                        'timestamp': event.start_time,
                        'description': event.description,
                        'confidence': event.confidence,
                        'evidence': event.evidence
                    }
                    for event in self.detected_events
                    if event.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]
                ],
                'timeline_analysis': {
                    'total_duration': max(event.end_time for event in self.detected_events) - 
                                    min(event.start_time for event in self.detected_events) 
                                    if self.detected_events else 0,
                    'event_density': len(self.detected_events) / 
                                   (max(event.end_time for event in self.detected_events) - 
                                    min(event.start_time for event in self.detected_events) + 1)
                                   if self.detected_events else 0,
                    'peak_activity_periods': []  # Would implement peak detection
                },
                'recommendations': [
                    "Review high-severity events for potential criminal activity",
                    "Investigate critical events with additional evidence",
                    "Cross-reference timeline with external data sources",
                    "Validate automated detections with human analysis"
                ]
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Forensic report exported to {output_path}")

# Example usage and testing
if __name__ == "__main__":
    from scene_graph_generation import SceneGraphGenerator, SceneObject
    
    # Create sample scene graph
    sample_objects = [
        SceneObject(
            object_id="person_001",
            object_type="person",
            confidence=0.95,
            bbox=(100, 100, 200, 300),
            center=(150, 200),
            area=20000,
            frame_id=1,
            timestamp=0.0,
            camera_id="cam_001"
        ),
        SceneObject(
            object_id="person_002",
            object_type="person",
            confidence=0.88,
            bbox=(180, 120, 280, 320),
            center=(230, 220),
            area=20000,
            frame_id=1,
            timestamp=0.0,
            camera_id="cam_001"
        )
    ]
    
    # Generate scene graph
    sg_generator = SceneGraphGenerator()
    scene_graph = sg_generator.generate_scene_graph(
        objects=sample_objects,
        frame_id=1,
        timestamp=0.0,
        scene_id="test_scene"
    )
    
    # Initialize event detection system
    event_detector = EventDetectionSystem()
    
    # Detect events
    events = event_detector.detect_events(scene_graph)
    
    # Print results
    print(f"Detected {len(events)} events:")
    for event in events:
        print(f"  - {event.event_type.value}: {event.description} (confidence: {event.confidence:.2f})")
    
    # Export events
    event_detector.export_events("detected_events.json", format='json')
    print("Events exported successfully!")
