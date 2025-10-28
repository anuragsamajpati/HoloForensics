"""
Scene Graph Generation for Forensic Video Analysis
Generates semantic scene graphs from multi-camera forensic footage
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RelationType(Enum):
    """Types of relationships between objects in scene graph"""
    SPATIAL_NEAR = "near"
    SPATIAL_FAR = "far"
    SPATIAL_LEFT = "left_of"
    SPATIAL_RIGHT = "right_of"
    SPATIAL_ABOVE = "above"
    SPATIAL_BELOW = "below"
    TEMPORAL_BEFORE = "before"
    TEMPORAL_AFTER = "after"
    TEMPORAL_DURING = "during"
    INTERACTION_TOUCHING = "touching"
    INTERACTION_HOLDING = "holding"
    INTERACTION_CARRYING = "carrying"
    INTERACTION_FOLLOWING = "following"
    INTERACTION_APPROACHING = "approaching"
    INTERACTION_AVOIDING = "avoiding"
    FUNCTIONAL_USING = "using"
    FUNCTIONAL_ENTERING = "entering"
    FUNCTIONAL_EXITING = "exiting"
    FUNCTIONAL_OPERATING = "operating"

@dataclass
class SceneObject:
    """Represents an object in the forensic scene"""
    object_id: str
    object_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[float, float]
    area: float
    frame_id: int
    timestamp: float
    camera_id: str
    features: Optional[np.ndarray] = None
    attributes: Dict[str, Any] = None

@dataclass
class SceneRelation:
    """Represents a relationship between objects"""
    subject_id: str
    object_id: str
    relation_type: RelationType
    confidence: float
    frame_id: int
    timestamp: float
    spatial_context: Dict[str, float] = None
    temporal_context: Dict[str, float] = None

@dataclass
class SceneGraph:
    """Complete scene graph for a forensic scene"""
    scene_id: str
    frame_id: int
    timestamp: float
    objects: List[SceneObject]
    relations: List[SceneRelation]
    metadata: Dict[str, Any] = None

class SpatialRelationExtractor:
    """Extracts spatial relationships between objects"""
    
    def __init__(self, proximity_threshold: float = 100.0):
        self.proximity_threshold = proximity_threshold
    
    def extract_spatial_relations(self, objects: List[SceneObject]) -> List[SceneRelation]:
        """Extract spatial relationships between objects"""
        relations = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:  # Avoid duplicate pairs and self-relations
                    continue
                
                # Calculate spatial relationships
                spatial_relations = self._compute_spatial_relations(obj1, obj2)
                relations.extend(spatial_relations)
        
        return relations
    
    def _compute_spatial_relations(self, obj1: SceneObject, obj2: SceneObject) -> List[SceneRelation]:
        """Compute spatial relationships between two objects"""
        relations = []
        
        # Calculate distance
        distance = np.sqrt((obj1.center[0] - obj2.center[0])**2 + 
                          (obj1.center[1] - obj2.center[1])**2)
        
        # Proximity relation
        if distance < self.proximity_threshold:
            relations.append(SceneRelation(
                subject_id=obj1.object_id,
                object_id=obj2.object_id,
                relation_type=RelationType.SPATIAL_NEAR,
                confidence=max(0.0, 1.0 - distance / self.proximity_threshold),
                frame_id=obj1.frame_id,
                timestamp=obj1.timestamp,
                spatial_context={'distance': distance}
            ))
        else:
            relations.append(SceneRelation(
                subject_id=obj1.object_id,
                object_id=obj2.object_id,
                relation_type=RelationType.SPATIAL_FAR,
                confidence=min(1.0, distance / self.proximity_threshold - 1.0),
                frame_id=obj1.frame_id,
                timestamp=obj1.timestamp,
                spatial_context={'distance': distance}
            ))
        
        # Directional relations
        dx = obj2.center[0] - obj1.center[0]
        dy = obj2.center[1] - obj1.center[1]
        
        if abs(dx) > abs(dy):  # Horizontal relationship is stronger
            if dx > 0:
                relations.append(SceneRelation(
                    subject_id=obj1.object_id,
                    object_id=obj2.object_id,
                    relation_type=RelationType.SPATIAL_LEFT,
                    confidence=abs(dx) / (abs(dx) + abs(dy)),
                    frame_id=obj1.frame_id,
                    timestamp=obj1.timestamp,
                    spatial_context={'dx': dx, 'dy': dy}
                ))
            else:
                relations.append(SceneRelation(
                    subject_id=obj1.object_id,
                    object_id=obj2.object_id,
                    relation_type=RelationType.SPATIAL_RIGHT,
                    confidence=abs(dx) / (abs(dx) + abs(dy)),
                    frame_id=obj1.frame_id,
                    timestamp=obj1.timestamp,
                    spatial_context={'dx': dx, 'dy': dy}
                ))
        else:  # Vertical relationship is stronger
            if dy > 0:
                relations.append(SceneRelation(
                    subject_id=obj1.object_id,
                    object_id=obj2.object_id,
                    relation_type=RelationType.SPATIAL_ABOVE,
                    confidence=abs(dy) / (abs(dx) + abs(dy)),
                    frame_id=obj1.frame_id,
                    timestamp=obj1.timestamp,
                    spatial_context={'dx': dx, 'dy': dy}
                ))
            else:
                relations.append(SceneRelation(
                    subject_id=obj1.object_id,
                    object_id=obj2.object_id,
                    relation_type=RelationType.SPATIAL_BELOW,
                    confidence=abs(dy) / (abs(dx) + abs(dy)),
                    frame_id=obj1.frame_id,
                    timestamp=obj1.timestamp,
                    spatial_context={'dx': dx, 'dy': dy}
                ))
        
        return relations

class TemporalRelationExtractor:
    """Extracts temporal relationships between objects across frames"""
    
    def __init__(self, temporal_window: int = 30):
        self.temporal_window = temporal_window
        self.object_history = defaultdict(list)
    
    def extract_temporal_relations(self, current_objects: List[SceneObject], 
                                 frame_id: int) -> List[SceneRelation]:
        """Extract temporal relationships"""
        relations = []
        
        # Update object history
        for obj in current_objects:
            self.object_history[obj.object_id].append({
                'frame_id': frame_id,
                'timestamp': obj.timestamp,
                'center': obj.center,
                'bbox': obj.bbox
            })
            
            # Keep only recent history
            if len(self.object_history[obj.object_id]) > self.temporal_window:
                self.object_history[obj.object_id].pop(0)
        
        # Analyze temporal patterns
        for obj_id, history in self.object_history.items():
            if len(history) >= 2:
                temporal_relations = self._analyze_temporal_patterns(obj_id, history, frame_id)
                relations.extend(temporal_relations)
        
        return relations
    
    def _analyze_temporal_patterns(self, obj_id: str, history: List[Dict], 
                                 current_frame: int) -> List[SceneRelation]:
        """Analyze temporal patterns for an object"""
        relations = []
        
        if len(history) < 2:
            return relations
        
        # Analyze movement patterns
        recent_positions = [h['center'] for h in history[-5:]]
        if len(recent_positions) >= 2:
            # Calculate velocity and acceleration
            velocities = []
            for i in range(1, len(recent_positions)):
                dx = recent_positions[i][0] - recent_positions[i-1][0]
                dy = recent_positions[i][1] - recent_positions[i-1][1]
                velocities.append((dx, dy))
            
            # Detect movement patterns
            if len(velocities) >= 2:
                avg_velocity = np.mean(velocities, axis=0)
                velocity_magnitude = np.linalg.norm(avg_velocity)
                
                # Object is moving significantly
                if velocity_magnitude > 5.0:
                    # Check for interactions with other objects
                    for other_obj_id, other_history in self.object_history.items():
                        if other_obj_id != obj_id and other_history:
                            interaction_relations = self._detect_interactions(
                                obj_id, history, other_obj_id, other_history, current_frame
                            )
                            relations.extend(interaction_relations)
        
        return relations
    
    def _detect_interactions(self, obj1_id: str, obj1_history: List[Dict],
                           obj2_id: str, obj2_history: List[Dict],
                           current_frame: int) -> List[SceneRelation]:
        """Detect interactions between two objects over time"""
        relations = []
        
        if len(obj1_history) < 2 or len(obj2_history) < 2:
            return relations
        
        # Get recent positions
        obj1_recent = obj1_history[-3:]
        obj2_recent = obj2_history[-3:]
        
        # Calculate distances over time
        distances = []
        for i in range(min(len(obj1_recent), len(obj2_recent))):
            dist = np.sqrt((obj1_recent[i]['center'][0] - obj2_recent[i]['center'][0])**2 +
                          (obj1_recent[i]['center'][1] - obj2_recent[i]['center'][1])**2)
            distances.append(dist)
        
        if len(distances) >= 2:
            # Approaching behavior
            if distances[-1] < distances[0] and distances[-1] < 100:
                relations.append(SceneRelation(
                    subject_id=obj1_id,
                    object_id=obj2_id,
                    relation_type=RelationType.INTERACTION_APPROACHING,
                    confidence=1.0 - distances[-1] / 200.0,
                    frame_id=current_frame,
                    timestamp=obj1_history[-1]['timestamp'],
                    temporal_context={'distance_change': distances[0] - distances[-1]}
                ))
            
            # Following behavior
            if self._is_following(obj1_recent, obj2_recent):
                relations.append(SceneRelation(
                    subject_id=obj1_id,
                    object_id=obj2_id,
                    relation_type=RelationType.INTERACTION_FOLLOWING,
                    confidence=0.8,
                    frame_id=current_frame,
                    timestamp=obj1_history[-1]['timestamp']
                ))
        
        return relations
    
    def _is_following(self, obj1_history: List[Dict], obj2_history: List[Dict]) -> bool:
        """Detect if obj1 is following obj2"""
        if len(obj1_history) < 3 or len(obj2_history) < 3:
            return False
        
        # Calculate movement vectors
        obj1_movement = np.array(obj1_history[-1]['center']) - np.array(obj1_history[-3]['center'])
        obj2_movement = np.array(obj2_history[-1]['center']) - np.array(obj2_history[-3]['center'])
        
        # Check if movements are in similar direction
        if np.linalg.norm(obj1_movement) > 0 and np.linalg.norm(obj2_movement) > 0:
            cosine_similarity = np.dot(obj1_movement, obj2_movement) / (
                np.linalg.norm(obj1_movement) * np.linalg.norm(obj2_movement)
            )
            return cosine_similarity > 0.7
        
        return False

class InteractionDetector:
    """Detects complex interactions between objects"""
    
    def __init__(self):
        self.interaction_patterns = self._load_interaction_patterns()
    
    def _load_interaction_patterns(self) -> Dict[str, Any]:
        """Load predefined interaction patterns"""
        return {
            'person_vehicle_entering': {
                'objects': ['person', 'vehicle'],
                'spatial_threshold': 50.0,
                'temporal_window': 10,
                'pattern': 'person_approaches_then_disappears'
            },
            'person_door_interaction': {
                'objects': ['person', 'door'],
                'spatial_threshold': 30.0,
                'temporal_window': 15,
                'pattern': 'person_near_door_movement'
            },
            'object_handoff': {
                'objects': ['person', 'person'],
                'spatial_threshold': 40.0,
                'temporal_window': 20,
                'pattern': 'close_proximity_brief_contact'
            }
        }
    
    def detect_interactions(self, scene_graphs: List[SceneGraph]) -> List[SceneRelation]:
        """Detect complex interactions across multiple frames"""
        interactions = []
        
        for pattern_name, pattern_config in self.interaction_patterns.items():
            pattern_interactions = self._detect_pattern(scene_graphs, pattern_config)
            interactions.extend(pattern_interactions)
        
        return interactions
    
    def _detect_pattern(self, scene_graphs: List[SceneGraph], 
                       pattern_config: Dict[str, Any]) -> List[SceneRelation]:
        """Detect a specific interaction pattern"""
        interactions = []
        
        # Implementation would depend on specific pattern
        # This is a simplified version
        for graph in scene_graphs:
            pattern_interactions = self._analyze_frame_for_pattern(graph, pattern_config)
            interactions.extend(pattern_interactions)
        
        return interactions
    
    def _analyze_frame_for_pattern(self, scene_graph: SceneGraph,
                                 pattern_config: Dict[str, Any]) -> List[SceneRelation]:
        """Analyze a single frame for interaction patterns"""
        interactions = []
        
        # Find objects matching pattern requirements
        relevant_objects = [
            obj for obj in scene_graph.objects 
            if obj.object_type in pattern_config['objects']
        ]
        
        # Analyze spatial relationships for pattern matching
        for i, obj1 in enumerate(relevant_objects):
            for j, obj2 in enumerate(relevant_objects):
                if i >= j:
                    continue
                
                distance = np.sqrt((obj1.center[0] - obj2.center[0])**2 + 
                                 (obj1.center[1] - obj2.center[1])**2)
                
                if distance < pattern_config['spatial_threshold']:
                    # Detected potential interaction
                    interaction_type = self._classify_interaction(obj1, obj2, pattern_config)
                    if interaction_type:
                        interactions.append(SceneRelation(
                            subject_id=obj1.object_id,
                            object_id=obj2.object_id,
                            relation_type=interaction_type,
                            confidence=1.0 - distance / pattern_config['spatial_threshold'],
                            frame_id=scene_graph.frame_id,
                            timestamp=scene_graph.timestamp
                        ))
        
        return interactions
    
    def _classify_interaction(self, obj1: SceneObject, obj2: SceneObject,
                            pattern_config: Dict[str, Any]) -> Optional[RelationType]:
        """Classify the type of interaction"""
        # Simplified classification logic
        if obj1.object_type == 'person' and obj2.object_type == 'vehicle':
            return RelationType.FUNCTIONAL_ENTERING
        elif obj1.object_type == 'person' and obj2.object_type == 'person':
            return RelationType.INTERACTION_TOUCHING
        
        return None

class SceneGraphGenerator:
    """Main class for generating scene graphs from forensic footage"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.spatial_extractor = SpatialRelationExtractor(
            proximity_threshold=self.config['spatial_proximity_threshold']
        )
        self.temporal_extractor = TemporalRelationExtractor(
            temporal_window=self.config['temporal_window']
        )
        self.interaction_detector = InteractionDetector()
        
        # Storage for scene graphs
        self.scene_graphs = []
        self.global_graph = nx.DiGraph()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for scene graph generation"""
        return {
            'spatial_proximity_threshold': 100.0,
            'temporal_window': 30,
            'confidence_threshold': 0.5,
            'max_objects_per_frame': 50,
            'enable_temporal_analysis': True,
            'enable_interaction_detection': True
        }
    
    def generate_scene_graph(self, objects: List[SceneObject], 
                           frame_id: int, timestamp: float,
                           scene_id: str = "default") -> SceneGraph:
        """Generate scene graph for a single frame"""
        
        # Filter objects by confidence
        filtered_objects = [
            obj for obj in objects 
            if obj.confidence >= self.config['confidence_threshold']
        ]
        
        # Limit number of objects if necessary
        if len(filtered_objects) > self.config['max_objects_per_frame']:
            filtered_objects = sorted(filtered_objects, key=lambda x: x.confidence, reverse=True)
            filtered_objects = filtered_objects[:self.config['max_objects_per_frame']]
        
        # Extract spatial relations
        spatial_relations = self.spatial_extractor.extract_spatial_relations(filtered_objects)
        
        # Extract temporal relations if enabled
        temporal_relations = []
        if self.config['enable_temporal_analysis']:
            temporal_relations = self.temporal_extractor.extract_temporal_relations(
                filtered_objects, frame_id
            )
        
        # Combine all relations
        all_relations = spatial_relations + temporal_relations
        
        # Create scene graph
        scene_graph = SceneGraph(
            scene_id=scene_id,
            frame_id=frame_id,
            timestamp=timestamp,
            objects=filtered_objects,
            relations=all_relations,
            metadata={
                'num_objects': len(filtered_objects),
                'num_relations': len(all_relations),
                'generation_time': datetime.now().isoformat()
            }
        )
        
        # Store scene graph
        self.scene_graphs.append(scene_graph)
        
        # Update global graph
        self._update_global_graph(scene_graph)
        
        return scene_graph
    
    def _update_global_graph(self, scene_graph: SceneGraph):
        """Update the global scene graph with new frame data"""
        
        # Add nodes for objects
        for obj in scene_graph.objects:
            self.global_graph.add_node(
                obj.object_id,
                object_type=obj.object_type,
                confidence=obj.confidence,
                first_seen=scene_graph.timestamp,
                last_seen=scene_graph.timestamp
            )
        
        # Add edges for relations
        for relation in scene_graph.relations:
            edge_key = f"{relation.subject_id}_{relation.object_id}_{relation.relation_type.value}"
            
            if self.global_graph.has_edge(relation.subject_id, relation.object_id):
                # Update existing edge
                edge_data = self.global_graph[relation.subject_id][relation.object_id]
                edge_data['relations'].append(relation)
                edge_data['last_seen'] = scene_graph.timestamp
            else:
                # Add new edge
                self.global_graph.add_edge(
                    relation.subject_id,
                    relation.object_id,
                    relations=[relation],
                    first_seen=scene_graph.timestamp,
                    last_seen=scene_graph.timestamp
                )
    
    def detect_complex_interactions(self) -> List[SceneRelation]:
        """Detect complex interactions across multiple frames"""
        if not self.config['enable_interaction_detection']:
            return []
        
        return self.interaction_detector.detect_interactions(self.scene_graphs)
    
    def export_scene_graph(self, output_path: str, format: str = 'json'):
        """Export scene graph to file"""
        output_path = Path(output_path)
        
        if format == 'json':
            self._export_json(output_path)
        elif format == 'graphml':
            self._export_graphml(output_path)
        elif format == 'visualization':
            self._export_visualization(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, output_path: Path):
        """Export scene graphs as JSON"""
        export_data = {
            'scene_graphs': [],
            'global_graph': {
                'nodes': [],
                'edges': []
            },
            'metadata': {
                'total_frames': len(self.scene_graphs),
                'export_time': datetime.now().isoformat()
            }
        }
        
        # Export individual scene graphs
        for sg in self.scene_graphs:
            sg_data = {
                'scene_id': sg.scene_id,
                'frame_id': sg.frame_id,
                'timestamp': sg.timestamp,
                'objects': [
                    {
                        'object_id': obj.object_id,
                        'object_type': obj.object_type,
                        'confidence': obj.confidence,
                        'bbox': obj.bbox,
                        'center': obj.center,
                        'area': obj.area
                    }
                    for obj in sg.objects
                ],
                'relations': [
                    {
                        'subject_id': rel.subject_id,
                        'object_id': rel.object_id,
                        'relation_type': rel.relation_type.value,
                        'confidence': rel.confidence,
                        'spatial_context': rel.spatial_context,
                        'temporal_context': rel.temporal_context
                    }
                    for rel in sg.relations
                ],
                'metadata': sg.metadata
            }
            export_data['scene_graphs'].append(sg_data)
        
        # Export global graph
        for node_id, node_data in self.global_graph.nodes(data=True):
            export_data['global_graph']['nodes'].append({
                'id': node_id,
                'data': node_data
            })
        
        for source, target, edge_data in self.global_graph.edges(data=True):
            export_data['global_graph']['edges'].append({
                'source': source,
                'target': target,
                'data': {
                    'relations': [
                        {
                            'relation_type': rel.relation_type.value,
                            'confidence': rel.confidence,
                            'timestamp': rel.timestamp
                        }
                        for rel in edge_data.get('relations', [])
                    ],
                    'first_seen': edge_data.get('first_seen'),
                    'last_seen': edge_data.get('last_seen')
                }
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Scene graphs exported to {output_path}")
    
    def _export_graphml(self, output_path: Path):
        """Export global graph as GraphML"""
        nx.write_graphml(self.global_graph, output_path)
        logger.info(f"Global graph exported to {output_path}")
    
    def _export_visualization(self, output_path: Path):
        """Export scene graph visualization"""
        plt.figure(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(self.global_graph, k=1, iterations=50)
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        for node_id, node_data in self.global_graph.nodes(data=True):
            if node_data.get('object_type') == 'person':
                node_colors.append('lightblue')
                node_sizes.append(500)
            elif node_data.get('object_type') == 'vehicle':
                node_colors.append('lightcoral')
                node_sizes.append(700)
            else:
                node_colors.append('lightgreen')
                node_sizes.append(300)
        
        nx.draw_networkx_nodes(self.global_graph, pos, 
                              node_color=node_colors, 
                              node_size=node_sizes,
                              alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(self.global_graph, pos, 
                              edge_color='gray', 
                              alpha=0.5,
                              arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(self.global_graph, pos, 
                               font_size=8, 
                               font_weight='bold')
        
        plt.title("Forensic Scene Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Scene graph visualization saved to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated scene graphs"""
        if not self.scene_graphs:
            return {}
        
        total_objects = sum(len(sg.objects) for sg in self.scene_graphs)
        total_relations = sum(len(sg.relations) for sg in self.scene_graphs)
        
        object_types = defaultdict(int)
        relation_types = defaultdict(int)
        
        for sg in self.scene_graphs:
            for obj in sg.objects:
                object_types[obj.object_type] += 1
            for rel in sg.relations:
                relation_types[rel.relation_type.value] += 1
        
        return {
            'total_frames': len(self.scene_graphs),
            'total_objects': total_objects,
            'total_relations': total_relations,
            'avg_objects_per_frame': total_objects / len(self.scene_graphs),
            'avg_relations_per_frame': total_relations / len(self.scene_graphs),
            'object_type_distribution': dict(object_types),
            'relation_type_distribution': dict(relation_types),
            'global_graph_nodes': self.global_graph.number_of_nodes(),
            'global_graph_edges': self.global_graph.number_of_edges()
        }

# Example usage and testing
if __name__ == "__main__":
    # Create sample objects for testing
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
            object_id="vehicle_001",
            object_type="vehicle",
            confidence=0.88,
            bbox=(300, 150, 500, 250),
            center=(400, 200),
            area=20000,
            frame_id=1,
            timestamp=0.0,
            camera_id="cam_001"
        )
    ]
    
    # Initialize scene graph generator
    generator = SceneGraphGenerator()
    
    # Generate scene graph
    scene_graph = generator.generate_scene_graph(
        objects=sample_objects,
        frame_id=1,
        timestamp=0.0,
        scene_id="test_scene"
    )
    
    # Print statistics
    stats = generator.get_statistics()
    print("Scene Graph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Export scene graph
    generator.export_scene_graph("test_scene_graph.json", format='json')
    print("Scene graph exported successfully!")
