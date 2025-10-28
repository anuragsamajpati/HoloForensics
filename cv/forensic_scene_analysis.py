"""
Forensic Scene Analysis Integration
Integrates scene graph generation and event detection for forensic video analysis
"""

import numpy as np
import cv2
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
from datetime import datetime
import uuid
import hashlib
from collections import defaultdict

from scene_graph_generation import SceneGraphGenerator, SceneGraph, SceneObject
from event_detection import EventDetectionSystem, DetectedEvent, EventType, EventSeverity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ForensicAnalysisConfig:
    """Configuration for forensic scene analysis"""
    case_id: str
    operator_id: str
    analysis_type: str = "comprehensive"
    enable_scene_graphs: bool = True
    enable_event_detection: bool = True
    enable_chain_of_custody: bool = True
    confidence_threshold: float = 0.6
    temporal_analysis_window: int = 30
    spatial_proximity_threshold: float = 100.0
    evidence_preservation: bool = True
    generate_visualizations: bool = True
    export_formats: List[str] = None
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ['json', 'report', 'timeline']

@dataclass
class ForensicEvidence:
    """Represents a piece of forensic evidence"""
    evidence_id: str
    evidence_type: str
    timestamp: float
    frame_id: int
    camera_id: str
    location: Tuple[float, float]
    confidence: float
    description: str
    raw_data: Dict[str, Any]
    chain_of_custody: List[Dict[str, Any]]
    hash_signature: str
    metadata: Dict[str, Any] = None

@dataclass
class ForensicAnalysisResult:
    """Complete forensic analysis result"""
    analysis_id: str
    case_id: str
    timestamp: datetime
    operator_id: str
    scene_graphs: List[SceneGraph]
    detected_events: List[DetectedEvent]
    evidence_items: List[ForensicEvidence]
    analysis_summary: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    chain_of_custody_log: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None

class ChainOfCustodyManager:
    """Manages chain of custody for forensic evidence"""
    
    def __init__(self, case_id: str, operator_id: str):
        self.case_id = case_id
        self.operator_id = operator_id
        self.custody_log = []
    
    def log_action(self, action: str, evidence_id: str, details: Dict[str, Any] = None):
        """Log a chain of custody action"""
        custody_entry = {
            'timestamp': datetime.now().isoformat(),
            'case_id': self.case_id,
            'operator_id': self.operator_id,
            'action': action,
            'evidence_id': evidence_id,
            'details': details or {},
            'system_info': {
                'version': '1.0.0',
                'analysis_system': 'HoloForensics'
            }
        }
        
        self.custody_log.append(custody_entry)
        logger.info(f"Chain of custody: {action} for evidence {evidence_id}")
    
    def get_custody_log(self) -> List[Dict[str, Any]]:
        """Get complete chain of custody log"""
        return self.custody_log.copy()
    
    def verify_integrity(self, evidence: ForensicEvidence) -> bool:
        """Verify evidence integrity using hash signatures"""
        # Recalculate hash from raw data
        data_string = json.dumps(evidence.raw_data, sort_keys=True)
        calculated_hash = hashlib.sha256(data_string.encode()).hexdigest()
        
        return calculated_hash == evidence.hash_signature

class ForensicQualityAssurance:
    """Quality assurance for forensic analysis"""
    
    def __init__(self):
        self.quality_checks = [
            'temporal_consistency',
            'spatial_accuracy',
            'confidence_validation',
            'evidence_integrity',
            'chain_of_custody_completeness'
        ]
    
    def assess_quality(self, analysis_result: ForensicAnalysisResult) -> Dict[str, Any]:
        """Assess quality of forensic analysis"""
        quality_metrics = {}
        
        # Temporal consistency check
        quality_metrics['temporal_consistency'] = self._check_temporal_consistency(
            analysis_result.scene_graphs
        )
        
        # Spatial accuracy check
        quality_metrics['spatial_accuracy'] = self._check_spatial_accuracy(
            analysis_result.scene_graphs
        )
        
        # Confidence validation
        quality_metrics['confidence_validation'] = self._validate_confidence_scores(
            analysis_result.detected_events
        )
        
        # Evidence integrity check
        quality_metrics['evidence_integrity'] = self._check_evidence_integrity(
            analysis_result.evidence_items
        )
        
        # Chain of custody completeness
        quality_metrics['chain_of_custody_completeness'] = self._check_custody_completeness(
            analysis_result.chain_of_custody_log
        )
        
        # Overall quality score
        quality_metrics['overall_score'] = np.mean([
            score for score in quality_metrics.values() 
            if isinstance(score, (int, float))
        ])
        
        return quality_metrics
    
    def _check_temporal_consistency(self, scene_graphs: List[SceneGraph]) -> float:
        """Check temporal consistency of scene graphs"""
        if len(scene_graphs) < 2:
            return 1.0
        
        inconsistencies = 0
        total_checks = 0
        
        for i in range(1, len(scene_graphs)):
            prev_graph = scene_graphs[i-1]
            curr_graph = scene_graphs[i]
            
            # Check timestamp ordering
            if curr_graph.timestamp <= prev_graph.timestamp:
                inconsistencies += 1
            total_checks += 1
            
            # Check frame ID ordering
            if curr_graph.frame_id <= prev_graph.frame_id:
                inconsistencies += 1
            total_checks += 1
        
        return 1.0 - (inconsistencies / max(total_checks, 1))
    
    def _check_spatial_accuracy(self, scene_graphs: List[SceneGraph]) -> float:
        """Check spatial accuracy of object positions"""
        if not scene_graphs:
            return 1.0
        
        accuracy_scores = []
        
        for graph in scene_graphs:
            for obj in graph.objects:
                # Check if bounding box is reasonable
                bbox_area = (obj.bbox[2] - obj.bbox[0]) * (obj.bbox[3] - obj.bbox[1])
                if bbox_area > 0 and obj.area > 0:
                    # Center should be within bounding box
                    center_in_bbox = (
                        obj.bbox[0] <= obj.center[0] <= obj.bbox[2] and
                        obj.bbox[1] <= obj.center[1] <= obj.bbox[3]
                    )
                    accuracy_scores.append(1.0 if center_in_bbox else 0.0)
        
        return np.mean(accuracy_scores) if accuracy_scores else 1.0
    
    def _validate_confidence_scores(self, events: List[DetectedEvent]) -> float:
        """Validate confidence scores of detected events"""
        if not events:
            return 1.0
        
        valid_scores = [
            1.0 if 0.0 <= event.confidence <= 1.0 else 0.0
            for event in events
        ]
        
        return np.mean(valid_scores)
    
    def _check_evidence_integrity(self, evidence_items: List[ForensicEvidence]) -> float:
        """Check integrity of evidence items"""
        if not evidence_items:
            return 1.0
        
        integrity_scores = []
        
        for evidence in evidence_items:
            # Check hash signature
            data_string = json.dumps(evidence.raw_data, sort_keys=True)
            calculated_hash = hashlib.sha256(data_string.encode()).hexdigest()
            
            integrity_scores.append(1.0 if calculated_hash == evidence.hash_signature else 0.0)
        
        return np.mean(integrity_scores)
    
    def _check_custody_completeness(self, custody_log: List[Dict[str, Any]]) -> float:
        """Check completeness of chain of custody log"""
        if not custody_log:
            return 0.0
        
        required_fields = ['timestamp', 'case_id', 'operator_id', 'action', 'evidence_id']
        completeness_scores = []
        
        for entry in custody_log:
            missing_fields = sum(1 for field in required_fields if field not in entry)
            completeness_scores.append(1.0 - missing_fields / len(required_fields))
        
        return np.mean(completeness_scores)

class ForensicSceneAnalyzer:
    """Main forensic scene analysis system"""
    
    def __init__(self, config: ForensicAnalysisConfig):
        self.config = config
        self.analysis_id = str(uuid.uuid4())
        
        # Initialize components
        if config.enable_scene_graphs:
            self.scene_graph_generator = SceneGraphGenerator({
                'spatial_proximity_threshold': config.spatial_proximity_threshold,
                'temporal_window': config.temporal_analysis_window,
                'confidence_threshold': config.confidence_threshold
            })
        
        if config.enable_event_detection:
            self.event_detector = EventDetectionSystem({
                'confidence_threshold': config.confidence_threshold,
                'proximity_threshold': config.spatial_proximity_threshold
            })
        
        # Chain of custody and quality assurance
        self.custody_manager = ChainOfCustodyManager(config.case_id, config.operator_id)
        self.quality_assurance = ForensicQualityAssurance()
        
        # Storage
        self.scene_graphs = []
        self.detected_events = []
        self.evidence_items = []
        
        # Log analysis start
        self.custody_manager.log_action(
            "analysis_started",
            self.analysis_id,
            {"analysis_type": config.analysis_type, "operator": config.operator_id}
        )
    
    def analyze_frame(self, objects: List[SceneObject], frame_id: int, 
                     timestamp: float, camera_id: str) -> Dict[str, Any]:
        """Analyze a single frame"""
        frame_results = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'camera_id': camera_id,
            'scene_graph': None,
            'events': [],
            'evidence': []
        }
        
        try:
            # Generate scene graph
            if self.config.enable_scene_graphs:
                scene_graph = self.scene_graph_generator.generate_scene_graph(
                    objects=objects,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    scene_id=self.config.case_id
                )
                
                self.scene_graphs.append(scene_graph)
                frame_results['scene_graph'] = scene_graph
                
                # Log scene graph generation
                self.custody_manager.log_action(
                    "scene_graph_generated",
                    f"frame_{frame_id}",
                    {
                        "num_objects": len(objects),
                        "num_relations": len(scene_graph.relations),
                        "camera_id": camera_id
                    }
                )
            
            # Detect events
            if self.config.enable_event_detection and frame_results['scene_graph']:
                events = self.event_detector.detect_events(frame_results['scene_graph'])
                self.detected_events.extend(events)
                frame_results['events'] = events
                
                # Log event detection
                if events:
                    self.custody_manager.log_action(
                        "events_detected",
                        f"frame_{frame_id}",
                        {
                            "num_events": len(events),
                            "event_types": [e.event_type.value for e in events],
                            "max_severity": max([e.severity.value for e in events]) if events else None
                        }
                    )
            
            # Create evidence items for significant findings
            evidence_items = self._create_evidence_items(
                frame_results['events'], frame_id, timestamp, camera_id
            )
            self.evidence_items.extend(evidence_items)
            frame_results['evidence'] = evidence_items
            
        except Exception as e:
            logger.error(f"Error analyzing frame {frame_id}: {e}")
            self.custody_manager.log_action(
                "analysis_error",
                f"frame_{frame_id}",
                {"error": str(e)}
            )
        
        return frame_results
    
    def _create_evidence_items(self, events: List[DetectedEvent], frame_id: int,
                             timestamp: float, camera_id: str) -> List[ForensicEvidence]:
        """Create evidence items from detected events"""
        evidence_items = []
        
        for event in events:
            # Only create evidence for significant events
            if event.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]:
                evidence_id = str(uuid.uuid4())
                
                raw_data = {
                    'event_data': asdict(event),
                    'frame_id': frame_id,
                    'timestamp': timestamp,
                    'camera_id': camera_id
                }
                
                # Calculate hash signature
                data_string = json.dumps(raw_data, sort_keys=True)
                hash_signature = hashlib.sha256(data_string.encode()).hexdigest()
                
                evidence = ForensicEvidence(
                    evidence_id=evidence_id,
                    evidence_type="detected_event",
                    timestamp=timestamp,
                    frame_id=frame_id,
                    camera_id=camera_id,
                    location=event.location,
                    confidence=event.confidence,
                    description=event.description,
                    raw_data=raw_data,
                    chain_of_custody=[],
                    hash_signature=hash_signature,
                    metadata={
                        'event_type': event.event_type.value,
                        'severity': event.severity.value,
                        'analysis_id': self.analysis_id
                    }
                )
                
                evidence_items.append(evidence)
                
                # Log evidence creation
                self.custody_manager.log_action(
                    "evidence_created",
                    evidence_id,
                    {
                        "event_type": event.event_type.value,
                        "severity": event.severity.value,
                        "confidence": event.confidence
                    }
                )
        
        return evidence_items
    
    def finalize_analysis(self) -> ForensicAnalysisResult:
        """Finalize the forensic analysis and generate results"""
        
        # Log analysis completion
        self.custody_manager.log_action(
            "analysis_completed",
            self.analysis_id,
            {
                "total_frames": len(self.scene_graphs),
                "total_events": len(self.detected_events),
                "total_evidence": len(self.evidence_items)
            }
        )
        
        # Generate analysis summary
        analysis_summary = self._generate_analysis_summary()
        
        # Create analysis result
        result = ForensicAnalysisResult(
            analysis_id=self.analysis_id,
            case_id=self.config.case_id,
            timestamp=datetime.now(),
            operator_id=self.config.operator_id,
            scene_graphs=self.scene_graphs,
            detected_events=self.detected_events,
            evidence_items=self.evidence_items,
            analysis_summary=analysis_summary,
            quality_metrics={},
            chain_of_custody_log=self.custody_manager.get_custody_log(),
            metadata={
                'config': asdict(self.config),
                'analysis_duration': 'calculated_in_export'
            }
        )
        
        # Assess quality
        result.quality_metrics = self.quality_assurance.assess_quality(result)
        
        # Log quality assessment
        self.custody_manager.log_action(
            "quality_assessment_completed",
            self.analysis_id,
            {"overall_quality_score": result.quality_metrics.get('overall_score', 0.0)}
        )
        
        return result
    
    def _generate_analysis_summary(self) -> Dict[str, Any]:
        """Generate comprehensive analysis summary"""
        
        # Event statistics
        event_stats = defaultdict(int)
        severity_stats = defaultdict(int)
        
        for event in self.detected_events:
            event_stats[event.event_type.value] += 1
            severity_stats[event.severity.value] += 1
        
        # Timeline analysis
        timeline_stats = {}
        if self.detected_events:
            timeline_stats = {
                'first_event': min(event.start_time for event in self.detected_events),
                'last_event': max(event.end_time for event in self.detected_events),
                'total_duration': max(event.end_time for event in self.detected_events) - 
                                min(event.start_time for event in self.detected_events),
                'event_density': len(self.detected_events) / 
                               (max(event.end_time for event in self.detected_events) - 
                                min(event.start_time for event in self.detected_events) + 1)
            }
        
        # Object analysis
        object_stats = defaultdict(int)
        for graph in self.scene_graphs:
            for obj in graph.objects:
                object_stats[obj.object_type] += 1
        
        # Critical findings
        critical_findings = [
            event for event in self.detected_events
            if event.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]
        ]
        
        return {
            'total_frames_analyzed': len(self.scene_graphs),
            'total_events_detected': len(self.detected_events),
            'total_evidence_items': len(self.evidence_items),
            'event_type_distribution': dict(event_stats),
            'severity_distribution': dict(severity_stats),
            'object_type_distribution': dict(object_stats),
            'timeline_analysis': timeline_stats,
            'critical_findings_count': len(critical_findings),
            'high_confidence_events': len([e for e in self.detected_events if e.confidence > 0.8]),
            'analysis_completeness': {
                'scene_graphs_generated': len(self.scene_graphs),
                'events_detected': len(self.detected_events),
                'evidence_preserved': len(self.evidence_items),
                'chain_of_custody_entries': len(self.custody_manager.get_custody_log())
            }
        }
    
    def export_results(self, output_dir: str, formats: List[str] = None) -> Dict[str, str]:
        """Export analysis results in specified formats"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        formats = formats or self.config.export_formats
        exported_files = {}
        
        # Finalize analysis if not done
        if not hasattr(self, '_finalized_result'):
            self._finalized_result = self.finalize_analysis()
        
        result = self._finalized_result
        
        try:
            # Export JSON format
            if 'json' in formats:
                json_file = output_dir / f"forensic_analysis_{self.analysis_id}.json"
                self._export_json(result, json_file)
                exported_files['json'] = str(json_file)
            
            # Export forensic report
            if 'report' in formats:
                report_file = output_dir / f"forensic_report_{self.analysis_id}.json"
                self._export_forensic_report(result, report_file)
                exported_files['report'] = str(report_file)
            
            # Export timeline visualization
            if 'timeline' in formats:
                timeline_file = output_dir / f"event_timeline_{self.analysis_id}.png"
                self._export_timeline_visualization(result, timeline_file)
                exported_files['timeline'] = str(timeline_file)
            
            # Export scene graph visualization
            if 'scene_graph' in formats and self.config.enable_scene_graphs:
                sg_file = output_dir / f"scene_graph_{self.analysis_id}.png"
                self.scene_graph_generator.export_scene_graph(sg_file, format='visualization')
                exported_files['scene_graph'] = str(sg_file)
            
            # Log export completion
            self.custody_manager.log_action(
                "results_exported",
                self.analysis_id,
                {
                    "formats": formats,
                    "output_directory": str(output_dir),
                    "exported_files": list(exported_files.keys())
                }
            )
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            self.custody_manager.log_action(
                "export_error",
                self.analysis_id,
                {"error": str(e)}
            )
        
        return exported_files
    
    def _export_json(self, result: ForensicAnalysisResult, output_path: Path):
        """Export complete analysis as JSON"""
        export_data = {
            'analysis_id': result.analysis_id,
            'case_id': result.case_id,
            'timestamp': result.timestamp.isoformat(),
            'operator_id': result.operator_id,
            'analysis_summary': result.analysis_summary,
            'quality_metrics': result.quality_metrics,
            'scene_graphs': [
                {
                    'scene_id': sg.scene_id,
                    'frame_id': sg.frame_id,
                    'timestamp': sg.timestamp,
                    'objects': [asdict(obj) for obj in sg.objects],
                    'relations': [
                        {
                            'subject_id': rel.subject_id,
                            'object_id': rel.object_id,
                            'relation_type': rel.relation_type.value,
                            'confidence': rel.confidence,
                            'frame_id': rel.frame_id,
                            'timestamp': rel.timestamp,
                            'spatial_context': rel.spatial_context,
                            'temporal_context': rel.temporal_context
                        }
                        for rel in sg.relations
                    ],
                    'metadata': sg.metadata
                }
                for sg in result.scene_graphs
            ],
            'detected_events': [
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
                for event in result.detected_events
            ],
            'evidence_items': [asdict(evidence) for evidence in result.evidence_items],
            'chain_of_custody_log': result.chain_of_custody_log,
            'metadata': result.metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Complete analysis exported to {output_path}")
    
    def _export_forensic_report(self, result: ForensicAnalysisResult, output_path: Path):
        """Export forensic report"""
        report = {
            'forensic_analysis_report': {
                'case_information': {
                    'case_id': result.case_id,
                    'analysis_id': result.analysis_id,
                    'analysis_date': result.timestamp.isoformat(),
                    'operator_id': result.operator_id,
                    'analysis_type': self.config.analysis_type
                },
                'executive_summary': {
                    'total_frames_analyzed': result.analysis_summary['total_frames_analyzed'],
                    'total_events_detected': result.analysis_summary['total_events_detected'],
                    'critical_findings': result.analysis_summary['critical_findings_count'],
                    'overall_quality_score': result.quality_metrics.get('overall_score', 0.0),
                    'analysis_completeness': result.analysis_summary['analysis_completeness']
                },
                'critical_findings': [
                    {
                        'event_id': event.event_id,
                        'event_type': event.event_type.value,
                        'severity': event.severity.value,
                        'timestamp': event.start_time,
                        'location': event.location,
                        'description': event.description,
                        'confidence': event.confidence,
                        'evidence_hash': next(
                            (e.hash_signature for e in result.evidence_items 
                             if e.raw_data.get('event_data', {}).get('event_id') == event.event_id),
                            None
                        )
                    }
                    for event in result.detected_events
                    if event.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]
                ],
                'timeline_analysis': result.analysis_summary.get('timeline_analysis', {}),
                'quality_assessment': result.quality_metrics,
                'chain_of_custody_summary': {
                    'total_entries': len(result.chain_of_custody_log),
                    'integrity_verified': result.quality_metrics.get('evidence_integrity', 0.0) > 0.9,
                    'completeness_score': result.quality_metrics.get('chain_of_custody_completeness', 0.0)
                },
                'recommendations': [
                    "Review all critical findings for potential criminal activity",
                    "Cross-reference detected events with external evidence sources",
                    "Validate high-confidence automated detections with human analysis",
                    "Preserve all evidence items with verified chain of custody",
                    "Consider additional analysis if quality scores are below threshold"
                ],
                'technical_details': {
                    'analysis_configuration': asdict(self.config),
                    'quality_metrics_breakdown': result.quality_metrics,
                    'evidence_integrity_status': 'VERIFIED' if result.quality_metrics.get('evidence_integrity', 0.0) > 0.9 else 'NEEDS_REVIEW'
                }
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Forensic report exported to {output_path}")
    
    def _export_timeline_visualization(self, result: ForensicAnalysisResult, output_path: Path):
        """Export timeline visualization"""
        if self.config.enable_event_detection and result.detected_events:
            self.event_detector.export_events(str(output_path), format='timeline')
        else:
            logger.warning("No events to visualize in timeline")

# Example usage and testing
if __name__ == "__main__":
    # Create sample configuration
    config = ForensicAnalysisConfig(
        case_id="CASE_2024_001",
        operator_id="forensic_analyst_001",
        analysis_type="comprehensive",
        confidence_threshold=0.7
    )
    
    # Initialize forensic analyzer
    analyzer = ForensicSceneAnalyzer(config)
    
    # Create sample objects
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
        )
    ]
    
    # Analyze frame
    frame_result = analyzer.analyze_frame(
        objects=sample_objects,
        frame_id=1,
        timestamp=0.0,
        camera_id="cam_001"
    )
    
    # Finalize and export results
    analysis_result = analyzer.finalize_analysis()
    exported_files = analyzer.export_results("./forensic_output")
    
    print("Forensic Analysis Complete:")
    print(f"  Analysis ID: {analysis_result.analysis_id}")
    print(f"  Events Detected: {len(analysis_result.detected_events)}")
    print(f"  Evidence Items: {len(analysis_result.evidence_items)}")
    print(f"  Quality Score: {analysis_result.quality_metrics.get('overall_score', 0.0):.2f}")
    print(f"  Exported Files: {list(exported_files.keys())}")
