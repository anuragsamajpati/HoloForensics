"""
Django REST API views for Scene Graph and Event Detection
"""

import json
import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.utils.decorators import method_decorator
from django.views import View
import numpy as np

# Import our forensic analysis modules
import sys
sys.path.append('/Users/anuragsamajpati/Desktop/holoforensics/cv')

from forensic_scene_analysis import (
    ForensicSceneAnalyzer, 
    ForensicAnalysisConfig,
    ForensicAnalysisResult
)
from scene_graph_generation import SceneObject
from event_detection import EventType, EventSeverity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Storage for active analysis jobs
active_jobs = {}

class SceneAnalysisJobManager:
    """Manages scene analysis jobs"""
    
    def __init__(self):
        self.jobs = {}
    
    def create_job(self, job_data: Dict[str, Any]) -> str:
        """Create a new scene analysis job"""
        job_id = str(uuid.uuid4())
        
        # Create forensic analysis config
        config = ForensicAnalysisConfig(
            case_id=job_data.get('case_id', f'case_{job_id[:8]}'),
            operator_id=job_data.get('operator_id', 'web_user'),
            analysis_type=job_data.get('analysis_type', 'comprehensive'),
            enable_scene_graphs=job_data.get('enable_scene_graphs', True),
            enable_event_detection=job_data.get('enable_event_detection', True),
            confidence_threshold=job_data.get('confidence_threshold', 0.6),
            temporal_analysis_window=job_data.get('temporal_window', 30),
            spatial_proximity_threshold=job_data.get('proximity_threshold', 100.0)
        )
        
        # Initialize analyzer
        analyzer = ForensicSceneAnalyzer(config)
        
        # Store job
        self.jobs[job_id] = {
            'job_id': job_id,
            'status': 'initialized',
            'created_at': datetime.now(),
            'config': config,
            'analyzer': analyzer,
            'progress': 0,
            'total_frames': 0,
            'processed_frames': 0,
            'results': None,
            'error': None,
            'logs': []
        }
        
        return job_id
    
    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job information"""
        return self.jobs.get(job_id)
    
    def update_job_status(self, job_id: str, status: str, **kwargs):
        """Update job status and metadata"""
        if job_id in self.jobs:
            self.jobs[job_id]['status'] = status
            self.jobs[job_id]['updated_at'] = datetime.now()
            
            for key, value in kwargs.items():
                self.jobs[job_id][key] = value
    
    def add_job_log(self, job_id: str, message: str, level: str = 'info'):
        """Add log entry to job"""
        if job_id in self.jobs:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message
            }
            self.jobs[job_id]['logs'].append(log_entry)
    
    def process_frame_data(self, job_id: str, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single frame of data"""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        analyzer = job['analyzer']
        
        try:
            # Convert frame data to SceneObject instances
            objects = []
            for obj_data in frame_data.get('objects', []):
                scene_obj = SceneObject(
                    object_id=obj_data.get('object_id', str(uuid.uuid4())),
                    object_type=obj_data.get('object_type', 'unknown'),
                    confidence=obj_data.get('confidence', 0.5),
                    bbox=tuple(obj_data.get('bbox', [0, 0, 100, 100])),
                    center=tuple(obj_data.get('center', [50, 50])),
                    area=obj_data.get('area', 2500),
                    frame_id=frame_data.get('frame_id', 0),
                    timestamp=frame_data.get('timestamp', 0.0),
                    camera_id=frame_data.get('camera_id', 'default'),
                    metadata=obj_data.get('metadata', {})
                )
                objects.append(scene_obj)
            
            # Analyze frame
            frame_result = analyzer.analyze_frame(
                objects=objects,
                frame_id=frame_data.get('frame_id', 0),
                timestamp=frame_data.get('timestamp', 0.0),
                camera_id=frame_data.get('camera_id', 'default')
            )
            
            # Update job progress
            job['processed_frames'] += 1
            if job['total_frames'] > 0:
                job['progress'] = (job['processed_frames'] / job['total_frames']) * 100
            
            self.add_job_log(job_id, f"Processed frame {frame_data.get('frame_id', 0)}")
            
            return {
                'frame_id': frame_result['frame_id'],
                'events_detected': len(frame_result['events']),
                'evidence_created': len(frame_result['evidence']),
                'scene_graph_objects': len(frame_result['scene_graph'].objects) if frame_result['scene_graph'] else 0,
                'scene_graph_relations': len(frame_result['scene_graph'].relations) if frame_result['scene_graph'] else 0
            }
            
        except Exception as e:
            self.add_job_log(job_id, f"Error processing frame: {str(e)}", 'error')
            raise e
    
    def finalize_job(self, job_id: str) -> ForensicAnalysisResult:
        """Finalize analysis job and generate results"""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        analyzer = job['analyzer']
        
        try:
            # Finalize analysis
            results = analyzer.finalize_analysis()
            
            # Export results
            output_dir = os.path.join(settings.MEDIA_ROOT, 'scene_analysis', job_id)
            exported_files = analyzer.export_results(output_dir)
            
            # Update job
            self.update_job_status(
                job_id, 
                'completed',
                results=results,
                exported_files=exported_files,
                progress=100
            )
            
            self.add_job_log(job_id, "Analysis completed successfully")
            
            return results
            
        except Exception as e:
            self.update_job_status(job_id, 'failed', error=str(e))
            self.add_job_log(job_id, f"Analysis failed: {str(e)}", 'error')
            raise e

# Global job manager
job_manager = SceneAnalysisJobManager()

@method_decorator([login_required, csrf_exempt], name='dispatch')
class StartSceneAnalysisView(View):
    """Start a new scene analysis job"""
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            
            # Validate required fields
            required_fields = ['case_id', 'analysis_type']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return JsonResponse({
                    'success': False,
                    'error': f'Missing required fields: {missing_fields}'
                }, status=400)
            
            # Add operator ID from authenticated user
            data['operator_id'] = request.user.username
            
            # Create job
            job_id = job_manager.create_job(data)
            
            # Set total frames if provided
            if 'total_frames' in data:
                job_manager.update_job_status(job_id, 'initialized', total_frames=data['total_frames'])
            
            logger.info(f"Started scene analysis job {job_id} for user {request.user.username}")
            
            return JsonResponse({
                'success': True,
                'job_id': job_id,
                'status': 'initialized',
                'message': 'Scene analysis job created successfully'
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON data'
            }, status=400)
        except Exception as e:
            logger.error(f"Error starting scene analysis: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

@method_decorator([login_required, csrf_exempt], name='dispatch')
class ProcessFrameView(View):
    """Process a single frame of scene data"""
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            
            # Validate required fields
            job_id = data.get('job_id')
            frame_data = data.get('frame_data')
            
            if not job_id or not frame_data:
                return JsonResponse({
                    'success': False,
                    'error': 'job_id and frame_data are required'
                }, status=400)
            
            # Check if job exists
            job = job_manager.get_job(job_id)
            if not job:
                return JsonResponse({
                    'success': False,
                    'error': f'Job {job_id} not found'
                }, status=404)
            
            # Process frame
            frame_result = job_manager.process_frame_data(job_id, frame_data)
            
            return JsonResponse({
                'success': True,
                'job_id': job_id,
                'frame_result': frame_result,
                'progress': job['progress']
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON data'
            }, status=400)
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

@method_decorator([login_required], name='dispatch')
class SceneAnalysisStatusView(View):
    """Get status of scene analysis job"""
    
    def get(self, request, job_id):
        try:
            job = job_manager.get_job(job_id)
            if not job:
                return JsonResponse({
                    'success': False,
                    'error': f'Job {job_id} not found'
                }, status=404)
            
            # Prepare response data
            response_data = {
                'success': True,
                'job_id': job_id,
                'status': job['status'],
                'progress': job['progress'],
                'processed_frames': job['processed_frames'],
                'total_frames': job['total_frames'],
                'created_at': job['created_at'].isoformat(),
                'logs': job['logs'][-10:]  # Last 10 log entries
            }
            
            # Add results if completed
            if job['status'] == 'completed' and job['results']:
                results = job['results']
                response_data['results_summary'] = {
                    'analysis_id': results.analysis_id,
                    'total_events': len(results.detected_events),
                    'total_evidence': len(results.evidence_items),
                    'total_scene_graphs': len(results.scene_graphs),
                    'quality_score': results.quality_metrics.get('overall_score', 0.0),
                    'critical_events': len([
                        e for e in results.detected_events 
                        if e.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]
                    ])
                }
                
                if 'exported_files' in job:
                    response_data['exported_files'] = job['exported_files']
            
            # Add error if failed
            if job['status'] == 'failed' and job['error']:
                response_data['error'] = job['error']
            
            return JsonResponse(response_data)
            
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

@method_decorator([login_required, csrf_exempt], name='dispatch')
class FinalizeSceneAnalysisView(View):
    """Finalize scene analysis and generate results"""
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            job_id = data.get('job_id')
            
            if not job_id:
                return JsonResponse({
                    'success': False,
                    'error': 'job_id is required'
                }, status=400)
            
            # Check if job exists
            job = job_manager.get_job(job_id)
            if not job:
                return JsonResponse({
                    'success': False,
                    'error': f'Job {job_id} not found'
                }, status=404)
            
            # Finalize analysis
            results = job_manager.finalize_job(job_id)
            
            # Prepare response
            response_data = {
                'success': True,
                'job_id': job_id,
                'analysis_id': results.analysis_id,
                'status': 'completed',
                'results_summary': {
                    'total_events': len(results.detected_events),
                    'total_evidence': len(results.evidence_items),
                    'total_scene_graphs': len(results.scene_graphs),
                    'quality_score': results.quality_metrics.get('overall_score', 0.0),
                    'analysis_summary': results.analysis_summary
                },
                'exported_files': job.get('exported_files', {}),
                'message': 'Scene analysis completed successfully'
            }
            
            return JsonResponse(response_data)
            
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON data'
            }, status=400)
        except Exception as e:
            logger.error(f"Error finalizing analysis: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

@method_decorator([login_required], name='dispatch')
class SceneAnalysisResultsView(View):
    """Get detailed results of completed scene analysis"""
    
    def get(self, request, job_id):
        try:
            job = job_manager.get_job(job_id)
            if not job:
                return JsonResponse({
                    'success': False,
                    'error': f'Job {job_id} not found'
                }, status=404)
            
            if job['status'] != 'completed':
                return JsonResponse({
                    'success': False,
                    'error': f'Job {job_id} is not completed (status: {job["status"]})'
                }, status=400)
            
            results = job['results']
            if not results:
                return JsonResponse({
                    'success': False,
                    'error': 'No results available'
                }, status=404)
            
            # Prepare detailed results
            response_data = {
                'success': True,
                'job_id': job_id,
                'analysis_id': results.analysis_id,
                'case_id': results.case_id,
                'timestamp': results.timestamp.isoformat(),
                'operator_id': results.operator_id,
                'analysis_summary': results.analysis_summary,
                'quality_metrics': results.quality_metrics,
                'detected_events': [
                    {
                        'event_id': event.event_id,
                        'event_type': event.event_type.value,
                        'severity': event.severity.value,
                        'confidence': event.confidence,
                        'start_time': event.start_time,
                        'end_time': event.end_time,
                        'location': event.location,
                        'description': event.description,
                        'involved_objects': event.involved_objects
                    }
                    for event in results.detected_events
                ],
                'evidence_summary': [
                    {
                        'evidence_id': evidence.evidence_id,
                        'evidence_type': evidence.evidence_type,
                        'timestamp': evidence.timestamp,
                        'confidence': evidence.confidence,
                        'description': evidence.description,
                        'location': evidence.location
                    }
                    for evidence in results.evidence_items
                ],
                'scene_graph_summary': [
                    {
                        'frame_id': sg.frame_id,
                        'timestamp': sg.timestamp,
                        'num_objects': len(sg.objects),
                        'num_relations': len(sg.relations),
                        'object_types': list(set(obj.object_type for obj in sg.objects))
                    }
                    for sg in results.scene_graphs
                ],
                'exported_files': job.get('exported_files', {})
            }
            
            return JsonResponse(response_data)
            
        except Exception as e:
            logger.error(f"Error getting analysis results: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

@method_decorator([login_required, csrf_exempt], name='dispatch')
class ValidateSceneDataView(View):
    """Validate scene data format before processing"""
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            scene_data = data.get('scene_data')
            
            if not scene_data:
                return JsonResponse({
                    'success': False,
                    'error': 'scene_data is required'
                }, status=400)
            
            validation_results = {
                'success': True,
                'valid': True,
                'issues': [],
                'statistics': {
                    'total_frames': 0,
                    'total_objects': 0,
                    'object_types': set(),
                    'time_range': [float('inf'), float('-inf')]
                }
            }
            
            # Validate scene data structure
            if not isinstance(scene_data, list):
                validation_results['valid'] = False
                validation_results['issues'].append('scene_data must be a list of frames')
                return JsonResponse(validation_results, status=400)
            
            for i, frame in enumerate(scene_data):
                # Check required frame fields
                required_fields = ['frame_id', 'timestamp', 'objects']
                missing_fields = [field for field in required_fields if field not in frame]
                if missing_fields:
                    validation_results['issues'].append(
                        f'Frame {i}: missing fields {missing_fields}'
                    )
                    validation_results['valid'] = False
                    continue
                
                # Update statistics
                validation_results['statistics']['total_frames'] += 1
                validation_results['statistics']['total_objects'] += len(frame.get('objects', []))
                
                timestamp = frame.get('timestamp', 0)
                validation_results['statistics']['time_range'][0] = min(
                    validation_results['statistics']['time_range'][0], timestamp
                )
                validation_results['statistics']['time_range'][1] = max(
                    validation_results['statistics']['time_range'][1], timestamp
                )
                
                # Validate objects
                for j, obj in enumerate(frame.get('objects', [])):
                    obj_required_fields = ['object_id', 'object_type', 'bbox', 'confidence']
                    obj_missing_fields = [field for field in obj_required_fields if field not in obj]
                    if obj_missing_fields:
                        validation_results['issues'].append(
                            f'Frame {i}, Object {j}: missing fields {obj_missing_fields}'
                        )
                        validation_results['valid'] = False
                    
                    # Check bbox format
                    bbox = obj.get('bbox', [])
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        validation_results['issues'].append(
                            f'Frame {i}, Object {j}: bbox must be [x1, y1, x2, y2]'
                        )
                        validation_results['valid'] = False
                    
                    # Check confidence range
                    confidence = obj.get('confidence', 0)
                    if not (0 <= confidence <= 1):
                        validation_results['issues'].append(
                            f'Frame {i}, Object {j}: confidence must be between 0 and 1'
                        )
                        validation_results['valid'] = False
                    
                    # Collect object types
                    validation_results['statistics']['object_types'].add(obj.get('object_type', 'unknown'))
            
            # Convert set to list for JSON serialization
            validation_results['statistics']['object_types'] = list(
                validation_results['statistics']['object_types']
            )
            
            # Fix infinite values
            if validation_results['statistics']['time_range'][0] == float('inf'):
                validation_results['statistics']['time_range'] = [0, 0]
            
            return JsonResponse(validation_results)
            
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON data'
            }, status=400)
        except Exception as e:
            logger.error(f"Error validating scene data: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

# URL patterns will be added to urls.py
