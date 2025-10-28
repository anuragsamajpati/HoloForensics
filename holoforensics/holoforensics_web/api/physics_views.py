import json
import os
import logging
from datetime import datetime
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.conf import settings
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import sys
import uuid
from datetime import datetime
import numpy as np

# Add CV module to path
cv_path = Path(__file__).parent.parent.parent.parent / 'cv'
sys.path.append(str(cv_path))

try:
    from forensic_physics_integration import (
        ForensicTrajectoryAnalyzer, ForensicPredictionConfig, 
        PredictionResult, ForensicScenario
    )
    from physics_prediction import ObjectTrajectory, TrajectoryPoint
except ImportError as e:
    logging.error(f"Failed to import physics modules: {e}")
    ForensicTrajectoryAnalyzer = None

logger = logging.getLogger(__name__)

@csrf_exempt
@require_http_methods(["POST"])
@login_required
def start_physics_prediction(request):
    """Start physics-informed trajectory prediction"""
    try:
        data = json.loads(request.body)
        
        # Validate required parameters
        required_fields = ['case_id', 'trajectory_data']
        for field in required_fields:
            if field not in data:
                return JsonResponse({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }, status=400)
        
        case_id = data['case_id']
        trajectory_data = data['trajectory_data']
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Parse configuration
        config_data = data.get('config', {})
        config = ForensicPredictionConfig(
            prediction_horizon=config_data.get('prediction_horizon', 60),
            confidence_threshold=config_data.get('confidence_threshold', 0.7),
            kalman_weight=config_data.get('kalman_weight', 0.6),
            scene_calibration=config_data.get('scene_calibration', True),
            evidence_preservation=config_data.get('evidence_preservation', True),
            uncertainty_quantification=config_data.get('uncertainty_quantification', True)
        )
        
        # Initialize analyzer
        if ForensicTrajectoryAnalyzer is None:
            return JsonResponse({
                'success': False,
                'error': 'Physics prediction system not available'
            }, status=500)
        
        analyzer = ForensicTrajectoryAnalyzer(config)
        
        # Load or create scenario
        scenario_data = data.get('scenario')
        if scenario_data:
            scenario_path = _create_scenario_file(scenario_data, case_id)
            analyzer.load_forensic_scenario(scenario_path)
        else:
            # Use default scenario
            scenario_path = _create_default_scenario(case_id)
            analyzer.load_forensic_scenario(scenario_path)
        
        # Parse trajectory data
        trajectories = []
        for traj_data in trajectory_data:
            trajectory = _parse_trajectory_data(traj_data)
            if trajectory:
                trajectories.append(trajectory)
        
        if not trajectories:
            return JsonResponse({
                'success': False,
                'error': 'No valid trajectories provided'
            }, status=400)
        
        # Store job metadata
        job_metadata = {
            'job_id': job_id,
            'case_id': case_id,
            'operator': request.user.username,
            'status': 'started',
            'created_at': datetime.now().isoformat(),
            'config': config_data,
            'trajectory_count': len(trajectories)
        }
        
        # Create output directory
        output_dir = Path(settings.MEDIA_ROOT) / 'physics_predictions' / case_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save job metadata
        job_file = output_dir / f"job_{job_id}.json"
        with open(job_file, 'w') as f:
            json.dump(job_metadata, f, indent=2)
        
        try:
            # Process trajectories
            results = []
            investigation_context = data.get('context', {})
            
            for trajectory in trajectories:
                result = analyzer.predict_suspect_trajectory(trajectory, investigation_context)
                results.append(result)
            
            # Generate forensic report
            report_path = analyzer.generate_forensic_report(
                results, case_id, str(output_dir)
            )
            
            # Update job status
            job_metadata['status'] = 'completed'
            job_metadata['completed_at'] = datetime.now().isoformat()
            job_metadata['results_count'] = len(results)
            job_metadata['report_path'] = str(report_path)
            
            with open(job_file, 'w') as f:
                json.dump(job_metadata, f, indent=2)
            
            # Prepare response data
            response_results = []
            for result in results:
                response_results.append({
                    'object_id': result.object_id,
                    'prediction_count': len(result.predicted_positions),
                    'avg_confidence': float(np.mean(result.confidence_scores)) if result.confidence_scores else 0.0,
                    'method_used': result.method_used,
                    'accuracy_metrics': result.accuracy_metrics
                })
            
            return JsonResponse({
                'success': True,
                'job_id': job_id,
                'status': 'completed',
                'results': response_results,
                'report_path': str(report_path)
            })
            
        except Exception as e:
            # Update job status with error
            job_metadata['status'] = 'failed'
            job_metadata['error'] = str(e)
            job_metadata['failed_at'] = datetime.now().isoformat()
            
            with open(job_file, 'w') as f:
                json.dump(job_metadata, f, indent=2)
            
            logger.error(f"Physics prediction job {job_id} failed: {e}")
            
            return JsonResponse({
                'success': False,
                'job_id': job_id,
                'error': str(e)
            }, status=500)
            
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        logger.error(f"Physics prediction API error: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@require_http_methods(["GET"])
@login_required
def get_physics_prediction_status(request, job_id):
    """Get status of physics prediction job"""
    try:
        # Find job file
        predictions_dir = Path(settings.MEDIA_ROOT) / 'physics_predictions'
        job_file = None
        
        for case_dir in predictions_dir.iterdir():
            if case_dir.is_dir():
                potential_job_file = case_dir / f"job_{job_id}.json"
                if potential_job_file.exists():
                    job_file = potential_job_file
                    break
        
        if not job_file:
            return JsonResponse({
                'success': False,
                'error': f'Job not found: {job_id}'
            }, status=404)
        
        # Load job metadata
        with open(job_file, 'r') as f:
            job_data = json.load(f)
        
        return JsonResponse({
            'success': True,
            'job_data': job_data
        })
        
    except Exception as e:
        logger.error(f"Error getting physics prediction status: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@require_http_methods(["GET"])
@login_required
def get_physics_prediction_results(request, job_id):
    """Get detailed results from physics prediction job"""
    try:
        # Find job file and results
        predictions_dir = Path(settings.MEDIA_ROOT) / 'physics_predictions'
        job_file = None
        case_dir = None
        
        for cd in predictions_dir.iterdir():
            if cd.is_dir():
                potential_job_file = cd / f"job_{job_id}.json"
                if potential_job_file.exists():
                    job_file = potential_job_file
                    case_dir = cd
                    break
        
        if not job_file:
            return JsonResponse({
                'success': False,
                'error': f'Job not found: {job_id}'
            }, status=404)
        
        # Load job metadata
        with open(job_file, 'r') as f:
            job_data = json.load(f)
        
        # Check if job belongs to current user
        if job_data.get('operator') != request.user.username:
            return JsonResponse({
                'success': False,
                'error': 'Access denied'
            }, status=403)
        
        # Load detailed results
        results = []
        prediction_files = []
        
        if case_dir:
            # Load individual object prediction files
            for result_file in case_dir.glob("object_*_predictions.json"):
                try:
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                        results.append(result_data)
                        
                    prediction_files.append({
                        'filename': result_file.name,
                        'path': str(result_file.relative_to(Path(settings.MEDIA_ROOT))),
                        'size': result_file.stat().st_size,
                        'type': 'prediction_result'
                    })
                except Exception as e:
                    logger.warning(f"Error loading result file {result_file}: {e}")
            
            # Load main report
            report_files = list(case_dir.glob("forensic_prediction_report_*.json"))
            for report_file in report_files:
                prediction_files.append({
                    'filename': report_file.name,
                    'path': str(report_file.relative_to(Path(settings.MEDIA_ROOT))),
                    'size': report_file.stat().st_size,
                    'type': 'forensic_report'
                })
        
        return JsonResponse({
            'success': True,
            'job_data': job_data,
            'prediction_results': results,
            'files': prediction_files
        })
        
    except Exception as e:
        logger.error(f"Error getting physics prediction results: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
@login_required
def validate_trajectory_data(request):
    """Validate trajectory data for physics prediction"""
    try:
        data = json.loads(request.body)
        trajectory_data = data.get('trajectory_data', [])
        
        validation_results = []
        
        for i, traj_data in enumerate(trajectory_data):
            result = {
                'trajectory_index': i,
                'valid': True,
                'issues': [],
                'object_id': traj_data.get('object_id'),
                'point_count': 0
            }
            
            # Check required fields
            if 'object_id' not in traj_data:
                result['valid'] = False
                result['issues'].append('Missing object_id')
            
            if 'points' not in traj_data:
                result['valid'] = False
                result['issues'].append('Missing trajectory points')
            else:
                points = traj_data['points']
                result['point_count'] = len(points)
                
                if len(points) < 3:
                    result['valid'] = False
                    result['issues'].append('Insufficient trajectory points (minimum 3 required)')
                
                # Validate point structure
                for j, point in enumerate(points):
                    if not all(key in point for key in ['x', 'y', 'timestamp']):
                        result['valid'] = False
                        result['issues'].append(f'Point {j} missing required fields (x, y, timestamp)')
                        break
            
            validation_results.append(result)
        
        # Overall validation summary
        valid_trajectories = sum(1 for r in validation_results if r['valid'])
        total_trajectories = len(validation_results)
        
        return JsonResponse({
            'success': True,
            'validation_results': validation_results,
            'summary': {
                'total_trajectories': total_trajectories,
                'valid_trajectories': valid_trajectories,
                'validation_passed': valid_trajectories == total_trajectories
            }
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        logger.error(f"Trajectory validation error: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def _parse_trajectory_data(traj_data: Dict[str, Any]) -> Optional[ObjectTrajectory]:
    """Parse trajectory data from API request"""
    try:
        object_id = traj_data['object_id']
        points_data = traj_data['points']
        
        points = []
        for point_data in points_data:
            point = TrajectoryPoint(
                x=float(point_data['x']),
                y=float(point_data['y']),
                timestamp=float(point_data['timestamp']),
                velocity_x=float(point_data.get('velocity_x', 0.0)),
                velocity_y=float(point_data.get('velocity_y', 0.0)),
                confidence=float(point_data.get('confidence', 1.0))
            )
            points.append(point)
        
        trajectory = ObjectTrajectory(
            object_id=object_id,
            points=points,
            object_type=traj_data.get('object_type', 'person'),
            start_frame=traj_data.get('start_frame', 0),
            end_frame=traj_data.get('end_frame', len(points)-1)
        )
        
        return trajectory
        
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Error parsing trajectory data: {e}")
        return None

def _create_scenario_file(scenario_data: Dict[str, Any], case_id: str) -> str:
    """Create scenario file from API data"""
    scenario_dir = Path(settings.MEDIA_ROOT) / 'scenarios'
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    scenario_file = scenario_dir / f"scenario_{case_id}.json"
    
    with open(scenario_file, 'w') as f:
        json.dump(scenario_data, f, indent=2)
    
    return str(scenario_file)

def _create_default_scenario(case_id: str) -> str:
    """Create default forensic scenario"""
    default_scenario = {
        'scenario_id': f'default_{case_id}',
        'scene_dimensions': [20.0, 15.0],
        'obstacles': [],
        'entry_points': [[0.0, 7.5], [20.0, 7.5]],
        'exit_points': [[0.0, 7.5], [20.0, 7.5], [10.0, 0.0], [10.0, 15.0]],
        'high_interest_zones': [],
        'timestamp': datetime.now().timestamp()
    }
    
    return _create_scenario_file(default_scenario, case_id)
