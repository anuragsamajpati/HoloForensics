from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.conf import settings
import json
import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any
import uuid
from datetime import datetime

# Add CV module to path
cv_path = Path(__file__).parent.parent.parent.parent / 'cv'
sys.path.append(str(cv_path))

try:
    from forensic_inpainting_integration import ForensicInpaintingIntegration, ForensicInpaintingConfig
    from video_inpainting import InpaintingConfig
except ImportError as e:
    logging.error(f"Failed to import inpainting modules: {e}")
    ForensicInpaintingIntegration = None

logger = logging.getLogger(__name__)

@csrf_exempt
@require_http_methods(["POST"])
@login_required
def start_inpainting_job(request):
    """Start video inpainting job for a forensic scene"""
    try:
        data = json.loads(request.body)
        
        # Validate required parameters
        required_fields = ['scene_id', 'case_id']
        for field in required_fields:
            if field not in data:
                return JsonResponse({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }, status=400)
        
        scene_id = data['scene_id']
        case_id = data['case_id']
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Get scene path
        scene_path = Path(settings.MEDIA_ROOT) / 'videos' / scene_id
        if not scene_path.exists():
            return JsonResponse({
                'success': False,
                'error': f'Scene not found: {scene_id}'
            }, status=404)
        
        # Create output directory
        output_dir = Path(settings.MEDIA_ROOT) / 'inpainted' / scene_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup forensic configuration
        config = ForensicInpaintingConfig(
            evidence_preservation=data.get('evidence_preservation', True),
            quality_validation=data.get('quality_validation', True),
            chain_of_custody=data.get('chain_of_custody', True),
            backup_originals=data.get('backup_originals', True)
        )
        
        # Initialize forensic inpainting system
        if ForensicInpaintingIntegration is None:
            return JsonResponse({
                'success': False,
                'error': 'Inpainting system not available'
            }, status=500)
        
        forensic_system = ForensicInpaintingIntegration(
            case_id=case_id,
            operator=request.user.username,
            config=config
        )
        
        # Store job metadata
        job_metadata = {
            'job_id': job_id,
            'scene_id': scene_id,
            'case_id': case_id,
            'operator': request.user.username,
            'status': 'started',
            'created_at': datetime.now().isoformat(),
            'scene_path': str(scene_path),
            'output_dir': str(output_dir),
            'config': config.__dict__
        }
        
        # Save job metadata
        job_file = output_dir / f"job_{job_id}.json"
        with open(job_file, 'w') as f:
            json.dump(job_metadata, f, indent=2)
        
        # Start processing (this should be async in production)
        try:
            processing_report = forensic_system.process_forensic_scene(
                str(scene_path), str(output_dir)
            )
            
            # Update job status
            job_metadata['status'] = 'completed'
            job_metadata['completed_at'] = datetime.now().isoformat()
            job_metadata['processing_report'] = processing_report
            
            with open(job_file, 'w') as f:
                json.dump(job_metadata, f, indent=2)
            
            return JsonResponse({
                'success': True,
                'job_id': job_id,
                'status': 'completed',
                'processing_report': processing_report
            })
            
        except Exception as e:
            # Update job status with error
            job_metadata['status'] = 'failed'
            job_metadata['error'] = str(e)
            job_metadata['failed_at'] = datetime.now().isoformat()
            
            with open(job_file, 'w') as f:
                json.dump(job_metadata, f, indent=2)
            
            logger.error(f"Inpainting job {job_id} failed: {e}")
            
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
        logger.error(f"Inpainting API error: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@require_http_methods(["GET"])
@login_required
def get_inpainting_job_status(request, job_id):
    """Get status of inpainting job"""
    try:
        # Find job file
        inpainted_dir = Path(settings.MEDIA_ROOT) / 'inpainted'
        job_file = None
        
        for scene_dir in inpainted_dir.iterdir():
            if scene_dir.is_dir():
                potential_job_file = scene_dir / f"job_{job_id}.json"
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
        logger.error(f"Error getting job status: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@require_http_methods(["GET"])
@login_required
def list_inpainting_jobs(request):
    """List all inpainting jobs for current user"""
    try:
        inpainted_dir = Path(settings.MEDIA_ROOT) / 'inpainted'
        
        if not inpainted_dir.exists():
            return JsonResponse({
                'success': True,
                'jobs': []
            })
        
        jobs = []
        
        for scene_dir in inpainted_dir.iterdir():
            if scene_dir.is_dir():
                for job_file in scene_dir.glob("job_*.json"):
                    try:
                        with open(job_file, 'r') as f:
                            job_data = json.load(f)
                        
                        # Filter by current user
                        if job_data.get('operator') == request.user.username:
                            jobs.append({
                                'job_id': job_data['job_id'],
                                'scene_id': job_data['scene_id'],
                                'case_id': job_data['case_id'],
                                'status': job_data['status'],
                                'created_at': job_data['created_at'],
                                'completed_at': job_data.get('completed_at'),
                                'error': job_data.get('error')
                            })
                    except Exception as e:
                        logger.warning(f"Error reading job file {job_file}: {e}")
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x['created_at'], reverse=True)
        
        return JsonResponse({
            'success': True,
            'jobs': jobs
        })
        
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@require_http_methods(["GET"])
@login_required
def get_inpainting_results(request, job_id):
    """Get results and files from completed inpainting job"""
    try:
        # Find job file
        inpainted_dir = Path(settings.MEDIA_ROOT) / 'inpainted'
        job_file = None
        scene_dir = None
        
        for sd in inpainted_dir.iterdir():
            if sd.is_dir():
                potential_job_file = sd / f"job_{job_id}.json"
                if potential_job_file.exists():
                    job_file = potential_job_file
                    scene_dir = sd
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
        
        # Get processed files
        processed_files = []
        evidence_files = []
        
        if scene_dir:
            # Processed videos
            processed_dir = scene_dir / 'processed'
            if processed_dir.exists():
                for video_file in processed_dir.glob("*.mp4"):
                    processed_files.append({
                        'filename': video_file.name,
                        'path': str(video_file.relative_to(Path(settings.MEDIA_ROOT))),
                        'size': video_file.stat().st_size,
                        'type': 'processed_video'
                    })
            
            # Evidence files
            evidence_dir = scene_dir / 'evidence'
            if evidence_dir.exists():
                for evidence_file in evidence_dir.glob("*.json"):
                    evidence_files.append({
                        'filename': evidence_file.name,
                        'path': str(evidence_file.relative_to(Path(settings.MEDIA_ROOT))),
                        'size': evidence_file.stat().st_size,
                        'type': 'evidence_report'
                    })
        
        return JsonResponse({
            'success': True,
            'job_data': job_data,
            'processed_files': processed_files,
            'evidence_files': evidence_files
        })
        
    except Exception as e:
        logger.error(f"Error getting inpainting results: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
@login_required
def validate_scene_for_inpainting(request):
    """Validate if a scene is ready for inpainting"""
    try:
        data = json.loads(request.body)
        scene_id = data.get('scene_id')
        
        if not scene_id:
            return JsonResponse({
                'success': False,
                'error': 'Missing scene_id'
            }, status=400)
        
        # Check if scene exists
        scene_path = Path(settings.MEDIA_ROOT) / 'videos' / scene_id
        
        if not scene_path.exists():
            return JsonResponse({
                'success': False,
                'error': f'Scene not found: {scene_id}',
                'ready_for_inpainting': False
            })
        
        # Find video files
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.MOV']:
            video_files.extend(scene_path.glob(f"*{ext}"))
        
        if not video_files:
            return JsonResponse({
                'success': True,
                'ready_for_inpainting': False,
                'error': 'No video files found in scene',
                'scene_info': {
                    'scene_id': scene_id,
                    'video_count': 0
                }
            })
        
        # Analyze video files
        scene_info = {
            'scene_id': scene_id,
            'video_count': len(video_files),
            'videos': []
        }
        
        total_size = 0
        
        for video_file in video_files:
            file_size = video_file.stat().st_size
            total_size += file_size
            
            scene_info['videos'].append({
                'filename': video_file.name,
                'size_mb': round(file_size / (1024 * 1024), 2),
                'camera_id': video_file.stem
            })
        
        scene_info['total_size_mb'] = round(total_size / (1024 * 1024), 2)
        
        # Check if scene is ready for inpainting
        ready_for_inpainting = (
            len(video_files) > 0 and
            total_size > 0 and
            total_size < 10 * 1024 * 1024 * 1024  # Less than 10GB
        )
        
        return JsonResponse({
            'success': True,
            'ready_for_inpainting': ready_for_inpainting,
            'scene_info': scene_info
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        logger.error(f"Scene validation error: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
