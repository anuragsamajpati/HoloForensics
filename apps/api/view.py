# apps/api/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
import json
import os
import uuid
from celery import shared_task
from .models import Scene, Camera, ProcessingJob
from cv.inpaint_e2fgvi import VideoInpainter
from cv.colmap_pipeline import run_colmap_pipeline
import logging

logger = logging.getLogger(__name__)

@csrf_exempt
def upload_scene(request):
    """Handle multi-camera video upload"""
    if request.method == 'POST':
        try:
            scene_name = request.POST.get('scene_name', 'Untitled Scene')
            description = request.POST.get('description', '')
            
            # Create new scene
            scene_id = str(uuid.uuid4())
            scene = Scene.objects.create(
                scene_id=scene_id,
                name=scene_name,
                description=description,
                status='uploading'
            )
            
            # Process uploaded videos
            uploaded_files = []
            for key, file in request.FILES.items():
                if key.startswith('video_'):
                    camera_id = key.replace('video_', '')
                    
                    # Save video file
                    file_path = f'scenes/{scene_id}/raw/{camera_id}.mp4'
                    saved_path = default_storage.save(file_path, file)
                    
                    # Create camera record
                    camera = Camera.objects.create(
                        scene=scene,
                        camera_id=camera_id,
                        video_file=saved_path
                    )
                    
                    uploaded_files.append({
                        'camera_id': camera_id,
                        'file_path': saved_path,
                        'file_size': file.size
                    })
            
            # Update scene status and trigger processing
            scene.status = 'processing'
            scene.save()
            
            # Start background processing
            process_scene_pipeline.delay(scene_id)
            
            return JsonResponse({
                'success': True,
                'scene_id': scene_id,
                'status': 'processing',
                'uploaded_files': uploaded_files,
                'message': f'Uploaded {len(uploaded_files)} camera files'
            })
            
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)
    
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)

@shared_task
def process_scene_pipeline(scene_id):
    """Background task for complete scene processing"""
    try:
        scene = Scene.objects.get(scene_id=scene_id)
        
        # Step 1: Extract keyframes and run COLMAP
        logger.info(f"Starting COLMAP for scene {scene_id}")
        scene.status = 'calibrating'
        scene.save()
        
        cameras = Camera.objects.filter(scene=scene)
        video_paths = [os.path.join(settings.MEDIA_ROOT, cam.video_file.name) 
                      for cam in cameras]
        
        # Run COLMAP pipeline
        colmap_result = run_colmap_pipeline(scene_id, video_paths)
        
        # Update camera intrinsics/extrinsics
        for cam in cameras:
            if cam.camera_id in colmap_result['cameras']:
                cam_data = colmap_result['cameras'][cam.camera_id]
                cam.intrinsics = cam_data['K']
                cam.extrinsics = {'R': cam_data['R'], 't': cam_data['t']}
                cam.calibration_error = cam_data['reprojection_error']
                cam.save()
        
        # Step 2: Object detection and tracking
        logger.info(f"Starting object detection for scene {scene_id}")
        scene.status = 'detecting'
        scene.save()
        
        # Run detection pipeline
        detection_results = run_detection_pipeline(scene_id, video_paths)
        
        # Step 3: Multi-view association and 3D tracking
        logger.info(f"Starting 3D tracking for scene {scene_id}")
        scene.status = 'tracking'
        scene.save()
        
        tracking_results = run_tracking_pipeline(scene_id, detection_results, colmap_result)
        
        # Step 4: Video inpainting for gaps
        logger.info(f"Starting inpainting for scene {scene_id}")
        scene.status = 'inpainting'
        scene.save()
        
        inpainting_results = run_inpainting_pipeline(scene_id, video_paths, tracking_results)
        
        # Step 5: Event detection and scene graph
        logger.info(f"Generating scene graph for scene {scene_id}")
        scene.status = 'analyzing'
        scene.save()
        
        scene_graph = generate_scene_graph(scene_id, tracking_results)
        
        # Mark as complete
        scene.status = 'ready'
        scene.save()
        
        logger.info(f"Scene {scene_id} processing completed successfully")
        
    except Exception as e:
        logger.error(f"Processing failed for scene {scene_id}: {str(e)}")
        scene = Scene.objects.get(scene_id=scene_id)
        scene.status = 'error'
        scene.save()

def get_processing_status(request, scene_id):
    """Get current processing status"""
    try:
        scene = Scene.objects.get(scene_id=scene_id)
        
        # Calculate progress percentage
        progress_map = {
            'uploading': 10,
            'processing': 15,
            'calibrating': 25,
            'detecting': 45,
            'tracking': 65,
            'inpainting': 80,
            'analyzing': 90,
            'ready': 100,
            'error': 0
        }
        
        return JsonResponse({
            'scene_id': scene_id,
            'status': scene.status,
            'progress': progress_map.get(scene.status, 0),
            'name': scene.name,
            'created_at': scene.created_at.isoformat()
        })
        
    except Scene.DoesNotExist:
        return JsonResponse({'error': 'Scene not found'}, status=404)

# cv/inpaint_e2fgvi.py
import torch
import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import os

class VideoInpainter:
    """E2FGVI video inpainting integration"""
    
    def __init__(self, model_path='models/E2FGVI-HQ-CVPR22.pth'):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """Load E2FGVI model"""
        # This assumes E2FGVI is installed and available
        try:
            import sys
            sys.path.append('third_party/E2FGVI')
            from model.e2fgvi import InpaintGenerator
            
            self.model = InpaintGenerator()
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            self.model.to(self.device)
            
        except ImportError:
            print("E2FGVI not found. Using placeholder implementation.")
            self.model = None
    
    def create_guidance_masks(self, frames, track_3d, camera_pose):
        """Create masks around projected 3D track positions"""
        masks = []
        
        for t, frame in enumerate(frames):
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            
            if t < len(track_3d):
                # Project 3D position to 2D
                pos_3d = track_3d[t]['position']
                pos_2d = self.project_3d_to_2d(pos_3d, camera_pose)
                
                if pos_2d is not None:
                    # Create circular mask around projected position
                    cv2.circle(mask, tuple(map(int, pos_2d)), radius=50, color=255, thickness=-1)
            
            masks.append(mask)
        
        return masks
    
    def project_3d_to_2d(self, pos_3d, camera_pose):
        """Project 3D point to 2D using camera parameters"""
        try:
            K = np.array(camera_pose['K'])
            R = np.array(camera_pose['R'])
            t = np.array(camera_pose['t'])
            
            # Transform to camera coordinates
            pos_cam = R @ pos_3d + t
            
            # Project to image plane
            if pos_cam[2] > 0:  # Point in front of camera
                pos_2d = K @ pos_cam
                return pos_2d[:2] / pos_2d[2]
            
        except Exception as e:
            print(f"Projection error: {e}")
        
        return None
    
    def inpaint_sequence(self, frames, masks):
        """Inpaint video sequence using E2FGVI"""
        if self.model is None:
            # Placeholder: return original frames
            print("E2FGVI model not available. Returning original frames.")
            return frames
        
        try:
            with torch.no_grad():
                # Convert frames to tensors
                frame_tensors = []
                for frame in frames:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
                    frame_tensors.append(frame_tensor)
                
                # Convert masks to tensors
                mask_tensors = []
                for mask in masks:
                    mask_tensor = torch.from_numpy(mask).float() / 255.0
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
                    mask_tensors.append(mask_tensor)
                
                # Stack into batch
                frames_batch = torch.cat(frame_tensors, dim=0).to(self.device)
                masks_batch = torch.cat(mask_tensors, dim=0).to(self.device)
                
                # Run inpainting
                inpainted_batch = self.model(frames_batch, masks_batch)
                
                # Convert back to numpy
                inpainted_frames = []
                for i in range(inpainted_batch.shape[0]):
                    frame = inpainted_batch[i].cpu().numpy()
                    frame = (frame * 255.0).astype(np.uint8)
                    frame = np.transpose(frame, (1, 2, 0))
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    inpainted_frames.append(frame)
                
                return inpainted_frames
                
        except Exception as e:
            print(f"Inpainting error: {e}")
            return frames

def run_inpainting_pipeline(scene_id, video_paths, tracking_results):
    """Run inpainting pipeline for a scene"""
    inpainter = VideoInpainter()
    results = {}
    
    for camera_id, video_path in enumerate(video_paths):
        print(f"Processing inpainting for camera {camera_id}")
        
        # Load video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        # Get 3D tracks for this camera
        if camera_id in tracking_results:
            tracks_3d = tracking_results[camera_id]['tracks_3d']
            camera_pose = tracking_results[camera_id]['camera_pose']
            
            # Create guidance masks from 3D tracks
            masks = inpainter.create_guidance_masks(frames, tracks_3d, camera_pose)
            
            # Run inpainting
            inpainted_frames = inpainter.inpaint_sequence(frames, masks)
            
            # Save inpainted video
            output_path = f'data/inpainted/scene_{scene_id}/cam_{camera_id}_inpainted.mp4'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, 
                                (frames[0].shape[1], frames[0].shape[0]))
            
            for frame in inpainted_frames:
                out.write(frame)
            out.release()
            
            # Calculate quality metrics
            quality_metrics = calculate_inpainting_quality(frames, inpainted_frames, masks)
            
            results[camera_id] = {
                'inpainted_video': output_path,
                'quality_metrics': quality_metrics,
                'frames_processed': len(frames)
            }
    
    return results

def calculate_inpainting_quality(original, inpainted, masks):
    """Calculate PSNR, SSIM for inpainted regions"""
    from skimage.metrics import structural_similarity as ssim
    
    psnr_scores = []
    ssim_scores = []
    
    for i, (orig, inp, mask) in enumerate(zip(original, inpainted, masks)):
        if mask.sum() > 0:  # Only calculate where inpainting occurred
            # Calculate PSNR in masked region
            mse = np.mean((orig[mask > 0] - inp[mask > 0]) ** 2)
            if mse > 0:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                psnr_scores.append(psnr)
            
            # Calculate SSIM for whole frame
            orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            inp_gray = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
            ssim_score = ssim(orig_gray, inp_gray)
            ssim_scores.append(ssim_score)
    
    return {
        'psnr_mean': np.mean(psnr_scores) if psnr_scores else 0,
        'psnr_std': np.std(psnr_scores) if psnr_scores else 0,
        'ssim_mean': np.mean(ssim_scores) if ssim_scores else 0,
        'frames_inpainted': len(psnr_scores)
    }

# apps/api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('upload-scene/', views.upload_scene, name='upload_scene'),
    path('status/<str:scene_id>/', views.get_processing_status, name='scene_status'),
    path('scene/<str:scene_id>/results/', views.get_scene_results, name='scene_results'),
]