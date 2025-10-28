"""
Django REST API views for 3D scene visualization data.
Provides endpoints for fetching scene objects, trajectories, events, and reconstruction data.
"""

import json
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

@login_required
@require_http_methods(["GET"])
def get_scene_data(request, scene_id):
    """
    Get comprehensive 3D scene data including objects, trajectories, events, and metadata.
    """
    try:
        # Mock scene data for demonstration
        scene_data = {
            "scene_id": scene_id,
            "metadata": {
                "name": f"Scene {scene_id.upper()}",
                "description": "Forensic scene reconstruction with multi-camera analysis",
                "created_at": "2024-01-15T10:30:00Z",
                "duration": 60.0,
                "fps": 30,
                "cameras": 4,
                "resolution": "1920x1080",
                "coordinate_system": "world",
                "units": "meters"
            },
            "bounds": {
                "min": [-10, -10, -2],
                "max": [10, 10, 5],
                "center": [0, 0, 1.5]
            },
            "objects": [
                {
                    "id": "person_001",
                    "type": "person",
                    "label": "Suspect A",
                    "confidence": 0.95,
                    "color": "#ff4757",
                    "visible": True,
                    "bbox": {
                        "x": -2.5, "y": 1.0, "z": 0.0,
                        "width": 0.6, "height": 1.8, "depth": 0.4
                    },
                    "attributes": {
                        "age_estimate": "25-35",
                        "gender": "male",
                        "clothing": "dark jacket, jeans",
                        "behavior": "suspicious"
                    }
                },
                {
                    "id": "person_002", 
                    "type": "person",
                    "label": "Witness B",
                    "confidence": 0.88,
                    "color": "#2ed573",
                    "visible": True,
                    "bbox": {
                        "x": 3.2, "y": -1.5, "z": 0.0,
                        "width": 0.5, "height": 1.7, "depth": 0.4
                    },
                    "attributes": {
                        "age_estimate": "40-50",
                        "gender": "female", 
                        "clothing": "red coat, black pants",
                        "behavior": "normal"
                    }
                },
                {
                    "id": "vehicle_001",
                    "type": "vehicle",
                    "label": "Sedan Car",
                    "confidence": 0.92,
                    "color": "#1e90ff",
                    "visible": True,
                    "bbox": {
                        "x": -5.0, "y": -3.0, "z": 0.0,
                        "width": 4.5, "height": 1.5, "depth": 2.0
                    },
                    "attributes": {
                        "make": "Toyota",
                        "model": "Camry",
                        "color": "silver",
                        "license": "ABC-123"
                    }
                }
            ],
            "trajectories": [
                {
                    "object_id": "person_001",
                    "points": [
                        {"time": 0.0, "x": -5.0, "y": 2.0, "z": 0.0},
                        {"time": 10.0, "x": -3.0, "y": 1.8, "z": 0.0},
                        {"time": 20.0, "x": -1.0, "y": 1.5, "z": 0.0},
                        {"time": 30.0, "x": 1.0, "y": 1.2, "z": 0.0},
                        {"time": 40.0, "x": 3.0, "y": 1.0, "z": 0.0},
                        {"time": 50.0, "x": 5.0, "y": 0.8, "z": 0.0},
                        {"time": 60.0, "x": 7.0, "y": 0.5, "z": 0.0}
                    ],
                    "interpolation": "cubic",
                    "confidence": 0.89
                },
                {
                    "object_id": "person_002",
                    "points": [
                        {"time": 0.0, "x": 4.0, "y": -2.0, "z": 0.0},
                        {"time": 15.0, "x": 3.5, "y": -1.8, "z": 0.0},
                        {"time": 30.0, "x": 3.2, "y": -1.5, "z": 0.0},
                        {"time": 45.0, "x": 3.0, "y": -1.2, "z": 0.0},
                        {"time": 60.0, "x": 2.8, "y": -1.0, "z": 0.0}
                    ],
                    "interpolation": "linear",
                    "confidence": 0.76
                },
                {
                    "object_id": "vehicle_001",
                    "points": [
                        {"time": 0.0, "x": -8.0, "y": -3.0, "z": 0.0},
                        {"time": 5.0, "x": -6.0, "y": -3.0, "z": 0.0},
                        {"time": 10.0, "x": -5.0, "y": -3.0, "z": 0.0},
                        {"time": 50.0, "x": -5.0, "y": -3.0, "z": 0.0},
                        {"time": 55.0, "x": -3.0, "y": -3.0, "z": 0.0},
                        {"time": 60.0, "x": -1.0, "y": -3.0, "z": 0.0}
                    ],
                    "interpolation": "linear",
                    "confidence": 0.94
                }
            ],
            "events": [
                {
                    "id": "event_001",
                    "type": "interaction",
                    "title": "Suspect Approach",
                    "description": "Suspect A approaches the scene from the west",
                    "time": 15.0,
                    "duration": 5.0,
                    "position": {"x": -3.0, "y": 1.8, "z": 0.0},
                    "severity": "medium",
                    "objects": ["person_001"],
                    "confidence": 0.87
                },
                {
                    "id": "event_002", 
                    "type": "anomaly",
                    "title": "Suspicious Behavior",
                    "description": "Suspect A exhibits unusual movement pattern",
                    "time": 25.0,
                    "duration": 8.0,
                    "position": {"x": 0.0, "y": 1.3, "z": 0.0},
                    "severity": "high",
                    "objects": ["person_001"],
                    "confidence": 0.92
                },
                {
                    "id": "event_003",
                    "type": "interaction",
                    "title": "Vehicle Departure",
                    "description": "Vehicle leaves the scene area",
                    "time": 52.0,
                    "duration": 8.0,
                    "position": {"x": -4.0, "y": -3.0, "z": 0.0},
                    "severity": "low",
                    "objects": ["vehicle_001"],
                    "confidence": 0.89
                }
            ],
            "reconstruction": {
                "available": True,
                "type": "nerf",
                "quality": "high",
                "mesh_url": "/media/reconstructions/sample_scene.glb",
                "point_cloud_url": "/media/reconstructions/sample_points.ply",
                "texture_resolution": 2048,
                "vertex_count": 125000,
                "face_count": 250000
            }
        }
        
        return JsonResponse({
            "success": True,
            "data": scene_data
        })
        
    except Exception as e:
        logger.error(f"Error fetching scene data for {scene_id}: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Failed to load scene data: {str(e)}"
        }, status=500)

@login_required  
@require_http_methods(["GET"])
def get_scene_objects(request, scene_id):
    """
    Get detailed object data for a specific scene.
    """
    try:
        # Extract query parameters
        time_filter = request.GET.get('time', None)
        object_type = request.GET.get('type', None)
        
        # Mock object data with filtering
        objects = [
            {
                "id": "person_001",
                "type": "person", 
                "label": "Suspect A",
                "confidence": 0.95,
                "position": {"x": -2.5, "y": 1.0, "z": 0.0},
                "rotation": {"x": 0, "y": 45, "z": 0},
                "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                "visible": True,
                "properties": {
                    "height": 1.8,
                    "age_estimate": "25-35",
                    "clothing": "dark jacket, jeans",
                    "behavior_score": 0.75
                }
            },
            {
                "id": "person_002",
                "type": "person",
                "label": "Witness B", 
                "confidence": 0.88,
                "position": {"x": 3.2, "y": -1.5, "z": 0.0},
                "rotation": {"x": 0, "y": -30, "z": 0},
                "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                "visible": True,
                "properties": {
                    "height": 1.7,
                    "age_estimate": "40-50",
                    "clothing": "red coat, black pants",
                    "behavior_score": 0.25
                }
            }
        ]
        
        # Apply filters
        if object_type:
            objects = [obj for obj in objects if obj['type'] == object_type]
            
        return JsonResponse({
            "success": True,
            "objects": objects,
            "count": len(objects)
        })
        
    except Exception as e:
        logger.error(f"Error fetching objects for scene {scene_id}: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Failed to load objects: {str(e)}"
        }, status=500)

@login_required
@require_http_methods(["GET"])
def get_scene_trajectories(request, scene_id):
    """
    Get trajectory data for scene objects.
    """
    try:
        start_time = float(request.GET.get('start_time', 0))
        end_time = float(request.GET.get('end_time', 60))
        
        trajectories = [
            {
                "object_id": "person_001",
                "type": "person",
                "keyframes": [
                    {"time": 0.0, "position": {"x": -5.0, "y": 2.0, "z": 0.0}, "rotation": {"x": 0, "y": 90, "z": 0}},
                    {"time": 15.0, "position": {"x": -3.0, "y": 1.8, "z": 0.0}, "rotation": {"x": 0, "y": 45, "z": 0}},
                    {"time": 30.0, "position": {"x": 0.0, "y": 1.3, "z": 0.0}, "rotation": {"x": 0, "y": 0, "z": 0}},
                    {"time": 45.0, "position": {"x": 3.0, "y": 1.0, "z": 0.0}, "rotation": {"x": 0, "y": -45, "z": 0}},
                    {"time": 60.0, "position": {"x": 7.0, "y": 0.5, "z": 0.0}, "rotation": {"x": 0, "y": -90, "z": 0}}
                ],
                "interpolation": "cubic",
                "confidence": 0.89,
                "color": "#ff4757"
            },
            {
                "object_id": "person_002", 
                "type": "person",
                "keyframes": [
                    {"time": 0.0, "position": {"x": 4.0, "y": -2.0, "z": 0.0}, "rotation": {"x": 0, "y": 180, "z": 0}},
                    {"time": 30.0, "position": {"x": 3.2, "y": -1.5, "z": 0.0}, "rotation": {"x": 0, "y": 150, "z": 0}},
                    {"time": 60.0, "position": {"x": 2.8, "y": -1.0, "z": 0.0}, "rotation": {"x": 0, "y": 120, "z": 0}}
                ],
                "interpolation": "linear",
                "confidence": 0.76,
                "color": "#2ed573"
            }
        ]
        
        # Filter by time range
        filtered_trajectories = []
        for traj in trajectories:
            filtered_keyframes = [
                kf for kf in traj['keyframes'] 
                if start_time <= kf['time'] <= end_time
            ]
            if filtered_keyframes:
                traj_copy = traj.copy()
                traj_copy['keyframes'] = filtered_keyframes
                filtered_trajectories.append(traj_copy)
        
        return JsonResponse({
            "success": True,
            "trajectories": filtered_trajectories,
            "time_range": {"start": start_time, "end": end_time}
        })
        
    except Exception as e:
        logger.error(f"Error fetching trajectories for scene {scene_id}: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Failed to load trajectories: {str(e)}"
        }, status=500)

@login_required
@require_http_methods(["GET"])
def get_scene_events(request, scene_id):
    """
    Get event data for scene timeline.
    """
    try:
        event_type = request.GET.get('type', None)
        
        events = [
            {
                "id": "event_001",
                "type": "interaction",
                "title": "Suspect Approach",
                "description": "Suspect A approaches the scene from the west side",
                "time": 15.0,
                "duration": 5.0,
                "position": {"x": -3.0, "y": 1.8, "z": 0.0},
                "severity": "medium",
                "confidence": 0.87,
                "objects": ["person_001"],
                "color": "#ffa502"
            },
            {
                "id": "event_002",
                "type": "anomaly", 
                "title": "Suspicious Behavior",
                "description": "Suspect A exhibits unusual movement pattern and loitering",
                "time": 25.0,
                "duration": 8.0,
                "position": {"x": 0.0, "y": 1.3, "z": 0.0},
                "severity": "high",
                "confidence": 0.92,
                "objects": ["person_001"],
                "color": "#ff4757"
            },
            {
                "id": "event_003",
                "type": "interaction",
                "title": "Vehicle Departure", 
                "description": "Vehicle leaves the scene area at high speed",
                "time": 52.0,
                "duration": 8.0,
                "position": {"x": -4.0, "y": -3.0, "z": 0.0},
                "severity": "low",
                "confidence": 0.89,
                "objects": ["vehicle_001"],
                "color": "#1e90ff"
            }
        ]
        
        # Filter by event type
        if event_type:
            events = [event for event in events if event['type'] == event_type]
            
        return JsonResponse({
            "success": True,
            "events": events,
            "count": len(events)
        })
        
    except Exception as e:
        logger.error(f"Error fetching events for scene {scene_id}: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Failed to load events: {str(e)}"
        }, status=500)

@login_required
@require_http_methods(["GET"])
def get_scene_reconstruction(request, scene_id):
    """
    Get 3D reconstruction data and mesh information.
    """
    try:
        reconstruction_data = {
            "available": True,
            "type": "nerf",
            "quality": "high",
            "created_at": "2024-01-15T12:00:00Z",
            "processing_time": 3600,
            "mesh": {
                "url": "/media/reconstructions/sample_scene.glb",
                "format": "glb",
                "size_mb": 15.2,
                "vertex_count": 125000,
                "face_count": 250000,
                "texture_resolution": 2048
            },
            "point_cloud": {
                "url": "/media/reconstructions/sample_points.ply", 
                "format": "ply",
                "size_mb": 8.7,
                "point_count": 500000,
                "color_channels": ["rgb", "normal"]
            },
            "cameras": [
                {
                    "id": "cam_001",
                    "position": {"x": -5.0, "y": 0.0, "z": 2.0},
                    "rotation": {"x": -10, "y": 45, "z": 0},
                    "fov": 60,
                    "resolution": "1920x1080"
                },
                {
                    "id": "cam_002", 
                    "position": {"x": 5.0, "y": 0.0, "z": 2.0},
                    "rotation": {"x": -10, "y": -45, "z": 0},
                    "fov": 60,
                    "resolution": "1920x1080"
                }
            ],
            "bounds": {
                "min": [-10, -10, -2],
                "max": [10, 10, 5],
                "center": [0, 0, 1.5]
            },
            "metadata": {
                "algorithm": "NeRF",
                "training_images": 120,
                "training_iterations": 50000,
                "psnr": 28.5,
                "ssim": 0.89
            }
        }
        
        return JsonResponse({
            "success": True,
            "reconstruction": reconstruction_data
        })
        
    except Exception as e:
        logger.error(f"Error fetching reconstruction for scene {scene_id}: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Failed to load reconstruction: {str(e)}"
        }, status=500)
