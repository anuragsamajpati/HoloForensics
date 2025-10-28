"""
Django REST API views for comprehensive case management system.
Handles case creation, updates, collaboration, and workflow management.
"""

import json
import os
from datetime import datetime, timedelta
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.conf import settings
from django.core.paginator import Paginator
from django.db.models import Q
import logging

logger = logging.getLogger(__name__)

@login_required
@require_http_methods(["POST"])
def create_case(request):
    """
    Create a new forensic case with metadata and initial configuration.
    """
    try:
        data = json.loads(request.body)
        
        # Validate required fields
        required_fields = ['title', 'description', 'case_type']
        for field in required_fields:
            if field not in data:
                return JsonResponse({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }, status=400)
        
        # Generate unique case ID
        case_id = f"CASE_{datetime.now().year}_{str(datetime.now().timestamp()).replace('.', '')[-6:]}"
        
        # Create case metadata
        case_data = {
            "id": case_id,
            "title": data['title'],
            "description": data['description'],
            "case_type": data['case_type'],
            "status": "created",
            "priority": data.get('priority', 'medium'),
            "created_by": request.user.username,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "assigned_to": data.get('assigned_to', [request.user.username]),
            "tags": data.get('tags', []),
            "evidence_files": [],
            "analysis_results": {},
            "timeline": [{
                "timestamp": datetime.now().isoformat(),
                "action": "case_created",
                "user": request.user.username,
                "details": f"Case '{data['title']}' created"
            }],
            "collaboration": {
                "active_users": [request.user.username],
                "comments": [],
                "shared_notes": ""
            },
            "workflow": {
                "current_stage": "evidence_collection",
                "stages": [
                    {"name": "evidence_collection", "status": "active", "started_at": datetime.now().isoformat()},
                    {"name": "analysis", "status": "pending"},
                    {"name": "reconstruction", "status": "pending"},
                    {"name": "review", "status": "pending"},
                    {"name": "completed", "status": "pending"}
                ]
            },
            "permissions": {
                "owner": request.user.username,
                "viewers": data.get('viewers', []),
                "editors": data.get('editors', []),
                "public": data.get('public', False)
            }
        }
        
        # Save case data (in production, this would go to a database)
        case_file_path = os.path.join(settings.MEDIA_ROOT, 'cases', f'{case_id}.json')
        os.makedirs(os.path.dirname(case_file_path), exist_ok=True)
        
        with open(case_file_path, 'w') as f:
            json.dump(case_data, f, indent=2)
        
        return JsonResponse({
            "success": True,
            "case": case_data,
            "message": f"Case {case_id} created successfully"
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            "success": False,
            "error": "Invalid JSON data"
        }, status=400)
    except Exception as e:
        logger.error(f"Error creating case: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Failed to create case: {str(e)}"
        }, status=500)

@login_required
@require_http_methods(["GET"])
def list_cases(request):
    """
    List cases with filtering, sorting, and pagination.
    """
    try:
        # Get query parameters
        page = int(request.GET.get('page', 1))
        per_page = int(request.GET.get('per_page', 20))
        status_filter = request.GET.get('status', '')
        priority_filter = request.GET.get('priority', '')
        search_query = request.GET.get('search', '')
        sort_by = request.GET.get('sort_by', 'created_at')
        sort_order = request.GET.get('sort_order', 'desc')
        
        # Mock case data for demonstration
        cases = [
            {
                "id": "CASE_2024_001",
                "title": "Indoor Incident Analysis",
                "description": "Multi-camera analysis of indoor incident with 4 camera angles",
                "status": "completed",
                "priority": "high",
                "created_by": "investigator1",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T16:45:00Z",
                "assigned_to": ["investigator1", "analyst2"],
                "case_type": "incident_analysis",
                "progress": 100,
                "objects_detected": 47,
                "confidence_score": 94.2,
                "tags": ["indoor", "multi-camera", "completed"]
            },
            {
                "id": "CASE_2024_002", 
                "title": "Outdoor Investigation",
                "description": "3D reconstruction of outdoor scene using COLMAP pipeline",
                "status": "processing",
                "priority": "medium",
                "created_by": "investigator2",
                "created_at": "2024-01-14T09:15:00Z",
                "updated_at": "2024-01-15T14:20:00Z",
                "assigned_to": ["investigator2"],
                "case_type": "scene_reconstruction",
                "progress": 65,
                "objects_detected": 23,
                "confidence_score": 87.6,
                "tags": ["outdoor", "3d-reconstruction", "in-progress"]
            },
            {
                "id": "CASE_2024_003",
                "title": "Evidence Timeline Analysis", 
                "description": "Timeline analysis and behavioral pattern recognition",
                "status": "ready",
                "priority": "high",
                "created_by": "analyst1",
                "created_at": "2024-01-13T14:00:00Z",
                "updated_at": "2024-01-15T11:30:00Z",
                "assigned_to": ["analyst1", "investigator1"],
                "case_type": "behavioral_analysis",
                "progress": 90,
                "objects_detected": 156,
                "confidence_score": 91.8,
                "tags": ["timeline", "behavior", "ready-for-review"]
            }
        ]
        
        # Apply filters
        filtered_cases = cases
        
        if status_filter:
            filtered_cases = [c for c in filtered_cases if c['status'] == status_filter]
        
        if priority_filter:
            filtered_cases = [c for c in filtered_cases if c['priority'] == priority_filter]
            
        if search_query:
            filtered_cases = [c for c in filtered_cases if 
                            search_query.lower() in c['title'].lower() or 
                            search_query.lower() in c['description'].lower()]
        
        # Apply sorting
        reverse_sort = sort_order == 'desc'
        if sort_by in ['created_at', 'updated_at']:
            filtered_cases.sort(key=lambda x: x[sort_by], reverse=reverse_sort)
        elif sort_by == 'title':
            filtered_cases.sort(key=lambda x: x['title'].lower(), reverse=reverse_sort)
        elif sort_by == 'priority':
            priority_order = {'low': 1, 'medium': 2, 'high': 3}
            filtered_cases.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=reverse_sort)
        
        # Pagination
        total_cases = len(filtered_cases)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_cases = filtered_cases[start_idx:end_idx]
        
        return JsonResponse({
            "success": True,
            "cases": paginated_cases,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total_cases,
                "pages": (total_cases + per_page - 1) // per_page
            },
            "filters": {
                "status": status_filter,
                "priority": priority_filter,
                "search": search_query
            }
        })
        
    except Exception as e:
        logger.error(f"Error listing cases: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Failed to list cases: {str(e)}"
        }, status=500)

@login_required
@require_http_methods(["GET"])
def get_case_details(request, case_id):
    """
    Get detailed information about a specific case.
    """
    try:
        # Mock detailed case data
        case_details = {
            "id": case_id,
            "title": "Indoor Incident Analysis",
            "description": "Multi-camera analysis of indoor incident with 4 camera angles and object tracking",
            "status": "completed",
            "priority": "high",
            "created_by": "investigator1",
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": "2024-01-15T16:45:00Z",
            "assigned_to": ["investigator1", "analyst2"],
            "case_type": "incident_analysis",
            "tags": ["indoor", "multi-camera", "completed"],
            "evidence_files": [
                {
                    "id": "evidence_001",
                    "name": "camera_01_footage.mp4",
                    "type": "video",
                    "size": "2.1 GB",
                    "uploaded_at": "2024-01-15T10:35:00Z",
                    "uploaded_by": "investigator1",
                    "status": "processed"
                },
                {
                    "id": "evidence_002", 
                    "name": "camera_02_footage.mp4",
                    "type": "video",
                    "size": "1.8 GB",
                    "uploaded_at": "2024-01-15T10:37:00Z",
                    "uploaded_by": "investigator1",
                    "status": "processed"
                }
            ],
            "analysis_results": {
                "objects_detected": 47,
                "trajectories_analyzed": 12,
                "events_identified": 8,
                "anomalies_found": 3,
                "confidence_score": 94.2,
                "processing_time": "2h 34m",
                "algorithms_used": ["YOLO", "DeepSORT", "Kalman Filter"]
            },
            "timeline": [
                {
                    "timestamp": "2024-01-15T10:30:00Z",
                    "action": "case_created",
                    "user": "investigator1",
                    "details": "Case created and initial evidence uploaded"
                },
                {
                    "timestamp": "2024-01-15T11:15:00Z",
                    "action": "analysis_started",
                    "user": "system",
                    "details": "Automated analysis pipeline initiated"
                },
                {
                    "timestamp": "2024-01-15T14:30:00Z",
                    "action": "analysis_completed",
                    "user": "system",
                    "details": "Analysis completed with 94.2% confidence"
                },
                {
                    "timestamp": "2024-01-15T16:45:00Z",
                    "action": "case_reviewed",
                    "user": "analyst2",
                    "details": "Case reviewed and marked as completed"
                }
            ],
            "collaboration": {
                "active_users": ["investigator1", "analyst2"],
                "comments": [
                    {
                        "id": "comment_001",
                        "user": "analyst2",
                        "timestamp": "2024-01-15T15:20:00Z",
                        "message": "Excellent detection accuracy. The trajectory analysis shows clear patterns.",
                        "type": "review"
                    }
                ],
                "shared_notes": "High-quality footage with good lighting conditions. All objects clearly identifiable."
            },
            "workflow": {
                "current_stage": "completed",
                "stages": [
                    {"name": "evidence_collection", "status": "completed", "started_at": "2024-01-15T10:30:00Z", "completed_at": "2024-01-15T11:00:00Z"},
                    {"name": "analysis", "status": "completed", "started_at": "2024-01-15T11:15:00Z", "completed_at": "2024-01-15T14:30:00Z"},
                    {"name": "reconstruction", "status": "completed", "started_at": "2024-01-15T14:30:00Z", "completed_at": "2024-01-15T15:45:00Z"},
                    {"name": "review", "status": "completed", "started_at": "2024-01-15T15:45:00Z", "completed_at": "2024-01-15T16:45:00Z"},
                    {"name": "completed", "status": "completed", "started_at": "2024-01-15T16:45:00Z", "completed_at": "2024-01-15T16:45:00Z"}
                ]
            },
            "permissions": {
                "owner": "investigator1",
                "viewers": ["supervisor1"],
                "editors": ["analyst2"],
                "public": False
            },
            "export_options": [
                {"format": "pdf", "name": "Comprehensive Report", "description": "Full case report with analysis results"},
                {"format": "json", "name": "Raw Data Export", "description": "Machine-readable data export"},
                {"format": "video", "name": "Annotated Footage", "description": "Video with detection overlays"},
                {"format": "3d", "name": "3D Scene Export", "description": "3D reconstruction files"}
            ]
        }
        
        return JsonResponse({
            "success": True,
            "case": case_details
        })
        
    except Exception as e:
        logger.error(f"Error getting case details for {case_id}: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Failed to get case details: {str(e)}"
        }, status=500)

@login_required
@require_http_methods(["PUT"])
def update_case(request, case_id):
    """
    Update case information and add timeline entries.
    """
    try:
        data = json.loads(request.body)
        
        # Mock update logic
        updated_fields = []
        
        if 'status' in data:
            updated_fields.append(f"Status changed to {data['status']}")
        
        if 'priority' in data:
            updated_fields.append(f"Priority changed to {data['priority']}")
            
        if 'assigned_to' in data:
            updated_fields.append(f"Assigned to {', '.join(data['assigned_to'])}")
            
        if 'tags' in data:
            updated_fields.append(f"Tags updated: {', '.join(data['tags'])}")
        
        # Add timeline entry
        timeline_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "case_updated",
            "user": request.user.username,
            "details": "; ".join(updated_fields) if updated_fields else "Case information updated"
        }
        
        return JsonResponse({
            "success": True,
            "message": f"Case {case_id} updated successfully",
            "timeline_entry": timeline_entry,
            "updated_fields": updated_fields
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            "success": False,
            "error": "Invalid JSON data"
        }, status=400)
    except Exception as e:
        logger.error(f"Error updating case {case_id}: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Failed to update case: {str(e)}"
        }, status=500)

@login_required
@require_http_methods(["POST"])
def add_case_comment(request, case_id):
    """
    Add a comment to a case for collaboration.
    """
    try:
        data = json.loads(request.body)
        
        if 'message' not in data:
            return JsonResponse({
                "success": False,
                "error": "Message is required"
            }, status=400)
        
        comment = {
            "id": f"comment_{datetime.now().timestamp()}",
            "user": request.user.username,
            "timestamp": datetime.now().isoformat(),
            "message": data['message'],
            "type": data.get('type', 'general')
        }
        
        return JsonResponse({
            "success": True,
            "comment": comment,
            "message": "Comment added successfully"
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            "success": False,
            "error": "Invalid JSON data"
        }, status=400)
    except Exception as e:
        logger.error(f"Error adding comment to case {case_id}: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Failed to add comment: {str(e)}"
        }, status=500)

@login_required
@require_http_methods(["GET"])
def get_case_statistics(request):
    """
    Get overall case statistics for dashboard.
    """
    try:
        stats = {
            "total_cases": 156,
            "active_cases": 12,
            "completed_cases": 134,
            "pending_review": 10,
            "cases_this_month": 23,
            "average_processing_time": "4.2 hours",
            "success_rate": 96.8,
            "top_case_types": [
                {"type": "incident_analysis", "count": 45},
                {"type": "scene_reconstruction", "count": 38},
                {"type": "behavioral_analysis", "count": 32},
                {"type": "evidence_timeline", "count": 28},
                {"type": "object_tracking", "count": 13}
            ],
            "monthly_trends": [
                {"month": "Jan 2024", "cases": 23, "success_rate": 96.8},
                {"month": "Dec 2023", "cases": 19, "success_rate": 95.2},
                {"month": "Nov 2023", "cases": 21, "success_rate": 97.1},
                {"month": "Oct 2023", "cases": 18, "success_rate": 94.8}
            ]
        }
        
        return JsonResponse({
            "success": True,
            "statistics": stats
        })
        
    except Exception as e:
        logger.error(f"Error getting case statistics: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Failed to get statistics: {str(e)}"
        }, status=500)
