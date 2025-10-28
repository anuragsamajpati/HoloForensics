from django.urls import path
from . import views, scene_3d_views, case_management_views, export_views, simple_analysis_views

urlpatterns = [
    path('upload-scene/', views.upload_scene, name='upload_scene'),
    path('status/<str:scene_id>/', views.get_processing_status, name='scene_status'),
    path('scene/<str:scene_id>/results/', views.get_scene_results, name='scene_results'),
    path('scene/<str:scene_id>/download/', views.download_scene_results, name='download_scene_results'),
    path('scene/<str:scene_id>/file/', views.download_scene_file, name='download_scene_file'),
    path('scene/<str:scene_id>/report/', views.scene_report, name='scene_report'),
    path('scenes/list/', views.list_processed_scenes, name='list_processed_scenes'),
    
    # Simple Analysis Tools (working without complex dependencies)
    path('analysis/object-detection/', simple_analysis_views.start_object_detection, name='start_object_detection'),
    path('analysis/3d-reconstruction/', simple_analysis_views.start_scene_reconstruction, name='start_scene_reconstruction'),
    path('analysis/video-inpainting/', simple_analysis_views.start_video_inpainting, name='start_video_inpainting'),
    path('analysis/physics-prediction/', simple_analysis_views.start_physics_prediction, name='start_physics_prediction'),
    path('analysis/status/<str:job_id>/', simple_analysis_views.get_analysis_status, name='get_analysis_status'),
    path('analysis/jobs/', simple_analysis_views.list_analysis_jobs, name='list_analysis_jobs'),
    
    # 3D Scene Viewer API endpoints
    path('3d/scene/<str:scene_id>/', scene_3d_views.get_scene_data, name='get_scene_data'),
    path('3d/scene/<str:scene_id>/objects/', scene_3d_views.get_scene_objects, name='get_scene_objects'),
    path('3d/scene/<str:scene_id>/trajectories/', scene_3d_views.get_scene_trajectories, name='get_scene_trajectories'),
    path('3d/scene/<str:scene_id>/events/', scene_3d_views.get_scene_events, name='get_scene_events'),
    path('3d/scene/<str:scene_id>/reconstruction/', scene_3d_views.get_scene_reconstruction, name='get_scene_reconstruction'),
    
    # Case Management API endpoints
    path('cases/create/', case_management_views.create_case, name='create_case'),
    path('cases/list/', case_management_views.list_cases, name='list_cases'),
    path('cases/<str:case_id>/', case_management_views.get_case_details, name='get_case_details'),
    path('cases/<str:case_id>/update/', case_management_views.update_case, name='update_case'),
    path('cases/<str:case_id>/comment/', case_management_views.add_case_comment, name='add_case_comment'),
    path('cases/statistics/', case_management_views.get_case_statistics, name='get_case_statistics'),
    
    # Export and Reporting API endpoints
    path('export/case/<str:case_id>/', export_views.export_case_report, name='export_case_report'),
    path('export/dashboard/', export_views.export_dashboard_report, name='export_dashboard_report'),
]
