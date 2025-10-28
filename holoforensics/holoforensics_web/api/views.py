import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.conf import settings
from datetime import datetime
from django.http import FileResponse, HttpResponse
import mimetypes
import io
import shutil
import uuid

@csrf_exempt
def upload_scene(request):
    """Handle video/image upload and create a simple scene record on disk.

    This endpoint accepts multiple files under the same key (e.g. 'files') and
    stores them in data/raw/<scene_id>/ preserving original filenames. It also
    creates a lightweight metadata JSON so that the existing status and results
    endpoints can surface something meaningful immediately.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)

    try:
        # Basic scene info
        scene_name = request.POST.get('scene_name', 'Untitled Scene')
        description = request.POST.get('description', '')
        analysis_type = (request.POST.get('analysis_type') or '').strip()

        # Generate scene_id and target directories in "Analysed Data/<scene_id>/"
        scene_id = request.POST.get('scene_id') or str(uuid.uuid4())
        analysed_base = os.path.join(settings.BASE_DIR, '..', '..', 'Analysed Data')
        scene_root = os.path.join(analysed_base, scene_id)
        raw_dir = os.path.join(scene_root, 'Raw Data')
        metadata_dir = os.path.join(scene_root, 'Metadata')
        detections_dir = os.path.join(scene_root, 'Object Detection')
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
        os.makedirs(detections_dir, exist_ok=True)

        # Accept files under common keys
        saved_files = []
        # Common patterns: 'files' list, or arbitrary keys in request.FILES
        files_list = request.FILES.getlist('files') or []
        if not files_list:
            # Fallback: iterate all files
            files_list = [f for _, f in request.FILES.items()]

        total_bytes = 0
        for f in files_list:
            safe_name = f.name
            dst_path = os.path.join(raw_dir, safe_name)
            # Stream save to disk
            with open(dst_path, 'wb') as out:
                for chunk in f.chunks():
                    out.write(chunk)
            total_bytes += getattr(f, 'size', 0)
            saved_files.append({
                'filename': safe_name,
                'path': os.path.relpath(dst_path, start=scene_root),
                'size': f.size
            })

        # Minimal metadata for status/results endpoints
        metadata = {
            'scene_id': scene_id,
            'name': scene_name,
            'description': description,
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'uploaded_files': saved_files,
            # Mark as processed=true so UI can immediately show something;
            # in real pipeline this would be updated asynchronously.
            'processing_stats': {
                'successful_cameras': len(saved_files),
                'failed_cameras': 0
            }
        }

        os.makedirs(analysed_base, exist_ok=True)
        meta_path = os.path.join(metadata_dir, 'scene_metadata.json')
        with open(meta_path, 'w') as mf:
            json.dump(metadata, mf, indent=2)

        # Create a placeholder detections file so results endpoint finds data
        placeholder = os.path.join(detections_dir, 'results.json')
        if not os.path.exists(placeholder):
            with open(placeholder, 'w') as pf:
                json.dump({'note': 'placeholder detections', 'files': [sf['filename'] for sf in saved_files]}, pf)

        # Create a quick preview: copy the first uploaded video into "Video Inpainting/preview.mp4"
        # Choose preview target folder label based on analysis type
        folder_map = {
            'object_detection': 'Object Detection',
            '3d_reconstruction': '3D Reconstruction',
            'timeline': 'Timeline Analysis',
            'video_inpainting': 'Video Inpainting',
            'physics_prediction': 'Physics Prediction',
            'scene_analysis': 'Scene Analysis',
            'forensic_qa': 'Forensic QA',
            'full': 'Video Inpainting'
        }
        preview_folder = folder_map.get(analysis_type, 'Video Inpainting')
        preview_dir = os.path.join(scene_root, preview_folder)
        os.makedirs(preview_dir, exist_ok=True)
        video_exts = {'.mp4', '.mov', '.avi', '.mkv'}
        for sf in saved_files:
            ext = os.path.splitext(sf['filename'])[1].lower()
            if ext in video_exts:
                src_path = os.path.join(scene_root, sf['path'])
                # If source isn't mp4, just copy with original name; else copy as preview.mp4
                dst_name = 'preview.mp4' if ext == '.mp4' else sf['filename']
                dst_path = os.path.join(preview_dir, dst_name)
                try:
                    shutil.copyfile(src_path, dst_path)
                except Exception:
                    pass
                break

        # Create very lightweight placeholder outputs per selected analysis type
        try:
            if analysis_type == 'object_detection':
                od_dir = os.path.join(scene_root, 'Object Detection')
                os.makedirs(od_dir, exist_ok=True)
                det_path = os.path.join(od_dir, 'detections.json')
                if not os.path.exists(det_path):
                    with open(det_path, 'w') as df:
                        json.dump({'summary': 'placeholder detections', 'count': 0}, df, indent=2)
                # Copy an annotated preview (here, just duplicate preview)
                if os.path.exists(dst_path):
                    try:
                        shutil.copyfile(dst_path, os.path.join(od_dir, 'annotated_preview.mp4'))
                    except Exception:
                        pass
            elif analysis_type == '3d_reconstruction':
                colmap_root = os.path.join(scene_root, '3D Reconstruction', 'sparse', '0')
                os.makedirs(colmap_root, exist_ok=True)
                for fname in ['cameras.txt', 'images.txt', 'points3D.txt']:
                    fpath = os.path.join(colmap_root, fname)
                    if not os.path.exists(fpath):
                        with open(fpath, 'w') as f:
                            f.write('# placeholder ' + fname + '\n')
                model_ply = os.path.join(scene_root, '3D Reconstruction', 'model.ply')
                if not os.path.exists(model_ply):
                    with open(model_ply, 'w') as f:
                        f.write('ply\nformat ascii 1.0\ncomment placeholder model\n')
            elif analysis_type == 'timeline':
                tl_dir = os.path.join(scene_root, 'Timeline Analysis')
                os.makedirs(tl_dir, exist_ok=True)
                tl_path = os.path.join(tl_dir, 'timeline.json')
                if not os.path.exists(tl_path):
                    with open(tl_path, 'w') as tf:
                        json.dump({'events': [], 'note': 'placeholder timeline'}, tf, indent=2)
            elif analysis_type == 'physics_prediction':
                ph_dir = os.path.join(scene_root, 'Physics Prediction')
                os.makedirs(ph_dir, exist_ok=True)
                ph_path = os.path.join(ph_dir, 'predictions.json')
                if not os.path.exists(ph_path):
                    with open(ph_path, 'w') as pf:
                        json.dump({'trajectories': [], 'note': 'placeholder predictions'}, pf, indent=2)
            elif analysis_type == 'scene_analysis':
                sa_dir = os.path.join(scene_root, 'Scene Analysis')
                os.makedirs(sa_dir, exist_ok=True)
                sa_path = os.path.join(sa_dir, 'scene_graph.json')
                if not os.path.exists(sa_path):
                    with open(sa_path, 'w') as sf:
                        json.dump({'nodes': [], 'edges': [], 'note': 'placeholder scene graph'}, sf, indent=2)
            elif analysis_type == 'forensic_qa':
                qa_dir = os.path.join(scene_root, 'Forensic QA')
                os.makedirs(qa_dir, exist_ok=True)
                qa_path = os.path.join(qa_dir, 'answers.json')
                if not os.path.exists(qa_path):
                    with open(qa_path, 'w') as qf:
                        json.dump({'answers': [], 'note': 'placeholder QA outputs'}, qf, indent=2)
        except Exception:
            # Non-fatal; placeholders are best-effort
            pass

        # Naive ETA estimate: ~150 MB/min processing rate, min 1 min
        mb = max(1, total_bytes / (1024*1024))
        eta_minutes = max(1, int(round(mb / 150.0)))
        eta_seconds = int(eta_minutes * 60)

        return JsonResponse({
            'success': True,
            'scene_id': scene_id,
            'uploaded_count': len(saved_files),
            'eta_seconds': eta_seconds,
            'message': f'Uploaded {len(saved_files)} file(s) successfully.'
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@csrf_exempt
def get_processing_status(request, scene_id):
    # Check if scene exists in Analysed Data directory
    analysed_base = os.path.join(settings.BASE_DIR, '..', '..', 'Analysed Data')
    metadata_path = os.path.join(analysed_base, scene_id, 'Metadata', 'scene_metadata.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return JsonResponse({
            'scene_id': scene_id,
            'status': 'completed',
            'processing_success': metadata.get('processing_stats', {}).get('successful_cameras', 0) > 0,
            'metadata': metadata
        })
    else:
        return JsonResponse({'scene_id': scene_id, 'status': 'not_found'})

@csrf_exempt
def get_scene_results(request, scene_id):
    analysed_base = os.path.join(settings.BASE_DIR, '..', '..', 'Analysed Data')
    scene_root = os.path.join(analysed_base, scene_id)
    
    # Check for various result files
    results = {
        'scene_id': scene_id,
        'available_data': {},
        'media': [],
        'artifacts': []
    }
    
    # Check for detections
    detections_dir = os.path.join(scene_root, 'Object Detection')
    if os.path.exists(detections_dir):
        detection_files = os.listdir(detections_dir)
        results['available_data']['detections'] = detection_files
    
    # Check for COLMAP reconstruction
    colmap_dir = os.path.join(scene_root, '3D Reconstruction')
    if os.path.exists(colmap_dir):
        colmap_files = os.listdir(colmap_dir)
        results['available_data']['colmap'] = colmap_files
    
    # Check for NeRF results
    nerf_dir = os.path.join(scene_root, 'NeRF Renders')
    if os.path.exists(nerf_dir):
        nerf_files = os.listdir(nerf_dir)
        results['available_data']['nerf'] = nerf_files
    
    # Check for keyframes
    keyframes_dir = os.path.join(scene_root, 'Keyframes')
    if os.path.exists(keyframes_dir):
        keyframe_dirs = os.listdir(keyframes_dir)
        results['available_data']['keyframes'] = keyframe_dirs

    # Additional analysis folders
    folders_extra = [
        ('Timeline Analysis', 'timeline'),
        ('Physics Prediction', 'physics'),
        ('Scene Analysis', 'scene_analysis'),
        ('Forensic QA', 'forensic_qa')
    ]
    for folder_name, key in folders_extra:
        fdir = os.path.join(scene_root, folder_name)
        if os.path.exists(fdir):
            results['available_data'][key] = os.listdir(fdir)

    # Collect media files (video previews and raw videos) with direct download URLs
    def _add_media_from(folder_rel):
        folder_abs = os.path.join(scene_root, folder_rel)
        if not os.path.exists(folder_abs):
            return
        for root, dirs, files in os.walk(folder_abs):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in {'.mp4', '.mov', '.avi', '.mkv'}:
                    abs_path = os.path.join(root, fname)
                    rel_to_scene = os.path.relpath(abs_path, start=scene_root)
                    results['media'].append({
                        'name': fname,
                        'path': rel_to_scene,
                        'folder': folder_rel,
                        'download_url': f"/api/scene/{scene_id}/file/?path=" + json.dumps(rel_to_scene)[1:-1]
                    })

    _add_media_from('Video Inpainting')
    _add_media_from('Raw Data')
    _add_media_from('Object Detection')

    # Collect non-video artifacts (json, ply, txt)
    def _add_artifacts_from(folder_rel):
        folder_abs = os.path.join(scene_root, folder_rel)
        if not os.path.exists(folder_abs):
            return
        for root, dirs, files in os.walk(folder_abs):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in {'.json', '.ply', '.txt'}:
                    abs_path = os.path.join(root, fname)
                    rel_to_scene = os.path.relpath(abs_path, start=scene_root)
                    results['artifacts'].append({
                        'name': fname,
                        'path': rel_to_scene,
                        'folder': folder_rel,
                        'download_url': f"/api/scene/{scene_id}/file/?path=" + json.dumps(rel_to_scene)[1:-1]
                    })

    for fr in ['Object Detection', '3D Reconstruction', 'Timeline Analysis', 'Physics Prediction', 'Scene Analysis', 'Forensic QA']:
        _add_artifacts_from(fr)

    return JsonResponse(results)

@csrf_exempt
def download_scene_results(request, scene_id):
    """Create a ZIP of the scene's raw uploads and any available outputs and stream it."""
    analysed_base = os.path.join(settings.BASE_DIR, '..', '..', 'Analysed Data')
    scene_root = os.path.join(analysed_base, scene_id)
    export_dir = os.path.join(analysed_base, 'Exports')
    os.makedirs(export_dir, exist_ok=True)

    zip_base = os.path.join(export_dir, f'{scene_id}_results')
    zip_path = f'{zip_base}.zip'

    # Create zip of the entire scene folder (mirrors Analysed Data structure)
    import zipfile
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(scene_root):
            for root, dirs, files in os.walk(scene_root):
                for fname in files:
                    fullp = os.path.join(root, fname)
                    arcname = os.path.relpath(fullp, start=scene_root)
                    zf.write(fullp, arcname)
        # Add a report.html into the ZIP
        try:
            report_html = _generate_scene_report_html(scene_root, scene_id)
            zf.writestr('report.html', report_html)
        except Exception:
            pass

    return FileResponse(open(zip_path, 'rb'), as_attachment=True, filename=f'{scene_id}_results.zip')

@csrf_exempt
def download_scene_file(request, scene_id):
    """Stream a single file from Analysed Data/<scene_id>/ by relative path (?path=...)."""
    analysed_base = os.path.join(settings.BASE_DIR, '..', '..', 'Analysed Data')
    scene_root = os.path.join(analysed_base, scene_id)
    rel_path = request.GET.get('path', '')
    if not rel_path:
        return JsonResponse({'error': 'path query parameter required'}, status=400)
    # Normalize and prevent path traversal
    abs_path = os.path.normpath(os.path.join(scene_root, rel_path))
    if not abs_path.startswith(os.path.abspath(scene_root)) or not os.path.exists(abs_path):
        return JsonResponse({'error': 'invalid path'}, status=400)
    filename = os.path.basename(abs_path)
    content_type, _ = mimetypes.guess_type(abs_path)
    force_download = request.GET.get('download') == '1'
    return FileResponse(open(abs_path, 'rb'), as_attachment=force_download, filename=filename, content_type=content_type or 'application/octet-stream')

def _generate_scene_report_html(scene_root, scene_id):
    """Build HTML string for the scene report (used by scene_report and ZIP bundling)."""
    metadata_path = os.path.join(scene_root, 'Metadata', 'scene_metadata.json')

    meta = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    media = []
    artifacts = []

    def _gather_media(folder_rel):
        folder_abs = os.path.join(scene_root, folder_rel)
        if not os.path.exists(folder_abs):
            return
        for root, dirs, files in os.walk(folder_abs):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in {'.mp4', '.mov', '.avi', '.mkv'}:
                    abs_path = os.path.join(root, fname)
                    rel_to_scene = os.path.relpath(abs_path, start=scene_root)
                    media.append({
                        'name': fname,
                        'folder': folder_rel,
                        'url': f"/api/scene/{scene_id}/file/?path=" + json.dumps(rel_to_scene)[1:-1]
                    })

    def _gather_artifacts(folder_rel):
        folder_abs = os.path.join(scene_root, folder_rel)
        if not os.path.exists(folder_abs):
            return
        for root, dirs, files in os.walk(folder_abs):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in {'.json', '.ply', '.txt'}:
                    abs_path = os.path.join(root, fname)
                    rel_to_scene = os.path.relpath(abs_path, start=scene_root)
                    artifacts.append({
                        'name': fname,
                        'folder': folder_rel,
                        'url': f"/api/scene/{scene_id}/file/?path=" + json.dumps(rel_to_scene)[1:-1]
                    })

    for fr in ['Video Inpainting', 'Raw Data', 'Object Detection']:
        _gather_media(fr)
    for fr in ['Object Detection', '3D Reconstruction', 'Timeline Analysis', 'Physics Prediction', 'Scene Analysis', 'Forensic QA']:
        _gather_artifacts(fr)

    title = meta.get('name') or f'Scene {scene_id}'
    html_parts = []
    html_parts.append('<!DOCTYPE html><html><head><meta charset="utf-8">')
    html_parts.append(f'<title>{title} - Report</title>')
    html_parts.append('<style>body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#fff;color:#111;margin:20px;} h1{margin:0 0 8px 0;} .muted{color:#666} .section{margin:24px 0;} .card{border:1px solid #eee;border-radius:8px;padding:14px;margin:10px 0;} .row{display:flex;gap:12px;flex-wrap:wrap} video{max-width:100%;border-radius:8px;background:#000} a.btn{display:inline-block;border:1px solid #ddd;border-radius:6px;padding:6px 10px;text-decoration:none;color:#333} .pill{display:inline-block;background:#f5f5f5;border:1px solid #eee;border-radius:999px;padding:4px 10px;margin-right:6px}</style>')
    html_parts.append('</head><body>')
    html_parts.append(f'<h1>{title} - Report</h1>')
    html_parts.append(f'<div class="muted">Scene ID: {scene_id}</div>')
    if meta:
        html_parts.append('<div class="section">')
        html_parts.append('<div class="pill">Uploaded files: ' + str(len(meta.get('uploaded_files', []))) + '</div>')
        html_parts.append('</div>')

    html_parts.append('<div class="section"><h2>Media Previews</h2>')
    if media:
        for m in media:
            html_parts.append('<div class="card">')
            html_parts.append(f'<div class="muted">{m["folder"]} / {m["name"]}</div>')
            html_parts.append(f'<video controls preload="metadata" src="{m["url"]}"></video>')
            html_parts.append(f'<div style="margin-top:8px"><a class="btn" href="{m["url"]}">Open/Download</a></div>')
            html_parts.append('</div>')
    else:
        html_parts.append('<div class="muted">No media available.</div>')
    html_parts.append('</div>')

    html_parts.append('<div class="section"><h2>Generated Files</h2>')
    if artifacts:
        for a in artifacts:
            html_parts.append('<div class="card">')
            html_parts.append(f'<div>{a["folder"]} / <strong>{a["name"]}</strong></div>')
            html_parts.append(f'<div class="muted" style=\"margin-top:6px\"><a class=\"btn\" href=\"{a["url"]}\">Open/Download</a></div>')
            html_parts.append('</div>')
    else:
        html_parts.append('<div class="muted">No generated files yet.</div>')
    html_parts.append('</div>')
    html_parts.append('</body></html>')
    return '\n'.join(html_parts)

@csrf_exempt
def scene_report(request, scene_id):
    """Render a simple white-page HTML report for a scene with media embeds and artifact links."""
    analysed_base = os.path.join(settings.BASE_DIR, '..', '..', 'Analysed Data')
    scene_root = os.path.join(analysed_base, scene_id)
    html = _generate_scene_report_html(scene_root, scene_id)
    if request.GET.get('download') == '1':
        resp = HttpResponse(html)
        resp['Content-Disposition'] = f'attachment; filename="{scene_id}_report.html"'
        return resp
    return HttpResponse(html)

@csrf_exempt
def list_processed_scenes(request):
    """List all processed scenes stored under 'Analysed Data/'"""
    analysed_base = os.path.join(settings.BASE_DIR, '..', '..', 'Analysed Data')
    scenes = []

    if not os.path.exists(analysed_base):
        return JsonResponse({'scenes': [], 'total_count': 0})

    for scene_id in os.listdir(analysed_base):
        scene_root = os.path.join(analysed_base, scene_id)
        if not os.path.isdir(scene_root):
            continue

        metadata_path = os.path.join(scene_root, 'Metadata', 'scene_metadata.json')
        try:
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            available_data = []
            if os.path.exists(os.path.join(scene_root, 'Object Detection')):
                available_data.append('detections')
            if os.path.exists(os.path.join(scene_root, '3D Reconstruction')):
                available_data.append('colmap')
            if os.path.exists(os.path.join(scene_root, 'NeRF Renders')):
                available_data.append('nerf')
            if os.path.exists(os.path.join(scene_root, 'Keyframes')):
                available_data.append('keyframes')

            scenes.append({
                'scene_id': scene_id,
                'metadata': metadata,
                'available_data': available_data,
                'status': 'completed' if available_data else 'processing'
            })

        except Exception as e:
            scenes.append({
                'scene_id': scene_id,
                'status': 'error',
                'error': str(e)
            })

    return JsonResponse({'scenes': scenes, 'total_count': len(scenes)})
