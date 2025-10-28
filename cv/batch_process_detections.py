# cv/batch_process_detections.py
import multiprocessing as mp
from pathlib import Path
import json
import time
from detect_objects import ObjectDetector, get_class_distribution
import cv2

def process_camera_worker(args):
    """Worker function for parallel processing"""
    scene_id, cam_id, frame_files, output_dir = args
    
    detector = ObjectDetector('s')
    all_detections = []
    
    start_time = time.time()
    worker_id = mp.current_process().name
    
    print(f"[{worker_id}] Starting {scene_id}/{cam_id} - {len(frame_files)} frames")
    
    for frame_idx, frame_file in enumerate(frame_files):
        try:
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue
            
            detections = detector.detect_frame(frame, conf_thresh=0.3)
            
            for det in detections:
                det['frame'] = frame_idx
                det['timestamp'] = frame_idx / 30.0
                det['frame_file'] = frame_file.name
            
            all_detections.extend(detections)
            
            if frame_idx % 100 == 0 and frame_idx > 0:
                elapsed = time.time() - start_time
                fps = frame_idx / elapsed
                print(f"[{worker_id}] {scene_id}/{cam_id}: {frame_idx}/{len(frame_files)} - {fps:.1f} FPS")
                
        except Exception as e:
            print(f"[{worker_id}] Error processing frame {frame_idx}: {e}")
            continue
    
    # Save results
    output_file = output_dir / f'{cam_id}_detections.json'
    detection_summary = {
        'scene_id': scene_id,
        'camera_id': cam_id,
        'total_frames': len(frame_files),
        'total_detections': len(all_detections),
        'processing_time': time.time() - start_time,
        'class_distribution': get_class_distribution(all_detections),
        'detections': all_detections
    }
    
    with open(output_file, 'w') as f:
        json.dump(detection_summary, f, indent=2)
    
    processing_time = time.time() - start_time
    print(f"[{worker_id}] Completed {scene_id}/{cam_id}: {len(all_detections)} detections in {processing_time:.1f}s")
    
    return {
        'scene_id': scene_id,
        'camera_id': cam_id,
        'detections': len(all_detections),
        'processing_time': processing_time
    }

def batch_process_all_scenes(max_workers=None):
    """Process all scenes in parallel"""
    keyframes_root = Path('data/keyframes')
    detections_root = Path('data/detections')
    detections_root.mkdir(exist_ok=True)
    
    if not keyframes_root.exists():
        print(f"Keyframes directory not found: {keyframes_root}")
        return
    
    # Collect all tasks
    tasks = []
    for scene_dir in keyframes_root.iterdir():
        if not scene_dir.is_dir():
            continue
            
        scene_id = scene_dir.name
        scene_output = detections_root / scene_id
        scene_output.mkdir(exist_ok=True)
        
        for cam_dir in scene_dir.iterdir():
            if not cam_dir.is_dir():
                continue
                
            cam_id = cam_dir.name
            frame_files = sorted(list(cam_dir.glob('*.jpg')) + list(cam_dir.glob('*.png')))
            
            if frame_files:
                tasks.append((scene_id, cam_id, frame_files, scene_output))
    
    if not tasks:
        print("No scenes found to process")
        return
    
    print(f"Found {len(tasks)} camera streams to process")
    
    # Process in parallel
    if max_workers is None:
        max_workers = min(len(tasks), mp.cpu_count())
    
    start_time = time.time()
    
    with mp.Pool(processes=max_workers) as pool:
        results = pool.map(process_camera_worker, tasks)
    
    # Summary
    total_time = time.time() - start_time
    total_detections = sum(r['detections'] for r in results)
    
    print(f"\nBatch Processing Complete!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Total detections: {total_detections}")
    print(f"Average processing speed: {len(tasks)/total_time:.2f} cameras/second")
    
    # Save summary
    summary = {
        'total_cameras': len(tasks),
        'total_detections': total_detections,
        'total_processing_time': total_time,
        'cameras': results
    }
    
    with open(detections_root / 'batch_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process object detection')
    parser.add_argument('--workers', type=int, default=None, 
                       help='Number of parallel workers (default: auto)')
    
    args = parser.parse_args()
    batch_process_all_scenes(max_workers=args.workers)