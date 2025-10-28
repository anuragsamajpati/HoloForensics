import cv2
import json
import numpy as np
from pathlib import Path
import sys

def visualize_tracks(scene_id, cam_id):
    """Create visualization of tracking results for any scene/camera"""
    
    # Load tracks
    tracks_file = Path(f"data/tracks/{scene_id}/{cam_id}_consistent.json")
    if not tracks_file.exists():
        print(f"Error: Tracking file not found: {tracks_file}")
        return None
    
    with open(tracks_file) as f:
        tracks_data = json.load(f)
    
    # Load frames
    frames_path = Path(f"data/keyframes/{scene_id}/{cam_id}")
    if not frames_path.exists():
        print(f"Error: Frames directory not found: {frames_path}")
        return None
    
    # Try different frame patterns
    frame_files = []
    for pattern in ["seq_*.jpg", "frame_*.jpg", "*.jpg", "*.png"]:
        frame_files = sorted(frames_path.glob(pattern))
        if frame_files:
            print(f"Found {len(frame_files)} frames with pattern {pattern}")
            break
    
    if not frame_files:
        print(f"Error: No image files found in {frames_path}")
        return None
    
    # Colors for different track IDs
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue  
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 255, 0),  # Light Green
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Light Blue
    ]
    
    # Group tracks by frame
    tracks_by_frame = {}
    for track in tracks_data["tracks"]:
        frame_idx = track["frame"]
        if frame_idx not in tracks_by_frame:
            tracks_by_frame[frame_idx] = []
        tracks_by_frame[frame_idx].append(track)
    
    print(f"Creating tracking visualization for {scene_id}/{cam_id}...")
    print(f"Processing {len(frame_files)} frames with {len(tracks_data['tracks'])} track instances")
    
    # Create output directory
    viz_dir = Path(f"data/tracks/{scene_id}/visualization")
    viz_dir.mkdir(exist_ok=True)
    
    frames_with_tracks = 0
    
    for frame_idx, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file))
        if frame is None:
            print(f"Warning: Could not load frame {frame_file}")
            continue
        
        # Draw tracks for this frame
        frame_tracks = 0
        if frame_idx in tracks_by_frame:
            for track in tracks_by_frame[frame_idx]:
                track_id = track["track_id"]
                bbox = track["bbox"]
                x, y, w, h = [int(v) for v in bbox]
                
                # Ensure bbox is within frame bounds
                h_img, w_img = frame.shape[:2]
                x = max(0, min(x, w_img-1))
                y = max(0, min(y, h_img-1))
                x2 = max(x+1, min(x+w, w_img))
                y2 = max(y+1, min(y+h, h_img))
                
                # Choose color based on track ID
                color = colors[(track_id - 1) % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x2, y2), color, 3)
                
                # Draw track ID with background
                label = f"ID: {track_id}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
                cv2.putText(frame, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                frame_tracks += 1
            
            if frame_tracks > 0:
                frames_with_tracks += 1
        
        # Add frame info
        cv2.rectangle(frame, (5, 5), (350, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Scene: {scene_id} Cam: {cam_id}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracks: {frame_tracks}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save visualization frame
        output_file = viz_dir / f"{cam_id}_frame_{frame_idx:03d}_tracked.jpg"
        cv2.imwrite(str(output_file), frame)
        
        if frame_idx % 5 == 0:  # Progress update every 5 frames
            print(f"  Processed frame {frame_idx}/{len(frame_files)}")
    
    print(f"Visualization complete!")
    print(f"  Frames processed: {len(frame_files)}")
    print(f"  Frames with tracks: {frames_with_tracks}")
    print(f"  Output directory: {viz_dir}")
    
    return viz_dir

def process_all_scenes():
    """Process visualization for all available scenes"""
    tracks_root = Path("data/tracks")
    if not tracks_root.exists():
        print("No tracks directory found")
        return
    
    scenes_processed = 0
    
    for scene_dir in tracks_root.iterdir():
        if not scene_dir.is_dir():
            continue
            
        scene_id = scene_dir.name
        print(f"\n=== Processing Scene: {scene_id} ===")
        
        # Find all tracking files in this scene
        for track_file in scene_dir.glob("*_consistent.json"):
            cam_id = track_file.name.replace("_consistent.json", "")
            print(f"Found tracking data for {scene_id}/{cam_id}")
            
            try:
                result = visualize_tracks(scene_id, cam_id)
                if result:
                    scenes_processed += 1
            except Exception as e:
                print(f"Error processing {scene_id}/{cam_id}: {e}")
    
    print(f"\nProcessed {scenes_processed} scene/camera combinations")

def process_scene_all_cameras(scene_id):
    """Process all cameras in a specific scene"""
    keyframes_path = Path(f"data/keyframes/{scene_id}")
    if not keyframes_path.exists():
        print(f"Scene not found: {scene_id}")
        return
    
    cameras_processed = 0
    
    for cam_dir in keyframes_path.iterdir():
        if not cam_dir.is_dir():
            continue
            
        cam_id = cam_dir.name
        
        # Check if tracking data exists for this camera
        tracks_file = Path(f"data/tracks/{scene_id}/{cam_id}_consistent.json")
        if tracks_file.exists():
            print(f"\nProcessing {scene_id}/{cam_id}...")
            try:
                result = visualize_tracks(scene_id, cam_id)
                if result:
                    cameras_processed += 1
            except Exception as e:
                print(f"Error processing {scene_id}/{cam_id}: {e}")
        else:
            print(f"No tracking data found for {scene_id}/{cam_id}")
    
    print(f"\nProcessed {cameras_processed} cameras in {scene_id}")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        # Command line usage: python cv/visualization.py scene_002 cam_001
        scene_id = sys.argv[1]
        cam_id = sys.argv[2]
        visualize_tracks(scene_id, cam_id)
    elif len(sys.argv) == 2:
        if sys.argv[1] == "all":
            # Process all scenes
            process_all_scenes()
        else:
            # Single argument assumed to be scene_id, process all cameras in that scene
            scene_id = sys.argv[1]
            process_scene_all_cameras(scene_id)
    else:
        print("Usage:")
        print("  python cv/visualization.py scene_002 cam_001  # Specific scene/camera")
        print("  python cv/visualization.py scene_002          # All cameras in scene_002")
        print("  python cv/visualization.py all               # All scenes and cameras")
        print("\nExample workflows:")
        print("  python cv/visualization.py scene_001         # Process all cameras in scene_001")
        print("  python cv/visualization.py scene_003 cam_002 # Process specific camera")