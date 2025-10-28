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
    for pattern in ["seq_*.jpg", "frame_*.jpg", "*.jpg"]:
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
    ]
    
    # Group tracks by frame
    tracks_by_frame = {}
    for track in tracks_data["tracks"]:
        frame_idx = track["frame"]
        if frame_idx not in tracks_by_frame:
            tracks_by_frame[frame_idx] = []
        tracks_by_frame[frame_idx].append(track)
    
    print(f"Creating tracking visualization for {scene_id}/{cam_id}...")
    
    # Create output directory
    viz_dir = Path(f"data/tracks/{scene_id}/visualization")
    viz_dir.mkdir(exist_ok=True)
    
    for frame_idx, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file))
        if frame is None:
            continue
        
        # Draw tracks for this frame
        frame_tracks = 0
        if frame_idx in tracks_by_frame:
            for track in tracks_by_frame[frame_idx]:
                track_id = track["track_id"]
                bbox = track["bbox"]
                x, y, w, h = [int(v) for v in bbox]
                
                # Choose color based on track ID
                color = colors[(track_id - 1) % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Draw track ID
                label = f"ID: {track_id}"
                cv2.putText(frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                frame_tracks += 1
        
        # Add frame info
        cv2.putText(frame, f"Scene: {scene_id} Cam: {cam_id}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_idx} Tracks: {frame_tracks}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save visualization frame
        output_file = viz_dir / f"{cam_id}_frame_{frame_idx:03d}_tracked.jpg"
        cv2.imwrite(str(output_file), frame)
    
    print(f"Visualization complete! Output: {viz_dir}")
    return viz_dir

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        scene_id = sys.argv[1]
        cam_id = sys.argv[2]
        visualize_tracks(scene_id, cam_id)
    else:
        # Default
        visualize_tracks("scene_001", "cam_001")
