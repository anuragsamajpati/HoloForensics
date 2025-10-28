# Create cv/synthetic_multicam.py
import cv2
import numpy as np
from pathlib import Path
import json

def create_synthetic_views(scene_id):
    print(f"Creating synthetic multi-camera views for {scene_id}")
    
    # Source keyframes directory
    keyframes_dir = Path(f"data/keyframes/{scene_id}/cam_001")
    synthetic_dir = Path(f"data/colmap/{scene_id}_synthetic/images")
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    # Get keyframes (every 20th frame to spread across timeline)
    keyframes = sorted(keyframes_dir.glob("*.jpg"))
    selected_frames = keyframes[::20]  # Every 20th frame
    
    for i, frame_path in enumerate(selected_frames):
        img = cv2.imread(str(frame_path))
        h, w = img.shape[:2]
        
        # Create 4 synthetic viewpoints
        transforms = [
            # Original view
            np.eye(3),
            # Slight zoom in (simulates closer camera)
            get_zoom_transform(w, h, 1.1),
            # Slight rotation (simulates different angle)
            get_rotation_transform(w, h, 3),
            # Perspective shift (simulates camera movement)
            get_perspective_transform(w, h, 10, 5)
        ]
        
        for j, transform in enumerate(transforms):
            if j == 0:
                # Original image
                result = img
            else:
                # Apply transformation
                result = cv2.warpPerspective(img, transform, (w, h))
            
            # Save synthetic view
            filename = f"view_{j+1}_frame_{i:03d}.jpg"
            cv2.imwrite(str(synthetic_dir / filename), result)
    
    print(f"Created {len(selected_frames) * 4} synthetic images")
    return synthetic_dir

def get_zoom_transform(w, h, zoom_factor):
    center_x, center_y = w // 2, h // 2
    M = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_factor)
    # Convert to 3x3
    transform = np.eye(3)
    transform[:2, :] = M
    return transform

def get_rotation_transform(w, h, angle):
    center_x, center_y = w // 2, h // 2
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    transform = np.eye(3)
    transform[:2, :] = M
    return transform

def get_perspective_transform(w, h, dx, dy):
    # Small perspective shift
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32([[dx, dy], [w-dx, dy], [w, h], [0, h]])
    return cv2.getPerspectiveTransform(src_points, dst_points)

if __name__ == "__main__":
    import sys
    scene_id = sys.argv[1] if len(sys.argv) > 1 else "scene_001"
    create_synthetic_views(scene_id)