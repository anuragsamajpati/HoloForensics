import cv2
import numpy as np
from pathlib import Path
import json
import sys

def extract_simple_features(image, bbox):
    """Extract simple color features from person crop"""
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # Crop person region safely
    h_img, w_img = image.shape[:2]
    x = max(0, min(x, w_img-1))
    y = max(0, min(y, h_img-1))
    x2 = max(x+1, min(x+w, w_img))
    y2 = max(y+1, min(y+h, h_img))
    
    person_crop = image[y:y2, x:x2]
    
    if person_crop.size == 0:
        return np.zeros(48)
    
    # Resize and extract color histogram
    person_crop = cv2.resize(person_crop, (32, 64))
    
    hist_b = cv2.calcHist([person_crop], [0], None, [16], [0, 256])
    hist_g = cv2.calcHist([person_crop], [1], None, [16], [0, 256]) 
    hist_r = cv2.calcHist([person_crop], [2], None, [16], [0, 256])
    
    features = np.concatenate([
        hist_b.flatten() / (hist_b.sum() + 1e-7),
        hist_g.flatten() / (hist_g.sum() + 1e-7),
        hist_r.flatten() / (hist_r.sum() + 1e-7)
    ])
    
    return features

def extract_reid_features(scene_id, cam_id):
    """Extract Re-ID features for a specific scene/camera"""
    # Load tracking results
    tracks_file = Path(f"data/tracks/{scene_id}/{cam_id}_consistent.json")
    if not tracks_file.exists():
        print(f"Tracking file not found: {tracks_file}")
        return None
    
    with open(tracks_file) as f:
        tracks_data = json.load(f)
    
    # Load frames
    frames_path = Path(f"data/keyframes/{scene_id}/{cam_id}")
    if not frames_path.exists():
        print(f"Frames path not found: {frames_path}")
        return None
    
    # Try different frame patterns
    frame_files = []
    for pattern in ['seq_*.jpg', 'frame_*.jpg', '*.jpg', '*.png']:
        frame_files = sorted(frames_path.glob(pattern))
        if frame_files:
            break
    
    if not frame_files:
        print(f"No frames found in {frames_path}")
        return None
    
    reid_features = {}
    
    print(f"Processing {len(tracks_data['tracks'])} track instances...")
    
    for track in tracks_data["tracks"]:
        frame_idx = track["frame"]
        track_id = track["track_id"]
        bbox = track["bbox"]
        
        if frame_idx < len(frame_files):
            frame = cv2.imread(str(frame_files[frame_idx]))
            if frame is not None:
                features = extract_simple_features(frame, bbox)
                
                if track_id not in reid_features:
                    reid_features[track_id] = []
                
                reid_features[track_id].append({
                    "frame": frame_idx,
                    "features": features.tolist(),
                    "bbox": bbox
                })
    
    # Save results
    output_file = Path(f"data/tracks/{scene_id}/{cam_id}_reid_features.json")
    with open(output_file, "w") as f:
        json.dump(reid_features, f, indent=2)
    
    print(f"Extracted Re-ID features for {len(reid_features)} unique tracks")
    
    # Show summary
    for track_id, track_features in reid_features.items():
        print(f"Track {track_id}: {len(track_features)} feature vectors")
    
    return reid_features

def process_all_reid():
    """Process Re-ID features for all available tracking results"""
    tracks_root = Path("data/tracks")
    if not tracks_root.exists():
        print("No tracks directory found")
        return
    
    for scene_dir in tracks_root.iterdir():
        if not scene_dir.is_dir():
            continue
            
        scene_id = scene_dir.name
        print(f"\n=== Processing Re-ID for Scene: {scene_id} ===")
        
        for track_file in scene_dir.glob("*_consistent.json"):
            cam_id = track_file.name.replace("_consistent.json", "")
            try:
                extract_reid_features(scene_id, cam_id)
            except Exception as e:
                print(f"Error processing Re-ID for {scene_id}/{cam_id}: {e}")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        scene_id = sys.argv[1]
        cam_id = sys.argv[2]
        extract_reid_features(scene_id, cam_id)
    elif len(sys.argv) == 2:
        if sys.argv[1] == "all":
            process_all_reid()
        else:
            print("Usage:")
            print("  python cv/reid_extractor.py scene_001 cam_001  # Specific scene/camera")
            print("  python cv/reid_extractor.py all               # All scenes")
    else:
        print("Usage:")
        print("  python cv/reid_extractor.py scene_001 cam_001  # Specific scene/camera")
        print("  python cv/reid_extractor.py all               # All scenes")