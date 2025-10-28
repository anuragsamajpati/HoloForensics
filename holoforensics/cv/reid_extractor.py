import cv2
import numpy as np
from pathlib import Path
import json

def extract_simple_features(image, bbox):
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

def main():
    # Load tracking results
    tracks_file = Path("data/tracks/scene_001/cam_001_consistent.json")
    with open(tracks_file) as f:
        tracks_data = json.load(f)
    
    # Load frames
    frames_path = Path("data/keyframes/scene_001/cam_001")
    frame_files = sorted(frames_path.glob("seq_*.jpg"))
    
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
    output_file = Path("data/tracks/scene_001/cam_001_reid_features.json")
    with open(output_file, "w") as f:
        json.dump(reid_features, f, indent=2)
    
    print(f"Extracted Re-ID features for {len(reid_features)} unique tracks")
    
    # Show summary
    for track_id, track_features in reid_features.items():
        print(f"Track {track_id}: {len(track_features)} feature vectors")

if __name__ == "__main__":
    main()
