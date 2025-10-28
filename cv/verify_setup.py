import json
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os

def verify_scene_processing(scene_id):
    metadata_file = Path(f"data/{scene_id}_metadata.json")
    
    if not metadata_file.exists():
        print(f"Metadata file not found: {metadata_file}")
        return False
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    print(f"HoloForensics Scene Verification")
    print("=" * 40)
    print(f"Scene: {scene_id}")
    print(f"Total cameras: {len(metadata['cameras'])}")
    
    total_keyframes = 0
    successful_cameras = 0
    
    for cam_id, cam_data in metadata['cameras'].items():
        print(f"\nCamera: {cam_id}")
        
        if not cam_data.get('processing_success', False):
            print(f"   Processing failed")
            continue
            
        successful_cameras += 1
        meta = cam_data['metadata']
        print(f"   Resolution: {meta['width']}x{meta['height']}")
        print(f"   FPS: {meta['fps']:.1f}")
        print(f"   Duration: {meta['duration']:.1f}s")
        print(f"   Keyframes: {cam_data['keyframe_count']}")
        
        total_keyframes += cam_data['keyframe_count']
        
        keyframes_dir = Path(cam_data['keyframes_dir'])
        if keyframes_dir.exists():
            keyframe_files = list(keyframes_dir.glob("*.jpg"))
            if len(keyframe_files) == cam_data['keyframe_count']:
                print(f"   All keyframes verified")
            else:
                print(f"   Keyframe count mismatch")
    
    print(f"\nSummary:")
    print(f"   Successful cameras: {successful_cameras}/{len(metadata['cameras'])}")
    print(f"   Total keyframes: {total_keyframes}")
    
    if successful_cameras > 0:
        print(f"\nScene {scene_id} verification PASSED!")
        return True
    else:
        print(f"\nScene {scene_id} verification FAILED!")
        return False

def show_sample_frames(scene_id):
    metadata_file = Path(f"data/{scene_id}_metadata.json")
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    for cam_id, cam_data in metadata['cameras'].items():
        if not cam_data.get('processing_success', False):
            continue
            
        keyframes_dir = Path(cam_data['keyframes_dir'])
        keyframe_files = sorted(keyframes_dir.glob("*.jpg"))
        
        if len(keyframe_files) >= 3:
            sample_frames = [keyframe_files[0], 
                           keyframe_files[len(keyframe_files)//2], 
                           keyframe_files[-1]]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for i, frame_path in enumerate(sample_frames):
                img = cv2.imread(str(frame_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(img_rgb)
                    axes[i].set_title(f'{frame_path.name}')
                    axes[i].axis('off')
            
            plt.suptitle(f'Sample Frames - Camera {cam_id}')
            plt.tight_layout()
            
            output_path = Path(f'docs/{scene_id}_{cam_id}_samples.png')
            output_path.parent.mkdir(exist_ok=True)
            plt.savefig(output_path)
            print(f"Saved sample frames: {output_path}")
            
            plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python cv/verify_setup.py scene_001")
        sys.exit(1)
    
    scene_id = sys.argv[1]
    
    if verify_scene_processing(scene_id):
        show_sample_frames(scene_id)
    else:
        print("Fix issues and try again")
        sys.exit(1)