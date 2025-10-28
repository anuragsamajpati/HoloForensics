import cv2
import numpy as np
import os
import json
import argparse
from pathlib import Path
import platform

class VideoIngestor:
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.MOV', '.MP4']
        self.is_apple_silicon = platform.machine() == 'arm64'
        
    def get_video_metadata(self, video_path):
        """Extract video metadata using OpenCV"""
        print(f"📊 Analyzing {video_path.name}...")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        metadata = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': 0
        }
        
        if metadata['fps'] > 0:
            metadata['duration'] = metadata['frame_count'] / metadata['fps']
        
        cap.release()
        return metadata
    
    def extract_keyframes(self, video_path, output_dir, frames_per_second=1):
        """Extract keyframes optimized for Mac"""
        print(f"🎬 Extracting keyframes from {video_path.name}...")
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            print(f"⚠️  Warning: Invalid FPS for {video_path.name}, using default 30")
            fps = 30.0
        
        frame_interval = max(1, int(fps / frames_per_second))
        frame_count = 0
        saved_count = 0
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"   📹 Video FPS: {fps:.1f}")
        print(f"   🎯 Saving every {frame_interval} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                filename = f"frame_{saved_count:06d}_t{timestamp:.2f}s.jpg"
                filepath = output_dir / filename
                
                cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                saved_count += 1
                
                if saved_count % 10 == 0:
                    print(f"   💾 Saved {saved_count} frames...")
            
            frame_count += 1
        
        cap.release()
        print(f"   ✅ Extracted {saved_count} keyframes")
        return saved_count
    
    def process_scene(self, scene_path, keyframe_fps=1.0):
        """Process all videos in a scene folder"""
        scene_path = Path(scene_path)
        
        if not scene_path.exists():
            raise ValueError(f"Scene path does not exist: {scene_path}")
        
        print(f"\n🎬 Processing Scene: {scene_path.name}")
        print("=" * 50)
        
        video_files = []
        for ext in self.supported_formats:
            video_files.extend(scene_path.glob(f"*{ext}"))
        
        if not video_files:
            raise ValueError(f"No video files found in {scene_path}")
        
        print(f"📹 Found {len(video_files)} videos:")
        for vf in video_files:
            print(f"   - {vf.name}")
        
        scene_metadata = {
            'scene_id': scene_path.name,
            'processed_on': 'Mac',
            'cameras': {},
            'processing_stats': {}
        }
        
        total_keyframes = 0
        
        for i, video_file in enumerate(video_files, 1):
            camera_id = video_file.stem.lower()
            
            print(f"\n📷 Processing Camera {i}/{len(video_files)}: {camera_id}")
            print("-" * 30)
            
            try:
                metadata = self.get_video_metadata(video_file)
                keyframes_dir = Path(f"data/keyframes/{scene_path.name}/{camera_id}")
                keyframe_count = self.extract_keyframes(video_file, keyframes_dir, keyframe_fps)
                
                total_keyframes += keyframe_count
                
                scene_metadata['cameras'][camera_id] = {
                    'video_file': str(video_file),
                    'video_size_mb': round(video_file.stat().st_size / (1024*1024), 2),
                    'metadata': metadata,
                    'keyframes_dir': str(keyframes_dir),
                    'keyframe_count': keyframe_count,
                    'processing_success': True
                }
                
                print(f"   ✅ {camera_id}: {keyframe_count} keyframes extracted")
                
            except Exception as e:
                print(f"   ❌ Error processing {camera_id}: {e}")
                scene_metadata['cameras'][camera_id] = {
                    'video_file': str(video_file),
                    'processing_success': False,
                    'error': str(e)
                }
        
        scene_metadata['processing_stats'] = {
            'total_cameras': len(video_files),
            'successful_cameras': sum(1 for cam in scene_metadata['cameras'].values() 
                                    if cam.get('processing_success', False)),
            'total_keyframes': total_keyframes,
            'keyframe_fps': keyframe_fps
        }
        
        metadata_file = Path(f"data/{scene_path.name}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(scene_metadata, f, indent=2)
        
        print(f"\n🎉 Scene Processing Complete!")
        print(f"📊 Processed {scene_metadata['processing_stats']['successful_cameras']}/{scene_metadata['processing_stats']['total_cameras']} cameras")
        print(f"🖼️  Total keyframes: {total_keyframes}")
        print(f"💾 Metadata saved: {metadata_file}")
        
        return scene_metadata

def main():
    parser = argparse.ArgumentParser(description='HoloForensics Video Processor (Mac Version)')
    parser.add_argument('--scene', required=True, help='Scene folder path (e.g., data/raw/scene_001)')
    parser.add_argument('--fps', type=float, default=1.0, help='Keyframes per second to extract')
    
    args = parser.parse_args()
    
    print("🍎 HoloForensics Video Processor - Mac Edition")
    print("=" * 50)
    
    ingestor = VideoIngestor()
    
    try:
        metadata = ingestor.process_scene(args.scene, args.fps)
        print(f"\n✅ SUCCESS!")
        print(f"Next: Run verification with:")
        print(f"python cv/verify_setup.py {Path(args.scene).name}")
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
