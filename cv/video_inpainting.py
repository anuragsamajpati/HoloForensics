import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import argparse
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InpaintingConfig:
    """Configuration for video inpainting parameters"""
    max_frames: int = 100  # Maximum frames to process at once
    flow_threshold: float = 0.5  # Optical flow confidence threshold
    temporal_window: int = 5  # Temporal window for context frames
    quality_threshold: float = 0.8  # Quality threshold for frame acceptance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_format: str = "mp4"
    fps: float = 30.0

class OpticalFlowEstimator:
    """Optical flow estimation for temporal consistency"""
    
    def __init__(self):
        self.flow_estimator = cv2.FarnebackOpticalFlow_create()
    
    def estimate_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Estimate optical flow between two frames"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, 
            corners=cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7),
            nextPts=None,
            winSize=(15, 15),
            maxLevel=2
        )
        
        return flow[0] if flow[0] is not None else np.array([])
    
    def warp_frame(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp frame using optical flow"""
        h, w = frame.shape[:2]
        flow_map = np.zeros((h, w, 2), dtype=np.float32)
        
        if len(flow) > 0:
            # Create dense flow field from sparse flow
            for i, (x, y) in enumerate(flow):
                if 0 <= int(y) < h and 0 <= int(x) < w:
                    flow_map[int(y), int(x)] = [x, y]
        
        # Apply flow to warp frame
        warped = cv2.remap(frame, flow_map, None, cv2.INTER_LINEAR)
        return warped

class TemporalConsistencyModule:
    """Ensures temporal consistency across inpainted frames"""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.flow_estimator = OpticalFlowEstimator()
    
    def get_temporal_context(self, frames: List[np.ndarray], target_idx: int) -> List[np.ndarray]:
        """Get temporal context frames around target frame"""
        start_idx = max(0, target_idx - self.window_size // 2)
        end_idx = min(len(frames), target_idx + self.window_size // 2 + 1)
        
        context_frames = []
        for i in range(start_idx, end_idx):
            if i != target_idx and i < len(frames):
                context_frames.append(frames[i])
        
        return context_frames
    
    def propagate_features(self, reference_frame: np.ndarray, target_frame: np.ndarray, 
                          mask: np.ndarray) -> np.ndarray:
        """Propagate features from reference to target frame"""
        # Estimate optical flow
        flow = self.flow_estimator.estimate_flow(reference_frame, target_frame)
        
        # Warp reference frame
        warped_ref = self.flow_estimator.warp_frame(reference_frame, flow)
        
        # Blend warped reference with target using mask
        mask_3d = np.stack([mask] * 3, axis=-1) / 255.0
        result = target_frame * (1 - mask_3d) + warped_ref * mask_3d
        
        return result.astype(np.uint8)

class E2FGVIInpainter:
    """End-to-End Flow-Guided Video Inpainting implementation"""
    
    def __init__(self, config: InpaintingConfig):
        self.config = config
        self.temporal_module = TemporalConsistencyModule(config.temporal_window)
        self.device = torch.device(config.device)
        
        # Initialize feature extraction network (simplified)
        self.feature_extractor = self._build_feature_extractor()
        
    def _build_feature_extractor(self) -> nn.Module:
        """Build feature extraction network"""
        class FeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                self.conv4 = nn.Conv2d(256, 128, 3, padding=1)
                self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
                self.conv6 = nn.Conv2d(64, 3, 3, padding=1)
                
            def forward(self, x):
                x1 = F.relu(self.conv1(x))
                x2 = F.relu(self.conv2(x1))
                x3 = F.relu(self.conv3(x2))
                x4 = F.relu(self.conv4(x3))
                x5 = F.relu(self.conv5(x4 + x1))  # Skip connection
                x6 = torch.sigmoid(self.conv6(x5))
                return x6
        
        model = FeatureExtractor().to(self.device)
        return model
    
    def detect_missing_regions(self, frame: np.ndarray) -> np.ndarray:
        """Detect missing or corrupted regions in frame"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect very dark regions (potential missing data)
        dark_mask = gray < 10
        
        # Detect regions with unusual patterns (corruption)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blur)
        corruption_mask = diff > 50
        
        # Combine masks
        combined_mask = (dark_mask | corruption_mask).astype(np.uint8) * 255
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask
    
    def inpaint_frame(self, frame: np.ndarray, mask: np.ndarray, 
                     context_frames: List[np.ndarray]) -> np.ndarray:
        """Inpaint a single frame using context frames"""
        if len(context_frames) == 0:
            # Fallback to traditional inpainting
            return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        
        # Use temporal consistency for better results
        best_result = frame.copy()
        best_score = 0
        
        for ref_frame in context_frames:
            # Propagate features from reference frame
            result = self.temporal_module.propagate_features(ref_frame, frame, mask)
            
            # Score the result (simplified quality metric)
            score = self._evaluate_inpainting_quality(result, mask)
            
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result
    
    def _evaluate_inpainting_quality(self, frame: np.ndarray, mask: np.ndarray) -> float:
        """Evaluate quality of inpainted region"""
        # Simple quality metric based on gradient consistency
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Evaluate smoothness in inpainted regions
        inpainted_region = mask > 0
        if np.sum(inpainted_region) == 0:
            return 1.0
        
        smoothness = 1.0 / (1.0 + np.mean(gradient_magnitude[inpainted_region]))
        return smoothness

class VideoInpaintingPipeline:
    """Complete video inpainting pipeline for forensic analysis"""
    
    def __init__(self, config: InpaintingConfig = None):
        self.config = config or InpaintingConfig()
        self.inpainter = E2FGVIInpainter(self.config)
        
    def process_video(self, video_path: str, output_path: str, 
                     mask_path: Optional[str] = None) -> Dict:
        """Process entire video for inpainting"""
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        logger.info(f"🎬 Starting video inpainting: {video_path.name}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or self.config.fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Load frames into memory (for temporal context)
        frames = []
        masks = []
        
        logger.info(f"📊 Loading {total_frames} frames...")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frames.append(frame)
            
            # Detect or load mask for this frame
            if mask_path:
                # Load pre-computed mask
                mask_file = Path(mask_path) / f"mask_{frame_idx:06d}.png"
                if mask_file.exists():
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                else:
                    mask = self.inpainter.detect_missing_regions(frame)
            else:
                # Auto-detect missing regions
                mask = self.inpainter.detect_missing_regions(frame)
            
            masks.append(mask)
            frame_idx += 1
            
            # Process in batches to manage memory
            if len(frames) >= self.config.max_frames:
                break
        
        cap.release()
        
        # Process frames with inpainting
        logger.info(f"🔧 Processing frames for inpainting...")
        
        processed_frames = 0
        frames_needing_inpainting = 0
        
        for i, (frame, mask) in enumerate(zip(frames, masks)):
            # Check if frame needs inpainting
            if np.sum(mask) > 0:
                frames_needing_inpainting += 1
                
                # Get temporal context
                context_frames = self.inpainter.temporal_module.get_temporal_context(frames, i)
                
                # Inpaint frame
                inpainted_frame = self.inpainter.inpaint_frame(frame, mask, context_frames)
                out.write(inpainted_frame)
                
                logger.info(f"🎨 Inpainted frame {i+1}/{len(frames)} "
                          f"(mask area: {np.sum(mask > 0)} pixels)")
            else:
                # No inpainting needed
                out.write(frame)
            
            processed_frames += 1
            
            if processed_frames % 10 == 0:
                logger.info(f"⏳ Processed {processed_frames}/{len(frames)} frames...")
        
        out.release()
        
        # Generate processing report
        report = {
            'input_video': str(video_path),
            'output_video': str(output_path),
            'total_frames': len(frames),
            'frames_inpainted': frames_needing_inpainting,
            'inpainting_percentage': (frames_needing_inpainting / len(frames)) * 100,
            'processing_time': time.time(),
            'config': {
                'temporal_window': self.config.temporal_window,
                'quality_threshold': self.config.quality_threshold,
                'device': self.config.device
            }
        }
        
        logger.info(f"✅ Video inpainting complete!")
        logger.info(f"📊 Inpainted {frames_needing_inpainting}/{len(frames)} frames "
                   f"({report['inpainting_percentage']:.1f}%)")
        
        return report
    
    def process_scene(self, scene_path: str, output_dir: str) -> Dict:
        """Process all videos in a forensic scene"""
        scene_path = Path(scene_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🏢 Processing forensic scene: {scene_path.name}")
        
        # Find all video files
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(scene_path.glob(f"*{ext}"))
        
        if not video_files:
            raise ValueError(f"No video files found in {scene_path}")
        
        scene_report = {
            'scene_id': scene_path.name,
            'total_videos': len(video_files),
            'processed_videos': [],
            'processing_stats': {}
        }
        
        total_frames_processed = 0
        total_frames_inpainted = 0
        
        for video_file in video_files:
            camera_id = video_file.stem
            output_file = output_dir / f"{camera_id}_inpainted.mp4"
            
            logger.info(f"📹 Processing camera: {camera_id}")
            
            try:
                report = self.process_video(str(video_file), str(output_file))
                scene_report['processed_videos'].append({
                    'camera_id': camera_id,
                    'status': 'success',
                    'report': report
                })
                
                total_frames_processed += report['total_frames']
                total_frames_inpainted += report['frames_inpainted']
                
            except Exception as e:
                logger.error(f"❌ Error processing {camera_id}: {e}")
                scene_report['processed_videos'].append({
                    'camera_id': camera_id,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Calculate scene-wide statistics
        scene_report['processing_stats'] = {
            'total_frames_processed': total_frames_processed,
            'total_frames_inpainted': total_frames_inpainted,
            'scene_inpainting_percentage': (total_frames_inpainted / total_frames_processed * 100) if total_frames_processed > 0 else 0,
            'successful_cameras': sum(1 for v in scene_report['processed_videos'] if v['status'] == 'success')
        }
        
        # Save scene report
        report_file = output_dir / f"{scene_path.name}_inpainting_report.json"
        with open(report_file, 'w') as f:
            json.dump(scene_report, f, indent=2)
        
        logger.info(f"🎉 Scene inpainting complete!")
        logger.info(f"📊 Processed {scene_report['processing_stats']['successful_cameras']}/{len(video_files)} cameras")
        logger.info(f"🎨 Inpainted {total_frames_inpainted}/{total_frames_processed} frames "
                   f"({scene_report['processing_stats']['scene_inpainting_percentage']:.1f}%)")
        
        return scene_report

def main():
    parser = argparse.ArgumentParser(description='HoloForensics Video Inpainting Pipeline')
    parser.add_argument('--input', required=True, help='Input video file or scene directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--mask-dir', help='Directory containing mask files (optional)')
    parser.add_argument('--temporal-window', type=int, default=5, help='Temporal window size')
    parser.add_argument('--quality-threshold', type=float, default=0.8, help='Quality threshold')
    parser.add_argument('--device', default='auto', help='Processing device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Setup configuration
    config = InpaintingConfig(
        temporal_window=args.temporal_window,
        quality_threshold=args.quality_threshold,
        device=args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Initialize pipeline
    pipeline = VideoInpaintingPipeline(config)
    
    input_path = Path(args.input)
    
    try:
        if input_path.is_file():
            # Process single video
            output_file = Path(args.output) / f"{input_path.stem}_inpainted.mp4"
            report = pipeline.process_video(str(input_path), str(output_file), args.mask_dir)
            print(f"✅ Single video processing complete: {output_file}")
            
        elif input_path.is_dir():
            # Process scene directory
            report = pipeline.process_scene(str(input_path), args.output)
            print(f"✅ Scene processing complete: {args.output}")
            
        else:
            raise ValueError(f"Invalid input path: {input_path}")
            
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
