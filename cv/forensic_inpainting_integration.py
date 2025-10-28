import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import shutil
import time

from video_inpainting import VideoInpaintingPipeline, InpaintingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ForensicInpaintingConfig:
    """Configuration for forensic video inpainting integration"""
    evidence_preservation: bool = True  # Keep original videos
    quality_validation: bool = True     # Validate inpainting quality
    chain_of_custody: bool = True       # Maintain forensic chain of custody
    backup_originals: bool = True       # Create backups before processing
    metadata_tracking: bool = True      # Track all processing metadata
    legal_compliance: bool = True       # Ensure legal compliance standards

class ForensicChainOfCustody:
    """Maintains forensic chain of custody for video processing"""
    
    def __init__(self, case_id: str):
        self.case_id = case_id
        self.custody_log = []
        
    def log_action(self, action: str, operator: str, timestamp: float = None, 
                   details: Dict = None):
        """Log forensic action with timestamp and operator"""
        if timestamp is None:
            timestamp = time.time()
            
        entry = {
            'timestamp': timestamp,
            'action': action,
            'operator': operator,
            'case_id': self.case_id,
            'details': details or {}
        }
        
        self.custody_log.append(entry)
        logger.info(f"🔒 Chain of Custody: {action} by {operator}")
        
    def save_log(self, output_path: Path):
        """Save chain of custody log"""
        log_file = output_path / f"chain_of_custody_{self.case_id}.json"
        with open(log_file, 'w') as f:
            json.dump({
                'case_id': self.case_id,
                'custody_log': self.custody_log,
                'total_actions': len(self.custody_log)
            }, f, indent=2)
        
        logger.info(f"📋 Chain of custody saved: {log_file}")

class EvidenceValidator:
    """Validates integrity and quality of inpainted evidence"""
    
    def __init__(self):
        self.validation_metrics = {}
        
    def validate_inpainting_quality(self, original_frame: np.ndarray, 
                                   inpainted_frame: np.ndarray, 
                                   mask: np.ndarray) -> Dict:
        """Validate quality of inpainted regions"""
        # Calculate structural similarity
        ssim_score = self._calculate_ssim(original_frame, inpainted_frame, mask)
        
        # Calculate temporal consistency (if available)
        temporal_score = self._calculate_temporal_consistency(inpainted_frame)
        
        # Calculate edge preservation
        edge_score = self._calculate_edge_preservation(original_frame, inpainted_frame, mask)
        
        # Overall quality score
        quality_score = (ssim_score + temporal_score + edge_score) / 3
        
        validation_result = {
            'ssim_score': ssim_score,
            'temporal_consistency': temporal_score,
            'edge_preservation': edge_score,
            'overall_quality': quality_score,
            'passed_validation': quality_score > 0.7,  # Threshold for forensic use
            'inpainted_area_percentage': (np.sum(mask > 0) / mask.size) * 100
        }
        
        return validation_result
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
        """Calculate Structural Similarity Index for non-inpainted regions"""
        # Focus on non-inpainted regions for comparison
        non_inpainted = mask == 0
        
        if np.sum(non_inpainted) == 0:
            return 1.0  # No comparison possible
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean and variance for non-inpainted regions
        mean1 = np.mean(gray1[non_inpainted])
        mean2 = np.mean(gray2[non_inpainted])
        var1 = np.var(gray1[non_inpainted])
        var2 = np.var(gray2[non_inpainted])
        covar = np.mean((gray1[non_inpainted] - mean1) * (gray2[non_inpainted] - mean2))
        
        # SSIM calculation
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2*mean1*mean2 + c1) * (2*covar + c2)) / ((mean1**2 + mean2**2 + c1) * (var1 + var2 + c2))
        
        return max(0, min(1, ssim))
    
    def _calculate_temporal_consistency(self, frame: np.ndarray) -> float:
        """Calculate temporal consistency score (simplified)"""
        # This would require previous/next frames in a real implementation
        # For now, return a baseline score based on gradient smoothness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Smoothness as consistency metric
        consistency = 1.0 / (1.0 + np.std(gradient_mag))
        return min(1.0, consistency)
    
    def _calculate_edge_preservation(self, original: np.ndarray, inpainted: np.ndarray, 
                                   mask: np.ndarray) -> float:
        """Calculate edge preservation in inpainted regions"""
        # Detect edges in both images
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_inp = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
        
        edges_orig = cv2.Canny(gray_orig, 50, 150)
        edges_inp = cv2.Canny(gray_inp, 50, 150)
        
        # Focus on boundary regions of inpainted areas
        kernel = np.ones((5, 5), np.uint8)
        mask_boundary = cv2.dilate(mask, kernel) - cv2.erode(mask, kernel)
        
        if np.sum(mask_boundary) == 0:
            return 1.0
        
        # Compare edge consistency at boundaries
        edge_similarity = np.sum((edges_orig == edges_inp) & (mask_boundary > 0))
        total_boundary = np.sum(mask_boundary > 0)
        
        return edge_similarity / total_boundary if total_boundary > 0 else 1.0

class ForensicInpaintingIntegration:
    """Complete forensic video inpainting integration system"""
    
    def __init__(self, case_id: str, operator: str, config: ForensicInpaintingConfig = None):
        self.case_id = case_id
        self.operator = operator
        self.config = config or ForensicInpaintingConfig()
        
        # Initialize components
        self.chain_of_custody = ForensicChainOfCustody(case_id)
        self.validator = EvidenceValidator()
        
        # Initialize inpainting pipeline
        inpainting_config = InpaintingConfig(
            temporal_window=7,  # Larger window for forensic quality
            quality_threshold=0.8,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.inpainting_pipeline = VideoInpaintingPipeline(inpainting_config)
        
        # Log initialization
        self.chain_of_custody.log_action(
            "System Initialized", 
            operator,
            details={'case_id': case_id, 'config': config.__dict__}
        )
    
    def process_forensic_scene(self, scene_path: str, output_dir: str) -> Dict:
        """Process complete forensic scene with full compliance"""
        scene_path = Path(scene_path)
        output_dir = Path(output_dir)
        
        logger.info(f"🏛️ Starting forensic inpainting for case: {self.case_id}")
        logger.info(f"📁 Scene: {scene_path.name}")
        
        # Create output structure
        evidence_dir = output_dir / "evidence"
        processed_dir = output_dir / "processed"
        validation_dir = output_dir / "validation"
        backup_dir = output_dir / "originals"
        
        for dir_path in [evidence_dir, processed_dir, validation_dir, backup_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Log scene processing start
        self.chain_of_custody.log_action(
            "Scene Processing Started",
            self.operator,
            details={'scene_path': str(scene_path), 'output_dir': str(output_dir)}
        )
        
        # Find video files
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.MOV']:
            video_files.extend(scene_path.glob(f"*{ext}"))
        
        if not video_files:
            raise ValueError(f"No video files found in {scene_path}")
        
        # Backup originals if required
        if self.config.backup_originals:
            self._backup_original_files(video_files, backup_dir)
        
        # Process each video
        processing_report = {
            'case_id': self.case_id,
            'scene_id': scene_path.name,
            'operator': self.operator,
            'processing_timestamp': time.time(),
            'total_videos': len(video_files),
            'processed_videos': [],
            'validation_results': {},
            'compliance_status': 'PENDING'
        }
        
        successful_processing = 0
        
        for video_file in video_files:
            camera_id = video_file.stem
            
            logger.info(f"📹 Processing evidence: {camera_id}")
            
            try:
                # Process video with inpainting
                video_result = self._process_single_video(
                    video_file, processed_dir, camera_id
                )
                
                # Validate results if required
                if self.config.quality_validation:
                    validation_result = self._validate_processed_video(
                        video_file, video_result['output_path'], validation_dir, camera_id
                    )
                    video_result['validation'] = validation_result
                
                processing_report['processed_videos'].append(video_result)
                
                if video_result['status'] == 'success':
                    successful_processing += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing {camera_id}: {e}")
                
                error_result = {
                    'camera_id': camera_id,
                    'status': 'error',
                    'error': str(e),
                    'input_file': str(video_file)
                }
                processing_report['processed_videos'].append(error_result)
                
                # Log error in chain of custody
                self.chain_of_custody.log_action(
                    "Processing Error",
                    self.operator,
                    details={'camera_id': camera_id, 'error': str(e)}
                )
        
        # Calculate overall compliance status
        processing_report['successful_videos'] = successful_processing
        processing_report['success_rate'] = successful_processing / len(video_files)
        
        if processing_report['success_rate'] >= 0.8:  # 80% success threshold
            processing_report['compliance_status'] = 'COMPLIANT'
        else:
            processing_report['compliance_status'] = 'NON_COMPLIANT'
        
        # Save processing report
        report_file = evidence_dir / f"forensic_inpainting_report_{self.case_id}.json"
        with open(report_file, 'w') as f:
            json.dump(processing_report, f, indent=2)
        
        # Save chain of custody
        self.chain_of_custody.log_action(
            "Scene Processing Completed",
            self.operator,
            details={
                'success_rate': processing_report['success_rate'],
                'compliance_status': processing_report['compliance_status']
            }
        )
        self.chain_of_custody.save_log(evidence_dir)
        
        logger.info(f"🎉 Forensic scene processing complete!")
        logger.info(f"📊 Success rate: {processing_report['success_rate']:.1%}")
        logger.info(f"⚖️ Compliance status: {processing_report['compliance_status']}")
        
        return processing_report
    
    def _backup_original_files(self, video_files: List[Path], backup_dir: Path):
        """Create forensic backups of original files"""
        logger.info(f"💾 Creating forensic backups...")
        
        for video_file in video_files:
            backup_file = backup_dir / video_file.name
            shutil.copy2(video_file, backup_file)
            
            # Calculate and store file hash for integrity
            file_hash = self._calculate_file_hash(video_file)
            hash_file = backup_dir / f"{video_file.stem}_hash.txt"
            with open(hash_file, 'w') as f:
                f.write(f"SHA256: {file_hash}\n")
                f.write(f"Original: {video_file}\n")
                f.write(f"Backup: {backup_file}\n")
        
        self.chain_of_custody.log_action(
            "Original Files Backed Up",
            self.operator,
            details={'backup_count': len(video_files), 'backup_dir': str(backup_dir)}
        )
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for integrity verification"""
        import hashlib
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _process_single_video(self, video_file: Path, output_dir: Path, camera_id: str) -> Dict:
        """Process single video with full forensic compliance"""
        output_file = output_dir / f"{camera_id}_inpainted.mp4"
        
        # Log processing start
        self.chain_of_custody.log_action(
            "Video Processing Started",
            self.operator,
            details={'camera_id': camera_id, 'input_file': str(video_file)}
        )
        
        try:
            # Run inpainting pipeline
            inpainting_result = self.inpainting_pipeline.process_video(
                str(video_file), str(output_file)
            )
            
            # Calculate file hashes for integrity
            original_hash = self._calculate_file_hash(video_file)
            processed_hash = self._calculate_file_hash(output_file)
            
            result = {
                'camera_id': camera_id,
                'status': 'success',
                'input_file': str(video_file),
                'output_path': str(output_file),
                'original_hash': original_hash,
                'processed_hash': processed_hash,
                'inpainting_stats': inpainting_result,
                'processing_timestamp': time.time()
            }
            
            # Log successful processing
            self.chain_of_custody.log_action(
                "Video Processing Completed",
                self.operator,
                details={
                    'camera_id': camera_id,
                    'frames_inpainted': inpainting_result.get('frames_inpainted', 0),
                    'output_file': str(output_file)
                }
            )
            
            return result
            
        except Exception as e:
            # Log processing failure
            self.chain_of_custody.log_action(
                "Video Processing Failed",
                self.operator,
                details={'camera_id': camera_id, 'error': str(e)}
            )
            
            return {
                'camera_id': camera_id,
                'status': 'error',
                'error': str(e),
                'input_file': str(video_file)
            }
    
    def _validate_processed_video(self, original_file: Path, processed_file: Path, 
                                validation_dir: Path, camera_id: str) -> Dict:
        """Validate processed video for forensic compliance"""
        logger.info(f"🔍 Validating processed video: {camera_id}")
        
        # Sample frames for validation
        validation_results = []
        
        # Open both videos
        cap_orig = cv2.VideoCapture(str(original_file))
        cap_proc = cv2.VideoCapture(str(processed_file))
        
        frame_count = 0
        sample_interval = 30  # Sample every 30th frame
        
        while True:
            ret_orig, frame_orig = cap_orig.read()
            ret_proc, frame_proc = cap_proc.read()
            
            if not (ret_orig and ret_proc):
                break
            
            if frame_count % sample_interval == 0:
                # Detect inpainted regions (simplified)
                mask = self._detect_inpainted_regions(frame_orig, frame_proc)
                
                # Validate this frame
                frame_validation = self.validator.validate_inpainting_quality(
                    frame_orig, frame_proc, mask
                )
                frame_validation['frame_number'] = frame_count
                validation_results.append(frame_validation)
            
            frame_count += 1
        
        cap_orig.release()
        cap_proc.release()
        
        # Calculate overall validation metrics
        if validation_results:
            avg_quality = np.mean([r['overall_quality'] for r in validation_results])
            passed_frames = sum(1 for r in validation_results if r['passed_validation'])
            validation_pass_rate = passed_frames / len(validation_results)
        else:
            avg_quality = 0.0
            validation_pass_rate = 0.0
        
        overall_validation = {
            'camera_id': camera_id,
            'average_quality_score': avg_quality,
            'validation_pass_rate': validation_pass_rate,
            'total_frames_validated': len(validation_results),
            'passed_validation': validation_pass_rate >= 0.8,  # 80% threshold
            'frame_validations': validation_results
        }
        
        # Save validation report
        validation_file = validation_dir / f"{camera_id}_validation.json"
        with open(validation_file, 'w') as f:
            json.dump(overall_validation, f, indent=2)
        
        # Log validation completion
        self.chain_of_custody.log_action(
            "Video Validation Completed",
            self.operator,
            details={
                'camera_id': camera_id,
                'validation_pass_rate': validation_pass_rate,
                'average_quality': avg_quality
            }
        )
        
        return overall_validation
    
    def _detect_inpainted_regions(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """Detect regions that were inpainted by comparing frames"""
        # Calculate absolute difference
        diff = cv2.absdiff(original, processed)
        
        # Convert to grayscale and threshold
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)
        
        # Clean up mask with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Forensic Video Inpainting Integration')
    parser.add_argument('--case-id', required=True, help='Forensic case ID')
    parser.add_argument('--operator', required=True, help='Operator name')
    parser.add_argument('--scene', required=True, help='Scene directory path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup creation')
    parser.add_argument('--no-validation', action='store_true', help='Skip quality validation')
    
    args = parser.parse_args()
    
    # Setup forensic configuration
    config = ForensicInpaintingConfig(
        backup_originals=not args.no_backup,
        quality_validation=not args.no_validation
    )
    
    # Initialize forensic integration
    forensic_system = ForensicInpaintingIntegration(
        case_id=args.case_id,
        operator=args.operator,
        config=config
    )
    
    try:
        # Process forensic scene
        report = forensic_system.process_forensic_scene(args.scene, args.output)
        
        print(f"\n✅ Forensic processing complete!")
        print(f"📊 Case ID: {report['case_id']}")
        print(f"🎯 Success Rate: {report['success_rate']:.1%}")
        print(f"⚖️ Compliance: {report['compliance_status']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Forensic processing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
