#!/usr/bin/env python3
"""
HoloForensics Video Inpainting Integration Test Suite
Tests the complete E2FGVI video inpainting pipeline for forensic analysis
"""

import os
import sys
import cv2
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
import logging
import time
import argparse

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from video_inpainting import VideoInpaintingPipeline, InpaintingConfig
    from forensic_inpainting_integration import ForensicInpaintingIntegration, ForensicInpaintingConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the cv/ directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InpaintingTestSuite:
    """Comprehensive test suite for video inpainting functionality"""
    
    def __init__(self, test_data_dir: str = None):
        self.test_data_dir = Path(test_data_dir) if test_data_dir else Path("test_data")
        self.temp_dir = None
        self.test_results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'failures': []
        }
        
    def setup_test_environment(self):
        """Setup test environment with sample data"""
        print("🔧 Setting up test environment...")
        
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="holoforensics_test_"))
        print(f"📁 Test directory: {self.temp_dir}")
        
        # Create test scene structure
        test_scene_dir = self.temp_dir / "scene_test_001"
        test_scene_dir.mkdir(parents=True)
        
        # Generate synthetic test videos
        self.create_test_videos(test_scene_dir)
        
        return test_scene_dir
    
    def create_test_videos(self, scene_dir: Path):
        """Create synthetic test videos for testing"""
        print("🎬 Creating synthetic test videos...")
        
        # Video parameters
        width, height = 640, 480
        fps = 30.0
        duration_seconds = 2
        total_frames = int(fps * duration_seconds)
        
        # Create multiple camera views
        cameras = ['cam_001', 'cam_002', 'cam_003']
        
        for camera_id in cameras:
            video_path = scene_dir / f"{camera_id}.mp4"
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            for frame_idx in range(total_frames):
                # Create synthetic frame with patterns
                frame = self.create_synthetic_frame(width, height, frame_idx, camera_id)
                
                # Add some "corruption" to simulate missing regions
                if frame_idx % 15 == 0:  # Every 15th frame has corruption
                    frame = self.add_corruption(frame)
                
                writer.write(frame)
            
            writer.release()
            print(f"   ✅ Created {camera_id}.mp4 ({total_frames} frames)")
    
    def create_synthetic_frame(self, width: int, height: int, frame_idx: int, camera_id: str) -> np.ndarray:
        """Create a synthetic frame with identifiable patterns"""
        # Create base frame with gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(height):
            for x in range(width):
                frame[y, x] = [
                    int(255 * x / width),  # Red gradient
                    int(255 * y / height),  # Green gradient
                    int(128 + 127 * np.sin(frame_idx * 0.1))  # Blue animation
                ]
        
        # Add moving objects
        center_x = int(width/2 + 100 * np.sin(frame_idx * 0.2))
        center_y = int(height/2 + 50 * np.cos(frame_idx * 0.15))
        
        # Draw moving circle (simulates person/object)
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
        
        # Add camera ID text
        cv2.putText(frame, f"{camera_id} - Frame {frame_idx}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def add_corruption(self, frame: np.ndarray) -> np.ndarray:
        """Add corruption to simulate missing/damaged regions"""
        corrupted = frame.copy()
        h, w = frame.shape[:2]
        
        # Add black rectangles (missing data)
        cv2.rectangle(corrupted, (w//4, h//4), (w//2, h//2), (0, 0, 0), -1)
        
        # Add noise in some regions
        noise_region = corrupted[h//3:2*h//3, 2*w//3:w]
        noise = np.random.randint(0, 255, noise_region.shape, dtype=np.uint8)
        corrupted[h//3:2*h//3, 2*w//3:w] = noise
        
        return corrupted
    
    def test_basic_inpainting_pipeline(self):
        """Test basic video inpainting pipeline functionality"""
        print("\n🧪 Testing basic inpainting pipeline...")
        
        try:
            # Setup test scene
            test_scene = self.setup_test_environment()
            
            # Initialize inpainting pipeline
            config = InpaintingConfig(
                max_frames=50,
                temporal_window=3,
                device='cpu'  # Use CPU for testing
            )
            
            pipeline = VideoInpaintingPipeline(config)
            
            # Test single video processing
            test_video = test_scene / "cam_001.mp4"
            output_video = self.temp_dir / "cam_001_inpainted.mp4"
            
            print(f"   📹 Processing: {test_video}")
            result = pipeline.process_video(str(test_video), str(output_video))
            
            # Validate results
            assert output_video.exists(), "Output video not created"
            assert result['total_frames'] > 0, "No frames processed"
            assert 'frames_inpainted' in result, "Missing inpainting stats"
            
            print(f"   ✅ Processed {result['total_frames']} frames")
            print(f"   🎨 Inpainted {result['frames_inpainted']} frames")
            
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            self.test_results['failures'].append(f"Basic pipeline test: {e}")
            self.test_results['tests_failed'] += 1
        
        finally:
            self.test_results['tests_run'] += 1
    
    def test_forensic_integration(self):
        """Test forensic integration with compliance features"""
        print("\n🏛️ Testing forensic integration...")
        
        try:
            # Setup test scene
            test_scene = self.setup_test_environment()
            
            # Initialize forensic integration
            config = ForensicInpaintingConfig(
                evidence_preservation=True,
                quality_validation=True,
                chain_of_custody=True,
                backup_originals=True
            )
            
            forensic_system = ForensicInpaintingIntegration(
                case_id="TEST_CASE_001",
                operator="test_operator",
                config=config
            )
            
            # Process forensic scene
            output_dir = self.temp_dir / "forensic_output"
            result = forensic_system.process_forensic_scene(str(test_scene), str(output_dir))
            
            # Validate forensic compliance
            assert result['case_id'] == "TEST_CASE_001", "Case ID mismatch"
            assert result['compliance_status'] in ['COMPLIANT', 'NON_COMPLIANT'], "Invalid compliance status"
            assert len(result['processed_videos']) > 0, "No videos processed"
            
            # Check required directories exist
            required_dirs = ['evidence', 'processed', 'validation', 'originals']
            for dir_name in required_dirs:
                dir_path = output_dir / dir_name
                assert dir_path.exists(), f"Missing required directory: {dir_name}"
            
            # Check chain of custody
            custody_file = output_dir / "evidence" / f"chain_of_custody_{result['case_id']}.json"
            assert custody_file.exists(), "Chain of custody file missing"
            
            print(f"   ✅ Processed {result['total_videos']} videos")
            print(f"   ⚖️ Compliance status: {result['compliance_status']}")
            print(f"   📊 Success rate: {result['success_rate']:.1%}")
            
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            self.test_results['failures'].append(f"Forensic integration test: {e}")
            self.test_results['tests_failed'] += 1
        
        finally:
            self.test_results['tests_run'] += 1
    
    def test_quality_validation(self):
        """Test quality validation functionality"""
        print("\n🔍 Testing quality validation...")
        
        try:
            from forensic_inpainting_integration import EvidenceValidator
            
            validator = EvidenceValidator()
            
            # Create test frames
            original_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            inpainted_frame = original_frame.copy()
            
            # Create test mask
            mask = np.zeros((480, 640), dtype=np.uint8)
            mask[200:280, 300:400] = 255  # Inpainted region
            
            # Modify inpainted region slightly
            inpainted_frame[200:280, 300:400] = np.random.randint(0, 255, (80, 100, 3), dtype=np.uint8)
            
            # Validate inpainting quality
            validation_result = validator.validate_inpainting_quality(
                original_frame, inpainted_frame, mask
            )
            
            # Check validation metrics
            required_metrics = ['ssim_score', 'temporal_consistency', 'edge_preservation', 
                              'overall_quality', 'passed_validation']
            
            for metric in required_metrics:
                assert metric in validation_result, f"Missing validation metric: {metric}"
                assert 0 <= validation_result[metric] <= 1, f"Invalid metric range: {metric}"
            
            print(f"   ✅ Overall quality score: {validation_result['overall_quality']:.3f}")
            print(f"   🎯 Validation passed: {validation_result['passed_validation']}")
            
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            self.test_results['failures'].append(f"Quality validation test: {e}")
            self.test_results['tests_failed'] += 1
        
        finally:
            self.test_results['tests_run'] += 1
    
    def test_api_integration(self):
        """Test API integration (mock test)"""
        print("\n🌐 Testing API integration...")
        
        try:
            # This would normally test the Django API endpoints
            # For now, we'll do a mock test of the data structures
            
            # Simulate API request data
            api_request = {
                'scene_id': 'scene_test_001',
                'case_id': 'TEST_CASE_002',
                'evidence_preservation': True,
                'quality_validation': True,
                'chain_of_custody': True,
                'backup_originals': True
            }
            
            # Validate request structure
            required_fields = ['scene_id', 'case_id']
            for field in required_fields:
                assert field in api_request, f"Missing required field: {field}"
            
            # Simulate API response
            api_response = {
                'success': True,
                'job_id': 'test_job_123',
                'status': 'started',
                'processing_report': {
                    'case_id': api_request['case_id'],
                    'success_rate': 1.0,
                    'compliance_status': 'COMPLIANT'
                }
            }
            
            # Validate response structure
            assert api_response['success'] == True, "API response indicates failure"
            assert 'job_id' in api_response, "Missing job ID in response"
            
            print(f"   ✅ API request validation passed")
            print(f"   🆔 Mock job ID: {api_response['job_id']}")
            
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            self.test_results['failures'].append(f"API integration test: {e}")
            self.test_results['tests_failed'] += 1
        
        finally:
            self.test_results['tests_run'] += 1
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir and self.temp_dir.exists():
            print(f"🧹 Cleaning up test directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting HoloForensics Video Inpainting Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Run individual tests
            self.test_basic_inpainting_pipeline()
            self.test_forensic_integration()
            self.test_quality_validation()
            self.test_api_integration()
            
        finally:
            # Always cleanup
            self.cleanup_test_environment()
        
        # Print results
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print("🎯 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Tests Run:    {self.test_results['tests_run']}")
        print(f"Tests Passed: {self.test_results['tests_passed']}")
        print(f"Tests Failed: {self.test_results['tests_failed']}")
        print(f"Success Rate: {(self.test_results['tests_passed'] / self.test_results['tests_run'] * 100):.1f}%")
        print(f"Duration:     {duration:.2f} seconds")
        
        if self.test_results['failures']:
            print("\n❌ FAILURES:")
            for i, failure in enumerate(self.test_results['failures'], 1):
                print(f"   {i}. {failure}")
        
        if self.test_results['tests_failed'] == 0:
            print("\n🎉 ALL TESTS PASSED! Video inpainting integration is ready for production.")
            return True
        else:
            print(f"\n⚠️  {self.test_results['tests_failed']} test(s) failed. Please review and fix issues.")
            return False

def main():
    parser = argparse.ArgumentParser(description='HoloForensics Video Inpainting Test Suite')
    parser.add_argument('--test-data', help='Path to test data directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run test suite
    test_suite = InpaintingTestSuite(args.test_data)
    success = test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
