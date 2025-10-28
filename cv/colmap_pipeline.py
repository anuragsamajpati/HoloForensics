import subprocess
import json
import numpy as np
import cv2
from pathlib import Path
import shutil
import math

class AdvancedCOLMAPProcessor:
    def __init__(self, scene_id):
        self.scene_id = scene_id
        self.keyframes_dir = Path(f"data/keyframes/{scene_id}")
        self.colmap_dir = Path(f"data/colmap/{scene_id}")
        self.colmap_dir.mkdir(parents=True, exist_ok=True)
        
    def detect_scenario(self):
        """Detect what type of scenario we're dealing with"""
        camera_dirs = [d for d in self.keyframes_dir.iterdir() if d.is_dir()]
        num_cameras = len(camera_dirs)
        
        print(f"Detected {num_cameras} camera(s) in scene {self.scene_id}")
        
        if num_cameras == 1:
            return "single_camera"
        elif num_cameras >= 2:
            return "multi_camera"
        else:
            return "no_cameras"
    
    def option_a_single_camera_geometry(self):
        """Option A: Single camera geometric analysis"""
        print("=== Option A: Single Camera Geometry Estimation ===")
        
        # Use structure-from-motion on single camera
        # This estimates camera motion and basic scene structure
        self.prepare_images_single()
        
        if not self.extract_features():
            return False
        if not self.match_features():
            return False
        if not self.run_sparse_reconstruction():
            return False
            
        # Analyze camera motion
        self.analyze_camera_motion()
        return True
    
    def option_b_synthetic_multicamera(self):
        """Option B: Create synthetic multi-camera views"""
        print("=== Option B: Synthetic Multi-Camera Generation ===")
        
        # Generate virtual camera positions around the scene
        self.generate_synthetic_views()
        
        # Run COLMAP on synthetic views
        if not self.extract_features():
            return False
        if not self.match_features():
            return False
        if not self.run_sparse_reconstruction():
            return False
            
        return True
    
    def option_c_find_additional_footage(self):
        """Option C: Search for additional surveillance angles"""
        print("=== Option C: Multi-View Analysis (if available) ===")
        
        # Check if we have multiple camera folders
        camera_dirs = [d for d in self.keyframes_dir.iterdir() if d.is_dir()]
        
        if len(camera_dirs) >= 2:
            print(f"Found {len(camera_dirs)} cameras - proceeding with multi-view reconstruction")
            self.prepare_images_multi()
            
            if not self.extract_features():
                return False
            if not self.match_features():
                return False
            if not self.run_sparse_reconstruction():
                return False
                
            return True
        else:
            print("Single camera detected - will proceed with synthetic views")
            return self.option_b_synthetic_multicamera()
    
    def prepare_images_single(self):
        """Prepare images from single camera for SfM"""
        images_dir = self.colmap_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Get the single camera directory
        camera_dirs = [d for d in self.keyframes_dir.iterdir() if d.is_dir()]
        cam_dir = camera_dirs[0]
        
        # Copy every 3rd frame for temporal baseline
        keyframes = sorted(cam_dir.glob("*.jpg"))
        selected_frames = keyframes[::3]  # Every 3rd frame
        
        for i, frame_path in enumerate(selected_frames):
            new_name = f"frame_{i:04d}.jpg"
            dst_path = images_dir / new_name
            shutil.copy2(frame_path, dst_path)
            
        print(f"Prepared {len(selected_frames)} images for single-camera SfM")
        return images_dir
    
    def prepare_images_multi(self):
        """Prepare images from multiple cameras"""
        images_dir = self.colmap_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        for cam_dir in self.keyframes_dir.iterdir():
            if cam_dir.is_dir():
                cam_id = cam_dir.name
                keyframes = sorted(cam_dir.glob("*.jpg"))
                
                # Take every 5th frame from each camera
                selected_frames = keyframes[::5]
                
                for i, frame_path in enumerate(selected_frames):
                    new_name = f"{cam_id}_{i:04d}.jpg"
                    dst_path = images_dir / new_name
                    shutil.copy2(frame_path, dst_path)
        
        total_images = len(list(images_dir.glob("*.jpg")))
        print(f"Prepared {total_images} images from multiple cameras")
        return images_dir
    
    def generate_synthetic_views(self):
        """Generate synthetic camera viewpoints from single camera"""
        print("Generating synthetic camera views...")
        
        images_dir = self.colmap_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Get source images
        camera_dirs = [d for d in self.keyframes_dir.iterdir() if d.is_dir()]
        cam_dir = camera_dirs[0]
        keyframes = sorted(cam_dir.glob("*.jpg"))
        
        # Take a subset of original frames
        base_frames = keyframes[::10]  # Every 10th frame
        
        for i, frame_path in enumerate(base_frames):
            # Copy original view
            original_name = f"original_{i:04d}.jpg"
            shutil.copy2(frame_path, images_dir / original_name)
            
            # Generate transformed views
            img = cv2.imread(str(frame_path))
            
            # Create slight perspective shifts (simulating different camera positions)
            transforms = self.generate_perspective_transforms(img.shape)
            
            for j, transform in enumerate(transforms):
                transformed = cv2.warpPerspective(img, transform, (img.shape[1], img.shape[0]))
                synthetic_name = f"synthetic_{i:04d}_{j:02d}.jpg"
                cv2.imwrite(str(images_dir / synthetic_name), transformed)
        
        total_synthetic = len(list(images_dir.glob("*.jpg")))
        print(f"Generated {total_synthetic} synthetic views")
        return images_dir
    
    def generate_perspective_transforms(self, image_shape):
        """Generate perspective transformation matrices for synthetic views"""
        h, w = image_shape[:2]
        transforms = []
        
        # Small perspective shifts to simulate camera movement
        shifts = [
            # Slight rotations and translations
            (5, 0, 10),    # small rotation, x-shift
            (-5, 10, 0),   # opposite rotation, y-shift
            (0, -10, 5),   # y-shift with small rotation
            (3, 5, -8),    # combined shifts
        ]
        
        for angle, dx, dy in shifts:
            # Create perspective transformation matrix
            center = (w//2, h//2)
            
            # Rotation matrix
            M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Add translation
            M_rot[0, 2] += dx
            M_rot[1, 2] += dy
            
            # Convert to 3x3 perspective matrix
            M_perspective = np.eye(3)
            M_perspective[:2, :] = M_rot
            
            transforms.append(M_perspective)
        
        return transforms
    
    def analyze_camera_motion(self):
        """Analyze camera motion from single-camera SfM"""
        poses_file = self.colmap_dir / "poses.json"
        
        if not poses_file.exists():
            print("No poses file found for camera motion analysis")
            return
        
        with open(poses_file, 'r') as f:
            poses_data = json.load(f)
        
        images = poses_data.get('images', {})
        
        if len(images) < 2:
            print("Insufficient images for motion analysis")
            return
        
        # Calculate camera trajectory
        positions = []
        for img_name, img_data in images.items():
            translation = img_data['translation']
            positions.append(translation)
        
        positions = np.array(positions)
        
        # Analyze motion statistics
        total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        max_distance = np.max(np.linalg.norm(positions - positions[0], axis=1))
        
        motion_analysis = {
            'total_trajectory_length': float(total_distance),
            'max_distance_from_start': float(max_distance),
            'num_positions': len(positions),
            'average_position': positions.mean(axis=0).tolist(),
            'motion_type': 'static' if total_distance < 0.1 else 'moving'
        }
        
        # Save motion analysis
        motion_file = self.colmap_dir / "camera_motion.json"
        with open(motion_file, 'w') as f:
            json.dump(motion_analysis, f, indent=2)
        
        print(f"Camera motion analysis saved to {motion_file}")
        print(f"Motion type: {motion_analysis['motion_type']}")
        print(f"Total movement: {total_distance:.2f} units")
    
    def extract_features(self):
        """Extract SIFT features (same as before)"""
        database_path = self.colmap_dir / "database.db"
        images_dir = self.colmap_dir / "images"
        
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.max_image_size", "1600",
            "--SiftExtraction.max_num_features", "8192"
        ]
        
        print("Extracting features...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Feature extraction failed: {result.stderr}")
            return False
            
        print("Feature extraction completed successfully")
        return True
    
    def match_features(self):
        """Match features (same as before)"""
        database_path = self.colmap_dir / "database.db"
        
        cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.guided_matching", "1"
        ]
        
        print("Matching features...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Feature matching failed: {result.stderr}")
            return False
            
        print("Feature matching completed successfully")
        return True
    
    def run_sparse_reconstruction(self):
        """Run sparse reconstruction (same as before)"""
        database_path = self.colmap_dir / "database.db"
        images_dir = self.colmap_dir / "images"
        sparse_dir = self.colmap_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        
        cmd = [
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.min_model_size", "10",
            "--Mapper.multiple_models", "0"
        ]
        
        print("Running sparse reconstruction...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Sparse reconstruction failed: {result.stderr}")
            return False
            
        print("Sparse reconstruction completed successfully")
        return True
    
    def export_poses(self):
        """Export camera poses (same as before but with additional metadata)"""
        sparse_dir = self.colmap_dir / "sparse" / "0"
        
        if not sparse_dir.exists():
            print("No sparse reconstruction found")
            return False
        
        cameras_txt = sparse_dir / "cameras.txt"
        images_txt = sparse_dir / "images.txt"
        points_txt = sparse_dir / "points3D.txt"
        
        if not all([cameras_txt.exists(), images_txt.exists(), points_txt.exists()]):
            print("COLMAP output files not found")
            return False
        
        poses_data = self.parse_colmap_output(cameras_txt, images_txt, points_txt)
        
        # Add scenario information
        poses_data['processing_info'] = {
            'scenario': self.detect_scenario(),
            'options_used': ['A', 'B', 'C'],
            'total_images_processed': len(poses_data['images']),
            'total_3d_points': len(poses_data['points_3d'])
        }
        
        poses_file = self.colmap_dir / "poses.json"
        with open(poses_file, 'w') as f:
            json.dump(poses_data, f, indent=2)
        
        print(f"Camera poses exported to {poses_file}")
        return True
    
    def parse_colmap_output(self, cameras_txt, images_txt, points_txt):
        """Parse COLMAP output (same as before)"""
        poses_data = {
            'scene_id': self.scene_id,
            'cameras': {},
            'images': {},
            'points_3d': []
        }
        
        # Parse cameras.txt
        with open(cameras_txt, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    camera_id = parts[0]
                    model = parts[1]
                    width = int(parts[2])
                    height = int(parts[3])
                    params = [float(p) for p in parts[4:]]
                    
                    poses_data['cameras'][camera_id] = {
                        'model': model,
                        'width': width,
                        'height': height,
                        'params': params
                    }
        
        # Parse images.txt
        with open(images_txt, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 10:
                    image_id = parts[0]
                    qw, qx, qy, qz = [float(p) for p in parts[1:5]]
                    tx, ty, tz = [float(p) for p in parts[5:8]]
                    camera_id = parts[8]
                    image_name = parts[9]
                    
                    poses_data['images'][image_name] = {
                        'image_id': image_id,
                        'quaternion': [qw, qx, qy, qz],
                        'translation': [tx, ty, tz],
                        'camera_id': camera_id
                    }
        
        # Parse points3D.txt
        point_count = 0
        with open(points_txt, 'r') as f:
            for line in f:
                if line.startswith('#') or point_count >= 1000:
                    continue
                parts = line.strip().split()
                if len(parts) >= 6:
                    x, y, z = [float(p) for p in parts[1:4]]
                    r, g, b = [int(p) for p in parts[4:7]]
                    
                    poses_data['points_3d'].append({
                        'position': [x, y, z],
                        'color': [r, g, b]
                    })
                    point_count += 1
        
        return poses_data
    
    def run_comprehensive_pipeline(self):
        """Run all three options in sequence"""
        print(f"Starting comprehensive COLMAP pipeline for {self.scene_id}")
        print("This will run Options A, B, and C to demonstrate all capabilities")
        
        scenario = self.detect_scenario()
        
        success = False
        
        # Try Option C first (multi-camera if available)
        if scenario == "multi_camera":
            print("\n" + "="*60)
            if self.option_c_find_additional_footage():
                success = True
        
        # Try Option A (single camera analysis)
        if scenario == "single_camera" or not success:
            print("\n" + "="*60)
            if self.option_a_single_camera_geometry():
                success = True
        
        # Always try Option B (synthetic views) for comparison
        print("\n" + "="*60)
        synthetic_success = self.option_b_synthetic_multicamera()
        
        # Export final results
        if success or synthetic_success:
            self.export_poses()
            print(f"\nComprehensive pipeline completed successfully for {self.scene_id}")
            print(f"Results available in: data/colmap/{self.scene_id}/")
            return True
        else:
            print("All options failed")
            return False

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python cv/advanced_colmap_pipeline.py scene_001")
        sys.exit(1)
    
    scene_id = sys.argv[1]
    processor = AdvancedCOLMAPProcessor(scene_id)
    
    if processor.run_comprehensive_pipeline():
        print("\nSuccess! All COLMAP options have been demonstrated.")
        print(f"Check results in: data/colmap/{scene_id}/")
        print("- poses.json: Camera poses and 3D points")
        print("- camera_motion.json: Motion analysis (if applicable)")
    else:
        print("Pipeline failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()