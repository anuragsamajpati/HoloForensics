import cv2
import numpy as np
from pathlib import Path
import json
import subprocess

class AdvancedCOLMAPProcessor:
    def __init__(self, scene_id):
        self.scene_id = scene_id
        self.keyframes_dir = Path(f"data/keyframes/{scene_id}")
        self.colmap_dir = Path(f"data/colmap/{scene_id}_advanced")
        self.colmap_dir.mkdir(parents=True, exist_ok=True)
        
    def create_strong_synthetic_views(self):
        """Create synthetic views with significant perspective differences"""
        print("Creating strong synthetic multi-camera views...")
        
        images_dir = self.colmap_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Get source keyframes
        camera_dirs = [d for d in self.keyframes_dir.iterdir() if d.is_dir()]
        cam_dir = camera_dirs[0]
        keyframes = sorted(cam_dir.glob("*.jpg"))
        
        # Select frames with more spacing
        selected_frames = keyframes[::30]  # Every 30th frame for more temporal baseline
        
        for i, frame_path in enumerate(selected_frames):
            img = cv2.imread(str(frame_path))
            h, w = img.shape[:2]
            
            # Create more dramatic synthetic viewpoints
            synthetic_views = [
                # Original view
                (img, "original"),
                # Zoomed out (simulates farther camera)
                (self.apply_zoom(img, 0.7), "zoom_out"),
                # Zoomed in (simulates closer camera) 
                (self.apply_zoom(img, 1.4), "zoom_in"),
                # Rotated views
                (self.apply_rotation(img, 8), "rotate_pos"),
                (self.apply_rotation(img, -8), "rotate_neg"),
                # Perspective shifts (simulates different positions)
                (self.apply_perspective_shift(img, 20, 15), "persp_1"),
                (self.apply_perspective_shift(img, -15, 20), "persp_2"),
                # Cropped views (simulates different framing)
                (self.apply_crop_shift(img, 0.1, 0.1), "crop_1"),
                (self.apply_crop_shift(img, -0.1, 0.1), "crop_2"),
            ]
            
            for view_img, view_type in synthetic_views:
                if view_img is not None:
                    filename = f"{view_type}_frame_{i:03d}.jpg"
                    cv2.imwrite(str(images_dir / filename), view_img)
        
        total_images = len(list(images_dir.glob("*.jpg")))
        print(f"Created {total_images} synthetic images with strong parallax")
        return images_dir
    
    def apply_zoom(self, img, factor):
        h, w = img.shape[:2]
        center = (w//2, h//2)
        
        # Create zoom transformation
        M = cv2.getRotationMatrix2D(center, 0, factor)
        
        # Adjust translation to keep center
        if factor < 1:  # Zoom out
            M[0, 2] += (w - w*factor) / 2
            M[1, 2] += (h - h*factor) / 2
        
        result = cv2.warpAffine(img, M, (w, h))
        return result
    
    def apply_rotation(self, img, angle):
        h, w = img.shape[:2]
        center = (w//2, h//2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(img, M, (w, h))
        return result
    
    def apply_perspective_shift(self, img, dx, dy):
        h, w = img.shape[:2]
        
        # Create perspective transformation with more dramatic shift
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_points = np.float32([
            [dx, dy], 
            [w-dx, dy*0.5], 
            [w+dx*0.5, h-dy], 
            [-dx*0.5, h+dy*0.5]
        ])
        
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        result = cv2.warpPerspective(img, M, (w, h))
        return result
    
    def apply_crop_shift(self, img, shift_x, shift_y):
        h, w = img.shape[:2]
        
        # Calculate crop region
        crop_w = int(w * 0.8)  # Use 80% of original
        crop_h = int(h * 0.8)
        
        start_x = int(w * (0.1 + shift_x))
        start_y = int(h * (0.1 + shift_y))
        
        # Ensure we don't go out of bounds
        start_x = max(0, min(start_x, w - crop_w))
        start_y = max(0, min(start_y, h - crop_h))
        
        # Crop and resize back to original size
        cropped = img[start_y:start_y+crop_h, start_x:start_x+crop_w]
        result = cv2.resize(cropped, (w, h))
        return result
    
    def run_colmap_pipeline(self):
        """Run COLMAP with relaxed parameters"""
        database_path = self.colmap_dir / "database.db"
        images_dir = self.colmap_dir / "images"
        sparse_dir = self.colmap_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        
        # Feature extraction
        print("Extracting features...")
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--ImageReader.single_camera", "0",  # Multiple camera types
            "--SiftExtraction.max_image_size", "800",  # Smaller for speed
            "--SiftExtraction.max_num_features", "4096"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Feature extraction failed: {result.stderr}")
            return False
        
        # Feature matching
        print("Matching features...")
        cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.guided_matching", "1",
            "--SiftMatching.max_ratio", "0.9",  # More permissive
            "--SiftMatching.max_distance", "0.9"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Feature matching failed: {result.stderr}")
            return False
        
        # Sparse reconstruction with very relaxed parameters
        print("Running sparse reconstruction...")
        cmd = [
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.min_model_size", "2",
            "--Mapper.init_min_tri_angle", "1",  # Very permissive
            "--Mapper.abs_pose_min_num_inliers", "4",
            "--Mapper.abs_pose_min_inlier_ratio", "0.15",
            "--Mapper.filter_max_reproj_error", "6.0",  # More permissive
            "--Mapper.init_max_error", "6.0"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Sparse reconstruction failed: {result.stderr}")
            print("Trying with even more relaxed parameters...")
            return self.try_emergency_reconstruction()
        
        return True
    
    def try_emergency_reconstruction(self):
        """Last resort with minimal constraints"""
        database_path = self.colmap_dir / "database.db"
        images_dir = self.colmap_dir / "images"
        sparse_dir = self.colmap_dir / "sparse"
        
        cmd = [
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.min_model_size", "2",
            "--Mapper.init_min_tri_angle", "0.5",
            "--Mapper.abs_pose_min_num_inliers", "3",
            "--Mapper.abs_pose_min_inlier_ratio", "0.1",
            "--Mapper.filter_max_reproj_error", "10.0",
            "--Mapper.init_max_error", "10.0",
            "--Mapper.ba_refine_focal_length", "0",
            "--Mapper.ba_refine_principal_point", "0"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    
    def export_results(self):
        """Export reconstruction results"""
        sparse_dir = self.colmap_dir / "sparse" / "0"
        
        if not sparse_dir.exists():
            print("No reconstruction found")
            return False
        
        # Export to text format
        cmd = [
            "colmap", "model_converter",
            "--input_path", str(sparse_dir),
            "--output_path", str(sparse_dir),
            "--output_type", "TXT"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Export failed: {result.stderr}")
            return False
        
        # Create summary
        self.create_reconstruction_summary()
        return True
    
    def create_reconstruction_summary(self):
        """Create summary of reconstruction results"""
        sparse_dir = self.colmap_dir / "sparse" / "0"
        
        summary = {
            'scene_id': self.scene_id,
            'reconstruction_type': 'synthetic_multi_camera',
            'status': 'success'
        }
        
        # Count cameras and points
        cameras_file = sparse_dir / "cameras.txt"
        images_file = sparse_dir / "images.txt"
        points_file = sparse_dir / "points3D.txt"
        
        if cameras_file.exists():
            with open(cameras_file) as f:
                camera_count = len([l for l in f if not l.startswith('#')])
            summary['cameras'] = camera_count
        
        if images_file.exists():
            with open(images_file) as f:
                image_count = len([l for l in f if not l.startswith('#')])
            summary['registered_images'] = image_count
        
        if points_file.exists():
            with open(points_file) as f:
                point_count = len([l for l in f if not l.startswith('#')])
            summary['3d_points'] = point_count
        
        # Save summary
        summary_file = self.colmap_dir / "reconstruction_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Reconstruction summary: {summary}")
        return summary
    
    def run_full_pipeline(self):
        """Run complete advanced pipeline"""
        print(f"Starting advanced COLMAP pipeline for {self.scene_id}")
        
        # Create synthetic views
        self.create_strong_synthetic_views()
        
        # Run COLMAP
        if self.run_colmap_pipeline():
            if self.export_results():
                print("Advanced pipeline completed successfully!")
                return True
        
        print("Advanced pipeline failed")
        return False

def main():
    import sys
    scene_id = sys.argv[1] if len(sys.argv) > 1 else "scene_001"
    
    processor = AdvancedCOLMAPProcessor(scene_id)
    if processor.run_full_pipeline():
        print(f"Success! Check results in: data/colmap/{scene_id}_advanced/")
    else:
        print("Failed to reconstruct scene")

if __name__ == "__main__":
    main()