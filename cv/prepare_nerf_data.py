# Update cv/prepare_nerf_data.py to fix the file path issue
import json
import numpy as np
import cv2
from pathlib import Path
import shutil

class NeRFDataPreparer:
    def __init__(self, scene_id):
        self.scene_id = scene_id
        self.colmap_dir = Path(f"data/colmap/{scene_id}_advanced")
        self.nerf_dir = Path(f"data/nerf/{scene_id}")
        self.nerf_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_colmap_to_nerf(self):
        """Convert COLMAP format to NeRF training format"""
        print("Converting COLMAP data to NeRF format...")
        
        # Create NeRF directory structure
        (self.nerf_dir / "images").mkdir(exist_ok=True)
        (self.nerf_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        
        # Copy COLMAP sparse reconstruction
        sparse_src = self.colmap_dir / "sparse" / "0"
        sparse_dst = self.nerf_dir / "sparse" / "0"
        
        for file in ["cameras.txt", "images.txt", "points3D.txt"]:
            src_file = sparse_src / file
            if src_file.exists():
                shutil.copy2(src_file, sparse_dst / file)
        
        # Copy and organize images with proper naming
        src_images = self.colmap_dir / "images"
        dst_images = self.nerf_dir / "images"
        
        # Clear existing images
        for old_img in dst_images.glob("*"):
            old_img.unlink()
        
        # Copy images with sequential naming
        image_files = sorted(src_images.glob("*.jpg"))
        for i, img_file in enumerate(image_files):
            new_name = f"image_{i:04d}.jpg"
            dst_path = dst_images / new_name
            shutil.copy2(img_file, dst_path)
            print(f"Copied {img_file.name} -> {new_name}")
        
        print(f"Prepared NeRF training data in {self.nerf_dir}")
        return self.nerf_dir
    
    def create_transforms_json(self):
        """Create transforms.json for NeRF training with fixed file paths"""
        sparse_dir = self.nerf_dir / "sparse" / "0"
        
        # Read COLMAP data
        cameras = self.read_colmap_cameras(sparse_dir / "cameras.txt")
        images = self.read_colmap_images(sparse_dir / "images.txt")
        
        # Convert to NeRF format
        transforms = {
            "camera_angle_x": 0.8,  # Approximate field of view
            "frames": []
        }
        
        # Map original image names to new sequential names
        image_mapping = {}
        src_images = self.colmap_dir / "images"
        image_files = sorted(src_images.glob("*.jpg"))
        
        for i, img_file in enumerate(image_files):
            new_name = f"image_{i:04d}.jpg"
            image_mapping[img_file.name] = new_name
        
        # Create frame entries with correct file paths
        for img_name, img_data in images.items():
            if img_name in image_mapping:
                # Convert COLMAP pose to NeRF format
                transform_matrix = self.colmap_to_nerf_transform(
                    img_data['rotation'], img_data['translation']
                )
                
                frame = {
                    "file_path": f"./images/{image_mapping[img_name]}",
                    "transform_matrix": transform_matrix.tolist()
                }
                transforms["frames"].append(frame)
        
        # Save transforms.json
        transforms_file = self.nerf_dir / "transforms.json"
        with open(transforms_file, 'w') as f:
            json.dump(transforms, f, indent=2)
        
        print(f"Created {transforms_file} with {len(transforms['frames'])} frames")
        return transforms_file
    
    def read_colmap_cameras(self, cameras_file):
        """Read COLMAP cameras.txt"""
        cameras = {}
        with open(cameras_file, 'r') as f:
            for line in f:
                if not line.startswith('#') and line.strip():
                    parts = line.strip().split()
                    cameras[parts[0]] = {
                        'model': parts[1],
                        'width': int(parts[2]),
                        'height': int(parts[3]),
                        'params': [float(p) for p in parts[4:]]
                    }
        return cameras
    
    def read_colmap_images(self, images_file):
        """Read COLMAP images.txt"""
        images = {}
        with open(images_file, 'r') as f:
            for line in f:
                if not line.startswith('#') and line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 10:
                        qw, qx, qy, qz = [float(p) for p in parts[1:5]]
                        tx, ty, tz = [float(p) for p in parts[5:8]]
                        img_name = parts[9]
                        
                        # Convert quaternion to rotation matrix
                        R = self.quaternion_to_matrix([qw, qx, qy, qz])
                        
                        images[img_name] = {
                            'rotation': R,
                            'translation': np.array([tx, ty, tz])
                        }
        return images
    
    def quaternion_to_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
    
    def colmap_to_nerf_transform(self, R, t):
        """Convert COLMAP pose to NeRF transform matrix"""
        # NeRF uses camera-to-world transform
        c2w = np.eye(4)
        c2w[:3, :3] = R.T  # Transpose for camera-to-world
        c2w[:3, 3] = -R.T @ t  # Camera center
        
        # NeRF coordinate system conversion
        transform = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        
        return transform @ c2w
    
    def prepare_full_dataset(self):
        """Prepare complete dataset for NeRF training"""
        print(f"Preparing NeRF dataset for {self.scene_id}")
        
        # Convert COLMAP data
        self.convert_colmap_to_nerf()
        
        # Create transforms.json
        self.create_transforms_json()
        
        # Verify dataset
        self.verify_dataset()
        
        print(f"NeRF dataset ready: {self.nerf_dir}")
        return self.nerf_dir
    
    def verify_dataset(self):
        """Verify the prepared dataset"""
        # Check transforms.json
        transforms_file = self.nerf_dir / "transforms.json"
        if transforms_file.exists():
            with open(transforms_file, 'r') as f:
                transforms = json.load(f)
                print(f"  Transforms: {len(transforms['frames'])} frames")
        
        # Check images
        images_dir = self.nerf_dir / "images"
        image_count = len(list(images_dir.glob("*.jpg")))
        print(f"  Images: {image_count}")
        
        # Verify image files exist
        with open(transforms_file, 'r') as f:
            transforms = json.load(f)
            for frame in transforms["frames"]:
                img_path = self.nerf_dir / frame["file_path"].replace("./", "")
                if not img_path.exists():
                    print(f"  Warning: Missing image {img_path}")
                    return False
        
        print("  Dataset verification: Complete")
        return True

def main():
    import sys
    scene_id = sys.argv[1] if len(sys.argv) > 1 else "scene_001"
    
    preparer = NeRFDataPreparer(scene_id)
    dataset_dir = preparer.prepare_full_dataset()
    
    print(f"Dataset prepared for NeRF training: {dataset_dir}")

if __name__ == "__main__":
    main()