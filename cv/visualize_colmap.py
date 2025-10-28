import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def visualize_colmap_reconstruction(scene_id):
    colmap_dir = Path(f"data/colmap/{scene_id}_advanced")
    sparse_dir = colmap_dir / "sparse" / "0"
    
    # Read camera poses
    cameras = {}
    images = {}
    points_3d = []
    
    # Parse cameras.txt
    cameras_file = sparse_dir / "cameras.txt"
    if cameras_file.exists():
        with open(cameras_file, 'r') as f:
            for line in f:
                if not line.startswith('#') and line.strip():
                    parts = line.strip().split()
                    cameras[parts[0]] = {
                        'model': parts[1],
                        'width': int(parts[2]),
                        'height': int(parts[3])
                    }
    
    # Parse images.txt (camera poses)
    images_file = sparse_dir / "images.txt"
    if images_file.exists():
        with open(images_file, 'r') as f:
            for line in f:
                if not line.startswith('#') and line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 10:
                        qw, qx, qy, qz = [float(p) for p in parts[1:5]]
                        tx, ty, tz = [float(p) for p in parts[5:8]]
                        
                        # Convert quaternion to rotation matrix
                        R = quaternion_to_rotation_matrix([qw, qx, qy, qz])
                        
                        # Camera center (not translation)
                        camera_center = -R.T @ np.array([tx, ty, tz])
                        
                        images[parts[9]] = {
                            'position': camera_center,
                            'rotation': R,
                            'translation': [tx, ty, tz]
                        }
    
    # Parse points3D.txt
    points_file = sparse_dir / "points3D.txt"
    if points_file.exists():
        with open(points_file, 'r') as f:
            for line in f:
                if not line.startswith('#') and line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        x, y, z = [float(p) for p in parts[1:4]]
                        r, g, b = [int(p) for p in parts[4:7]]
                        points_3d.append([x, y, z, r, g, b])
    
    # Create 3D visualization
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D points
    if points_3d:
        points_3d = np.array(points_3d)
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  c=points_3d[:, 3:6]/255.0, s=1, alpha=0.6)
    
    # Plot camera positions
    camera_positions = []
    for img_name, img_data in images.items():
        pos = img_data['position']
        camera_positions.append(pos)
        ax.scatter(pos[0], pos[1], pos[2], c='red', s=100, marker='^')
        ax.text(pos[0], pos[1], pos[2], f'  {img_name[:8]}', fontsize=8)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title(f'COLMAP 3D Reconstruction - {scene_id}\n{len(points_3d)} points, {len(images)} cameras')
    
    # Equal aspect ratio
    if len(points_3d) > 0:
        max_range = np.array([points_3d[:,0].max()-points_3d[:,0].min(),
                             points_3d[:,1].max()-points_3d[:,1].min(),
                             points_3d[:,2].max()-points_3d[:,2].min()]).max() / 2.0
        mid_x = (points_3d[:,0].max()+points_3d[:,0].min()) * 0.5
        mid_y = (points_3d[:,1].max()+points_3d[:,1].min()) * 0.5
        mid_z = (points_3d[:,2].max()+points_3d[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save plot
    output_path = f'docs/{scene_id}_colmap_reconstruction.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"3D reconstruction visualization saved: {output_path}")
    plt.show()

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix"""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])

if __name__ == "__main__":
    import sys
    scene_id = sys.argv[1] if len(sys.argv) > 1 else "scene_001"
    visualize_colmap_reconstruction(scene_id)