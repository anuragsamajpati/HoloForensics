import torch
import torch.nn as nn
import numpy as np
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt

class TinyNeRF(nn.Module):
    def __init__(self, pos_embed_dims=10, dir_embed_dims=4, hidden_dim=256):
        super().__init__()
        
        # Positional encoding dimensions
        self.pos_embed_dims = pos_embed_dims
        self.dir_embed_dims = dir_embed_dims
        
        # Input dimensions after positional encoding
        pos_input_dim = 3 + 3 * 2 * pos_embed_dims  # xyz + encoded
        dir_input_dim = 3 + 3 * 2 * dir_embed_dims  # direction + encoded
        
        # Network layers
        self.pos_encoder = PositionalEncoder(pos_embed_dims)
        self.dir_encoder = PositionalEncoder(dir_embed_dims)
        
        # Position network (outputs density + features)
        self.pos_net = nn.Sequential(
            nn.Linear(pos_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1)  # features + density
        )
        
        # Color network (takes features + direction)
        self.color_net = nn.Sequential(
            nn.Linear(hidden_dim + dir_input_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 3),  # RGB
            nn.Sigmoid()
        )
    
    def forward(self, positions, directions):
        # Encode positions and directions
        pos_encoded = self.pos_encoder(positions)
        dir_encoded = self.dir_encoder(directions)
        
        # Position network
        pos_out = self.pos_net(pos_encoded)
        density = torch.relu(pos_out[..., 0])  # Density (sigma)
        features = pos_out[..., 1:]  # Feature vector
        
        # Color network
        color_input = torch.cat([features, dir_encoded], dim=-1)
        color = self.color_net(color_input)
        
        return color, density

class PositionalEncoder:
    def __init__(self, embed_dims):
        self.embed_dims = embed_dims
        
    def __call__(self, x):
        # x shape: (..., 3)
        encoded = [x]
        
        for i in range(self.embed_dims):
            freq = 2.0 ** i
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
            
        return torch.cat(encoded, dim=-1)

class MacNeRFTrainer:
    def __init__(self, scene_id):
        self.scene_id = scene_id
        self.nerf_dir = Path(f"data/nerf/{scene_id}")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def load_data(self):
        """Load training data from NeRF format"""
        transforms_file = self.nerf_dir / "transforms.json"
        
        with open(transforms_file, 'r') as f:
            transforms = json.load(f)
        
        images = []
        poses = []
        
        for frame in transforms["frames"]:
            # Load image
            img_path = self.nerf_dir / frame["file_path"].replace("./", "")
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            images.append(img)
            
            # Load pose
            pose = np.array(frame["transform_matrix"])
            poses.append(pose)
        
        self.images = np.array(images)
        self.poses = np.array(poses)
        self.H, self.W = images[0].shape[:2]
        
        print(f"Loaded {len(images)} images of size {self.H}x{self.W}")
        return self.images, self.poses
    
    def get_rays(self, pose, focal=400):
        """Generate camera rays for given pose"""
        H, W = self.H, self.W
        
        # Camera intrinsics (simplified)
        cx, cy = W//2, H//2
        
        # Pixel coordinates - ensure they're on the correct device
        i, j = torch.meshgrid(
            torch.arange(W, device=self.device), 
            torch.arange(H, device=self.device), 
            indexing='xy'
        )
        
        # Ray directions in camera space
        dirs = torch.stack([
            (i - cx) / focal,
            -(j - cy) / focal,  # Negative for image convention
            -torch.ones_like(i)
        ], dim=-1).float()
        
        # Transform to world space - ensure pose is on correct device
        pose = pose.to(self.device)
        dirs = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        
        # Ray origins (camera center)
        origins = pose[:3, 3].expand(dirs.shape)
        
        return origins, dirs
    
    def render_rays(self, model, origins, directions, near=0.1, far=10.0, n_samples=64):
        """Render rays through the NeRF model"""
        batch_size = origins.shape[0]
        
        # Sample points along rays - ensure on correct device
        t_vals = torch.linspace(near, far, n_samples, device=self.device)
        t_vals = t_vals.expand(batch_size, n_samples)
        
        # Add noise for regularization during training
        noise = torch.rand_like(t_vals) * (far - near) / n_samples
        t_vals = t_vals + noise
        
        # Compute 3D points
        points = origins[..., None, :] + directions[..., None, :] * t_vals[..., :, None]
        
        # Expand directions to match points
        dirs_expanded = directions[..., None, :].expand_as(points)
        
        # Reshape for network
        points_flat = points.reshape(-1, 3)
        dirs_flat = dirs_expanded.reshape(-1, 3)
        
        # Forward pass
        colors, densities = model(points_flat, dirs_flat)
        
        # Reshape back
        colors = colors.reshape(*points.shape[:-1], 3)
        densities = densities.reshape(*points.shape[:-1])
        
        # Volume rendering
        deltas = t_vals[..., 1:] - t_vals[..., :-1]
        deltas = torch.cat([deltas, torch.full_like(deltas[..., :1], 1e10)], dim=-1)
        
        alphas = 1.0 - torch.exp(-densities * deltas)
        transmittance = torch.cumprod(1.0 - alphas + 1e-10, dim=-1)
        transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]], dim=-1)
        
        weights = alphas * transmittance
        rgb = torch.sum(weights[..., None] * colors, dim=-2)
        
        return rgb, weights, densities
    
    def train_simple(self, num_epochs=1000, batch_size=1024):
        """Simple training loop for Mac"""
        # Load data
        images, poses = self.load_data()
        
        # Convert to tensors and move to device
        images = torch.from_numpy(images).float().to(self.device)
        poses = torch.from_numpy(poses).float().to(self.device)
        
        # Create model
        model = TinyNeRF().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        
        print("Starting NeRF training...")
        
        for epoch in range(num_epochs):
            # Sample random rays
            img_idx = torch.randint(len(images), (batch_size//100,), device=self.device)
            
            total_loss = 0
            
            for idx in img_idx:
                # Get rays for this image
                origins, directions = self.get_rays(poses[idx])
                target_img = images[idx]
                
                # Sample random pixels
                coords = torch.randint(0, min(self.H, self.W), (100, 2), device=self.device)
                
                ray_origins = origins[coords[:, 1], coords[:, 0]]
                ray_dirs = directions[coords[:, 1], coords[:, 0]]
                target_colors = target_img[coords[:, 1], coords[:, 0]]
                
                # Render
                rgb, _, _ = self.render_rays(model, ray_origins, ray_dirs)
                
                # Loss
                loss = torch.mean((rgb - target_colors) ** 2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(img_idx):.6f}")
        
        # Save model
        model_path = self.nerf_dir / "tiny_nerf_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")
        
        return model
    
    def render_novel_view(self, model, pose, focal=400):
        """Render a novel view from the trained model"""
        model.eval()
        
        with torch.no_grad():
            # Get rays for the pose
            origins, directions = self.get_rays(pose, focal)
            
            # Render in smaller batches to avoid memory issues
            rendered_img = torch.zeros(self.H, self.W, 3, device=self.device)
            
            batch_size = 2048  # Smaller batches for memory efficiency
            
            for i in range(0, self.H, 32):
                for j in range(0, self.W, 32):
                    # Process 32x32 tiles
                    end_i = min(i + 32, self.H)
                    end_j = min(j + 32, self.W)
                    
                    tile_origins = origins[i:end_i, j:end_j].reshape(-1, 3)
                    tile_dirs = directions[i:end_i, j:end_j].reshape(-1, 3)
                    
                    # Render tile in smaller batches
                    tile_rgb = []
                    for b in range(0, len(tile_origins), batch_size):
                        batch_origins = tile_origins[b:b+batch_size]
                        batch_dirs = tile_dirs[b:b+batch_size]
                        
                        rgb, _, _ = self.render_rays(model, batch_origins, batch_dirs)
                        tile_rgb.append(rgb)
                    
                    tile_rgb = torch.cat(tile_rgb, dim=0)
                    tile_rgb = tile_rgb.reshape(end_i - i, end_j - j, 3)
                    
                    rendered_img[i:end_i, j:end_j] = tile_rgb
        
        return rendered_img.cpu().numpy()

def main():
    import sys
    scene_id = sys.argv[1] if len(sys.argv) > 1 else "scene_001"
    
    trainer = MacNeRFTrainer(scene_id)
    
    # Check if data exists
    if not (Path(f"data/nerf/{scene_id}") / "transforms.json").exists():
        print("Please run prepare_nerf_data.py first")
        return
    
    # Train model
    model = trainer.train_simple(num_epochs=500)
    print("NeRF training completed!")
    
    # Test novel view rendering
    print("Rendering test view...")
    test_pose = trainer.poses[0]  # Use first camera pose as test
    rendered_img = trainer.render_novel_view(model, torch.from_numpy(test_pose).float())
    
    # Save rendered image
    output_path = f"docs/{scene_id}_nerf_render.png"
    plt.figure(figsize=(10, 8))
    plt.imshow(rendered_img)
    plt.title(f"NeRF Novel View - {scene_id}")
    plt.axis('off')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Novel view saved: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()