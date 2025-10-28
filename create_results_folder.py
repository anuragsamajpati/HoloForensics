#!/usr/bin/env python3
"""
HoloForensics Results Organizer
Creates organized folder structure after analysis completion
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path

def create_results_folder(scene_id="scene_001"):
    """Create organized results folder structure after analysis completion"""
    
    base_dir = Path("/Users/anuragsamajpati/Desktop/holoforensics")
    data_dir = base_dir / "data"
    
    # Create timestamp for this analysis run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = base_dir / "results" / f"{scene_id}_{timestamp}"
    
    # Create main results structure
    folders_to_create = [
        results_dir / "raw_data",
        results_dir / "detections", 
        results_dir / "keyframes",
        results_dir / "3d_reconstruction",
        results_dir / "nerf_renders",
        results_dir / "analysis_reports",
        results_dir / "metadata"
    ]
    
    for folder in folders_to_create:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {folder}")
    
    # Copy and organize existing data
    try:
        # Copy scene metadata
        metadata_file = data_dir / f"{scene_id}_metadata.json"
        if metadata_file.exists():
            shutil.copy2(metadata_file, results_dir / "metadata" / f"{scene_id}_metadata.json")
            print(f"📋 Copied metadata: {metadata_file}")
        
        # Copy detections if they exist
        detections_src = data_dir / "detections" / scene_id
        if detections_src.exists():
            shutil.copytree(detections_src, results_dir / "detections" / scene_id, dirs_exist_ok=True)
            print(f"🔍 Copied detections: {detections_src}")
        
        # Copy keyframes if they exist
        keyframes_src = data_dir / "keyframes" / scene_id
        if keyframes_src.exists():
            shutil.copytree(keyframes_src, results_dir / "keyframes" / scene_id, dirs_exist_ok=True)
            print(f"🖼️ Copied keyframes: {keyframes_src}")
        
        # Copy COLMAP data if it exists
        colmap_src = data_dir / "colmap" / scene_id
        if colmap_src.exists():
            shutil.copytree(colmap_src, results_dir / "3d_reconstruction" / scene_id, dirs_exist_ok=True)
            print(f"🏗️ Copied COLMAP data: {colmap_src}")
        
        # Copy NeRF data if it exists
        nerf_src = data_dir / "nerf" / scene_id
        if nerf_src.exists():
            shutil.copytree(nerf_src, results_dir / "nerf_renders" / scene_id, dirs_exist_ok=True)
            print(f"🎨 Copied NeRF renders: {nerf_src}")
        
        # Copy raw data if it exists
        raw_src = data_dir / "raw" / scene_id
        if raw_src.exists():
            shutil.copytree(raw_src, results_dir / "raw_data" / scene_id, dirs_exist_ok=True)
            print(f"📹 Copied raw data: {raw_src}")
    
    except Exception as e:
        print(f"⚠️ Warning copying data: {e}")
    
    # Create analysis summary
    create_analysis_summary(results_dir, scene_id)
    
    print(f"\n🎉 Results folder created: {results_dir}")
    print(f"📁 Total size: {get_folder_size(results_dir):.2f} MB")
    
    return results_dir

def create_analysis_summary(results_dir, scene_id):
    """Create analysis summary report"""
    
    summary = {
        "scene_id": scene_id,
        "analysis_date": datetime.now().isoformat(),
        "results_location": str(results_dir),
        "folder_structure": {
            "raw_data": "Original video files and input data",
            "detections": "Object detection results (JSON, CSV, annotated images)",
            "keyframes": "Extracted keyframes from video analysis", 
            "3d_reconstruction": "COLMAP camera poses and 3D point clouds",
            "nerf_renders": "Neural Radiance Field rendered outputs",
            "analysis_reports": "Generated forensic analysis reports",
            "metadata": "Scene metadata and processing information"
        },
        "file_counts": {},
        "processing_info": {}
    }
    
    # Count files in each directory
    for folder_name in summary["folder_structure"].keys():
        folder_path = results_dir / folder_name
        if folder_path.exists():
            file_count = sum(1 for _ in folder_path.rglob("*") if _.is_file())
            summary["file_counts"][folder_name] = file_count
    
    # Save summary
    summary_file = results_dir / "analysis_reports" / f"{scene_id}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Created analysis summary: {summary_file}")

def get_folder_size(folder_path):
    """Calculate folder size in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB

if __name__ == "__main__":
    print("🔬 HoloForensics Results Organizer")
    print("=" * 50)
    
    # Check if analysis is complete
    data_dir = Path("/Users/anuragsamajpati/Desktop/holoforensics/data")
    scene_id = "scene_001"
    
    if (data_dir / f"{scene_id}_metadata.json").exists():
        print(f"📋 Found scene metadata for {scene_id}")
        results_folder = create_results_folder(scene_id)
        print(f"\n✅ Results organized in: {results_folder}")
    else:
        print(f"❌ No metadata found for {scene_id}")
        print("⏳ Wait for analysis to complete first")
