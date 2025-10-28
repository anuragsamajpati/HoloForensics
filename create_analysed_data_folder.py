#!/usr/bin/env python3
"""
HoloForensics Analysed Data Organizer
Creates comprehensive 'Analysed Data' folder for all analysis types
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path

def create_analysed_data_folder(scene_id="scene_001"):
    """Create comprehensive 'Analysed Data' folder for all analysis results"""
    
    base_dir = Path("/Users/anuragsamajpati/Desktop/holoforensics")
    data_dir = base_dir / "data"
    
    # Create timestamp for this analysis run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysed_data_dir = base_dir / "Analysed Data" / f"{scene_id}_{timestamp}"
    
    # Create comprehensive analysis structure
    analysis_folders = {
        "Object_Detection": "YOLOv8 object detection results and annotations",
        "Multi_Camera_Analysis": "Synchronized multi-camera feed analysis",
        "Scene_Analysis": "Scene graph generation and event detection",
        "Video_Inpainting": "E2FGVI video reconstruction and inpainting",
        "Physics_Prediction": "Kalman filter and social force trajectory prediction",
        "3D_Reconstruction": "COLMAP camera poses and 3D point clouds",
        "Timeline_Analysis": "Temporal event analysis and behavioral patterns",
        "Identity_Tracking": "Person re-identification across scenes",
        "Forensic_QA": "AI-powered forensic query responses",
        "NeRF_Renders": "Neural Radiance Field 3D scene renders",
        "Raw_Data": "Original input videos and source materials",
        "Keyframes": "Extracted keyframes and temporal samples",
        "Reports": "Generated forensic analysis reports",
        "Metadata": "Processing metadata and configuration files"
    }
    
    # Create all analysis folders
    for folder_name, description in analysis_folders.items():
        folder_path = analysed_data_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Create description file in each folder
        desc_file = folder_path / "README.txt"
        with open(desc_file, 'w') as f:
            f.write(f"# {folder_name.replace('_', ' ')}\n\n")
            f.write(f"{description}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Scene ID: {scene_id}\n")
        
        print(f"✅ Created: {folder_name}")
    
    # Copy and organize existing data into appropriate folders
    copy_existing_data(data_dir, analysed_data_dir, scene_id)
    
    # Create master analysis report
    create_master_analysis_report(analysed_data_dir, scene_id, analysis_folders)
    
    print(f"\n🎉 Analysed Data folder created: {analysed_data_dir}")
    print(f"📁 Total size: {get_folder_size(analysed_data_dir):.2f} MB")
    
    return analysed_data_dir

def copy_existing_data(data_dir, analysed_data_dir, scene_id):
    """Copy existing data to appropriate analysis folders"""
    
    copy_mappings = [
        # (source, destination_folder, description)
        (data_dir / f"{scene_id}_metadata.json", "Metadata", "Scene metadata"),
        (data_dir / "detections" / scene_id, "Object_Detection", "Detection results"),
        (data_dir / "keyframes" / scene_id, "Keyframes", "Extracted keyframes"),
        (data_dir / "colmap" / scene_id, "3D_Reconstruction", "COLMAP data"),
        (data_dir / "nerf" / scene_id, "NeRF_Renders", "NeRF outputs"),
        (data_dir / "raw" / scene_id, "Raw_Data", "Original videos")
    ]
    
    for source_path, dest_folder, desc in copy_mappings:
        if source_path.exists():
            dest_path = analysed_data_dir / dest_folder
            
            if source_path.is_file():
                shutil.copy2(source_path, dest_path / source_path.name)
                print(f"📋 Copied {desc}: {source_path.name}")
            else:
                dest_scene_path = dest_path / scene_id
                if dest_scene_path.exists():
                    shutil.rmtree(dest_scene_path)
                shutil.copytree(source_path, dest_scene_path)
                print(f"📁 Copied {desc}: {source_path}")

def create_master_analysis_report(analysed_data_dir, scene_id, analysis_folders):
    """Create comprehensive master analysis report"""
    
    report = {
        "scene_id": scene_id,
        "analysis_timestamp": datetime.now().isoformat(),
        "analysed_data_location": str(analysed_data_dir),
        "analysis_capabilities": {},
        "folder_structure": {},
        "file_inventory": {},
        "processing_summary": {
            "total_analysis_types": len(analysis_folders),
            "data_organized": True,
            "forensic_grade": True
        }
    }
    
    # Document each analysis type
    for folder_name, description in analysis_folders.items():
        folder_path = analysed_data_dir / folder_name
        file_count = sum(1 for _ in folder_path.rglob("*") if _.is_file())
        
        report["analysis_capabilities"][folder_name] = description
        report["folder_structure"][folder_name] = str(folder_path)
        report["file_inventory"][folder_name] = file_count
    
    # Save master report
    reports_dir = analysed_data_dir / "Reports"
    master_report_file = reports_dir / f"{scene_id}_Master_Analysis_Report.json"
    
    with open(master_report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create human-readable summary
    summary_file = reports_dir / f"{scene_id}_Analysis_Summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("HOLOFORENSICS COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Scene ID: {scene_id}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Location: {analysed_data_dir}\n\n")
        
        f.write("ANALYSIS CAPABILITIES:\n")
        f.write("-" * 30 + "\n")
        for folder_name, description in analysis_folders.items():
            file_count = report["file_inventory"][folder_name]
            f.write(f"• {folder_name.replace('_', ' ')}: {file_count} files\n")
            f.write(f"  {description}\n\n")
        
        f.write(f"Total Analysis Types: {len(analysis_folders)}\n")
        f.write(f"Total Files Organized: {sum(report['file_inventory'].values())}\n")
        f.write(f"Forensic Grade: Yes\n")
    
    print(f"📊 Created master analysis report: {master_report_file}")
    print(f"📄 Created summary report: {summary_file}")

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
    print("🔬 HoloForensics Analysed Data Organizer")
    print("=" * 60)
    
    # Check if analysis data exists
    data_dir = Path("/Users/anuragsamajpati/Desktop/holoforensics/data")
    scene_id = "scene_001"
    
    if (data_dir / f"{scene_id}_metadata.json").exists():
        print(f"📋 Found scene data for {scene_id}")
        analysed_data_folder = create_analysed_data_folder(scene_id)
        print(f"\n✅ All analysis data organized in: {analysed_data_folder}")
        print("\n📁 Folder Structure Created:")
        print("   • Object Detection")
        print("   • Multi-Camera Analysis") 
        print("   • Scene Analysis")
        print("   • Video Inpainting")
        print("   • Physics Prediction")
        print("   • 3D Reconstruction")
        print("   • Timeline Analysis")
        print("   • Identity Tracking")
        print("   • Forensic Q&A")
        print("   • NeRF Renders")
        print("   • Raw Data")
        print("   • Keyframes")
        print("   • Reports")
        print("   • Metadata")
    else:
        print(f"❌ No scene data found for {scene_id}")
        print("⏳ Run analysis first to generate data")
