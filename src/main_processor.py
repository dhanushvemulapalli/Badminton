# File: C:/Users/dhanu/OneDrive/Desktop/Projects/Badminton/src/main_processor.py

import os
from config import ProjectPaths, AnalysisConfig
from video_analyzer import BadmintonVideoAnalyzer
from analyze_motion import load_keypoints_from_csv
from pose_extractor import extract_keypoints  # Your existing function

def process_all_videos():
    """Process all 5 videos with complete analysis"""
    
    # Create necessary directories
    ProjectPaths.create_directories()
    
    video_files = [
        "Video-1.mp4", "Video-2.mp4", "Video-3.mp4", 
        "Video-4.mp4", "Video-5.mp4"
    ]
    
    analyzer = BadmintonVideoAnalyzer()
    
    for i, video_file in enumerate(video_files):
        print(f"Processing {video_file}...")
        
        # File paths
        video_path = os.path.join(ProjectPaths.VIDEOS_DIR, video_file)
        keypoints_csv = os.path.join(ProjectPaths.KEYPOINTS_DIR, f"video_{i+1}_keypoints.csv")
        output_video = os.path.join(ProjectPaths.ANALYZED_DIR, f"video_{i+1}_analyzed.mp4")
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Warning: {video_path} not found, skipping...")
            continue
        
        # Extract keypoints (if not already done)
        if not os.path.exists(keypoints_csv):
            print(f"Extracting keypoints for {video_file}...")
            extract_keypoints(video_path, keypoints_csv)
        
        # Load keypoints
        keypoints_data = load_keypoints_from_csv(keypoints_csv)
        
        # Create analysis video
        analyzer.render_analysis_video(video_path, keypoints_data, output_video)
        
        # Generate 3D visualizations for key frames
        key_frames = [0, len(keypoints_data)//4, len(keypoints_data)//2, 
                     3*len(keypoints_data)//4, len(keypoints_data)-1]
        
        for frame_idx in key_frames:
            viz_path = os.path.join(ProjectPaths.VIZ_3D_DIR, 
                                   f"video_{i+1}_frame_{frame_idx}_3d.png")
            analyzer.create_3d_visualization(keypoints_data, frame_idx, viz_path)
        
        print(f"✅ {video_file} processing complete!")
        print(f"   📹 Analyzed video: {output_video}")
        print(f"   📊 3D visualizations: {len(key_frames)} images created")

if __name__ == "__main__":
    process_all_videos()
