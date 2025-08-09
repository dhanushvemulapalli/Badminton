# File: C:/Users/dhanu/OneDrive/Desktop/Projects/Badminton/src/main_processor.py

import os
from config import ProjectPaths, AnalysisConfig
from video_analyzer import BadmintonVideoAnalyzer
from analyze_motion import load_keypoints_from_csv
from pose_extractor import extract_keypoints  # Your existing function

def process_all_videos():
    """Process all 5 videos with complete analysis"""
    ProjectPaths.create_directories()
    
    video_files = [
        "Video-1.mp4", "Video-2.mp4", "Video-3.mp4",
        "Video-4.mp4", "Video-5.mp4"
    ]

    for i, video_file in enumerate(video_files, 1):
        print(f"\n{'='*50}")
        print(f"🏸 Processing {video_file} ({i}/{len(video_files)})...")
        print(f"{'='*50}")

        # File paths using new config methods
        video_path = ProjectPaths.get_video_path(video_file)
        keypoints_csv = ProjectPaths.get_keypoints_path(i)
        output_video = ProjectPaths.get_analyzed_video_path(i)

        # Check if video exists
        if not os.path.exists(video_path):
            print(f"⚠️ Warning: {video_path} not found, skipping...")
            continue

        try:
            # CREATE NEW ANALYZER INSTANCE FOR EACH VIDEO
            # This prevents cached data from previous videos
            analyzer = BadmintonVideoAnalyzer()
            
            # Extract keypoints specifically for THIS video
            if not os.path.exists(keypoints_csv):
                print(f"📊 Extracting keypoints for {video_file}...")
                extract_keypoints(video_path, keypoints_csv, save_frames=False)  # Don't save frames to avoid conflicts
            else:
                print(f"✓ Keypoints already exist: {keypoints_csv}")

            # Load keypoints for THIS specific video
            print(f"📥 Loading keypoints data for {video_file}...")
            keypoints_data = load_keypoints_from_csv(keypoints_csv)
            
            if len(keypoints_data) == 0:
                print(f"❌ No keypoints data found for {video_file}")
                continue

            # Create analysis video with the correct data pairing
            print(f"🎥 Creating analysis video for {video_file}...")
            analyzer.render_analysis_video(video_path, keypoints_data, output_video)

            print(f"✅ {video_file} processing complete!")
            print(f" 📹 Analyzed video: {output_video}")
            print(f" 📋 Keypoints: {len(keypoints_data)} frames processed")

        except Exception as e:
            print(f"❌ Error processing {video_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*50}")
    print(f"🏁 Processing Summary:")
    # print(f"   📊 Videos processed: {processed_count}/{len(video_files)}")
    # print(f"   📁 Output directory: {ProjectPaths.OUTPUT_DIR}")
    print(f"{'='*50}")

if __name__ == "__main__":
    process_all_videos()
