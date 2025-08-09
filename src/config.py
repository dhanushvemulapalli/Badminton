# File: C:/Users/dhanu/OneDrive/Desktop/Projects/Badminton/src/config.py

import os

class ProjectPaths:
    # Use current working directory as base, making it portable
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Input paths
    VIDEOS_DIR = os.path.join(BASE_DIR, "Videos")
    
    # Output paths
    OUTPUT_DIR = os.path.join(BASE_DIR, "Output")
    KEYPOINTS_DIR = os.path.join(OUTPUT_DIR, "keypoints")
    ANALYZED_DIR = os.path.join(OUTPUT_DIR, "analyzed")
    VIZ_3D_DIR = os.path.join(OUTPUT_DIR, "3d_viz")
    FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
    ANNOTATED_FRAMES_DIR = os.path.join(OUTPUT_DIR, "annotated_frames")
    
    @classmethod
    def create_directories(cls):
        """Create all necessary output directories"""
        directories = [
            cls.OUTPUT_DIR,
            cls.KEYPOINTS_DIR, 
            cls.ANALYZED_DIR, 
            cls.VIZ_3D_DIR,
            cls.FEATURES_DIR,
            cls.ANNOTATED_FRAMES_DIR
        ]
        
        for path in directories:
            os.makedirs(path, exist_ok=True)
            
    @classmethod
    def get_video_path(cls, video_filename):
        """Get full path for a video file"""
        return os.path.join(cls.VIDEOS_DIR, video_filename)
    
    @classmethod
    def get_keypoints_path(cls, video_number):
        """Get keypoints CSV path for a video number"""
        return os.path.join(cls.KEYPOINTS_DIR, f"video_{video_number}_keypoints.csv")
    
    @classmethod
    def get_analyzed_video_path(cls, video_number):
        """Get analyzed video output path"""
        return os.path.join(cls.ANALYZED_DIR, f"video_{video_number}_analyzed.mp4")

class AnalysisConfig:
    # Video processing
    FPS = 30
    
    # Movement thresholds (calibrated for badminton)
    SHOT_SPEED_THRESHOLD = 5.0      # Increased for actual badminton shot speeds
    REACTION_SPEED_THRESHOLD = 3.0   # Increased for realistic player reactions
    MIN_SUSTAINED_FRAMES = 5         # Increased to reduce false positives
    MAX_REALISTIC_SPEED = 15.0       # Increased for fast badminton movements
    MIN_CONFIDENCE = 0.6             # Slightly lower to capture more movements
    SMOOTHING_WINDOW = 5             # Increased for better noise reduction
    
    # Pose detection thresholds
    MIN_VISIBILITY_THRESHOLD = 0.5   # Minimum landmark visibility to use
    LUNGE_KNEE_ANGLE_THRESHOLD = 100 # Knee angle for lunge detection
    STANCE_WIDTH_MIN = 0.25          # Minimum stance width ratio
    
    # 3D Visualization settings
    VIZ_FIGURE_SIZE = (12, 9)
    VIZ_POINT_SIZE = 60
    VIZ_LINE_WIDTH = 3
