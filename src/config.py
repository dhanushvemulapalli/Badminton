# File: C:/Users/dhanu/OneDrive/Desktop/Projects/Badminton/src/config.py

import os

class ProjectPaths:
    BASE_DIR = "C:/Users/dhanu/OneDrive/Desktop/Projects/Badminton"
    
    # Input paths
    VIDEOS_DIR = os.path.join(BASE_DIR, "Videos")
    
    # Output paths
    OUTPUT_DIR = os.path.join(BASE_DIR, "Output")
    KEYPOINTS_DIR = os.path.join(OUTPUT_DIR, "keypoints")
    ANALYZED_DIR = os.path.join(OUTPUT_DIR, "analyzed")
    VIZ_3D_DIR = os.path.join(OUTPUT_DIR, "3d_viz")
    FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
    
    @classmethod
    def create_directories(cls):
        """Create all necessary output directories"""
        for path in [cls.ANALYZED_DIR, cls.VIZ_3D_DIR]:
            os.makedirs(path, exist_ok=True)

class AnalysisConfig:
    # Video processing
    FPS = 30
    
    # Movement thresholds
    SHOT_SPEED_THRESHOLD = 2.0
    REACTION_SPEED_THRESHOLD = 2.5
    MIN_SUSTAINED_FRAMES = 3
    MAX_REALISTIC_SPEED = 10.0
    MIN_CONFIDENCE = 0.7
    SMOOTHING_WINDOW = 3
