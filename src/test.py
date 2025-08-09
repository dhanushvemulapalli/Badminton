"""
Test script for the Badminton Pose Analysis System.
This script performs basic system checks and validates dependencies.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print(f"✅ MediaPipe: {mp.__version__}")
    except ImportError as e:
        print(f"❌ MediaPipe failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib failed: {e}")
        return False
    
    return True

def test_camera():
    """Test camera access (optional)."""
    print("\n📹 Testing camera access...")
    
    try:
        import cv2
        import mediapipe as mp
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️ Camera not available (this is optional)")
            return True
        
        pose = mp.solutions.pose.Pose()
        
        # Test one frame
        ret, frame = cap.read()
        if ret:
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarks_detected = results.pose_landmarks is not None
            print(f"✅ Camera test successful, pose landmarks: {landmarks_detected}")
        else:
            print("⚠️ Could not read from camera")
        
        cap.release()
        pose.close()
        return True
        
    except Exception as e:
        print(f"⚠️ Camera test failed: {e}")
        return True  # Camera is optional

def test_config():
    """Test configuration system."""
    print("\n⚙️ Testing configuration...")
    
    try:
        # Add src to path if needed
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        
        from config import ProjectPaths, AnalysisConfig
        
        print(f"✅ Base directory: {ProjectPaths.BASE_DIR}")
        print(f"✅ Videos directory: {ProjectPaths.VIDEOS_DIR}")
        print(f"✅ Output directory: {ProjectPaths.OUTPUT_DIR}")
        
        # Test directory creation
        ProjectPaths.create_directories()
        print("✅ Directory creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🏸 Badminton Pose Analysis System - Test Suite")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test configuration
    if not test_config():
        all_passed = False
    
    # Test camera (optional)
    test_camera()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! System is ready to use.")
        print("\nTo run the analysis:")
        print("python main.py")
    else:
        print("❌ Some tests failed. Please install missing dependencies:")
        print("pip install -r requirements.txt")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
