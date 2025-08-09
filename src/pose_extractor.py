from pose_normalizer import normalize_pose
from pose_features import extract_pose_features
from config import ProjectPaths, AnalysisConfig
import mediapipe as mp
import cv2
import os
import numpy as np
import pandas as pd

# Initialize MediaPipe once
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(video_path, output_csv, save_frames=False, frame_dir=""):
    """Extract pose keypoints from video and save to CSV"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Initialize pose detection
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        frame_count = 0
        keypoints_list = []
        
        print(f"Processing video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                keypoints = []
                for lm in landmarks:
                    keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
                
                # Reshape to 33x4 (x, y, z, visibility)
                xyzv = np.array(keypoints).reshape((33, 4))
                
                # Normalize only the xyz coordinates
                try:
                    normalized_xyz = normalize_pose(xyzv[:, :3])  # Returns (33, 3)
                    
                    # Combine normalized xyz with original visibility
                    normalized_xyzv = np.hstack([normalized_xyz, xyzv[:, 3:4]])  # (33, 4)
                    
                    # Add frame number and flatten for CSV storage
                    row_data = [frame_count] + normalized_xyzv.flatten().tolist()
                    keypoints_list.append(row_data)

                    # Extract pose features (optional, for debugging)
                    if frame_count % 30 == 0:  # Only print every 30 frames to reduce spam
                        try:
                            pose_features = extract_pose_features(normalized_xyzv)
                            print(f"Frame {frame_count}/{total_frames}: Pose features extracted")
                        except Exception as e:
                            print(f"Warning: Could not extract features for frame {frame_count}: {e}")

                except Exception as e:
                    print(f"Warning: Could not normalize pose for frame {frame_count}: {e}")
                    continue

                # Optionally save annotated frame
                if save_frames:
                    if not frame_dir:
                        frame_dir = ProjectPaths.ANNOTATED_FRAMES_DIR
                    
                    annotated = frame.copy()
                    mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    os.makedirs(frame_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(frame_dir, f"frame_{frame_count:04d}.jpg"), annotated)

            # Progress indicator
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
                
            frame_count += 1

        cap.release()

        # Create column names: frame + 33 joints * 4 values each
        columns = ["frame"] + [f"joint_{i}_{axis}" for i in range(33) for axis in ["x", "y", "z", "v"]]
        
        if keypoints_list:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            
            df = pd.DataFrame(keypoints_list, columns=columns)
            df.to_csv(output_csv, index=False)
            print(f"✅ Keypoints saved to {output_csv}")
            print(f"📊 Processed {len(keypoints_list)} frames with pose data")
        else:
            print("❌ No keypoints detected in video! Check video quality and lighting.")

# Example usage
if __name__ == "__main__":
    # Example: Extract keypoints from a single video
    video_path = ProjectPaths.get_video_path("Video-1.mp4")
    output_csv = ProjectPaths.get_keypoints_path(1)
    
    if os.path.exists(video_path):
        extract_keypoints(
            video_path,
            output_csv,
            save_frames=True,
            frame_dir=ProjectPaths.ANNOTATED_FRAMES_DIR
        )
    else:
        print(f"❌ Video not found: {video_path}")
        print("Please place your video files in the Videos/ directory")
