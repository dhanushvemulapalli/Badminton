from pose_normalizer import normalize_pose  # assumes you have this implemented
from pose_features import extract_pose_features
import mediapipe as mp


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

import cv2
import os
import numpy as np
import pandas as pd

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(video_path, output_csv, save_frames=False, frame_dir=""):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    keypoints_list = []

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
            normalized_xyz = normalize_pose(xyzv[:, :3])  # Returns (33, 3)
            
            # Combine normalized xyz with original visibility
            normalized_xyzv = np.hstack([normalized_xyz, xyzv[:, 3:4]])  # (33, 4)
            
            # Add frame number and flatten for CSV storage
            row_data = [frame_count] + normalized_xyzv.flatten().tolist()
            keypoints_list.append(row_data)

            # Extract pose features
            try:
                pose_features = extract_pose_features(normalized_xyzv)
                print(f"Frame {frame_count} features: {pose_features}")
            except Exception as e:
                print(f"Error extracting features for frame {frame_count}: {e}")

            # Optionally save annotated frame
            if save_frames:
                annotated = frame.copy()
                mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if not os.path.exists(frame_dir):
                    os.makedirs(frame_dir)
                cv2.imwrite(os.path.join(frame_dir, f"frame_{frame_count:04d}.jpg"), annotated)

        print(f"Frame {frame_count}: landmarks found? {results.pose_landmarks is not None}")
        frame_count += 1

    cap.release()

    # Create column names: frame + 33 joints * 4 values each
    columns = ["frame"] + [f"joint_{i}_{axis}" for i in range(33) for axis in ["x", "y", "z", "v"]]
    
    if keypoints_list:
        df = pd.DataFrame(keypoints_list, columns=columns)
        df.to_csv(output_csv, index=False)
        print(f"Keypoints saved to {output_csv}")
    else:
        print("No keypoints detected in video!")

# Example usage
if __name__ == "__main__":
    video_path = "C:/Users/dhanu/OneDrive/Desktop/Projects/Badminton/Videos/Video-2.mp4"
    extract_keypoints(
        video_path,
        "C:/Users/dhanu/OneDrive/Desktop/Projects/Badminton/Output/keypoints/sample1_keypoints.csv",
        save_frames=True,
        frame_dir="C:/Users/dhanu/OneDrive/Desktop/Projects/Badminton/Output/annotated_frames"
    )
