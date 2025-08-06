import cv2
import mediapipe as mp
import os
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
            keypoints_list.append([frame_count] + keypoints)

            if save_frames:
                annotated = frame.copy()
                mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if not os.path.exists(frame_dir):
                    os.makedirs(frame_dir)
                cv2.imwrite(os.path.join(frame_dir, f"frame_{frame_count:04d}.jpg"), annotated)
        print(f"Frame {frame_count}: landmarks found? {results.pose_landmarks is not None}")
        frame_count += 1

    if not cap.isOpened():
        print("Failed to open video!")

    df = pd.DataFrame(keypoints_list)
    df.to_csv(output_csv, index=False)
    print(f"Keypoints saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    video_path = "C:/Users/dhanu/OneDrive/Desktop/Projects/Badminton/Videos/Video-1.mp4"  # replace with actual filename
    extract_keypoints(video_path, "C:/Users/dhanu/OneDrive/Desktop/Projects/Badminton/Output/keypoints/sample1_keypoints.csv", save_frames=True, frame_dir="C:/Users/dhanu/OneDrive/Desktop/Projects/Badminton/Output/annotated_frames")