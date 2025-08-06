import pandas as pd
import numpy as np

# List of Mediapipe pose landmarks
POSE_LANDMARKS = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]

def get_landmark_xy(joints, name):
    idx = POSE_LANDMARKS.index(name)

    # print("Type of joints:", type(joints))
    # print("Shape of joints:", np.shape(joints))
    # print("Index:", idx)
    # print("joints[idx]:", joints[idx])

    x, y, _, visibility = joints[idx]
    return x, y

import math

def visibility_check(joints, names, threshold=0.5):
    for name in names:
        idx = POSE_LANDMARKS.index(name)
        if joints[idx][3] < threshold:
            return False
    return True

def angle(a, b, c):
    """Returns angle ABC (in degrees) between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def stride_length(joints):
    left = get_landmark_xy(joints, 'left_ankle')
    right = get_landmark_xy(joints, 'right_ankle')
    return abs(left[0] - right[0])

def is_lunge(joints):
    if not visibility_check(joints, ['left_hip', 'left_knee', 'left_ankle',
                                     'right_hip', 'right_knee', 'right_ankle']):
        return False

    # Left leg lunge
    left_hip = get_landmark_xy(joints, 'left_hip')
    left_knee = get_landmark_xy(joints, 'left_knee')
    left_ankle = get_landmark_xy(joints, 'left_ankle')
    left_knee_angle = angle(left_hip, left_knee, left_ankle)

    # Right leg lunge
    right_hip = get_landmark_xy(joints, 'right_hip')
    right_knee = get_landmark_xy(joints, 'right_knee')
    right_ankle = get_landmark_xy(joints, 'right_ankle')
    right_knee_angle = angle(right_hip, right_knee, right_ankle)

    # Heuristic conditions:
    # - One leg bent (angle < 120), one relatively straight (angle > 150)
    left_lunge = left_knee_angle < 120 and right_knee_angle > 150
    right_lunge = right_knee_angle < 120 and left_knee_angle > 150

    if stride_length(joints) < 0.1:
        return False

    return left_lunge or right_lunge

def is_split_step(prev, curr, next_):
    try:
        # Track Y of hips (center of mass approximation)
        prev_center = (get_landmark_xy(prev, 'left_hip')[1] + get_landmark_xy(prev, 'right_hip')[1]) / 2
        curr_center = (get_landmark_xy(curr, 'left_hip')[1] + get_landmark_xy(curr, 'right_hip')[1]) / 2
        next_center = (get_landmark_xy(next_, 'left_hip')[1] + get_landmark_xy(next_, 'right_hip')[1]) / 2

        # Sudden dip and rise (simple second derivative like logic)
        drop = curr_center - prev_center
        rise = next_center - curr_center

        # Feet distance
        left_ankle = np.array(get_landmark_xy(curr, 'left_ankle'))
        right_ankle = np.array(get_landmark_xy(curr, 'right_ankle'))
        feet_distance = np.linalg.norm(left_ankle - right_ankle)

        return drop > 0.015 and rise < -0.015 and feet_distance > 0.1
    except:
        return False

def detect_split_steps(frames):
    split_frames = []

    for i in range(1, len(frames) - 1):
        if is_split_step(frames[i - 1], frames[i], frames[i + 1]):
            split_frames.append(i)

    return split_frames


def detect_lunges(frames):
    lunge_frames = []

    for i, joints in enumerate(frames):
        if is_lunge(joints):
            lunge_frames.append(i)

    return lunge_frames

def load_keypoints(csv_path):
    # print("Called load_keypoints function")

    df = pd.read_csv(csv_path)
    # Remove frame index column
    frames = df.iloc[:, 1:].to_numpy().reshape(-1, 33, 4)  # (num_frames, 33 joints, 4 coords)
    return frames

# Example usage
if __name__ == "__main__":
    # print("Entered main function")

    frames = load_keypoints("C:/Users/dhanu/OneDrive/Desktop/Projects/Badminton/Output/keypoints/sample1_keypoints.csv")
    print("Shape of keypoints:", frames.shape)  # Expect (N, 33, 4)

    lunges = detect_lunges(frames)
    print(f"Lunges detected at frames: {lunges}")

    split_steps = detect_split_steps(frames)
    print(f"Split Steps detected at frames: {split_steps}")



    # df = pd.read_csv('C:/Users/dhanu/OneDrive/Desktop/Projects/Badminton/Output/keypoints/sample1_keypoints.csv')
    # frames = df.values.tolist()
    # lunge_frames = detect_lunges(frames)
    # print("Detected lunges at frames:", lunge_frames)
