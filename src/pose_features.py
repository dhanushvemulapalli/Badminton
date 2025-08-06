# pose_features.py

import numpy as np

# MediaPipe pose indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

def compute_angle(a, b, c):
    """Returns angle (in degrees) at point b formed by vectors ba and bc."""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_pose_features(pose_landmarks):
    """
    pose_landmarks: np.array of shape (33, 3)
    Returns: dict of joint angles and body distances
    """

    features = {}

    # Joint angles
    features['left_elbow_angle'] = compute_angle(
        pose_landmarks[LEFT_SHOULDER],
        pose_landmarks[LEFT_ELBOW],
        pose_landmarks[LEFT_WRIST]
    )
    features['right_elbow_angle'] = compute_angle(
        pose_landmarks[RIGHT_SHOULDER],
        pose_landmarks[RIGHT_ELBOW],
        pose_landmarks[RIGHT_WRIST]
    )
    features['left_knee_angle'] = compute_angle(
        pose_landmarks[LEFT_HIP],
        pose_landmarks[LEFT_KNEE],
        pose_landmarks[LEFT_ANKLE]
    )
    features['right_knee_angle'] = compute_angle(
        pose_landmarks[RIGHT_HIP],
        pose_landmarks[RIGHT_KNEE],
        pose_landmarks[RIGHT_ANKLE]
    )

    # Distances
    features['shoulder_width'] = np.linalg.norm(
        pose_landmarks[LEFT_SHOULDER] - pose_landmarks[RIGHT_SHOULDER]
    )
    features['hip_width'] = np.linalg.norm(
        pose_landmarks[LEFT_HIP] - pose_landmarks[RIGHT_HIP]
    )
    features['torso_length'] = np.linalg.norm(
        (pose_landmarks[LEFT_SHOULDER] + pose_landmarks[RIGHT_SHOULDER]) / 2 -
        (pose_landmarks[LEFT_HIP] + pose_landmarks[RIGHT_HIP]) / 2
    )

    return features
