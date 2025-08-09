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
    pose_landmarks: np.array of shape (33, 4) with x,y,z,visibility
    Returns: dict of joint angles and body distances
    """
    from config import AnalysisConfig
    
    # Take only x,y,z coordinates (ignore visibility for now, but check later)
    landmarks = pose_landmarks[:, :3]
    visibility = pose_landmarks[:, 3]
    
    features = {}
    
    # Helper function to check if landmarks are visible enough
    def is_visible(indices):
        return all(visibility[idx] > AnalysisConfig.MIN_VISIBILITY_THRESHOLD for idx in indices)

    # Joint angles (only compute if landmarks are visible)
    if is_visible([LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]):
        features['left_elbow_angle'] = compute_angle(
            landmarks[LEFT_SHOULDER],
            landmarks[LEFT_ELBOW],
            landmarks[LEFT_WRIST]
        )
    else:
        features['left_elbow_angle'] = None
        
    if is_visible([RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]):
        features['right_elbow_angle'] = compute_angle(
            landmarks[RIGHT_SHOULDER],
            landmarks[RIGHT_ELBOW],
            landmarks[RIGHT_WRIST]
        )
    else:
        features['right_elbow_angle'] = None
        
    if is_visible([LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]):
        features['left_knee_angle'] = compute_angle(
            landmarks[LEFT_HIP],
            landmarks[LEFT_KNEE],
            landmarks[LEFT_ANKLE]
        )
    else:
        features['left_knee_angle'] = None
        
    if is_visible([RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]):
        features['right_knee_angle'] = compute_angle(
            landmarks[RIGHT_HIP],
            landmarks[RIGHT_KNEE],
            landmarks[RIGHT_ANKLE]
        )
    else:
        features['right_knee_angle'] = None

    # Distances (only compute if landmarks are visible)
    if is_visible([LEFT_SHOULDER, RIGHT_SHOULDER]):
        features['shoulder_width'] = np.linalg.norm(
            landmarks[LEFT_SHOULDER] - landmarks[RIGHT_SHOULDER]
        )
    else:
        features['shoulder_width'] = None
        
    if is_visible([LEFT_HIP, RIGHT_HIP]):
        features['hip_width'] = np.linalg.norm(
            landmarks[LEFT_HIP] - landmarks[RIGHT_HIP]
        )
    else:
        features['hip_width'] = None
        
    if is_visible([LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]):
        features['torso_length'] = np.linalg.norm(
            (landmarks[LEFT_SHOULDER] + landmarks[RIGHT_SHOULDER]) / 2 -
            (landmarks[LEFT_HIP] + landmarks[RIGHT_HIP]) / 2
        )
    else:
        features['torso_length'] = None

    return features
