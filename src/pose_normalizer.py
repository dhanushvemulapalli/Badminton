import numpy as np

# Pose indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

def normalize_pose(pose_landmarks):
    """
    pose_landmarks: np.array shape (33, 3) with (x, y, z) in image or world coordinates.
    Returns: normalized_pose -> shape (33, 3)
    """
    if pose_landmarks.shape != (33, 3):
        raise ValueError(f"Expected shape (33, 3) for pose landmarks, got {pose_landmarks.shape}")

    # Step 1: Centering - use mid-hip as origin
    mid_hip = (pose_landmarks[LEFT_HIP] + pose_landmarks[RIGHT_HIP]) / 2.0
    centered = pose_landmarks - mid_hip

    # Step 2: Scaling - use shoulder width
    # shoulder_dist = np.linalg.norm(pose_landmarks[LEFT_SHOULDER] - pose_landmarks[RIGHT_SHOULDER])
    shoulder_dist = pose_landmarks[LEFT_SHOULDER][0] - pose_landmarks[RIGHT_SHOULDER][0]
    
    if shoulder_dist < 0.01:
        shoulder_dist = 0.1  # prevent division by zero

    scaled = centered / shoulder_dist

    return scaled  # Return as (33, 3) array, not flattened

