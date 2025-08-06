import numpy as np

# Pose indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
def normalize_pose(pose_landmarks):
    """
    pose_landmarks: np.array shape (33, 3) with (x, y, z) in image or world coordinates.
    Returns: flattened normalized_pose -> shape (99,)
    """

    if pose_landmarks.shape != (33, 3):
        raise ValueError("Expected shape (33, 3) for pose landmarks")

    # Step 1: Centering - use mid-hip as origin
    mid_hip = (pose_landmarks[LEFT_HIP] + pose_landmarks[RIGHT_HIP]) / 2.0
    centered = pose_landmarks - mid_hip

    # Step 2: Scaling - use shoulder width
    shoulder_dist = np.linalg.norm(pose_landmarks[LEFT_SHOULDER] - pose_landmarks[RIGHT_SHOULDER])
    if shoulder_dist == 0:
        shoulder_dist = 1e-6  # prevent division by zero
    scaled = centered / shoulder_dist

    # Flatten to 1D list [x1, y1, z1, x2, y2, z2, ...]
    return scaled.flatten().tolist()
