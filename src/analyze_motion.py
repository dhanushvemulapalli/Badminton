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
    x, y, _, visibility = joints[idx]
    return x, y

def visibility_check(joints, names, threshold=0.5):
    for name in names:
        idx = POSE_LANDMARKS.index(name)
        if joints[idx][3] < threshold:
            return False
    return True

def angle(a, b, c):
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

    left_hip = get_landmark_xy(joints, 'left_hip')
    left_knee = get_landmark_xy(joints, 'left_knee')
    left_ankle = get_landmark_xy(joints, 'left_ankle')
    left_knee_angle = angle(left_hip, left_knee, left_ankle)

    right_hip = get_landmark_xy(joints, 'right_hip')
    right_knee = get_landmark_xy(joints, 'right_knee')
    right_ankle = get_landmark_xy(joints, 'right_ankle')
    right_knee_angle = angle(right_hip, right_knee, right_ankle)

    left_lunge = left_knee_angle < 110 and right_knee_angle > 160
    right_lunge = right_knee_angle < 110 and left_knee_angle > 160

    if stride_length(joints) < 0.1:
        return False

    return left_lunge or right_lunge

def is_split_step(prev, curr, next_):
    try:
        prev_center = (get_landmark_xy(prev, 'left_hip')[1] + get_landmark_xy(prev, 'right_hip')[1]) / 2
        curr_center = (get_landmark_xy(curr, 'left_hip')[1] + get_landmark_xy(curr, 'right_hip')[1]) / 2
        next_center = (get_landmark_xy(next_, 'left_hip')[1] + get_landmark_xy(next_, 'right_hip')[1]) / 2

        drop = curr_center - prev_center
        rise = next_center - curr_center

        left_ankle = np.array(get_landmark_xy(curr, 'left_ankle'))
        right_ankle = np.array(get_landmark_xy(curr, 'right_ankle'))
        feet_distance = np.linalg.norm(left_ankle - right_ankle)

        # print(f"[Split Step Debug] drop: {drop:.4f}, rise: {rise:.4f}, feet_distance: {feet_distance:.4f}")

        return drop > 0.015 and rise < -0.015 and feet_distance > 0.1
    except Exception as e:
        print("Error in split step detection:", e)
        return False

def normalize_frame(joints):
    # Midpoint between hips as origin
    left_hip = joints[POSE_LANDMARKS.index('left_hip')]
    right_hip = joints[POSE_LANDMARKS.index('right_hip')]
    origin_x = (left_hip[0] + right_hip[0]) / 2
    origin_y = (left_hip[1] + right_hip[1]) / 2

    # Distance between shoulders as scale
    left_shoulder = joints[POSE_LANDMARKS.index('left_shoulder')]
    right_shoulder = joints[POSE_LANDMARKS.index('right_shoulder')]
    scale = np.linalg.norm(np.array([left_shoulder[0], left_shoulder[1]]) -
                           np.array([right_shoulder[0], right_shoulder[1]]))
    scale = max(scale, 1e-5)

    normalized = []
    for x, y, z, vis in joints:
        norm_x = (x - origin_x) / scale
        norm_y = (y - origin_y) / scale
        normalized.append([norm_x, norm_y, z, vis])

    return np.array(normalized)

def detect_lunges(frames):
    lunge_frames = []
    for i, joints in enumerate(frames):
        joints = normalize_frame(joints)
        if is_lunge(joints):
            lunge_frames.append(i)
    return lunge_frames

def detect_split_steps(frames):
    split_frames = []
    for i in range(1, len(frames) - 1):
        prev = normalize_frame(frames[i - 1])
        curr = normalize_frame(frames[i])
        next_ = normalize_frame(frames[i + 1])
        if is_split_step(prev, curr, next_):
            split_frames.append(i)
    return split_frames

def load_keypoints(csv_path):
    df = pd.read_csv(csv_path)
    frames = df.iloc[:, 1:].to_numpy().reshape(-1, 33, 4)
    return frames

# ---------------- MAIN ----------------
if __name__ == "__main__":
    frames = load_keypoints("C:/Users/dhanu/OneDrive/Desktop/Projects/Badminton/Output/keypoints/sample1_keypoints.csv")
    print("Shape of keypoints:", frames.shape)

    lunges = detect_lunges(frames)
    print(f"Lunges detected at frames: {lunges}")

    split_steps = detect_split_steps(frames)
    print(f"Split Steps detected at frames: {split_steps}")
