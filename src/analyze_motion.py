import pandas as pd
import ast
import cv2
import numpy as np
import os

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


# Define the constants locally in this file instead
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
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

def angle_between(p1, p2, p3):
    """Calculate angle at point p2 between vectors p1-p2 and p3-p2"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def compute_stance_features(pose):
    """Compute stance-related features from pose landmarks"""
    features = {}
    
    # Take only x,y,z coordinates (exclude visibility)
    foot_l = pose[LEFT_FOOT_INDEX][:3]
    foot_r = pose[RIGHT_FOOT_INDEX][:3]
    
    features['stance_width'] = np.linalg.norm(foot_l - foot_r)

    left_knee_angle = angle_between(pose[LEFT_HIP][:3], pose[LEFT_KNEE][:3], pose[LEFT_ANKLE][:3])
    right_knee_angle = angle_between(pose[RIGHT_HIP][:3], pose[RIGHT_KNEE][:3], pose[RIGHT_ANKLE][:3])
    features['left_knee_angle'] = left_knee_angle
    features['right_knee_angle'] = right_knee_angle

    mid_shoulder = (pose[LEFT_SHOULDER][:3] + pose[RIGHT_SHOULDER][:3]) / 2
    mid_hip = (pose[LEFT_HIP][:3] + pose[RIGHT_HIP][:3]) / 2
    mid_knee = (pose[LEFT_KNEE][:3] + pose[RIGHT_KNEE][:3]) / 2
    hip_bend_angle = angle_between(mid_shoulder, mid_hip, mid_knee)
    features['hip_bend_angle'] = hip_bend_angle

    return features

def get_landmark_xy(joints, joint_idx):
    """Get x,y coordinates of a joint by index"""
    return joints[joint_idx][0], joints[joint_idx][1]

def visibility_check(joints, joint_indices, threshold=0.5):
    """Check if joints meet visibility threshold"""
    for idx in joint_indices:
        if joints[idx][3] < threshold:  # visibility is 4th element
            return False
    return True

def angle(a, b, c):
    """Calculate angle at point b"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def stride_length(joints):
    """Calculate stride length between ankles"""
    left = get_landmark_xy(joints, LEFT_ANKLE)
    right = get_landmark_xy(joints, RIGHT_ANKLE)
    return abs(left[0] - right[0])

def is_lunge(joints):
    """Detect if pose represents a lunge"""
    if not visibility_check(joints, [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE,
                                     RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]):
        return False

    left_hip = get_landmark_xy(joints, LEFT_HIP)
    left_knee = get_landmark_xy(joints, LEFT_KNEE)
    left_ankle = get_landmark_xy(joints, LEFT_ANKLE)
    left_knee_angle = angle(left_hip, left_knee, left_ankle)

    right_hip = get_landmark_xy(joints, RIGHT_HIP)
    right_knee = get_landmark_xy(joints, RIGHT_KNEE)
    right_ankle = get_landmark_xy(joints, RIGHT_ANKLE)
    right_knee_angle = angle(right_hip, right_knee, right_ankle)

    left_lunge = left_knee_angle < 100 and right_knee_angle > 160
    right_lunge = right_knee_angle < 100 and left_knee_angle > 160

    if stride_length(joints) < 0.1:
        return False

    return left_lunge or right_lunge

def is_split_step(prev, curr, next_):
    """Detect split step movement"""
    try:
        prev_center = (get_landmark_xy(prev, LEFT_HIP)[1] + get_landmark_xy(prev, RIGHT_HIP)[1]) / 2
        curr_center = (get_landmark_xy(curr, LEFT_HIP)[1] + get_landmark_xy(curr, RIGHT_HIP)[1]) / 2
        next_center = (get_landmark_xy(next_, LEFT_HIP)[1] + get_landmark_xy(next_, RIGHT_HIP)[1]) / 2

        drop = curr_center - prev_center
        rise = next_center - curr_center

        left_ankle = np.array(get_landmark_xy(curr, LEFT_ANKLE))
        right_ankle = np.array(get_landmark_xy(curr, RIGHT_ANKLE))
        feet_distance = np.linalg.norm(left_ankle - right_ankle)

        return (drop > 0.015 or rise < -0.015) and feet_distance > 0.1
    except Exception as e:
        print("Error in split step detection:", e)
        return False

def compute_chasse_features(frames, fps=30):  # Accept numpy array directly
    chasse_events = []
    
    prev_l = None
    prev_r = None
    prev_sep = None
    
    for i, frame in enumerate(frames):
        try:
            # Access numpy array directly instead of dictionary keys
            foot_l = frame[LEFT_FOOT_INDEX][:3]  # x, y, z coordinates
            foot_r = frame[RIGHT_FOOT_INDEX][:3]  # x, y, z coordinates
            
            sep = np.linalg.norm(foot_l[:2] - foot_r[:2])  # xy distance
            mid_pos = (foot_l[:2] + foot_r[:2]) / 2        # avg foot position
            
            if prev_l is not None and prev_r is not None:
                # Foot speeds
                speed_l = np.linalg.norm(foot_l[:2] - prev_l[:2])
                speed_r = np.linalg.norm(foot_r[:2] - prev_r[:2])
                avg_speed = (speed_l + speed_r) / 2
                
                # Foot separation change
                sep_delta = sep - prev_sep if prev_sep is not None else 0
                
                # Direction (x axis)
                dir_l = foot_l[0] - prev_l[0]
                dir_r = foot_r[0] - prev_r[0]
                
                same_direction = np.sign(dir_l) == np.sign(dir_r) and abs(dir_l) > 0.01 and abs(dir_r) > 0.01
                
                # Heuristic: Chassé-like pattern
                if same_direction and avg_speed > 0.1 and abs(sep_delta) > 0.07:
                    chasse_events.append({
                        'frame': i,
                        'foot_sep': sep,
                        'sep_delta': sep_delta,
                        'avg_speed': avg_speed,
                        'direction': np.sign(dir_l),
                        'center_x': mid_pos[0],
                        'center_y': mid_pos[1],
                    })
            
            prev_l = foot_l
            prev_r = foot_r
            prev_sep = sep
            
        except (KeyError, IndexError):
            continue  # skip frames missing data
    
    return chasse_events


def normalize_frame(joints):
    # Convert to float first
    joints = np.array(joints, dtype=float)
    
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
    scale = max(scale, 1e-3)  # Increased minimum scale

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
        # print(is_lunge(joints))
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

def get_velocity(current_pos, prev_pos, fps):
    """Calculate velocity with proper scaling for normalized coordinates"""
    distance = np.linalg.norm(current_pos - prev_pos)
    
    # If using normalized coordinates, you might need to scale
    # Assuming shoulder width ≈ 0.3 in normalized coords ≈ 50cm in real world
    scale_factor = 50.0 / 0.3  # Convert to cm
    real_distance = distance * scale_factor
    
    time_delta = 1.0 / fps
    velocity = real_distance / time_delta  # cm per second
    
    return velocity


def compute_stroke_mechanics(normed_frames, fps=30):
    angles = []
    velocities = []
    contact_frames = []

    prev_wrist = None

    for i, frame in enumerate(normed_frames):
        try:
            # Access numpy array directly using landmark indices
            shoulder = frame[RIGHT_SHOULDER][:3]  # x, y, z coordinates
            elbow = frame[RIGHT_ELBOW][:3]       # x, y, z coordinates  
            wrist = frame[RIGHT_WRIST][:3]       # x, y, z coordinates
            
            # Calculate angle at elbow
            arm_angle = angle_between(shoulder, elbow, wrist)  # Use your existing function
            angles.append(arm_angle)
            
            # Calculate wrist velocity
            if prev_wrist is not None:
                velocity = get_velocity(wrist, prev_wrist, fps)
                velocities.append(velocity)
            else:
                velocities.append(0)
            
            # Detect probable contact point (heuristic)
            if arm_angle > 165 and velocities[-1] > 1.0:
                contact_frames.append({
                    'frame': i,
                    'angle': arm_angle,
                    'wrist_speed': velocities[-1],
                })
            
            prev_wrist = wrist.copy()

            
        except (IndexError, ValueError):
            angles.append(None)
            velocities.append(0)
            continue
    
    return {
        'angles': angles,
        'wrist_speeds': velocities,
        'contact_frames': contact_frames
    }

class ReactionConfig:
    SHOT_SPEED_THRESHOLD = 2.0
    REACTION_SPEED_THRESHOLD = 2.5  
    MIN_SUSTAINED_FRAMES = 3
    MAX_REALISTIC_SPEED = 10.0
    MIN_CONFIDENCE = 0.7
    SMOOTHING_WINDOW = 3

def detect_opponent_shot(frames, fps=30, config=None):
    if config is None:
        config = ReactionConfig()
    
    RIGHT_WRIST = 16
    LEFT_WRIST = 15
    
    if len(frames) == 0:
        return None
    
    speeds = []
    prev_right = None
    prev_left = None

    try:
        for i, frame in enumerate(frames):
            # Check confidence/visibility if available
            right_confidence = frame[RIGHT_WRIST][3] if len(frame[RIGHT_WRIST]) > 3 else 1.0
            left_confidence = frame[LEFT_WRIST][3] if len(frame[LEFT_WRIST]) > 3 else 1.0
            
            right_wrist = frame[RIGHT_WRIST][:3]
            left_wrist = frame[LEFT_WRIST][:3]
            
            max_speed = 0
            
            if prev_right is not None and right_confidence > config.MIN_CONFIDENCE:
                right_speed = np.linalg.norm(right_wrist - prev_right) * fps
                if right_speed < config.MAX_REALISTIC_SPEED:  # Filter out noise
                    max_speed = max(max_speed, right_speed)
            
            if prev_left is not None and left_confidence > config.MIN_CONFIDENCE:
                left_speed = np.linalg.norm(left_wrist - prev_left) * fps
                if left_speed < config.MAX_REALISTIC_SPEED:  # Filter out noise
                    max_speed = max(max_speed, left_speed)
            
            speeds.append(max_speed)
            
            prev_right = right_wrist
            prev_left = left_wrist

        # Apply smoothing filter
        window_size = config.SMOOTHING_WINDOW
        for i in range(window_size, len(speeds)):
            avg_speed = np.mean(speeds[i-window_size:i])
            print(f"Opponent Smoothed Speed (frame {i}): {avg_speed:.2f}")
            if avg_speed > config.SHOT_SPEED_THRESHOLD:
                return i
        return None
        
    except (IndexError, ValueError) as e:
        print(f"Error in shot detection: {e}")
        return None


def detect_player_reaction(frames, start_frame, fps=30, config=None):
    if config is None:
        config = ReactionConfig()
    
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    if start_frame >= len(frames):
        return None

    prev_pos = None
    movement_count = 0

    try:
        for i in range(start_frame, len(frames)):
            frame = frames[i]
            
            # Check confidence if available
            left_conf = frame[LEFT_ANKLE][3] if len(frame[LEFT_ANKLE]) > 3 else 1.0
            right_conf = frame[RIGHT_ANKLE][3] if len(frame[RIGHT_ANKLE]) > 3 else 1.0
            
            if left_conf < config.MIN_CONFIDENCE or right_conf < config.MIN_CONFIDENCE:
                continue
                
            l_ankle = frame[LEFT_ANKLE][:3]
            r_ankle = frame[RIGHT_ANKLE][:3]
            center = (l_ankle + r_ankle) / 2

            if prev_pos is not None:
                speed = np.linalg.norm(center - prev_pos) * fps
                
                # Filter out unrealistic speeds
                if speed < config.MAX_REALISTIC_SPEED:
                    print(f"Player Speed (frame {i}): {speed:.2f}")
                    
                    if speed > config.REACTION_SPEED_THRESHOLD:
                        movement_count += 1
                        if movement_count >= config.MIN_SUSTAINED_FRAMES:
                            print(f"Sustained movement detected starting at frame {i - movement_count + 1}")
                            return i - movement_count + 1  # Return start of movement
                    else:
                        movement_count = 0  # Reset counter if movement stops
                else:
                    print(f"WARNING: Unrealistic speed filtered out: {speed:.2f}")
                    movement_count = 0  # Reset on noise
            
            prev_pos = center
        return None
        
    except (IndexError, ValueError) as e:
        print(f"Error in reaction detection: {e}")
        return None


def compute_reaction_time(normed_frames, fps=30, config=None):
    """
    Compute reaction time with enhanced error handling and validation
    """
    if config is None:
        config = ReactionConfig()
    
    result = {
        'reaction_time': None,
        'shot_frame': None,
        'reaction_frame': None,
        'details': None
    }
    
    if len(normed_frames) == 0:
        result['details'] = 'No frames to analyze'
        return result
    
    print(f"Analyzing {len(normed_frames)} frames at {fps} FPS...")
    
    shot_frame = detect_opponent_shot(normed_frames, fps, config)
    if shot_frame is None:
        result['details'] = f'No shot detected (threshold: {config.SHOT_SPEED_THRESHOLD})'
        return result
    
    print(f"Shot detected at frame {shot_frame}")
    result['shot_frame'] = shot_frame

    react_frame = detect_player_reaction(normed_frames, shot_frame + 1, fps, config)
    if react_frame is None:
        result['details'] = f'No player movement after shot (threshold: {config.REACTION_SPEED_THRESHOLD})'
        return result
    
    print(f"Player reaction detected at frame {react_frame}")
    result['reaction_frame'] = react_frame

    reaction_time = (react_frame - shot_frame) / fps
    result['reaction_time'] = round(reaction_time, 3)
    result['details'] = 'Analysis successful'
    
    # Add realism check
    if reaction_time < 0.12:  # 120ms is near human limit
        result['details'] += f' (WARNING: {reaction_time}s may be unrealistically fast)'
    elif reaction_time > 1.0:  # 1 second is very slow
        result['details'] += f' (WARNING: {reaction_time}s may be unrealistically slow)'
    
    return result


def load_keypoints_from_csv(csv_path):
    """Load keypoints from CSV file created by pose_extractor.py"""
    df = pd.read_csv(csv_path)
    
    print(f"CSV shape: {df.shape}")
    print(f"Columns: {df.columns[:5].tolist()}...")  # Show first 5 columns
    
    # Skip frame column, get pose data
    pose_data = df.iloc[:, 1:].values  # Shape: (num_frames, 132) - 33 joints * 4 values
    
    # Reshape to (num_frames, 33, 4)
    num_frames = pose_data.shape[0]
    frames = pose_data.reshape(num_frames, 33, 4).astype(float)
    
    return frames

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def main():
    """Main function to run the analysis"""
    try:
        from config import ProjectPaths
        
        # Load keypoints (use first available video)
        csv_path = ProjectPaths.get_keypoints_path(1)
        
        if not os.path.exists(csv_path):
            print(f"❌ Keypoints file not found: {csv_path}")
            print("Please run the main processor first to extract keypoints")
            return
            
        frames = load_keypoints_from_csv(csv_path)
        print(f"Loaded keypoints with shape: {frames.shape}")
        
        # Compute stance features for each frame
        stance_features_list = []
        for i, joints in enumerate(frames):
            try:
                stance_feats = compute_stance_features(joints)
                stance_feats["frame"] = i
                stance_features_list.append(stance_feats)
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                continue
        
        if stance_features_list:
            stance_df = pd.DataFrame(stance_features_list)
            output_path = os.path.join(ProjectPaths.FEATURES_DIR, "stance_features.csv")
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            stance_df.to_csv(output_path, index=False)
            print(f"Saved stance features to {output_path}")
        
        # Detect lunges
        lunges = []
        for i, joints in enumerate(frames):
            try:
                if is_lunge(joints):
                    lunges.append(i)
            except Exception as e:
                print(f"Error detecting lunge in frame {i}: {e}")
        
        print(f"Lunges detected at frames: {lunges}")
        
        # Detect split steps
        split_steps = []
        for i in range(1, len(frames) - 1):
            try:
                if is_split_step(frames[i-1], frames[i], frames[i+1]):
                    split_steps.append(i)
            except Exception as e:
                print(f"Error detecting split step at frame {i}: {e}")
        
        print(f"Split Steps detected at frames: {split_steps}")
        
        # Detect chasse
        chasse_steps = compute_chasse_features(frames)

        for event in chasse_steps:
            print(f"Chasse at frame {event['frame']} with speed {event['avg_speed']:.3f}")

        # Compute stroke mechanics
        stroke_feats = compute_stroke_mechanics(frames)
        print("Likely contact points:")
        for c in stroke_feats['contact_frames']:
            print(f"Frame {c['frame']} | Angle: {c['angle']:.2f}° | Wrist Speed: {c['wrist_speed']:.2f} m/s")
        
        # Use first available video for FPS
        video_path = ProjectPaths.get_video_path("Video-1.mp4")
        if os.path.exists(video_path):
            print("Video FPS:", get_video_fps(video_path))
            reaction_time = compute_reaction_time(frames, fps=get_video_fps(video_path))
            print("Reaction Time:", reaction_time['reaction_time'], "seconds")
        else:
            print("Video file not found for FPS analysis")

    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    import os
    main()
