# File: C:/Users/dhanu/OneDrive/Desktop/Projects/Badminton/src/video_analyzer.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp
import os
from pose_extractor import extract_keypoints

# Import your existing functions
from analyze_motion import (
    compute_stance_features, is_lunge, is_split_step, 
    compute_chasse_features, compute_stroke_mechanics,
    compute_reaction_time, load_keypoints_from_csv,
    POSE_LANDMARKS, LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE
)

class BadmintonVideoAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()
        
    def render_analysis_video(self, video_path, keypoints_data, output_path):
        """Use pre-generated annotated frames with pose lines"""
        
        # Path to your annotated frames (from your working pose extraction)
        annotated_frames_dir = "C:/Users/dhanu/OneDrive/Desktop/Projects/Badminton/Output/annotated_frames"
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, original_frame = cap.read()
            if not ret:
                break
            
            # Load the pre-annotated frame with perfect pose lines
            annotated_frame_path = os.path.join(annotated_frames_dir, f"frame_{frame_idx:04d}.jpg")
            
            if os.path.exists(annotated_frame_path):
                annotated_frame = cv2.imread(annotated_frame_path)
                
                # Add analysis overlays to the frame with pose lines
                analysis_frame = self.add_analysis_overlay(annotated_frame, keypoints_data, frame_idx)
            else:
                # Fallback to original frame
                analysis_frame = original_frame.copy()
            
            # Side-by-side: original | analysis with pose lines
            combined = np.hstack([original_frame, analysis_frame])
            out.write(combined)
            frame_idx += 1
        
        cap.release()
        out.release()
        print(f"✅ Analysis video with pose lines saved: {output_path}")

        
    def add_analysis_overlay(self, frame, keypoints_data, frame_idx):
        """Create comprehensive analysis overlay for a single frame"""
        overlay = frame.copy()
        
        if frame_idx < len(keypoints_data):
            current_pose = keypoints_data[frame_idx]
            
            # 1. Draw pose landmarks
            # self.draw_enhanced_pose_lines(overlay, current_pose)
            self.draw_pose_landmarks(overlay, current_pose)  
            
            # 2. Add movement analysis
            # self.add_movement_analysis(overlay, keypoints_data, frame_idx)
            
            # 3. Add stroke mechanics
            self.add_stroke_analysis(overlay, keypoints_data, frame_idx)
            
            # 4. Add reaction time analysis
            self.add_reaction_analysis(overlay, keypoints_data, frame_idx)
            
            # 5. Add corrective feedback
            self.add_corrective_feedback(overlay, current_pose, frame_idx)
            
        return overlay
    
    def draw_enhanced_pose_lines(self, frame, pose_landmarks):
        """Draw pose with dynamic coloring based on movement/analysis"""
        height, width = frame.shape[:2]
        
        # Analyze current pose for dynamic coloring
        stance_features = compute_stance_features(pose_landmarks)
        
        # Define connections with categories
        connections = {
            'arms': [(LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
                    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST)],
            'torso': [(LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_HIP, RIGHT_HIP),
                    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP)],
            'legs': [(LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
                    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE)]
        }
        
        # Dynamic colors based on analysis
        colors = {
            'arms': (0, 255, 0) if stance_features.get('hip_bend_angle', 180) < 160 else (0, 255, 255),
            'torso': (255, 255, 0),  # Always yellow for torso
            'legs': (255, 0, 0) if stance_features.get('stance_width', 0) > 0.3 else (0, 100, 255)
        }
        
        # Draw connections with dynamic coloring
        for category, connection_list in connections.items():
            for start_idx, end_idx in connection_list:
                if (pose_landmarks[start_idx][3] > 0.5 and pose_landmarks[end_idx][3] > 0.5):
                    start_point = (int(pose_landmarks[start_idx][0] * width),
                                int(pose_landmarks[start_idx][1] * height))
                    end_point = (int(pose_landmarks[end_idx][0] * width),
                                int(pose_landmarks[end_idx][1] * height))
                    
                    cv2.line(frame, start_point, end_point, colors[category], 4)
        
        # Draw joint circles
        for i, landmark in enumerate(pose_landmarks):
            if landmark[3] > 0.5:
                x, y = int(landmark[0] * width), int(landmark[1] * height)
                cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), 8, (0, 0, 0), 2)

    def draw_pose_landmarks(self, frame, pose_landmarks):
        """Draw pose landmarks using MediaPipe's built-in drawing (like your working code)"""
        height, width = frame.shape[:2]
        
        # Convert your pose_landmarks back to MediaPipe format
        landmarks_list = []
        for i, landmark in enumerate(pose_landmarks):
            if len(landmark) >= 4:
                # Create MediaPipe landmark object
                mp_landmark = mp.framework.formats.landmark_pb2.NormalizedLandmark()
                mp_landmark.x = float(landmark[0])
                mp_landmark.y = float(landmark[1])
                mp_landmark.z = float(landmark[2])
                mp_landmark.visibility = float(landmark[3])
                landmarks_list.append(mp_landmark)
        
        # Create landmark collection
        pose_landmarks_proto = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(landmarks_list)
        
        # Draw using MediaPipe's method (same as your working code!)
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        
        mp_drawing.draw_landmarks(
            frame, 
            pose_landmarks_proto, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Landmarks
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)  # Connections
        )

    def add_movement_analysis(self, frame, keypoints_data, frame_idx):
        """Add movement detection overlays"""
        height, width = frame.shape[:2]
        
        # Detect current movements
        movements = self.analyze_current_movements(keypoints_data, frame_idx)
        
        # Create status panel
        panel_height = 150
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        
        y_offset = 30
        for movement, detected in movements.items():
            color = (0, 255, 0) if detected else (100, 100, 100)
            text = f"{movement}: {'DETECTED' if detected else 'Not Detected'}"
            cv2.putText(panel, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
        
        # Overlay panel on frame
        frame[height-panel_height:height, :] = panel

    def add_stroke_analysis(self, frame, keypoints_data, frame_idx):
        """Add stroke mechanics visualization"""
        if frame_idx < len(keypoints_data):
            try:
                current_pose = keypoints_data[frame_idx]
                
                # Calculate stroke angles
                stroke_data = compute_stroke_mechanics([current_pose])
                
                if stroke_data['angles']:
                    angle = stroke_data['angles'][0]
                    speed = stroke_data['wrist_speeds'][0] if stroke_data['wrist_speeds'] else 0
                    
                    # Draw stroke analysis
                    height, width = frame.shape[:2]
                    
                    # Get arm positions
                    wrist_pos = current_pose[RIGHT_WRIST][:2]
                    elbow_pos = current_pose[RIGHT_ELBOW][:2]
                    shoulder_pos = current_pose[RIGHT_SHOULDER][:2]
                    
                    # Convert to pixel coordinates
                    wrist_pixel = (int(wrist_pos[0] * width), int(wrist_pos[1] * height))
                    elbow_pixel = (int(elbow_pos[0] * width), int(elbow_pos[1] * height))
                    shoulder_pixel = (int(shoulder_pos[0] * width), int(shoulder_pos[1] * height))
                    
                    # Ensure coordinates are within frame bounds
                    wrist_pixel = (max(0, min(width-1, wrist_pixel[0])), max(0, min(height-1, wrist_pixel[1])))
                    elbow_pixel = (max(0, min(width-1, elbow_pixel[0])), max(0, min(height-1, elbow_pixel[1])))
                    shoulder_pixel = (max(0, min(width-1, shoulder_pixel[0])), max(0, min(height-1, shoulder_pixel[1])))
                    
                    # Draw arm lines
                    cv2.line(frame, shoulder_pixel, elbow_pixel, (255, 255, 0), 3)
                    cv2.line(frame, elbow_pixel, wrist_pixel, (255, 255, 0), 3)
                    
                    # Add angle and speed text
                    cv2.putText(frame, f"Arm Angle: {angle:.1f}°", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f"Wrist Speed: {speed:.2f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            except Exception as e:
                # Skip this frame if there's an error
                pass

    def add_reaction_analysis(self, frame, keypoints_data, frame_idx):
        """Add reaction time visualization"""
        try:
            # Compute reaction time for current segment
            segment_start = max(0, frame_idx - 30)  # Look back 30 frames
            segment_end = min(len(keypoints_data), frame_idx + 30)  # Look ahead 30 frames
            
            if segment_end - segment_start > 10:
                segment_data = keypoints_data[segment_start:segment_end]
                reaction_result = compute_reaction_time(segment_data, fps=30)
                
                if reaction_result['reaction_time']:
                    # Visual indicator for reaction time
                    height, width = frame.shape[:2]
                    
                    # Draw reaction time indicator
                    reaction_color = (0, 255, 0) if reaction_result['reaction_time'] > 0.15 else (0, 255, 255)
                    
                    cv2.putText(frame, f"Reaction Time: {reaction_result['reaction_time']}s", 
                               (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, reaction_color, 2)
                    
                    # Draw timeline
                    timeline_y = height - 50
                    cv2.line(frame, (50, timeline_y), (width-50, timeline_y), (255, 255, 255), 2)
                    
                    if reaction_result['shot_frame'] is not None:
                        shot_x = int(50 + (reaction_result['shot_frame'] / len(segment_data)) * (width-100))
                        cv2.circle(frame, (shot_x, timeline_y), 5, (0, 0, 255), -1)
                        cv2.putText(frame, "Shot", (shot_x-15, timeline_y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    
                    if reaction_result['reaction_frame'] is not None:
                        react_x = int(50 + (reaction_result['reaction_frame'] / len(segment_data)) * (width-100))
                        cv2.circle(frame, (react_x, timeline_y), 5, (0, 255, 0), -1)
                        cv2.putText(frame, "React", (react_x-15, timeline_y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        except Exception as e:
            # Skip reaction analysis for this frame if there's an error
            pass
    def generate_detailed_feedback(self, pose_landmarks):
        """Generate more specific and actionable feedback"""
        feedback = []
        
        try:
            # Analyze stance with specific measurements
            stance_features = compute_stance_features(pose_landmarks)
            
            # Stance width analysis
            if stance_features['stance_width'] < 0.25:
                feedback.append(('Stance', 'Widen stance by 6 inches', 'critical'))
            elif stance_features['stance_width'] < 0.35:
                feedback.append(('Stance', 'Slightly wider stance needed', 'warning'))
            else:
                feedback.append(('Stance', 'Good stance width', 'good'))
            
            # Knee bend analysis
            avg_knee_angle = (stance_features['left_knee_angle'] + stance_features['right_knee_angle']) / 2
            if avg_knee_angle > 170:
                feedback.append(('Knees', 'Bend knees more for agility', 'critical'))
            elif avg_knee_angle > 160:
                feedback.append(('Knees', 'Slight knee bend needed', 'warning'))
            else:
                feedback.append(('Knees', 'Good knee position', 'good'))
            
            # Racket position analysis
            shoulder_pos = pose_landmarks[RIGHT_SHOULDER][:3]
            wrist_pos = pose_landmarks[RIGHT_WRIST][:3]
            
            if wrist_pos[1] > shoulder_pos[1] + 0.1:
                feedback.append(('Racket', 'Raise racket higher', 'critical'))
            elif wrist_pos[1] > shoulder_pos[1]:
                feedback.append(('Racket', 'Keep racket ready position', 'warning'))
            else:
                feedback.append(('Racket', 'Good racket position', 'good'))
                
            # Balance analysis
            left_foot = pose_landmarks[LEFT_FOOT_INDEX][:3]
            right_foot = pose_landmarks[RIGHT_FOOT_INDEX][:3]
            foot_separation = abs(left_foot[0] - right_foot[0])
            
            if foot_separation < 0.15:
                feedback.append(('Balance', 'Spread feet for stability', 'warning'))
            else:
                feedback.append(('Balance', 'Well balanced', 'good'))
                
        except Exception as e:
            feedback = [('System', 'Analysis in progress...', 'warning')]
        
        return feedback
    
    def add_body_part_indicators(self, frame, pose_landmarks, feedback):
        """Add visual indicators pointing to specific body parts that need attention"""
        height, width = frame.shape[:2]
        
        for category, message, priority in feedback:
            if priority == 'critical':
                # Draw attention circles around problem areas
                if 'Stance' in category or 'Balance' in category:
                    # Highlight feet
                    for foot_idx in [LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX]:
                        foot_pos = pose_landmarks[foot_idx][:2]
                        foot_pixel = (int(foot_pos[0] * width), int(foot_pos[1] * height))
                        cv2.circle(frame, foot_pixel, 30, (0, 100, 255), 3)
                        
                elif 'Knees' in category:
                    # Highlight knees
                    for knee_idx in [LEFT_KNEE, RIGHT_KNEE]:
                        knee_pos = pose_landmarks[knee_idx][:2]
                        knee_pixel = (int(knee_pos[0] * width), int(knee_pos[1] * height))
                        cv2.circle(frame, knee_pixel, 25, (0, 100, 255), 3)
                        
                elif 'Racket' in category:
                    # Highlight wrist/racket area
                    wrist_pos = pose_landmarks[RIGHT_WRIST][:2]
                    wrist_pixel = (int(wrist_pos[0] * width), int(wrist_pos[1] * height))
                    cv2.circle(frame, wrist_pixel, 35, (0, 100, 255), 3)
                    # Add arrow pointing up
                    cv2.arrowedLine(frame, wrist_pixel, 
                                (wrist_pixel[0], wrist_pixel[1] - 50), 
                                (0, 255, 255), 3)


    def add_corrective_feedback(self, frame, pose_landmarks, frame_idx):
        """Add improved AI-powered corrective feedback with better visual design"""
        try:
            feedback = self.generate_detailed_feedback(pose_landmarks)
            height, width = frame.shape[:2]
            
            # Create a semi-transparent overlay instead of solid black
            overlay = frame.copy()
            
            # Create feedback panel with better design
            panel_width = 400
            panel_height = 120
            panel_x = width - panel_width - 10  # Position on right side
            panel_y = 10  # Position at top
            
            # Create rounded rectangle background
            panel_bg = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
            panel_bg[:] = (40, 40, 40)  # Dark gray instead of black
            
            # Add border
            cv2.rectangle(panel_bg, (0, 0), (panel_width-1, panel_height-1), (100, 100, 100), 2)
            
            # Add title with better styling
            cv2.putText(panel_bg, "TECHNIQUE ANALYSIS", 
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.line(panel_bg, (15, 30), (panel_width-15, 30), (100, 150, 255), 2)
            
            # Add feedback with icons and colors
            y_start = 50
            for i, (category, message, priority) in enumerate(feedback[:3]):
                # Choose color based on priority
                color = (0, 255, 0) if priority == 'good' else (0, 255, 255) if priority == 'warning' else (0, 100, 255)
                
                # Add icon
                icon = "✓" if priority == 'good' else "⚠" if priority == 'warning' else "!"
                cv2.putText(panel_bg, f"{icon} {message}", 
                        (15, y_start + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            
            # Blend the panel with the frame
            roi = frame[panel_y:panel_y+panel_height, panel_x:panel_x+panel_width]
            blended = cv2.addWeighted(roi, 0.3, panel_bg, 0.7, 0)
            frame[panel_y:panel_y+panel_height, panel_x:panel_x+panel_width] = blended
            
            # Add specific body part indicators
            self.add_body_part_indicators(frame, pose_landmarks, feedback)
            
        except Exception as e:
            pass


    def generate_feedback(self, pose_landmarks):
        """Generate AI-powered corrective feedback"""
        feedback = []
        
        try:
            # Analyze stance
            stance_features = compute_stance_features(pose_landmarks)
            
            if stance_features['stance_width'] < 0.3:
                feedback.append("Widen your stance for better balance")
            
            if stance_features['left_knee_angle'] > 170 or stance_features['right_knee_angle'] > 170:
                feedback.append("Bend your knees slightly for better agility")
            
            # Analyze arm position
            shoulder_pos = pose_landmarks[RIGHT_SHOULDER][:3]
            wrist_pos = pose_landmarks[RIGHT_WRIST][:3]
            
            if wrist_pos[1] > shoulder_pos[1]:  # Wrist below shoulder
                feedback.append("Keep your racket up and ready")
            
            # Analyze foot positioning
            left_foot = pose_landmarks[LEFT_FOOT_INDEX][:3]
            right_foot = pose_landmarks[RIGHT_FOOT_INDEX][:3]
            
            if abs(left_foot[0] - right_foot[0]) < 0.2:
                feedback.append("Position feet parallel for better court coverage")
        except Exception as e:
            # Return default feedback if analysis fails
            pass
        
        return feedback if feedback else ["Good form! Keep it up!"]

    def create_3d_visualization(self, keypoints_data, frame_idx, save_path):
        """Create 3D pose visualization"""
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            if frame_idx < len(keypoints_data):
                pose = keypoints_data[frame_idx]
                
                # Plot 3D landmarks
                xs = [landmark[0] for landmark in pose]
                ys = [landmark[1] for landmark in pose]
                zs = [landmark[2] for landmark in pose]
                
                ax.scatter(xs, ys, zs, c='red', s=50)
                
                # Draw connections
                connections = [
                    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
                    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
                    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
                    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
                    (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_HIP, RIGHT_HIP)
                ]
                
                for start, end in connections:
                    ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], 'b-')
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'3D Pose Visualization - Frame {frame_idx}')
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                plt.close()
        except Exception as e:
            print(f"Error creating 3D visualization: {e}")

    def analyze_current_movements(self, keypoints_data, frame_idx):
        """Analyze movements for current frame"""
        movements = {
            'Lunge': False,
            'Split Step': False,
            'Chassé': False,
            'Stroke Contact': False
        }
        
        try:
            if frame_idx < len(keypoints_data):
                current_pose = keypoints_data[frame_idx]
                
                # Check lunge
                movements['Lunge'] = is_lunge(current_pose)
                
                # Check split step (need previous and next frames)
                if 0 < frame_idx < len(keypoints_data) - 1:
                    prev_pose = keypoints_data[frame_idx - 1]
                    next_pose = keypoints_data[frame_idx + 1]
                    movements['Split Step'] = is_split_step(prev_pose, current_pose, next_pose)
                
                # Check chassé (need sequence analysis)
                if frame_idx >= 5:
                    segment = keypoints_data[frame_idx-5:frame_idx+1]
                    chasse_events = compute_chasse_features(segment)
                    movements['Chassé'] = len(chasse_events) > 0
                
                # Check stroke contact
                stroke_data = compute_stroke_mechanics([current_pose])
                movements['Stroke Contact'] = len(stroke_data['contact_frames']) > 0
        except Exception as e:
            # Return default movements if analysis fails
            pass
        
        return movements
