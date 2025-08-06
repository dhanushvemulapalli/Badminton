import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)
pose = mp.solutions.pose.Pose()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    print("Landmarks found?", results.pose_landmarks is not None)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
