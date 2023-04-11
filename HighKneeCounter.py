import cv2
from cv2 import destroyAllWindows
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# Youtube video
right_angle_min = []
right_angle_min_hip = []
cap = cv2.VideoCapture(0)

# Curl counter variables
left_counter = 0
right_counter = 0 
min_ang = 0
max_ang = 0
min_ang_hip = 0
max_ang_hip = 0
left_stage = None
right_stage = None


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is not None:
            frame_ = rescale_frame(frame, percent=75)
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            
            # Get coordinates
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            
           
            
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            right_angle_knee = calculate_angle(right_hip, right_knee, right_ankle) # Knee joint angle
            right_angle_knee = round(right_angle_knee,2)
            
            right_angle_hip = calculate_angle(shoulder, right_hip, right_knee)
            right_angle_hip = round(right_angle_hip,2)
            
            right_hip_angle = 180-right_angle_hip
            right_knee_angle = 180-right_angle_knee
            
            


            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            left_angle_knee = calculate_angle(left_hip, left_knee, left_ankle) # Knee joint angle
            left_angle_knee = round(left_angle_knee,2)
            
            left_angle_hip = calculate_angle(shoulder, left_hip, left_knee)
            left_angle_hip = round(left_angle_hip,2)
            
            left_hip_angle = 180-left_angle_hip
            left_knee_angle = 180-left_angle_knee
            
            
            
            # High knee counter logic
            if left_angle_knee > 169:
                left_stage = "up"
            if left_angle_knee <= 90 and left_stage == 'up':
                left_stage= "down"
                left_counter += 1
                
            if right_angle_knee > 169:
                right_stage = "up"
            if right_angle_knee <= 90 and right_stage =='up':
                right_stage="down"
                right_counter +=1
                # min_ang = min(right_angle_min)
                # max_ang = max(right_angle_min)
                
                # min_ang_hip = min(right_angle_min_hip)
                # max_ang_hip = max(right_angle_min_hip)
                
                # right_angle_min = []
                # right_angle_min_hip = []
        except:
            pass
        
        # Render squat counterq
        # Setup status box
        cv2.rectangle(image, (20,20), (435,160), (0,0,0), -1)
        # Rep data
        """cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)"""
        cv2.putText(image, "Right knee: " + str(right_counter),
                    (30,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, "Left knee: " + str(left_counter),
                    (30,120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(203,17,17), thickness=2, circle_radius=2) 
                                 )               
        
        # out.write(image)
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()