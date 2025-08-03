# -*- coding: utf-8 -*-
"""
Enhanced Blink Detection with Drunk Driver Detection
Combines original drowsiness detection with drunk driver indicators
"""

import dlib
import sys
import cv2
import time
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread
import playsound
import queue
import math
from collections import deque
import smtplib
from email.mime.text import MIMEText

FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 460

# Original drowsiness parameters
thresh = 0.27
blinkTime = 0.15
drowsyTime = 1.5

# New drunk detection parameters
HEAD_MOVEMENT_THRESHOLD = 15  # degrees
HEAD_STABILITY_WINDOW = 30    # frames
BLINK_DELAY_THRESHOLD = 1.5   # seconds
RED_EYE_THRESHOLD = 0.3       # red dominance ratio

modelPath = "models/shape_predictor_68_face_landmarks.dat"
sound_path = "alarm.wav"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

leftEyeIndex = [36, 37, 38, 39, 40, 41]
rightEyeIndex = [42, 43, 44, 45, 46, 47]

# Original variables
blinkCount = 0
drowsy = 0
state = 0
ALARM_ON = False
GAMMA = 1.5
threadStatusQ = queue.Queue()

# New drunk detection variables
head_positions = deque(maxlen=HEAD_STABILITY_WINDOW)
blink_durations = deque(maxlen=10)
drunk_indicators = {
    'head_sway': False,
    'delayed_blink': False,
    'red_eyes': False,
    'droopy_eyelids': False
}
drunk_alert_sent = False
emergency_contacts = ["emergency@example.com"]  # Add real email addresses

invGamma = 1.0/GAMMA
table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(0, 256)]).astype("uint8")

def gamma_correction(image):
    return cv2.LUT(image, table)

def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray) 

def soundAlert(path, threadStatusQ):
    import traceback
    while True:
        if not threadStatusQ.empty():
            FINISHED = threadStatusQ.get()
            if FINISHED:
                break
        try:
            playsound.playsound(path)
        except Exception as e:
            print(f"Error playing sound: {e}")
            traceback.print_exc()
            break

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_head_pose(landmarks, img_shape):
    """Calculate head pose angles for drunk detection"""
    # 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    
    # 2D image points from landmarks
    image_points = np.array([
        landmarks[30],     # Nose tip
        landmarks[8],      # Chin
        landmarks[36],     # Left eye left corner
        landmarks[45],     # Right eye right corner
        landmarks[48],     # Left mouth corner
        landmarks[54]      # Right mouth corner
    ], dtype=np.float64)
    
    # Camera internals
    focal_length = img_shape[1]
    center = (img_shape[1]/2, img_shape[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4, 1))
    
    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )
        
        if success:
            # Convert rotation vector to angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            angles = rotation_matrix_to_euler_angles(rotation_matrix)
            return angles
    except:
        pass
    
    return [0, 0, 0]

def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to euler angles"""
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    
    return [math.degrees(x), math.degrees(y), math.degrees(z)]

def detect_head_sway(angles):
    """Detect excessive head movement"""
    head_positions.append(angles)
    
    if len(head_positions) < HEAD_STABILITY_WINDOW:
        return False
    
    # Calculate movement variance
    positions_array = np.array(head_positions)
    variance = np.var(positions_array, axis=0)
    
    # Check if movement exceeds threshold
    excessive_movement = any(var > HEAD_MOVEMENT_THRESHOLD**2 for var in variance)
    return excessive_movement

def analyze_eye_redness(eye_region):
    """Analyze eye region for redness"""
    if eye_region.size == 0:
        return False
    
    # Convert to HSV for better red detection
    hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
    
    # Define red color range
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = eye_region.shape[0] * eye_region.shape[1]
    red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0
    
    return red_ratio > RED_EYE_THRESHOLD

def detect_delayed_blink(ear, current_time):
    """Detect delayed blinking patterns"""
    global blink_start_time
    
    if ear < thresh:
        if 'blink_start_time' not in globals():
            blink_start_time = current_time
    else:
        if 'blink_start_time' in globals():
            duration = current_time - blink_start_time
            blink_durations.append(duration)
            del globals()['blink_start_time']
    
    if len(blink_durations) >= 3:
        avg_duration = np.mean(list(blink_durations))
        return avg_duration > BLINK_DELAY_THRESHOLD
    
    return False

def detect_droopy_eyelids(landmarks):
    """Detect droopy eyelids"""
    left_eye_landmarks = [landmarks[i] for i in leftEyeIndex]
    right_eye_landmarks = [landmarks[i] for i in rightEyeIndex]
    
    left_ear = eye_aspect_ratio(left_eye_landmarks)
    right_ear = eye_aspect_ratio(right_eye_landmarks)
    avg_ear = (left_ear + right_ear) / 2.0
    
    # Eyes partially closed but not blinking
    return 0.15 < avg_ear < 0.22

def send_emergency_alert():
    """Send emergency email alert"""
    global drunk_alert_sent
    
    if drunk_alert_sent:
        return
    
    def send_email():
        global drunk_alert_sent
        try:
            # Configure email settings (replace with your settings)
            sender_email = "your_email@gmail.com"
            password = "your_app_password"
            
            subject = "URGENT: Driver Impairment Alert"
            body = f"""
            EMERGENCY ALERT: Driver Impairment Detected
            
            Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
            
            Detected Indicators:
            - Head Sway: {'Yes' if drunk_indicators['head_sway'] else 'No'}
            - Delayed Blinking: {'Yes' if drunk_indicators['delayed_blink'] else 'No'}
            - Red Eyes: {'Yes' if drunk_indicators['red_eyes'] else 'No'}
            - Droopy Eyelids: {'Yes' if drunk_indicators['droopy_eyelids'] else 'No'}
            - Drowsiness: {'Yes' if drowsy else 'No'}
            
            Immediate attention required!
            """
            
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = sender_email
            
            import smtplib
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender_email, password)
                for contact in emergency_contacts:
                    msg['To'] = contact
                    server.send_message(msg)
                    del msg['To']
            
            print("Emergency alert sent!")
            drunk_alert_sent = True
            
        except Exception as e:
            print(f"Failed to send alert: {e}")
    
    # Send in separate thread
    alert_thread = Thread(target=send_email)
    alert_thread.daemon = True
    alert_thread.start()

def checkEyeStatus(landmarks):
    mask = np.zeros(frame.shape[:2], dtype = np.float32)
    
    hullLeftEye = []
    for i in range(0, len(leftEyeIndex)):
        hullLeftEye.append((landmarks[leftEyeIndex[i]][0], landmarks[leftEyeIndex[i]][1]))

    cv2.fillConvexPoly(mask, np.int32(hullLeftEye), 255)

    hullRightEye = []
    for i in range(0, len(rightEyeIndex)):
        hullRightEye.append((landmarks[rightEyeIndex[i]][0], landmarks[rightEyeIndex[i]][1]))

    cv2.fillConvexPoly(mask, np.int32(hullRightEye), 255)

    leftEAR = eye_aspect_ratio(hullLeftEye)
    rightEAR = eye_aspect_ratio(hullRightEye)

    ear = (leftEAR + rightEAR) / 2.0

    eyeStatus = 1
    if (ear < thresh):
        eyeStatus = 0

    return eyeStatus, ear

def checkBlinkStatus(eyeStatus):
    global state, blinkCount, drowsy
    if(state >= 0 and state <= falseBlinkLimit):
        if(eyeStatus):
            state = 0
        else:
            state += 1

    elif(state >= falseBlinkLimit and state < drowsyLimit):
        if(eyeStatus):
            blinkCount += 1 
            state = 0
        else:
            state += 1

    else:
        if(eyeStatus):
            state = 0
            drowsy = 3
            blinkCount += 1
        else:
            drowsy = 3

def getLandmarks(im):
    imSmall = cv2.resize(im, None, 
                            fx = 1.0/FACE_DOWNSAMPLE_RATIO, 
                            fy = 1.0/FACE_DOWNSAMPLE_RATIO, 
                            interpolation = cv2.INTER_LINEAR)

    rects = detector(imSmall, 0)
    if len(rects) == 0:
        return 0

    newRect = dlib.rectangle(int(rects[0].left() * FACE_DOWNSAMPLE_RATIO),
                            int(rects[0].top() * FACE_DOWNSAMPLE_RATIO),
                            int(rects[0].right() * FACE_DOWNSAMPLE_RATIO),
                            int(rects[0].bottom() * FACE_DOWNSAMPLE_RATIO))

    points = []
    [points.append((p.x, p.y)) for p in predictor(im, newRect).parts()]
    return points

def analyze_drunk_indicators(landmarks, frame, ear, current_time):
    """Analyze all drunk driver indicators"""
    global drunk_indicators
    
    # Head pose analysis
    head_angles = get_head_pose(landmarks, frame.shape)
    drunk_indicators['head_sway'] = detect_head_sway(head_angles)
    
    # Delayed blink detection
    drunk_indicators['delayed_blink'] = detect_delayed_blink(ear, current_time)
    
    # Droopy eyelids detection
    drunk_indicators['droopy_eyelids'] = detect_droopy_eyelids(landmarks)
    
    # Eye redness analysis
    try:
        # Extract eye regions for color analysis
        left_eye_points = np.array([landmarks[i] for i in leftEyeIndex])
        right_eye_points = np.array([landmarks[i] for i in rightEyeIndex])
        
        left_rect = cv2.boundingRect(left_eye_points)
        right_rect = cv2.boundingRect(right_eye_points)
        
        # Extract eye regions with padding
        padding = 5
        left_eye_region = frame[
            max(0, left_rect[1]-padding):left_rect[1]+left_rect[3]+padding,
            max(0, left_rect[0]-padding):left_rect[0]+left_rect[2]+padding
        ]
        right_eye_region = frame[
            max(0, right_rect[1]-padding):right_rect[1]+right_rect[3]+padding,
            max(0, right_rect[0]-padding):right_rect[0]+right_rect[2]+padding
        ]
        
        left_red = analyze_eye_redness(left_eye_region)
        right_red = analyze_eye_redness(right_eye_region)
        drunk_indicators['red_eyes'] = left_red or right_red
        
    except Exception as e:
        print(f"Error in eye redness analysis: {e}")
        drunk_indicators['red_eyes'] = False
    
    return head_angles

def draw_drunk_indicators(frame, head_angles):
    """Draw drunk detection indicators on frame"""
    y_offset = 120
    
    # Count active indicators
    active_indicators = sum(drunk_indicators.values())
    
    if active_indicators > 0:
        # Main warning
        cv2.putText(frame, f"IMPAIRMENT INDICATORS: {active_indicators}/4", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        y_offset += 30
        
        # Individual indicators
        for indicator, status in drunk_indicators.items():
            if status:
                indicator_text = indicator.replace('_', ' ').title()
                cv2.putText(frame, f"- {indicator_text}", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                y_offset += 25
    
    # Display head angles
    cv2.putText(frame, f"Head: P{head_angles[0]:.1f} Y{head_angles[1]:.1f} R{head_angles[2]:.1f}",
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Initialize camera
capture = cv2.VideoCapture(0)

for i in range(10):
    ret, frame = capture.read()
    if not capture.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()

totalTime = 0.0
validFrames = 0
dummyFrames = 100

print("Calibration in Progress!")
while(validFrames < dummyFrames):
    validFrames += 1
    t = time.time()
    ret, frame = capture.read()
    if not ret or frame is None:
        print("Error: Could not read frame from webcam.")
        break 

    height, width = frame.shape[:2]
    IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
    frame = cv2.resize(frame, None, 
                        fx = 1/IMAGE_RESIZE, 
                        fy = 1/IMAGE_RESIZE, 
                        interpolation = cv2.INTER_LINEAR)

    adjusted = histogram_equalization(frame)
    landmarks = getLandmarks(adjusted)
    timeLandmarks = time.time() - t

    if landmarks == 0:
        validFrames -= 1
        cv2.putText(frame, "Unable to detect face, Please check proper lighting", 
                   (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "or decrease FACE_DOWNSAMPLE_RATIO", 
                   (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("Enhanced Blink Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        totalTime += timeLandmarks

print("Calibration Complete!")

spf = totalTime/dummyFrames
print("Current SPF (seconds per frame) is {:.2f} ms".format(spf * 1000))

drowsyLimit = drowsyTime/spf
falseBlinkLimit = blinkTime/spf
print("drowsy limit: {}, false blink limit: {}".format(drowsyLimit, falseBlinkLimit))

if __name__ == "__main__":
    vid_writer = cv2.VideoWriter('output-enhanced-detection.avi',
                                cv2.VideoWriter_fourcc('M','J','P','G'), 
                                15, (frame.shape[1],frame.shape[0]))
    
    print("\nEnhanced Driver Detection System Started")
    print("Features: Drowsiness + Drunk Driver Detection")
    print("Press 'q' to quit, 'r' to reset")
    
    while(1):
        try:
            t = time.time()
            current_time = time.time()
            ret, frame = capture.read()
            
            height, width = frame.shape[:2]
            IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
            frame = cv2.resize(frame, None, 
                                fx = 1/IMAGE_RESIZE, 
                                fy = 1/IMAGE_RESIZE, 
                                interpolation = cv2.INTER_LINEAR)

            adjusted = histogram_equalization(frame)
            landmarks = getLandmarks(adjusted)
            
            if landmarks == 0:
                cv2.putText(frame, "Unable to detect face, Please check proper lighting", 
                           (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "or decrease FACE_DOWNSAMPLE_RATIO", 
                           (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow("Enhanced Blink Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # Original drowsiness detection
            eyeStatus, ear = checkEyeStatus(landmarks)
            checkBlinkStatus(eyeStatus)

            # New drunk detection analysis
            head_angles = analyze_drunk_indicators(landmarks, frame, ear, current_time)

            # Draw eye landmarks
            for i in range(0, len(leftEyeIndex)):
                cv2.circle(frame, (landmarks[leftEyeIndex[i]][0], landmarks[leftEyeIndex[i]][1]), 
                          1, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            for i in range(0, len(rightEyeIndex)):
                cv2.circle(frame, (landmarks[rightEyeIndex[i]][0], landmarks[rightEyeIndex[i]][1]), 
                          1, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            # Display EAR
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Original drowsiness alert
            if drowsy:
                cv2.putText(frame, "! ! ! DROWSINESS ALERT ! ! !", (70, 60), 
                           cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if not ALARM_ON:
                    ALARM_ON = True
                    threadStatusQ.put(not ALARM_ON)
                    thread = Thread(target=soundAlert, args=(sound_path, threadStatusQ,))
                    thread.setDaemon(True)
                    thread.start()
            else:
                cv2.putText(frame, "Blinks : {}".format(blinkCount), (460, 80), 
                           cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                ALARM_ON = False

            # Draw drunk indicators
            draw_drunk_indicators(frame, head_angles)

            # Check for emergency alert conditions
            drunk_indicators_count = sum(drunk_indicators.values())
            if (drunk_indicators_count >= 2 and drowsy) or drunk_indicators_count >= 3:
                send_emergency_alert()
                
                # Visual emergency warning
                cv2.putText(frame, "!!! EMERGENCY ALERT SENT !!!", (50, frame.shape[0] - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.imshow("Enhanced Blink Detection", frame)
            vid_writer.write(frame)

            k = cv2.waitKey(1) 
            if k == ord('r'):
                # Reset all states
                state = 0
                drowsy = 0
                ALARM_ON = False
                drunk_alert_sent = False
                drunk_indicators = {k: False for k in drunk_indicators}
                threadStatusQ.put(not ALARM_ON)
                print("All alerts reset")

            elif k == ord('q'):
                break

        except Exception as e:
            print(f"Error in main loop: {e}")

    capture.release()
    vid_writer.release()
    cv2.destroyAllWindows()
