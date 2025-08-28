# -*- coding: utf-8 -*-
"""
Enhanced Blink Detection with Drunk Driver Detection and Session Tracking
Combines drowsiness detection, drunk driver indicators, and session management
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
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json

# Configuration Constants
FACE_DOWNSAMPLE_RATIO = 0.5
RESIZE_HEIGHT = 460

# Original drowsiness parameters
thresh = 0.27
blinkTime = 0.15
drowsyTime = 1.5

# Drunk detection parameters
HEAD_MOVEMENT_THRESHOLD = 15  # degrees
HEAD_STABILITY_WINDOW = 30    # frames
BLINK_DELAY_THRESHOLD = 1.5   # seconds
RED_EYE_THRESHOLD = 0.3       # red dominance ratio

# File paths
modelPath = "models/shape_predictor_68_face_landmarks.dat"
sound_path = "alarm.wav"

# Initialize dlib detectors
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(modelPath)
    print("✅ Facial detection models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Please ensure the shape_predictor_68_face_landmarks.dat file is in the models/ directory")
    sys.exit(1)

# Eye landmark indices
leftEyeIndex = [36, 37, 38, 39, 40, 41]
rightEyeIndex = [42, 43, 44, 45, 46, 47]

# Original drowsiness variables
blinkCount = 0
drowsy = 0
state = 0
ALARM_ON = False
GAMMA = 1.5
threadStatusQ = queue.Queue()

# Drunk detection variables
head_positions = deque(maxlen=HEAD_STABILITY_WINDOW)
blink_durations = deque(maxlen=10)
drunk_indicators = {
    'head_sway': False,
    'delayed_blink': False,
    'red_eyes': False,
    'droopy_eyelids': False
}
drunk_alert_sent = False

# Session tracking variables
current_ear = 0.0
session_ear_values = []
session_alerts = 0
session_start_time = None
session_active = False

# Load emergency configuration
def load_emergency_config():
    """Load emergency contacts and email configuration"""
    try:
        with open("emergency_config.json", 'r') as f:
            config = json.load(f)
        return config["emergency_contacts"], config["email_settings"]
    except Exception as e:
        print(f"Warning: Could not load emergency config: {e}")
        return ["emergency@example.com"], {
            "sender_email": "your_email@gmail.com", 
            "sender_password": "your_app_password"
        }

emergency_contacts, email_settings = load_emergency_config()

# Gamma correction setup
invGamma = 1.0/GAMMA
table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(0, 256)]).astype("uint8")

class SessionTracker:
    """Enhanced session tracking with both drowsiness and drunk detection metrics"""
    
    def __init__(self):
        global session_start_time, session_active
        session_start_time = datetime.now()
        session_active = True
        self.drunk_alerts = 0
        self.impairment_events = []
        self.total_frames = 0
        print(f"📊 Enhanced Detection Session started at: {session_start_time.strftime('%H:%M:%S')}")
    
    def add_ear_value(self, ear_value):
        global session_ear_values, current_ear
        current_ear = ear_value
        self.total_frames += 1
        timestamp = datetime.now()
        session_ear_values.append({
            "value": round(ear_value, 4),
            "timestamp": timestamp.isoformat()
        })
    
    def add_drowsy_alert(self):
        global session_alerts
        session_alerts += 1
        timestamp = datetime.now()
        print(f"😴 Drowsiness Alert #{session_alerts} at: {timestamp.strftime('%H:%M:%S')}")
    
    def add_drunk_alert(self, indicators):
        self.drunk_alerts += 1
        timestamp = datetime.now()
        event = {
            "timestamp": timestamp.isoformat(),
            "indicators": indicators.copy(),
            "alert_number": self.drunk_alerts
        }
        self.impairment_events.append(event)
        print(f"🚨 Impairment Alert #{self.drunk_alerts} at: {timestamp.strftime('%H:%M:%S')}")
        print(f"   Active indicators: {[k for k, v in indicators.items() if v]}")
    
    def get_session_stats(self):
        if not session_ear_values:
            return {}
        
        ear_values = [item["value"] for item in session_ear_values]
        return {
            "avg_ear": np.mean(ear_values),
            "min_ear": np.min(ear_values),
            "max_ear": np.max(ear_values),
            "total_frames": self.total_frames,
            "drowsy_alerts": session_alerts,
            "impairment_alerts": self.drunk_alerts,
            "total_blinks": blinkCount
        }
    
    def end_session(self):
        global session_start_time, session_active
        if session_active:
            end_time = datetime.now()
            duration_minutes = (end_time - session_start_time).total_seconds() / 60
            stats = self.get_session_stats()
            
            print("\n" + "="*60)
            print("🏁 ENHANCED DETECTION SESSION SUMMARY")
            print("="*60)
            print(f"⏱️  Session Duration: {duration_minutes:.2f} minutes")
            print(f"📊 Total Frames Processed: {stats.get('total_frames', 0):,}")
            print(f"👁️  Average EAR: {stats.get('avg_ear', 0):.4f}")
            print(f"👁️  EAR Range: {stats.get('min_ear', 0):.4f} - {stats.get('max_ear', 0):.4f}")
            print(f"😴 Drowsiness Alerts: {stats.get('drowsy_alerts', 0)}")
            print(f"🚨 Impairment Alerts: {stats.get('impairment_alerts', 0)}")
            print(f"👀 Total Blinks Detected: {stats.get('total_blinks', 0)}")
            
            if self.impairment_events:
                print(f"🔍 Impairment Event Details:")
                for event in self.impairment_events:
                    indicators = [k for k, v in event['indicators'].items() if v]
                    timestamp = datetime.fromisoformat(event['timestamp']).strftime('%H:%M:%S')
                    print(f"   • Alert #{event['alert_number']} at {timestamp}: {', '.join(indicators)}")
            
            print("="*60)
            session_active = False

def gamma_correction(image):
    """Apply gamma correction to improve image contrast"""
    return cv2.LUT(image, table)

def histogram_equalization(image):
    """Apply histogram equalization for better lighting conditions"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def soundAlert(path, threadStatusQ):
    """Play sound alert in separate thread"""
    import traceback
    while True:
        if not threadStatusQ.empty():
            FINISHED = threadStatusQ.get()
            if FINISHED:
                break
        try:
            playsound.playsound(path)
        except Exception as e:
            print(f"🔊 Audio alert error: {e}")
            # Try alternative audio playback methods
            try:
                import os
                if os.name == 'nt':  # Windows
                    import winsound
                    winsound.PlaySound(path, winsound.SND_FILENAME)
                else:  # Linux/Mac
                    os.system(f"aplay {path} 2>/dev/null || afplay {path} 2>/dev/null")
            except:
                print("⚠️  Could not play audio alert")
            break

def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR)"""
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
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            angles = rotation_matrix_to_euler_angles(rotation_matrix)
            return angles
    except Exception as e:
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
    """Detect excessive head movement indicating impairment"""
    head_positions.append(angles)
    
    if len(head_positions) < HEAD_STABILITY_WINDOW:
        return False
    
    positions_array = np.array(head_positions)
    variance = np.var(positions_array, axis=0)
    
    # Check for excessive movement in any direction
    excessive_movement = any(var > HEAD_MOVEMENT_THRESHOLD**2 for var in variance)
    return excessive_movement

def analyze_eye_redness(eye_region):
    """Analyze eye region for redness indicating impairment"""
    if eye_region.size == 0 or eye_region.shape[0] < 5 or eye_region.shape[1] < 5:
        return False
    
    try:
        hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
        
        # Define red color ranges in HSV
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
    except Exception as e:
        return False

def detect_delayed_blink(ear, current_time):
    """Detect delayed blinking patterns"""
    if not hasattr(detect_delayed_blink, 'blink_start_time'):
        detect_delayed_blink.blink_start_time = None
    
    if ear < thresh:
        if detect_delayed_blink.blink_start_time is None:
            detect_delayed_blink.blink_start_time = current_time
    else:
        if detect_delayed_blink.blink_start_time is not None:
            duration = current_time - detect_delayed_blink.blink_start_time
            blink_durations.append(duration)
            detect_delayed_blink.blink_start_time = None
    
    if len(blink_durations) >= 3:
        avg_duration = np.mean(list(blink_durations))
        return avg_duration > BLINK_DELAY_THRESHOLD
    
    return False

def detect_droopy_eyelids(landmarks):
    """Detect droopy eyelids indicating impairment"""
    left_eye_landmarks = [landmarks[i] for i in leftEyeIndex]
    right_eye_landmarks = [landmarks[i] for i in rightEyeIndex]
    
    left_ear = eye_aspect_ratio(left_eye_landmarks)
    right_ear = eye_aspect_ratio(right_eye_landmarks)
    avg_ear = (left_ear + right_ear) / 2.0
    
    # Eyes partially closed but not blinking (between normal open and closed)
    return 0.15 < avg_ear < 0.22

def send_emergency_alert(alert_type="IMPAIRMENT", severity="HIGH"):
    """Send emergency email alert for severe impairment"""
    global drunk_alert_sent
    
    if drunk_alert_sent:
        return
    
    def send_email():
        global drunk_alert_sent
        try:
            sender_email = email_settings["sender_email"]
            password = email_settings["sender_password"]
            
            message = MIMEMultipart()
            message["From"] = sender_email
            message["Subject"] = f"🚨 URGENT: Driver {alert_type} Alert - {severity} Priority"
            
            # Get current session stats
            stats = session_tracker.get_session_stats() if 'session_tracker' in globals() else {}
            
            body = f"""
🚨 EMERGENCY ALERT: Driver Impairment Detected 🚨

⏰ Alert Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
🔍 Alert Type: {alert_type}
⚠️  Severity Level: {severity}

📊 Current Detection Status:
{'='*50}
🎯 Active Impairment Indicators:
• Head Sway/Movement: {'✅ DETECTED' if drunk_indicators['head_sway'] else '❌ Normal'}
• Delayed Blinking: {'✅ DETECTED' if drunk_indicators['delayed_blink'] else '❌ Normal'}
• Eye Redness: {'✅ DETECTED' if drunk_indicators['red_eyes'] else '❌ Normal'}
• Droopy Eyelids: {'✅ DETECTED' if drunk_indicators['droopy_eyelids'] else '❌ Normal'}
• Drowsiness State: {'✅ ACTIVE' if drowsy else '❌ Alert'}

📈 Session Statistics:
• Current EAR Value: {current_ear:.4f}
• Total Drowsiness Alerts: {stats.get('drowsy_alerts', 0)}
• Total Impairment Alerts: {stats.get('impairment_alerts', 0)}
• Detected Blinks: {stats.get('total_blinks', 0)}

⚠️  IMMEDIATE ACTION RECOMMENDED ⚠️

This is an automated alert from the Enhanced Driver Detection System.
The system has detected concerning patterns that may indicate driver impairment.

🔧 System Information:
• Detection Algorithm: Enhanced Multi-Modal Analysis
• Confidence Level: High
• Monitoring Duration: Continuous

📞 Emergency Response Protocol:
1. Attempt to contact the driver immediately
2. If no response, consider emergency services
3. Check driver location and status

This message was generated automatically by the driver safety monitoring system.
            """
            
            message.attach(MIMEText(body, "plain"))
            
            # Send email to all emergency contacts
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender_email, password)
                for contact in emergency_contacts:
                    message["To"] = contact
                    text = message.as_string()
                    server.sendmail(sender_email, contact, text)
                    del message["To"]
            
            print(f"📧 Emergency alert sent to {len(emergency_contacts)} contact(s)")
            drunk_alert_sent = True
            
        except Exception as e:
            print(f"❌ Failed to send emergency alert: {e}")
            print("   Please check email configuration in emergency_config.json")
    
    # Send in separate thread to avoid blocking detection
    alert_thread = Thread(target=send_email)
    alert_thread.daemon = True
    alert_thread.start()

def checkEyeStatus(landmarks, frame):
    """Check eye status and calculate EAR"""
    global session_tracker, current_ear
    
    mask = np.zeros(frame.shape[:2], dtype=np.float32)
    
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
    
    if 'session_tracker' in globals() and session_tracker:
        session_tracker.add_ear_value(ear)

    eyeStatus = 1 if ear >= thresh else 0
    return eyeStatus, ear

def checkBlinkStatus(eyeStatus):
    """Check blink status and drowsiness"""
    global state, blinkCount, drowsy, session_tracker, falseBlinkLimit, drowsyLimit
    
    if state >= 0 and state <= falseBlinkLimit:
        if eyeStatus:
            state = 0
        else:
            state += 1
    elif state >= falseBlinkLimit and state < drowsyLimit:
        if eyeStatus:
            blinkCount += 1 
            state = 0
        else:
            state += 1
    else:
        if eyeStatus:
            state = 0
            drowsy = 3
            blinkCount += 1
            if 'session_tracker' in globals() and session_tracker:
                session_tracker.add_drowsy_alert()
        else:
            drowsy = 3
            if 'session_tracker' in globals() and session_tracker:
                session_tracker.add_drowsy_alert()

def getLandmarks(im):
    """Extract facial landmarks"""
    imSmall = cv2.resize(im, None, 
                        fx=1.0/FACE_DOWNSAMPLE_RATIO, 
                        fy=1.0/FACE_DOWNSAMPLE_RATIO, 
                        interpolation=cv2.INTER_LINEAR)

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
    global drunk_indicators, session_tracker
    
    # Head pose analysis
    head_angles = get_head_pose(landmarks, frame.shape)
    drunk_indicators['head_sway'] = detect_head_sway(head_angles)
    
    # Delayed blink detection
    drunk_indicators['delayed_blink'] = detect_delayed_blink(ear, current_time)
    
    # Droopy eyelids detection
    drunk_indicators['droopy_eyelids'] = detect_droopy_eyelids(landmarks)
    
    # Eye redness analysis with improved error handling
    try:
        left_eye_points = np.array([landmarks[i] for i in leftEyeIndex])
        right_eye_points = np.array([landmarks[i] for i in rightEyeIndex])
        
        left_rect = cv2.boundingRect(left_eye_points)
        right_rect = cv2.boundingRect(right_eye_points)
        
        padding = 8
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
        drunk_indicators['red_eyes'] = False
    
    # Track impairment alerts
    active_indicators = sum(drunk_indicators.values())
    if active_indicators > 0 and 'session_tracker' in globals() and session_tracker:
        session_tracker.add_drunk_alert(drunk_indicators)
    
    return head_angles

def draw_drunk_indicators(frame, head_angles, ear):
    """Draw drunk detection indicators and enhanced UI on frame"""
    
    # Background overlay for better text visibility
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 200), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Main EAR display
    ear_color = (0, 255, 0) if ear > thresh else (0, 0, 255)
    cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, ear_color, 2)
    
    # Blink counter
    cv2.putText(frame, f"Blinks: {blinkCount}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Head pose information
    cv2.putText(frame, f"Head: P{head_angles[0]:.1f}° Y{head_angles[1]:.1f}° R{head_angles[2]:.1f}°",
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # Impairment indicators
    active_indicators = sum(drunk_indicators.values())
    
    if active_indicators > 0:
        # Main indicator count
        indicator_color = (0, 165, 255) if active_indicators < 3 else (0, 0, 255)
        cv2.putText(frame, f"IMPAIRMENT INDICATORS: {active_indicators}/4", 
                   (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, indicator_color, 2)
        
        # Individual indicators
        y_offset = 155
        indicator_names = {
            'head_sway': 'Head Movement',
            'delayed_blink': 'Slow Blinking', 
            'red_eyes': 'Eye Redness',
            'droopy_eyelids': 'Droopy Eyes'
        }
        
        for key, name in indicator_names.items():
            if drunk_indicators[key]:
                cv2.putText(frame, f"• {name}", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                y_offset += 25
    
    # Session info (bottom right)
    if 'session_tracker' in globals() and session_tracker:
        session_duration = (datetime.now() - session_start_time).total_seconds() / 60
        cv2.putText(frame, f"Session: {session_duration:.1f}min", 
                   (frame.shape[1] - 180, frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Alerts: D{session_alerts} I{session_tracker.drunk_alerts}", 
                   (frame.shape[1] - 180, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def main():
    """Main detection loop with enhanced error handling and features"""
    global session_tracker, drowsyLimit, falseBlinkLimit, frame, drowsy, ALARM_ON
    
    print("🚀 Starting Enhanced Driver Detection System...")
    print("🔧 Initializing camera and calibration...")
    
    # Initialize camera with better error handling
    for camera_index in range(3):  # Try multiple camera indices
        capture = cv2.VideoCapture(camera_index)
        if capture.isOpened():
            print(f"📹 Camera {camera_index} initialized successfully")
            break
        capture.release()
    else:
        print("❌ Error: Could not access any camera")
        input("Press Enter to exit...")
        sys.exit(1)

    # Test camera capture
    for i in range(10):
        ret, frame = capture.read()
        if ret and frame is not None:
            break
        time.sleep(0.1)
    else:
        print("❌ Error: Could not read frames from camera")
        capture.release()
        sys.exit(1)

    # Calibration phase with progress indication
    totalTime = 0.0
    validFrames = 0
    dummyFrames = 100

    print(f"🎯 Calibrating detection system ({dummyFrames} frames)...")
    
    while validFrames < dummyFrames:
        validFrames += 1
        t = time.time()
        ret, frame = capture.read()
        
        if not ret or frame is None:
            print("⚠️  Frame read error during calibration")
            validFrames -= 1
            continue

        height, width = frame.shape[:2]
        IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
        frame = cv2.resize(frame, None, 
                           fx=1/IMAGE_RESIZE, 
                           fy=1/IMAGE_RESIZE, 
                           interpolation=cv2.INTER_LINEAR)

        adjusted = histogram_equalization(frame)
        landmarks = getLandmarks(adjusted)
        timeLandmarks = time.time() - t

        if landmarks == 0:
            validFrames -= 1
            # Show calibration progress
            progress = f"Calibration Progress: {validFrames}/{dummyFrames}"
            cv2.putText(frame, progress, (10, 30), 
                       cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Position your face clearly in camera view", 
                       (10, 70), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "Ensure good lighting conditions", 
                       (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Enhanced Detection System - Calibration", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                capture.release()
                cv2.destroyAllWindows()
                sys.exit(0)
        else:
            totalTime += timeLandmarks
            # Show successful calibration frame
            progress = f"Calibration: {validFrames}/{dummyFrames}"
            cv2.putText(frame, progress, (10, 30), 
                       cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Face detected successfully!", 
                       (10, 70), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Enhanced Detection System - Calibration", frame)
            cv2.waitKey(1)

    print("✅ Calibration completed successfully!")

    spf = totalTime/dummyFrames
    print(f"📊 Performance: {spf * 1000:.2f} ms per frame")

    drowsyLimit = int(drowsyTime/spf)
    falseBlinkLimit = int(blinkTime/spf)
    print(f"🎯 Detection thresholds - Drowsy: {drowsyLimit}, Blink: {falseBlinkLimit}")

    # Initialize session tracking
    session_tracker = SessionTracker()
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vid_writer = cv2.VideoWriter(f'enhanced_detection_session_{timestamp}.avi',
                                fourcc, 15, (frame.shape[1], frame.shape[0]))
    
    print("\n" + "="*60)
    print("🚀 ENHANCED DRIVER DETECTION SYSTEM - ACTIVE")
    print("="*60)
    print("🎯 Features Active:")
    print("   • Real-time drowsiness detection")
    print("   • Advanced impairment analysis") 
    print("   • Emergency alert system")
    print("   • Session tracking & analytics")
    print("\n⌨️  Controls:")
    print("   • 'q' - Quit system")
    print("   • 'r' - Reset all alerts") 
    print("   • ESC - Emergency exit")
    print("="*60 + "\n")
    
    frame_count = 0
    last_stats_time = time.time()
    
    # Main detection loop
    try:
        while True:
            current_time = time.time()
            ret, frame = capture.read()
            
            if not ret or frame is None:
                print("⚠️  Camera disconnected or frame read error")
                break
                
            frame_count += 1
            
            # Resize frame for processing
            height, width = frame.shape[:2]
            IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
            frame = cv2.resize(frame, None, 
                               fx=1/IMAGE_RESIZE, 
                               fy=1/IMAGE_RESIZE, 
                               interpolation=cv2.INTER_LINEAR)

            adjusted = histogram_equalization(frame)
            landmarks = getLandmarks(adjusted)
            
            if landmarks == 0:
                cv2.putText(frame, "⚠️  No face detected - Please check lighting and position", 
                           (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "System Status: Waiting for face detection", 
                           (10, 80), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("Enhanced Detection System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Core drowsiness detection
            eyeStatus, ear = checkEyeStatus(landmarks, frame)
            checkBlinkStatus(eyeStatus)

            # Advanced impairment detection
            head_angles = analyze_drunk_indicators(landmarks, frame, ear, current_time)

            # Draw eye landmarks
            for i in leftEyeIndex + rightEyeIndex:
                cv2.circle(frame, (landmarks[i][0], landmarks[i][1]), 
                          2, (0, 255, 0), -1, lineType=cv2.LINE_AA)

            # Enhanced UI display
            draw_drunk_indicators(frame, head_angles, ear)

            # Drowsiness alert handling
            if drowsy > 0:
                drowsy -= 1
                cv2.putText(frame, "🚨 DROWSINESS ALERT 🚨", (50, frame.shape[0] - 100), 
                           cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, "Driver appears to be falling asleep!", (50, frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                
                if not ALARM_ON:
                    ALARM_ON = True
                    threadStatusQ.put(not ALARM_ON)
                    thread = Thread(target=soundAlert, args=(sound_path, threadStatusQ,))
                    thread.setDaemon(True)
                    thread.start()
            else:
                if ALARM_ON:
                    ALARM_ON = False
                    threadStatusQ.put(True)  # Stop alarm

            # Emergency alert conditions
            drunk_indicators_count = sum(drunk_indicators.values())
            severe_impairment = (drunk_indicators_count >= 2 and drowsy > 0) or drunk_indicators_count >= 3
            
            if severe_impairment and not drunk_alert_sent:
                severity = "CRITICAL" if drunk_indicators_count >= 3 and drowsy > 0 else "HIGH"
                send_emergency_alert("SEVERE IMPAIRMENT", severity)
                
                # Visual emergency alert
                cv2.rectangle(frame, (0, frame.shape[0] - 150), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
                cv2.putText(frame, "🆘 EMERGENCY ALERT SENT 🆘", (50, frame.shape[0] - 100),
                           cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, "Severe impairment detected - Contacts notified", (50, frame.shape[0] - 60),
                           cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Show main detection window
            cv2.imshow("Enhanced Detection System", frame)
            vid_writer.write(frame)

            # Print periodic statistics
            if current_time - last_stats_time > 30:  # Every 30 seconds
                if session_tracker:
                    stats = session_tracker.get_session_stats()
                    print(f"📊 Session Update - EAR: {current_ear:.3f}, "
                          f"Blinks: {stats.get('total_blinks', 0)}, "
                          f"Alerts: D{stats.get('drowsy_alerts', 0)}/I{stats.get('impairment_alerts', 0)}")
                last_stats_time = current_time

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quit command received")
                break
            elif key == ord('r'):
                # Reset all detection states
                print("🔄 Resetting all alerts and detection states...")
                state = 0
                drowsy = 0
                ALARM_ON = False
                drunk_alert_sent = False
                drunk_indicators = {k: False for k in drunk_indicators}
                head_positions.clear()
                blink_durations.clear()
                threadStatusQ.put(True)  # Stop any active alarms
                print("✅ All alerts reset successfully")
            elif key == 27:  # ESC key
                print("🚨 Emergency exit")
                break

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"❌ Unexpected error in main loop: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup and session summary
        print("\n🔄 Shutting down Enhanced Detection System...")
        
        # Stop any active alarms
        if ALARM_ON:
            threadStatusQ.put(True)
            
        # End session and show summary
        if session_tracker:
            session_tracker.end_session()

        # Release resources
        try:
            capture.release()
            vid_writer.release()
            cv2.destroyAllWindows()
            print("✅ Resources released successfully")
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")
        
        print("👋 Enhanced Detection System shutdown complete")
        print("Thank you for using the driver safety system!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")