#!/usr/bin/env python3
"""
Enhanced Driver Impairment Detection System
Combines drowsiness detection with drunk driver detection features
"""

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time
import threading
import queue
import math
from collections import deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class ImpairedDriverDetector:
    def __init__(self):
        # Load face detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        
        # Eye landmarks indices
        self.left_eye = [36, 37, 38, 39, 40, 41]
        self.right_eye = [42, 43, 44, 45, 46, 47]
        
        # Drowsiness detection parameters
        self.ear_threshold = 0.25
        self.ear_consec_frames = 20
        self.ear_counter = 0
        
        # Drunk detection parameters
        self.head_movement_threshold = 15  # degrees
        self.head_stability_window = 30    # frames
        self.blink_delay_threshold = 2.0   # seconds
        self.red_eye_threshold = 0.4       # red channel dominance
        
        # Data storage for analysis
        self.head_positions = deque(maxlen=self.head_stability_window)
        self.blink_times = deque(maxlen=10)
        self.eye_colors = deque(maxlen=20)
        
        # Alert system
        self.alert_active = False
        self.emergency_contacts = ["emergency@example.com"]  # Add real contacts
        
        # Status tracking
        self.drowsy_detected = False
        self.drunk_indicators = {
            'head_sway': False,
            'delayed_blink': False,
            'red_eyes': False,
            'droopy_eyelids': False
        }
        
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        # Vertical eye landmarks
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        # Horizontal eye landmark
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_head_pose(self, landmarks, img_shape):
        """Calculate head pose angles"""
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
                angles = self.rotation_matrix_to_euler_angles(rotation_matrix)
                return angles
        except:
            pass
        
        return [0, 0, 0]  # Default values if calculation fails
    
    def rotation_matrix_to_euler_angles(self, R):
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
    
    def detect_head_sway(self, angles):
        """Detect excessive head movement indicating impairment"""
        self.head_positions.append(angles)
        
        if len(self.head_positions) < self.head_stability_window:
            return False
        
        # Calculate movement variance
        positions_array = np.array(self.head_positions)
        variance = np.var(positions_array, axis=0)
        
        # Check if movement exceeds threshold
        excessive_movement = any(var > self.head_movement_threshold**2 for var in variance)
        
        return excessive_movement
    
    def analyze_eye_color(self, eye_region):
        """Analyze eye region for redness indicating alcohol consumption"""
        if eye_region.size == 0:
            return False
        
        # Convert to different color spaces for better analysis
        hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
        
        # Define red color range in HSV
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red areas
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Calculate red pixel percentage
        red_pixels = cv2.countNonZero(red_mask)
        total_pixels = eye_region.shape[0] * eye_region.shape[1]
        red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0
        
        return red_ratio > self.red_eye_threshold
    
    def detect_delayed_blink(self, ear, timestamp):
        """Detect delayed or slow blinking patterns"""
        # Detect blink events (when EAR drops significantly)
        if ear < self.ear_threshold:
            if not hasattr(self, 'blink_start_time'):
                self.blink_start_time = timestamp
        else:
            if hasattr(self, 'blink_start_time'):
                blink_duration = timestamp - self.blink_start_time
                self.blink_times.append(blink_duration)
                delattr(self, 'blink_start_time')
        
        # Analyze blink patterns
        if len(self.blink_times) >= 3:
            avg_blink_duration = np.mean(list(self.blink_times))
            return avg_blink_duration > self.blink_delay_threshold
        
        return False
    
    def detect_droopy_eyelids(self, landmarks):
        """Detect droopy eyelids indicating impairment"""
        # Calculate eyelid openness for both eyes
        left_eye_landmarks = [landmarks[i] for i in self.left_eye]
        right_eye_landmarks = [landmarks[i] for i in self.right_eye]
        
        left_ear = self.calculate_ear(left_eye_landmarks)
        right_ear = self.calculate_ear(right_eye_landmarks)
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Check if eyes are consistently partially closed (not fully closed like blinking)
        return 0.15 < avg_ear < 0.22  # Between fully closed and normal open
    
    def send_emergency_alert(self, alert_type, severity):
        """Send emergency alert to contacts"""
        def send_email():
            try:
                # Configure your email settings here
                sender_email = "your_email@gmail.com"
                password = "your_password"  # Use app password for Gmail
                
                message = MIMEMultipart()
                message["From"] = sender_email
                message["Subject"] = f"URGENT: Driver Impairment Alert - {alert_type}"
                
                body = f"""
                EMERGENCY ALERT: Driver Impairment Detected
                
                Alert Type: {alert_type}
                Severity: {severity}
                Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
                
                Detected Indicators:
                - Head Sway: {'Yes' if self.drunk_indicators['head_sway'] else 'No'}
                - Delayed Blinking: {'Yes' if self.drunk_indicators['delayed_blink'] else 'No'}
                - Red Eyes: {'Yes' if self.drunk_indicators['red_eyes'] else 'No'}
                - Droopy Eyelids: {'Yes' if self.drunk_indicators['droopy_eyelids'] else 'No'}
                - Drowsiness: {'Yes' if self.drowsy_detected else 'No'}
                
                Immediate action may be required.
                """
                
                message.attach(MIMEText(body, "plain"))
                
                # Send email to all emergency contacts
                with smtplib.SMTP("smtp.gmail.com", 587) as server:
                    server.starttls()
                    server.login(sender_email, password)
                    for contact in self.emergency_contacts:
                        message["To"] = contact
                        text = message.as_string()
                        server.sendmail(sender_email, contact, text)
                
                print("Emergency alert sent successfully!")
                
            except Exception as e:
                print(f"Failed to send emergency alert: {e}")
        
        # Send email in separate thread to avoid blocking main process
        email_thread = threading.Thread(target=send_email)
        email_thread.daemon = True
        email_thread.start()
    
    def process_frame(self, frame):
        """Main processing function for each frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        current_time = time.time()
        
        for face in faces:
            landmarks = self.predictor(gray, face)
            landmarks = [(p.x, p.y) for p in landmarks.parts()]
            
            # Extract eye regions
            left_eye_landmarks = [landmarks[i] for i in self.left_eye]
            right_eye_landmarks = [landmarks[i] for i in self.right_eye]
            
            # Calculate EAR for drowsiness detection
            left_ear = self.calculate_ear(left_eye_landmarks)
            right_ear = self.calculate_ear(right_eye_landmarks)
            ear = (left_ear + right_ear) / 2.0
            
            # Drowsiness detection
            if ear < self.ear_threshold:
                self.ear_counter += 1
                if self.ear_counter >= self.ear_consec_frames:
                    self.drowsy_detected = True
            else:
                self.ear_counter = 0
                self.drowsy_detected = False
            
            # Get head pose for drunk detection
            head_angles = self.get_head_pose(landmarks, frame.shape)
            
            # Drunk driver detection
            self.drunk_indicators['head_sway'] = self.detect_head_sway(head_angles)
            self.drunk_indicators['delayed_blink'] = self.detect_delayed_blink(ear, current_time)
            self.drunk_indicators['droopy_eyelids'] = self.detect_droopy_eyelids(landmarks)
            
            # Extract eye regions for color analysis
            left_eye_rect = cv2.boundingRect(np.array(left_eye_landmarks))
            right_eye_rect = cv2.boundingRect(np.array(right_eye_landmarks))
            
            left_eye_region = frame[left_eye_rect[1]:left_eye_rect[1]+left_eye_rect[3],
                                   left_eye_rect[0]:left_eye_rect[0]+left_eye_rect[2]]
            right_eye_region = frame[right_eye_rect[1]:right_eye_rect[1]+right_eye_rect[3],
                                    right_eye_rect[0]:right_eye_rect[0]+right_eye_rect[2]]
            
            # Analyze eye color for redness
            left_red = self.analyze_eye_color(left_eye_region)
            right_red = self.analyze_eye_color(right_eye_region)
            self.drunk_indicators['red_eyes'] = left_red or right_red
            
            # Draw landmarks and information on frame
            self.draw_analysis_results(frame, landmarks, ear, head_angles)
            
            # Check for alerts
            self.check_and_trigger_alerts()
        
        return frame
    
    def draw_analysis_results(self, frame, landmarks, ear, head_angles):
        """Draw analysis results on the frame"""
        # Draw eye landmarks
        for i in self.left_eye + self.right_eye:
            cv2.circle(frame, landmarks[i], 2, (0, 255, 0), -1)
        
        # Display EAR
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display head angles
        cv2.putText(frame, f"Head: P{head_angles[0]:.1f} Y{head_angles[1]:.1f} R{head_angles[2]:.1f}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display alerts
        y_offset = 90
        if self.drowsy_detected:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            y_offset += 40
        
        # Display drunk indicators
        drunk_count = sum(self.drunk_indicators.values())
        if drunk_count > 0:
            cv2.putText(frame, f"IMPAIRMENT INDICATORS: {drunk_count}/4", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            y_offset += 30
            
            for indicator, status in self.drunk_indicators.items():
                if status:
                    cv2.putText(frame, f"- {indicator.replace('_', ' ').title()}", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    y_offset += 25
    
    def check_and_trigger_alerts(self):
        """Check conditions and trigger appropriate alerts"""
        drunk_indicators_count = sum(self.drunk_indicators.values())
        
        # High severity: Multiple drunk indicators + drowsiness
        if drunk_indicators_count >= 2 and self.drowsy_detected:
            if not self.alert_active:
                self.send_emergency_alert("SEVERE IMPAIRMENT", "HIGH")
                self.alert_active = True
        
        # Medium severity: Multiple drunk indicators OR drowsiness with some indicators
        elif drunk_indicators_count >= 3 or (self.drowsy_detected and drunk_indicators_count >= 1):
            if not self.alert_active:
                self.send_emergency_alert("MODERATE IMPAIRMENT", "MEDIUM")
                self.alert_active = True
        
        # Reset alert flag if conditions improve
        elif drunk_indicators_count == 0 and not self.drowsy_detected:
            self.alert_active = False

# Example usage
if __name__ == "__main__":
    detector = ImpairedDriverDetector()
    cap = cv2.VideoCapture(0)
    
    print("Enhanced Driver Impairment Detection System Started")
    print("Press 'q' to quit, 'r' to reset alerts")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame = detector.process_frame(frame)
        
        # Display result
        cv2.imshow('Enhanced Driver Impairment Detection', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset detection states
            detector.alert_active = False
            detector.drowsy_detected = False
            detector.drunk_indicators = {k: False for k in detector.drunk_indicators}
            print("Alerts reset")
    
    cap.release()
    cv2.destroyAllWindows()
