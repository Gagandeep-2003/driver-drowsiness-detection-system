# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:51:37 2019

@author: Lenovo
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
from datetime import datetime


FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 460

thresh = 0.27

# IMPORTANT: You must download the shape_predictor_68_face_landmarks.dat file from
# https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# and place it in the 'models' folder
modelPath = "models/shape_predictor_68_face_landmarks.dat"
sound_path = "alarm.wav"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

leftEyeIndex = [36, 37, 38, 39, 40, 41]
rightEyeIndex = [42, 43, 44, 45, 46, 47]

blinkCount = 0
drowsy = 0
state = 0
blinkTime = 0.15 #150ms
drowsyTime = 1.5  #1200ms
ALARM_ON = False
GAMMA = 1.5
threadStatusQ = queue.Queue()

# Phase 2: Session tracking variables (temporary until SessionManager is ready)
current_ear = 0.0
session_ear_values = []
session_alerts = 0
session_start_time = None
session_active = False

# Temporary Session Tracker - will be replaced by real SessionManager
class TempSessionTracker:
    def __init__(self):
        global session_start_time, session_active
        session_start_time = datetime.now()
        session_active = True
        print(f"Session started at: {session_start_time}")
    
    def add_ear_value(self, ear_value):
        global session_ear_values, current_ear
        current_ear = ear_value
        timestamp = datetime.now()
        session_ear_values.append({
            "value": round(ear_value, 4),
            "timestamp": timestamp.isoformat()
        })
    
    def add_alert(self):
        global session_alerts
        session_alerts += 1
        timestamp = datetime.now()
        print(f"Alert #{session_alerts} triggered at: {timestamp}")
    
    def end_session(self):
        global session_start_time, session_active
        if session_active:
            end_time = datetime.now()
            duration = (end_time - session_start_time).total_seconds() / 60
            avg_ear = sum(item["value"] for item in session_ear_values) / len(session_ear_values) if session_ear_values else 0
            
            print(f"\n=== Session Summary ===")
            print(f"Duration: {duration:.2f} minutes")
            print(f"Total EAR readings: {len(session_ear_values)}")
            print(f"Average EAR: {avg_ear:.4f}")
            print(f"Alerts triggered: {session_alerts}")
            print(f"Total blinks: {blinkCount}")
            
            session_active = False

# Initialize session tracker
session_tracker = None

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


def checkEyeStatus(landmarks):
    global session_tracker, current_ear
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
    
    if session_tracker:
        session_tracker.add_ear_value(ear)

    eyeStatus = 1          # 1 = Open, 0 = closed
    if (ear < thresh):
        eyeStatus = 0

    return eyeStatus  

def checkBlinkStatus(eyeStatus):
    global state, blinkCount, drowsy, session_tracker
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
            # Phase 2: Track alert when drowsiness is detected
            if session_tracker:
                session_tracker.add_alert()

        else:
            drowsy = 3
            # Phase 2: Track alert when drowsiness persists
            if session_tracker:
                session_tracker.add_alert()

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

# Phase 2: Getter functions for external access (for session_history.py)
def get_current_ear():
    """Get the current EAR value"""
    return current_ear

def get_current_blink_count():
    """Get the current blink count"""
    return blinkCount

def get_session_data():
    """Get all current session data"""
    return {
        'ear': current_ear,
        'blink_count': blinkCount,
        'alerts': session_alerts,
        'drowsy_state': drowsy,
        'eye_state': state,
        'session_active': session_active
    }

def get_session_ear_values():
    """Get all EAR values collected in current session"""
    return session_ear_values

def start_new_session():
    """Start a new tracking session"""
    global session_tracker
    if session_tracker:
        session_tracker.end_session()
    session_tracker = TempSessionTracker()

def end_current_session():
    """End the current tracking session"""
    global session_tracker
    if session_tracker:
        session_tracker.end_session()
        session_tracker = None

capture = cv2.VideoCapture(0)

for i in range(10):
    ret, frame = capture.read()
    if not capture.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()

totalTime = 0.0
validFrames = 0
dummyFrames = 100

print("Caliberation in Progress!")
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

    #adjusted = gamma_correction(frame)
    adjusted = histogram_equalization(frame)

    landmarks = getLandmarks(adjusted)
    timeLandmarks = time.time() - t

    if landmarks == 0:
        validFrames -= 1
        cv2.putText(frame, "Unable to detect face, Please check proper lighting", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "or decrease FACE_DOWNSAMPLE_RATIO", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("Blink Detection Demo", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    else:
        totalTime += timeLandmarks
print("Caliberation Complete!")

spf = totalTime/dummyFrames
print("Current SPF (seconds per frame) is {:.2f} ms".format(spf * 1000))

drowsyLimit = drowsyTime/spf
falseBlinkLimit = blinkTime/spf
print("drowsy limit: {}, false blink limit: {}".format(drowsyLimit, falseBlinkLimit))

# Phase 2: Start session tracking
session_tracker = TempSessionTracker()

if __name__ == "__main__":
    vid_writer = cv2.VideoWriter('output-low-light-2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))
    while(1):
        try:
            t = time.time()
            ret, frame = capture.read()
            height, width = frame.shape[:2]
            IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
            frame = cv2.resize(frame, None, 
                                fx = 1/IMAGE_RESIZE, 
                                fy = 1/IMAGE_RESIZE, 
                                interpolation = cv2.INTER_LINEAR)

            # adjusted = gamma_correction(frame)
            adjusted = histogram_equalization(frame)

            landmarks = getLandmarks(adjusted)
            if landmarks == 0:
                validFrames -= 1
                cv2.putText(frame, "Unable to detect face, Please check proper lighting", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "or decrease FACE_DOWNSAMPLE_RATIO", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow("Blink Detection Demo", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            eyeStatus = checkEyeStatus(landmarks)
            checkBlinkStatus(eyeStatus)

            for i in range(0, len(leftEyeIndex)):
                cv2.circle(frame, (landmarks[leftEyeIndex[i]][0], landmarks[leftEyeIndex[i]][1]), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            for i in range(0, len(rightEyeIndex)):
                cv2.circle(frame, (landmarks[rightEyeIndex[i]][0], landmarks[rightEyeIndex[i]][1]), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            if drowsy:
                cv2.putText(frame, "! ! ! DROWSINESS ALERT ! ! !", (70, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if not ALARM_ON:
                    ALARM_ON = True
                    threadStatusQ.put(not ALARM_ON)
                    thread = Thread(target=soundAlert, args=(sound_path, threadStatusQ,))
                    thread.setDaemon(True)
                    thread.start()

            else:
                cv2.putText(frame, "Blinks : {}".format(blinkCount), (460, 80), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                # (0, 400)
                ALARM_ON = False


            cv2.imshow("Blink Detection", frame)
            vid_writer.write(frame)

            k = cv2.waitKey(1) 
            if k == ord('r'):
                state = 0
                drowsy = 0
                ALARM_ON = False
                threadStatusQ.put(not ALARM_ON)

            elif k == ord('q'):
                break

            # print("Time taken", time.time() - t)

        except Exception as e:
            print(e)

    # Phase 2: End session when detection stops
    if session_tracker:
        session_tracker.end_session()

    capture.release()
    vid_writer.release()
    cv2.destroyAllWindows()
