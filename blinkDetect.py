# -*- coding: utf-8 -*-
"""
Driver Drowsiness Detection – CSV logging with drowsy duration
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
import csv
from datetime import datetime

# 📂 CSV File create/open
csv_file = open("blink_log.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Date", "Time", "Status", "Duration (sec)"])

# 👁‍🗨 Eye state tracking variables
eye_closed_start_time = None 
is_eye_closed = False   

# 🔧 Model & settings
FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 460
thresh = 0.27  # EAR threshold (eyes closed if below this)

modelPath = "models/shape_predictor_68_face_landmarks.dat"
sound_path = "alarm.wav"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

leftEyeIndex = [36, 37, 38, 39, 40, 41]
rightEyeIndex = [42, 43, 44, 45, 46, 47]

blinkCount = 0
drowsy = 0
state = 0
blinkTime = 0.15  # 150ms
drowsyTime = 1.5  # 1500ms (1.5 sec)
ALARM_ON = False
GAMMA = 1.5
threadStatusQ = queue.Queue()

# Gamma correction table
invGamma = 1.0 / GAMMA
table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(0, 256)]).astype("uint8")


# 📌 Utility Functions
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
    mask = np.zeros(frame.shape[:2], dtype=np.float32)
    
    hullLeftEye = [(landmarks[i][0], landmarks[i][1]) for i in leftEyeIndex]
    hullRightEye = [(landmarks[i][0], landmarks[i][1]) for i in rightEyeIndex]

    cv2.fillConvexPoly(mask, np.int32(hullLeftEye), 255)
    cv2.fillConvexPoly(mask, np.int32(hullRightEye), 255)

    leftEAR = eye_aspect_ratio(hullLeftEye)
    rightEAR = eye_aspect_ratio(hullRightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return 1 if ear >= thresh else 0   # 1=open, 0=closed

def getLandmarks(im):
    imSmall = cv2.resize(im, None, fx=1.0/FACE_DOWNSAMPLE_RATIO, fy=1.0/FACE_DOWNSAMPLE_RATIO, interpolation=cv2.INTER_LINEAR)
    rects = detector(imSmall, 0)
    if len(rects) == 0:
        return 0

    newRect = dlib.rectangle(int(rects[0].left() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].top() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].right() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].bottom() * FACE_DOWNSAMPLE_RATIO))

    points = [(p.x, p.y) for p in predictor(im, newRect).parts()]
    return points


# 🎥 Camera setup
capture = cv2.VideoCapture(0)

for i in range(10):
    ret, frame = capture.read()
    if not capture.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()

totalTime = 0.0
validFrames = 0
dummyFrames = 100

print("Calibrating camera...")
while validFrames < dummyFrames:
    validFrames += 1
    t = time.time()
    ret, frame = capture.read()
    if not ret or frame is None:
        print("Error: Could not read frame from webcam.")
        break 

    height, width = frame.shape[:2]
    IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
    frame = cv2.resize(frame, None, fx=1/IMAGE_RESIZE, fy=1/IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)

    adjusted = histogram_equalization(frame)
    landmarks = getLandmarks(adjusted)
    timeLandmarks = time.time() - t

    if landmarks == 0:
        validFrames -= 1
        continue
    else:
        totalTime += timeLandmarks

print("Calibration Complete!")

spf = totalTime / dummyFrames
print("SPF: {:.2f} ms".format(spf * 1000))

# 🏁 Main Loop
if __name__ == "__main__":
    vid_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1], frame.shape[0]))

    while True:
        try:
            ret, frame = capture.read()
            if not ret:
                break

            height, width = frame.shape[:2]
            IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
            frame = cv2.resize(frame, None, fx=1/IMAGE_RESIZE, fy=1/IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)

            adjusted = histogram_equalization(frame)
            landmarks = getLandmarks(adjusted)

            if landmarks == 0:
                cv2.putText(frame, "Unable to detect face", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow("Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # EAR calculation
            leftEAR = eye_aspect_ratio([landmarks[i] for i in leftEyeIndex])
            rightEAR = eye_aspect_ratio([landmarks[i] for i in rightEyeIndex])
            ear = (leftEAR + rightEAR) / 2.0


            if ear < thresh:
                if not is_eye_closed:
                    eye_closed_start_time = time.time()
                    is_eye_closed = True
            else:
                if is_eye_closed:
                    eye_closed_duration = time.time() - eye_closed_start_time
                    is_eye_closed = False

                    if eye_closed_duration > 2.5:
                        current_time = datetime.now()
                        date_str = current_time.strftime("%d-%m-%Y")
                        time_str = current_time.strftime("%H:%M:%S")
                        csv_writer.writerow([date_str, time_str, "Drowsy", f"{eye_closed_duration:.2f}"])
                        print(f"[LOG] Drowsy for {eye_closed_duration:.2f} sec")

            for i in leftEyeIndex:
                cv2.circle(frame, landmarks[i], 1, (0, 0, 255), -1)
            for i in rightEyeIndex:
                cv2.circle(frame, landmarks[i], 1, (0, 0, 255), -1)

            cv2.imshow("Drowsiness Detection", frame)
            vid_writer.write(frame)

            k = cv2.waitKey(1)
            if k == ord('q'):
                break

        except Exception as e:
            print("Error:", e)

    capture.release()
    vid_writer.release()
    csv_file.close()
    cv2.destroyAllWindows()
