# -- coding: utf-8 --
"""
Created on Sun Dec 29 18:48:12 2019

@author: Lenovo
"""
import os
import cv2
import sys

# Load Haar cascade
#Absolute path
#cascade_path = '/home/happy/gssoc/driver-drowsiness-detection-system/models/haarcascade_frontalface_default.xml'

#relative path
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(BASE_DIR,'models','haarcascade_frontalface_default.xml')


face_cascade = cv2.CascadeClassifier(cascade_path)

# Check if cascade loaded successfully
if face_cascade.empty():
    print(f"[ERROR] Failed to load cascade from {cascade_path}")
    sys.exit(1)

print("[INFO] Trying to open default webcam (0)...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[WARNING] Default webcam not available. Trying external camera (1)...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("[WARNING] No webcam found. Falling back to test video (test2.mp4)...")
    cap = cv2.VideoCapture("test2.mp4")

if not cap.isOpened():
    print("[ERROR] No camera or video source available.")
    sys.exit(1)


try:
    while True:
        ret, img = cap.read()

        # Check if frame was captured
        if not ret or img is None:
            print("[WARNING] Failed to read frame from webcam.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        try:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        except cv2.error as e:
            print(f"[ERROR] detectMultiScale failed: {e}")
            continue

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

        cv2.imshow('Face Detection', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting on user request.")
            break

except KeyboardInterrupt:
    print("\n[INFO] Exiting on Ctrl+C")

finally:
    cap.release()
    cv2.destroyAllWindows()
