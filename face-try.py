# -- coding: utf-8 --
"""
Created on Sun Dec 29 18:48:12 2019

@author: Lenovo
"""
import cv2
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'models', 'haarcascade_frontalface_default.xml')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

face_cascade = cv2.CascadeClassifier(model_path)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y , w ,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)

    cv2.imshow('Face Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()