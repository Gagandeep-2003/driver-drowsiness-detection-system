# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 18:48:12 2019

@author: Lenovo
"""
import cv2

face_cascade = cv2.CascadeClassifier(r'models\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, img = cap.read()
    if img is None:
        print("Can't acess camera! Ending process..")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y , w ,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)

    cv2.imshow('Face Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #ADDED FUNCTIONALITY OF TERMINATING BY CLICKING ON 'X'
    if cv2.getWindowProperty('Face Detection', cv2.WND_PROP_VISIBLE) < 1:
        break       
cap.release()
cv2.destroyAllWindows()
