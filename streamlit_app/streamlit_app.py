import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import threading
import time
import base64

# Eye aspect ratio threshold and consecutive frames
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20

# Load alarm
def play_alarm():
    try:
        from playsound import playsound
        playsound('alarm.mp3')
    except:
        st.warning("Audio alert failed to play.")

alarm_thread = None

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, eye_indices):
    # Eye landmarks
    p1, p2 = landmarks[eye_indices[1]], landmarks[eye_indices[5]]
    p3, p4 = landmarks[eye_indices[2]], landmarks[eye_indices[4]]
    p5, p6 = landmarks[eye_indices[0]], landmarks[eye_indices[3]]

    vertical1 = euclidean(p2, p4)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p6)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

class DrowsinessDetector(VideoTransformerBase):
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(refine_landmarks=True)
        self.frame_count = 0
        self.drowsy = False

    def transform(self, frame):
        global alarm_thread
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = img.shape
                landmarks = [(int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark]

                # Left and right eyes
                left_eye = [362, 385, 387, 263, 373, 380]
                right_eye = [33, 160, 158, 133, 153, 144]

                left_ear = calculate_ear(landmarks, left_eye)
                right_ear = calculate_ear(landmarks, right_eye)
                ear = (left_ear + right_ear) / 2.0

                if ear < EAR_THRESHOLD:
                    self.frame_count += 1
                else:
                    self.frame_count = 0
                    self.drowsy = False

                if self.frame_count >= CONSEC_FRAMES:
                    self.drowsy = True
                    cv2.putText(img, "DROWSY!", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    # Play alarm in separate thread
                    if alarm_thread is None or not alarm_thread.is_alive():
                        alarm_thread = threading.Thread(target=play_alarm)
                        alarm_thread.start()
                else:
                    cv2.putText(img, f"EAR: {ear:.2f}", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return img

# Streamlit app
st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")
st.title("🚗 Driver Drowsiness Detection")
st.markdown("Detects eye closure and triggers an audio alert if drowsiness is detected.")

# Run webcam with Streamlit
webrtc_streamer(
    key="drowsiness-app",
    video_processor_factory=DrowsinessDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Embed alarm.wav to trigger download if needed
with open("alarm.wav", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<a href="data:audio/wav;base64,{b64}" download="alarm.wav">Download alarm.wav if needed</a>',
        unsafe_allow_html=True,
    )
