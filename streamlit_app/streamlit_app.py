import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import threading
import time
import base64
import zipfile
import os
import zipfile
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import load_model
MODEL_ZIP = r"models/ML_Models/resnet_model.zip"
MODEL_DIR = r"models/ML_Models/resnet_model"  # Folder where it will be extracted

# Unzip only if the folder doesn't already exist
if not os.path.exists(MODEL_DIR):
    with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)

# Now load the model
try:
    model = load_model(MODEL_DIR)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Failed to load model:", e)

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

    def preprocess_eye(self, img, eye_landmarks):
        # Crop around the eye landmarks
        x_coords = [p[0] for p in eye_landmarks]
        y_coords = [p[1] for p in eye_landmarks]
        x_min, x_max = max(min(x_coords)-5, 0), min(max(x_coords)+5, img.shape[1])
        y_min, y_max = max(min(y_coords)-5, 0), min(max(y_coords)+5, img.shape[0])
        
        eye_img = img[y_min:y_max, x_min:x_max]
        eye_img = cv2.resize(eye_img, (224, 224))  # Match your ResNet input size
        eye_img = eye_img / 255.0
        return np.expand_dims(eye_img, axis=0)

    def transform(self, frame):
        global alarm_thread
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = img.shape
                landmarks = [(int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark]

                left_eye_idx = [362, 385, 387, 263, 373, 380]
                right_eye_idx = [33, 160, 158, 133, 153, 144]

                left_eye_pts = [landmarks[i] for i in left_eye_idx]
                right_eye_pts = [landmarks[i] for i in right_eye_idx]

                # EAR Calculation
                left_ear = calculate_ear(landmarks, left_eye_idx)
                right_ear = calculate_ear(landmarks, right_eye_idx)
                ear = (left_ear + right_ear) / 2.0

                # Model Prediction
                left_eye_img = self.preprocess_eye(img, left_eye_pts)
                left_eye_img = preprocess_input(left_eye_img)  # Preprocess for ResNet
                left_eye_img = np.expand_dims(left_eye_img, axis=0)  # Add batch dimension
                right_eye_img = self.preprocess_eye(img, right_eye_pts)
                right_eye_img = preprocess_input(right_eye_img)  # Preprocess for ResNet
                right_eye_img = np.expand_dims(right_eye_img, axis=0)  # Add batch dimension
                
                pred_l = model.predict(left_eye_img, verbose=0)[0][0]  # assuming binary output: 0=open, 1=closed
                pred_r = model.predict(right_eye_img, verbose=0)[0][0]
                pred= (pred_l + pred_r) / 2.0  # Average prediction for both eyes
                cv2.putText(img, f"Left EAR: {left_ear:.2f} | Right EAR: {right_ear:.2f}", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, f"Left Pred: {pred_l:.2f} | Right Pred: {pred_r:.2f}", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, f"Combined Pred: {pred:.2f}", (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # Drowsiness detection logic
                if ear < EAR_THRESHOLD or pred > 0.5:  # combine EAR + model
                    self.frame_count += 1
                else:
                    self.frame_count = 0
                    self.drowsy = False

                if self.frame_count >= CONSEC_FRAMES:
                    self.drowsy = True
                    cv2.putText(img, "DROWSY!", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    if alarm_thread is None or not alarm_thread.is_alive():
                        alarm_thread = threading.Thread(target=play_alarm)
                        alarm_thread.start()
                else:
                    cv2.putText(img, f"EAR: {ear:.2f} | Pred: {pred:.2f}", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return img


# Streamlit app
st.set_page_config(
    page_title="Driver Drowsiness Detection", 
    layout="centered",
    page_icon="🚗",
    initial_sidebar_state="collapsed"
)

# Add PWA support with custom HTML
pwa_html = """
<link rel="manifest" href="/app/static/manifest.json">
<meta name="theme-color" content="#ff6b6b">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="DrowsinessDetect">
<link rel="apple-touch-icon" href="/app/static/icons/icon-192x192.png">
<meta name="msapplication-TileColor" content="#ff6b6b">
<meta name="msapplication-TileImage" content="/app/static/icons/icon-144x144.png">

<script>
// Register service worker for PWA functionality
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/app/static/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            }, function(err) {
                console.log('ServiceWorker registration failed: ', err);
            });
    });
}

// Add install prompt
let deferredPrompt;
window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    
    // Show install button
    const installButton = document.createElement('button');
    installButton.textContent = '📱 Install App';
    installButton.style.cssText = `
        position: fixed;
        top: 10px;
        right: 10px;
        background: #ff6b6b;
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 5px;
        cursor: pointer;
        z-index: 1000;
        font-size: 14px;
    `;
    
    installButton.addEventListener('click', () => {
        deferredPrompt.prompt();
        deferredPrompt.userChoice.then((choiceResult) => {
            if (choiceResult.outcome === 'accepted') {
                console.log('User accepted the install prompt');
                installButton.remove();
            }
            deferredPrompt = null;
        });
    });
    
    document.body.appendChild(installButton);
});
</script>
"""

st.markdown(pwa_html, unsafe_allow_html=True)

st.title("🚗 Driver Drowsiness Detection")
st.markdown("Detects eye closure and triggers an audio alert if drowsiness is detected.")

# PWA Installation Instructions
with st.expander("📱 Install as Mobile App", expanded=False):
    st.markdown("""
    **Install this app on your device for the best experience:**
    
    **On Mobile (Chrome/Safari):**
    1. Tap the menu button (⋮ or share icon)
    2. Select "Add to Home Screen" or "Install App"
    3. Confirm installation
    
    **On Desktop (Chrome/Edge):**
    1. Click the install icon in the address bar
    2. Or use the "Install App" button above
    3. Confirm installation
    
    **Benefits of installing:**
    - ✅ Works offline
    - ✅ Faster loading
    - ✅ Full-screen experience
    - ✅ App-like behavior
    """)

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
