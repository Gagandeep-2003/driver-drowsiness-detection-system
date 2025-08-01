#!/usr/bin/env python3
"""
Enhanced Streamlit App with PWA Support
This version includes proper static file serving for PWA assets
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import threading
import time
import base64
import os
import mimetypes
from pathlib import Path

# Eye aspect ratio threshold and consecutive frames
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20

# PWA Static Files Setup
def serve_pwa_files():
    """Add routes for PWA files"""
    
    # Add custom CSS and PWA meta tags
    pwa_head = """
    <head>
        <link rel="manifest" href="data:application/json;base64,""" + base64.b64encode(
            open('manifest.json', 'rb').read() if os.path.exists('manifest.json') else b'{}'
        ).decode() + """">
        <meta name="theme-color" content="#ff6b6b">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="default">
        <meta name="apple-mobile-web-app-title" content="DrowsinessDetect">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        
        <!-- PWA Icons -->
        <link rel="icon" type="image/png" sizes="32x32" href="data:image/png;base64,""" + (
            base64.b64encode(open('icons/icon-192x192.png', 'rb').read()).decode() 
            if os.path.exists('icons/icon-192x192.png') else ""
        ) + """">
        <link rel="apple-touch-icon" href="data:image/png;base64,""" + (
            base64.b64encode(open('icons/icon-192x192.png', 'rb').read()).decode() 
            if os.path.exists('icons/icon-192x192.png') else ""
        ) + """">
        
        <style>
            .install-button {
                position: fixed;
                top: 10px;
                right: 10px;
                background: #ff6b6b;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 6px;
                cursor: pointer;
                z-index: 1000;
                font-size: 12px;
                display: none;
            }
            .install-button:hover {
                background: #ff5252;
            }
            
            /* PWA-friendly responsive design */
            @media (max-width: 768px) {
                .main .block-container {
                    padding-top: 2rem;
                    padding-bottom: 2rem;
                }
            }
            
            /* Hide Streamlit branding for PWA */
            .viewerBadge_container__1QSob {
                display: none;
            }
            
            footer {
                visibility: hidden;
            }
        </style>
    </head>
    """
    
    # PWA JavaScript for service worker and install prompt
    pwa_js = """
    <script>
        // Register service worker
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', function() {
                // Create service worker content as data URL
                const swContent = `
                    const CACHE_NAME = 'drowsiness-detection-v1';
                    const urlsToCache = ['/'];
                    
                    self.addEventListener('install', (event) => {
                        event.waitUntil(
                            caches.open(CACHE_NAME)
                                .then((cache) => cache.addAll(urlsToCache))
                        );
                    });
                    
                    self.addEventListener('fetch', (event) => {
                        if (event.request.url.includes('webrtc') || 
                            event.request.url.includes('ws://') || 
                            event.request.url.includes('wss://')) {
                            return;
                        }
                        
                        event.respondWith(
                            caches.match(event.request)
                                .then((response) => response || fetch(event.request))
                        );
                    });
                `;
                
                const blob = new Blob([swContent], { type: 'application/javascript' });
                const swUrl = URL.createObjectURL(blob);
                
                navigator.serviceWorker.register(swUrl)
                    .then(function(registration) {
                        console.log('ServiceWorker registration successful');
                    })
                    .catch(function(err) {
                        console.log('ServiceWorker registration failed: ', err);
                    });
            });
        }
        
        // Install prompt
        let deferredPrompt;
        const installButton = document.createElement('button');
        installButton.textContent = '📱 Install App';
        installButton.className = 'install-button';
        
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredPrompt = e;
            installButton.style.display = 'block';
            document.body.appendChild(installButton);
        });
        
        installButton.addEventListener('click', () => {
            if (deferredPrompt) {
                deferredPrompt.prompt();
                deferredPrompt.userChoice.then((choiceResult) => {
                    if (choiceResult.outcome === 'accepted') {
                        installButton.style.display = 'none';
                    }
                    deferredPrompt = null;
                });
            }
        });
        
        // Add to home screen for iOS
        if (/iPhone|iPad|iPod/.test(navigator.userAgent) && !window.navigator.standalone) {
            const iosInstall = document.createElement('div');
            iosInstall.innerHTML = `
                <div style="position: fixed; bottom: 20px; left: 20px; right: 20px; background: #ff6b6b; color: white; padding: 15px; border-radius: 8px; text-align: center; z-index: 1000;">
                    📱 Install this app: Tap <strong>Share</strong> then <strong>Add to Home Screen</strong>
                    <button onclick="this.parentElement.remove()" style="position: absolute; top: 5px; right: 10px; background: none; border: none; color: white; font-size: 16px;">×</button>
                </div>
            `;
            document.body.appendChild(iosInstall);
            
            // Auto-hide after 10 seconds
            setTimeout(() => {
                if (iosInstall.parentElement) {
                    iosInstall.remove();
                }
            }, 10000);
        }
    </script>
    """
    
    # Inject PWA components
    st.markdown(pwa_head + pwa_js, unsafe_allow_html=True)

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

# Main Streamlit app
def main():
    # Set page config
    st.set_page_config(
        page_title="Driver Drowsiness Detection", 
        layout="centered",
        page_icon="🚗",
        initial_sidebar_state="collapsed"
    )
    
    # Serve PWA files
    serve_pwa_files()
    
    # App header
    st.title("🚗 Driver Drowsiness Detection")
    st.markdown("**Real-time drowsiness detection using computer vision**")
    
    # PWA status indicator
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("ℹ️ PWA Info"):
            st.info("This app supports offline use when installed!")
    
    # Installation guide
    with st.expander("📱 Install as Mobile App", expanded=False):
        st.markdown("""
        ### Install this Progressive Web App for the best experience:
        
        **📱 On Mobile:**
        - **Chrome/Edge**: Tap menu → "Add to Home Screen"
        - **Safari**: Tap share → "Add to Home Screen"
        
        **💻 On Desktop:**
        - **Chrome/Edge**: Click install icon in address bar
        - Or look for "Install App" button
        
        **✨ Benefits:**
        - 🚀 Faster loading
        - 📱 App-like experience
        - 🔒 Works offline
        - 💾 Cached for speed
        """)
    
    # Main camera interface
    st.markdown("---")
    st.markdown("### Live Detection")
    st.markdown("Allow camera access to start drowsiness detection.")
    
    # Run webcam with Streamlit
    webrtc_streamer(
        key="drowsiness-app",
        video_processor_factory=DrowsinessDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Additional info
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("EAR Threshold", f"{EAR_THRESHOLD}")
    with col2:
        st.metric("Alert Frames", f"{CONSEC_FRAMES}")
    with col3:
        st.metric("Status", "🟢 Ready")
    
    # Footer with PWA info
    st.markdown("---")
    st.markdown(
        "<small>💡 **Tip**: Install this app for offline access and better performance!</small>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
