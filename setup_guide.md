# Enhanced Driver Drowsiness & Impairment Detection System - Setup Guide

## 🚀 Quick Start

This enhanced system now includes drunk driver detection capabilities alongside the original drowsiness detection.

### New Features Added:
- **Head Movement Analysis** - Detects excessive swaying indicating impairment
- **Delayed Blink Detection** - Identifies slow or delayed blinking patterns
- **Eye Redness Detection** - Analyzes eye color for signs of alcohol consumption
- **Droopy Eyelid Detection** - Detects partially closed eyes indicating impairment
- **Emergency Alert System** - Sends email alerts to emergency contacts
- **Enhanced UI** - Improved interface with configuration options

## 📋 Prerequisites

- Python 3.7 - 3.11
- Webcam/Camera
- Internet connection (for emergency alerts)
- Gmail account (for email alerts)

## 🛠 Installation Steps

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/Parthavi19/driver-drowsiness-detection-system.git
cd driver-drowsiness-detection-system

# Create virtual environment
python -m venv drowsiness_env

# Activate virtual environment
# On Windows:
drowsiness_env\Scripts\activate
# On macOS/Linux:
source drowsiness_env/bin/activate
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# If dlib installation fails, try:
pip install cmake
pip install dlib
```

### 3. Download Required Model Files

Create a `models` directory and download these files:

#### Facial Landmark Predictor:
```bash
# Create models directory
mkdir models

# Download shape predictor (68-point facial landmarks)
# Download from: https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Extract and place in models/ folder
```

#### Haar Cascade (if not present):
```bash
# Download from OpenCV repository or use the existing one
# URL: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
```

### 4. Add Audio Alert File

```bash
# Add alarm.wav file to root directory
# You can use any .wav audio file and rename it to alarm.wav
```

### 5. Configure Email Settings

#### Gmail Setup (Recommended):
1. Enable 2-factor authentication on your Gmail account
2. Generate an App Password:
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate password for "Mail"
3. Use this app password (not your regular Gmail password)

## 🚦 Running the System

### Option 1: Enhanced Detection (Recommended)
```bash
python main.py
```
Then click "🚨 Enhanced Detection" button.

### Option 2: Direct Enhanced Detection
```bash
python enhanced_driver_detection.py
```

### Option 3: Original Modules
```bash
# Original blink detection (now enhanced)
python blinkDetect.py

# Face detection only
python face-try.py

# Lane detection
python lanedetection.py
```

## ⚙️ Configuration

### Emergency Contacts Setup:
1. Run `python main.py`
2. Click "⚙️ Configure Emergency Contacts"
3. Add your email settings and emergency contact emails
4. Save configuration

### Manual Configuration:
Create `emergency_config.json`:
```json
{
  "emergency_contacts": [
    "emergency1@example.com",
    "emergency2@example.com"
  ],
  "email_settings": {
    "sender_email": "your_email@gmail.com",
    "sender_password": "your_app_password"
  }
}
```

## 🎯 Usage Instructions

### Controls During Detection:
- **'q'** - Quit the application
- **'r'** - Reset all alerts and counters
- **ESC** - Exit (in some modules)

### Detection Indicators:

#### Drowsiness Detection:
- **EAR (Eye Aspect Ratio)** - Shows eye openness level
- **Blink Count** - Number of blinks detected
- **DROWSINESS ALERT** - Triggered when eyes closed too long

#### Impairment Detection:
- **Head Pose** - Shows pitch, yaw, roll angles
- **Impairment Indicators** - Shows count of active indicators:
  - Head Sway
  - Delayed Blinking
  - Red Eyes
  - Droopy Eyelids

### Alert Levels:
- **LOW** - Single indicator detected
- **MEDIUM** - Multiple indicators OR drowsiness + some indicators
- **HIGH** - Multiple indicators + drowsiness (emergency alert sent)

## 🔧 Troubleshooting

### Common Issues:

#### 1. Camera Not Working:
```python
# Try different camera indices
cap = cv2.VideoCapture(1)  # Try 1, 2, etc. instead of 0
```

#### 2. Dlib Installation Failed:
```bash
# On Windows:
# Install Visual Studio Build Tools from Microsoft
pip install cmake
pip install dlib

# On macOS:
xcode-select --install
pip install cmake dlib

# On Linux:
sudo apt-get install cmake libopenblas-dev liblapack-dev
pip install dlib
```

#### 3. Face Detection Not Working:
- Ensure good lighting
- Check if `models/shape_predictor_68_face_landmarks.dat` exists
- Try adjusting `FACE_DOWNSAMPLE_RATIO` in the code

#### 4. Email Alerts Not Sending:
- Verify Gmail app password (not regular password)
- Check internet connection
- Ensure 2-factor authentication is enabled
- Try with a test email first

#### 5. Performance Issues:
```python
# Reduce processing load by adjusting these parameters:
FACE_DOWNSAMPLE_RATIO = 2.0  # Increase for faster processing
RESIZE_HEIGHT = 360          # Decrease for better performance
```

## 📁 File Structure

```
driver-drowsiness-detection-system/
├── main.py                          # Enhanced main UI
├── enhanced_driver_detection.py     # New comprehensive detection
├── blinkDetect.py                   # Enhanced with drunk detection
├── face-try.py                      # Original face detection
├── lanedetection.py                 # Lane detection
├── requirements.txt                 # Updated dependencies
├── emergency_config.json            # Email configuration
├── models/
│   ├── shape_predictor_68_face_landmarks.dat
│   └── haarcascade_frontalface_default.xml
├── alarm.wav                        # Audio alert file
└── icons/                           # PWA icons (for web version)
```

## 🧪 Testing the System

### 1. Test Face Detection:
```bash
python face-try.py
# Should show rectangles around detected faces
```

### 2. Test Enhanced Detection:
```bash
python enhanced_driver_detection.py
# Should show multiple detection indicators
```

### 3. Test Email Alerts:
- Configure emergency contacts
- Simulate impairment (look away, close eyes partially)
- Check if email alert is received

## 🔐 Security Notes

- **Never commit email passwords to version control**
- Use Gmail App Passwords, not regular passwords
- Consider using environment variables for sensitive data
- Regularly update dependencies for security patches

## 📊 Performance Optimization

### For Better Performance:
1. **Adjust Resolution**:
   ```python
   RESIZE_HEIGHT = 320  # Lower resolution
   ```

2. **Reduce Face Detection Area**:
   ```python
   FACE_DOWNSAMPLE_RATIO = 2.0  # Process smaller image
   ```

3. **Optimize Detection Frequency**:
   ```python
   # Process every N frames instead of every frame
   if frame_count % 3 == 0:  # Process every 3rd frame
       # Run detection
   ```

## 🆘 Emergency Features

### Automatic Emergency Response:
- System automatically sends emails when severe impairment detected
- Multiple detection methods ensure accuracy
- Configurable sensitivity levels
- Manual reset capability

### Emergency Contact Best Practices:
- Add multiple emergency contacts
- Include family members and local emergency services
- Test email delivery periodically
- Keep contact list updated

## 🔄 Updates and Maintenance

### Regular Maintenance:
1. Update dependencies monthly:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. Check model file integrity
3. Test emergency alert system
4. Review and update emergency contacts

### Contributing:
- Report issues on GitHub
- Submit pull requests for improvements
- Share detection accuracy feedback
- Suggest new safety features

## 📞 Support

For issues and support:
1. Check this troubleshooting guide
2. Review GitHub issues
3. Test with different lighting conditions
4. Verify all dependencies are installed correctly

---

**⚠️ Important Safety Note**: This system is designed to assist in detecting driver impairment but should not be the sole safety measure. Always prioritize responsible driving practices and seek help if experiencing impairment.
