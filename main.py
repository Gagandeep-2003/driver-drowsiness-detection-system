import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import json
import os

face_proc = None  
blink_proc = None
enhanced_proc = None

def run_face_detection():
    global face_proc
    try:
        face_proc = subprocess.Popen(["python", "face-try.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run face detection:\n{e}")

def run_blink_detection():
    global blink_proc
    try:
        blink_proc = subprocess.Popen(["python", "blinkDetect.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run blink detection:\n{e}")

def run_enhanced_detection():
    global enhanced_proc
    try:
        enhanced_proc = subprocess.Popen(["python", "enhanced_driver_detection.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run enhanced detection:\n{e}")

def run_lane_detection():
    try:
        subprocess.Popen(["python", "lanedetection.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run lane detection:\n{e}")

def configure_emergency_contacts():
    """Open configuration window for emergency contacts"""
    config_window = tk.Toplevel()
    config_window.title("Emergency Contacts Configuration")
    config_window.geometry("500x400")
    config_window.grab_set()
    
    # Load existing config
    config_file = "emergency_config.json"
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except:
        config = {
            "emergency_contacts": ["emergency@example.com"],
            "email_settings": {
                "sender_email": "your_email@gmail.com",
                "sender_password": "your_app_password"
            }
        }
    
    # Create form
    tk.Label(config_window, text="Emergency Contacts Configuration", 
             font=('Arial', 14, 'bold')).pack(pady=10)
    
    # Email settings frame
    email_frame = ttk.LabelFrame(config_window, text="Email Settings", padding=10)
    email_frame.pack(fill='x', padx=20, pady=10)
    
    tk.Label(email_frame, text="Sender Email:").grid(row=0, column=0, sticky='w')
    sender_email_var = tk.StringVar(value=config["email_settings"]["sender_email"])
    tk.Entry(email_frame, textvariable=sender_email_var, width=35).grid(row=0, column=1, padx=5)
    
    tk.Label(email_frame, text="App Password:").grid(row=1, column=0, sticky='w')
    sender_password_var = tk.StringVar(value=config["email_settings"]["sender_password"])
    tk.Entry(email_frame, textvariable=sender_password_var, width=35, show='*').grid(row=1, column=1, padx=5)
    
    # Help text for Gmail setup
    help_text = tk.Text(email_frame, height=3, width=50, wrap=tk.WORD, font=('Arial', 8))
    help_text.insert(tk.END, "Gmail Setup: Enable 2FA → Go to Google Account Settings → Security → App passwords → Generate app-specific password")
    help_text.config(state=tk.DISABLED, bg='#f0f0f0')
    help_text.grid(row=2, column=0, columnspan=2, pady=5, sticky='ew')
    
    # Emergency contacts frame
    contacts_frame = ttk.LabelFrame(config_window, text="Emergency Contacts", padding=10)
    contacts_frame.pack(fill='both', expand=True, padx=20, pady=10)
    
    # Contacts listbox with scrollbar
    listbox_frame = tk.Frame(contacts_frame)
    listbox_frame.pack(fill='both', expand=True, pady=(0, 10))
    
    contacts_listbox = tk.Listbox(listbox_frame, height=8)
    scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
    contacts_listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=contacts_listbox.yview)
    
    contacts_listbox.pack(side=tk.LEFT, fill='both', expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Populate listbox
    for contact in config["emergency_contacts"]:
        contacts_listbox.insert(tk.END, contact)
    
    # Buttons frame
    buttons_frame = tk.Frame(contacts_frame)
    buttons_frame.pack(fill='x')
    
    # Add contact
    def add_contact():
        contact = tk.simpledialog.askstring("Add Contact", "Enter email address:")
        if contact and '@' in contact and '.' in contact:
            contacts_listbox.insert(tk.END, contact)
        elif contact:
            messagebox.showwarning("Invalid Email", "Please enter a valid email address")
    
    # Remove contact
    def remove_contact():
        selection = contacts_listbox.curselection()
        if selection:
            contacts_listbox.delete(selection)
        else:
            messagebox.showwarning("No Selection", "Please select a contact to remove")
    
    tk.Button(buttons_frame, text="Add Contact", command=add_contact, 
              bg='#4CAF50', fg='white', font=('Arial', 9, 'bold')).pack(side='left', padx=5)
    tk.Button(buttons_frame, text="Remove Contact", command=remove_contact,
              bg='#f44336', fg='white', font=('Arial', 9, 'bold')).pack(side='left', padx=5)
    
    # Test email button
    def test_email():
        try:
            test_config = {
                "emergency_contacts": [sender_email_var.get()],
                "email_settings": {
                    "sender_email": sender_email_var.get(),
                    "sender_password": sender_password_var.get()
                }
            }
            
            # Simple test - just validate format for now
            if '@' in sender_email_var.get() and sender_password_var.get():
                messagebox.showinfo("Test", "Email configuration looks valid!\nActual test will occur during alert.")
            else:
                messagebox.showerror("Test Failed", "Please provide valid email and password")
                
        except Exception as e:
            messagebox.showerror("Test Failed", f"Configuration error:\n{e}")
    
    tk.Button(buttons_frame, text="Test Config", command=test_email,
              bg='#FF9800', fg='white', font=('Arial', 9, 'bold')).pack(side='right', padx=5)
    
    # Save configuration
    def save_config():
        try:
            contacts_list = list(contacts_listbox.get(0, tk.END))
            if not contacts_list:
                messagebox.showwarning("Warning", "Please add at least one emergency contact")
                return
                
            new_config = {
                "emergency_contacts": contacts_list,
                "email_settings": {
                    "sender_email": sender_email_var.get(),
                    "sender_password": sender_password_var.get()
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(new_config, f, indent=2)
            
            messagebox.showinfo("Success", f"Configuration saved successfully!\nContacts: {len(contacts_list)}")
            config_window.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration:\n{e}")
    
    # Save button
    tk.Button(config_window, text="Save Configuration", command=save_config,
              bg='#2196F3', fg='white', font=('Arial', 11, 'bold'), 
              padx=20, pady=8).pack(pady=15)

def show_help():
    """Show detailed help information"""
    help_window = tk.Toplevel()
    help_window.title("Enhanced Driver Detection System - Help")
    help_window.geometry("700x600")
    help_window.grab_set()
    
    # Create scrollable text widget
    text_frame = tk.Frame(help_window)
    text_frame.pack(fill='both', expand=True, padx=10, pady=10)
    
    help_text = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10))
    scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=help_text.yview)
    help_text.config(yscrollcommand=scrollbar.set)
    
    help_content = """
    🚗 ENHANCED DRIVER DROWSINESS & IMPAIRMENT DETECTION SYSTEM 🚗
    
    ═══════════════════════════════════════════════════════════════
    
    📋 SYSTEM OVERVIEW:
    This advanced system combines multiple detection algorithms to monitor driver
    safety in real-time, detecting both drowsiness and signs of impairment.
    
    🔧 DETECTION MODULES:
    
    1. 👤 Face Detection (Basic)
       • Simple face recognition using Haar cascades
       • Foundation for other detection systems
       • Good for testing camera functionality
    
    2. 👁️ Blink Detection (Original)
       • Eye Aspect Ratio (EAR) based drowsiness detection
       • Blink counting and pattern analysis
       • Audio alerts for drowsiness
       • Session tracking with statistics
    
    3. 🚨 Enhanced Detection (RECOMMENDED)
       • Complete drowsiness detection (EAR analysis)
       • Advanced impairment detection:
         ✓ Head movement/sway analysis
         ✓ Delayed blinking patterns
         ✓ Eye redness detection
         ✓ Droopy eyelid detection
       • Emergency alert system via email
       • Real-time visual indicators
       • Comprehensive session tracking
    
    4. 🛣️ Lane Detection
       • Road lane tracking using computer vision
       • Hough transform line detection
       • Video processing capabilities
    
    ⚙️ SETUP REQUIREMENTS:
    
    📁 Required Files:
    • models/shape_predictor_68_face_landmarks.dat
    • models/haarcascade_frontalface_default.xml
    • alarm.wav (audio alert file)
    
    📧 Email Configuration:
    • Gmail account with 2-factor authentication enabled
    • App-specific password (NOT regular password)
    • Emergency contact email addresses
    
    🔧 Installation Steps:
    1. Install Python dependencies: pip install -r requirements.txt
    2. Download dlib face landmarks model
    3. Configure emergency contacts via settings
    4. Test camera access
    
    🎯 ENHANCED DETECTION FEATURES:
    
    Drowsiness Indicators:
    • Eye Aspect Ratio (EAR) monitoring
    • Consecutive frame analysis
    • Blink frequency patterns
    • Audio and visual alerts
    
    Impairment Indicators:
    • Head Pose Analysis: Detects excessive head swaying/movement
    • Delayed Blink Detection: Identifies slow or delayed blinking
    • Eye Color Analysis: Detects redness indicating possible impairment
    • Eyelid Position: Monitors for droopy or partially closed eyelids
    
    🚨 Alert System:
    • Visual warnings on screen
    • Audio alarm for drowsiness
    • Email alerts to emergency contacts
    • Severity-based escalation
    
    📊 Session Tracking:
    • Real-time EAR values
    • Alert frequency
    • Session duration
    • Detection statistics
    
    ⌨️ KEYBOARD CONTROLS:
    • 'q' - Quit detection
    • 'r' - Reset all alerts
    • ESC - Exit (in some modules)
    
    🔍 TROUBLESHOOTING:
    
    Camera Issues:
    • Check camera permissions
    • Close other applications using camera
    • Try different camera indices (0, 1, 2...)
    
    Detection Issues:
    • Ensure good lighting conditions
    • Position face clearly in camera view
    • Check if required model files exist
    
    Email Issues:
    • Verify 2FA is enabled on Gmail
    • Use app password, not regular password
    • Check internet connection
    • Verify email addresses are valid
    
    Performance Issues:
    • Close unnecessary applications
    • Reduce camera resolution if needed
    • Ensure adequate system resources
    
    📈 OPTIMAL USAGE:
    • Use Enhanced Detection for comprehensive monitoring
    • Configure email alerts before driving
    • Test system in safe environment first
    • Ensure stable camera mounting
    • Maintain good lighting in vehicle
    
    ⚠️ IMPORTANT NOTES:
    • This system is a safety aid, not a replacement for responsible driving
    • Regular breaks and proper rest are essential
    • System accuracy depends on lighting and camera quality
    • Email alerts require internet connection
    
    🔒 PRIVACY:
    • All processing is done locally
    • No data is sent to external servers
    • Video is processed in real-time only
    • Email alerts contain summary information only
    
    For technical support or questions, check the project documentation.
    """
    
    help_text.insert(tk.END, help_content)
    help_text.config(state=tk.DISABLED)
    
    help_text.pack(side=tk.LEFT, fill='both', expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    tk.Button(help_window, text="Close", command=help_window.destroy,
              bg='#607D8B', fg='white', font=('Arial', 10, 'bold')).pack(pady=10)

def show_about():
    """Show about information"""
    about_text = """
    Enhanced Driver Drowsiness & Impairment Detection System
    Version 2.0
    
    🎯 Mission: Enhance road safety through advanced computer vision
    
    🔬 Technology Stack:
    • OpenCV - Computer vision processing
    • dlib - Facial landmark detection  
    • NumPy/SciPy - Mathematical computations
    • Python - Core development
    
    🚀 Features:
    • Real-time drowsiness detection
    • Advanced impairment analysis
    • Emergency alert system
    • Session tracking & analytics
    
    👥 Developed for: Driver Safety & Road Traffic Management
    
    ⚖️ License: Educational and Research Use
    
    ⚠️ Disclaimer: This system is designed as a safety aid. 
    It does not replace the need for responsible driving practices, 
    adequate rest, and adherence to traffic laws.
    """
    
    messagebox.showinfo("About - Enhanced Detection System", about_text)

def check_system_requirements():
    """Check if system has required components"""
    missing_items = []
    
    # Check required files
    required_files = [
        ("models/shape_predictor_68_face_landmarks.dat", "Facial landmarks model"),
        ("models/haarcascade_frontalface_default.xml", "Face detection cascade"),
        ("alarm.wav", "Audio alert file")
    ]
    
    for file_path, description in required_files:
        if not os.path.exists(file_path):
            missing_items.append(f"• {description}: {file_path}")
    
    # Check Python modules
    required_modules = ['cv2', 'dlib', 'numpy', 'scipy']
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_items.append(f"• Python module: {module}")
    
    if missing_items:
        warning_msg = "⚠️ MISSING SYSTEM COMPONENTS:\n\n" + "\n".join(missing_items)
        warning_msg += "\n\n📥 Download required files and install dependencies for full functionality."
        warning_msg += "\n\n💡 See Help section for detailed setup instructions."
        messagebox.showwarning("System Requirements Check", warning_msg)
        return False
    else:
        messagebox.showinfo("System Check", "✅ All required components found!\nSystem ready for operation.")
        return True

def on_quit(root):
    global face_proc, blink_proc, enhanced_proc
    
    # Terminate all running processes
    processes = [
        (face_proc, "Face Detection"), 
        (blink_proc, "Blink Detection"), 
        (enhanced_proc, "Enhanced Detection")
    ]
    
    terminated_count = 0
    for proc, name in processes:
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                terminated_count += 1
                print(f"Terminated {name} process")
            except Exception as e:
                print(f"Error terminating {name}: {e}")
    
    if terminated_count > 0:
        print(f"Terminated {terminated_count} running detection process(es)")
    
    print("Enhanced Detection System shutdown complete")
    root.destroy()

def main():
    root = tk.Tk()
    root.title("Enhanced Driver Drowsiness & Impairment Detection System v2.0")
    root.geometry("800x700")
    root.configure(bg='#1a237e')
    
    # Configure modern styling
    style = ttk.Style()
    style.theme_use('clam')
    
    # Custom color scheme
    style.configure('Title.TLabel', font=('Arial', 18, 'bold'), 
                   background='#1a237e', foreground='white')
    style.configure('Subtitle.TLabel', font=('Arial', 11), 
                   background='#1a237e', foreground='#e3f2fd')
    style.configure('Custom.TButton', font=('Arial', 12, 'bold'), padding=12)
    
    # Header section
    header_frame = tk.Frame(root, bg='#1a237e')
    header_frame.pack(fill='x', pady=20)
    
    title_label = ttk.Label(header_frame, text="🚗 Enhanced Driver Safety System",
                           style='Title.TLabel')
    title_label.pack()
    
    subtitle_label = ttk.Label(header_frame, text="Advanced Drowsiness & Impairment Detection with Emergency Alerts",
                              style='Subtitle.TLabel')
    subtitle_label.pack(pady=(5, 0))
    
    # Main content frame
    main_frame = tk.Frame(root, bg='#f5f5f5')
    main_frame.pack(expand=True, fill='both', padx=20, pady=10)
    
    # Detection modules frame
    detection_frame = ttk.LabelFrame(main_frame, text="🔍 Detection Modules", padding=20)
    detection_frame.pack(fill='x', pady=10)
    
    # Create detection buttons in a 2x2 grid
    buttons_info = [
        ("👤 Face Detection", run_face_detection, "Basic face recognition & testing", '#2196F3', 'white'),
        ("👁️ Blink Detection", run_blink_detection, "Original drowsiness detection system", '#FF5722', 'white'),
        ("🚨 Enhanced Detection", run_enhanced_detection, "Complete impairment detection + alerts", '#4CAF50', 'white'),
        ("🛣️ Lane Detection", run_lane_detection, "Road lane tracking system", '#9C27B0', 'white')
    ]
    
    buttons_grid = tk.Frame(detection_frame)
    buttons_grid.pack(expand=True, fill='both')
    
    for i, (text, command, desc, bg_color, fg_color) in enumerate(buttons_info):
        row = i // 2
        col = i % 2
        
        btn_frame = tk.Frame(buttons_grid, bg='white', relief='raised', bd=2, padx=10, pady=10)
        btn_frame.grid(row=row, column=col, padx=15, pady=15, sticky='ew')
        
        btn = tk.Button(btn_frame, text=text, command=command,
                       font=('Arial', 12, 'bold'), bg=bg_color, fg=fg_color,
                       relief='flat', padx=25, pady=18, cursor='hand2')
        btn.pack(fill='x')
        
        desc_label = tk.Label(btn_frame, text=desc, font=('Arial', 9),
                             bg='white', fg='#555', wraplength=200)
        desc_label.pack(pady=(8, 5))
    
    # Configure grid weights for responsive layout
    buttons_grid.columnconfigure(0, weight=1)
    buttons_grid.columnconfigure(1, weight=1)
    
    # Configuration and help frame
    config_frame = ttk.LabelFrame(main_frame, text="⚙️ Configuration & Support", padding=15)
    config_frame.pack(fill='x', pady=10)
    
    config_buttons = [
        ("📧 Configure Emergency Contacts", configure_emergency_contacts, '#E91E63'),
        ("❓ Help & Setup Guide", show_help, '#607D8B'),
        ("🔍 Check System Requirements", check_system_requirements, '#795548'),
        ("ℹ️ About System", show_about, '#455A64')
    ]
    
    config_grid = tk.Frame(config_frame)
    config_grid.pack(fill='x')
    
    for i, (text, command, color) in enumerate(config_buttons):
        btn = tk.Button(config_grid, text=text, command=command,
                       font=('Arial', 10, 'bold'), bg=color, fg='white',
                       relief='flat', padx=20, pady=10, cursor='hand2')
        btn.grid(row=i//2, column=i%2, padx=8, pady=5, sticky='ew')
    
    config_grid.columnconfigure(0, weight=1)
    config_grid.columnconfigure(1, weight=1)
    
    # Status and tips frame
    status_frame = tk.Frame(main_frame, bg='#e8f5e8', relief='groove', bd=2)
    status_frame.pack(fill='x', pady=15)
    
    tip_label = tk.Label(status_frame, 
                        text="💡 Recommendation: Use Enhanced Detection for comprehensive safety monitoring",
                        font=('Arial', 10, 'italic'), bg='#e8f5e8', fg='#2e7d32', 
                        pady=12)
    tip_label.pack()
    
    # Footer with quit button
    footer_frame = tk.Frame(root, bg='#1a237e')
    footer_frame.pack(fill='x', side='bottom')
    
    quit_btn = tk.Button(footer_frame, text="🚪 Exit System", 
                        command=lambda: on_quit(root),
                        font=('Arial', 12, 'bold'), bg='#d32f2f', fg='white',
                        relief='flat', padx=40, pady=12, cursor='hand2')
    quit_btn.pack(pady=20)
    
    # Bind window close event
    root.protocol("WM_DELETE_WINDOW", lambda: on_quit(root))
    
    # Auto-check requirements after window loads
    root.after(2000, check_system_requirements)
    
    print("🚀 Enhanced Driver Detection System v2.0 Started")
    print("🔧 Features: Drowsiness Detection + Impairment Analysis + Emergency Alerts")
    
    root.mainloop()

if __name__ == "__main__":
    import tkinter.simpledialog
    main()
