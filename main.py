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
    config_window.geometry("400x300")
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
    tk.Entry(email_frame, textvariable=sender_email_var, width=30).grid(row=0, column=1, padx=5)
    
    tk.Label(email_frame, text="App Password:").grid(row=1, column=0, sticky='w')
    sender_password_var = tk.StringVar(value=config["email_settings"]["sender_password"])
    tk.Entry(email_frame, textvariable=sender_password_var, width=30, show='*').grid(row=1, column=1, padx=5)
    
    # Emergency contacts frame
    contacts_frame = ttk.LabelFrame(config_window, text="Emergency Contacts", padding=10)
    contacts_frame.pack(fill='both', expand=True, padx=20, pady=10)
    
    # Contacts listbox
    contacts_listbox = tk.Listbox(contacts_frame, height=6)
    contacts_listbox.pack(fill='both', expand=True, pady=(0, 10))
    
    # Populate listbox
    for contact in config["emergency_contacts"]:
        contacts_listbox.insert(tk.END, contact)
    
    # Buttons frame
    buttons_frame = tk.Frame(contacts_frame)
    buttons_frame.pack(fill='x')
    
    # Add contact
    def add_contact():
        contact = tk.simpledialog.askstring("Add Contact", "Enter email address:")
        if contact and '@' in contact:
            contacts_listbox.insert(tk.END, contact)
    
    # Remove contact
    def remove_contact():
        selection = contacts_listbox.curselection()
        if selection:
            contacts_listbox.delete(selection)
    
    tk.Button(buttons_frame, text="Add Contact", command=add_contact).pack(side='left', padx=5)
    tk.Button(buttons_frame, text="Remove Contact", command=remove_contact).pack(side='left', padx=5)
    
    # Save configuration
    def save_config():
        try:
            new_config = {
                "emergency_contacts": list(contacts_listbox.get(0, tk.END)),
                "email_settings": {
                    "sender_email": sender_email_var.get(),
                    "sender_password": sender_password_var.get()
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(new_config, f, indent=2)
            
            messagebox.showinfo("Success", "Configuration saved successfully!")
            config_window.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration:\n{e}")
    
    # Save button
    tk.Button(config_window, text="Save Configuration", command=save_config,
              bg='#4CAF50', fg='white', font=('Arial', 10, 'bold')).pack(pady=10)

def show_help():
    """Show help information"""
    help_text = """
    Enhanced Driver Drowsiness Detection System
    
    Features:
    • Face Detection - Basic face recognition
    • Blink Detection - Original drowsiness detection
    • Enhanced Detection - Combines drowsiness + drunk detection
    • Lane Detection - Road lane tracking
    
    Enhanced Detection includes:
    ✓ Head movement analysis (swaying)
    ✓ Delayed blinking detection
    ✓ Eye redness detection
    ✓ Droopy eyelid detection
    ✓ Emergency alert system
    
    Setup Required:
    1. Configure emergency contacts
    2. Set up email credentials (use Gmail app password)
    3. Ensure webcam is connected
    4. Install required models and audio files
    
    Tips:
    • Use 'r' key to reset alerts during detection
    • Ensure good lighting for face detection
    • Configure email settings for emergency alerts
    """
    
    messagebox.showinfo("Help - Enhanced Detection System", help_text)

def on_quit(root):
    global face_proc, blink_proc, enhanced_proc
    
    # Terminate all running processes
    for proc in [face_proc, blink_proc, enhanced_proc]:
        if proc and proc.poll() is None:
            proc.terminate()
    
    root.destroy()

def main():
    root = tk.Tk()
    root.title("Enhanced Driver Drowsiness Detection System")
    root.geometry("600x500")
    root.configure(bg='#2c3e50')
    
    # Style configuration
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('Title.TLabel', font=('Arial', 16, 'bold'), 
                   background='#2c3e50', foreground='white')
    style.configure('Subtitle.TLabel', font=('Arial', 10), 
                   background='#2c3e50', foreground='#ecf0f1')
    style.configure('Custom.TButton', font=('Arial', 12, 'bold'), padding=10)
    
    # Main title
    title_label = ttk.Label(root, text="Enhanced Driver Safety System",
                           style='Title.TLabel')
    title_label.pack(pady=20)
    
    subtitle_label = ttk.Label(root, text="Drowsiness & Impairment Detection",
                              style='Subtitle.TLabel')
    subtitle_label.pack(pady=(0, 30))
    
    # Main frame
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(expand=True, fill='both')
    
    # Detection buttons frame
    detection_frame = ttk.LabelFrame(main_frame, text="Detection Modules", padding=15)
    detection_frame.pack(fill='x', pady=10)
    
    # Button grid
    buttons_info = [
        ("Face Detection", run_face_detection, "Basic face recognition", '#3498db'),
        ("Blink Detection", run_blink_detection, "Original drowsiness detection", '#e74c3c'),
        ("🚨 Enhanced Detection", run_enhanced_detection, "Drowsiness + Drunk detection", '#e67e22'),
        ("Lane Detection", run_lane_detection, "Road lane tracking", '#27ae60')
    ]
    
    for i, (text, command, desc, color) in enumerate(buttons_info):
        row = i // 2
        col = i % 2
        
        btn_frame = tk.Frame(detection_frame, bg='white', relief='raised', bd=1)
        btn_frame.grid(row=row, column=col, padx=10, pady=10, sticky='ew')
        
        btn = tk.Button(btn_frame, text=text, command=command,
                       font=('Arial', 11, 'bold'), bg=color, fg='white',
                       relief='flat', padx=20, pady=15)
        btn.pack(fill='x')
        
        desc_label = tk.Label(btn_frame, text=desc, font=('Arial', 8),
                             bg='white', fg='#666')
        desc_label.pack(pady=(5, 10))
    
    # Configure grid weights
    detection_frame.columnconfigure(0, weight=1)
    detection_frame.columnconfigure(1, weight=1)
    
    # Configuration frame
    config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding=15)
    config_frame.pack(fill='x', pady=10)
    
    config_btn = tk.Button(config_frame, text="⚙️ Configure Emergency Contacts",
                          command=configure_emergency_contacts,
                          font=('Arial', 10, 'bold'), bg='#9b59b6', fg='white',
                          relief='flat', padx=15, pady=8)
    config_btn.pack(side='left', padx=5)
    
    help_btn = tk.Button(config_frame, text="❓ Help & Setup Guide",
                        command=show_help,
                        font=('Arial', 10, 'bold'), bg='#34495e', fg='white',
                        relief='flat', padx=15, pady=8)
    help_btn.pack(side='right', padx=5)
    
    # Status frame
    status_frame = ttk.Frame(main_frame)
    status_frame.pack(fill='x', pady=20)
    
    status_label = tk.Label(status_frame, 
                           text="💡 Tip: Use Enhanced Detection for comprehensive safety monitoring",
                           font=('Arial', 9, 'italic'), bg='#2c3e50', fg='#f39c12')
    status_label.pack()
    
    # Quit button
    quit_btn = tk.Button(root, text="Exit System", 
                        command=lambda: on_quit(root),
                        font=('Arial', 11, 'bold'), bg='#c0392b', fg='white',
                        relief='flat', padx=30, pady=10)
    quit_btn.pack(side='bottom', pady=20)
    
    # Check for required files
    def check_requirements():
        missing_files = []
        required_files = [
            "models/shape_predictor_68_face_landmarks.dat",
            "models/haarcascade_frontalface_default.xml",
            "alarm.wav"
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            warning_msg = "Missing required files:\n" + "\n".join(missing_files)
            warning_msg += "\n\nPlease download these files for full functionality."
            messagebox.showwarning("Missing Files", warning_msg)
    
    # Check requirements after window is shown
    root.after(1000, check_requirements)
    
    root.mainloop()

if __name__ == "__main__":
    import tkinter.simpledialog
    main()
