import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import threading

face_proc = None
is_dark_mode = False  # Tracks current theme state

def run_face_detection(btn_face=None):
    global face_proc
    script = "face-try.py"
    if not os.path.exists(script):
        messagebox.showerror("Error", f"Script '{script}' not found.")
        return
    if btn_face:
        btn_face.config(state=tk.DISABLED)
    try:
        face_proc = subprocess.Popen(["python", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        def check_proc():
            try:
                stdout, stderr = face_proc.communicate()
                exit_code = face_proc.returncode
                if exit_code != 0:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    messagebox.showerror("Face Detection Error", f"face-try.py exited with code {exit_code}:\n{error_msg}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to run face detection:\n{e}")
            finally:
                if btn_face:
                    btn_face.config(state=tk.NORMAL)
        threading.Thread(target=check_proc, daemon=True).start()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start face detection:\n{e}")
        if btn_face:
            btn_face.config(state=tk.NORMAL)

def run_blink_detection(btn_blink=None):
    script = "blinkDetect.py"
    if not os.path.exists(script):
        messagebox.showerror("Error", f"Script '{script}' not found.")
        return
    if btn_blink:
        btn_blink.config(state=tk.DISABLED)
    def call_script():
        try:
            subprocess.call(["python", script])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run blink detection:\n{e}")
        finally:
            if btn_blink:
                btn_blink.config(state=tk.NORMAL)
    threading.Thread(target=call_script, daemon=True).start()

def toggle_theme(root, frame, toggle_btn):
    global is_dark_mode

    if is_dark_mode:
        # Switch to light mode
        root.configure(bg="#f0f0f0")
        frame.configure(style="Light.TFrame")
        toggle_btn.config(text="Switch to Dark Mode")
        ttk.Style().configure('TButton', background="#ffffff", foreground="#000000")
    else:
        # Switch to dark mode
        root.configure(bg="#2e2e2e")
        frame.configure(style="Dark.TFrame")
        toggle_btn.config(text="Switch to Light Mode")
        ttk.Style().configure('TButton', background="#444444", foreground="#ffffff")

    is_dark_mode = not is_dark_mode

def run_analysis():
    try:
        subprocess.call(["python", "Session_Dashboard.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run Session Dashboard:\n{e}")
    

def on_quit(root):
    if face_proc and face_proc.poll() is None:
        face_proc.terminate()
    root.destroy()

def main():
    root = tk.Tk()
    root.title("Driver Drowsiness Detection System")
    root.geometry("500x500")
    root.configure(bg="#f0f0f0")  # Default light background
    style = ttk.Style()
    style.theme_use("clam")

    # Frame styles
    style.configure("Light.TFrame", background="#f0f0f0")
    style.configure("Dark.TFrame", background="#2e2e2e")

    # Button styles
    style.configure('TButton',
                    font=('Segoe UI', 14, 'bold'),
                    padding=10,
                    borderwidth=1,
                    relief="raised")

    frame = ttk.Frame(root, padding=20, style="Light.TFrame")
    frame.pack(expand=True)

    btn_face = ttk.Button(frame, text="Face Detection")
    btn_face.config(command=lambda: run_face_detection(btn_face))
    btn_face.grid(row=0, column=0, padx=15, pady=15)

    btn_blink = ttk.Button(frame, text="Blink Detection")
    btn_blink.config(command=lambda: run_blink_detection(btn_blink))
    btn_blink.grid(row=0, column=1, padx=15, pady=15)

    btn_analysis = ttk.Button(frame, text="Blink Analysis", command=run_analysis)
    btn_analysis.grid(row=1, column=0, columnspan=2, padx=15, pady=15)

    # Toggle button
    btn_toggle = ttk.Button(root, text="Switch to Dark Mode")
    btn_toggle.config(command=lambda: toggle_theme(root, frame, btn_toggle))
    btn_toggle.pack(pady=10)

    # Quit button
    btn_quit = ttk.Button(root, text="Quit", command=lambda: on_quit(root))
    btn_quit.pack(side=tk.BOTTOM, pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()