
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
    
    
    frame = Frame(root)
    frame.pack(side=TOP, pady=40)

    button1 = Button(frame, text="Face Detection", command=face)
    button1.pack(side=LEFT, padx=10, pady=10)

    button2 = Button(frame, text="Blink Detection", command=blink)
    button2.pack(side=LEFT, padx=10, pady=10)

    button3 = Button(root, text="Quit", command=root.destroy)
    button3.pack(side=BOTTOM, pady=30)

    root.mainloop()


    

    

if __name__ == "__main__":
    main()