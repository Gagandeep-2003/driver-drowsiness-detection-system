import tkinter as tk
from tkinter import ttk, messagebox
import subprocess

face_proc = None  

def run_face_detection():
    global face_proc
    try:
        face_proc = subprocess.Popen(["python", "face-try.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run face detection:\n{e}")


def run_blink_detection():
    try:
        subprocess.call(["python", "blinkDetect.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run blink detection:\n{e}")

def on_quit(root):
    if face_proc and face_proc.poll() is None:
        face_proc.terminate()
    root.destroy()


def main():
    root = tk.Tk()
    root.title("Driver Drowsiness Detection System")
    root.geometry("500x500")

    style = ttk.Style()
    style.configure('TButton', font=('Calibri', 20, 'bold'), borderwidth=2)

    frame = ttk.Frame(root, padding=20)
    frame.pack(expand=True)

    btn_face = ttk.Button(frame, text="Face Detection", command=run_face_detection)
    btn_face.pack(side=tk.LEFT, padx=10, pady=10)

    btn_blink = ttk.Button(frame, text="Blink Detection", command=run_blink_detection)
    btn_blink.pack(side=tk.RIGHT, padx=10, pady=10)

    btn_quit = ttk.Button(root, text="Quit", command=lambda: on_quit(root))
    btn_quit.pack(side=tk.BOTTOM, pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()