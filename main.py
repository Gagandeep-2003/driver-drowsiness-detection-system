import tkinter as tk
from tkinter import ttk
import subprocess


def run_face_detection():
    subprocess.call(["python", "face-try.py"])


def run_blink_detection():
    subprocess.call(["python", "blinkDetect.py"])


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

    btn_quit = ttk.Button(root, text="Quit", command=root.destroy)
    btn_quit.pack(side=tk.BOTTOM, pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
