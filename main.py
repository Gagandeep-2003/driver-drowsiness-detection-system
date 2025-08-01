import tkinter as tk
from tkinter.ttk import *
import subprocess
import sys

def blink():
    subprocess.call([sys.executable, "face-try.py"])

def lane():
    subprocess.call([sys.executable, "blinkDetect.py"])

root = tk.Tk()
root.title("Driver Alert System")
root.geometry("500x500")

frame = tk.Frame(root)
frame.pack(pady=50)

button1 = tk.Button(frame, text="Face Detection", fg="red", command=blink, height=2, width=20)
button1.pack(side=tk.LEFT, padx=10)

button2 = tk.Button(frame, text="Blink Detection", fg="red", command=lane, height=2, width=20)
button2.pack(side=tk.RIGHT, padx=10)

button3 = tk.Button(root, text="Quit", fg="red", command=root.destroy, height=2, width=20)
button3.pack(side=tk.BOTTOM, pady=20)

root.mainloop()
