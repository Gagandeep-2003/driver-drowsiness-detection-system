
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



def face():
    subprocess.Popen(["python", "face-try.py"])

def blink():
    subprocess.Popen(["python", "blinkDetect.py"])

def on_quit():
    global current_process
    # If a process is running, terminate it!
    if current_process and current_process.poll() is None:
        current_process.terminate()  # Send SIGTERM or equivalent
        try:
            current_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            current_process.kill()  # Force kill if still running
    root.destroy()  # Close the GUI
    
def main():
    root = tk.Tk()
    root.title("Driver Drowsiness Detection System")
    root.geometry("500x500")

    style = ttk.Style()
    style.configure('TButton', font=('Calibri', 20, 'bold'), borderwidth=2)

    frame = ttk.Frame(root, padding=20)
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