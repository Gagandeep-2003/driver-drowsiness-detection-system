from tkinter import *
from tkinter.ttk import *
import subprocess



root = Tk()
root.geometry('500x500')
style = Style()

style.configure('TButton', font =('calibri', 20, 'bold'), borderwidth = '2')
#root.title('The game')
root.geometry("500x500") 
#tk.resizable(0, 0)
frame = Frame(root)
frame.pack()




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
frame = Frame(root)
frame.pack(side=TOP, pady=40)

button1 = Button(frame, text="Face Detection", command=face)
button1.pack(side=LEFT, padx=10, pady=10)

button2 = Button(frame, text="Blink Detection", command=blink)
button2.pack(side=LEFT, padx=10, pady=10)

button3 = Button(root, text="Quit", command=root.destroy)
button3.pack(side=BOTTOM, pady=30)

root.mainloop()
