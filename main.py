# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:04:42 2019

@author: Lenovo
"""

from tkinter import * 
from tkinter.ttk import * 
import subprocess    

def blink():
    subprocess.call(["python", "face-try.py"])
    
def lane():
    subprocess.call(["python", "blinkDetect.py"])

root = Tk()
root.geometry('500x500')
style = Style()
style.configure('TButton', font =('calibri', 20, 'bold'), borderwidth = '2')
#root.title('The game')
root.geometry("500x500") 
#tk.resizable(0, 0)
frame = Frame(root)
frame.pack()


button1 = Button(frame, text="Face Detection", fg="red", command=blink,height=25, width=10)
button1.pack(side=root.LEFT)

button2 = Button(frame, text="Blink Detection", fg="red",command=lane)
button2.pack(side=root.RIGHT)

button3 = Button(frame, text="Quit", fg="red",command=root.destroy)
button3.pack(side=root.BOTTOM)


root.mainloop()
