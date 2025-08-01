# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 17:27:12 2019
@author: Lenovo
"""

import tkinter as tk
from tkinter.ttk import *
import subprocess
import sys
from threading import Thread

def run_script(script_name):
    def target():
        subprocess.call([sys.executable, script_name])
    Thread(target=target).start()

def face():
    run_script("face-try.py")

def blink():
    run_script("blinkDetect.py")

def lane():
    run_script("lanedetection.py")

# GUI setup
root = tk.Tk()
root.geometry('300x550')
root.title('Drowsiness Detection System')

style = Style()
style.configure('TButton', font=('calibri', 16, 'bold'), borderwidth='2', padding=10)

# Buttons
btn1 = Button(root, text='Face Detection', command=face)
btn1.pack(pady=40)

btn2 = Button(root, text='Blink Detection', command=blink)
btn2.pack(pady=20)

btn4 = Button(root, text='Lane Detection', command=lane)
btn4.pack(pady=30)

btn3 = Button(root, text='Quit', command=root.destroy)
btn3.pack(pady=30)

root.mainloop()
