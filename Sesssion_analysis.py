import tkinter as tk
from tkinter import ttk
from tkinter.ttk import *

root = tk.Tk()
root.geometry('600x400')
root.title("Tab Widget")

notebook = ttk.Notebook(root)
live_track = ttk.Frame(notebook)
session_his = ttk.Frame(notebook)
notebook.add(live_track, text = 'Live Tracking')
notebook.add(session_his, text = 'Session History')

notebook.pack()


root.mainloop()
