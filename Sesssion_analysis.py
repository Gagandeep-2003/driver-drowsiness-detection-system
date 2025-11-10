import tkinter as tk
from tkinter import ttk

# Constants
WINDOW_SIZE = "600x400"
APP_TITLE = "Session Analysis"


def build_live_tracking_tab(parent):
    """Build the Live Tracking tab interface."""
    frame = ttk.Frame(parent, padding=10)
    ttk.Label(frame, text="Live Tracking — TODO").pack(anchor="w")
    # TODO: Add live tracking functionality here
    return frame


def build_session_history_tab(parent):
    """Build the Session History tab interface."""
    frame = ttk.Frame(parent, padding=10)
    ttk.Label(frame, text="Session History — TODO").pack(anchor="w")
    # TODO: Add session history functionality here
    return frame


def main():
    """Main application entry point."""
    root = tk.Tk()
    root.geometry(WINDOW_SIZE)
    root.title(APP_TITLE)

    notebook = ttk.Notebook(root)
    
    # Build tabs using helper functions
    live_track = build_live_tracking_tab(notebook)
    session_his = build_session_history_tab(notebook)

    notebook.add(live_track, text="Live Tracking")
    notebook.add(session_his, text="Session History")
    notebook.pack(expand=True, fill="both")

    root.mainloop()


if __name__ == "__main__":
    main()
