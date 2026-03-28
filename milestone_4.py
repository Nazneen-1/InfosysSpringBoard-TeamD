import tkinter as tk
from tkinter import Frame, Label

class GestureDashboardUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Milestone 4: Gesture Control Interface")
        self.root.geometry("1000x650")
        self.root.configure(bg="#f8f9fa")

        self.setup_layout()

    def setup_layout(self):
        # Header
        header = Frame(self.root, bg="#ffffff", height=60, bd=1, relief="ridge")
        header.pack(fill="x", side="top")
        Label(header, text="Gesture Control Interface", font=("Segoe UI", 16, "bold"), fg="#e67e22", bg="#ffffff").pack(side="left", padx=20, pady=15)

        # Main Body
        main_body = Frame(self.root, bg="#f8f9fa")
        main_body.pack(fill="both", expand=True, padx=20, pady=20)

        # Left Panel (Video)
        left_panel = Frame(main_body, bg="#ffffff", bd=1, relief="ridge")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        Label(left_panel, text="📷 Live Gesture Control", font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#333333").pack(anchor="w", padx=15, pady=10)
        
        self.video_lbl = Label(left_panel, bg="black")
        self.video_lbl.pack(padx=15, pady=(0, 15), fill="both", expand=True)

        # Right Panel
        right_panel = Frame(main_body, bg="#f8f9fa", width=350)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)

        # --- Gesture Recognition Box ---
        gesture_frame = Frame(right_panel, bg="#ffffff", bd=1, relief="ridge")
        gesture_frame.pack(fill="x", pady=(0, 15))
        Label(gesture_frame, text="🖐 Gesture Recognition", font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#333333").pack(anchor="w", padx=15, pady=10)

        self.lbl_open = self.create_gesture_row(gesture_frame, "Open Hand", "Distance > 80mm", "Inactive")
        self.lbl_pinch = self.create_gesture_row(gesture_frame, "Pinch", "20mm < Distance < 80mm", "Inactive")
        self.lbl_closed = self.create_gesture_row(gesture_frame, "Closed", "Distance < 20mm", "Inactive")

        # --- Performance Metrics Box ---
        metrics_container = Frame(right_panel, bg="#ffffff", bd=1, relief="ridge")
        metrics_container.pack(fill="both", expand=True)
        Label(metrics_container, text="📊 Performance Metrics", font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#333333").pack(anchor="w", padx=15, pady=10)

        grid_frame = Frame(metrics_container, bg="#ffffff")
        grid_frame.pack(padx=15, pady=5, fill="both", expand=True)

        self.val_vol = self.create_metric_tile(grid_frame, "Current Volume", "0%", 0, 0)
        self.val_dist = self.create_metric_tile(grid_frame, "Finger Distance", "0mm", 0, 1)
        self.val_acc = self.create_metric_tile(grid_frame, "Accuracy", "0%", 1, 0)
        self.val_time = self.create_metric_tile(grid_frame, "Response Time", "0ms", 1, 1)

    def create_gesture_row(self, parent, title, desc, status):
        row = Frame(parent, bg="#fafafa", bd=1, relief="solid")
        row.pack(fill="x", padx=15, pady=5)
        
        info_frame = Frame(row, bg="#fafafa")
        info_frame.pack(side="left", padx=10, pady=8)
        Label(info_frame, text=title, font=("Segoe UI", 10, "bold"), bg="#fafafa", fg="#333333").pack(anchor="w")
        Label(info_frame, text=desc, font=("Segoe UI", 8), bg="#fafafa", fg="#7f8c8d").pack(anchor="w")
        
        status_lbl = Label(row, text=status, font=("Segoe UI", 10, "bold"), bg="#fafafa", fg="#e74c3c")
        status_lbl.pack(side="right", padx=15)
        return status_lbl

    def create_metric_tile(self, parent, title, value, row, col):
        tile = Frame(parent, bg="#fdfdfe", bd=1, relief="ridge", width=140, height=90)
        tile.grid(row=row, column=col, padx=5, pady=5)
        tile.grid_propagate(False)
        
        val_lbl = Label(tile, text=value, font=("Segoe UI", 20, "bold"), fg="#e67e22", bg="#fdfdfe")
        val_lbl.pack(expand=True, pady=(10, 0))
        Label(tile, text=title, font=("Segoe UI", 9), fg="#7f8c8d", bg="#fdfdfe").pack(side="bottom", pady=(0, 10))
        return val_lbl

    def update_video(self, img_tk):
        """Updates the main camera label."""
        self.video_lbl.imgtk = img_tk
        self.video_lbl.configure(image=img_tk)

    def update_gesture_status(self, active_gesture):
        """Highlights the active gesture in the UI list."""
        # Reset all
        self.lbl_open.config(text="Inactive", fg="#e74c3c")
        self.lbl_pinch.config(text="Inactive", fg="#e74c3c")
        self.lbl_closed.config(text="Inactive", fg="#e74c3c")

        # Highlight active
        if active_gesture == "Open Palm":
            self.lbl_open.config(text="Active", fg="#27ae60")
        elif active_gesture == "Pinch":
            self.lbl_pinch.config(text="Active", fg="#27ae60")
        elif active_gesture in ["Fist", "Closed"]:
            self.lbl_closed.config(text="Active", fg="#27ae60")

    def update_metrics(self, vol, dist, acc, response_time):
        """Updates the 4 metric tiles."""
        self.val_vol.config(text=f"{vol}%")
        self.val_dist.config(text=f"{dist}mm")
        self.val_acc.config(text=f"{acc}%")
        self.val_time.config(text=f"{response_time}ms")
