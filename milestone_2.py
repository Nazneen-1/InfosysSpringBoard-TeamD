import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, simpledialog
from PIL import Image, ImageTk
import math
import threading
import time

class GestureRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Milestone 2: Gesture Recognition and Distance Measurement Module")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1100x720")

        self.cap = None
        self.running = False
        self.paused = False

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands_model = None

        self.distance_mm = 0
        self.gesture_name = "Unknown"
        self.focal_length = 800  # tunable in Settings

        # Gesture thresholds (mm)
        self.open_hand_dist  = 80
        self.pinch_dist_min  = 20
        self.pinch_dist_max  = 60
        self.closed_dist     = 20

        self.build_ui()

    # ──────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────
    def build_ui(self):
        # Title bar
        title_frame = tk.Frame(self.root, bg="#6A0DAD", pady=12)
        title_frame.pack(fill="x")

        tk.Label(title_frame,
                 text="Gesture Recognition and Distance Measurement Module",
                 font=("Arial", 13, "bold"), fg="white", bg="#6A0DAD").pack(anchor="w", padx=15)

        tk.Label(title_frame,
                 text="Module: Gesture Recognition and Distance Measurement Module • "
                      "Distance calculation between thumb and index finger tips • "
                      "Gesture classification based on distance • Real-time gesture annotations",
                 font=("Arial", 9), fg="#E1BEE7", bg="#6A0DAD",
                 wraplength=1060, justify="left").pack(anchor="w", padx=15)

        # Sub-header
        sub = tk.Frame(self.root, bg="#f5f5f5", pady=7)
        sub.pack(fill="x")

        tk.Label(sub, text="Gesture Recognition Interface",
                 font=("Arial", 11, "bold"), bg="#f5f5f5").pack(side="left", padx=15)

        btn_frame = tk.Frame(sub, bg="#f5f5f5")
        btn_frame.pack(side="right", padx=10)

        tk.Button(btn_frame, text="▶  Start",    bg="#4CAF50", fg="white",
                  font=("Arial", 9, "bold"), relief="flat", padx=10, pady=4,
                  command=self.start_camera).pack(side="left", padx=3)

        tk.Button(btn_frame, text="⏸  Pause",   bg="#FF9800", fg="white",
                  font=("Arial", 9, "bold"), relief="flat", padx=10, pady=4,
                  command=self.toggle_pause).pack(side="left", padx=3)

        tk.Button(btn_frame, text="⚙  Settings", bg="#607D8B", fg="white",
                  font=("Arial", 9, "bold"), relief="flat", padx=10, pady=4,
                  command=self.open_settings).pack(side="left", padx=3)

        # Main content
        content = tk.Frame(self.root, bg="#e0e0e0")
        content.pack(fill="both", expand=True, padx=10, pady=8)

        # Video feed
        self.video_label = tk.Label(content, bg="black", width=700, height=480)
        self.video_label.pack(side="left", padx=(0, 8))

        # Right panel
        right = tk.Frame(content, bg="#e0e0e0", width=300)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # Distance Measurement card
        dist_card = tk.Frame(right, bg="white", relief="flat", bd=1)
        dist_card.pack(fill="x", pady=(0, 8))

        header = tk.Frame(dist_card, bg="#424242")
        header.pack(fill="x")
        tk.Label(header, text="▬  Distance Measurement",
                 font=("Arial", 10, "bold"), fg="white", bg="#424242",
                 pady=6).pack(anchor="w", padx=10)

        self.dist_var = tk.StringVar(value="0")
        tk.Label(dist_card, textvariable=self.dist_var,
                 font=("Arial", 44, "bold"), fg="#212121",
                 bg="white").pack(pady=(10, 0))

        tk.Label(dist_card, text="millimeters",
                 font=("Arial", 10), fg="#888", bg="white").pack()

        # Slider (read-only visual indicator)
        slider_frame = tk.Frame(dist_card, bg="white")
        slider_frame.pack(fill="x", padx=15, pady=(8, 4))

        tk.Label(slider_frame, text="0mm",   font=("Arial", 7), bg="white", fg="#aaa").pack(side="left")
        tk.Label(slider_frame, text="50mm",  font=("Arial", 7), bg="white", fg="#aaa").pack(side="left", expand=True)
        tk.Label(slider_frame, text="100mm", font=("Arial", 7), bg="white", fg="#aaa").pack(side="right")

        self.dist_slider = ttk.Scale(dist_card, from_=0, to=150,
                                     orient="horizontal", state="disabled")
        self.dist_slider.pack(fill="x", padx=15, pady=(0, 10))

        # Gesture States card
        gesture_card = tk.Frame(right, bg="white", relief="flat", bd=1)
        gesture_card.pack(fill="x")

        g_header = tk.Frame(gesture_card, bg="#424242")
        g_header.pack(fill="x")
        tk.Label(g_header, text="●  Gesture States",
                 font=("Arial", 10, "bold"), fg="white", bg="#424242",
                 pady=6).pack(anchor="w", padx=10)

        gestures = [
            ("Open Hand", "#4CAF50", f"Distance > {self.open_hand_dist}mm"),
            ("Pinch",     "#FF9800", f"{self.pinch_dist_min}mm > Distance < {self.pinch_dist_max}mm"),
            ("Closed",    "#f44336", f"Distance < {self.closed_dist}mm"),
        ]

        self.gesture_rows = {}
        for name, color, desc in gestures:
            row = tk.Frame(gesture_card, bg="white", pady=4)
            row.pack(fill="x", padx=10)

            dot = tk.Label(row, text="●", font=("Arial", 14), fg=color, bg="white")
            dot.pack(side="left")

            info = tk.Frame(row, bg="white")
            info.pack(side="left", padx=6)

            name_lbl = tk.Label(info, text=name,
                                font=("Arial", 10, "bold"), bg="white", fg="#212121")
            name_lbl.pack(anchor="w")

            desc_lbl = tk.Label(info, text=desc,
                                font=("Arial", 8), bg="white", fg="#888")
            desc_lbl.pack(anchor="w")

            self.gesture_rows[name] = row
            ttk.Separator(gesture_card, orient="horizontal").pack(fill="x", padx=10)

        tk.Frame(gesture_card, height=6, bg="white").pack()

    # ──────────────────────────────────────────────
    # Camera
    # ──────────────────────────────────────────────
    def start_camera(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            return
        self.hands_model = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.running = True
        self.paused  = False
        threading.Thread(target=self.update_frame, daemon=True).start()

    def toggle_pause(self):
        if not self.running:
            return
        self.paused = not self.paused

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()

    # ──────────────────────────────────────────────
    # Settings popup
    # ──────────────────────────────────────────────
    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.geometry("340x260")
        win.configure(bg="white")
        win.grab_set()

        tk.Label(win, text="⚙  Settings", font=("Arial", 12, "bold"),
                 bg="white").pack(pady=(12, 4))
        ttk.Separator(win, orient="horizontal").pack(fill="x", padx=15)

        params = [
            ("Open Hand threshold (mm)",  "open_hand_dist"),
            ("Pinch min distance (mm)",   "pinch_dist_min"),
            ("Pinch max distance (mm)",   "pinch_dist_max"),
            ("Closed threshold (mm)",     "closed_dist"),
        ]
        self.setting_vars = {}
        for label, attr in params:
            row = tk.Frame(win, bg="white")
            row.pack(fill="x", padx=20, pady=4)
            tk.Label(row, text=label, font=("Arial", 9),
                     bg="white", width=28, anchor="w").pack(side="left")
            var = tk.StringVar(value=str(getattr(self, attr)))
            entry = tk.Entry(row, textvariable=var, width=6,
                             font=("Arial", 9))
            entry.pack(side="right")
            self.setting_vars[attr] = var

        def apply():
            for attr, var in self.setting_vars.items():
                try:
                    setattr(self, attr, int(var.get()))
                except ValueError:
                    pass
            win.destroy()

        tk.Button(win, text="Apply", bg="#6A0DAD", fg="white",
                  font=("Arial", 10, "bold"), relief="flat",
                  padx=20, pady=6, command=apply).pack(pady=14)

    # ──────────────────────────────────────────────
    # Distance & Gesture
    # ──────────────────────────────────────────────
    def calc_distance_px(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def px_to_mm(self, px_dist, frame_width):
        """Rough conversion using a reference focal length."""
        # Average real finger-tip spread ~80mm when held ~400mm from camera
        REAL_WIDTH_MM = 80
        DISTANCE_MM   = 400
        scale = (self.focal_length * REAL_WIDTH_MM) / (DISTANCE_MM * frame_width)
        return px_dist * scale

    def classify_gesture(self, dist_mm):
        if dist_mm > self.open_hand_dist:
            return "Open Hand"
        elif self.pinch_dist_min <= dist_mm <= self.pinch_dist_max:
            return "Pinch"
        elif dist_mm < self.closed_dist:
            return "Closed"
        else:
            return "Pinch"   # in-between treated as pinch

    # ──────────────────────────────────────────────
    # Frame loop
    # ──────────────────────────────────────────────
    def update_frame(self):
        while self.running:
            if self.paused:
                time.sleep(0.03)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands_model.process(rgb)

            gesture_label = "No Hand"
            dist_mm       = 0

            if result.multi_hand_landmarks:
                for hand_lm in result.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_lm,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(180, 0, 255), thickness=2, circle_radius=4),
                        self.mp_draw.DrawingSpec(color=(180, 0, 255), thickness=2)
                    )

                    lm = hand_lm.landmark
                    thumb_tip = (int(lm[4].x * w),  int(lm[4].y * h))
                    index_tip = (int(lm[8].x * w),  int(lm[8].y * h))

                    px_dist  = self.calc_distance_px(thumb_tip, index_tip)
                    dist_mm  = int(self.px_to_mm(px_dist, w))
                    gesture_label = self.classify_gesture(dist_mm)

                    # Draw line between tips
                    cv2.line(frame, thumb_tip, index_tip, (200, 0, 255), 2)
                    cv2.circle(frame, thumb_tip, 6, (255, 20, 147), -1)
                    cv2.circle(frame, index_tip, 6, (255, 20, 147), -1)

                    # Midpoint label
                    mid = ((thumb_tip[0] + index_tip[0]) // 2,
                           (thumb_tip[1] + index_tip[1]) // 2)
                    cv2.rectangle(frame,
                                  (mid[0] - 28, mid[1] - 16),
                                  (mid[0] + 42, mid[1] + 6),
                                  (30, 30, 30), -1)
                    cv2.putText(frame, f"{dist_mm}mm",
                                (mid[0] - 24, mid[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 1, cv2.LINE_AA)

                # Gesture label top-left
                cv2.rectangle(frame, (8, 8), (160, 36), (80, 0, 120), -1)
                cv2.putText(frame, gesture_label,
                            (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Update UI
            self.dist_var.set(str(dist_mm))
            self.gesture_name = gesture_label

            # Slider (re-enable briefly to set value)
            self.dist_slider.config(state="normal")
            self.dist_slider.set(min(dist_mm, 150))
            self.dist_slider.config(state="disabled")

            # Highlight active gesture row
            self.highlight_gesture(gesture_label)

            # Render frame
            img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img   = img.resize((700, 460))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.video_label.config(image="", bg="black")

    def highlight_gesture(self, active):
        colors = {"Open Hand": "#E8F5E9", "Pinch": "#FFF3E0",
                  "Closed": "#FFEBEE", "No Hand": "white", "Unknown": "white"}
        for name, row in self.gesture_rows.items():
            bg = colors.get(name, "white") if name == active else "white"
            for widget in row.winfo_children():
                try:
                    widget.config(bg=bg)
                except Exception:
                    pass
            row.config(bg=bg)

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = GestureRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()