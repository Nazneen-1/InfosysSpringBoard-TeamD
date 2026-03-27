import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import threading

class HandDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Input and Hand Detection Module")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1100x700")

        self.cap = None
        self.running = False
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = None
        self.fps = 0
        self.prev_time = 0
        self.detection_confidence = tk.DoubleVar(value=0.75)
        self.tracking_confidence = tk.DoubleVar(value=0.80)
        self.max_hands = tk.IntVar(value=2)

        self.build_ui()

    def build_ui(self):
        # ─── Title Bar ───────────────────────────────────────────────
        title_frame = tk.Frame(self.root, bg="#1565C0", pady=12)
        title_frame.pack(fill="x")

        tk.Label(
            title_frame,
            text="Webcam Input and Hand Detection Module",
            font=("Arial", 14, "bold"), fg="white", bg="#1565C0"
        ).pack(anchor="w", padx=15)

        tk.Label(
            title_frame,
            text="Module: Webcam Input and Hand Detection Module • Webcam feed integration using OpenCV • "
                 "MediaPipe Hands implementation for real-time hand detection • Landmark coordinate extraction",
            font=("Arial", 9), fg="#BBDEFB", bg="#1565C0", wraplength=1050, justify="left"
        ).pack(anchor="w", padx=15)

        # ─── Sub-header ───────────────────────────────────────────────
        sub_frame = tk.Frame(self.root, bg="#f5f5f5", pady=8)
        sub_frame.pack(fill="x")

        tk.Label(sub_frame, text="Hand Detection Interface",
                 font=("Arial", 12, "bold"), bg="#f5f5f5").pack(side="left", padx=15)

        btn_frame = tk.Frame(sub_frame, bg="#f5f5f5")
        btn_frame.pack(side="right", padx=10)

        tk.Button(btn_frame, text="▶  Start Camera", bg="#1565C0", fg="white",
                  font=("Arial", 9, "bold"), relief="flat", padx=10, pady=4,
                  command=self.start_camera).pack(side="left", padx=4)

        tk.Button(btn_frame, text="■  Stop Camera", bg="#424242", fg="white",
                  font=("Arial", 9, "bold"), relief="flat", padx=10, pady=4,
                  command=self.stop_camera).pack(side="left", padx=4)

        tk.Button(btn_frame, text="📷  Capture", bg="#424242", fg="white",
                  font=("Arial", 9, "bold"), relief="flat", padx=10, pady=4,
                  command=self.capture_frame).pack(side="left", padx=4)

        # ─── Main Content ─────────────────────────────────────────────
        content_frame = tk.Frame(self.root, bg="#e0e0e0")
        content_frame.pack(fill="both", expand=True, padx=10, pady=8)

        # Camera Feed
        self.video_label = tk.Label(content_frame, bg="black", width=780, height=480)
        self.video_label.pack(side="left", padx=(0, 8))

        # Right Panel
        right_panel = tk.Frame(content_frame, bg="#e0e0e0", width=280)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)

        # ── Detection Status ──
        status_card = tk.Frame(right_panel, bg="white", relief="flat", bd=1)
        status_card.pack(fill="x", pady=(0, 8))

        tk.Label(status_card, text="ℹ  Detection Status",
                 font=("Arial", 10, "bold"), bg="white").pack(anchor="w", padx=10, pady=(8, 4))
        ttk.Separator(status_card, orient="horizontal").pack(fill="x", padx=10)

        self.status_vars = {
            "Camera Status":  tk.StringVar(value="Inactive"),
            "Hands Detected": tk.StringVar(value="0"),
            "Detection FPS":  tk.StringVar(value="0"),
            "Model Status":   tk.StringVar(value="Not Loaded"),
        }
        self.status_colors = {
            "Camera Status":  {"Active": "#4CAF50", "Inactive": "#f44336"},
            "Model Status":   {"Loaded": "#4CAF50", "Not Loaded": "#f44336"},
        }

        for label, var in self.status_vars.items():
            row = tk.Frame(status_card, bg="white")
            row.pack(fill="x", padx=10, pady=3)
            tk.Label(row, text=label + ":", font=("Arial", 9),
                     bg="white", fg="#555").pack(side="left")
            color = "#4CAF50" if label == "Model Status" else "#333"
            lbl = tk.Label(row, textvariable=var, font=("Arial", 9, "bold"),
                           bg="white", fg=color)
            lbl.pack(side="right")
            if label in self.status_colors:
                self.status_labels = getattr(self, 'status_labels', {})
                self.status_labels[label] = lbl

        tk.Frame(status_card, height=8, bg="white").pack()

        # ── Detection Parameters ──
        params_card = tk.Frame(right_panel, bg="white", relief="flat", bd=1)
        params_card.pack(fill="x")

        tk.Label(params_card, text="≡  Detection Parameters",
                 font=("Arial", 10, "bold"), bg="white").pack(anchor="w", padx=10, pady=(8, 4))
        ttk.Separator(params_card, orient="horizontal").pack(fill="x", padx=10)

        self._add_slider(params_card, "Detection Confidence", self.detection_confidence, 0, 1, self.update_model)
        self._add_slider(params_card, "Tracking Confidence",  self.tracking_confidence,  0, 1, self.update_model)
        self._add_slider(params_card, "Max Number of Hands",  self.max_hands,             1, 4, self.update_model, integer=True)

        tk.Frame(params_card, height=8, bg="white").pack()

    def _add_slider(self, parent, label, variable, from_, to, command, integer=False):
        frame = tk.Frame(parent, bg="white")
        frame.pack(fill="x", padx=10, pady=6)

        tk.Label(frame, text=label, font=("Arial", 9), bg="white").pack(anchor="w")

        tick_frame = tk.Frame(frame, bg="white")
        tick_frame.pack(fill="x")

        tk.Label(tick_frame, text=str(from_), font=("Arial", 7), bg="white", fg="#888").pack(side="left")
        mid = int((from_ + to) / 2) if integer else f"{(from_ + to)/2:.0%}"
        tk.Label(tick_frame, text=str(mid), font=("Arial", 7), bg="white", fg="#888").pack(side="left", expand=True)
        tk.Label(tick_frame, text=str(to) if not integer else str(to),
                 font=("Arial", 7), bg="white", fg="#888").pack(side="right")

        resolution = 1 if integer else 0.05
        slider = ttk.Scale(frame, from_=from_, to=to, variable=variable,
                           orient="horizontal", command=lambda e: command())
        slider.pack(fill="x")

    def update_model(self):
        if self.running:
            self.hands = self.mp_hands.Hands(
                max_num_hands=self.max_hands.get(),
                min_detection_confidence=round(self.detection_confidence.get(), 2),
                min_tracking_confidence=round(self.tracking_confidence.get(), 2)
            )

    def start_camera(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_vars["Camera Status"].set("Error")
            return
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_hands.get(),
            min_detection_confidence=round(self.detection_confidence.get(), 2),
            min_tracking_confidence=round(self.tracking_confidence.get(), 2)
        )
        self.running = True
        self.status_vars["Camera Status"].set("Active")
        self.status_vars["Model Status"].set("Loaded")
        if hasattr(self, 'status_labels'):
            self.status_labels.get("Camera Status") and self.status_labels["Camera Status"].config(fg="#4CAF50")
            self.status_labels.get("Model Status")  and self.status_labels["Model Status"].config(fg="#4CAF50")

        thread = threading.Thread(target=self.update_frame, daemon=True)
        thread.start()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.status_vars["Camera Status"].set("Inactive")
        self.status_vars["Hands Detected"].set("0")
        self.status_vars["Detection FPS"].set("0")
        self.status_vars["Model Status"].set("Not Loaded")
        self.video_label.config(image="", bg="black")
        if hasattr(self, 'status_labels'):
            self.status_labels.get("Camera Status") and self.status_labels["Camera Status"].config(fg="#f44336")
            self.status_labels.get("Model Status")  and self.status_labels["Model Status"].config(fg="#f44336")

    def capture_frame(self):
        if self.cap and self.running:
            ret, frame = self.cap.read()
            if ret:
                filename = f"capture_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"[Captured] Saved as {filename}")

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            num_hands = 0
            if result.multi_hand_landmarks:
                num_hands = len(result.multi_hand_landmarks)
                for hand_landmarks in result.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(255, 20, 147), thickness=2, circle_radius=4),
                        self.mp_draw.DrawingSpec(color=(255, 20, 147), thickness=2)
                    )

            # FPS
            curr_time = time.time()
            self.fps = int(1 / (curr_time - self.prev_time + 1e-5))
            self.prev_time = curr_time

            self.status_vars["Hands Detected"].set(str(num_hands))
            self.status_vars["Detection FPS"].set(str(min(self.fps, 60)))

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((780, 460))
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.video_label.config(image="", bg="black")

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = HandDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()