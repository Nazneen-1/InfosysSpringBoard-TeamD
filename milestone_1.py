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
        self.root.title("Milestone 1: Webcam Input and Hand Detection Module")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1100x700")

        self.cap        = None
        self.running    = False
        self.paused     = False
        self.mp_hands   = mp.solutions.hands
        self.mp_draw    = mp.solutions.drawing_utils
        self.hands      = None

        # Thread-safe shared state
        self.latest_frame  = None
        self.num_hands     = 0
        self.current_fps   = 0
        self.prev_time     = 0

        self.detection_confidence = tk.DoubleVar(value=0.75)
        self.tracking_confidence  = tk.DoubleVar(value=0.80)
        self.max_hands_var        = tk.IntVar(value=2)

        self.build_ui()
        self.poll_ui()      # start main-thread UI loop

    # ══════════════════════════════════════════════
    # BUILD UI
    # ══════════════════════════════════════════════
    def build_ui(self):
        # ── Title bar ─────────────────────────────
        title_frame = tk.Frame(self.root, bg="#1565C0", pady=12)
        title_frame.pack(fill="x")

        tk.Label(title_frame,
                 text="Webcam Input and Hand Detection Module ",
                 font=("Arial", 14, "bold"),
                 fg="white", bg="#1565C0").pack(anchor="w", padx=15)

        tk.Label(title_frame,
                 text="Module: Webcam Input and Hand Detection Module  •  Webcam feed integration "
                      "using OpenCV  •  MediaPipe Hands implementation for real-time hand detection  "
                      "•  Landmark coordinate extraction",
                 font=("Arial", 9), fg="#BBDEFB", bg="#1565C0",
                 wraplength=1050, justify="left").pack(anchor="w", padx=15)

        # ── Sub-header ────────────────────────────
        sub_frame = tk.Frame(self.root, bg="#f5f5f5", pady=8)
        sub_frame.pack(fill="x")

        tk.Label(sub_frame, text="Hand Detection Interface",
                 font=("Arial", 12, "bold"),
                 bg="#f5f5f5").pack(side="left", padx=15)

        btn_frame = tk.Frame(sub_frame, bg="#f5f5f5")
        btn_frame.pack(side="right", padx=10)

        tk.Button(btn_frame, text="▶  Start Camera",
                  bg="#1565C0", fg="white",
                  font=("Arial", 9, "bold"), relief="flat",
                  padx=10, pady=4,
                  command=self.start_camera).pack(side="left", padx=4)

        tk.Button(btn_frame, text="⏸  Pause",
                  bg="#FF9800", fg="white",
                  font=("Arial", 9, "bold"), relief="flat",
                  padx=10, pady=4,
                  command=self.toggle_pause).pack(side="left", padx=4)

        tk.Button(btn_frame, text="■  Stop Camera",
                  bg="#424242", fg="white",
                  font=("Arial", 9, "bold"), relief="flat",
                  padx=10, pady=4,
                  command=self.stop_camera).pack(side="left", padx=4)

        tk.Button(btn_frame, text="📷  Capture",
                  bg="#424242", fg="white",
                  font=("Arial", 9, "bold"), relief="flat",
                  padx=10, pady=4,
                  command=self.capture_frame).pack(side="left", padx=4)

        # ── Main content ──────────────────────────
        content_frame = tk.Frame(self.root, bg="#e0e0e0")
        content_frame.pack(fill="both", expand=True, padx=10, pady=8)

        # ── Camera feed (large, fixed) ────────────
        cam_wrap = tk.Frame(content_frame, bg="white", relief="flat", bd=1)
        cam_wrap.pack(side="left", fill="both", expand=True, padx=(0, 8))

        cam_hdr = tk.Frame(cam_wrap, bg="white")
        cam_hdr.pack(fill="x", padx=10, pady=(8, 4))

        tk.Label(cam_hdr, text="📷  Live Hand Detection Feed",
                 font=("Arial", 10, "bold"),
                 bg="white").pack(side="left")

        self.detect_badge = tk.Label(cam_hdr, text="● No Hand",
                                     font=("Arial", 9, "bold"),
                                     fg="#888", bg="white")
        self.detect_badge.pack(side="right")

        # Fixed-size video label — prevents layout shifts
        self.video_label = tk.Label(cam_wrap, bg="#111111",
                                    width=760, height=470)
        self.video_label.pack(fill="both", expand=True,
                              padx=8, pady=(0, 8))

        # ── Right panel ───────────────────────────
        right_panel = tk.Frame(content_frame, bg="#e0e0e0", width=280)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)

        # Detection Status card
        status_card = tk.Frame(right_panel, bg="white",
                               relief="flat", bd=1)
        status_card.pack(fill="x", pady=(0, 8))

        tk.Label(status_card, text="ℹ  Detection Status",
                 font=("Arial", 10, "bold"),
                 bg="white").pack(anchor="w", padx=10, pady=(8, 4))
        ttk.Separator(status_card, orient="horizontal").pack(
            fill="x", padx=10)

        self.status_vars = {
            "Camera Status":  tk.StringVar(value="Inactive"),
            "Hands Detected": tk.StringVar(value="0"),
            "Detection FPS":  tk.StringVar(value="0"),
            "Model Status":   tk.StringVar(value="Not Loaded"),
        }
        self.status_labels = {}

        for label, var in self.status_vars.items():
            row = tk.Frame(status_card, bg="white")
            row.pack(fill="x", padx=10, pady=3)
            tk.Label(row, text=label + ":",
                     font=("Arial", 9),
                     bg="white", fg="#555").pack(side="left")
            lbl = tk.Label(row, textvariable=var,
                           font=("Arial", 9, "bold"),
                           bg="white", fg="#f44336")
            lbl.pack(side="right")
            self.status_labels[label] = lbl

        tk.Frame(status_card, height=8, bg="white").pack()

        # Detection Parameters card
        params_card = tk.Frame(right_panel, bg="white",
                               relief="flat", bd=1)
        params_card.pack(fill="x")

        tk.Label(params_card, text="≡  Detection Parameters",
                 font=("Arial", 10, "bold"),
                 bg="white").pack(anchor="w", padx=10, pady=(8, 4))
        ttk.Separator(params_card, orient="horizontal").pack(
            fill="x", padx=10)

        self._add_slider(params_card, "Detection Confidence",
                         self.detection_confidence, 0, 1,
                         self.update_model)
        self._add_slider(params_card, "Tracking Confidence",
                         self.tracking_confidence,  0, 1,
                         self.update_model)
        self._add_slider(params_card, "Max Number of Hands",
                         self.max_hands_var, 1, 4,
                         self.update_model, integer=True)

        tk.Frame(params_card, height=8, bg="white").pack()

        # Capture log card
        log_card = tk.Frame(right_panel, bg="white",
                            relief="flat", bd=1)
        log_card.pack(fill="x", pady=(8, 0))

        tk.Label(log_card, text="📁  Capture Log",
                 font=("Arial", 10, "bold"),
                 bg="white").pack(anchor="w", padx=10, pady=(8, 4))
        ttk.Separator(log_card, orient="horizontal").pack(
            fill="x", padx=10)

        self.log_var = tk.StringVar(value="No captures yet.")
        tk.Label(log_card, textvariable=self.log_var,
                 font=("Arial", 8), fg="#555",
                 bg="white", wraplength=240,
                 justify="left").pack(anchor="w", padx=10, pady=6)

    # ══════════════════════════════════════════════
    # SLIDER HELPER
    # ══════════════════════════════════════════════
    def _add_slider(self, parent, label, variable,
                    from_, to, command, integer=False):
        frame = tk.Frame(parent, bg="white")
        frame.pack(fill="x", padx=10, pady=6)

        tk.Label(frame, text=label,
                 font=("Arial", 9), bg="white").pack(anchor="w")

        tick_frame = tk.Frame(frame, bg="white")
        tick_frame.pack(fill="x")

        tk.Label(tick_frame, text=str(from_),
                 font=("Arial", 7), bg="white",
                 fg="#888").pack(side="left")
        mid = int((from_ + to) / 2) if integer \
              else f"{int((from_ + to) / 2 * 100)}%"
        tk.Label(tick_frame, text=str(mid),
                 font=("Arial", 7), bg="white",
                 fg="#888").pack(side="left", expand=True)
        tk.Label(tick_frame, text=str(to),
                 font=("Arial", 7), bg="white",
                 fg="#888").pack(side="right")

        ttk.Scale(frame, from_=from_, to=to,
                  variable=variable, orient="horizontal",
                  command=lambda e: command()).pack(fill="x")

    # ══════════════════════════════════════════════
    # MODEL UPDATE
    # ══════════════════════════════════════════════
    def update_model(self):
        if self.running and self.hands:
            self.hands = self.mp_hands.Hands(
                max_num_hands=self.max_hands_var.get(),
                min_detection_confidence=round(
                    self.detection_confidence.get(), 2),
                min_tracking_confidence=round(
                    self.tracking_confidence.get(), 2)
            )

    # ══════════════════════════════════════════════
    # CAMERA CONTROLS
    # ══════════════════════════════════════════════
    def start_camera(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_vars["Camera Status"].set("Error")
            return
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_hands_var.get(),
            min_detection_confidence=round(
                self.detection_confidence.get(), 2),
            min_tracking_confidence=round(
                self.tracking_confidence.get(), 2)
        )
        self.running = True
        self.paused  = False
        threading.Thread(target=self.update_frame, daemon=True).start()

    def toggle_pause(self):
        if self.running:
            self.paused = not self.paused

    def stop_camera(self):
        self.running = False
        self.paused  = False
        if self.cap:
            self.cap.release()
        self.latest_frame = None
        self.num_hands    = 0
        self.current_fps  = 0

    def capture_frame(self):
        if self.cap and self.running and not self.paused:
            ret, frame = self.cap.read()
            if ret:
                filename = f"capture_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                self.log_var.set(f"Saved: {filename}")
                print(f"[Captured] {filename}")

    # ══════════════════════════════════════════════
    # POLL UI — main thread every 30ms (NO jitter)
    # ══════════════════════════════════════════════
    def poll_ui(self):
        # Update video label
        if self.latest_frame is not None:
            self.video_label.imgtk = self.latest_frame
            self.video_label.config(image=self.latest_frame)
        elif not self.running:
            self.video_label.config(image="", bg="#111111")

        # Camera / model status
        if self.running:
            self.status_vars["Camera Status"].set("Active")
            self.status_vars["Model Status"].set("Loaded")
            self.status_labels["Camera Status"].config(fg="#4CAF50")
            self.status_labels["Model Status"].config(fg="#4CAF50")
        else:
            self.status_vars["Camera Status"].set("Inactive")
            self.status_vars["Model Status"].set("Not Loaded")
            self.status_labels["Camera Status"].config(fg="#f44336")
            self.status_labels["Model Status"].config(fg="#f44336")

        # Hands detected & FPS
        self.status_vars["Hands Detected"].set(str(self.num_hands))
        self.status_vars["Detection FPS"].set(str(self.current_fps))

        # Detection badge
        if self.num_hands > 0:
            self.detect_badge.config(
                text=f"● {self.num_hands} Hand(s) Detected",
                fg="#4CAF50")
        else:
            self.detect_badge.config(
                text="● No Hand", fg="#888")

        self.root.after(30, self.poll_ui)

    # ══════════════════════════════════════════════
    # FRAME LOOP — background thread, NEVER touches UI
    # ══════════════════════════════════════════════
    def update_frame(self):
        while self.running:
            if self.paused:
                time.sleep(0.03)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            num_hands = 0
            if result.multi_hand_landmarks:
                num_hands = len(result.multi_hand_landmarks)
                for hand_landmarks in result.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(
                            color=(255, 20, 147),
                            thickness=2, circle_radius=4),
                        self.mp_draw.DrawingSpec(
                            color=(255, 20, 147),
                            thickness=2)
                    )

                    # Draw landmark index numbers
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        cx = int(lm.x * w)
                        cy = int(lm.y * h)
                        if idx in [4, 8, 12, 16, 20]:  # fingertips only
                            cv2.circle(frame, (cx, cy), 8,
                                       (0, 255, 200), -1)

            # FPS calculation
            curr_time      = time.time()
            fps            = int(1 / (curr_time - self.prev_time + 1e-5))
            self.prev_time = curr_time

            # Draw FPS on frame
            cv2.rectangle(frame, (w-110, 8), (w-8, 38),
                          (20, 20, 20), -1)
            cv2.putText(frame, f"FPS: {min(fps, 60)}",
                        (w-104, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 180), 2)

            # Draw hands count on frame
            cv2.rectangle(frame, (8, 8), (180, 40),
                          (21, 101, 192), -1)
            cv2.putText(frame,
                        f"Hands: {num_hands}",
                        (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2)

            # Store state for poll_ui (thread-safe primitives)
            self.num_hands   = num_hands
            self.current_fps = min(fps, 60)

            # Prepare image — resize to fit label
            lw = self.video_label.winfo_width()
            lh = self.video_label.winfo_height()
            tw = lw if lw > 10 else 760
            th = lh if lh > 10 else 470

            img   = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img   = img.resize((tw, th))
            # Store — poll_ui picks it up
            self.latest_frame = ImageTk.PhotoImage(image=img)

        # Cleanup after loop ends
        self.latest_frame = None
        self.num_hands    = 0
        self.current_fps  = 0

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = HandDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()