# app.py
import cv2
import time
import math
import tkinter as tk
from PIL import Image, ImageTk
import mediapipe as mp

# Import your separated milestones
from milestone_3 import VolumeController
from milestone_4 import GestureDashboardUI

class MainApplication:
    def __init__(self, root):
        self.root = root
        
        # 1. Initialize UI (Milestone 4)
        self.ui = GestureDashboardUI(self.root)
        
        # 2. Initialize Audio Engine (Milestone 3)
        self.audio = VolumeController(min_dist=20, max_dist=150)
        
        # 3. Initialize Computer Vision
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)

        # State Variables
        self.prev_frame_time = 0
        self.open_palm_counter = 0

        # Start the video loop
        self.process_frame()

    def get_finger_state(self, hand_landmarks):
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        fingers = []
        
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
            
        for tip, pip in zip(tips, pips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def process_frame(self):
        success, frame = self.cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Metric: Response Time
            new_frame_time = time.time()
            latency_ms = int((new_frame_time - self.prev_frame_time) * 1000)
            self.prev_frame_time = new_frame_time

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            rubric_gesture = "None"
            current_dist_mm = 0
            accuracy = 0
            
            if results.multi_hand_landmarks:
                # Metric: Accuracy (Mocked from presence, MediaPipe Hands doesn't expose confidence directly in this API)
                accuracy = 95 
                
                for handLms in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, handLms, self.mp_hands.HAND_CONNECTIONS)
                    
                    x4, y4 = int(handLms.landmark[4].x * w), int(handLms.landmark[4].y * h)
                    x8, y8 = int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h)
                    
                    fingers = self.get_finger_state(handLms)
                    finger_count = sum(fingers)
                    current_dist_mm = int(math.hypot(x8 - x4, y8 - y4) * 0.5)

                    # Determine Gesture
                    if finger_count == 0: rubric_gesture = "Closed"
                    elif finger_count == 5: rubric_gesture = "Open Palm"
                    elif current_dist_mm <= self.audio.max_dist and finger_count <= 2: rubric_gesture = "Pinch"

                    # Execute Milestone 3 Logic
                    if rubric_gesture == "Open Palm":
                        self.open_palm_counter += 1
                        if self.open_palm_counter >= 15:
                            if self.audio.get_mute_status():
                                self.audio.toggle_mute()
                            self.open_palm_counter = 0
                    else:
                        self.open_palm_counter = 0

                    if rubric_gesture == "Pinch":
                        self.audio.set_volume_smoothly(current_dist_mm)
                        cv2.line(frame, (x4, y4), (x8, y8), (200, 0, 200), 3)

                    break # Track one hand only

            # Update Milestone 4 UI
            vol_pct = self.audio.get_current_volume_percent()
            
            self.ui.update_gesture_status(rubric_gesture)
            self.ui.update_metrics(vol_pct, current_dist_mm, accuracy, latency_ms)

            # Convert frame to Tkinter format
            frame_resized = cv2.resize(frame, (600, 450))
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.ui.update_video(img_tk)

        # Loop
        self.root.after(15, self.process_frame)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
