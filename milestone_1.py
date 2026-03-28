import cv2
import mediapipe as mp
import time
import threading
from PIL import Image


class HandDetectionBackend:
   
    def __init__(self):
        # MediaPipe
        self._mp_hands  = mp.solutions.hands
        self._mp_draw   = mp.solutions.drawing_utils
        self._hands     = None

        # Camera
        self._cap       = None
        self._thread    = None

        # Detection parameters (updated via configure())
        self._det_conf  = 0.75
        self._trk_conf  = 0.80
        self._max_hands = 2

        # ── Public state (frontend reads these) ──
        self.latest_frame  = None   # PIL.Image or None
        self.num_hands     = 0
        self.current_fps   = 0
        self.is_running    = False
        self.is_paused     = False

        # Internal
        self._prev_time    = 0
        self._last_log     = ""     # capture log message

    # ──────────────────────────────────────────────
    # Configuration
    # ──────────────────────────────────────────────
    def configure(self, det_conf: float = None,
                  trk_conf: float = None,
                  max_hands: int  = None):
        
        if det_conf  is not None: self._det_conf  = round(det_conf,  2)
        if trk_conf  is not None: self._trk_conf  = round(trk_conf,  2)
        if max_hands is not None: self._max_hands = max_hands

        # Rebuild model if already running
        if self.is_running and self._hands:
            self._build_model()

    # ──────────────────────────────────────────────
    # Camera controls
    # ──────────────────────────────────────────────
    def start(self) -> bool:
    
        if self.is_running:
            return True

        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            return False

        self._build_model()
        self.is_running = True
        self.is_paused  = False
        self._thread    = threading.Thread(
            target=self._frame_loop, daemon=True)
        self._thread.start()
        return True

    def pause(self):
        
        if self.is_running:
            self.is_paused = not self.is_paused

    def stop(self):
        """Stop camera and clean up."""
        self.is_running   = False
        self.is_paused    = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self.latest_frame = None
        self.num_hands    = 0
        self.current_fps  = 0

    def capture_snapshot(self) -> str:
       
        if not (self.is_running and self._cap and not self.is_paused):
            return ""
        ret, frame = self._cap.read()
        if not ret:
            return ""
        filename = f"capture_{int(time.time())}.png"
        cv2.imwrite(filename, frame)
        self._last_log = f"Saved: {filename}"
        print(f"[Captured] {filename}")
        return filename

    def get_last_log(self) -> str:
        return self._last_log

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────
    def _build_model(self):
       
        self._hands = self._mp_hands.Hands(
            max_num_hands=self._max_hands,
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._trk_conf,
        )

    def _annotate_frame(self, frame, result, fps: int):
        
        h, w = frame.shape[:2]

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Skeleton
                self._mp_draw.draw_landmarks(
                    frame, hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_draw.DrawingSpec(
                        color=(255, 20, 147),
                        thickness=2, circle_radius=4),
                    self._mp_draw.DrawingSpec(
                        color=(255, 20, 147),
                        thickness=2),
                )
                # Fingertip highlights
                for idx, lm in enumerate(hand_landmarks.landmark):
                    if idx in [4, 8, 12, 16, 20]:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 8, (0, 255, 200), -1)

        # FPS badge — top right
        cv2.rectangle(frame, (w - 110, 8), (w - 8, 38), (20, 20, 20), -1)
        cv2.putText(frame, f"FPS: {fps}",
                    (w - 104, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 180), 2)

        # Hands badge — top left
        num = len(result.multi_hand_landmarks) \
              if result.multi_hand_landmarks else 0
        cv2.rectangle(frame, (8, 8), (180, 40), (21, 101, 192), -1)
        cv2.putText(frame, f"Hands: {num}",
                    (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2)

        return frame

    # ──────────────────────────────────────────────
    # Background frame loop  (NO UI calls inside)
    # ──────────────────────────────────────────────
    def _frame_loop(self):
        while self.is_running:
            if self.is_paused:
                time.sleep(0.03)
                continue

            ret, frame = self._cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._hands.process(rgb)

            # FPS
            curr_time       = time.time()
            fps             = int(1 / (curr_time - self._prev_time + 1e-5))
            self._prev_time = curr_time
            fps             = min(fps, 60)

            # Annotate
            frame = self._annotate_frame(frame, result, fps)

            # Update public state
            self.num_hands   = len(result.multi_hand_landmarks) \
                               if result.multi_hand_landmarks else 0
            self.current_fps = fps

            # Convert to PIL Image (frontend will resize + display)
            self.latest_frame = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Cleanup
        self.latest_frame = None
        self.num_hands    = 0
        self.current_fps  = 0
