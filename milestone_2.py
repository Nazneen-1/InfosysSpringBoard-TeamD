import cv2
import mediapipe as mp
import math
import time
import threading
import queue
from PIL import Image


class GestureDistanceBackend:
   
    def __init__(self):
        self._mp_hands   = mp.solutions.hands
        self._mp_draw    = mp.solutions.drawing_utils
        self._hands      = None
        self._cap        = None
        self._thread     = None

        # ── Gesture thresholds (mm) ──
        self.open_hand_dist  = 50
        self.pinch_dist_min  = 10
        self.pinch_dist_max  = 50
        self.closed_dist     = 10
        self.focal_length    = 800

        # ── Public state (frontend reads these) ──
        self.latest_frame  = None   # PIL.Image or None
        self.distance_mm   = 0
        self.gesture_name  = "No Hand"
        self.is_running    = False
        self.is_paused     = False

        # Internal frame queue (maxsize=1 → always latest frame)
        self._frame_queue  = queue.Queue(maxsize=1)

    # ──────────────────────────────────────────────
    # Configuration
    # ──────────────────────────────────────────────
    def configure(self,
                  open_hand_dist: int  = None,
                  pinch_dist_min: int  = None,
                  pinch_dist_max: int  = None,
                  closed_dist:    int  = None,
                  focal_length:   int  = None):
        if open_hand_dist is not None: self.open_hand_dist = open_hand_dist
        if pinch_dist_min is not None: self.pinch_dist_min = pinch_dist_min
        if pinch_dist_max is not None: self.pinch_dist_max = pinch_dist_max
        if closed_dist    is not None: self.closed_dist    = closed_dist
        if focal_length   is not None: self.focal_length   = focal_length

    # ──────────────────────────────────────────────
    # Camera controls
    # ──────────────────────────────────────────────
    def start(self) -> bool:
        if self.is_running:
            return True

        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS,          30)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        self._hands = self._mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
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
        self.is_running  = False
        self.is_paused   = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self.latest_frame = None
        self.distance_mm  = 0
        self.gesture_name = "No Hand"

    # ──────────────────────────────────────────────
    # Distance & gesture helpers
    # ──────────────────────────────────────────────
    def _calc_distance_px(self, p1, p2) -> float:
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def _px_to_mm(self, px_dist: float, frame_width: int) -> int:
        REAL_WIDTH_MM = 80
        DISTANCE_MM   = 400
        scale = (self.focal_length * REAL_WIDTH_MM) / (DISTANCE_MM * frame_width)
        return int(px_dist * scale)

    def _classify_gesture(self, dist_mm: int) -> str:
        if dist_mm > self.open_hand_dist:
            return "Open Hand"
        if dist_mm < self.closed_dist:
            return "Closed"
        return "Pinch"   # everything in between is Pinch

    # ──────────────────────────────────────────────
    # Frame annotation
    # ──────────────────────────────────────────────
    def _annotate_frame(self, frame, result, dist_mm: int,
                        gesture_label: str):
        h, w = frame.shape[:2]

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                # Skeleton
                self._mp_draw.draw_landmarks(
                    frame, hand_lm,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_draw.DrawingSpec(
                        color=(180, 0, 255),
                        thickness=2, circle_radius=4),
                    self._mp_draw.DrawingSpec(
                        color=(180, 0, 255), thickness=2),
                )

                lm        = hand_lm.landmark
                thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))
                index_tip = (int(lm[8].x * w), int(lm[8].y * h))

                # Thumb-index line + endpoints
                cv2.line(frame, thumb_tip, index_tip, (200, 0, 255), 2)
                cv2.circle(frame, thumb_tip, 6, (255, 20, 147), -1)
                cv2.circle(frame, index_tip, 6, (255, 20, 147), -1)

                # Distance label at midpoint
                mid = ((thumb_tip[0] + index_tip[0]) // 2,
                       (thumb_tip[1] + index_tip[1]) // 2)
                cv2.rectangle(frame,
                              (mid[0]-28, mid[1]-16),
                              (mid[0]+42, mid[1]+6),
                              (30, 30, 30), -1)
                cv2.putText(frame, f"{dist_mm}mm",
                            (mid[0]-24, mid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1, cv2.LINE_AA)

            # Gesture badge — top left
            cv2.rectangle(frame, (8, 8), (160, 36), (80, 0, 120), -1)
            cv2.putText(frame, gesture_label,
                        (14, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    # ──────────────────────────────────────────────
    # Background frame loop  (NO UI calls inside)
    # ──────────────────────────────────────────────
    def _frame_loop(self):
        TARGET_FPS  = 30
        FRAME_DELAY = 1.0 / TARGET_FPS

        while self.is_running:
            loop_start = time.time()

            if self.is_paused:
                time.sleep(FRAME_DELAY)
                continue

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame  = cv2.flip(frame, 1)
            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._hands.process(rgb)

            dist_mm       = 0
            gesture_label = "No Hand"

            if result.multi_hand_landmarks:
                hand_lm   = result.multi_hand_landmarks[0]
                lm        = hand_lm.landmark
                thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))
                index_tip = (int(lm[8].x * w), int(lm[8].y * h))

                px_dist       = self._calc_distance_px(thumb_tip, index_tip)
                dist_mm       = self._px_to_mm(px_dist, w)
                gesture_label = self._classify_gesture(dist_mm)

            frame = self._annotate_frame(frame, result,
                                         dist_mm, gesture_label)

            # Update public state
            self.distance_mm  = dist_mm
            self.gesture_name = gesture_label

            # Convert to PIL Image
            self.latest_frame = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Pace to TARGET_FPS
            elapsed = time.time() - loop_start
            sleep_t = FRAME_DELAY - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        # Cleanup
        self.latest_frame = None
        self.distance_mm  = 0
        self.gesture_name = "No Hand"
