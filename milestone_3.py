# milestone 3
import numpy as np
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Disable PyAutoGUI delay for lightning-fast volume steps
pyautogui.PAUSE = 0 

class VolumeController:
    def __init__(self, min_dist=20, max_dist=150):
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.smoothed_dist = 0
        
        # Initialize Windows Audio Control
        devices = AudioUtilities.GetSpeakers()
        try:
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
        except Exception as e:
            print(f"Audio init error: {e}")
            self.volume_ctrl = None

    def get_current_volume_percent(self):
        """Returns volume as a 0-100 integer."""
        if self.volume_ctrl:
            return round(self.volume_ctrl.GetMasterVolumeLevelScalar() * 100)
        return 0

    def get_mute_status(self):
        """Returns True if muted, False otherwise."""
        if self.volume_ctrl:
            return self.volume_ctrl.GetMute()
        return False

    def toggle_mute(self):
        """Uses PyAutoGUI to toggle the system mute."""
        pyautogui.press('volumemute')

    def step_volume(self, direction, steps=5):
        """Increments or decrements volume by fixed steps."""
        if direction == "up":
            pyautogui.press('volumeup', presses=steps)
        elif direction == "down":
            pyautogui.press('volumedown', presses=steps)

    def set_volume_smoothly(self, current_distance_mm):
        """Applies smoothing and maps distance to system volume."""
        # 1. Exponential Smoothing
        if self.smoothed_dist == 0:
            self.smoothed_dist = current_distance_mm
        else:
            self.smoothed_dist = (0.35 * current_distance_mm) + (0.65 * self.smoothed_dist)
            
        # 2. Map smoothed distance to 0-100%
        target_vol = int(np.interp(self.smoothed_dist, [self.min_dist, self.max_dist], [0, 100]))
        current_vol = self.get_current_volume_percent()
        
        diff = target_vol - current_vol
        
        # 3. Apply changes via PyAutoGUI (only if difference is significant)
        if abs(diff) >= 2:
            steps = abs(diff) // 2  
            if diff > 0: 
                pyautogui.press('volumeup', presses=steps)
            else: 
                pyautogui.press('volumedown', presses=steps)
                
        return target_vol
