import tkinter as tk
from tkinter import ttk
from PIL import ImageTk

from milestone_1 import HandDetectionBackend
from milestone_2 import GestureDistanceBackend


# ══════════════════════════════════════════════════════════
#  MILESTONE 1 TAB  —  Hand Detection
# ══════════════════════════════════════════════════════════
class Milestone1Tab:
    

    def __init__(self, parent: tk.Frame):
        self.backend        = HandDetectionBackend()
        self._current_photo = None
        self._build_ui(parent)
        self._poll_ui()

    # ── UI ────────────────────────────────────────────────
    def _build_ui(self, parent):
        # Title bar
        title = tk.Frame(parent, bg="#1565C0", pady=10)
        title.pack(fill="x")
        tk.Label(title,
                 text=" Webcam Input and Hand Detection Module ",
                 font=("Arial", 13, "bold"), fg="white", bg="#1565C0"
                 ).pack(anchor="w", padx=15)
        tk.Label(title,
                 text="Webcam feed integration using OpenCV  •  MediaPipe Hands  •  Landmark extraction",
                 font=("Arial", 9), fg="#BBDEFB", bg="#1565C0"
                 ).pack(anchor="w", padx=15)

        # Sub-header
        sub = tk.Frame(parent, bg="#f5f5f5", pady=7)
        sub.pack(fill="x")
        tk.Label(sub, text="Hand Detection Interface",
                 font=("Arial", 11, "bold"), bg="#f5f5f5"
                 ).pack(side="left", padx=15)

        btn_frame = tk.Frame(sub, bg="#f5f5f5")
        btn_frame.pack(side="right", padx=10)
        for label, color, cmd in [
            ("▶  Start Camera", "#1565C0", self._on_start),
            ("⏸  Pause",        "#FF9800", self._on_pause),
            ("■  Stop",         "#424242", self._on_stop),
            ("📷  Capture",     "#424242", self._on_capture),
        ]:
            tk.Button(btn_frame, text=label, bg=color, fg="white",
                      font=("Arial", 9, "bold"), relief="flat",
                      padx=10, pady=4, command=cmd
                      ).pack(side="left", padx=3)

        # Content
        content = tk.Frame(parent, bg="#e0e0e0")
        content.pack(fill="both", expand=True, padx=10, pady=8)

        # Video
        cam_wrap = tk.Frame(content, bg="white", relief="flat", bd=1)
        cam_wrap.pack(side="left", fill="both", expand=True, padx=(0, 8))

        cam_hdr = tk.Frame(cam_wrap, bg="white")
        cam_hdr.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(cam_hdr, text="📷  Live Hand Detection Feed",
                 font=("Arial", 10, "bold"), bg="white").pack(side="left")
        self._detect_badge = tk.Label(cam_hdr, text="● No Hand",
                                      font=("Arial", 9, "bold"),
                                      fg="#888", bg="white")
        self._detect_badge.pack(side="right")

        self._video_label = tk.Label(cam_wrap, bg="#111111",
                                     width=760, height=470)
        self._video_label.pack(fill="both", expand=True,
                               padx=8, pady=(0, 8))

        # Right panel
        right = tk.Frame(content, bg="#e0e0e0", width=280)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # Status card
        status_card = tk.Frame(right, bg="white", relief="flat", bd=1)
        status_card.pack(fill="x", pady=(0, 8))
        tk.Label(status_card, text="ℹ  Detection Status",
                 font=("Arial", 10, "bold"), bg="white"
                 ).pack(anchor="w", padx=10, pady=(8, 4))
        ttk.Separator(status_card).pack(fill="x", padx=10)

        self._status_vars = {
            "Camera Status":  tk.StringVar(value="Inactive"),
            "Hands Detected": tk.StringVar(value="0"),
            "Detection FPS":  tk.StringVar(value="0"),
            "Model Status":   tk.StringVar(value="Not Loaded"),
        }
        self._status_labels = {}
        for label, var in self._status_vars.items():
            row = tk.Frame(status_card, bg="white")
            row.pack(fill="x", padx=10, pady=3)
            tk.Label(row, text=label + ":", font=("Arial", 9),
                     bg="white", fg="#555").pack(side="left")
            lbl = tk.Label(row, textvariable=var,
                           font=("Arial", 9, "bold"),
                           bg="white", fg="#f44336")
            lbl.pack(side="right")
            self._status_labels[label] = lbl
        tk.Frame(status_card, height=8, bg="white").pack()

        # Parameters card
        params_card = tk.Frame(right, bg="white", relief="flat", bd=1)
        params_card.pack(fill="x")
        tk.Label(params_card, text="≡  Detection Parameters",
                 font=("Arial", 10, "bold"), bg="white"
                 ).pack(anchor="w", padx=10, pady=(8, 4))
        ttk.Separator(params_card).pack(fill="x", padx=10)

        self._det_conf  = tk.DoubleVar(value=0.75)
        self._trk_conf  = tk.DoubleVar(value=0.80)
        self._max_hands = tk.IntVar(value=2)

        self._add_slider(params_card, "Detection Confidence",
                         self._det_conf, 0, 1, self._on_params_change)
        self._add_slider(params_card, "Tracking Confidence",
                         self._trk_conf, 0, 1, self._on_params_change)
        self._add_slider(params_card, "Max Number of Hands",
                         self._max_hands, 1, 4, self._on_params_change,
                         integer=True)
        tk.Frame(params_card, height=8, bg="white").pack()

        # Capture log card
        log_card = tk.Frame(right, bg="white", relief="flat", bd=1)
        log_card.pack(fill="x", pady=(8, 0))
        tk.Label(log_card, text="📁  Capture Log",
                 font=("Arial", 10, "bold"), bg="white"
                 ).pack(anchor="w", padx=10, pady=(8, 4))
        ttk.Separator(log_card).pack(fill="x", padx=10)
        self._log_var = tk.StringVar(value="No captures yet.")
        tk.Label(log_card, textvariable=self._log_var,
                 font=("Arial", 8), fg="#555", bg="white",
                 wraplength=240, justify="left"
                 ).pack(anchor="w", padx=10, pady=6)

    def _add_slider(self, parent, label, variable,
                    from_, to, command, integer=False):
        frame = tk.Frame(parent, bg="white")
        frame.pack(fill="x", padx=10, pady=6)
        tk.Label(frame, text=label, font=("Arial", 9),
                 bg="white").pack(anchor="w")
        tick = tk.Frame(frame, bg="white")
        tick.pack(fill="x")
        mid = int((from_ + to) / 2) if integer \
              else f"{int((from_ + to) / 2 * 100)}%"
        tk.Label(tick, text=str(from_), font=("Arial", 7),
                 fg="#888", bg="white").pack(side="left")
        tk.Label(tick, text=str(mid), font=("Arial", 7),
                 fg="#888", bg="white").pack(side="left", expand=True)
        tk.Label(tick, text=str(to), font=("Arial", 7),
                 fg="#888", bg="white").pack(side="right")
        ttk.Scale(frame, from_=from_, to=to, variable=variable,
                  orient="horizontal",
                  command=lambda _: command()).pack(fill="x")

    # ── Button handlers ───────────────────────────────────
    def _on_start(self):
        self.backend.start()

    def _on_pause(self):
        self.backend.pause()

    def _on_stop(self):
        self.backend.stop()

    def _on_capture(self):
        fname = self.backend.capture_snapshot()
        self._log_var.set(
            f"Saved: {fname}" if fname else "Capture failed.")

    def _on_params_change(self):
        self.backend.configure(
            det_conf=self._det_conf.get(),
            trk_conf=self._trk_conf.get(),
            max_hands=self._max_hands.get(),
        )

    # ── Poll loop ─────────────────────────────────────────
    def _poll_ui(self):
        b = self.backend

        # Video
        if b.latest_frame is not None:
            lw = self._video_label.winfo_width()
            lh = self._video_label.winfo_height()
            tw = lw if lw > 10 else 760
            th = lh if lh > 10 else 470
            photo = ImageTk.PhotoImage(image=b.latest_frame.resize((tw, th)))
            self._current_photo         = photo
            self._video_label.imgtk     = photo
            self._video_label.config(image=photo)
        elif not b.is_running:
            self._video_label.config(image="", bg="#111111")

        # Status
        running = b.is_running
        self._status_vars["Camera Status"].set(
            "Active" if running else "Inactive")
        self._status_vars["Model Status"].set(
            "Loaded" if running else "Not Loaded")
        self._status_labels["Camera Status"].config(
            fg="#4CAF50" if running else "#f44336")
        self._status_labels["Model Status"].config(
            fg="#4CAF50" if running else "#f44336")
        self._status_vars["Hands Detected"].set(str(b.num_hands))
        self._status_vars["Detection FPS"].set(str(b.current_fps))

        # Badge
        if b.num_hands > 0:
            self._detect_badge.config(
                text=f"● {b.num_hands} Hand(s) Detected", fg="#4CAF50")
        else:
            self._detect_badge.config(text="● No Hand", fg="#888")

        self._video_label.after(30, self._poll_ui)

    def stop(self):
        """Called when app closes."""
        self.backend.stop()


# ══════════════════════════════════════════════════════════
#  MILESTONE 2 TAB  —  Gesture Recognition & Distance
# ══════════════════════════════════════════════════════════
class Milestone2Tab:
   

    def __init__(self, parent: tk.Frame):
        self.backend        = GestureDistanceBackend()
        self._current_photo = None
        self._build_ui(parent)
        self._poll_ui()

    # ── UI ────────────────────────────────────────────────
    def _build_ui(self, parent):
        # Title bar
        title = tk.Frame(parent, bg="#6A0DAD", pady=10)
        title.pack(fill="x")
        tk.Label(title,
                 text=" Gesture Recognition and Distance Measurement Module",
                 font=("Arial", 13, "bold"), fg="white", bg="#6A0DAD"
                 ).pack(anchor="w", padx=15)
        tk.Label(title,
                 text="Distance calculation between thumb & index  •  "
                      "Gesture classification  •  Real-time annotations",
                 font=("Arial", 9), fg="#E1BEE7", bg="#6A0DAD"
                 ).pack(anchor="w", padx=15)

        # Sub-header
        sub = tk.Frame(parent, bg="#f5f5f5", pady=7)
        sub.pack(fill="x")
        tk.Label(sub, text="Gesture Recognition Interface",
                 font=("Arial", 11, "bold"), bg="#f5f5f5"
                 ).pack(side="left", padx=15)

        btn_frame = tk.Frame(sub, bg="#f5f5f5")
        btn_frame.pack(side="right", padx=10)
        for label, color, cmd in [
            ("▶  Start",     "#4CAF50", self._on_start),
            ("⏸  Pause",    "#FF9800", self._on_pause),
            ("⚙  Settings", "#607D8B", self._on_settings),
        ]:
            tk.Button(btn_frame, text=label, bg=color, fg="white",
                      font=("Arial", 9, "bold"), relief="flat",
                      padx=10, pady=4, command=cmd
                      ).pack(side="left", padx=3)

        # Content
        content = tk.Frame(parent, bg="#e0e0e0")
        content.pack(fill="both", expand=True, padx=10, pady=8)

        # Video
        self._video_label = tk.Label(content, bg="black",
                                     width=700, height=480)
        self._video_label.pack(side="left", padx=(0, 8))

        # Right panel
        right = tk.Frame(content, bg="#e0e0e0", width=300)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # Distance card
        dist_card = tk.Frame(right, bg="white", relief="flat", bd=1)
        dist_card.pack(fill="x", pady=(0, 8))
        hdr = tk.Frame(dist_card, bg="#424242")
        hdr.pack(fill="x")
        tk.Label(hdr, text="▬  Distance Measurement",
                 font=("Arial", 10, "bold"), fg="white",
                 bg="#424242", pady=6).pack(anchor="w", padx=10)

        self._dist_var = tk.StringVar(value="0")
        tk.Label(dist_card, textvariable=self._dist_var,
                 font=("Arial", 44, "bold"),
                 fg="#212121", bg="white").pack(pady=(10, 0))
        tk.Label(dist_card, text="millimeters",
                 font=("Arial", 10), fg="#888",
                 bg="white").pack()

        sf = tk.Frame(dist_card, bg="white")
        sf.pack(fill="x", padx=15, pady=(8, 4))
        for txt, side, expand in [
            ("0mm", "left", False),
            ("50mm", "left", True),
            ("100mm", "right", False),
        ]:
            tk.Label(sf, text=txt, font=("Arial", 7),
                     fg="#aaa", bg="white").pack(side=side, expand=expand)

        self._dist_slider = ttk.Scale(dist_card, from_=0, to=150,
                                      orient="horizontal")
        self._dist_slider.pack(fill="x", padx=15, pady=(0, 10))

        # Gesture States card
        g_card = tk.Frame(right, bg="white", relief="flat", bd=1)
        g_card.pack(fill="x")
        g_hdr = tk.Frame(g_card, bg="#424242")
        g_hdr.pack(fill="x")
        tk.Label(g_hdr, text="●  Gesture States",
                 font=("Arial", 10, "bold"), fg="white",
                 bg="#424242", pady=6).pack(anchor="w", padx=10)

        b = self.backend
        gestures = [
            ("Open Hand", "#4CAF50", f"Distance > {b.open_hand_dist}mm"),
            ("Pinch",     "#FF9800", f"{b.pinch_dist_min}mm – {b.pinch_dist_max}mm"),
            ("Closed",    "#f44336", f"Distance < {b.closed_dist}mm"),
        ]
        self._gesture_rows = {}
        for name, color, desc in gestures:
            row = tk.Frame(g_card, bg="white", pady=4)
            row.pack(fill="x", padx=10)
            tk.Label(row, text="●", font=("Arial", 14),
                     fg=color, bg="white").pack(side="left")
            info = tk.Frame(row, bg="white")
            info.pack(side="left", padx=6)
            tk.Label(info, text=name, font=("Arial", 10, "bold"),
                     bg="white", fg="#212121").pack(anchor="w")
            tk.Label(info, text=desc, font=("Arial", 8),
                     bg="white", fg="#888").pack(anchor="w")
            self._gesture_rows[name] = row
            ttk.Separator(g_card).pack(fill="x", padx=10)
        tk.Frame(g_card, height=6, bg="white").pack()

    # ── Button handlers ───────────────────────────────────
    def _on_start(self):
        self.backend.start()

    def _on_pause(self):
        self.backend.pause()

    def _on_settings(self):
        win = tk.Toplevel()
        win.title("Settings")
        win.geometry("340x260")
        win.configure(bg="white")
        win.grab_set()
        tk.Label(win, text="⚙  Settings",
                 font=("Arial", 12, "bold"), bg="white"
                 ).pack(pady=(12, 4))
        ttk.Separator(win).pack(fill="x", padx=15)

        fields = [
            ("Open Hand threshold (mm)", "open_hand_dist"),
            ("Pinch min distance (mm)",  "pinch_dist_min"),
            ("Pinch max distance (mm)",  "pinch_dist_max"),
            ("Closed threshold (mm)",    "closed_dist"),
        ]
        svars = {}
        for label, attr in fields:
            row = tk.Frame(win, bg="white")
            row.pack(fill="x", padx=20, pady=4)
            tk.Label(row, text=label, font=("Arial", 9),
                     bg="white", width=28, anchor="w").pack(side="left")
            var = tk.StringVar(value=str(getattr(self.backend, attr)))
            tk.Entry(row, textvariable=var, width=6,
                     font=("Arial", 9)).pack(side="right")
            svars[attr] = var

        def apply():
            kwargs = {}
            for attr, var in svars.items():
                try:
                    kwargs[attr] = int(var.get())
                except ValueError:
                    pass
            self.backend.configure(**kwargs)
            win.destroy()

        tk.Button(win, text="Apply", bg="#6A0DAD", fg="white",
                  font=("Arial", 10, "bold"), relief="flat",
                  padx=20, pady=6, command=apply).pack(pady=14)

    # ── Poll loop ─────────────────────────────────────────
    def _poll_ui(self):
        b = self.backend

        # Video
        if b.latest_frame is not None:
            img   = b.latest_frame.resize((700, 460))
            photo = ImageTk.PhotoImage(image=img)
            self._current_photo         = photo
            self._video_label.imgtk     = photo
            self._video_label.config(image=photo)
        elif not b.is_running:
            self._video_label.config(image="", bg="black")

        # Distance
        self._dist_var.set(str(b.distance_mm))
        self._dist_slider.set(min(b.distance_mm, 150))

        # Gesture rows
        highlight = {
            "Open Hand": "#E8F5E9",
            "Pinch":     "#FFF3E0",
            "Closed":    "#FFEBEE",
        }
        for name, row in self._gesture_rows.items():
            bg = highlight[name] if name == b.gesture_name else "white"
            row.config(bg=bg)
            for child in row.winfo_children():
                try:
                    child.config(bg=bg)
                except Exception:
                    pass

        self._video_label.after(33, self._poll_ui)

    def stop(self):
        """Called when app closes."""
        self.backend.stop()


# ══════════════════════════════════════════════════════════
#  MAIN APP  —  Tab switcher
# ══════════════════════════════════════════════════════════
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hand Gesture Control System")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1150x780")

        # ── Tab bar ───────────────────────────────────────
        tab_bar = tk.Frame(self.root, bg="#0d0d1a", pady=6)
        tab_bar.pack(fill="x")

        tk.Label(tab_bar, text="🖐  Hand Gesture System",
                 font=("Arial", 13, "bold"),
                 fg="white", bg="#0d0d1a").pack(side="left", padx=16)

        self._tab_btns = {}
        btn_area = tk.Frame(tab_bar, bg="#0d0d1a")
        btn_area.pack(side="left", padx=20)

        for name, label in [
            ("m1", "📷  Milestone 1 — Hand Detection"),
            ("m2", "✌  Milestone 2 — Gesture & Distance"),
        ]:
            btn = tk.Button(
                btn_area, text=label,
                font=("Arial", 10, "bold"),
                relief="flat", padx=14, pady=6,
                command=lambda n=name: self._switch_tab(n),
            )
            btn.pack(side="left", padx=4)
            self._tab_btns[name] = btn

        # ── Tab content frames ────────────────────────────
        self._frames = {}
        for name in ("m1", "m2"):
            f = tk.Frame(self.root, bg="#1a1a2e")
            self._frames[name] = f
            f.pack(fill="both", expand=True)
            f.pack_forget()

        # ── Build each milestone tab ──────────────────────
        self._tabs = {
            "m1": Milestone1Tab(self._frames["m1"]),
            "m2": Milestone2Tab(self._frames["m2"]),
        }

        # ── Show Milestone 1 by default ───────────────────
        self._active = None
        self._switch_tab("m1")

    def _switch_tab(self, name: str):
        if self._active:
            self._frames[self._active].pack_forget()
            self._tab_btns[self._active].config(
                bg="#2a2a3a", fg="#aaaaaa")
        self._frames[name].pack(fill="both", expand=True)
        self._tab_btns[name].config(bg="#1565C0", fg="white")
        self._active = name

    def on_close(self):
        for tab in self._tabs.values():
            tab.stop()
        self.root.destroy()


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
