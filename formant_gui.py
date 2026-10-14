"""Slow Formant Wave - live drone with spider-graph control.

Drag the radar handles or the sliders: the sound updates in real time.
Press Play once - it drones until you press Stop.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import wave
import threading
import math
import os

try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False

SR = 44100
BLOCK = 2205  # 50 ms blocks
N_HARM = 24

VOWELS = {
    "oo": np.array([300.0, 870.0, 2240.0]),
    "oh": np.array([570.0, 840.0, 2410.0]),
    "ah": np.array([730.0, 1090.0, 2440.0]),
    "eh": np.array([530.0, 1840.0, 2480.0]),
    "ee": np.array([270.0, 2290.0, 3010.0]),
}
AMP = np.array([1.0, 0.5, 0.25])
BW_BASE = np.array([90.0, 120.0, 180.0])

# radar axes: (key, label, min, max)
AXES = [
    ("pitch", "Pitch", 40.0, 100.0),
    ("cycle", "Cycle s", 8.0, 40.0),
    ("bright", "Bright", 0.7, 1.4),
    ("bw", "BW", 0.6, 1.6),
    ("volume", "Volume", 0.2, 1.0),
]


class LiveParams:
    """Plain floats shared between GUI thread and audio thread."""
    def __init__(self):
        self.lock = threading.Lock()
        self.pitch = 62.0
        self.cycle = 24.0
        self.bright = 1.0
        self.bw = 1.0
        self.volume = 0.85
        self.path = [VOWELS["oo"].copy(), VOWELS["ah"].copy(), VOWELS["ee"].copy()]


class Drone:
    def __init__(self, p: LiveParams):
        self.p = p
        self.stream = None
        self.harm_phase = np.zeros(N_HARM)
        self.morph_phase = 0.0
        self.prev = None

    def _snap(self):
        with self.p.lock:
            return (self.p.pitch, self.p.cycle, self.p.bright,
                    self.p.bw, self.p.volume,
                    [v.copy() for v in self.p.path])

    def callback(self, outdata, frames, time_info, status):
        F0, cyc, roll, bw, vol, path = self._snap()
        if self.prev is None:
            self.prev = (F0, cyc, roll, bw, vol)
        pF0, pCyc, pRoll, pBw, pVol = self.prev
        n = np.arange(frames, dtype=np.float64)
        f = n / max(frames - 1, 1)
        F0a = pF0 + (F0 - pF0) * f
        cycA = pCyc + (cyc - pCyc) * f
        rollA = pRoll + (roll - pRoll) * f
        bwA = pBw + (bw - pBw) * f
        volA = pVol + (vol - pVol) * f
        self.prev = (F0, cyc, roll, bw, vol)

        mph = self.morph_phase + np.cumsum(2 * np.pi / (cycA * SR))
        self.morph_phase = float(mph[-1] % (2 * np.pi * 1024))
        m = 0.5 - 0.5 * np.cos(mph % (2 * np.pi))
        F = np.zeros((frames, 3))
        mask = m <= 0.5
        k1 = np.empty_like(m)
        k1[mask] = m[mask] * 2.0
        k1[~mask] = (m[~mask] - 0.5) * 2.0
        for i in range(3):
            F[mask, i] = (1 - k1[mask]) * path[0][i] + k1[mask] * path[1][i]
            F[~mask, i] = (1 - k1[~mask]) * path[1][i] + k1[~mask] * path[2][i]

        out = np.zeros(frames)
        for k in range(1, N_HARM + 1):
            dphi = 2 * np.pi * F0a * k / SR
            ph = self.harm_phase[k - 1] + np.cumsum(dphi)
            self.harm_phase[k - 1] = float(ph[-1] % (2 * np.pi * 1024))
            fk = F0 * k
            env = np.zeros(frames)
            for i in range(3):
                d = (fk - F[:, i]) / (BW_BASE[i] * bwA)
                env += AMP[i] * np.exp(-0.5 * d * d)
            out += (1.0 / (k ** rollA)) * env * np.sin(ph)
        out = np.tanh(out * 0.35) * volA
        outdata[:, 0] = out.astype(np.float32)

    def start(self):
        if self.stream is not None:
            return
        self.prev = None
        self.stream = sd.OutputStream(samplerate=SR, channels=1,
                                      blocksize=BLOCK, callback=self.callback)
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


def render_offline(F0, dur, cycle, path_names, rolloff, bw_scale, volume=0.85):
    path = [VOWELS[p] for p in path_names]
    N = int(SR * dur)
    t = np.arange(N, dtype=np.float64) / SR
    m = 0.5 - 0.5 * np.cos(2 * np.pi * t / cycle)
    F = np.zeros((N, 3))
    mask = m <= 0.5
    k1 = np.empty_like(m)
    k1[mask] = m[mask] * 2.0
    k1[~mask] = (m[~mask] - 0.5) * 2.0
    for i in range(3):
        F[mask, i] = (1 - k1[mask]) * path[0][i] + k1[mask] * path[1][i]
        F[~mask, i] = (1 - k1[~mask]) * path[1][i] + k1[~mask] * path[2][i]
    BW = BW_BASE * bw_scale
    y = np.zeros(N)
    for k in range(1, N_HARM + 1):
        fk = F0 * k
        env = np.zeros(N)
        for i in range(3):
            d = (fk - F[:, i]) / BW[i]
            env += AMP[i] * np.exp(-0.5 * d * d)
        y += (1.0 / (k ** rolloff)) * env * np.sin(2 * np.pi * fk * t)
    y /= np.max(np.abs(y)) + 1e-9
    y *= volume
    fade = min(int(SR * 1.0), N // 4)
    win = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
    y[:fade] *= win
    y[-fade:] *= win[::-1]
    return (y * 32767).astype(np.int16)


class Radar(tk.Canvas):
    """Spider graph: drag any handle to drive its parameter live."""
    def __init__(self, parent, get01, set01, size=320):
        super().__init__(parent, width=size, height=size,
                         bg="#101418", highlightthickness=0)
        self.size = size
        self.get01 = get01
        self.set01 = set01
        self.drag_axis = None
        self.bind("<Button-1>", self._down)
        bindata = self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", self._up)
        self.draw()

    def _geom(self):
        c = self.size / 2
        R = self.size / 2 - 44
        angs = [ -math.pi / 2 + 2 * math.pi * i / len(AXES) for i in range(len(AXES))]
        return c, R, angs

    def _pt(self, c, R, ang, v01):
        return (c + R * v01 * math.cos(ang), c + R * v01 * math.sin(ang))

    def draw(self):
        self.delete("all")
        c, R, angs = self._geom()
        for ring in (0.25, 0.5, 0.75, 1.0):
            pts = []
            for ang in angs:
                pts += [c + R * ring * math.cos(ang), c + R * ring * math.sin(ang)]
            self.create_polygon(pts, outline="#2c3640", fill="")
        vals = [self.get01(k) for k, _, _, _ in AXES]
        for (key, label, lo, hi), ang, v in zip(AXES, angs, vals):
            x2, y2 = self._pt(c, R, ang, 1.0)
            self.create_line(c, c, x2, y2, fill="#2c3640")
            lx, ly = self._pt(c, R, ang, 1.18)
            real = lo + v * (hi - lo)
            self.create_text(lx, ly, text=f"{label}\n{real:.2f}",
                             fill="#9fb4c8", font=("Segoe UI", 8), justify="center")
        poly = []
        for ang, v in zip(angs, vals):
            poly += list(self._pt(c, R, ang, v))
        if len(poly) >= 6:
            self.create_polygon(poly, fill="#1c2836", outline="#3fa9f5", width=2)
        for ang, v in zip(angs, vals):
            x, y = self._pt(c, R, ang, v)
            self.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#3fa9f5", outline="white")

    def _axis_from_xy(self, x, y):
        c, R, angs = self._geom()
        best, bestd = 0, 1e9
        for i, ang in enumerate(angs):
            hx, hy = self._pt(c, R, ang, self.get01(AXES[i][0]))
            d = (x - hx) ** 2 + (y - hy) ** 2
            if d < bestd:
                best, bestd = i, d
        return best if bestd < 24 ** 2 else None

    def _set_from_xy(self, i, x, y):
        c, R, angs = self._geom()
        dx, dy = x - c, y - c
        v = (dx * math.cos(angs[i]) + dy * math.sin(angs[i])) / R
        self.set01(AXES[i][0], max(0.0, min(1.0, v)))
        self.draw()

    def _down(self, e):
        self.drag_axis = self._axis_from_xy(e.x, e.y)

    def _move(self, e):
        if self.drag_axis is not None:
            self._set_from_xy(self.drag_axis, e.x, e.y)

    def _up(self, e):
        self.drag_axis = None

    def refresh(self):
        self.draw()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Slow Formant Wave - live")
        self.live = LiveParams()
        self.drone = Drone(self.live) if HAS_SD else None
        self.playing = False
        self._syncing = False
        self.vars = {}
        self._build()
        self._push_to_live()

    def _build(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="both", expand=True)

        right = ttk.Frame(top)
        right.pack(side="left", fill="y")
        self._slider(right, "pitch", "Pitch F0 (Hz)", 40, 100, 62)
        self._slider(right, "cycle", "Cycle / wave time (s)", 8, 40, 24)
        self._slider(right, "length", "Save length (s)", 4, 30, 12)
        self._slider(right, "bright", "Brightness (low=dark)", 0.7, 1.4, 1.0)
        self._slider(right, "bw", "BW scale (low=sharp)", 0.6, 1.6, 1.0)
        self._slider(right, "volume", "Volume", 0.2, 1.0, 0.85)

        row = ttk.Frame(right)
        row.pack(fill="x", pady=8)
        self.v1 = tk.StringVar(value="oo")
        self.v2 = tk.StringVar(value="ah")
        self.v3 = tk.StringVar(value="ee")
        for lbl, var in (("Vowel A", self.v1), ("Vowel B", self.v2), ("Vowel C", self.v3)):
            c = ttk.Frame(row)
            c.pack(side="left", expand=True)
            ttk.Label(c, text=lbl).pack()
            ttk.OptionMenu(c, var, var.get(), *VOWELS.keys()).pack()
        for v in (self.v1, self.v2, self.v3):
            v.trace_add("write", lambda *a: self._push_to_live())

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=10)
        self.play_btn = ttk.Button(btns, text="Play", command=self.toggle)
        self.play_btn.pack(side="left", expand=True, padx=2)
        ttk.Button(btns, text="Save...", command=self.save).pack(side="left", expand=True, padx=2)
        self.status = ttk.Label(right, text="hit Play - drones until Stop")
        self.status.pack(pady=4)
        if not HAS_SD:
            self.status.config(text="no sounddevice: pip install sounddevice for live sound")
            self.play_btn.config(state="disabled")

        left = ttk.Frame(top)
        left.pack(side="left", padx=(10, 0))
        self.radar = Radar(left, self._get01, self._set01)
        self.radar.pack()
        ttk.Label(left, text="drag the blue handles",
                  foreground="gray").pack()

    def _slider(self, parent, key, label, frm, to, default):
        ttk.Label(parent, text=label).pack(anchor="w")
        var = tk.DoubleVar(value=default)
        s = ttk.Scale(parent, from_=frm, to=to, variable=var, orient="horizontal", length=220)
        s.pack(fill="x")
        val = ttk.Label(parent, text=str(default))
        val.pack(anchor="e")
        self.vars[key] = (var, val, frm, to)
        var.trace_add("write", lambda *a, k=key: self._on_slider(k))

    def _on_slider(self, key):
        if self._syncing:
            return
        var, val, frm, to = self.vars[key]
        val.config(text=f"{var.get():.2f}")
        self._push_to_live()
        if hasattr(self, "radar"):
            self.radar.refresh()

    def _get01(self, key):
        var, _, lo, hi = self.vars[key]
        return (var.get() - lo) / (hi - lo)

    def _set01(self, key, v01):
        var, val, lo, hi = self.vars[key]
        self._syncing = True
        try:
            var.set(lo + v01 * (hi - lo))
            val.config(text=f"{var.get():.2f}")
        finally:
            self._syncing = False
        self._push_to_live()

    def _push_to_live(self):
        with self.live.lock:
            self.live.pitch = self.vars["pitch"][0].get()
            self.live.cycle = self.vars["cycle"][0].get()
            self.live.bright = self.vars["bright"][0].get()
            self.live.bw = self.vars["bw"][0].get()
            self.live.volume = self.vars["volume"][0].get()
            self.live.path = [VOWELS[self.v1.get()].copy(),
                              VOWELS[self.v2.get()].copy(),
                              VOWELS[self.v3.get()].copy()]

    def toggle(self):
        if self.drone is None:
            return
        if not self.playing:
            try:
                self.drone.start()
            except Exception as e:
                messagebox.showerror("Audio error", str(e))
                return
            self.playing = True
            self.play_btn.config(text="Stop")
            self.status.config(text="playing - drag radar, sound follows")
        else:
            self.drone.stop()
            self.playing = False
            self.play_btn.config(text="Play")
            self.status.config(text="stopped")

    def save(self):
        p = filedialog.asksaveasfilename(defaultextension=".wav",
            filetypes=[("WAV", "*.wav")],
            initialfile=f"formant_{int(self.vars['pitch'][0].get())}Hz.wav")
        if not p:
            return
        v = self.vars
        pcm = render_offline(v["pitch"][0].get(), v["length"][0].get(),
                             v["cycle"][0].get(),
                             [self.v1.get(), self.v2.get(), self.v3.get()],
                             v["bright"][0].get(), v["bw"][0].get(), v["volume"][0].get())
        with wave.open(p, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            wf.writeframes(pcm.tobytes())
        self.status.config(text=f"saved {os.path.basename(p)}")


if __name__ == "__main__":
    App().mainloop()
