"""Slow Formant Wave - live layered drones + deep beat, spider-graph control.

Layers 1-3: independent pitch / vowels / volume drones (all at once).
Beat: deep kick + hats with BPM, pitch, decay, pattern controls.
Radar handles + sliders update the live sound in real time.
Play drones until Stop.
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
BLOCK = 2205  # 50 ms
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

AXES = [
    ("pitch", "Pitch", 40.0, 100.0),
    ("cycle", "Cycle s", 8.0, 40.0),
    ("bright", "Bright", 0.7, 1.4),
    ("bw", "BW", 0.6, 1.6),
    ("layervol", "LayerVol", 0.0, 1.0),
]

PATTERNS = {
    "Four": {"kicks": [0, 1, 2, 3], "hats": [0.5, 1.5, 2.5, 3.5]},
    "Half": {"kicks": [0, 2], "hats": [0.5, 1.5, 2.5, 3.5]},
    "Dub": {"kicks": [0], "hats": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]},
}


class LiveParams:
    def __init__(self):
        self.lock = threading.Lock()
        self.cycle = 24.0
        self.bright = 1.0
        self.bw = 1.0
        self.master = 0.85
        self.layers = [
            {"pitch": 55.0, "vol": 0.8, "on": True, "path": ["oo", "ah", "ee"]},
            {"pitch": 65.41, "vol": 0.6, "on": False, "path": ["oo", "oh", "ah"]},
            {"pitch": 49.0, "vol": 0.6, "on": False, "path": ["ee", "ah", "oo"]},
        ]
        self.beat_on = False
        self.pattern = "Four"
        self.bpm = 100.0
        self.kpitch = 55.0
        self.kdecay = 0.25
        self.kvol = 0.8
        self.hatvol = 0.3
        self.beatvol = 0.8


class Drone:
    def __init__(self, p: LiveParams):
        self.p = p
        self.stream = None
        self.harm = [np.zeros(N_HARM) for _ in range(3)]
        self.mph = [0.0, 2.0, 4.0]
        self.prev_gains = None
        self.prev_master = 0.85
        self.beat_phase = 0.0  # in beats

    def _snap(self):
        with self.p.lock:
            import copy
            return (self.p.cycle, self.p.bright, self.p.bw, self.p.master,
                    copy.deepcopy(self.p.layers),
                    self.p.beat_on, self.p.pattern, self.p.bpm,
                    self.p.kpitch, self.p.kdecay, self.p.kvol,
                    self.p.hatvol, self.p.beatvol)

    def _drone_block(self, li, F0, cyc, roll, bw, vol, path, frames):
        n = np.arange(frames, dtype=np.float64)
        mph = self.mph[li] + np.cumsum(np.full(frames, 2 * np.pi / (cyc * SR)))
        self.mph[li] = float(mph[-1] % (2 * np.pi * 1024))
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
            ph = self.harm[li][k - 1] + np.cumsum(np.full(frames, 2 * np.pi * F0 * k / SR))
            self.harm[li][k - 1] = float(ph[-1] % (2 * np.pi * 1024))
            fk = F0 * k
            env = np.zeros(frames)
            for i in range(3):
                d = (fk - F[:, i]) / (BW_BASE[i] * bw)
                env += AMP[i] * np.exp(-0.5 * d * d)
            out += (1.0 / (k ** roll)) * env * np.sin(ph)
        return np.tanh(out * 0.35) * vol

    def _beat_block(self, frames, beat_on, pattern, bpm, kpitch, kdecay,
                    kvol, hatvol, beatvol):
        out = np.zeros(frames)
        if not beat_on:
            self.beat_phase += frames / (SR * 60.0 / bpm)
            return out
        spb = SR * 60.0 / bpm
        pat = PATTERNS.get(pattern, PATTERNS["Four"])
        start = self.beat_phase
        end = start + frames / spb
        bar = 4.0
        # kick events
        b = math.floor(start)
        while b < end:
            in_bar = b % bar
            if any(abs(in_bar - k) < 1e-9 for k in pat["kicks"]):
                off = int(round((b - start) * spb))
                if 0 <= off < frames:
                    L = min(frames - off, int(SR * 0.8))
                    tt = np.arange(L) / SR
                    fr = kpitch * (1 + 2.2 * np.exp(-tt / 0.02))
                    ph = np.cumsum(2 * np.pi * fr / SR)
                    k = np.sin(ph) * np.exp(-tt / kdecay) * kvol
                    out[off:off + L] += k
            if any(abs(in_bar - h) < 1e-9 for h in pat["hats"]):
                off = int(round((b - start) * spb))
                if 0 <= off < frames:
                    L = min(frames - off, int(SR * 0.05))
                    nz = np.random.randn(L)
                    hp = np.diff(nz, prepend=0.0)
                    h = hp * np.exp(-np.arange(L) / SR / 0.012) * hatvol * 0.5
                    out[off:off + L] += h
            b += 0.5
        self.beat_phase = end
        return out * beatvol

    def callback(self, outdata, frames, time_info, status):
        cyc, roll, bw, master, layers, bon, pat, bpm, kp, kd, kv, hv, bv = self._snap()
        f = np.linspace(0, 1, frames)
        if self.prev_gains is None:
            self.prev_gains = [L["vol"] if L["on"] else 0.0 for L in layers]
        mix = np.zeros(frames)
        cur_gains = []
        for li, L in enumerate(layers):
            g = L["vol"] if L["on"] else 0.0
            cur_gains.append(g)
            ga = self.prev_gains[li] + (g - self.prev_gains[li]) * f
            path = [VOWELS[p] for p in L["path"]]
            mix += self._drone_block(li, L["pitch"], cyc, roll, bw, 1.0, path, frames) * ga
        self.prev_gains = cur_gains
        mix += self._beat_block(frames, bon, pat, bpm, kp, kd, kv, hv, bv)
        ma = self.prev_master + (master - self.prev_master) * f
        self.prev_master = master
        outdata[:, 0] = (np.tanh(mix) * ma).astype(np.float32)

    def start(self):
        if self.stream is not None:
            return
        self.stream = sd.OutputStream(samplerate=SR, channels=1,
                                      blocksize=BLOCK, callback=self.callback)
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


def render_offline(layers, cyc, roll, bw, master, beat, dur):
    N = int(SR * dur)
    t = np.arange(N, dtype=np.float64) / SR
    mix = np.zeros(N)
    for L in layers:
        if not L["on"]:
            continue
        path = [VOWELS[p] for p in L["path"]]
        m = 0.5 - 0.5 * np.cos(2 * np.pi * t / cyc)
        F = np.zeros((N, 3))
        mask = m <= 0.5
        k1 = np.empty_like(m)
        k1[mask] = m[mask] * 2.0
        k1[~mask] = (m[~mask] - 0.5) * 2.0
        for i in range(3):
            F[mask, i] = (1 - k1[mask]) * path[0][i] + k1[mask] * path[1][i]
            F[~mask, i] = (1 - k1[~mask]) * path[1][i] + k1[~mask] * path[2][i]
        BW = BW_BASE * bw
        y = np.zeros(N)
        for k in range(1, N_HARM + 1):
            fk = L["pitch"] * k
            env = np.zeros(N)
            for i in range(3):
                d = (fk - F[:, i]) / BW[i]
                env += AMP[i] * np.exp(-0.5 * d * d)
            y += (1.0 / (k ** roll)) * env * np.sin(2 * np.pi * fk * t)
        y = np.tanh(y * 0.35)
        mix += y * L["vol"]
    if beat["on"]:
        bv = beat["beatvol"]
        spb = SR * 60.0 / beat["bpm"]
        pat = PATTERNS.get(beat["pattern"], PATTERNS["Four"])
        nb = int(dur * beat["bpm"] / 60.0) + 1
        for b in range(nb):
            in_bar = b % 4
            off = int(round(b * spb))
            if off >= N:
                break
            if in_bar in pat["kicks"]:
                L = min(N - off, int(SR * 0.8))
                tt = np.arange(L) / SR
                fr = beat["kpitch"] * (1 + 2.2 * np.exp(-tt / 0.02))
                mix[off:off + L] += np.sin(np.cumsum(2 * np.pi * fr / SR)) * np.exp(-tt / beat["kdecay"]) * beat["kvol"] * bv
        nh = int(dur * beat["bpm"] / 60.0 * 2) + 1
        for h in range(nh):
            if (h * 0.5) % 4 in pat["hats"]:
                off = int(round(h * 0.5 * spb))
                if off >= N:
                    break
                L = min(N - off, int(SR * 0.05))
                nz = np.random.randn(L)
                hp = np.diff(nz, prepend=0.0)
                mix[off:off + L] += hp * np.exp(-np.arange(L) / SR / 0.012) * beat["hatvol"] * 0.5 * bv
    mix = np.tanh(mix) * master
    mix /= max(np.max(np.abs(mix)), 1e-9)
    mix *= 0.9 * master + 0.1
    fade = min(int(SR * 1.0), N // 4)
    win = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
    mix[:fade] *= win
    mix[-fade:] *= win[::-1]
    return (mix * 32767).astype(np.int16)


class Radar(tk.Canvas):
    def __init__(self, parent, get01, set01, size=300):
        super().__init__(parent, width=size, height=size,
                         bg="#101418", highlightthickness=0)
        self.size = size
        self.get01 = get01
        self.set01 = set01
        self.drag_axis = None
        self.bind("<Button-1>", self._down)
        self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", self._up)
        self.draw()

    def _geom(self):
        c = self.size / 2
        R = self.size / 2 - 44
        angs = [-math.pi / 2 + 2 * math.pi * i / len(AXES) for i in range(len(AXES))]
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
            lx, ly = self._pt(c, R, ang, 1.2)
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
        self.title("Slow Formant Wave - layers + deep beat")
        self.live = LiveParams()
        self.drone = Drone(self.live) if HAS_SD else None
        self.playing = False
        self._syncing = False
        self.sel_layer = tk.IntVar(value=0)
        self.vars = {}
        self._build()
        self._load_layer(0)
        self._push_to_live()

    def _slider(self, parent, key, label, frm, to, default):
        ttk.Label(parent, text=label).pack(anchor="w")
        var = tk.DoubleVar(value=default)
        ttk.Scale(parent, from_=frm, to=to, variable=var,
                  orient="horizontal", length=200).pack(fill="x")
        val = ttk.Label(parent, text=str(default))
        val.pack(anchor="e")
        self.vars[key] = (var, val, frm, to)
        var.trace_add("write", lambda *a, k=key: self._on_slider(k))

    def _build(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="both", expand=True)

        # ---- drone column ----
        d = ttk.Frame(top)
        d.pack(side="left", fill="y")
        ttk.Label(d, text="DRONE LAYERS", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        lr = ttk.Frame(d)
        lr.pack(fill="x", pady=4)
        for i in range(3):
            ttk.Radiobutton(lr, text=f"{i + 1}", variable=self.sel_layer,
                            value=i, command=self._on_sel_layer).pack(side="left")
        self.layer_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(d, text="layer on", variable=self.layer_on,
                        command=self._on_layer_toggle).pack(anchor="w")
        self._slider(d, "pitch", "Pitch F0 (Hz)", 40, 100, 55)
        self._slider(d, "cycle", "Cycle / wave time (s)", 8, 40, 24)
        self._slider(d, "length", "Save length (s)", 4, 30, 12)
        self._slider(d, "bright", "Brightness (low=dark)", 0.7, 1.4, 1.0)
        self._slider(d, "bw", "BW scale (low=sharp)", 0.6, 1.6, 1.0)
        self._slider(d, "layervol", "Layer volume", 0.0, 1.0, 0.8)
        self._slider(d, "master", "Master volume", 0.2, 1.0, 0.85)
        vr = ttk.Frame(d)
        vr.pack(fill="x", pady=6)
        self.v1 = tk.StringVar(value="oo")
        self.v2 = tk.StringVar(value="ah")
        self.v3 = tk.StringVar(value="ee")
        for lbl, var in (("A", self.v1), ("B", self.v2), ("C", self.v3)):
            c = ttk.Frame(vr)
            c.pack(side="left", expand=True)
            ttk.Label(c, text=lbl).pack()
            ttk.OptionMenu(c, var, var.get(), *VOWELS.keys()).pack()
        for v in (self.v1, self.v2, self.v3):
            v.trace_add("write", lambda *a: self._on_vowel())

        # ---- radar column ----
        mid = ttk.Frame(top)
        mid.pack(side="left", padx=10)
        self.radar = Radar(mid, self._get01, self._set01)
        self.radar.pack()
        ttk.Label(mid, text="drag the blue handles\n(radar = selected layer)",
                  foreground="gray", justify="center").pack()
        btns = ttk.Frame(mid)
        btns.pack(fill="x", pady=10)
        self.play_btn = ttk.Button(btns, text="Play", command=self.toggle)
        self.play_btn.pack(side="left", expand=True, padx=2)
        ttk.Button(btns, text="Save...", command=self.save).pack(side="left", expand=True, padx=2)
        self.status = ttk.Label(mid, text="hit Play - drones until Stop")
        self.status.pack(pady=4)
        if not HAS_SD:
            self.status.config(text="no sounddevice: pip install sounddevice")
            self.play_btn.config(state="disabled")

        # ---- beat column ----
        b = ttk.Frame(top)
        b.pack(side="left", fill="y")
        ttk.Label(b, text="DEEP BEAT", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.beat_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(b, text="beat on", variable=self.beat_on,
                        command=self._push_to_live).pack(anchor="w", pady=4)
        self.pattern = tk.StringVar(value="Four")
        pr = ttk.Frame(b)
        pr.pack(fill="x")
        ttk.Label(pr, text="Pattern:").pack(side="left")
        ttk.OptionMenu(pr, self.pattern, "Four", *PATTERNS.keys()).pack(side="left")
        self.pattern.trace_add("write", lambda *a: self._push_to_live())
        self._slider(b, "bpm", "BPM", 60, 140, 100)
        self._slider(b, "kpitch", "Kick pitch (Hz)", 40, 80, 55)
        self._slider(b, "kdecay", "Kick decay (s)", 0.1, 0.6, 0.25)
        self._slider(b, "kvol", "Kick volume", 0.0, 1.0, 0.8)
        self._slider(b, "hatvol", "Hat volume", 0.0, 1.0, 0.3)
        self._slider(b, "beatvol", "Beat volume", 0.0, 1.0, 0.8)

    # ---- sync helpers ----
    def _on_slider(self, key):
        if self._syncing:
            return
        var, val, _, _ = self.vars[key]
        val.config(text=f"{var.get():.2f}")
        if key in ("pitch", "layervol"):
            self._store_layer()
        self._push_to_live()
        self.radar.refresh()

    def _store_layer(self):
        i = self.sel_layer.get()
        with self.live.lock:
            self.live.layers[i]["pitch"] = self.vars["pitch"][0].get()
            self.live.layers[i]["vol"] = self.vars["layervol"][0].get()

    def _load_layer(self, i):
        with self.live.lock:
            L = dict(self.live.layers[i])
        self._syncing = True
        try:
            self.vars["pitch"][0].set(L["pitch"])
            self.vars["layervol"][0].set(L["vol"])
            self.vars["pitch"][1].config(text=f"{L['pitch']:.2f}")
            self.vars["layervol"][1].config(text=f"{L['vol']:.2f}")
            self.layer_on.set(L["on"])
            for var, name in ((self.v1, L["path"][0]), (self.v2, L["path"][1]), (self.v3, L["path"][2])):
                var.set(name)
        finally:
            self._syncing = False
        self.radar.refresh()

    def _on_sel_layer(self):
        self._load_layer(self.sel_layer.get())

    def _on_layer_toggle(self):
        i = self.sel_layer.get()
        on = self.layer_on.get()
        with self.live.lock:
            self.live.layers[i]["on"] = on
            if on and self.live.layers[i]["vol"] < 0.05:
                self.live.layers[i]["vol"] = 0.7
        if on:
            self._syncing = True
            try:
                with self.live.lock:
                    v = self.live.layers[i]["vol"]
                self.vars["layervol"][0].set(v)
                self.vars["layervol"][1].config(text=f"{v:.2f}")
            finally:
                self._syncing = False
            self.radar.refresh()
        self._push_to_live()

    def _on_vowel(self):
        if self._syncing:
            return
        i = self.sel_layer.get()
        with self.live.lock:
            self.live.layers[i]["path"] = [self.v1.get(), self.v2.get(), self.v3.get()]
        self._push_to_live()

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
        if key in ("pitch", "layervol"):
            self._store_layer()
        self._push_to_live()

    def _push_to_live(self):
        with self.live.lock:
            self.live.cycle = self.vars["cycle"][0].get()
            self.live.bright = self.vars["bright"][0].get()
            self.live.bw = self.vars["bw"][0].get()
            self.live.master = self.vars["master"][0].get()
            self.live.beat_on = self.beat_on.get()
            self.live.pattern = self.pattern.get()
            self.live.bpm = self.vars["bpm"][0].get()
            self.live.kpitch = self.vars["kpitch"][0].get()
            self.live.kdecay = self.vars["kdecay"][0].get()
            self.live.kvol = self.vars["kvol"][0].get()
            self.live.hatvol = self.vars["hatvol"][0].get()
            self.live.beatvol = self.vars["beatvol"][0].get()

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
            self.status.config(text="playing - everything is live")
        else:
            self.drone.stop()
            self.playing = False
            self.play_btn.config(text="Play")
            self.status.config(text="stopped")

    def save(self):
        p = filedialog.asksaveasfilename(defaultextension=".wav",
            filetypes=[("WAV", "*.wav")],
            initialfile="formant_layers_beat.wav")
        if not p:
            return
        with self.live.lock:
            import copy
            layers = copy.deepcopy(self.live.layers)
            cyc = self.live.cycle
            roll = self.live.bright
            bw = self.live.bw
            master = self.live.master
            beat = {"on": self.live.beat_on, "pattern": self.live.pattern,
                    "bpm": self.live.bpm, "kpitch": self.live.kpitch,
                    "kdecay": self.live.kdecay, "kvol": self.live.kvol,
                    "hatvol": self.live.hatvol, "beatvol": self.live.beatvol}
        dur = self.vars["length"][0].get()
        self.status.config(text="rendering...")
        def work():
            try:
                pcm = render_offline(layers, cyc, roll, bw, master, beat, dur)
                with wave.open(p, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SR)
                    wf.writeframes(pcm.tobytes())
                self.status.config(text=f"saved {os.path.basename(p)}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        threading.Thread(target=work, daemon=True).start()


if __name__ == "__main__":
    App().mainloop()
