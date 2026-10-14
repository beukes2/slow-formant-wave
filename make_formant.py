import numpy as np
import wave

sr = 44100
dur = 30.0
N = int(sr * dur)
t = np.arange(N, dtype=np.float64) / sr

F0 = 110.0  # A2 drone, steady pitch
n_harm = 32

# Vowel formants F1,F2,F3
F_oo = np.array([300.0, 870.0, 2240.0])
F_ah = np.array([730.0, 1090.0, 2440.0])
F_ee = np.array([270.0, 2290.0, 3010.0])
A = np.array([1.0, 0.5, 0.25])
BW = np.array([90.0, 120.0, 180.0])

# Slow morph 0->1->0 over full duration (30s cycle)
m = 0.5 - 0.5 * np.cos(2 * np.pi * t / dur)

F = np.zeros((N, 3), dtype=np.float64)
mask = m <= 0.5
k1 = np.zeros_like(m)
k1[mask] = m[mask] * 2.0
k1[~mask] = (m[~mask] - 0.5) * 2.0
# oo->ah when mask, ah->ee when not
for i in range(3):
    F[mask, i] = (1 - k1[mask]) * F_oo[i] + k1[mask] * F_ah[i]
    F[~mask, i] = (1 - k1[~mask]) * F_ah[i] + k1[~mask] * F_ee[i]

y = np.zeros(N, dtype=np.float64)

# Additive saw + time-varying formant envelope
for k in range(1, n_harm + 1):
    fk = F0 * k
    # envelope from 3 formants
    env = np.zeros(N, dtype=np.float64)
    for i in range(3):
        d = (fk - F[:, i]) / BW[i]
        env += A[i] * np.exp(-0.5 * d * d)
    amp = (1.0 / (k ** 0.9)) * env
    y += amp * np.sin(2 * np.pi * fk * t)

# normalize
y /= np.max(np.abs(y)) + 1e-9
y *= 0.85

# 2s fade in/out
fade = int(sr * 2.0)
win_in = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
y[:fade] *= win_in
y[-fade:] *= win_in[::-1]

# to 16-bit
pcm = (y * 32767).astype(np.int16)
with wave.open("slow_formant_wave.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(pcm.tobytes())

print(f"wrote slow_formant_wave.wav {dur}s @ {sr}Hz, F0={F0}Hz, oo->ah->ee->ah->oo")
