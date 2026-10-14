import numpy as np
import wave

sr = 44100

F_OO = np.array([300.0, 870.0, 2240.0])
F_AH = np.array([730.0, 1090.0, 2440.0])
F_EE = np.array([270.0, 2290.0, 3010.0])
F_OH = np.array([570.0, 840.0, 2410.0])
F_EH = np.array([530.0, 1840.0, 2480.0])
A = np.array([1.0, 0.5, 0.25])

VOWELS = {"oo": F_OO, "ah": F_AH, "ee": F_EE, "oh": F_OH, "eh": F_EH}

# 10 variations, all lower than 110Hz
variations = [
    {"name": "slow_formant_v01_55Hz_oo-ah-ee",  "F0": 55.0,  "dur": 24.0, "path": ["oo","ah","ee"], "rolloff": 0.9, "bw_scale": 1.0},
    {"name": "slow_formant_v02_58Hz_oo-oh-ah",  "F0": 58.27, "dur": 24.0, "path": ["oo","oh","ah"], "rolloff": 1.0, "bw_scale": 1.2},
    {"name": "slow_formant_v03_61Hz_ah-ee-eh",  "F0": 61.74, "dur": 26.0, "path": ["ah","ee","eh"], "rolloff": 1.1, "bw_scale": 0.9},
    {"name": "slow_formant_v04_65Hz_oo-ee-ah",  "F0": 65.41, "dur": 26.0, "path": ["oo","ee","ah"], "rolloff": 0.9, "bw_scale": 1.0},
    {"name": "slow_formant_v05_69Hz_oh-ah-oo",  "F0": 69.30, "dur": 22.0, "path": ["oh","ah","oo"], "rolloff": 1.2, "bw_scale": 1.3},
    {"name": "slow_formant_v06_73Hz_ee-ah-oo",  "F0": 73.42, "dur": 22.0, "path": ["ee","ah","oo"], "rolloff": 1.0, "bw_scale": 0.8},
    {"name": "slow_formant_v07_77Hz_oo-eh-ee",  "F0": 77.78, "dur": 24.0, "path": ["oo","eh","ee"], "rolloff": 0.8, "bw_scale": 1.0},
    {"name": "slow_formant_v08_82Hz_ah-oh-oo",  "F0": 82.41, "dur": 24.0, "path": ["ah","oh","oo"], "rolloff": 1.1, "bw_scale": 1.1},
    {"name": "slow_formant_v09_55Hz_ee-oh-oo",  "F0": 55.0,  "dur": 28.0, "path": ["ee","oh","oo"], "rolloff": 1.3, "bw_scale": 1.4},
    {"name": "slow_formant_v10_65Hz_oo-ah-oh",  "F0": 65.41, "dur": 28.0, "path": ["oo","ah","oh"], "rolloff": 0.85,"bw_scale": 0.7},
]

n_harm = 24
BW_BASE = np.array([90.0, 120.0, 180.0])

def render_one(v):
    dur = v["dur"]
    N = int(sr * dur)
    t = np.arange(N, dtype=np.float64) / sr
    F0 = v["F0"]
    path = [VOWELS[p] for p in v["path"]]
    # cycle through path and back: e.g. oo->ah->ee->ah->oo
    # m goes 0->1->0
    m = 0.5 - 0.5 * np.cos(2 * np.pi * t / dur)
    # map m to 2-segment path (3 vowels)
    F = np.zeros((N, 3))
    mask = m <= 0.5
    k1 = np.empty_like(m)
    k1[mask] = m[mask] * 2.0
    k1[~mask] = (m[~mask] - 0.5) * 2.0
    for i in range(3):
        F[mask, i] = (1 - k1[mask]) * path[0][i] + k1[mask] * path[1][i]
        F[~mask, i] = (1 - k1[~mask]) * path[1][i] + k1[~mask] * path[2][i]

    BW = BW_BASE * v["bw_scale"]
    y = np.zeros(N)
    for k in range(1, n_harm + 1):
        fk = F0 * k
        env = np.zeros(N)
        for i in range(3):
            d = (fk - F[:, i]) / BW[i]
            env += A[i] * np.exp(-0.5 * d * d)
        y += (1.0 / (k ** v["rolloff"])) * env * np.sin(2 * np.pi * fk * t)

    y /= np.max(np.abs(y)) + 1e-9
    y *= 0.85
    fade = int(sr * 1.5)
    win = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
    y[:fade] *= win
    y[-fade:] *= win[::-1]
    pcm = (y * 32767).astype(np.int16)
    with wave.open(v["name"] + ".wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    print(f"wrote {v['name']}.wav F0={F0}Hz dur={dur}s path={'-'.join(v['path'])}")

for v in variations:
    render_one(v)
