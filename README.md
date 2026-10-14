# slow-formant-wave

Long, slow formant / vowel-morph drones in the spirit of Kraftwerk-style
robotic vowels. Steady pitch, moving mouth — like opening and closing your
mouth while holding one note.

## What it does

- `formant_gui.py` — live app: 3 independent drone layers (pitch / vowels /
  volume each, all at once) plus a deep beat (BPM, pattern, kick pitch/decay/
  volume, hats). Spider-graph + sliders, everything live in real time.
  Press Play and it drones until Stop. Save renders the mix to WAV.
- `make_formant.py` — renders one 30s example (`oo -> ah -> ee -> ah -> oo`,
  110 Hz drone) with NumPy only.
- `make_formant_10.py` — renders 10 lower-pitched variations (55–82 Hz).

## Requirements

- Python 3.10+
- NumPy (`pip install -r requirements.txt`)
- Tkinter ships with standard Python. Live sound needs `sounddevice`
  (`pip install -r requirements.txt`); without it, use Save and play the WAV.

## Use

```bash
pip install -r requirements.txt
python formant_gui.py
```

Or render headlessly:

```bash
python make_formant.py
python make_formant_10.py
```

## How it works

Sawtooth drone at a fixed F0 (pitch never changes) passed through a
time-varying 3-formant (F1/F2/F3) spectral envelope. A sub-audio LFO
(0.03–0.12 Hz, i.e. an 8–30 s cycle) morphs the formants between vowels,
so you hear a slow `ah-oh-oo` wave over a constant note.
