# OIG — Object-centric Image/Video/Game World Model

A four-phase research project building toward a controllable game world model using differentiable object-centric representations.

## Overview

| Phase | What it learns | Input → Output |
|-------|---------------|----------------|
| **1 — Static** | Decompose images into primitives | Image → Slots → Canvas → Reconstructed image |
| **2 — Temporal** | How slots evolve over time | Slots(t) → Slots(t+1) → Next image |
| **3 — Controllable** | What actions cause which changes | (Slots, Action code) → Slots(t+1) → Next image |
| **4 — Planning** | *(future)* Goal-conditioned action search | — |

---

## Requirements

```bash
pip install torch torchvision numpy opencv-python-headless Pillow tqdm decord
```

A CUDA GPU is strongly recommended. All phases were developed with PyTorch 2.x on Windows.

---

## Repository structure

```
config.py              # Phase 1 hyperparameters
config_p2.py           # Phase 2 overrides (imports config.py)
config_p3.py           # Phase 3 overrides (imports config_p2.py)
model.py               # All neural network architectures (O, I, N, LAM)
rasterizer.py          # Differentiable primitive renderer
dataset_video.py       # Video frame dataset (JPEG / decord / OpenCV backends)
train.py               # Phase 1 training loop
train_p2.py            # Phase 2 training loop
train_p3.py            # Phase 3 training loop
preprocess_videos.py   # Extract JPEG frames from videos (run once before Phase 2/3)
remux_faststart.py     # Remux MP4s with faststart header (improves decord seek speed)
engine.py              # Interactive inference engine (run after Phase 3)
loss.ipynb             # Notebook to plot training losses
```

---

## Phase 1 — Static image reconstruction

### Data

Images from [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) or any folder of images.

Set `FLICKR_DIR` in `config.py` to point to your image folder.

### Configure

Key settings in `config.py`:

```python
IMAGE_SIZE  = 64       # resolution (square); increase to 128 for higher quality
MAX_SLOTS   = 32       # number of primitives per image
PRIMITIVE   = "mixed"  # "circle" or "mixed" (circles + lines)
BATCH_SIZE  = 32
NUM_EPOCHS  = 100
LR          = 3e-4
```

### Train

```bash
python train.py
```

Samples are saved to `samples/` every 500 steps. Checkpoints to `checkpoints/`.

---

## Phase 2 — Temporal dynamics

### Data

Any folder of video files (`.mp4`, `.avi`, `.mov`, …). Nested subdirectories are supported.

Set `VIDEO_DIR` in `config_p2.py`:

```python
VIDEO_DIR = r"path/to/your/videos"
```

### Preprocess videos (recommended for long files)

For short clips (< 30 seconds) decord handles random seeking natively — skip this step.

For long recordings (game captures, screencasts), extract frames once:

```bash
# Extract at 6 fps, resize to 128×128 (recommended — allows changing IMAGE_SIZE later)
python preprocess_videos.py --config p2 --fps 6 --size 128

# Or at the exact training resolution (fastest loading, but locked to IMAGE_SIZE=64)
python preprocess_videos.py --config p2 --fps 6 --size 64
```

The script prints the output folder name (e.g. `frames_p2_6fps_128px`). Set it in `config_p2.py`:

```python
FRAMES_DIR = r"frames_p2_6fps_128px"
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `p3` | Which config to read `VIDEO_DIR`/`TRIM_*` from (`p2` or `p3`) |
| `--fps` | `6` | Frames per second to extract |
| `--size` | *(original)* | Resize to `SIZE×SIZE` pixels. Strongly recommended — loading 1080p JPEGs is ~100× slower than 64px ones |
| `--quality` | `4` | JPEG quality (`-q:v` scale: 1=best, 31=worst) |

The script is **resumable** — already-extracted clips are skipped automatically.

**For long H.264 files you can also remux for faster decord seeking** (no re-encode, ~10–30s per file):

```bash
python remux_faststart.py          # remux in-place (renames originals to .bak)
python remux_faststart.py --copy   # write to videos/games_faststart/ instead
```

### Configure

Key settings in `config_p2.py`:

```python
VIDEO_DIR        = r"path/to/videos"
FRAMES_DIR       = r"frames_p2_6fps_128px"   # or None to use decord directly
FRAME_SKIP       = 1     # sample every Nth frame pair; increase for faster videos
VIDEO_WORKERS    = 4
N_HIDDEN_DIM     = 128   # SlotTransition capacity; increase to 256 for better quality
N_LAYERS         = 3     # increase to 6 for better quality
LOAD_P1_WEIGHTS  = True  # transfer O and I from Phase 1 checkpoint
```

### Train

```bash
python train_p2.py
```

Samples are saved to `samples_p2/`. Phase 2 trains for 300 epochs by default.

---

## Phase 3 — Action world model

### Data

Game recordings or any video where a player/agent is controlling something. Long recordings work well.

**Recommended preprocessing:**

```bash
# Rename files to avoid Unicode issues (optional but avoids path problems on Windows)
# Then extract frames:
python preprocess_videos.py --config p3 --fps 6 --size 64

# Set FRAMES_DIR in config_p3.py to the output folder
```

### Configure

Key settings in `config_p3.py`:

```python
VIDEO_DIR        = r"path/to/game/videos"
FRAMES_DIR       = r"frames_p3_6fps_64px"   # output of preprocess_videos.py
FRAME_SKIP       = 10    # 60fps ÷ 10 = 6fps effective; adjust to your video FPS
TRIM_START_SEC   = 120.0 # skip first 2 minutes (intros)
TRIM_END_SEC     = 120.0 # skip last  2 minutes (outros)
N_ACTIONS        = 16    # number of binary action bits
LOAD_P2_WEIGHTS  = True  # transfer O, N, I from Phase 2 checkpoint
```

### Train

```bash
python train_p3.py
```

Every 500 steps, four GIF files are saved to `samples_p3/`:

- `rollout_*_g0.gif` — action bits null, 0–3 (left: canvas, right: reconstructed image)
- `rollout_*_g1.gif` — bits 4–7
- `rollout_*_g2.gif` — bits 8–11
- `rollout_*_g3.gif` — bits 12–15
- `train_*.png` — teacher-forced 2-step reconstruction quality

Training is automatically resumed from `checkpoints_p3/latest_mixed.pth` if it exists.

---

## Interactive engine

Once Phase 3 is trained, run the world model in real time:

```bash
python engine.py                          # random seed frame from dataset
python engine.py --seed my_frame.jpg      # specific starting image
python engine.py --fps 6 --scale 10      # slower, bigger window
```

The window shows **reconstructed image | canvas** side by side.

**Default key bindings** (hold to keep active, press again to toggle off):

| Keys | Bits |
|------|------|
| `1 2 3 4 5 6 7 8` | 0–7 |
| `9 0 q w e r t y` | 8–15 |
| `n` | New random seed |
| `s` | Save current frame as `seed.png` |
| `x` / ESC | Quit |

Edit `KEY_TO_BIT` in `engine.py` to remap to WASD or any other layout once you know which bits do what.

---

## Plotting losses

```python
import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("samples_p3/losses.csv")
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, col in zip(axes.flat, ["r1", "r2", "canvas1", "canvas2", "sparse", "anchor"]):
    ax.plot(df["step"], df[col])
    ax.set_title(col); ax.set_xlabel("step")
plt.tight_layout(); plt.show()
```

---

## Architecture summary

**O — ObjectGenerator** (`model.py`)
CNN encoder: 64×64 → 4 stride-2 stages (32→64→128→256 ch) → FC → `(B, MAX_SLOTS, SLOT_DIM)` raw slot parameters.

**I — ImageReconstructorP2** (`model.py`)
U-Net: takes canvas + previous frame (6 channels) → predicts residual delta → adds to previous frame. Decoder has skip connections from encoder.

**N — SlotTransition** (`model.py`)
Transformer: slot pairs `(params_prev, params_curr)` as queries, image patch tokens as keys/values. 3 cross-attention + self-attention blocks at `hidden_dim=128`. Outputs `(B, MAX_SLOTS, SLOT_DIM)` delta bounded by `tanh * MAX_DELTA`.

**LAM — LatentActionModel** (`model.py`)
Small CNN: 6-channel input `(frame_t, frame_{t+1})` → 16 logits → straight-through binary code `(B, 16)`.

**Rasterizer** (`rasterizer.py`)
Differentiable soft rendering: each slot is rendered as a soft coverage map (circle or line), composited back-to-front using Porter-Duff alpha. Per-slot learnable edge sharpness.

---

## Slot layout (`PRIMITIVE = "mixed"`)

Each of the 32 slots is a 13-dimensional vector:

| Index | Key | Range | Meaning |
|-------|-----|-------|---------|
| 0 | `exists` | [0, 1] | Slot occupancy / opacity mask |
| 1 | `type` | [0, 1] | 0 = circle, 1 = line |
| 2 | `cx` | [0, 1] | Centre x (normalised) |
| 3 | `cy` | [0, 1] | Centre y (normalised) |
| 4 | `p1` | [0, MAX_P1] | Radius (circle) or half-length (line) |
| 5 | `p2` | [0, π] | Line angle |
| 6 | `p3` | [0, MAX_LINE_WIDTH] | Line half-width |
| 7–9 | `r g b` | [0, 1] | Colour |
| 10 | `alpha` | [0, 1] | Transparency |
| 11 | `sharpness` | [SIGMA_MIN, SIGMA_MAX] | Learnable edge softness (per slot) |
| 12 | `depth` | (−∞, +∞) | Compositing order (lower = behind) |
