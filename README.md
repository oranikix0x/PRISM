# PRISM — Primitive Rendering for Interactive Scene Modeling

A research project building a controllable game world model from unlabelled video, using differentiable object-centric representations and unsupervised action discovery.

## Overview

| Phase | What it learns | Key models |
|-------|---------------|------------|
| **1 — Static** | Decompose images into geometric primitives | O, I |
| **2 — Temporal** | How primitives evolve over time | O, N, I |
| **3 — Controllable** | Unsupervised action codes from frame pairs | O, N, I, LAM |
| **3b — Long-horizon** | Stable multi-step rollouts | O, N, I, LAM (unrolled) |
| **4 — Multi-agent** *(in progress)* | Consistent shared world state across two independent agents | O, N, I, LAM + LSE + GA |
| **5 — Language** *(planned)* | Text-conditioned world editing | O, N, I, LAM + language encoder |

### Core idea

Rather than predicting pixels directly, PRISM decomposes each frame into a set of soft geometric primitives (circles and line segments). A differentiable rasterizer renders these into an optical flow field. A UNet image reconstructor warps the previous frame by that flow and refines the result — separating *geometric structure* (primitives → flow) from *appearance* (warp + refinement). This inductive bias keeps rollouts geometrically coherent across many steps.

Action codes are discovered entirely without supervision: a Latent Action Model (LAM) encodes consecutive frame pairs into a discrete VQ code, which conditions the slot transition model N. Backpropagating reconstruction loss through the straight-through estimator teaches LAM to extract semantically meaningful action axes.

---

## Requirements

```bash
pip install torch torchvision numpy opencv-python Pillow tqdm decord lpips
```

CUDA GPU required for reasonable training speed. Developed with PyTorch 2.x on Windows.

---

## Repository structure

```
config.py              # Phase 1 hyperparameters (shared base)
config_p2.py           # Phase 2 overrides
config_p3.py           # Phase 3 overrides
config_p3b.py          # Phase 3b overrides
model.py               # All neural network architectures (O, I, N, LAM, VQCodebook)
rasterizer.py          # Differentiable soft primitive renderer + optical flow
dataset.py             # Image dataset (Phase 1)
dataset_video.py       # Video frame dataset (JPEG / decord backends)
train.py               # Phase 1 training
train_p2.py            # Phase 2 training
train_p3.py            # Phase 3 training
train_p3b.py           # Phase 3b training (long-horizon unrolling)
preprocess_videos.py   # Extract JPEG frames from videos (run once before Phase 2/3)
remux_faststart.py     # Remux MP4s with faststart header (faster decord seeking)
engine.py              # Interactive inference engine + observe/atlas mode
loss.ipynb             # Notebook to plot training losses
```

---

## Phase 1 — Static image reconstruction

**Goal:** Learn to represent images as sets of soft geometric primitives.

### Data

Images from any folder (e.g. [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)).

```python
# config.py
FLICKR_DIR = r"path/to/images"
IMAGE_SIZE  = 128      # resolution (square)
MAX_SLOTS   = 32       # number of primitives per image
PRIMITIVE   = "mixed"  # "circle" or "mixed" (circles + lines)
```

### Train

```bash
python train.py
```

Samples → `samples/`.  Checkpoints → `checkpoints/`.

---

## Phase 2 — Temporal dynamics

**Goal:** Learn how primitive slots evolve between consecutive frames.

### Data

Any folder of video files (`.mp4`, `.avi`, `.mov`, …).

```bash
# Extract frames once (recommended for long recordings)
python preprocess_videos.py --config p2 --fps 6 --size 128
```

```python
# config_p2.py
VIDEO_DIR  = r"path/to/videos"
FRAMES_DIR = r"frames_p2_6fps_128px"   # output of preprocess_videos.py
```

### Train

```bash
python train_p2.py
```

Samples → `samples_p2/`. Loads Phase 1 weights automatically (`LOAD_P1_WEIGHTS = True`).

---

## Phase 3 — Action world model

**Goal:** Discover discrete action codes from unlabelled video and learn to predict future frames conditioned on them.

The Latent Action Model (LAM) encodes `(frame_t, frame_{t+1})` into a **multi-codebook VQ code** — 4 independent codebooks × 4 entries each = 256 possible action combinations. The slot transition model N is conditioned on the VQ embedding via straight-through estimator, so reconstruction gradients flow back to teach LAM what actions matter.

### Data

Game recordings or any controlled-agent video.

```bash
python preprocess_videos.py --config p3 --fps 6 --size 128
```

```python
# config_p3.py
VIDEO_DIR  = r"path/to/game/videos"
FRAMES_DIR = r"frames_p3_6fps_128px"

# Freeze components to speed up training (optional)
FREEZE_O   = False   # O is fast to run frozen
FREEZE_LAM = False   # freeze once LAM has converged
```

### Train

```bash
python train_p3.py
python train_p3.py --no-output   # skip GIF saving (avoids CUDA TDR on Windows)
```

Samples → `samples_p3/`:
- `rollout_*_g0.gif` … `g2.gif` — action-code rollouts (canvas | reconstructed image)
- `train_*.png` — teacher-forced 2-step reconstruction quality

### Observe + code atlas

```bash
python engine.py --observe          # play real video clips, show LAM codes live
```

While observing:
- `s` — save `code_atlas.png` (grid: each row = one code, columns = prev|diff×5|next pairs) + `code_atlas.json`
- `space` — next random clip
- `x` — quit (auto-saves atlas)

Once `code_atlas.json` exists, rollout GIFs automatically use the most frequent real gameplay codes instead of all theoretical combinations.

---

## Phase 3b — Long-horizon fine-tuning

**Goal:** Stable multi-step rollouts that don't drift. N and I are fine-tuned on unrolled sequences (10+ steps) while O stays frozen.

```python
# config_p3b.py
P3B_UNROLL_STEPS  = 10   # autoregressive steps per training sequence
P3B_TRUNCATE_BPTT = 5    # detach graph every N steps to bound VRAM
FREEZE_O          = True  # O doesn't benefit from longer horizons
```

```bash
python train_p3b.py
python train_p3b.py --no-output
```

Samples → `samples_p3b/`:
- `rollout_*_g*.gif` — action-code rollouts over 8+ steps
- `diversity_rollout_*.gif` — same null action, different noise seeds (shows stochasticity)

---

## Phase 4 — Multi-agent *(in progress)*

**Goal:** Two independent agents sharing a common world state. Each agent has its own view and action history; a shared global embedding makes their worlds consistent when they occupy the same space.

Architecture addition:
- **LSE (Local State Encoder)** — encodes each agent's `(frame, slots, action)` into a local latent
- **GA (Global Aggregator)** — attention-pools the two local latents into a shared world embedding
- N and I are conditioned on `world_emb` in addition to slots and action

Training: **convergence loss** — two agents given the same starting frame and shuffled identical action sequences must produce the same final frame, enforcing world-state consistency.

The interactive engine supports multiplayer mode today:
```bash
python engine.py --multiplayer
```

---

## Phase 5 — Language conditioning *(planned)*

**Goal:** A "god mode" chat box: type anything and it materialises in the game world (*"place a bridge over this river"*, *"switch to first-person view"*).

Approach: every N batches, inject a training sample of `(before_frame, after_frame, caption)` describing the visual change. A language encoder (frozen CLIP or small trained embedding) conditions N and I, so the world model learns to associate text descriptions with visual transitions.

Key design choice: the language embedding is trained end-to-end within the existing pipeline — no separate inpainting or diffusion model needed. The same N and I that handle action codes also handle language codes, unified through the world embedding pathway.

---

## Interactive engine

```bash
python engine.py                     # normal mode — hold keys to fire action codes
python engine.py --seed frame.jpg    # specific starting image
python engine.py --fps 6 --scale 8  # slower, bigger window
python engine.py --observe           # observe mode — watch real video + code HUD
python engine.py --multiplayer       # two agents sharing world state
```

The window shows **reconstructed image | canvas** side by side.
Canvas side shows a live 4×4 HUD: rows = codebooks, columns = entries, lit square = active code.

**Key bindings (normal mode):**

| Keys | Action |
|------|--------|
| Letter/number keys | Hold to activate a specific codebook entry |
| `n` | New random seed frame |
| `s` | Save current frame |
| `x` / ESC | Quit |

Remap `KEY_TO_BIT` in `engine.py` to WASD or any layout once you've used the code atlas to identify which codes correspond to which in-game actions.

---

## Architecture

**O — ObjectGenerator** (`model.py`)

CNN encoder (4 stride-2 stages, 3→32→64→128→256 channels) → AdaptiveAvgPool(4×4) → MLP → `(B, MAX_SLOTS, SLOT_DIM)` raw slot parameters. Resolution-agnostic: weights trained at any resolution transfer to any other.

**N — SlotTransition** (`model.py`)

Transformer operating on slot pairs `(params_prev, params_curr)` as queries, image patch tokens as cross-attention keys/values. Conditioned on the VQ action embedding (`z_q_st`, 128-dim) and optionally a world embedding. Outputs `(B, MAX_SLOTS, SLOT_DIM)` delta bounded by `tanh × MAX_DELTA`.

**I — ImageReconstructorP2** (`model.py`)

Warp-then-refine UNet. Takes `[flow(2ch), warped_frame(3ch)]` as input:
1. Warp `frame_t` by the predicted optical flow (bilinear backward warp)
2. UNet refines the warp artifact → residual delta
3. Output = `clamp(warped + delta, 0, 1)`

Also conditioned on the VQ action embedding for non-spatial effects (colour changes, UI elements, occlusion).

**LAM — LatentActionModel** (`model.py`)

6-channel CNN encoder `(frame_t ∥ frame_{t+1})` → GlobalAvgPool → MLP → **multi-codebook VQ**. C=4 codebooks × E=4 entries × D=32-dim embeddings. Returns:
- `z_q_st` (B, C×D) — quantized embedding with straight-through gradient, passed to N
- `action_code` (B, C×E) — one-hot code, used for logging and engine HUD only

**Rasterizer** (`rasterizer.py`)

Differentiable soft rendering: each slot rendered as a soft coverage map, composited back-to-front (Porter-Duff). Also computes per-pixel optical flow from slot motion between `params_curr` and `params_next`.

---

## Slot layout (`PRIMITIVE = "mixed"`)

Each of the MAX_SLOTS slots is a SLOT_DIM-dimensional vector:

| Index | Key | Range | Meaning |
|-------|-----|-------|---------|
| 0 | `exists` | [0, 1] | Slot occupancy / opacity |
| 1 | `type` | [0, 1] | 0 = circle, 1 = line |
| 2 | `cx` | [0, 1] | Centre x (normalised) |
| 3 | `cy` | [0, 1] | Centre y (normalised) |
| 4 | `p1` | [0, MAX_P1] | Radius (circle) or half-length (line) |
| 5 | `p2` | [0, π] | Line angle |
| 6 | `p3` | [0, MAX_LINE_WIDTH] | Line half-width |
| 7–9 | `r g b` | [0, 1] | Colour |
| 10 | `alpha` | [0, 1] | Transparency |
| 11 | `sharpness` | [σ_min, σ_max] | Learnable edge softness |
| 12 | `depth` | (−∞, +∞) | Compositing order (lower = behind) |
