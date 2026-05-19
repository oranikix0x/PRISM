# ── Phase 3b configuration ────────────────────────────────────────────────────
# Inherits all Phase 3 settings; override only what changes.
from config_p3 import *   # noqa: F401, F403

# ── Unrolling ─────────────────────────────────────────────────────────────────
# Number of autoregressive prediction steps (beyond the 2 ground-truth context
# frames).  seq_len fed to the dataset = P3B_UNROLL_STEPS + 2.
P3B_UNROLL_STEPS  = 10

# Detach all state tensors every N unroll steps to keep the live computation
# graph bounded in VRAM.  Memory cost ≈ N × single-step cost.
# 0 = full BPTT (best gradient quality, but O(T) memory — use only for small T).
P3B_TRUNCATE_BPTT = 5

# ── Frozen models ─────────────────────────────────────────────────────────────
# O encodes single frames independently; LAM is used only to provide ground-
# truth action codes from real frame pairs.  Neither benefits from longer
# unrolling — keep them frozen so only N and I are updated.
FREEZE_O   = True
FREEZE_LAM = False

# ── Training ──────────────────────────────────────────────────────────────────
# Batch size is much smaller than P3 because up to P3B_TRUNCATE_BPTT forward
# passes' worth of computation graphs live in VRAM simultaneously.
# Probability of zeroing a slot's exists in params_curr each step.
# Simulates temporary occlusion/disappearance: N must re-derive the slot's
# state from params_prev (last known position) and the image context.
# ~10% → ~3 slots dropped per item per step out of MAX_SLOTS=32.
P3B_SLOT_DROPOUT = 0.1

# ── Loss weights ──────────────────────────────────────────────────────────────
# Canvas is a structural scaffold for I, not a full scene reconstruction.
# A much lower weight lets N focus objects on what matters (salient structure)
# rather than trying to match every pixel of the background.
P3B_CANVAS_AUX_WEIGHT = 0.01   # 10× lower than P3's 0.05

# ── Stochastic reconstruction (noise + LPIPS) ─────────────────────────────────
# Gaussian noise std injected at I's bottleneck during training.
# Gives I the freedom to commit to one sharp texture instead of blurring.
# Paired with LPIPS loss which rewards sharpness over blurry MSE averages.
P3B_NOISE_STD    = 0.1
P3B_LPIPS_WEIGHT = 0.1

P3B_BATCH_SIZE   = 6
P3B_NUM_EPOCHS   = 100
P3B_LR           = 1e-5    # fine-tuning: 5× lower than P3
P3B_LR_MIN       = 1e-7
P3B_WEIGHT_DECAY = 1e-5

# ── Paths ─────────────────────────────────────────────────────────────────────
P3B_CHECKPOINT_DIR = "checkpoints_p3b"
P3B_SAMPLE_DIR     = "samples_p3b"
P3B_SAMPLE_EVERY     = 200
P3B_CHECKPOINT_EVERY = 2000
