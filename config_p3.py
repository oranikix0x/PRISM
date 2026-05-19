# ── Phase 3 configuration ────────────────────────────────────────────────────
# Inherits all Phase 2 settings; override only what changes.
from config_p2 import *   # noqa: F401, F403

# ── Latent Action Model ───────────────────────────────────────────────────────
# N receives z_q_st (the quantized embedding with STE gradient), NOT the one-hot
# action code.  This is the correct VQ-VAE architecture: reconstruction gradients
# flow back to the LAM encoder through z_q_st → STE → z → encoder.
# N's action_proj therefore takes C*D = 128 dims (not C*E = 16).
import config as _cfg
N_ACTIONS = _cfg.VQ_NUM_CODEBOOKS * _cfg.VQ_EMBEDDING_DIM  # = 128 (4 codebooks × 32-dim embedding)

# ── Stochastic reconstruction (noise + LPIPS) ─────────────────────────────────
P3_NOISE_STD    = 0.2
P3_LPIPS_WEIGHT = 0.02

# ── Phase 3 training ─────────────────────────────────────────────────────────
P3_BATCH_SIZE   = 16
P3_NUM_EPOCHS   = 500
P3_LR           = 5e-5    # lower than P2 — O, N, I are already trained
P3_LR_MIN       = 1e-6
P3_WEIGHT_DECAY = 1e-5
MAX_VIDEO_PAIRS = 20000

# Whether to load O, N, I weights from the Phase 2 checkpoint.
# Set to False to train Phase 3 from scratch (not recommended).
LOAD_P2_WEIGHTS = True

# ── Freeze individual models to speed up training ─────────────────────────────
# Frozen models are not updated by the optimizer (gradients disabled).
# Useful when one component is already converged or to reduce GPU load.
FREEZE_O   = False
FREEZE_N   = False
FREEZE_I   = False
FREEZE_LAM = False

# Re-seed VQ codebook from the first training batch's encoder outputs.
# Set to True once after a resolution change (e.g. 64→128px) to fix
# codebook collapse caused by the distribution shift.  Reset to False
# after the first successful run so subsequent resumes don't re-seed.
RESET_VQ_CODEBOOK = False

# ── Paths ─────────────────────────────────────────────────────────────────────
VIDEO_DIR        = r"C:\Projects\Personal\OIG\videos\games\minecraft"
VIDEO_WORKERS    = 4
FRAME_SKIP       = 10          # 60fps ÷ 10 = 6fps effective — matches motion visible to humans
TRIM_START_SEC   = 120.0
TRIM_END_SEC     = 120.0
# Pre-extracted JPEG frames directory (fast mode).
# Run:  python preprocess_videos.py --fps 6
# Set to None to use decord directly on the raw video files (slower for long H.264).
FRAMES_DIR = r"C:\Projects\Personal\OIG\frames_p3_6fps_128px"
if PRIMITIVE == "mixed":
    P2_CHECKPOINT     = "checkpoints_p2/latest_mixed.pth"
else:
    P2_CHECKPOINT     = "checkpoints_p2/latest.pth"
P3_CHECKPOINT_DIR   = "checkpoints_p3"
P3_SAMPLE_DIR       = "samples_p3"
P3_SAMPLE_EVERY     = 200
P3_CHECKPOINT_EVERY = 300
