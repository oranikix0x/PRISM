# ── Phase 2 configuration ────────────────────────────────────────────────────
# Inherits all Phase 1 settings; override only what changes.
from config import *   # noqa: F401, F403

# ── Video dataset ─────────────────────────────────────────────────────────────
# Videos are organised as  videos/train/<category>/<clip>.avi  (or any
# extension).  _find_videos() recurses the full tree so nested folders are
# handled automatically.
#
# FRAME_SKIP — sample every Nth frame pair.  Use 1 for maximum density, or a
# larger value to sub-sample fast-motion videos.
VIDEO_DIR        = r"data/videos"          # set to your video folder
# Pre-extracted JPEG frames (same ones used by Phase 3 — no need to re-extract).
# Set to None to fall back to decoding raw video files via decord (slower).
FRAMES_DIR       = r"data/frames_6fps_128px"   # output of preprocess_videos.py, or None
PRIMITIVE = "mixed"
FRAME_SKIP       = 1          # stride between consecutive frame pairs in JPEG index
MAX_VIDEO_PAIRS  = 20000       # int to cap dataset size, None = use all
VIDEO_WORKERS    = 4           # 0 = main process; avoids Windows cv2 worker deadlocks
# Seconds to skip from the start/end of each video — avoids intros and outros
# without needing to re-encode or trim the files.  0.0 = use the full video.
TRIM_START_SEC   = 0.0
TRIM_END_SEC     = 0.0
# Scene-cut filter: skip frame tuples where any consecutive pair has a mean
# absolute pixel diff above this threshold.  Catches hard cuts (teleports,
# loading screens, YouTube chapter jumps) that would otherwise produce
# misleading LAM signals and potentially long GPU kernels during saves.
# Set to None or 0.0 to disable. default is 0.15
SCENE_CUT_THRESHOLD = 0.0

# ── N model (SlotTransition) ──────────────────────────────────────────────────
N_HIDDEN_DIM  = 256
N_HEADS       = 4
N_LAYERS      = 3
MAX_DELTA     = 0.5   # tanh bound on each delta component

# ── Phase 2 training ─────────────────────────────────────────────────────────
P2_BATCH_SIZE    = 64
P2_NUM_EPOCHS    = 300
P2_LR            = 1e-4
P2_LR_MIN        = 1e-6
P2_WEIGHT_DECAY  = 1e-5

# Same aux-loss weights as Phase 1 (canvas MSE keeps N honest).
P2_CANVAS_AUX_WEIGHT = 0.0  # for O only — keeps slots active and on objects; I never sees RGB canvas
# Small L2 regularisation on N's delta output — encourages smooth/slow motion.
DELTA_REG_WEIGHT     = 0

# Noise added to the conditioning frame during training to simulate imperfect
# rollout context (exposure-bias robustness). Set to 0.0 to disable.
CONTEXT_NOISE_STD    = 0.03
P2_NOISE_STD         = 0.2   # bottleneck noise in I — same as P3 to avoid distribution shift
# Probability of zeroing the frame_t input channels of I per sample during
# training. Forces I to use canvas information rather than the frame shortcut.
# 0.0 = disabled. Recommended: 0.2–0.3.
P2_FRAME_DROP_P      = 0.0

# Weight applied to the step-2 losses in the 2-step unrolled training.
# Step 2 is trained on model-generated context (recon_t) rather than a real
# frame, so it directly closes the train-rollout gap.
# 1.0 = equal weight to both steps; lower if step-2 gradients destabilise training.
MULTISTEP_LOSS_WEIGHT = 1.0

# Anchor loss: after generating recon_t, run O on it and compare the resulting
# params to params_t_hat.  Penalises N for predicting slots that O would not
# produce from a realistic frame — directly combats slot-space drift.
# All dims are normalised to comparable scale before MSE so depth and sharpness
# now contribute correctly (depth → tanh(v/2), sharpness → (v-MIN)/(MAX-MIN)).
# Start low; increase gradually once recon quality stabilises.
# Set to 0.0 to disable.
ANCHOR_LOSS_WEIGHT = 0.0

# Noise added to delta_2 before apply_delta in step 2.  Simulates accumulated
# slot perturbations so N and I learn to tolerate slightly imperfect slot states.
# Set to 0.0 to disable.
DELTA_NOISE_STD = 0.005

# In save_rollout: re-encode recon through O every N steps to reset slot drift.
# 0 = never ground (pure autoregressive); 4 = ground every 4th step.
ROLLOUT_GROUND_EVERY = 0

# ── Paths ─────────────────────────────────────────────────────────────────────
if PRIMITIVE == "mixed":
    P1_CHECKPOINT     = "checkpoints/latest_mixed.pth"
else:
    P1_CHECKPOINT     = "checkpoints/latest.pth"
P2_CHECKPOINT_DIR = "checkpoints_p2"

# Set to False to start Phase 2 with randomly initialised O and I weights
# instead of transferring them from the Phase 1 checkpoint.
# N (SlotTransition) is always randomly initialised — it is new in Phase 2.
LOAD_P1_WEIGHTS = False
P2_SAMPLE_DIR     = "samples_p2"
P2_SAMPLE_EVERY   = 500
P2_CHECKPOINT_EVERY = 312
