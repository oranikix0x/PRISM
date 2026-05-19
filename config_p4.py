# ── Phase 4 configuration ──────────────────────────────────────────────────────
# Inherits all Phase 3b settings (video paths, model sizes, etc.).
# Only Phase-4-specific values are defined here.
from config_p3b import *   # noqa: F401, F403

# ── Multi-agent latent communication ──────────────────────────────────────────
# Dimensionality of z_local and z_global produced by LocalSceneEncoder and
# GlobalAggregator.  Must match the n_world_dim passed to SlotTransition and
# the world_dim passed to ImageReconstructorP2 when constructing models.
P4_LATENT_DIM = 128

# ── Rollout length ─────────────────────────────────────────────────────────────
# Number of generative steps per training example.  Keep short (5-8) so that
# action permutations are approximately commutative — longer sequences increase
# the chance that non-commutative actions (jump→land, trigger→consequence) make
# the two endpoints genuinely different, turning the convergence loss wrong.
# DataLoader seq_len = P4_SEQ_LEN + 1  (seed frame + P4_SEQ_LEN targets).
P4_SEQ_LEN = 6

# ── Convergence loss ──────────────────────────────────────────────────────────
# Weight of MSE(recon_1_T, recon_2_T) at the final step.
# Ramped from 0 → P4_CONVERGENCE_WEIGHT over the first P4_CONVERGENCE_RAMP_STEPS
# training steps so agent 1's reconstruction stabilises before convergence
# pressure is applied (avoids the model collapsing both agents to a blur early).
P4_CONVERGENCE_WEIGHT     = 1.0
P4_CONVERGENCE_RAMP_STEPS = 2000

# ── Training ──────────────────────────────────────────────────────────────────
# Smaller batch than P3b: two full generative unrolls fit in VRAM simultaneously.
P4_BATCH_SIZE    = 4
P4_NUM_EPOCHS    = 100

# LocalSceneEncoder + GlobalAggregator are new → higher LR.
# N and I are fine-tuned from P3b → much lower LR.
P4_LR_NEW        = 3e-4   # LSE and GA
P4_LR_FINETUNE   = 5e-6   # N and I  (10× lower than P3b fine-tune LR)
P4_LR_MIN        = 1e-7
P4_WEIGHT_DECAY  = 1e-5

# ── Paths ──────────────────────────────────────────────────────────────────────
P4_CHECKPOINT_DIR  = "checkpoints_p4"
P4_SAMPLE_DIR      = "samples_p4"
P4_SAMPLE_EVERY    = 500
P4_CHECKPOINT_EVERY = 2000
