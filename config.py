IMAGE_SIZE  = 128       # input/output resolution (square)
MAX_SLOTS   = 32       # number of primitives per image

# ── Primitive type ────────────────────────────────────────────────────────────
# "circle"  — slots are circles only  [exists, cx, cy, radius, r, g, b, alpha, depth]
# "mixed"   — slots can be circles OR lines, selected by a soft learned 'type'
#             field that blends both renderings:
#               type ≈ 0  →  circle  (cx, cy, p1=radius)
#               type ≈ 1  →  line    (cx, cy, p1=half-length, p2=angle, p3=half-width)
#             layout: [exists, type, cx, cy, p1, p2, p3, r, g, b, alpha, depth]
#
# Changing PRIMITIVE requires re-training from scratch (different SLOT_DIM).
PRIMITIVE   = "mixed"   # set to "mixed" to add lines

if PRIMITIVE == "circle":
    SLOT_DIM  = 10
    SLOT_KEYS = ("exists", "cx", "cy", "radius", "r", "g", "b", "alpha", "sharpness", "depth")
else:  # "mixed"
    SLOT_DIM  = 13
    SLOT_KEYS = ("exists", "type", "cx", "cy", "p1", "p2", "p3",
                 "r", "g", "b", "alpha", "sharpness", "depth")

# cx/cy are allowed to exceed [0,1] so that objects can park off-screen and
# re-enter the viewport without being hard-snapped to the edge.
# O always initialises slots in [0,1] (sigmoid activation); N accumulates
# deltas that can push cx/cy into the margin defined here.
POSITION_MIN    = -1.0  # one full viewport to the left / above
POSITION_MAX    =  2.0  # one full viewport to the right / below

MAX_RADIUS      = 0.5   # max circle radius as fraction of image width
MAX_LINE_LENGTH = 0.5   # max half-length of a line  (normalised [0,1])
MAX_LINE_WIDTH  = 0.1   # max half-width of a line   (normalised [0,1])
# MAX_P1: unified p1 bound used for both circle radius and line half-length
MAX_P1          = max(MAX_RADIUS, MAX_LINE_LENGTH)
EDGE_SIGMA_MIN = 0.5 / IMAGE_SIZE   # ~0.5 px  — floor keeps gradients alive
EDGE_SIGMA_MAX = 6.5 / IMAGE_SIZE   # ~6.5 px  — ceiling prevents infinite blur
# Init target: sigmoid(bias) * (MAX-MIN) + MIN ≈ old 1.5px default
# → bias ≈ logit((1.5-0.5)/(6.5-0.5)) = logit(1/6) ≈ -1.6
EDGE_SIGMA_INIT_BIAS = -1.6
BG_COLOR    = 0.0      # background fill value

BATCH_SIZE  = 32
NUM_EPOCHS  = 100
LR          = 3e-4
LR_MIN      = 1e-5
WEIGHT_DECAY = 1e-5

SPARSITY_WEIGHT  = 1e-3  # L1 penalty on exists to encourage empty slots
CAT_WEIGHT        = 1e-3  # L1 penalty on exists*(1-exists) to encourage categorical distribution
CANVAS_AUX_WEIGHT = 0.1   # auxiliary MSE loss directly on canvas vs. input
                           # keeps O honest — prevents it from offloading all
                           # reconstruction work to I

# ── VQ (Latent Action Model) ──────────────────────────────────────────────────
# Multi-codebook VQ replaces binary STE + sparsity loss.
# action code shape = (B, VQ_NUM_CODEBOOKS * VQ_NUM_ENTRIES) — same as N_ACTIONS.
# Representable action combinations = VQ_NUM_ENTRIES ^ VQ_NUM_CODEBOOKS = 4^4 = 256.
VQ_NUM_CODEBOOKS   = 4      # independent action axes (movement, camera, interact, modifier)
VQ_NUM_ENTRIES     = 4      # options per axis
VQ_EMBEDDING_DIM   = 32     # dimension of each codebook entry / encoder output vector
VQ_COMMITMENT_COST = 0.05  # higher early on to break VQ collapse; reduce to 0.05 once vq_uniq > 10   # weight on encoder commitment loss
VQ_EMA_DECAY       = 0.9   # 0.9 → dead entry detected in ~7 steps; 0.99 → ~70 steps; 0.999 → ~700 steps

FLICKR_DIR  = r"data/images"   # set to your image folder
MAX_IMAGES  = 100000      # set to an int (e.g. 5000) to cap dataset size
NUM_WORKERS = 4

CHECKPOINT_DIR  = "checkpoints"
SAMPLE_DIR      = "samples"
SAMPLE_EVERY    = 500   # steps between sample images
CHECKPOINT_EVERY = 2000 # steps between checkpoint saves
