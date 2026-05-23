"""
Interactive World Model Engine — Phase 3 / 4
==============================================
Runs the trained world model in real-time.
Each keyboard key maps to one action bit.
Press keys to "act" in the imagined world.

Single-player controls:
  Hold a,z,e,r,t,y,u,i,o,p,q,s,d,f,g,h  → action bits 0-15 (hold to activate)
  n                   → new random seed
  s                   → save current frame as seed.png
  x / ESC             → quit

Multiplayer mode (--multiplayer):
  Player 1 uses the same key bindings above.
  Player 2 uses number keys 1-9,0 and arrow keys for action bits 0-15.
  Both agents share z_global via LocalSceneEncoder + GlobalAggregator at
  every step.  Two side-by-side windows are displayed (one per player).

Usage:
  python engine.py                          # random seed from dataset
  python engine.py --seed path/to/img.jpg   # specific seed image
  python engine.py --scale 8                # display scale (default 8 → 512×512)
  python engine.py --multiplayer            # two-player mode (Phase 4)
"""

import argparse
import ctypes
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as T

import config
import config_p3 as cfg
import config_p3b as cfgb
from model import (ObjectGenerator, ImageReconstructorP2, SlotTransition,
                   LatentActionModel, LocalSceneEncoder, GlobalAggregator)
from rasterizer import decode_slots, rasterize_from_params, rasterize_flow_from_params
from dataset_video import VideoPairDataset

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--seed",  type=str,   default=None,
                    help="Path to seed image. Omit for random frame from dataset.")
parser.add_argument("--scale", type=int,   default=8,
                    help="Display upscale factor (default 8 → 512×512 at IMAGE_SIZE=64).")
parser.add_argument("--fps",   type=float, default=12.0,
                    help="Target playback speed in steps/second.")
parser.add_argument("--steps-before-ground", type=int, default=0,
                    help="Re-run O every N steps to prevent drift (0 = never).")
parser.add_argument("--multiplayer", action="store_true",
                    help="Run two independent agents sharing z_global (Phase 4).")
parser.add_argument("--observe", action="store_true",
                    help="Observe mode: play a real video clip and show LAM codes live.")
args = parser.parse_args()

# ── Key → bit mapping ─────────────────────────────────────────────────────────
# Customise freely — these are just defaults.
# Keys are OpenCV waitKey codes (ord of the character).
KEY_TO_BIT: dict[int, int] = {
    ord('a'): 0,   ord('z'): 1,   ord('e'): 2,   ord('r'): 3,
    ord('t'): 4,   ord('y'): 5,   ord('u'): 6,   ord('i'): 7,
    ord('o'): 8,   ord('p'): 9,   ord('g'): 10,  ord('h'): 11,
    ord('j'): 12,  ord("k"): 13,  ord('l'): 14,  ord('m'): 15,
}
RESET_KEY  = ord('n')   # n = new seed
SAVE_KEY   = ord('s')
QUIT_KEYS  = {27, ord('x')}   # ESC or x

# ── Hold-to-activate: poll physical key state each frame ──────────────────────
# Windows GetAsyncKeyState returns bit-15 set while the key is physically held.
# This bypasses OS key-repeat entirely — no toggle, no repeat-rate jank.
if sys.platform == "win32":
    _user32 = ctypes.windll.user32
    def _key_held(vk: int) -> bool:
        return bool(_user32.GetAsyncKeyState(vk) & 0x8000)
else:
    # Fallback for non-Windows: always report no key held (use toggle below).
    def _key_held(vk: int) -> bool:
        return False

# OpenCV letter key codes → Windows Virtual Key codes (uppercase VK == ord of uppercase)
_CV_ARROW_TO_VK = {81: 0x25, 82: 0x26, 83: 0x27, 84: 0x28}  # L / U / R / D
_BRACKET_VK     = {ord('['): 0xDB, ord(']'): 0xDD}

def _cv_to_vk(cv_key: int) -> int | None:
    """Convert an OpenCV waitKey code to a Windows VK code, or None if unknown."""
    ch = chr(cv_key) if 0 < cv_key < 128 else None
    if ch and ch.isalpha():
        return ord(ch.upper())
    if ch and ch.isdigit():
        return cv_key          # digit VK == ord('0'-'9')
    if cv_key in _CV_ARROW_TO_VK:
        return _CV_ARROW_TO_VK[cv_key]
    if cv_key in _BRACKET_VK:
        return _BRACKET_VK[cv_key]
    return None

# bit → VK code for polling
BIT_TO_VK:  dict[int, int] = {bit: vk for cv, bit in KEY_TO_BIT.items()
                                if (vk := _cv_to_vk(cv)) is not None}


def _poll_keys_until(deadline: float, special_keys: set[int]) -> int:
    """
    Wait until deadline via short waitKey(1) polls, then sleep any remainder.

    Unlike waitKey(wait_ms), OS key-repeat on held keys does not return early,
    so simulation speed stays independent of which keys are held.
    """
    key = 0
    while time.time() < deadline:
        k = cv2.waitKey(1) & 0xFF
        if k in special_keys:
            key = k
            break
    remaining = deadline - time.time()
    if remaining > 0:
        time.sleep(remaining)
    return key

# ── Setup ─────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

_resize = T.Resize(
    (config.IMAGE_SIZE, config.IMAGE_SIZE),
    interpolation=T.InterpolationMode.BICUBIC,
    antialias=True,
)

def load_image(path: str) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return _resize(t).clamp(0, 1)

def random_seed_frame() -> torch.Tensor:
    ds = VideoPairDataset(
        video_dir  = cfg.VIDEO_DIR,
        frames_dir = getattr(cfg, "FRAMES_DIR", None),
        augment    = False,
    )
    _, _, frame, _ = ds[random.randrange(len(ds))]
    return frame

def to_display(tensor: torch.Tensor, scale: int) -> np.ndarray:
    """(1,3,H,W) float32 → BGR uint8 scaled for display."""
    arr = (tensor[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    return cv2.resize(arr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

# ── Load models ───────────────────────────────────────────────────────────────

import config_p4 as cfg4

_LATENT_DIM = cfg4.P4_LATENT_DIM if args.multiplayer else 0

o_model = ObjectGenerator().to(device)
n_model = SlotTransition(
    hidden_dim  = cfg.N_HIDDEN_DIM,
    n_heads     = cfg.N_HEADS,
    n_layers    = cfg.N_LAYERS,
    max_delta   = cfg.MAX_DELTA,
    n_actions   = cfg.N_ACTIONS,
    n_world_dim = _LATENT_DIM,
).to(device)
i_model = ImageReconstructorP2(world_dim=_LATENT_DIM).to(device)
lam     = LatentActionModel().to(device)

# Phase 4 communication models (only allocated when --multiplayer)
lse = ga = None
if args.multiplayer:
    import config
    lse = LocalSceneEncoder(
        latent_dim = _LATENT_DIM,
        n_actions  = cfg.N_ACTIONS,
        slot_dim   = config.SLOT_DIM,
    ).to(device)
    ga  = GlobalAggregator(latent_dim=_LATENT_DIM).to(device)
ckpt_name = "latest_mixed.pth" if cfg.PRIMITIVE == "mixed" else "latest.pth"

def _find_ckpt(*dirs):
    for d in dirs:
        p = os.path.join(d, ckpt_name)
        if os.path.isfile(p):
            return p
    return None

if args.multiplayer:
    ckpt_path = _find_ckpt(cfg4.P4_CHECKPOINT_DIR, cfgb.P3B_CHECKPOINT_DIR,
                            cfg.P3_CHECKPOINT_DIR)
else:
    ckpt_path = _find_ckpt(cfgb.P3B_CHECKPOINT_DIR, cfg.P3_CHECKPOINT_DIR)

if ckpt_path is None:
    raise FileNotFoundError("No checkpoint found. Train Phase 3 / 3b / 4 first.")

ckpt = torch.load(ckpt_path, map_location=device)
o_model.load_state_dict(ckpt["o_model"])
# strict=False so engine works with both P3b checkpoints (missing world_proj /
# world_adaLN) and P4 checkpoints (has them).
n_model.load_state_dict(ckpt["n_model"], strict=False)
i_model.load_state_dict(ckpt["i_model"], strict=False)
lam.load_state_dict(ckpt["lam"])
if args.multiplayer and "lse" in ckpt and lse is not None:
    lse.load_state_dict(ckpt["lse"])
    ga.load_state_dict(ckpt["ga"])
o_model.eval(); n_model.eval(); i_model.eval(); lam.eval()
if lse is not None:
    lse.eval(); ga.eval()
print(f"Loaded checkpoint from step {ckpt.get('step', '?')}  [{ckpt_path}]")

# ── Engine state ──────────────────────────────────────────────────────────────

def init_state(seed_img: torch.Tensor):
    """Initialise rollout state from a seed image."""
    img = seed_img.unsqueeze(0).to(device)
    with torch.no_grad():
        params = decode_slots(o_model(img))
        canvas = rasterize_from_params(params)
        # True null: LAM(img, img) = "nothing changed" — the real null embedding.
        null_emb, _, _ = lam(img, img)
    return {
        "params_prev": params,
        "params_curr": params,
        "canvas_curr": canvas,
        "prev_recon":  img,
        "emb_prev":    null_emb,
        "step":        0,
    }

print("\nLoading seed frame...")
if args.seed:
    seed_img = load_image(args.seed)
    print(f"  Using: {args.seed}")
else:
    seed_img = random_seed_frame()
    print("  Using random frame from dataset.")

state = init_state(seed_img)

# Pre-compute the null embedding once (LAM on a black frame → "nothing changed").
# Used as the no-key-pressed base for _build_vq_emb and as agent-2's fixed action.
with torch.no_grad():
    _black = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    _z_q_st, null_onehot_cached, _ = lam(_black, _black)
    null_emb_cached = _z_q_st  # (1, C*D) — already the STE embedding

# Second agent state (multiplayer only) — starts from the same seed
state2: dict | None = None
active_bits2: set[int] = set()
if args.multiplayer:
    state2 = init_state(seed_img)
    print("Multiplayer mode: two agents sharing z_global.")

# Player 2 key → bit mapping (number row + arrow keys)
KEY_TO_BIT_P2: dict[int, int] = {
    ord('1'): 0,  ord('2'): 1,  ord('3'): 2,  ord('4'): 3,
    ord('5'): 4,  ord('6'): 5,  ord('7'): 6,  ord('8'): 7,
    ord('9'): 8,  ord('0'): 9,
    82: 10,  # OpenCV up-arrow
    84: 11,  # down-arrow
    81: 12,  # left-arrow
    83: 13,  # right-arrow
    ord('['): 14,  ord(']'): 15,
}
BIT_TO_VK_P2: dict[int, int] = {bit: vk for cv, bit in KEY_TO_BIT_P2.items()
                                  if (vk := _cv_to_vk(cv)) is not None}

# ── Observe mode ──────────────────────────────────────────────────────────────

if args.observe:
    import glob as _glob
    import json as _json
    import random as _random
    from collections import defaultdict
    from PIL import Image as _PilImage

    WIN_OBS = "OIG Observe  [space=next clip | s=save atlas | x=quit]"
    cv2.namedWindow(WIN_OBS, cv2.WINDOW_NORMAL)

    E_obs = cfg.VQ_NUM_ENTRIES
    C_obs = cfg.VQ_NUM_CODEBOOKS
    sq, gap = 12, 4
    colours = [(80,80,200),(80,200,80),(200,80,80),(200,200,80)]

    # ── Code atlas accumulator ────────────────────────────────────────────────
    # key: tuple of winners e.g. (2, 0, 3, 1)
    # value: list of (prev_np_rgb, curr_np_rgb) uint8 arrays, capped at MAX_PER_CODE
    MAX_PER_CODE = 6
    code_buckets: dict[tuple, list] = defaultdict(list)
    code_counts:  dict[tuple, int]  = defaultdict(int)

    def _code_label(winners: list[int]) -> str:
        return "_".join(f"cb{c}{w}" for c, w in enumerate(winners))

    def _tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
        """(1,3,H,W) float [0,1] → (H,W,3) uint8."""
        return (t[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    def _make_diff(prev_rgb: np.ndarray, curr_rgb: np.ndarray,
                   amplify: float = 5.0) -> np.ndarray:
        """Amplified absolute difference — bright pixels = large motion/change."""
        diff = np.abs(curr_rgb.astype(np.float32) - prev_rgb.astype(np.float32))
        diff = (diff * amplify).clip(0, 255).astype(np.uint8)
        return diff

    def _save_atlas(out_path: str = "code_atlas.png") -> None:
        if not code_counts:
            print("[observe] No codes collected yet.")
            return
        # Sort by frequency descending
        sorted_codes = sorted(code_counts.items(), key=lambda x: -x[1])
        H   = config.IMAGE_SIZE
        pad = 4
        label_w  = 90        # text label column
        triplet_w = 3 * H + 2 * pad   # prev | diff | next
        cols     = MAX_PER_CODE
        cell_w   = triplet_w + pad
        img_w    = label_w + cols * cell_w
        img_h    = len(sorted_codes) * (H + pad) + pad

        atlas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        for row_i, (code_key, count) in enumerate(sorted_codes):
            y0 = pad + row_i * (H + pad)
            # Label: code digits + count
            label = f"{''.join(str(w) for w in code_key)} x{count}"
            cv2.putText(atlas, label, (4, y0 + H // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            # Triplets: prev | diff×5 | next
            for col_i, (prev_rgb, curr_rgb) in enumerate(code_buckets[code_key][:cols]):
                x0       = label_w + col_i * cell_w
                prev_bgr = cv2.cvtColor(cv2.resize(prev_rgb, (H, H)), cv2.COLOR_RGB2BGR)
                diff_bgr = cv2.cvtColor(cv2.resize(_make_diff(prev_rgb, curr_rgb), (H, H)),
                                        cv2.COLOR_RGB2BGR)
                curr_bgr = cv2.cvtColor(cv2.resize(curr_rgb, (H, H)), cv2.COLOR_RGB2BGR)
                atlas[y0:y0+H, x0         : x0+H]           = prev_bgr
                atlas[y0:y0+H, x0+H+pad   : x0+2*H+pad]     = diff_bgr
                atlas[y0:y0+H, x0+2*H+2*pad : x0+triplet_w] = curr_bgr

        cv2.imwrite(out_path, atlas)

        # Save machine-readable metadata for save_rollout to consume
        json_path = out_path.replace(".png", ".json")
        meta = {
            "num_codebooks": C_obs,
            "num_entries":   E_obs,
            "codes": [
                {"winners": list(code_key), "count": count}
                for code_key, count in sorted_codes
            ],
        }
        with open(json_path, "w") as _f:
            _json.dump(meta, _f, indent=2)

        print(f"[observe] Atlas saved → {out_path}  "
              f"({len(sorted_codes)} unique codes, "
              f"{sum(code_counts.values())} total transitions)\n"
              f"         Metadata  → {json_path}")

    def _draw_hud(img_bgr, winners, n_unique):
        code_str = "  ".join(f"cb{c}:{w}" for c, w in enumerate(winners))
        cv2.putText(img_bgr, f"{code_str}   [{n_unique} unique]", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)
        for c, w in enumerate(winners):
            for e in range(E_obs):
                x = 8 + e * (sq + gap)
                y = 24 + c * (sq + gap)
                col = colours[c] if e == w else (40, 40, 40)
                cv2.rectangle(img_bgr, (x, y), (x + sq, y + sq), col, -1)

    frame_interval_obs = 1.0 / args.fps
    frames_dir = cfg.FRAMES_DIR
    clip_dirs  = sorted(_glob.glob(str(frames_dir) + "/*/"))
    if not clip_dirs:
        clip_dirs = [str(frames_dir)]

    with torch.no_grad():
        while True:
            clip  = _random.choice(clip_dirs)
            jpegs = sorted(_glob.glob(clip + "*.jpg") + _glob.glob(clip + "*.jpeg"))
            if len(jpegs) < 2:
                continue
            print(f"[observe] clip: {clip}  ({len(jpegs)} frames)")
            prev_frame    = None
            prev_frame_np = None

            for path in jpegs:
                t0    = time.time()
                frame = load_image(path).unsqueeze(0).to(device)
                frame_np = _tensor_to_rgb(frame)

                if prev_frame is not None:
                    _, code_oh, _ = lam(prev_frame, frame)
                    winners = tuple(
                        int(code_oh[0, c * E_obs:(c + 1) * E_obs].argmax())
                        for c in range(C_obs)
                    )
                    code_counts[winners] += 1
                    if len(code_buckets[winners]) < MAX_PER_CODE:
                        code_buckets[winners].append((prev_frame_np, frame_np))
                else:
                    winners = tuple([0] * C_obs)

                disp = to_display(frame, args.scale)
                _draw_hud(disp, list(winners), len(code_counts))
                cv2.imshow(WIN_OBS, disp)

                prev_frame    = frame
                prev_frame_np = frame_np
                key = _poll_keys_until(
                    t0 + frame_interval_obs,
                    {ord('x'), ord('s'), ord(' ')},
                )
                if key == ord('x'):
                    _save_atlas()
                    cv2.destroyAllWindows()
                    raise SystemExit(0)
                if key == ord('s'):
                    _save_atlas()
                if key == ord(' '):
                    break  # next random clip

    cv2.destroyAllWindows()
    raise SystemExit(0)

# ── Display loop ──────────────────────────────────────────────────────────────

WIN  = "OIG Engine  [a/z/e/r... = bits | n = new seed | s = save | x = quit]"
WIN2 = "OIG Engine P2  [1-9/0/arrows = bits]"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
if args.multiplayer:
    cv2.namedWindow(WIN2, cv2.WINDOW_NORMAL)

active_bits: set[int] = set()
frame_interval = 1.0 / args.fps

print(f"\nRunning at target {args.fps} fps  |  display {config.IMAGE_SIZE * args.scale}×"
      f"{config.IMAGE_SIZE * args.scale}  |  bits: {cfg.N_ACTIONS}")
if args.multiplayer:
    print("Player 1: a/z/e/r/t/y/u/i/o/p/q/s/d/f/g/h keys")
    print("Player 2: 1-9/0/arrow keys")
else:
    print("Hold letter keys to activate action bits.")
print()

with torch.no_grad():
    while True:
        t_start = time.time()

        # ── Build action embeddings (VQ: one-hot → codebook lookup) ──────────
        # True null: lam(prev_recon, prev_recon) — "nothing changed" for this frame.
        # Recomputed each step so the base tracks the current frame distribution.
        _, null_onehot_step, _ = lam(state["prev_recon"], state["prev_recon"])
        null_emb_step = lam.code_to_embedding(null_onehot_step)  # (1, C*D)

        def _build_vq_emb(bits: set[int]) -> tuple[torch.Tensor, torch.Tensor]:
            """Returns (embedding, onehot) for the action to inject this step."""
            if not bits:
                return null_emb_step, null_onehot_step
            onehot = null_onehot_step.clone()          # (1, C*E) per-step null base
            E = cfg.VQ_NUM_ENTRIES
            for bit in bits:
                if bit < cfg.VQ_NUM_CODEBOOKS * E:
                    cb_idx = bit // E
                    entry  = bit  % E
                    base   = cb_idx * E
                    onehot[0, base:base + E] = 0.0
                    onehot[0, base + entry]  = 1.0
            return lam.code_to_embedding(onehot), onehot  # (1, C*D), (1, C*E)

        emb, action_onehot = _build_vq_emb(active_bits)
        # Agent 2 always receives the cached black-frame null — no key input.
        emb2: torch.Tensor | None = null_emb_cached if args.multiplayer else None

        # ── Compute z_global (multiplayer only) ───────────────────────────────
        world_emb: torch.Tensor | None = None
        if args.multiplayer and lse is not None and ga is not None:
            p1_t = SlotTransition.params_to_tensor(state["params_curr"])
            p2_t = SlotTransition.params_to_tensor(state2["params_curr"])
            z1   = lse(state["prev_recon"],  p1_t, emb)
            z2   = lse(state2["prev_recon"], p2_t, emb2)
            world_emb = ga([z1, z2])

        # ── One world-model step (agent 1) ────────────────────────────────────
        delta       = n_model(state["params_prev"], state["params_curr"],
                              state["prev_recon"], emb, state["emb_prev"],
                              world_emb=world_emb)
        params_next = SlotTransition.apply_delta(state["params_curr"], delta)
        canvas      = rasterize_from_params(params_next)
        flow        = rasterize_flow_from_params(state["params_curr"], params_next)
        recon       = i_model(flow, state["prev_recon"],
                              action_emb=emb, world_emb=world_emb)

        state["params_prev"] = state["params_curr"]
        state["params_curr"] = params_next
        state["canvas_curr"] = canvas
        state["prev_recon"]  = recon
        state["emb_prev"]    = emb
        state["step"]       += 1

        # Optional grounding (agent 1)
        if (args.steps_before_ground > 0
                and state["step"] % args.steps_before_ground == 0):
            state["params_curr"] = decode_slots(o_model(recon))
            state["canvas_curr"] = rasterize_from_params(state["params_curr"])

        # ── One world-model step (agent 2, multiplayer only) ──────────────────
        recon2 = canvas2 = None
        if args.multiplayer and state2 is not None:
            delta2       = n_model(state2["params_prev"], state2["params_curr"],
                                   state2["prev_recon"], emb2, state2["emb_prev"],
                                   world_emb=world_emb)
            params_next2 = SlotTransition.apply_delta(state2["params_curr"], delta2)
            canvas2      = rasterize_from_params(params_next2)
            flow2        = rasterize_flow_from_params(state2["params_curr"], params_next2)
            recon2       = i_model(flow2, state2["prev_recon"],
                                   action_emb=emb2, world_emb=world_emb)

            state2["params_prev"] = state2["params_curr"]
            state2["params_curr"] = params_next2
            state2["canvas_curr"] = canvas2
            state2["prev_recon"]  = recon2
            state2["emb_prev"]    = emb2
            state2["step"]       += 1

            if (args.steps_before_ground > 0
                    and state2["step"] % args.steps_before_ground == 0):
                state2["params_curr"] = decode_slots(o_model(recon2))
                state2["canvas_curr"] = rasterize_from_params(state2["params_curr"])

        # ── Display ───────────────────────────────────────────────────────────
        E = cfg.VQ_NUM_ENTRIES
        C = cfg.VQ_NUM_CODEBOOKS
        # Winners from the action onehot actually injected this step (not a probe).
        winners = [int(action_onehot[0, c * E:(c + 1) * E].argmax()) for c in range(C)]

        recon_disp  = to_display(recon,  args.scale)
        canvas_disp = to_display(canvas, args.scale)
        separator   = np.zeros((recon_disp.shape[0], 4, 3), dtype=np.uint8)
        combined    = np.concatenate([recon_disp, separator, canvas_disp], axis=1)

        # Step info on the recon (left) side
        bit_str = " ".join(str(b) for b in sorted(active_bits)) or "—"
        cv2.putText(combined, f"bits:{bit_str}  step:{state['step']}",
                    (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Code HUD: 4×4 grid on the canvas (right) side
        # Row = codebook, Column = entry; lit square = active code for that codebook.
        sq, gap = 12, 4
        colours = [(80,80,200),(80,200,80),(200,80,80),(200,200,80)]
        canvas_x0 = recon_disp.shape[1] + separator.shape[1] + 8
        canvas_y0 = 8
        for c, w in enumerate(winners):
            for e in range(E):
                x = canvas_x0 + e * (sq + gap)
                y = canvas_y0 + c * (sq + gap)
                col = colours[c] if e == w else (40, 40, 40)
                cv2.rectangle(combined, (x, y), (x + sq, y + sq), col, -1)
        cv2.imshow(WIN, combined)

        if args.multiplayer and recon2 is not None and canvas2 is not None:
            r2_disp = to_display(recon2,  args.scale)
            c2_disp = to_display(canvas2, args.scale)
            combined2 = np.concatenate([r2_disp, separator, c2_disp], axis=1)
            cv2.putText(combined2, f"P2 (null action)  step: {state2['step']}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            cv2.imshow(WIN2, combined2)

        # ── Key handling ──────────────────────────────────────────────────────
        key = _poll_keys_until(
            t_start + frame_interval,
            QUIT_KEYS | {RESET_KEY, SAVE_KEY},
        )

        if key in QUIT_KEYS:
            break
        elif key == RESET_KEY:
            print("New random seed...")
            seed_img = random_seed_frame()
            state = init_state(seed_img)
            if args.multiplayer:
                state2 = init_state(seed_img)
        elif key == SAVE_KEY:
            cv2.imwrite("seed.png", to_display(recon, 1))
            print("Saved current frame as seed.png")

        # Poll which action keys are physically held right now.
        prev_bits  = active_bits
        active_bits  = {bit for bit, vk in BIT_TO_VK.items() if _key_held(vk)}
        # Agent 2 has no key input — null action only, coupled via world_emb.
        if active_bits != prev_bits:
            print(f"  P1 active bits: {sorted(active_bits) or '—'}")

cv2.destroyAllWindows()
print("Done.")
