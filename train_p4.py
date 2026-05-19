"""
Phase 4 — Multiplayer Latent Communication
============================================
Fine-tunes N and I from a Phase 3b checkpoint by training two agents
simultaneously, each running generative rollouts with the SAME action codes
in a DIFFERENT ORDER.

Because small game actions are approximately commutative (left then right ≈
right then left), both agents arrive at approximately the same world state
after P4_SEQ_LEN steps despite taking different intermediate paths.  The
convergence loss MSE(recon_1_T, recon_2_T) forces the models to produce
identical reconstructions at the shared endpoint.  The only communication
channel between agents is z_global produced by LocalSceneEncoder +
GlobalAggregator, so the gradient MUST flow through that channel — teaching
z_global to encode a compact shared world-state representation.

Training loop:
──────────────────────────────────────────────────────────────────────────────
  Precompute all T action codes via LAM on real consecutive frames.
  Shuffle action codes per batch item → agent 2's permuted sequence.

  Both agents share the same starting state: params = O(frames[0]).

  Generative unroll for t = 0 … T-1:
    z_1     = LSE(prev_recon_1, params_1, code_1_t)
    z_2     = LSE(prev_recon_2, params_2, code_2_t)
    z_global = GA([z_1, z_2])

    delta_1  = N(params_prev_1, params_curr_1, prev_recon_1, code_1_t, world_emb=z_global)
    recon_1  = I(canvas_curr_1, canvas_next_1, prev_recon_1,           world_emb=z_global)
    loss_recon += MSE(recon_1, frames[t+1])   ← agent 1 has ground truth

    (mirror for agent 2 — no per-step ground truth)

  Convergence loss at final step T:
    lambda_c = ramp(step) * P4_CONVERGENCE_WEIGHT
    loss_conv = MSE(recon_1_T, recon_2_T)

  total_loss = loss_recon + lambda_c * loss_conv
──────────────────────────────────────────────────────────────────────────────

Frozen: O, LAM.
Fine-tuned: N, I  (loaded from P3b checkpoint, low LR).
Trained from scratch: LocalSceneEncoder, GlobalAggregator (higher LR).
"""

import os
import threading

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import config
import config_p4 as cfg
from model import (
    ObjectGenerator,
    ImageReconstructorP2,
    SlotTransition,
    LatentActionModel,
    LocalSceneEncoder,
    GlobalAggregator,
)
from rasterizer import decode_slots, rasterize_from_params
from dataset_video import PermutedActionDataset, _worker_init_fn


# ── Device ────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Phase 4 training on {device}")

# ── Models ────────────────────────────────────────────────────────────────────

o_model = ObjectGenerator().to(device)
n_model = SlotTransition(
    hidden_dim  = cfg.N_HIDDEN_DIM,
    n_heads     = cfg.N_HEADS,
    n_layers    = cfg.N_LAYERS,
    max_delta   = cfg.MAX_DELTA,
    n_actions   = cfg.N_ACTIONS,
    n_world_dim = cfg.P4_LATENT_DIM,
).to(device)
i_model = ImageReconstructorP2(world_dim=cfg.P4_LATENT_DIM).to(device)
lam     = LatentActionModel().to(device)
lse     = LocalSceneEncoder(
    latent_dim = cfg.P4_LATENT_DIM,
    n_actions  = cfg.N_ACTIONS,
    slot_dim   = config.SLOT_DIM,
).to(device)
ga      = GlobalAggregator(latent_dim=cfg.P4_LATENT_DIM).to(device)

# ── Load P3b checkpoint ───────────────────────────────────────────────────────

ckpt_name = "latest_mixed.pth" if config.PRIMITIVE == "mixed" else "latest.pth"
ckpt_path = os.path.join(cfg.P3B_CHECKPOINT_DIR, ckpt_name)
if not os.path.isfile(ckpt_path):
    ckpt_path = os.path.join(cfg.P3_CHECKPOINT_DIR, ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"No P3b (or P3) checkpoint found. Train Phase 3b first."
        )
print(f"Loading checkpoint: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location=device)
o_model.load_state_dict(ckpt["o_model"])
lam.load_state_dict(ckpt["lam"])

# N: load with strict=False — world_proj is new and absent from P3b checkpoints.
n_model.load_state_dict(ckpt["n_model"], strict=False)

# I: filter out any tensors whose shape changed (e.g. enc1 6→9 ch migration)
# AND skip new keys (world_adaLN).  Only copy weights where shapes match.
own_i  = i_model.state_dict()
ckpt_i = ckpt["i_model"]
filtered = {k: v for k, v in ckpt_i.items()
            if k in own_i and own_i[k].shape == v.shape}
own_i.update(filtered)
i_model.load_state_dict(own_i)

start_step = ckpt.get("step", 0)
print(f"  Resumed from step {start_step}")

# Freeze O and LAM
for p in o_model.parameters():
    p.requires_grad_(False)
for p in lam.parameters():
    p.requires_grad_(False)

o_model.eval(); lam.eval()

# ── Optimiser — two param groups with different LRs ───────────────────────────

optimiser = torch.optim.AdamW([
    {"params": list(lse.parameters()) + list(ga.parameters()),
     "lr": cfg.P4_LR_NEW},
    {"params": list(n_model.parameters()) + list(i_model.parameters()),
     "lr": cfg.P4_LR_FINETUNE},
], weight_decay=cfg.P4_WEIGHT_DECAY)

total_steps = cfg.P4_NUM_EPOCHS * 1000  # approximate
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser, T_max=total_steps, eta_min=cfg.P4_LR_MIN
)

# ── Dataset ───────────────────────────────────────────────────────────────────

dataset = PermutedActionDataset(
    seq_len    = cfg.P4_SEQ_LEN + 1,      # seed frame + T targets
    video_dir  = cfg.VIDEO_DIR,
    frames_dir = getattr(cfg, "FRAMES_DIR", None),
    augment    = True,
)
loader = DataLoader(
    dataset,
    batch_size  = cfg.P4_BATCH_SIZE,
    shuffle     = True,
    num_workers = 2,
    pin_memory  = False,   # pinned memory crashes on Windows TDR resets
    drop_last   = True,
    worker_init_fn = _worker_init_fn,
)

# ── Visualisation ─────────────────────────────────────────────────────────────

def save_samples(frames: list[torch.Tensor], step: int, n_show: int = 8):
    """
    Show the two-agent rollout side-by-side for the first n_show batch items.
    Left pair: agent 1 (real action order).  Right pair: agent 2 (shuffled).
    Each pair: [canvas | recon].
    """
    n_model.eval(); i_model.eval(); lse.eval(); ga.eval()
    T = cfg.P4_SEQ_LEN

    with torch.no_grad():
        seed = frames[0][:n_show].to(device)
        B    = seed.shape[0]

        # Initialise both agents identically from seed
        params_0   = decode_slots(o_model(seed))
        canvas_0   = rasterize_from_params(params_0)

        s1 = {
            "params_prev": params_0, "params_curr": params_0,
            "canvas_curr": canvas_0, "prev_recon":  seed,
            "code_prev":   torch.zeros(B, cfg.N_ACTIONS, device=device),
        }
        s2 = {k: (v.clone() if isinstance(v, torch.Tensor) else {k2: v2.clone() for k2, v2 in v.items()})
              for k, v in s1.items()}

        # Precompute action codes
        with torch.no_grad():
            codes_stack = []
            for t in range(T):
                f_t  = frames[t][:n_show].to(device)
                f_t1 = frames[t + 1][:n_show].to(device)
                code, _ = lam(f_t, f_t1)
                codes_stack.append(code)
            codes_stack = torch.stack(codes_stack, dim=1)   # (B, T, N_ACTIONS)
            shuffled    = codes_stack.clone()
            for b in range(B):
                perm        = torch.randperm(T, device=device)
                shuffled[b] = codes_stack[b, perm]

        recon_1_final = recon_2_final = None
        for t in range(T):
            c1 = codes_stack[:, t, :]
            c2 = shuffled[:, t, :]

            p1_t = SlotTransition.params_to_tensor(s1["params_curr"])
            p2_t = SlotTransition.params_to_tensor(s2["params_curr"])
            z1   = lse(s1["prev_recon"], p1_t, c1)
            z2   = lse(s2["prev_recon"], p2_t, c2)
            zg   = ga([z1, z2])

            for s, code in [(s1, c1), (s2, c2)]:
                delta      = n_model(s["params_prev"], s["params_curr"],
                                     s["prev_recon"], code, s["code_prev"],
                                     world_emb=zg)
                params_next = SlotTransition.apply_delta(s["params_curr"], delta)
                canvas_next = rasterize_from_params(params_next)
                recon       = i_model(s["canvas_curr"], canvas_next,
                                      s["prev_recon"], world_emb=zg)
                s["params_prev"] = s["params_curr"]
                s["params_curr"] = params_next
                s["canvas_curr"] = canvas_next
                s["prev_recon"]  = recon
                s["code_prev"]   = code

            recon_1_final = s1["prev_recon"]
            recon_2_final = s2["prev_recon"]

        gt_final = frames[T][:n_show].to(device)

    grid = torch.cat([
        gt_final.cpu(),
        recon_1_final.cpu(),
        recon_2_final.cpu(),
    ], dim=0)
    os.makedirs(cfg.P4_SAMPLE_DIR, exist_ok=True)
    save_image(grid, os.path.join(cfg.P4_SAMPLE_DIR, f"train_{step:07d}.png"),
               nrow=n_show, padding=2)
    n_model.train(); i_model.train(); lse.train(); ga.train()


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(step: int, tag: str = "latest"):
    os.makedirs(cfg.P4_CHECKPOINT_DIR, exist_ok=True)
    fname = f"{tag}_mixed.pth" if config.PRIMITIVE == "mixed" else f"{tag}.pth"
    torch.save({
        "step":    step,
        "o_model": o_model.state_dict(),
        "n_model": n_model.state_dict(),
        "i_model": i_model.state_dict(),
        "lam":     lam.state_dict(),
        "lse":     lse.state_dict(),
        "ga":      ga.state_dict(),
        "optimiser": optimiser.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, os.path.join(cfg.P4_CHECKPOINT_DIR, fname))


# ── Training loop ─────────────────────────────────────────────────────────────

global_step = start_step
T           = cfg.P4_SEQ_LEN

n_model.train(); i_model.train(); lse.train(); ga.train()

start_epoch = global_step // max(len(loader), 1)

for epoch in range(start_epoch, cfg.P4_NUM_EPOCHS):
    dataset.reshuffle()
    pbar = tqdm(loader, desc=f"P4 epoch {epoch+1}/{cfg.P4_NUM_EPOCHS}  "
                              f"step {global_step}", leave=False)

    for frames in pbar:
        # frames: tuple of (T+1) tensors each (B, 3, H, W)
        frames = [f.to(device) for f in frames]
        B = frames[0].shape[0]

        # ── Precompute action codes from real consecutive frames ───────────
        with torch.no_grad():
            codes_list = []
            for t in range(T):
                code, _ = lam(frames[t], frames[t + 1])
                codes_list.append(code)
            # (B, T, N_ACTIONS) — agent 1 uses real order
            codes_stack = torch.stack(codes_list, dim=1)
            # Per-batch-item random permutation — agent 2's different path
            shuffled = codes_stack.clone()
            for b in range(B):
                perm        = torch.randperm(T, device=device)
                shuffled[b] = codes_stack[b, perm]

        # ── Initialise both agents from the same seed frame ───────────────
        with torch.no_grad():
            params_0 = decode_slots(o_model(frames[0]))
            canvas_0 = rasterize_from_params(params_0)

        # Agent state dicts (all tensors start identical)
        s1 = {
            "params_prev": {k: v.clone() for k, v in params_0.items()},
            "params_curr": {k: v.clone() for k, v in params_0.items()},
            "canvas_curr": canvas_0.clone(),
            "prev_recon":  frames[0].clone(),
            "code_prev":   torch.zeros(B, cfg.N_ACTIONS, device=device),
        }
        s2 = {
            "params_prev": {k: v.clone() for k, v in params_0.items()},
            "params_curr": {k: v.clone() for k, v in params_0.items()},
            "canvas_curr": canvas_0.clone(),
            "prev_recon":  frames[0].clone(),
            "code_prev":   torch.zeros(B, cfg.N_ACTIONS, device=device),
        }

        # ── Generative unroll ─────────────────────────────────────────────
        loss_recon = torch.tensor(0.0, device=device)
        recon_1_final = recon_2_final = None

        for t in range(T):
            c1 = codes_stack[:, t, :]   # (B, N_ACTIONS) real order
            c2 = shuffled[:, t, :]      # (B, N_ACTIONS) shuffled

            # Encode local scenes and aggregate global embedding
            p1_t = SlotTransition.params_to_tensor(s1["params_curr"])
            p2_t = SlotTransition.params_to_tensor(s2["params_curr"])
            z1   = lse(s1["prev_recon"], p1_t, c1)
            z2   = lse(s2["prev_recon"], p2_t, c2)
            zg   = ga([z1, z2])

            # Agent 1 — step
            delta_1      = n_model(s1["params_prev"], s1["params_curr"],
                                   s1["prev_recon"], c1, s1["code_prev"],
                                   world_emb=zg)
            params_next_1 = SlotTransition.apply_delta(s1["params_curr"], delta_1)
            canvas_next_1 = rasterize_from_params(params_next_1)
            recon_1       = i_model(s1["canvas_curr"], canvas_next_1,
                                    s1["prev_recon"], world_emb=zg)

            # Agent 1 reconstruction loss vs ground truth
            loss_recon = loss_recon + F.mse_loss(recon_1, frames[t + 1])

            # Agent 2 — step (no per-step ground truth)
            delta_2      = n_model(s2["params_prev"], s2["params_curr"],
                                   s2["prev_recon"], c2, s2["code_prev"],
                                   world_emb=zg)
            params_next_2 = SlotTransition.apply_delta(s2["params_curr"], delta_2)
            canvas_next_2 = rasterize_from_params(params_next_2)
            recon_2       = i_model(s2["canvas_curr"], canvas_next_2,
                                    s2["prev_recon"], world_emb=zg)

            # Advance states for next step
            s1["params_prev"] = s1["params_curr"]
            s1["params_curr"] = params_next_1
            s1["canvas_curr"] = canvas_next_1
            s1["prev_recon"]  = recon_1
            s1["code_prev"]   = c1

            s2["params_prev"] = s2["params_curr"]
            s2["params_curr"] = params_next_2
            s2["canvas_curr"] = canvas_next_2
            s2["prev_recon"]  = recon_2
            s2["code_prev"]   = c2

            recon_1_final = recon_1
            recon_2_final = recon_2

        # Normalise reconstruction loss by number of steps
        loss_recon = loss_recon / T

        # ── Convergence loss (ramped) ──────────────────────────────────────
        ramp     = min(1.0, global_step / max(cfg.P4_CONVERGENCE_RAMP_STEPS, 1))
        lambda_c = ramp * cfg.P4_CONVERGENCE_WEIGHT
        loss_conv = F.mse_loss(recon_1_final, recon_2_final)

        loss = loss_recon + lambda_c * loss_conv

        if not torch.isfinite(loss):
            pbar.set_postfix({"skip": "non-finite loss"})
            optimiser.zero_grad()
            continue

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(n_model.parameters()) + list(i_model.parameters()) +
            list(lse.parameters()) + list(ga.parameters()),
            1.0,
        )
        optimiser.step()
        scheduler.step()
        global_step += 1

        pbar.set_postfix({
            "recon": f"{loss_recon.item():.4f}",
            "conv":  f"{loss_conv.item():.4f}",
            "λc":    f"{lambda_c:.3f}",
        })

        # ── Visualisation ──────────────────────────────────────────────────
        if global_step % cfg.P4_SAMPLE_EVERY == 0:
            try:
                save_samples(frames, global_step)
            except Exception as exc:
                print(f"[warn] save_samples failed: {exc}")
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # ── Checkpoint ────────────────────────────────────────────────────
        if global_step % cfg.P4_CHECKPOINT_EVERY == 0:
            save_checkpoint(global_step, tag="latest")
            print(f"\nCheckpoint saved at step {global_step}")

    save_checkpoint(global_step, tag="latest")
    print(f"Epoch {epoch+1} done — step {global_step}")

print("Phase 4 training complete.")
