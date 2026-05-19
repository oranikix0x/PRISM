"""
Phase 2 training loop — 2-step unrolled prediction.

Quadruplet per step: (frame_{t-2}, frame_{t-1}, frame_t, frame_{t+1})

  frame_pp   = frame_{t-2}
  frame_prev = frame_{t-1}
  frame_t    = frame_t
  frame_t1   = frame_{t+1}

── Step 1 (teacher-forced) ───────────────────────────────────────────────────
  O(frame_pp)   → params_pp    (B, K) dict, activated
  O(frame_prev) → params_prev  (B, K) dict, activated
  cond_1        = frame_prev + Gaussian noise   ← frame at params_curr's time, matches rollout
  N(params_pp, params_prev, cond_1) → delta_1
  apply_delta(params_prev, delta_1) → params_t_hat
  rasterize_from_params(params_t_hat) → canvas_t
  I_P2(canvas_t, cond_1) → recon_t

  loss_step1 = MSE(recon_t, frame_t)
             + P2_CANVAS_AUX_WEIGHT * MSE(canvas_t, frame_t)
             + DELTA_REG_WEIGHT * ||delta_1||²

── Step 2 (model-conditioned, step 1 outputs detached) ───────────────────────
  N(params_prev.detach(), params_t_hat.detach(), recon_t.detach()) → delta_2
  apply_delta(params_t_hat.detach(), delta_2) → params_t1_hat
  rasterize_from_params(params_t1_hat) → canvas_t1
  I_P2(canvas_t1, recon_t.detach()) → recon_t1

  loss_step2 = MSE(recon_t1, frame_t1)
             + P2_CANVAS_AUX_WEIGHT * MSE(canvas_t1, frame_t1)
             + DELTA_REG_WEIGHT * ||delta_2||²

── Total loss ────────────────────────────────────────────────────────────────
  loss = loss_step1 + MULTISTEP_LOSS_WEIGHT * loss_step2

WHY frame_prev as context for step 1 (not frame_pp):
  In the rollout, prev_recon ≈ frame at params_CURR's time (not params_prev's).
  After each iteration, prev_recon is updated to the just-generated recon and
  params_curr advances — they stay temporally aligned.  In step 1 of training,
  params_pp ↔ rollout's params_prev and params_prev ↔ rollout's params_curr,
  so the correct context is frame at params_prev's time = frame_prev.

WHY detach step-1 outputs for step 2:
  Step 2 directly trains N and I to work with their own (imperfect) outputs,
  closing the train-rollout gap without expensive backprop through time.
  Gradients still flow through N and I in step 2; they just don't propagate
  back through step 1 (which keeps training stable).

Transfer learning:
  O  — loaded from Phase 1 checkpoint, fine-tuned from step 0.
  I  — ImageReconstructorP2 (6-ch input), loaded from Phase 1 I via
       load_from_p1(): canvas channels get Phase 1 weights, frame_prev
       channels start at zero (so I is a drop-in for Phase 1 I at init).
  N  — randomly initialised.
  All three models are trained jointly from the first step.
"""

import csv
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import config
import config_p2 as cfg
from model      import ObjectGenerator, ImageReconstructorP2, SlotTransition
from rasterizer import decode_slots, rasterize_from_params, rasterize_flow_from_params
from dataset_video import VideoPairDataset, _worker_init_fn


def _params_normalised(params: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Stack slot params into (B, K, SLOT_DIM) with every dimension mapped to
    a comparable scale for anchor-loss MSE.

    Mapping per field:
      exists / type / cx / cy / p1 / p3 / r / g / b / alpha  already [0,1] → as-is
      p2   (angle, [-π, π])   → v / π          → [-1, 1]
      sharpness ([MIN, MAX])  → (v-MIN)/(MAX-MIN) → [0, 1]
      depth (ℝ, sort-order)  → tanh(v/2)       → (-1, 1)
    """
    dims = []
    sigma_range = config.EDGE_SIGMA_MAX - config.EDGE_SIGMA_MIN
    for k in config.SLOT_KEYS:
        v = params[k]
        if k == "depth":
            dims.append(torch.tanh(v / 2.0))
        elif k == "sharpness":
            dims.append((v - config.EDGE_SIGMA_MIN) / sigma_range)
        elif k == "p2":
            dims.append(v / math.pi)
        else:
            dims.append(v)
    return torch.stack(dims, dim=-1)   # (B, K, SLOT_DIM)


# ── Helpers ──────────────────────────────────────────────────────────────────

def save_samples(
    o_model:   ObjectGenerator,
    n_model:   SlotTransition,
    i_model:   ImageReconstructorP2,
    batch:     tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    step:      int,
    out_dir:   str,
    device:    torch.device,
    n_show:    int = 8,
):
    """
    Two-step evaluation grid (uses real frames — for training diagnostics).

    Context for each step = frame corresponding to params_curr (matches rollout).

    Rows:
      1 — target frame_t    (step-1 target)
      2 — recon_t           (step-1 output)
      3 — target frame_t1   (step-2 target)
      4 — recon_t1          (step-2 output, conditioned on recon_t)
    """
    o_model.eval()
    n_model.eval()
    i_model.eval()
    with torch.no_grad():
        frame_pp, frame_prev, frame_t, frame_t1 = batch
        frame_pp   = frame_pp[:n_show].to(device)
        frame_prev = frame_prev[:n_show].to(device)
        frame_t    = frame_t[:n_show].to(device)
        frame_t1   = frame_t1[:n_show].to(device)

        params_pp   = decode_slots(o_model(frame_pp))
        params_prev = decode_slots(o_model(frame_prev))

        # Step 1: predict frame_t
        delta_1  = n_model(params_pp, params_prev, frame_prev)
        params_t = SlotTransition.apply_delta(params_prev, delta_1)
        flow_1   = rasterize_flow_from_params(params_prev, params_t)
        recon_t  = i_model(flow_1, frame_prev)

        # Step 2: predict frame_t1, conditioned on recon_t (model's own output)
        delta_2   = n_model(params_prev, params_t, recon_t)
        params_t1 = SlotTransition.apply_delta(params_t, delta_2)
        flow_2    = rasterize_flow_from_params(params_t, params_t1)
        recon_t1  = i_model(flow_2, recon_t)

        # Flow ablation: zero flow → Warp(frame_prev, 0) = frame_prev (identity).
        # recon_t_noflow shows what I produces with no motion signal at all.
        # Any difference vs recon_t is due to the warp + learned refinement.
        zeros_flow       = torch.zeros_like(flow_1)
        recon_t_noflow   = i_model(zeros_flow, frame_prev)

        # RGB canvas for visualisation only (not fed to I).
        canvas_t = rasterize_from_params(params_t)

    grid = torch.cat([
        frame_t.cpu(), recon_t.cpu(), recon_t_noflow.cpu(),
        frame_t1.cpu(), recon_t1.cpu(), canvas_t.cpu(),
    ], dim=0)
    os.makedirs(out_dir, exist_ok=True)
    save_image(grid, os.path.join(out_dir, f"step_{step:07d}.png"),
               nrow=n_show, padding=2)
    o_model.train()
    n_model.train()
    i_model.train()


def save_rollout(
    o_model:    ObjectGenerator,
    n_model:    SlotTransition,
    i_model:    ImageReconstructorP2,
    seed_frame: torch.Tensor,
    step:       int,
    out_dir:    str,
    device:     torch.device,
    n_steps:    int = 8,
    n_show:     int = 4,
):
    """
    Autoregressive multi-step rollout — O runs once, then N+I loop.

    Saves two files to out_dir:

    rollout_NNNNNNN.png  — static grid, per-sample layout:
        For each sample (pair of rows):
          row A  seed | canvas_1 | canvas_2 | … | canvas_T
          row B  seed | recon_1  | recon_2  | … | recon_T
        → n_steps+1 columns, 2*n_show rows.

    rollout_NNNNNNN.gif  — animated GIF, one frame per timestep:
        Each frame shows a 2×n_show grid:
          top row    canvas_t for each sample
          bottom row recon_t  for each sample
        seed is shown as frame 0.
    """
    from PIL import Image as PilImage

    o_model.eval()
    n_model.eval()
    i_model.eval()

    with torch.no_grad():
        imgs = seed_frame[:n_show].to(device)

        params_curr = decode_slots(o_model(imgs))
        params_prev = params_curr
        prev_recon  = imgs

        # each list entry: (canvas_rgb, recon) tensors (n_show, 3, H, W) on CPU
        # canvas_rgb is the RGB rasterisation for visualisation only.
        frames: list[tuple[torch.Tensor, torch.Tensor]] = []

        for step_idx in range(n_steps):
            delta       = n_model(params_prev, params_curr, prev_recon)
            params_next = SlotTransition.apply_delta(params_curr, delta)
            flow        = rasterize_flow_from_params(params_curr, params_next)
            recon       = i_model(flow, prev_recon)
            canvas_rgb  = rasterize_from_params(params_next)   # visualisation only

            frames.append((canvas_rgb.cpu(), recon.cpu()))

            params_prev = params_curr
            params_curr = params_next
            prev_recon  = recon

            # Slot grounding: re-encode recon through O to reset accumulated drift.
            if (cfg.ROLLOUT_GROUND_EVERY > 0
                    and (step_idx + 1) % cfg.ROLLOUT_GROUND_EVERY == 0):
                params_curr = decode_slots(o_model(recon))

    seed_cpu = imgs.cpu()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"rollout_{step:07d}")

    # ── Static PNG (per-sample rows) ─────────────────────────────────────────
    # For each sample i, build two consecutive rows of (n_steps+1) images:
    #   canvas row: [seed_i, canvas_1_i, canvas_2_i, …]
    #   recon  row: [seed_i, recon_1_i,  recon_2_i,  …]
    rows = []
    for i in range(n_show):
        canvas_timeline = torch.stack(
            [seed_cpu[i]] + [c[i] for c, _ in frames]
        )  # (n_steps+1, 3, H, W)
        recon_timeline  = torch.stack(
            [seed_cpu[i]] + [r[i] for _, r in frames]
        )
        rows.append(canvas_timeline)
        rows.append(recon_timeline)

    grid = torch.cat(rows, dim=0)   # (2*n_show*(n_steps+1), 3, H, W)
    save_image(grid, f"{base}.png", nrow=n_steps + 1, padding=2)

    # ── Animated GIF ──────────────────────────────────────────────────────────
    # Frame 0 = seed; frames 1..T = (canvas_t on top, recon_t on bottom)
    # Each GIF frame is a 2×n_show image grid rendered as a PIL image.
    def _to_pil(tensor_batch: torch.Tensor) -> PilImage.Image:
        """(N, 3, H, W) float32 → PIL image with N images tiled in one row."""
        from torchvision.utils import make_grid
        grid_t = make_grid(tensor_batch.clamp(0, 1), nrow=n_show, padding=2)
        arr    = (grid_t.permute(1, 2, 0).numpy() * 255).astype("uint8")
        return PilImage.fromarray(arr)

    # Seed frame as frame 0
    gif_frames = [_to_pil(torch.cat([seed_cpu, seed_cpu], dim=0))]

    for canvas, recon in frames:
        combined = torch.cat([canvas, recon], dim=0)  # canvas top, recon bottom
        gif_frames.append(_to_pil(combined))

    gif_frames[0].save(
        f"{base}.gif",
        save_all    = True,
        append_images = gif_frames[1:],
        duration    = 150,    # ms per frame
        loop        = 0,      # loop forever
    )

    o_model.train()
    n_model.train()
    i_model.train()


def cosine_lr(optimizer, step: int, total_steps: int):
    progress = min(step / max(total_steps, 1), 1.0)
    scale    = cfg.P2_LR_MIN / cfg.P2_LR + 0.5 * (1.0 - cfg.P2_LR_MIN / cfg.P2_LR) * (
        1.0 + math.cos(math.pi * progress)
    )
    for pg in optimizer.param_groups:
        pg["lr"] = cfg.P2_LR * scale


# ── Main ─────────────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset = VideoPairDataset(augment=True)
    loader  = DataLoader(
        dataset,
        batch_size         = cfg.P2_BATCH_SIZE,
        shuffle            = True,
        num_workers        = cfg.VIDEO_WORKERS,
        pin_memory         = False,
        drop_last          = True,
        persistent_workers = False,  # must be False: persistent workers keep a frozen copy of dataset.index and never see reshuffle() updates
        worker_init_fn     = _worker_init_fn,
    )

    # ── Models ───────────────────────────────────────────────────────────────
    o_model = ObjectGenerator().to(device)
    n_model = SlotTransition(
        hidden_dim = cfg.N_HIDDEN_DIM,
        n_heads    = cfg.N_HEADS,
        n_layers   = cfg.N_LAYERS,
        max_delta  = cfg.MAX_DELTA,
    ).to(device)
    i_model = ImageReconstructorP2().to(device)

    # ── Load Phase 1 checkpoint ───────────────────────────────────────────────
    if cfg.LOAD_P1_WEIGHTS:
        if not os.path.isfile(cfg.P1_CHECKPOINT):
            raise FileNotFoundError(
                f"Phase 1 checkpoint not found: {cfg.P1_CHECKPOINT!r}\n"
                "Train Phase 1 first (python train.py), or set LOAD_P1_WEIGHTS=False."
            )
        p1_ckpt = torch.load(cfg.P1_CHECKPOINT, map_location=device)
        o_model.load_state_dict(p1_ckpt["o_model"])
        print(f"Loaded O from {cfg.P1_CHECKPOINT!r}")
        i_model.load_from_p1(p1_ckpt["i_model"])
        print("Transferred I weights (Phase 1 → Phase 2, frame_t channels zeroed).")
    else:
        print("LOAD_P1_WEIGHTS=False — O and I initialised from scratch.")

    # ── Optimiser — all three models trained jointly from step 0 ─────────────
    optimizer = torch.optim.AdamW(
        list(o_model.parameters())
        + list(n_model.parameters())
        + list(i_model.parameters()),
        lr           = cfg.P2_LR,
        weight_decay = cfg.P2_WEIGHT_DECAY,
    )

    total_steps = len(loader) * cfg.P2_NUM_EPOCHS
    # ── Resume Phase 2 checkpoint if present ─────────────────────────────────
    global_step = 0
    os.makedirs(cfg.P2_CHECKPOINT_DIR, exist_ok=True)
    if cfg.PRIMITIVE == "mixed":
        ckpt_path = os.path.join(cfg.P2_CHECKPOINT_DIR, "latest_mixed.pth")
    else:
        ckpt_path = os.path.join(cfg.P2_CHECKPOINT_DIR, "latest.pth")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        o_model.load_state_dict(ckpt["o_model"])
        # strict=False: gate_head may be missing in older P2 checkpoints
        missing, _ = n_model.load_state_dict(ckpt["n_model"], strict=False)
        new_n_keys = [k for k in missing if "gate_head" in k]
        if set(missing) - set(new_n_keys):
            raise RuntimeError(f"Unexpected missing N keys: {set(missing)-set(new_n_keys)}")
        if new_n_keys:
            print(f"  N: initialised new gate_head from scratch ({len(new_n_keys)} tensors)")
        i_model.load_state_dict(ckpt["i_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt.get("step", 0)
        print(f"Resumed Phase 2 from step {global_step}")

    start_epoch = global_step // max(len(loader), 1)
    print(f"Steps/epoch: {len(loader)}  |  Total steps: {total_steps}  |  "
          f"Resuming from epoch {start_epoch + 1}")

    o_model.train()
    n_model.train()
    i_model.train()

    # ── Loss CSV ──────────────────────────────────────────────────────────────
    os.makedirs(cfg.P2_SAMPLE_DIR, exist_ok=True)
    log_path = os.path.join(cfg.P2_SAMPLE_DIR, "losses.csv")
    log_exists = os.path.isfile(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.DictWriter(
        log_file,
        fieldnames=["epoch", "step", "loss", "r1", "r2", "canvas1", "canvas2",
                    "delta1", "delta2", "anchor", "lr"],
    )
    if not log_exists:
        log_writer.writeheader()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.P2_NUM_EPOCHS):
        dataset.reshuffle()
        epoch_loss   = 0.0
        ep_r1        = 0.0
        ep_r2        = 0.0
        ep_canvas1   = 0.0
        ep_canvas2   = 0.0
        ep_delta1    = 0.0
        ep_delta2    = 0.0
        ep_anchor    = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.P2_NUM_EPOCHS} (step {global_step})", leave=False)

        for frame_pp, frame_prev, frame_t, frame_t1 in pbar:
            frame_pp   = frame_pp.to(device,   non_blocking=True)
            frame_prev = frame_prev.to(device, non_blocking=True)
            frame_t    = frame_t.to(device,    non_blocking=True)
            frame_t1   = frame_t1.to(device,   non_blocking=True)

            # ── Step 1: predict frame_t (teacher-forced) ──────────────────────
            params_pp   = decode_slots(o_model(frame_pp))
            params_prev = decode_slots(o_model(frame_prev))

            # Add noise to conditioning frame (exposure-bias robustness)
            cond_1 = frame_prev
            if cfg.CONTEXT_NOISE_STD > 0.0:
                cond_1 = (frame_prev + cfg.CONTEXT_NOISE_STD * torch.randn_like(frame_prev)).clamp(0.0, 1.0)

            delta_1      = n_model(params_pp, params_prev, cond_1)
            params_t_hat = SlotTransition.apply_delta(params_prev, delta_1)
            flow_1       = rasterize_flow_from_params(params_prev, params_t_hat)
            recon_t      = i_model(flow_1, cond_1, noise_std=cfg.P2_NOISE_STD, frame_drop_p=cfg.P2_FRAME_DROP_P)

            loss_recon_1  = F.mse_loss(recon_t, frame_t)
            if cfg.P2_CANVAS_AUX_WEIGHT > 0.0:
                canvas_prev   = rasterize_from_params(params_prev)
                canvas_t      = rasterize_from_params(params_t_hat)
                loss_canvas_1 = (F.mse_loss(canvas_prev, frame_prev)
                                 + F.mse_loss(canvas_t,   frame_t))
            else:
                loss_canvas_1 = torch.tensor(0.0, device=device)
            loss_delta_1  = delta_1.pow(2).mean()

            # ── Step 2: predict frame_t1 (model-conditioned) ──────────────────
            # Detach step-1 outputs so step-2 gradients don't flow back through
            # step 1, but N and I still receive their own imperfect outputs as
            # training signal — directly closing the train-rollout gap.
            params_prev_sg = {k: v.detach() for k, v in params_prev.items()}
            params_t_sg    = {k: v.detach() for k, v in params_t_hat.items()}
            cond_2         = recon_t.detach()

            delta_2 = n_model(params_prev_sg, params_t_sg, cond_2)

            # Delta noise: inject small perturbation before apply_delta to teach
            # N and I to tolerate imperfect slot states (slot-level robustness).
            if cfg.DELTA_NOISE_STD > 0.0:
                delta_2 = delta_2 + cfg.DELTA_NOISE_STD * torch.randn_like(delta_2)

            params_t1_hat = SlotTransition.apply_delta(params_t_sg, delta_2)
            flow_2        = rasterize_flow_from_params(params_t_sg, params_t1_hat)
            recon_t1      = i_model(flow_2, cond_2, noise_std=cfg.P2_NOISE_STD, frame_drop_p=cfg.P2_FRAME_DROP_P)

            loss_recon_2  = F.mse_loss(recon_t1, frame_t1)
            if cfg.P2_CANVAS_AUX_WEIGHT > 0.0:
                canvas_t1     = rasterize_from_params(params_t1_hat)
                loss_canvas_2 = F.mse_loss(canvas_t1, frame_t1)
            else:
                loss_canvas_2 = torch.tensor(0.0, device=device)
            loss_delta_2  = delta_2.pow(2).mean()

            # ── Anchor loss ───────────────────────────────────────────────────
            # Run O on the generated recons (no grad through O here — O is used
            # as a fixed "slot-space oracle" to define realistic slot targets).
            # Penalises N for predicting slots that O would not produce from a
            # real-looking frame — combats slot-space drift over long rollouts.
            loss_anchor = torch.tensor(0.0, device=device)
            if cfg.ANCHOR_LOSS_WEIGHT > 0.0:
                with torch.no_grad():
                    anchor_t  = decode_slots(o_model(recon_t.detach()))
                    anchor_t1 = decode_slots(o_model(recon_t1.detach()))
                # All dims normalised to ~[-1,1]/[0,1] so every field
                # (including depth and sharpness) contributes at the same scale.
                loss_anchor = (
                    F.mse_loss(_params_normalised(params_t_hat),
                               _params_normalised(anchor_t).detach())
                    + F.mse_loss(_params_normalised(params_t1_hat),
                                 _params_normalised(anchor_t1).detach())
                )

            # ── Combined loss ─────────────────────────────────────────────────
            w2   = cfg.MULTISTEP_LOSS_WEIGHT
            loss = (loss_recon_1
                    + cfg.P2_CANVAS_AUX_WEIGHT * loss_canvas_1
                    + cfg.DELTA_REG_WEIGHT      * loss_delta_1
                    + w2 * loss_recon_2
                    + w2 * cfg.P2_CANVAS_AUX_WEIGHT * loss_canvas_2
                    + cfg.DELTA_REG_WEIGHT      * loss_delta_2
                    + cfg.ANCHOR_LOSS_WEIGHT    * loss_anchor)

            # ── Backward ─────────────────────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(o_model.parameters())
                + list(n_model.parameters())
                + list(i_model.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            cosine_lr(optimizer, global_step, total_steps)

            epoch_loss  += loss.item()
            ep_r1       += loss_recon_1.item()
            ep_r2       += loss_recon_2.item()
            ep_canvas1  += loss_canvas_1.item()
            ep_canvas2  += loss_canvas_2.item()
            ep_delta1   += loss_delta_1.item()
            ep_delta2   += loss_delta_2.item()
            ep_anchor   += loss_anchor.item()
            global_step += 1

            pbar.set_postfix(
                r1     = f"{loss_recon_1.item():.4f}",
                r2     = f"{loss_recon_2.item():.4f}",
                anchor = f"{loss_anchor.item():.4f}",
                delta  = f"{loss_delta_1.item():.5f}",
                lr     = f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            # ── Samples ──────────────────────────────────────────────────────
            if global_step % cfg.P2_SAMPLE_EVERY == 0:
                save_samples(o_model, n_model, i_model,
                             (frame_pp, frame_prev, frame_t, frame_t1), global_step,
                             cfg.P2_SAMPLE_DIR, device)
                save_rollout(o_model, n_model, i_model,
                             frame_prev, global_step,
                             cfg.P2_SAMPLE_DIR, device)

            # ── Checkpoint ───────────────────────────────────────────────────
            if global_step % cfg.P2_CHECKPOINT_EVERY == 0:
                torch.save(
                    {"o_model":   o_model.state_dict(),
                     "n_model":   n_model.state_dict(),
                     "i_model":   i_model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "step":      global_step},
                    ckpt_path,
                )

        n = len(loader)
        avg = epoch_loss / n
        print(f"Epoch {epoch+1:3d}  avg_loss={avg:.4f}  r1={ep_r1/n:.4f}  r2={ep_r2/n:.4f}  step={global_step}")
        log_writer.writerow({
            "epoch":   epoch + 1,
            "step":    global_step,
            "loss":    f"{avg:.6f}",
            "r1":      f"{ep_r1/n:.6f}",
            "r2":      f"{ep_r2/n:.6f}",
            "canvas1": f"{ep_canvas1/n:.6f}",
            "canvas2": f"{ep_canvas2/n:.6f}",
            "delta1":  f"{ep_delta1/n:.6f}",
            "delta2":  f"{ep_delta2/n:.6f}",
            "anchor":  f"{ep_anchor/n:.6f}",
            "lr":      f"{optimizer.param_groups[0]['lr']:.2e}",
        })
        log_file.flush()

    log_file.close()

    # Final save
    torch.save(
        {"o_model":   o_model.state_dict(),
         "n_model":   n_model.state_dict(),
         "i_model":   i_model.state_dict(),
         "optimizer": optimizer.state_dict(),
         "step":      global_step},
        ckpt_path,
    )
    print("Phase 2 training complete.")


if __name__ == "__main__":
    train()
