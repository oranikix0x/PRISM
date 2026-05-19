"""
Phase 3b — Long-Horizon Fine-Tuning
=====================================
Fine-tunes N (SlotTransition) and I (ImageReconstructorP2) from a Phase 3
checkpoint by unrolling predictions for P3B_UNROLL_STEPS steps instead of
the 2-step unroll used in Phase 3.

O (ObjectGenerator) and LAM (LatentActionModel) are kept frozen:
  - O encodes single frames independently; longer rollouts don't help it.
  - LAM provides ground-truth action codes from real frame pairs throughout
    training, so it doesn't benefit from multi-step gradients either.

Training loop (T = P3B_UNROLL_STEPS):
──────────────────────────────────────────────────────────────────────────────
  Context init (teacher-forced, no grad through O/LAM):
    params_prev = O(frame[0])
    params_curr = O(frame[1])
    code_prev   = LAM(frame[0], frame[1])
    prev_recon  = frame[1]

  Autoregressive unroll for t = 0 … T-1:
    code        = LAM(frame[t+1], frame[t+2])        # ground-truth action
    delta       = N(params_prev, params_curr, prev_recon, code, code_prev)
    params_next = apply_delta(params_curr, delta)
    canvas      = rasterize(params_next)
    recon       = I(canvas, prev_recon)

    loss_t = MSE(recon, frame[t+2])
           + P2_CANVAS_AUX_WEIGHT * MSE(canvas, frame[t+2])
           + DELTA_REG_WEIGHT * ||delta||²

    Truncated BPTT: every P3B_TRUNCATE_BPTT steps, detach all state tensors
    so the live computation graph stays bounded in VRAM.

  total_loss = mean(loss_0 … loss_{T-1})
──────────────────────────────────────────────────────────────────────────────

Visualisation:
  save_samples — 4-row grid (same as P3) using the first 2 frames of the batch.
  save_rollout — action-dictionary GIF (same as P3) using the first frame.
"""

import contextlib
import math
import os
import threading

import lpips
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import config
import config_p3b as cfg
from model import ObjectGenerator, ImageReconstructorP2, SlotTransition, LatentActionModel
from rasterizer import decode_slots, rasterize_from_params, rasterize_flow_from_params
from dataset_video import VideoSequenceDataset, _worker_init_fn


# ── Visualisation (identical logic to train_p3.py) ────────────────────────────

def save_samples(
    o_model:   ObjectGenerator,
    n_model:   SlotTransition,
    i_model:   ImageReconstructorP2,
    lam:       LatentActionModel,
    frames:    list[torch.Tensor],
    step:      int,
    out_dir:   str,
    device:    torch.device,
    n_show:    int = 8,
):
    """2-step evaluation grid from the first two context frames of the batch."""
    o_model.eval(); n_model.eval(); i_model.eval(); lam.eval()
    with torch.no_grad():
        frame_pp   = frames[0][:n_show].to(device)
        frame_prev = frames[1][:n_show].to(device)
        frame_t    = frames[2][:n_show].to(device)
        frame_t1   = frames[3][:n_show].to(device)

        params_pp   = decode_slots(o_model(frame_pp))
        params_prev = decode_slots(o_model(frame_prev))

        emb_pp, _, _ = lam(frame_pp,   frame_prev)
        emb_1,  _, _ = lam(frame_prev, frame_t)
        delta_1    = n_model(params_pp, params_prev, frame_prev, emb_1, emb_pp)
        params_t   = SlotTransition.apply_delta(params_prev, delta_1)
        flow_1     = rasterize_flow_from_params(params_prev, params_t)
        recon_t    = i_model(flow_1, frame_prev, action_emb=emb_1)

        emb_2, _, _ = lam(frame_t, frame_t1)
        delta_2    = n_model(params_prev, params_t, recon_t, emb_2, emb_1)
        params_t1  = SlotTransition.apply_delta(params_t, delta_2)
        flow_2     = rasterize_flow_from_params(params_t, params_t1)
        recon_t1   = i_model(flow_2, recon_t, action_emb=emb_2)

    grid = torch.cat([
        frame_t.cpu(), recon_t.cpu(),
        frame_t1.cpu(), recon_t1.cpu(),
    ], dim=0)
    os.makedirs(out_dir, exist_ok=True)
    save_image(grid, os.path.join(out_dir, f"train_{step:07d}.png"),
               nrow=n_show, padding=2)

    # ── Diversity grid: same inputs, different noise seeds ─────────────────
    N_DIV = 4
    noise_std = getattr(cfg, "P3B_NOISE_STD", 0.0)
    if noise_std > 0.0:
        with torch.no_grad():
            div_rows = [frame_t.cpu()]
            for _ in range(N_DIV):
                r = i_model(flow_1, frame_prev, action_emb=emb_1, noise_std=noise_std).cpu()
                div_rows.append(r)
        div_grid = torch.cat(div_rows, dim=0)   # ((N_DIV+1)*n_show, 3, H, W)
        save_image(div_grid, os.path.join(out_dir, f"diversity_{step:07d}.png"),
                   nrow=n_show, padding=2)

    for _m in [o_model, n_model, i_model, lam]:
        if any(p.requires_grad for p in _m.parameters()):
            _m.train()


def save_rollout(
    o_model:    ObjectGenerator,
    n_model:    SlotTransition,
    i_model:    ImageReconstructorP2,
    lam:        LatentActionModel,
    seed_frame: torch.Tensor,
    step:       int,
    out_dir:    str,
    device:     torch.device,
    n_steps:    int = 8,
):
    """
    Action-dictionary rollout — same as Phase 3.
    GPU inference runs synchronously; PIL/disk write runs in a daemon thread.
    """
    from PIL import Image as PilImage
    from torchvision.utils import make_grid

    o_model.eval(); n_model.eval(); i_model.eval(); lam.eval()

    with torch.no_grad():
        img         = seed_frame[:1].to(device)
        params_seed = decode_slots(o_model(img))

        # Build one-hot codes in C*E space then convert to embeddings.
        # N uses z_q_st (C*D = 128 dim), not the one-hot.
        _, null_onehot, _ = lam(img, img)               # (1, C*E) — true null one-hot
        E    = cfg.VQ_NUM_ENTRIES
        C    = cfg.VQ_NUM_CODEBOOKS

        # Load atlas codes if available (top MAX_ROLLOUT_CODES), else all combos.
        MAX_ROLLOUT_CODES = 8
        import json as _json
        atlas_path = "code_atlas.json"
        if os.path.isfile(atlas_path):
            with open(atlas_path) as _f:
                atlas = _json.load(_f)
            onehots = [null_onehot]
            for entry in atlas["codes"][:MAX_ROLLOUT_CODES]:
                row = torch.zeros_like(null_onehot)
                for cb, w in enumerate(entry["winners"]):
                    if cb < C and w < E:
                        row[0, cb * E + w] = 1.0
                onehots.append(row)
            print(f"  [rollout] Using top {len(onehots)-1} atlas codes from {atlas_path}")
        else:
            onehots = [null_onehot]
            for c in range(C):
                for e in range(E):
                    row = null_onehot.clone()
                    base = c * E
                    row[0, base:base + E] = 0.0
                    row[0, base + e]      = 1.0
                    onehots.append(row)

        onehots    = torch.cat(onehots, dim=0)           # (1+N, C*E)
        embeddings = lam.code_to_embedding(onehots)      # (1+N, C*D)
        null_emb_r = embeddings[:1]                      # (1, C*D)

        all_frames: list[list[tuple]] = []
        for emb_vec in embeddings:
            code        = emb_vec.unsqueeze(0)           # (1, C*D)
            params_curr = {k: v.clone() for k, v in params_seed.items()}
            params_prev = params_curr
            prev_recon  = img
            code_prev   = null_emb_r.clone()
            timeline    = []

            for step_idx in range(n_steps):
                delta       = n_model(params_prev, params_curr, prev_recon,
                                      code, code_prev)
                params_next = SlotTransition.apply_delta(params_curr, delta)
                flow        = rasterize_flow_from_params(params_curr, params_next)
                recon       = i_model(flow, prev_recon, action_emb=code)
                canvas_rgb  = rasterize_from_params(params_next)   # visualisation only
                timeline.append((canvas_rgb.cpu(), recon.cpu()))
                params_prev = params_curr
                params_curr = params_next
                prev_recon  = recon
                code_prev   = code

                if (cfg.ROLLOUT_GROUND_EVERY > 0
                        and (step_idx + 1) % cfg.ROLLOUT_GROUND_EVERY == 0):
                    params_curr = decode_slots(o_model(recon))

                if device.type == "cuda":
                    torch.cuda.synchronize()

            all_frames.append(timeline)

        # ── Diversity rollout: null action, N noise seeds ─────────────────
        noise_std   = getattr(cfg, "P3B_NOISE_STD", 0.0)
        N_DIV       = 4
        div_timelines: list[list[torch.Tensor]] = []
        null_emb, _, _ = lam(img, img)
        null_emb_prev  = null_emb.clone()
        for _ in range(N_DIV):
            params_curr = {k: v.clone() for k, v in params_seed.items()}
            params_prev = params_curr
            prev_recon  = img
            emb_prev    = null_emb_prev
            tl = []
            for step_idx in range(n_steps):
                delta       = n_model(params_prev, params_curr, prev_recon,
                                      null_emb, emb_prev)
                params_next = SlotTransition.apply_delta(params_curr, delta)
                flow        = rasterize_flow_from_params(params_curr, params_next)
                recon       = i_model(flow, prev_recon, action_emb=null_emb, noise_std=noise_std)
                tl.append(recon.cpu())
                params_prev = params_curr
                params_curr = params_next
                prev_recon  = recon
                emb_prev    = null_emb
            div_timelines.append(tl)
        if device.type == "cuda":
            torch.cuda.synchronize()

    seed_cpu    = img.cpu()
    seed_canvas = rasterize_from_params(decode_slots(o_model(img))).cpu()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    for _m in [o_model, n_model, i_model, lam]:
        if any(p.requires_grad for p in _m.parameters()):
            _m.train()

    os.makedirs(out_dir, exist_ok=True)
    n_codes  = len(all_frames)
    grp_size = 4
    groups = [
        (list(range(i, min(i + grp_size, n_codes))), f"g{i // grp_size}")
        for i in range(0, n_codes, grp_size)
    ]

    def _make_frame(t, code_indices):
        rows = []
        for ci in code_indices:
            c_img, r_img = (seed_canvas, seed_cpu) if t is None else all_frames[ci][t]
            rows.append(torch.cat([c_img, r_img], dim=0))
        imgs = torch.cat(rows, dim=0).clamp(0, 1)
        g    = make_grid(imgs, nrow=2, padding=2, pad_value=0.3)
        arr  = (g.permute(1, 2, 0).numpy() * 255).astype("uint8")
        return PilImage.fromarray(arr)

    def _write_gifs():
        # ── Diversity GIF (null action, different noise seeds) ─────────────
        if noise_std > 0.0:
            def _make_div_frame(t):
                rows = [seed_cpu] * N_DIV if t is None else [tl[t] for tl in div_timelines]
                imgs = torch.cat(rows, dim=0).clamp(0, 1)
                g    = make_grid(imgs, nrow=1, padding=2, pad_value=0.3)
                arr  = (g.permute(1, 2, 0).numpy() * 255).astype("uint8")
                return PilImage.fromarray(arr)

            div_gif = [_make_div_frame(None)]
            for t in range(n_steps):
                div_gif.append(_make_div_frame(t))
            div_path = os.path.join(out_dir, f"diversity_rollout_{step:07d}.gif")
            try:
                div_gif[0].save(div_path, save_all=True, append_images=div_gif[1:],
                                duration=300, loop=0)
            except Exception as e:
                print(f"\n[save_rollout] WARNING: could not write {div_path}: {e}")

        for code_indices, suffix in groups:
            frames_gif = [_make_frame(None, code_indices)]
            for t in range(n_steps):
                frames_gif.append(_make_frame(t, code_indices))
            path = os.path.join(out_dir, f"rollout_{step:07d}_{suffix}.gif")
            try:
                frames_gif[0].save(
                    path, save_all=True, append_images=frames_gif[1:],
                    duration=250, loop=0,
                )
            except Exception as e:
                print(f"\n[save_rollout] WARNING: could not write {path}: {e}")

    threading.Thread(target=_write_gifs, daemon=True).start()


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_lr(optimizer, step: int, total: int):
    t  = step / max(total, 1)
    lr = cfg.P3B_LR_MIN + 0.5 * (cfg.P3B_LR - cfg.P3B_LR_MIN) * (1 + math.cos(math.pi * t))
    for g in optimizer.param_groups:
        g["lr"] = lr


# ── Main ──────────────────────────────────────────────────────────────────────

def train(no_output: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = False

    seq_len = cfg.P3B_UNROLL_STEPS + 2   # 2 context + T prediction frames
    dataset = VideoSequenceDataset(
        seq_len    = seq_len,
        video_dir  = cfg.VIDEO_DIR,
        frames_dir = getattr(cfg, "FRAMES_DIR", None),
        max_seqs   = getattr(cfg, "MAX_VIDEO_PAIRS", None),
        augment    = True,
    )
    loader = DataLoader(
        dataset,
        batch_size         = cfg.P3B_BATCH_SIZE,
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
        n_actions  = cfg.N_ACTIONS,
    ).to(device)
    i_model = ImageReconstructorP2().to(device)
    lam     = LatentActionModel().to(device)

    # LPIPS perceptual loss — frozen VGG, eval mode, not saved in checkpoint.
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # ── Load Phase 3 checkpoint ───────────────────────────────────────────────
    p3_ckpt_path = os.path.join(
        cfg.P3_CHECKPOINT_DIR,
        "latest_mixed.pth" if cfg.PRIMITIVE == "mixed" else "latest.pth",
    )
    if not os.path.isfile(p3_ckpt_path):
        raise FileNotFoundError(
            f"Phase 3 checkpoint not found: {p3_ckpt_path!r}\n"
            "Train Phase 3 first (python train_p3.py)."
        )
    p3_ckpt = torch.load(p3_ckpt_path, map_location=device, weights_only=True)
    o_model.load_state_dict(p3_ckpt["o_model"])
    # strict=False: gate_head is new (init: zero-weight, bias=2.0 → gate starts open)
    missing, _ = n_model.load_state_dict(p3_ckpt["n_model"], strict=False)
    new_n_keys = [k for k in missing if "gate_head" in k]
    if set(missing) - set(new_n_keys):
        raise RuntimeError(f"Unexpected missing N keys from P3 ckpt: {set(missing)-set(new_n_keys)}")
    if new_n_keys:
        print(f"  N: initialised new gate_head from scratch ({len(new_n_keys)} tensors)")
    i_model.load_state_dict(p3_ckpt["i_model"])
    lam.load_state_dict(p3_ckpt["lam"])
    print(f"Loaded Phase 3 weights from {p3_ckpt_path!r}")

    # ── Freeze models ─────────────────────────────────────────────────────────
    def _freeze(model, name):
        model.requires_grad_(False)
        model.eval()
        print(f"  Frozen: {name}")

    if getattr(cfg, "FREEZE_O",   False): _freeze(o_model, "O")
    if getattr(cfg, "FREEZE_N",   False): _freeze(n_model, "N")
    if getattr(cfg, "FREEZE_I",   False): _freeze(i_model, "I")
    if getattr(cfg, "FREEZE_LAM", False): _freeze(lam,     "LAM")

    def _restore_train():
        for m in [o_model, n_model, i_model, lam]:
            if any(p.requires_grad for p in m.parameters()):
                m.train()

    # ── Optimiser (only trainable params) ────────────────────────────────────
    trainable = [p for m in [o_model, n_model, i_model, lam]
                 for p in m.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("All models are frozen — nothing to train.")
    optimizer = torch.optim.AdamW(
        trainable,
        lr           = cfg.P3B_LR,
        weight_decay = cfg.P3B_WEIGHT_DECAY,
    )

    # ── Mixed precision ───────────────────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_steps = len(loader) * cfg.P3B_NUM_EPOCHS

    # ── Resume Phase 3b checkpoint ────────────────────────────────────────────
    global_step = 0
    os.makedirs(cfg.P3B_CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(
        cfg.P3B_CHECKPOINT_DIR,
        "latest_mixed.pth" if cfg.PRIMITIVE == "mixed" else "latest.pth",
    )
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        o_model.load_state_dict(ckpt["o_model"])
        missing, _ = n_model.load_state_dict(ckpt["n_model"], strict=False)
        new_n_keys = [k for k in missing if "gate_head" in k]
        if set(missing) - set(new_n_keys):
            raise RuntimeError(f"Unexpected missing N keys: {set(missing)-set(new_n_keys)}")
        if new_n_keys:
            print(f"  N: initialised new gate_head from scratch ({len(new_n_keys)} tensors)")
        i_model.load_state_dict(ckpt["i_model"])
        lam.load_state_dict(ckpt["lam"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        global_step = ckpt.get("step", 0)
        print(f"Resumed Phase 3b from step {global_step}")

    start_epoch = global_step // max(len(loader), 1)
    print(f"Steps/epoch: {len(loader)}  |  Total steps: {total_steps}  |  "
          f"Resuming from epoch {start_epoch + 1}  |  "
          f"Unroll steps: {cfg.P3B_UNROLL_STEPS}  |  "
          f"Truncate BPTT every: {cfg.P3B_TRUNCATE_BPTT or 'never'}")

    _restore_train()

    # ── Loss log ──────────────────────────────────────────────────────────────
    log_path  = os.path.join(cfg.P3B_SAMPLE_DIR, "losses.csv")
    LOG_EVERY = 10
    os.makedirs(cfg.P3B_SAMPLE_DIR, exist_ok=True)
    _write_header = not os.path.isfile(log_path) or global_step == 0
    log_file = open(log_path, "a", buffering=1)
    if _write_header:
        log_file.write("step,epoch,lr,loss,recon_mean,canvas_mean,delta_mean,commit_mean,lpips_mean\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.P3B_NUM_EPOCHS):
        dataset.reshuffle()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.P3B_NUM_EPOCHS}", leave=False)

        for batch in pbar:
            # batch is a tuple of seq_len tensors each (B, 3, H, W)
            frames = [f.to(device, non_blocking=True) for f in batch]

            with torch.autocast(device_type=device.type, enabled=use_amp):

                # ── Context init (teacher-forced, frozen models) ──────────────
                o_ctx = torch.inference_mode() if not any(p.requires_grad for p in o_model.parameters()) else contextlib.nullcontext()
                with o_ctx:
                    params_prev = decode_slots(o_model(frames[0]))
                    params_curr = decode_slots(o_model(frames[1]))

                with torch.no_grad():
                    emb_prev, _, _ = lam(frames[0], frames[1])

                # Slot dropout: randomly blank exists in params_curr to simulate
                # temporary occlusion. N must re-derive those slots from the
                # image context and their last-known state in params_prev.
                # Only exists is masked — N can still read position/colour from
                # the slot token; it just sees the object as "not currently visible."
                if cfg.P3B_SLOT_DROPOUT > 0.0:
                    keep = (torch.rand(
                        params_curr["exists"].shape, device=device
                    ) > cfg.P3B_SLOT_DROPOUT).float()
                    params_curr = {
                        k: (v * keep if k == "exists" else v)
                        for k, v in params_curr.items()
                    }

                prev_recon  = frames[1]

                # ── Autoregressive unroll ─────────────────────────────────────
                total_loss   = torch.tensor(0.0, device=device)
                sum_recon    = 0.0
                sum_canvas   = 0.0
                sum_delta    = 0.0
                sum_commit   = 0.0
                sum_lpips    = 0.0

                for t in range(cfg.P3B_UNROLL_STEPS):
                    frame_target = frames[t + 2]

                    _lam_frozen = not any(p.requires_grad for p in lam.parameters())
                    lam_ctx = torch.inference_mode() if _lam_frozen else contextlib.nullcontext()
                    with lam_ctx:
                        emb, _, commit_loss = lam(frames[t + 1], frame_target)

                    delta       = n_model(params_prev, params_curr, prev_recon,
                                         emb, emb_prev)
                    params_next = SlotTransition.apply_delta(params_curr, delta)
                    flow        = rasterize_flow_from_params(params_curr, params_next)
                    recon       = i_model(flow, prev_recon, action_emb=emb, noise_std=cfg.P3B_NOISE_STD)

                    loss_recon  = F.mse_loss(recon, frame_target)
                    loss_canvas = torch.tensor(0.0, device=device)
                    loss_delta  = delta.pow(2).mean()
                    _B = min(8, recon.size(0))
                    loss_lpips  = lpips_fn(recon[:_B] * 2.0 - 1.0,
                                           frame_target[:_B] * 2.0 - 1.0).mean()

                    step_loss  = (loss_recon
                                  + cfg.DELTA_REG_WEIGHT * loss_delta
                                  + cfg.P3B_LPIPS_WEIGHT * loss_lpips)

                    if not _lam_frozen:
                        step_loss  = step_loss + commit_loss
                        sum_commit += commit_loss.item()

                    total_loss = total_loss + step_loss

                    sum_recon  += loss_recon.item()
                    sum_canvas += loss_canvas.item()
                    sum_delta  += loss_delta.item()
                    sum_lpips  += loss_lpips.item()

                    # ── Truncated BPTT ────────────────────────────────────────
                    trunc = cfg.P3B_TRUNCATE_BPTT
                    if trunc > 0 and (t + 1) % trunc == 0:
                        params_prev = {k: v.detach() for k, v in params_curr.items()}
                        params_curr = {k: v.detach() for k, v in params_next.items()}
                        prev_recon  = recon.detach()
                        emb_prev    = emb.detach()
                    else:
                        params_prev = params_curr
                        params_curr = params_next
                        prev_recon  = recon
                        emb_prev    = emb

                loss = total_loss / cfg.P3B_UNROLL_STEPS

            if not torch.isfinite(loss):
                print(f"\n[step {global_step}] WARNING: non-finite loss ({loss.item():.4f}), skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            cosine_lr(optimizer, global_step, total_steps)

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % LOG_EVERY == 0:
                lr = optimizer.param_groups[0]["lr"]
                T  = cfg.P3B_UNROLL_STEPS
                log_file.write(
                    f"{global_step},{epoch+1},{lr:.6e},"
                    f"{loss.item():.6f},"
                    f"{sum_recon/T:.6f},{sum_canvas/T:.6f},{sum_delta/T:.6f},"
                    f"{sum_commit/T:.6f},{sum_lpips/T:.6f}\n"
                )

            postfix = dict(
                recon  = f"{sum_recon  / cfg.P3B_UNROLL_STEPS:.4f}",
                canvas = f"{sum_canvas / cfg.P3B_UNROLL_STEPS:.4f}",
                lpips  = f"{sum_lpips  / cfg.P3B_UNROLL_STEPS:.4f}",
                lr     = f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            if not cfg.FREEZE_LAM:
                postfix["commit"] = f"{sum_commit / cfg.P3B_UNROLL_STEPS:.4f}"
            pbar.set_postfix(**postfix)

            # ── Samples ───────────────────────────────────────────────────────
            if not no_output and global_step % cfg.P3B_SAMPLE_EVERY == 0:
                try:
                    save_samples(o_model, n_model, i_model, lam,
                                 frames, global_step, cfg.P3B_SAMPLE_DIR, device)
                    save_rollout(o_model, n_model, i_model, lam,
                                 frames[1], global_step, cfg.P3B_SAMPLE_DIR, device)
                except Exception as e:
                    print(f"\n[samples] WARNING: save failed at step {global_step}: {e}")
                    if device.type == "cuda" and "cuda" in str(e).lower():
                        print("[samples] CUDA context appears dead — exiting for auto-restart.")
                        raise SystemExit(1)
                finally:
                    if device.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            print("[samples] empty_cache failed — CUDA context dead, exiting.")
                            raise SystemExit(1)

            # ── Checkpoint ────────────────────────────────────────────────────
            if global_step % cfg.P3B_CHECKPOINT_EVERY == 0:
                torch.save(
                    {"o_model":   o_model.state_dict(),
                     "n_model":   n_model.state_dict(),
                     "i_model":   i_model.state_dict(),
                     "lam":       lam.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "step":      global_step},
                    ckpt_path,
                )

        avg = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{cfg.P3B_NUM_EPOCHS}  avg_loss={avg:.4f}")

        # Save at end of every epoch so a crash never loses more than 1 epoch
        torch.save(
            {"o_model":   o_model.state_dict(),
             "n_model":   n_model.state_dict(),
             "i_model":   i_model.state_dict(),
             "lam":       lam.state_dict(),
             "optimizer": optimizer.state_dict(),
             "scaler":    scaler.state_dict(),
             "step":      global_step},
            ckpt_path,
        )

    # Final checkpoint
    torch.save(
        {"o_model":   o_model.state_dict(),
         "n_model":   n_model.state_dict(),
         "i_model":   i_model.state_dict(),
         "lam":       lam.state_dict(),
         "optimizer": optimizer.state_dict(),
         "scaler":    scaler.state_dict(),
         "step":      global_step},
        ckpt_path,
    )
    log_file.close()
    print("Phase 3b training complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-output", action="store_true",
                        help="Skip saving samples and rollout GIFs (avoids CUDA TDR on Windows).")
    args = parser.parse_args()
    train(no_output=args.no_output)
