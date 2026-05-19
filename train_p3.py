"""
Phase 3 — Action World Model
=============================
Adds a Latent Action Model (LAM) to the Phase 2 pipeline.

LAM infers a discrete 16-bit binary action code from consecutive frame pairs:
  (frame_t, frame_{t+1})  →  LAM  →  code  (B, N_ACTIONS)

SlotTransition N is conditioned on the code in addition to slots and frame:
  N(params_prev, params_curr, frame_t, code)  →  delta

Training (2-step unrolled, same structure as Phase 2):

── Step 1 (teacher-forced) ─────────────────────────────────────────────────
  code_1 = LAM(frame_prev, frame_t)          # action that produced frame_t
  delta_1 = N(params_pp, params_prev, frame_prev, code_1)
  params_t_hat = apply_delta(params_prev, delta_1)
  canvas_t  = rasterize(params_t_hat)
  recon_t   = I(canvas_t, frame_prev)

  loss_step1 = MSE(recon_t, frame_t)
             + P3_CANVAS_AUX_WEIGHT * MSE(canvas_t, frame_t)
             + DELTA_REG_WEIGHT * ||delta_1||²

── Step 2 (model-conditioned) ──────────────────────────────────────────────
  code_2 = LAM(frame_t, frame_t1)            # action that produced frame_t1
  delta_2 = N(params_prev.detach(), params_t_hat.detach(), recon_t.detach(), code_2)
  params_t1_hat = apply_delta(params_t_hat.detach(), delta_2)
  canvas_t1 = rasterize(params_t1_hat)
  recon_t1  = I(canvas_t1, recon_t.detach())

  loss_step2 = MSE(recon_t1, frame_t1)
             + P3_CANVAS_AUX_WEIGHT * MSE(canvas_t1, frame_t1)
             + DELTA_REG_WEIGHT * ||delta_2||²

── VQ commitment loss ────────────────────────────────────────────────────────
  loss_commit = commit_1 + commit_2   # from VQCodebook (already weighted by VQ_COMMITMENT_COST)

── Total ────────────────────────────────────────────────────────────────────
  loss = loss_step1
       + MULTISTEP_LOSS_WEIGHT * loss_step2
       + loss_commit
       + ANCHOR_LOSS_WEIGHT * loss_anchor      (same as P2)

At inference:
  LAM is not needed — pick any 16-bit code manually to control rollout.
  Flip individual bits one at a time to discover what each one does.
"""

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
import config_p3 as cfg
from model import ObjectGenerator, ImageReconstructorP2, SlotTransition, LatentActionModel
from rasterizer import decode_slots, rasterize_from_params, rasterize_flow_from_params
from dataset_video import VideoPairDataset, _worker_init_fn


def _params_normalised(params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Normalise slot param dict to comparable scale for anchor-loss MSE."""
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
    return torch.stack(dims, dim=-1)


# ── Visualisation helpers ─────────────────────────────────────────────────────

def save_samples(
    o_model:   ObjectGenerator,
    n_model:   SlotTransition,
    i_model:   ImageReconstructorP2,
    lam:       LatentActionModel,
    batch:     tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    step:      int,
    out_dir:   str,
    device:    torch.device,
    n_show:    int = 8,
):
    """
    Two-step evaluation grid.

    Rows:
      1 — target frame_t    (step-1 target)
      2 — recon_t           (step-1 output)
      3 — target frame_t1   (step-2 target)
      4 — recon_t1          (step-2 output, conditioned on recon_t)
    """
    o_model.eval(); n_model.eval(); i_model.eval(); lam.eval()
    with torch.no_grad():
        frame_pp, frame_prev, frame_t, frame_t1 = batch
        frame_pp   = frame_pp[:n_show].to(device)
        frame_prev = frame_prev[:n_show].to(device)
        frame_t    = frame_t[:n_show].to(device)
        frame_t1   = frame_t1[:n_show].to(device)

        params_pp   = decode_slots(o_model(frame_pp))
        params_prev = decode_slots(o_model(frame_prev))

        emb_pp, _, _ = lam(frame_pp,   frame_prev)
        emb_1,  _, _ = lam(frame_prev, frame_t)
        delta_1   = n_model(params_pp, params_prev, frame_prev, emb_1, emb_pp)
        params_t  = SlotTransition.apply_delta(params_prev, delta_1)
        flow_1    = rasterize_flow_from_params(params_prev, params_t)
        recon_t   = i_model(flow_1, frame_prev, action_emb=emb_1, noise_std=cfg.P3_NOISE_STD)

        emb_2, _, _ = lam(frame_t, frame_t1)
        delta_2   = n_model(params_prev, params_t, recon_t, emb_2, emb_1)
        params_t1 = SlotTransition.apply_delta(params_t, delta_2)
        flow_2    = rasterize_flow_from_params(params_t, params_t1)
        recon_t1  = i_model(flow_2, recon_t, action_emb=emb_2, noise_std=cfg.P3_NOISE_STD)

        # ── Diversity grid: same inputs, different noise seeds ─────────────
        N_DIV = 4
        div_rows = [frame_t.cpu()]
        for _ in range(N_DIV):
            r = i_model(flow_1, frame_prev, action_emb=emb_1, noise_std=cfg.P3_NOISE_STD).cpu()
            div_rows.append(r)

    grid = torch.cat([
        frame_t.cpu(), recon_t.cpu(),
        frame_t1.cpu(), recon_t1.cpu(),
    ], dim=0)
    os.makedirs(out_dir, exist_ok=True)
    save_image(grid, os.path.join(out_dir, f"train_{step:07d}.png"),
               nrow=n_show, padding=2)

    if cfg.P3_NOISE_STD > 0.0:
        div_grid = torch.cat(div_rows, dim=0)
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
    Action-dictionary rollout.

    If code_atlas.json exists (saved by engine --observe), uses those real
    gameplay codes (sorted by frequency) instead of all theoretical combos.
    Falls back to all C*E one-hot codes + null when no atlas is present.

    Creates animated GIFs grouped into blocks of 4–5 codes each.
    Each GIF frame = one timestep; each row = one action [canvas | recon].

    GPU inference runs synchronously; PIL encoding + disk writing runs in a
    daemon thread so the training loop resumes immediately.
    """
    import json as _json
    from PIL import Image as PilImage
    from torchvision.utils import make_grid

    o_model.eval(); n_model.eval(); i_model.eval(); lam.eval()

    # ── GPU inference (synchronous) ───────────────────────────────────────────
    with torch.no_grad():
        img = seed_frame[:1].to(device)
        params_seed = decode_slots(o_model(img))

        # Build one-hot codes then convert to embeddings via the codebook.
        # N uses z_q_st (C*D dimensional) not the one-hot — see LAM.forward.
        _, null_onehot, _ = lam(img, img)               # (1, C*E) — true null one-hot
        E    = cfg.VQ_NUM_ENTRIES
        C    = cfg.VQ_NUM_CODEBOOKS

        # Load atlas codes if available (top MAX_ROLLOUT_CODES), else all combos.
        # Capped to avoid TDR: at 128px each step is 4× heavier than 64px.
        MAX_ROLLOUT_CODES = 9
        atlas_path = "code_atlas.json"
        if os.path.isfile(atlas_path):
            with open(atlas_path) as _f:
                atlas = _json.load(_f)
            onehots = [null_onehot]
            for entry in atlas["codes"][:MAX_ROLLOUT_CODES]:
                row = torch.zeros_like(null_onehot)
                for cb, w in enumerate(entry["winners"]):
                    if cb < C and w < E:          # guard against stale atlas
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
        onehots     = torch.cat(onehots, dim=0)         # (1+N, C*E)
        embeddings  = lam.code_to_embedding(onehots)    # (1+N, C*D)
        null_emb    = embeddings[:1]                    # (1, C*D)

        # all_frames[code_idx][step] = (canvas_cpu, recon_cpu)  each (1,3,H,W)
        all_frames: list[list[tuple]] = []

        for emb_vec in embeddings:
            code        = emb_vec.unsqueeze(0)          # (1, C*D)
            params_curr = {k: v.clone() for k, v in params_seed.items()}
            params_prev = params_curr
            prev_recon  = img
            code_prev   = null_emb.clone()
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

                # Sync after every step to prevent TDR at higher resolutions.
                if device.type == "cuda":
                    torch.cuda.synchronize()

            all_frames.append(timeline)

    seed_cpu    = img.cpu()
    seed_canvas = rasterize_from_params(decode_slots(o_model(img))).cpu()

    # Release cached VRAM before handing control back to the training loop.
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for _m in [o_model, n_model, i_model, lam]:
        if any(p.requires_grad for p in _m.parameters()):
            _m.train()

    # ── PIL encoding + disk write (background thread, CPU-only) ───────────────
    os.makedirs(out_dir, exist_ok=True)

    # Split all codes into groups of 4 for separate GIFs
    n_codes   = len(all_frames)
    grp_size  = 4
    groups = [
        (list(range(i, min(i + grp_size, n_codes))), f"g{i // grp_size}")
        for i in range(0, n_codes, grp_size)
    ]

    def _make_frame(t: int | None, code_indices: list[int]) -> PilImage.Image:
        rows = []
        for ci in code_indices:
            if t is None:
                c_img = seed_canvas
                r_img = seed_cpu
            else:
                c_img, r_img = all_frames[ci][t]
            pair = torch.cat([c_img, r_img], dim=0)   # (2, 3, H, W)
            rows.append(pair)
        imgs = torch.cat(rows, dim=0).clamp(0, 1)     # (2*n, 3, H, W)
        g    = make_grid(imgs, nrow=2, padding=2, pad_value=0.3)
        arr  = (g.permute(1, 2, 0).numpy() * 255).astype("uint8")
        return PilImage.fromarray(arr)

    def _write_gifs():
        for code_indices, suffix in groups:
            frames = [_make_frame(None, code_indices)]
            for t in range(n_steps):
                frames.append(_make_frame(t, code_indices))

            path = os.path.join(out_dir, f"rollout_{step:07d}_{suffix}.gif")
            try:
                frames[0].save(
                    path, save_all=True, append_images=frames[1:],
                    duration=250, loop=0,
                )
            except Exception as e:
                print(f"\n[save_rollout] WARNING: could not write {path}: {e}")

    t = threading.Thread(target=_write_gifs, daemon=True)
    t.start()


# ── LR schedule (cosine, same as P2) ─────────────────────────────────────────

def cosine_lr(optimizer, step: int, total: int):
    t = step / max(total, 1)
    scale = 0.5 * (1 + math.cos(math.pi * t))
    for g in optimizer.param_groups:
        base_lr = g.get("initial_lr", g["lr"])
        g["initial_lr"] = base_lr
        g["lr"] = cfg.P3_LR_MIN + (base_lr - cfg.P3_LR_MIN) * scale


# ── Main ─────────────────────────────────────────────────────────────────────

def train(no_output: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # Disable cuDNN auto-tuner: prevents multi-second kernel searches when
    # save_samples/save_rollout use different shapes than the training batch,
    # which would otherwise trigger Windows TDR and kill the CUDA context.
    torch.backends.cudnn.benchmark = False

    # Dataset — same as Phase 2 (4-frame tuples)
    dataset = VideoPairDataset(
        video_dir  = cfg.VIDEO_DIR,
        frames_dir = getattr(cfg, "FRAMES_DIR", None),
        augment    = True,
    )
    loader  = DataLoader(
        dataset,
        batch_size         = cfg.P3_BATCH_SIZE,
        shuffle            = True,
        num_workers        = cfg.VIDEO_WORKERS,
        pin_memory         = False,   # avoid CUDA teardown crash on Windows
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

    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # ── Load Phase 2 checkpoint ───────────────────────────────────────────────
    if cfg.LOAD_P2_WEIGHTS:
        if not os.path.isfile(cfg.P2_CHECKPOINT):
            raise FileNotFoundError(
                f"Phase 2 checkpoint not found: {cfg.P2_CHECKPOINT!r}\n"
                "Train Phase 2 first (python train_p2.py), or set LOAD_P2_WEIGHTS=False."
            )
        p2_ckpt = torch.load(cfg.P2_CHECKPOINT, map_location=device, weights_only=True)
        o_model.load_state_dict(p2_ckpt["o_model"])
        i_model.load_state_dict(p2_ckpt["i_model"], strict=False)  # action_proj is new
        # N gains new action_proj + gate_head layers — load everything else, skip new keys
        p2_n = p2_ckpt["n_model"]
        missing, unexpected = n_model.load_state_dict(p2_n, strict=False)
        new_keys = [k for k in missing if "action_proj" in k or "gate_head" in k]
        if set(missing) - set(new_keys):
            print(f"  WARNING: unexpected missing keys in N: {set(missing)-set(new_keys)}")
        print(f"Loaded O, I, N from {cfg.P2_CHECKPOINT!r}  "
              f"(N gained {len(new_keys)} new params: action_proj + gate_head)")
    else:
        print("LOAD_P2_WEIGHTS=False — all models initialised from scratch.")

    # ── Freeze models ─────────────────────────────────────────────────────────
    def _freeze(model, name):
        model.requires_grad_(False)
        model.eval()
        print(f"  Frozen: {name}")

    if getattr(cfg, "FREEZE_O",   False): _freeze(o_model, "O")
    if getattr(cfg, "FREEZE_N",   False): _freeze(n_model, "N")
    if getattr(cfg, "FREEZE_I",   False): _freeze(i_model, "I")
    if getattr(cfg, "FREEZE_LAM", False): _freeze(lam,     "LAM")

    # ── Optimiser ────────────────────────────────────────────────────────────
    # Only include trainable (non-frozen) parameters in the optimizer.
    # LAM trains from random init → needs higher LR than the pre-trained O/N/I.
    _base_params = [p for m in [o_model, n_model, i_model]
                    for p in m.parameters() if p.requires_grad]
    _lam_params  = [p for p in lam.parameters() if p.requires_grad]
    _param_groups = []
    if _base_params:
        _param_groups.append({"params": _base_params, "lr": cfg.P3_LR})
    if _lam_params:
        _param_groups.append({"params": _lam_params, "lr": cfg.P3_LR * 10})
    if not _param_groups:
        raise ValueError("All models are frozen — nothing to train.")
    optimizer = torch.optim.AdamW(_param_groups, weight_decay=cfg.P3_WEIGHT_DECAY)

    def _restore_train():
        """Return non-frozen models to train mode after eval-only operations."""
        for m in [o_model, n_model, i_model, lam]:
            if any(p.requires_grad for p in m.parameters()):
                m.train()

    # ── Mixed precision ───────────────────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_steps  = len(loader) * cfg.P3_NUM_EPOCHS

    # ── Resume Phase 3 checkpoint ─────────────────────────────────────────────
    global_step = 0
    os.makedirs(cfg.P3_CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(
        cfg.P3_CHECKPOINT_DIR,
        "latest_mixed.pth" if cfg.PRIMITIVE == "mixed" else "latest.pth"
    )
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        o_model.load_state_dict(ckpt["o_model"])
        # strict=False: gate_head is new (zero-weight, bias=2.0 init → gate starts open)
        missing, unexpected = n_model.load_state_dict(ckpt["n_model"], strict=False)
        new_keys = [k for k in missing if "gate_head" in k]
        if set(missing) - set(new_keys):
            raise RuntimeError(f"Unexpected missing N keys: {set(missing) - set(new_keys)}")
        if new_keys:
            print(f"  N: initialised new gate_head from scratch ({len(new_keys)} tensors)")
        i_model.load_state_dict(ckpt["i_model"], strict=False)  # action_proj may be new
        lam.load_state_dict(ckpt["lam"])
        if "optimizer" in ckpt:
            saved_opt   = ckpt["optimizer"]
            flat_params = [p for g in optimizer.param_groups for p in g["params"]]
            saved_state = saved_opt.get("state", {})
            shape_ok = all(
                saved_state[i]["exp_avg"].shape == p.shape
                for i, p in enumerate(flat_params)
                if i in saved_state and "exp_avg" in saved_state[i]
            )
            if shape_ok:
                optimizer.load_state_dict(saved_opt)
            else:
                print("  WARNING: optimizer state skipped (parameter shapes changed — starting with fresh Adam state).")
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        global_step = ckpt.get("step", 0)
        print(f"Resumed Phase 3 from step {global_step}")


    start_epoch = global_step // len(loader)
    print(f"Steps/epoch: {len(loader)}  |  Total steps: {total_steps}  |  "
          f"Resuming from epoch {start_epoch + 1}")

    _restore_train()

    # ── Loss log (CSV, appended every LOG_EVERY steps) ───────────────────────
    log_path   = os.path.join(cfg.P3_SAMPLE_DIR, "losses.csv")
    LOG_EVERY  = 10
    os.makedirs(cfg.P3_SAMPLE_DIR, exist_ok=True)
    _write_header = not os.path.isfile(log_path) or global_step == 0
    log_file = open(log_path, "a", buffering=1)   # line-buffered
    if _write_header:
        log_file.write("step,epoch,lr,loss,r1,r2,lpips1,lpips2,canvas_prev,"
                       "canvas1,canvas2,delta1,delta2,commit,anchor\n")

    # ── One-time VQ codebook re-seed (use after a resolution change) ─────────
    _vq_reset_done = not getattr(cfg, "RESET_VQ_CODEBOOK", False)

    # Pre-check which models are frozen (avoids per-step parameter iteration)
    _o_frozen   = not any(p.requires_grad for p in o_model.parameters())
    _lam_frozen = not any(p.requires_grad for p in lam.parameters())

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.P3_NUM_EPOCHS):
        dataset.reshuffle()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.P3_NUM_EPOCHS}", leave=False)

        for frame_pp, frame_prev, frame_t, frame_t1 in pbar:
            frame_pp   = frame_pp.to(device,   non_blocking=True)
            frame_prev = frame_prev.to(device, non_blocking=True)
            frame_t    = frame_t.to(device,    non_blocking=True)
            frame_t1   = frame_t1.to(device,   non_blocking=True)

            # Re-seed VQ codebook from first batch if flagged (use after resolution change)
            if not _vq_reset_done:
                with torch.no_grad():
                    lam.vq.reinit_from_batch(lam.encode_z(frame_t, frame_t1))
                _vq_reset_done = True
                print("  [VQ] Codebook re-seeded. Set RESET_VQ_CODEBOOK=False in config_p3.py.")

            # ── Frozen model precompute (no graph, no grad) ───────────────────
            _inf = torch.inference_mode
            if _o_frozen:
                with _inf():
                    params_pp   = decode_slots(o_model(frame_pp))
                    params_prev = decode_slots(o_model(frame_prev))
            if _lam_frozen:
                with _inf():
                    emb_pp, _,      _        = lam(frame_pp,   frame_prev)
                    emb_1,  code_1, commit_1 = lam(frame_prev, frame_t)
                    emb_2,  _,      commit_2 = lam(frame_t,    frame_t1)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                # ── Step 1: predict frame_t (teacher-forced) ──────────────────
                if not _o_frozen:
                    params_pp   = decode_slots(o_model(frame_pp))
                    params_prev = decode_slots(o_model(frame_prev))

                cond_1 = frame_prev
                if cfg.CONTEXT_NOISE_STD > 0.0:
                    cond_1 = (frame_prev + cfg.CONTEXT_NOISE_STD
                              * torch.randn_like(frame_prev)).clamp(0.0, 1.0)

                # LAM infers action embeddings (z_q_st) for both transitions.
                # z_q_st has STE gradients — reconstruction loss flows back to encoder.
                # action_code (one-hot) is used only for logging vq_uniq.
                if not _lam_frozen:
                    emb_pp, _,        _         = lam(frame_pp,   frame_prev)
                    emb_1,  code_1,   commit_1  = lam(frame_prev, frame_t)

                delta_1      = n_model(params_pp, params_prev, cond_1, emb_1, emb_pp)
                params_t_hat = SlotTransition.apply_delta(params_prev, delta_1)
                flow_1       = rasterize_flow_from_params(params_prev, params_t_hat)
                recon_t      = i_model(flow_1, cond_1, action_emb=emb_1, noise_std=cfg.P3_NOISE_STD)

                loss_recon_1     = F.mse_loss(recon_t, frame_t)
                _B = min(16, recon_t.size(0))
                loss_lpips_1     = lpips_fn(recon_t[:_B] * 2.0 - 1.0,
                                            frame_t[:_B] * 2.0 - 1.0).mean()
                loss_canvas_1    = torch.tensor(0.0, device=device)
                loss_canvas_prev = torch.tensor(0.0, device=device)
                loss_delta_1     = delta_1.pow(2).mean()

                # ── Step 2: predict frame_t1 (model-conditioned) ──────────────
                params_prev_sg = {k: v.detach() for k, v in params_prev.items()}
                params_t_sg    = {k: v.detach() for k, v in params_t_hat.items()}
                cond_2         = recon_t.detach()

                # LAM infers the action that produced frame_t1 from frame_t
                if not _lam_frozen:
                    emb_2, _, commit_2 = lam(frame_t, frame_t1)

                delta_2 = n_model(params_prev_sg, params_t_sg, cond_2,
                                   emb_2, emb_1.detach())
                if cfg.DELTA_NOISE_STD > 0.0:
                    delta_2 = delta_2 + cfg.DELTA_NOISE_STD * torch.randn_like(delta_2)

                params_t1_hat = SlotTransition.apply_delta(params_t_sg, delta_2)
                flow_2        = rasterize_flow_from_params(params_t_sg, params_t1_hat)
                recon_t1      = i_model(flow_2, cond_2, action_emb=emb_2, noise_std=cfg.P3_NOISE_STD)

                loss_recon_2  = F.mse_loss(recon_t1, frame_t1)
                loss_lpips_2  = lpips_fn(recon_t1[:_B] * 2.0 - 1.0,
                                         frame_t1[:_B] * 2.0 - 1.0).mean()
                loss_canvas_2 = torch.tensor(0.0, device=device)
                loss_delta_2  = delta_2.pow(2).mean()

                # ── VQ commitment loss + codebook usage ──────────────────────
                loss_commit = (commit_1 + commit_2) if not _lam_frozen else torch.tensor(0.0, device=device)
                with torch.no_grad():
                    n_unique = len(set(map(tuple, code_1.cpu().tolist())))

                # ── Anchor loss (same as P2) ──────────────────────────────────
                loss_anchor = torch.tensor(0.0, device=device)
                if cfg.ANCHOR_LOSS_WEIGHT > 0.0:
                    with torch.no_grad():
                        anchor_t  = decode_slots(o_model(recon_t.detach()))
                        anchor_t1 = decode_slots(o_model(recon_t1.detach()))
                    loss_anchor = (
                        F.mse_loss(_params_normalised(params_t_hat),
                                   _params_normalised(anchor_t).detach())
                        + F.mse_loss(_params_normalised(params_t1_hat),
                                     _params_normalised(anchor_t1).detach())
                    )

                # ── Combined loss ─────────────────────────────────────────────
                w2   = cfg.MULTISTEP_LOSS_WEIGHT
                loss = (loss_recon_1
                        + cfg.P3_LPIPS_WEIGHT       * loss_lpips_1
                        + cfg.P2_CANVAS_AUX_WEIGHT  * loss_canvas_1
                        + cfg.P2_CANVAS_AUX_WEIGHT  * loss_canvas_prev
                        + cfg.DELTA_REG_WEIGHT       * loss_delta_1
                        + w2 * loss_recon_2
                        + w2 * cfg.P3_LPIPS_WEIGHT  * loss_lpips_2
                        + w2 * cfg.P2_CANVAS_AUX_WEIGHT * loss_canvas_2
                        + cfg.DELTA_REG_WEIGHT       * loss_delta_2
                        + loss_commit
                        + cfg.ANCHOR_LOSS_WEIGHT     * loss_anchor)

            if not torch.isfinite(loss):
                print(f"\n[step {global_step}] WARNING: non-finite loss ({loss.item():.4f}), skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(o_model.parameters())
                + list(n_model.parameters())
                + list(i_model.parameters())
                + list(lam.parameters()),
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            cosine_lr(optimizer, global_step, total_steps)

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % LOG_EVERY == 0:
                lr = optimizer.param_groups[0]['lr']
                log_file.write(
                    f"{global_step},{epoch+1},{lr:.6e},"
                    f"{loss.item():.6f},"
                    f"{loss_recon_1.item():.6f},{loss_recon_2.item():.6f},"
                    f"{loss_lpips_1.item():.6f},{loss_lpips_2.item():.6f},"
                    f"{loss_canvas_prev.item():.6f},"
                    f"{loss_canvas_1.item():.6f},{loss_canvas_2.item():.6f},"
                    f"{loss_delta_1.item():.6f},{loss_delta_2.item():.6f},"
                    f"{loss_commit.item():.6f},{loss_anchor.item():.6f}\n"
                )

            pbar.set_postfix(
                r1      = f"{loss_recon_1.item():.4f}",
                r2      = f"{loss_recon_2.item():.4f}",
                lpips   = f"{loss_lpips_1.item():.4f}",
                commit  = f"{loss_commit.item():.4f}",
                vq_uniq = n_unique,
                anchor  = f"{loss_anchor.item():.4f}",
                lr      = f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            # ── Samples ──────────────────────────────────────────────────────
            if not no_output and global_step % cfg.P3_SAMPLE_EVERY == 0:
                try:
                    save_samples(o_model, n_model, i_model, lam,
                                 (frame_pp, frame_prev, frame_t, frame_t1),
                                 global_step, cfg.P3_SAMPLE_DIR, device)
                    save_rollout(o_model, n_model, i_model, lam,
                                 frame_prev,
                                 global_step, cfg.P3_SAMPLE_DIR, device)
                except Exception as e:
                    print(f"\n[samples] WARNING: save failed at step {global_step}: {e}")
                    # If the CUDA context is dead (TDR / illegal memory access),
                    # empty_cache and the next training step will both fail.
                    # Exit cleanly so run_p3.py can restart from the last checkpoint.
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

            # ── Checkpoint ───────────────────────────────────────────────────
            if global_step % cfg.P3_CHECKPOINT_EVERY == 0:
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

        avg = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{cfg.P3_NUM_EPOCHS}  avg_loss={avg:.4f}")

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
    print("Training complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-output", action="store_true",
                        help="Skip saving samples and rollout GIFs (avoids CUDA TDR on Windows).")
    args = parser.parse_args()
    train(no_output=args.no_output)
