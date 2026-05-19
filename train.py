"""
Phase 1 training loop.

Full pipeline per step:
  image  →  O (ObjectGenerator)   →  slots (B, K, 9)
         →  rasterize(slots)      →  canvas (B, 3, H, W)
         →  I (ImageReconstructor)→  recon  (B, 3, H, W)

Losses:
  recon_loss   = MSE(recon,  image)              — primary: I vs. input
  canvas_loss  = MSE(canvas, image) * CANVAS_AUX_WEIGHT  — aux: keeps O honest
  sparse_loss  = mean(exists) * SPARSITY_WEIGHT  — encourage empty slots

Both O and I share a single AdamW optimiser.
"""

import os
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import config
from model      import ObjectGenerator, ImageReconstructor
from rasterizer import rasterize
from dataset    import Flickr30kImages


# ── Helpers ──────────────────────────────────────────────────────────────────

def save_samples(
    o_model:   ObjectGenerator,
    i_model:   ImageReconstructor,
    batch:     torch.Tensor,
    step:      int,
    out_dir:   str,
    device:    torch.device,
    n_show:    int = 8,
):
    """
    Save a three-row sample grid:
      row 1 — original images
      row 2 — rasterized canvas  (output of O + rasterizer)
      row 3 — reconstructed image (output of I)
    """
    o_model.eval()
    i_model.eval()
    with torch.no_grad():
        imgs   = batch[:n_show].to(device)
        slots  = o_model(imgs)
        canvas = rasterize(slots)
        recon  = i_model(canvas)

    grid = torch.cat([imgs.cpu(), canvas.cpu(), recon.cpu()], dim=0)
    os.makedirs(out_dir, exist_ok=True)
    save_image(grid, os.path.join(out_dir, f"step_{step:07d}.png"),
               nrow=n_show, padding=2)
    o_model.train()
    i_model.train()


def cosine_lr(optimizer, step: int, total_steps: int):
    """In-place cosine annealing from LR to LR_MIN."""
    progress = min(step / max(total_steps, 1), 1.0)
    scale    = config.LR_MIN / config.LR + 0.5 * (1.0 - config.LR_MIN / config.LR) * (
        1.0 + math.cos(math.pi * progress)
    )
    for pg in optimizer.param_groups:
        pg["lr"] = config.LR * scale


# ── Main ─────────────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset = Flickr30kImages(augment=True)
    loader  = DataLoader(
        dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = True,
        num_workers = config.NUM_WORKERS,
        pin_memory  = device.type == "cuda",
        drop_last   = True,
        persistent_workers = config.NUM_WORKERS > 0,
    )

    # ── Models & optimiser ───────────────────────────────────────────────────
    o_model = ObjectGenerator().to(device)
    i_model = ImageReconstructor().to(device)

    optimizer = torch.optim.AdamW(
        list(o_model.parameters()) + list(i_model.parameters()),
        lr           = config.LR,
        weight_decay = config.WEIGHT_DECAY,
    )

    total_steps = len(loader) * config.NUM_EPOCHS
    print(f"Steps per epoch: {len(loader)}  |  Total steps: {total_steps}")

    # Resume from latest checkpoint if present.
    global_step = 0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    if config.PRIMITIVE == "mixed":
        ckpt_path   = os.path.join(config.CHECKPOINT_DIR, "latest_mixed.pth")
    else:
        ckpt_path   = os.path.join(config.CHECKPOINT_DIR, "latest.pth")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        o_model.load_state_dict(ckpt["o_model"])
        i_model.load_state_dict(ckpt["i_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt.get("step", 0)
        print(f"Resumed from step {global_step}")

    o_model.train()
    i_model.train()

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(config.NUM_EPOCHS):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}", leave=False)

        for imgs in pbar:
            imgs = imgs.to(device, non_blocking=True)  # (B, 3, H, W)

            # ── Forward ──────────────────────────────────────────────────────
            slots  = o_model(imgs)      # (B, K, 9)
            canvas = rasterize(slots)   # (B, 3, H, W)
            recon  = i_model(canvas)    # (B, 3, H, W)

            # ── Losses ───────────────────────────────────────────────────────
            recon_loss  = F.mse_loss(recon,  imgs)
            canvas_loss = F.mse_loss(canvas, imgs)
            exists      = torch.sigmoid(slots[..., 0])   # (B, K) ∈ [0,1]
            cat_loss = (exists*(1-exists)).mean()
            sparse_loss = exists.mean()
            if config.PRIMITIVE == "mixed":
                type = torch.sigmoid(slots[..., 1])
                cat_loss += (type*(1-type)).mean()

            loss = (recon_loss
                    + config.CANVAS_AUX_WEIGHT  * canvas_loss
                    + config.SPARSITY_WEIGHT    * sparse_loss
                    + config.CAT_WEIGHT         * cat_loss)

            # ── Backward ─────────────────────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(o_model.parameters()) + list(i_model.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            cosine_lr(optimizer, global_step, total_steps)

            epoch_loss  += loss.item()
            global_step += 1

            pbar.set_postfix(
                recon  = f"{recon_loss.item():.4f}",
                canvas = f"{canvas_loss.item():.4f}",
                sparse = f"{sparse_loss.item():.4f}",
                lr     = f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            # ── Samples ──────────────────────────────────────────────────────
            if global_step % config.SAMPLE_EVERY == 0:
                save_samples(o_model, i_model, imgs, global_step,
                             config.SAMPLE_DIR, device)

            # ── Checkpoint ───────────────────────────────────────────────────
            if global_step % config.CHECKPOINT_EVERY == 0:
                torch.save(
                    {"o_model":   o_model.state_dict(),
                     "i_model":   i_model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "step":      global_step},
                    ckpt_path,
                )

        avg = epoch_loss / len(loader)
        print(f"Epoch {epoch+1:3d}  avg_loss={avg:.4f}  step={global_step}")

    # Final save
    torch.save(
        {"o_model":   o_model.state_dict(),
         "i_model":   i_model.state_dict(),
         "optimizer": optimizer.state_dict(),
         "step":      global_step},
        ckpt_path,
    )
    print("Training complete.")


if __name__ == "__main__":
    train()
