"""
O — Object Generator
I — Image Reconstructor (Phase 1 and Phase 2)
N — Slot Transition (Phase 2)
LAM — Latent Action Model (Phase 3)

Full Phase-1 pipeline:
  input  →  O  →  slots (B, K, 9)
         →  rasterize(slots)  →  canvas (B, 3, H, W)
         →  I  →  reconstructed (B, 3, H, W)

O architecture:
  Input  : (B, 3, H, W) image, H = W = IMAGE_SIZE
  Encoder: four stride-2 conv blocks, each doubling channels.
           64×64 → 32×32 → 16×16 → 8×8 → 4×4, final 256 ch.
  Head   : flatten → two FC layers → (B, MAX_SLOTS * SLOT_DIM)
  Output : (B, MAX_SLOTS, SLOT_DIM) raw logits (not activated here;
           the rasterizer applies sigmoid/identity per slot dimension).

I architecture  (U-Net encoder-decoder with skip connections):
  Input  : rasterized canvas (B, 3, H, W)
  Output : reconstructed image (B, 3, H, W) in [0, 1]
  The skip connections let I correct fine detail that the circle
  representation can't capture, while still being forced to pass
  through the rasterized bottleneck.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ── Helpers ──────────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """Conv → GroupNorm → GELU, optional stride-2 downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResBlock(nn.Module):
    """Two conv-GN-GELU layers with an identity / projection skip."""

    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, ch), ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, ch), ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + x)


# ── O model ──────────────────────────────────────────────────────────────────

class ObjectGenerator(nn.Module):
    """
    Predicts a set of circle primitives from an input image.

    Returns:
        slots: (B, MAX_SLOTS, SLOT_DIM) raw logits.
                Each slot encodes one circle as
                [exists_logit, cx_logit, cy_logit, radius_logit,
                 r_logit, g_logit, b_logit, alpha_logit, depth_raw].
                Apply sigmoid to the first 8 dims, leave depth raw.
    """

    def __init__(
        self,
        max_slots: int = config.MAX_SLOTS,
        slot_dim:  int = config.SLOT_DIM,
        image_size: int = config.IMAGE_SIZE,  # kept for API compat, no longer used
    ):
        super().__init__()
        self.max_slots = max_slots
        self.slot_dim  = slot_dim

        # ── Encoder ──────────────────────────────────────────────────────────
        # Each stage: conv (preserve) + strided conv (downsample) + residual.
        # Channels:  3 → 32 → 64 → 128 → 256
        # Spatial: INPUT → INPUT/2 → .../4 → .../8 → .../16
        self.encoder = nn.Sequential(
            # stage 1
            _ConvBlock(3,   32),
            _ConvBlock(32,  32, stride=2),
            _ResBlock(32),
            # stage 2
            _ConvBlock(32,  64),
            _ConvBlock(64,  64, stride=2),
            _ResBlock(64),
            # stage 3
            _ConvBlock(64,  128),
            _ConvBlock(128, 128, stride=2),
            _ResBlock(128),
            # stage 4
            _ConvBlock(128, 256),
            _ConvBlock(256, 256, stride=2),
            _ResBlock(256),
        )

        # ── Head ─────────────────────────────────────────────────────────────
        # AdaptiveAvgPool2d(4) normalises any input resolution to a 4×4
        # spatial map before flattening, making O resolution-agnostic while
        # preserving spatial structure.  At the training resolution (64px)
        # it is a no-op (encoder already outputs 4×4), so the head weights
        # from existing checkpoints transfer exactly — no re-initialisation.
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.GELU(),
            nn.Linear(1024, max_slots * slot_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # Initialise slot positions on a regular grid and bias exists=1.
        #
        # With flow canvas there is no canvas-aux loss to push slots toward objects,
        # so we need structure from the very start:
        #   - All slots exist (sigmoid(+2) ≈ 0.88 → STE hard=1).
        #   - Each slot is placed at a unique grid cell so the flow canvas
        #     immediately covers the whole image and I sees different flow vectors
        #     in different image regions.  Slots then migrate toward moving objects
        #     via the reconstruction gradient.
        #
        # Grid layout: n_cols × n_rows tiles, each slot at cell centre in [0, 1].
        # tanh(logit) * half_range + mid = target  →  logit = atanh((t-mid)/half_range)
        with torch.no_grad():
            bias = self.head[-1].bias              # (max_slots * slot_dim,)
            bias = bias.view(self.max_slots, self.slot_dim)
            bias[:, 0] = +2.0                      # exists logit → sigmoid(+2) ≈ 0.88

            cx_idx = config.SLOT_KEYS.index("cx")
            cy_idx = config.SLOT_KEYS.index("cy")
            pos_mid  = (config.POSITION_MAX + config.POSITION_MIN) * 0.5   # 0.5
            pos_half = (config.POSITION_MAX - config.POSITION_MIN) * 0.5   # 1.5

            n_cols = math.isqrt(self.max_slots * 2)   # wider than tall (landscape)
            n_rows = math.ceil(self.max_slots / n_cols)
            for k in range(self.max_slots):
                col = k % n_cols
                row = k // n_cols
                cx_t = (col + 0.5) / n_cols          # target centre x in [0, 1]
                cy_t = (row + 0.5) / n_rows          # target centre y in [0, 1]
                bias[k, cx_idx] = math.atanh((cx_t - pos_mid) / pos_half)
                bias[k, cy_idx] = math.atanh((cy_t - pos_mid) / pos_half)

            # Bias sharpness so initial sigma ≈ 1.5px  (old fixed EDGE_SIGMA default).
            sharpness_idx = config.SLOT_KEYS.index("sharpness")
            bias[:, sharpness_idx] = config.EDGE_SIGMA_INIT_BIAS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalised image in [0, 1].
        Returns:
            slots: (B, MAX_SLOTS, SLOT_DIM) raw logits.
        """
        B = x.shape[0]
        features = self.encoder(x)                          # (B, 256, H/16, W/16)
        pooled   = self.pool(features)                      # (B, 256, 4, 4) — any resolution
        out      = self.head(pooled)                        # (B, MAX_SLOTS*SLOT_DIM)
        return out.view(B, self.max_slots, self.slot_dim)


# ── I model ──────────────────────────────────────────────────────────────────

class _UpBlock(nn.Module):
    """Bilinear upsample × 2, then concat skip, then two conv layers."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            _ConvBlock(in_ch + skip_ch, out_ch),
            _ResBlock(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class ImageReconstructor(nn.Module):
    """
    I — Image Reconstructor.

    Takes the rasterized canvas (output of the differentiable rasterizer)
    and maps it back to the original image space.

    U-Net with three encoder stages and three decoder stages.
    Skip connections give the network access to spatial detail from the
    canvas at each resolution, while still being bound to the circle
    representation as its sole input.

    Channels / spatial sizes (for IMAGE_SIZE=64):
      enc1: 3  → 32,  64×64 → 32×32   (skip s1: 32 ch, 32×32)
      enc2: 32 → 64,  32×32 → 16×16   (skip s2: 64 ch, 16×16)
      enc3: 64 → 128, 16×16 →  8×8    (skip s3: 128 ch, 8×8)
      bottleneck: 128 ch, 8×8
      dec3: upsample 8→16,  concat s2 (64 ch) → in=128+64=192 → out 64 ch
      dec2: upsample 16→32, concat s1 (32 ch) → in= 64+32= 96 → out 32 ch
      dec1: upsample 32→64, no skip              in= 32       → out 16 ch
      out : 16 → 3  +  sigmoid
    """

    def __init__(self):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = nn.Sequential(          # 64→32  |  skip s1: 32 ch @ 32×32
            _ConvBlock(3,  32),
            _ConvBlock(32, 32, stride=2),
            _ResBlock(32),
        )
        self.enc2 = nn.Sequential(          # 32→16  |  skip s2: 64 ch @ 16×16
            _ConvBlock(32, 64),
            _ConvBlock(64, 64, stride=2),
            _ResBlock(64),
        )
        self.enc3 = nn.Sequential(          # 16→8   |  skip s3: 128 ch @ 8×8
            _ConvBlock(64,  128),
            _ConvBlock(128, 128, stride=2),
            _ResBlock(128),
        )

        # ── Bottleneck (8×8) ─────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            _ResBlock(128),
            _ResBlock(128),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        # dec3: 8×8 → 16×16, concat s2 (64 ch)  → 128+64=192 in
        self.dec3 = _UpBlock(128, 64, 64)
        # dec2: 16×16 → 32×32, concat s1 (32 ch) → 64+32=96 in
        self.dec2 = _UpBlock(64, 32, 32)
        # dec1: 32×32 → 64×64, no skip
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            _ConvBlock(32, 16),
            _ResBlock(16),
        )

        # ── Output ───────────────────────────────────────────────────────────
        self.out_conv = nn.Sequential(
            _ConvBlock(16, 16),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, canvas: torch.Tensor) -> torch.Tensor:
        """
        Args:
            canvas: (B, 3, H, W) rasterized circle image in [0, 1].
        Returns:
            recon: (B, 3, H, W) reconstructed image in [0, 1].
        """
        s1 = self.enc1(canvas)        # (B,  32, H/2, W/2)
        s2 = self.enc2(s1)            # (B,  64, H/4, W/4)
        s3 = self.enc3(s2)            # (B, 128, H/8, W/8)

        x  = self.bottleneck(s3)      # (B, 128, H/8, W/8)

        x  = self.dec3(x,  s2)        # (B,  64, H/4, W/4)   ← skip from 16×16
        x  = self.dec2(x,  s1)        # (B,  32, H/2, W/2)   ← skip from 32×32
        x  = self.dec1(x)             # (B,  16, H,   W  )

        return self.out_conv(x)       # (B,   3, H,   W  )


# ── N model ──────────────────────────────────────────────────────────────────

class _ImagePatchEncoder(nn.Module):
    """
    Lightweight CNN: (B, 3, H, W) → (B, n_patches, hidden_dim) context tokens.

    Three stride-2 stages reduce H×W to (H/8)×(W/8); the spatial positions
    are then flattened into a sequence of patch tokens and projected to
    hidden_dim.  For IMAGE_SIZE=64 this gives 64 tokens of hidden_dim dims.
    """

    def __init__(self, hidden_dim: int, image_size: int = config.IMAGE_SIZE):
        super().__init__()
        self.cnn = nn.Sequential(
            _ConvBlock(3,   32),
            _ConvBlock(32,  32, stride=2),   # 64 → 32
            _ConvBlock(32,  64),
            _ConvBlock(64,  64, stride=2),   # 32 → 16
            _ConvBlock(64,  128),
            _ConvBlock(128, 128, stride=2),  # 16 →  8
        )
        self.proj = nn.Linear(128, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)                           # (B, 128, H/8, W/8)
        B, C, H, W = f.shape
        tokens = f.flatten(2).transpose(1, 2)     # (B, H*W, 128)
        return self.proj(tokens)                  # (B, H*W, hidden_dim)


class _CrossSelfBlock(nn.Module):
    """
    One transformer block for slot tokens:
      1. Cross-attention  — each slot attends to image patch tokens (scene context)
      2. Self-attention   — slots attend to each other (object interactions)
      3. Feed-forward     — per-slot MLP
    Pre-LN (norm_first) throughout for training stability.
    """

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead,
                                                dropout=dropout, batch_first=True)
        self.self_attn  = nn.MultiheadAttention(d_model, nhead,
                                                dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_self  = nn.LayerNorm(d_model)
        self.norm_ffn   = nn.LayerNorm(d_model)
        self.drop       = nn.Dropout(dropout)

    def forward(self, slots: torch.Tensor, img_tokens: torch.Tensor) -> torch.Tensor:
        # Cross-attention: slots (query) attend to image patches (key/value)
        s = self.norm_cross(slots)
        slots = slots + self.drop(self.cross_attn(s, img_tokens, img_tokens)[0])
        # Self-attention: slots attend to each other
        s = self.norm_self(slots)
        slots = slots + self.drop(self.self_attn(s, s, s)[0])
        # FFN
        s = self.norm_ffn(slots)
        slots = slots + self.drop(self.ffn(s))
        return slots


class SlotTransition(nn.Module):
    """
    N — Slot Transition model (Phase 2).

    Conditioned on three inputs:
      params_prev : activated slots at t-1  (B, K, 9)  — encodes past position
      params_curr : activated slots at t    (B, K, 9)  — encodes current position
      frame_t     : current image           (B, 3, H, W) — rich visual context

    Concatenating params_prev and params_curr gives each slot implicit velocity
    (Δpos = curr - prev), without needing an explicit velocity field.
    Image cross-attention lets slots query the scene for appearance cues that
    the circle representation cannot capture.

    Architecture:
      slot_tokens  = Linear(2*SLOT_DIM → hidden_dim)  applied to cat(prev, curr)
      img_tokens   = _ImagePatchEncoder(frame_t)       (B, n_patches, hidden_dim)
      n_layers × _CrossSelfBlock(cross-attn + self-attn + FFN)
      output_proj  = Linear(hidden_dim → SLOT_DIM)  zero-initialised
      delta        = tanh(output_proj(slots)) * max_delta

    Output: (B, K, SLOT_DIM) delta, bounded by ±max_delta per component.
    """

    SLOT_KEYS = config.SLOT_KEYS   # driven by config.PRIMITIVE

    def __init__(
        self,
        slot_dim:    int   = config.SLOT_DIM,
        hidden_dim:  int   = 128,
        n_heads:     int   = 4,
        n_layers:    int   = 3,
        max_delta:   float = 0.05,
        n_actions:   int   = 0,
        n_world_dim: int   = 0,
    ):
        super().__init__()
        self.max_delta = max_delta

        # Slot projection: cat(params_prev, params_curr) → hidden_dim
        self.slot_proj  = nn.Linear(2 * slot_dim, hidden_dim)

        # Optional action conditioning (Phase 3).  n_actions=0 → disabled.
        # Concatenates prev and curr action codes before projecting, so N can
        # distinguish "first frame of jump" from "second frame of jump".
        if n_actions > 0:
            self.action_proj = nn.Linear(2 * n_actions, hidden_dim)

        # Optional world-embedding conditioning (Phase 4).  n_world_dim=0 → disabled.
        # Adds z_global as a global bias on every slot token, letting N know
        # about the partner agent's current world state.  Zero-initialised so
        # P3b checkpoints load unchanged and Phase 4 fine-tunes from there.
        if n_world_dim > 0:
            self.world_proj = nn.Linear(n_world_dim, hidden_dim)
            nn.init.zeros_(self.world_proj.weight)
            nn.init.zeros_(self.world_proj.bias)

        # Image context encoder
        self.img_encoder = _ImagePatchEncoder(hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            _CrossSelfBlock(hidden_dim, n_heads, hidden_dim * 4)
            for _ in range(n_layers)
        ])

        # Small random init so N produces non-zero deltas from step 0.
        # With a flow canvas, zero deltas → zero flow → I never sees a signal
        # and learns to ignore the flow entirely (chicken-and-egg deadlock).
        # Xavier init gives small but non-zero deltas so the flow canvas is
        # immediately non-trivial and I can start learning to exploit it.
        self.output_proj = nn.Linear(hidden_dim, slot_dim)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.bias)

        # Per-slot update gate via STE: binary decision whether to apply delta.
        # Large positive bias → gate starts mostly open (≈ current behaviour).
        # The model gradually learns to close it for stable/off-screen slots.
        self.gate_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, 2.0)   # sigmoid(2.0) ≈ 0.88

    @staticmethod
    def params_to_tensor(params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Stack activated-param dict → (B, K, SLOT_DIM) tensor."""
        return torch.stack(
            [params[k] for k in SlotTransition.SLOT_KEYS], dim=-1
        )

    @staticmethod
    def apply_delta(
        params:     dict[str, torch.Tensor],
        delta:      torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Add delta to activated params and clamp each field to its valid range.

        Args:
            params: dict of activated tensors matching config.SLOT_KEYS, each (B, K).
            delta:  (B, K, SLOT_DIM) output of SlotTransition.forward().
        Returns:
            New params dict with clamped values.
        """
        import math
        _clamps: dict[str, tuple] = {
            # circle-only fields
            "exists":    (0.0,                    1.0),
            "cx":        (config.POSITION_MIN,    config.POSITION_MAX),
            "cy":        (config.POSITION_MIN,    config.POSITION_MAX),
            "radius":    (0.0,                    config.MAX_RADIUS),
            "r":         (0.0,                    1.0),
            "g":         (0.0,                    1.0),
            "b":         (0.0,                    1.0),
            "alpha":     (0.0,                    1.0),
            "sharpness": (config.EDGE_SIGMA_MIN,  config.EDGE_SIGMA_MAX),
            "depth":     (None,                   None),
            # mixed-mode extra fields
            "type":      (0.0,                    1.0),
            "p1":        (0.0,                    config.MAX_P1),
            "p2":        (-math.pi,               math.pi),
            "p3":        (0.0,                    config.MAX_LINE_WIDTH),
        }
        out = {}
        for i, k in enumerate(config.SLOT_KEYS):
            lo, hi = _clamps[k]
            val    = params[k] + delta[..., i]
            out[k] = val.clamp(lo, hi) if lo is not None else val
        return out

    def forward(
        self,
        params_prev:      dict[str, torch.Tensor],
        params_curr:      dict[str, torch.Tensor],
        frame_t:          torch.Tensor,
        action_code:      torch.Tensor | None = None,
        action_code_prev: torch.Tensor | None = None,
        world_emb:        torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            params_prev:      activated-slot dict at t-1.
            params_curr:      activated-slot dict at t.
            frame_t:          (B, 3, H, W) current image in [0, 1].
            action_code:      (B, N_ACTIONS) binary code for the current
                              transition, or None (Phase 2 backward-compatible).
            action_code_prev: (B, N_ACTIONS) binary code for the previous
                              transition.  When None, treated as all-zeros so
                              N sees "no prior action" at the start of a rollout.
            world_emb:        (B, LATENT_DIM) global world embedding from
                              GlobalAggregator, or None (Phase 3 backward-compatible).
        Returns:
            delta: (B, K, SLOT_DIM) bounded by tanh * max_delta.
        """
        p_prev = self.params_to_tensor(params_prev)           # (B, K, SLOT_DIM)
        p_curr = self.params_to_tensor(params_curr)           # (B, K, SLOT_DIM)
        slots  = self.slot_proj(torch.cat([p_prev, p_curr], dim=-1))  # (B, K, hidden_dim)

        # Inject action history as a global bias added to every slot token.
        # cat(code_prev, code_curr) lets N distinguish action duration from
        # action onset — e.g. (0,1) = first jump frame vs (1,1) = sustained.
        if action_code is not None and hasattr(self, "action_proj"):
            if action_code_prev is None:
                action_code_prev = torch.zeros_like(action_code)
            action_cat = torch.cat([action_code_prev, action_code], dim=-1)  # (B, 2*N)
            action_emb = self.action_proj(action_cat)          # (B, hidden_dim)
            slots = slots + action_emb.unsqueeze(1)            # broadcast over K slots

        # Inject global world embedding (Phase 4 cross-agent communication).
        # Zero-initialised projection → no-op when loading P3b checkpoints.
        if world_emb is not None and hasattr(self, "world_proj"):
            slots = slots + self.world_proj(world_emb).unsqueeze(1)

        img_tokens = self.img_encoder(frame_t)                 # (B, n_patches, hidden_dim)

        for block in self.blocks:
            slots = block(slots, img_tokens)

        delta_raw = torch.tanh(self.output_proj(slots)) * self.max_delta  # (B, K, SLOT_DIM)

        # Per-slot update gate: STE binary — hard 0/1 forward, soft gradient backward.
        # gate=0 → slot frozen this step; gate=1 → delta applied normally.
        gate_soft = torch.sigmoid(self.gate_head(slots)).squeeze(-1)      # (B, K)
        gate_hard = (gate_soft > 0.5).float()
        gate      = gate_soft + (gate_hard - gate_soft).detach()          # STE (B, K)

        return gate.unsqueeze(-1) * delta_raw                             # (B, K, SLOT_DIM)


# ── LAM (Phase 3) ─────────────────────────────────────────────────────────────

class VQCodebook(nn.Module):
    """
    Multi-codebook Vector Quantization with EMA codebook updates.

    Each of the C codebooks independently finds the nearest entry (L2) in its
    E-entry table and returns a one-hot selection.  The C one-hot vectors are
    concatenated to form the final action code of length C*E.

    Gradients flow back to the encoder via a straight-through estimator:
        z_q_st = z + (quantized - z).detach()

    Codebook entries are updated with exponential moving averages of the
    assigned encoder vectors (no gradient through the codebook itself).
    A small Laplace smoothing term prevents dead entries from causing
    division-by-zero.

    Args:
        num_codebooks:   C — number of independent codebooks.
        num_entries:     E — number of entries per codebook.
        embedding_dim:   D — dimension of each entry / encoder output vector.
        commitment_cost: weight on the encoder commitment loss.
        ema_decay:       EMA decay for codebook update (0.999 → slow update).
    """

    def __init__(
        self,
        num_codebooks:   int,
        num_entries:     int,
        embedding_dim:   int,
        commitment_cost: float = 0.05,
        ema_decay:       float = 0.999,
    ):
        super().__init__()
        self.num_codebooks   = num_codebooks
        self.num_entries     = num_entries
        self.embedding_dim   = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay       = ema_decay

        # Codebooks: (C, E, D) — random init, one row per entry
        codebooks = torch.randn(num_codebooks, num_entries, embedding_dim)
        self.register_buffer("codebooks",        codebooks)
        self.register_buffer("ema_cluster_size", torch.ones(num_codebooks, num_entries))
        self.register_buffer("ema_embed_sum",    codebooks.clone())

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, C*D) encoder output.
        Returns:
            z_q_st:      (B, C*D) quantized embeddings with STE.
            action_code: (B, C*E) concatenated one-hot codes.
            commit_loss: scalar — encoder commitment loss.
        """
        B = z.shape[0]
        z_cb = z.reshape(B, self.num_codebooks, self.embedding_dim)  # (B, C, D)

        # L2-normalise encoder outputs before lookup so that the codebook entries
        # and z vectors live on the same unit sphere regardless of encoder scale.
        # Without this, a scale mismatch causes one entry to win by accident and
        # EMA locks it in — the classic dead-entry feedback loop.
        z_norm  = F.normalize(z_cb, dim=-1)                           # (B, C, D)
        cb_norm = F.normalize(self.codebooks.to(z_cb.dtype), dim=-1) # (C, E, D) — match AMP dtype

        # L2 distance on unit sphere = 2 - 2*cosine_similarity
        dist = (
            z_norm.unsqueeze(2) - cb_norm.unsqueeze(0)
        ).pow(2).sum(-1)  # (B, C, E)

        indices   = dist.argmin(dim=-1)                              # (B, C)
        one_hot   = F.one_hot(indices, self.num_entries).float()     # (B, C, E)
        quantized = torch.einsum(
            "bce,ced->bcd", one_hot, self.codebooks.to(z_cb.dtype)
        )                                                             # (B, C, D)

        # Straight-through: forward=quantized, backward=gradient of z
        z_q_st = z_cb + (quantized - z_cb).detach()                 # (B, C, D)

        # EMA codebook update + dead-entry restart (training only).
        if self.training:
            with torch.no_grad():
                for c in range(self.num_codebooks):
                    oh_c = one_hot[:, c, :].float()                # (B, E) hard assignments
                    z_c  = z_cb[:, c, :].detach().float()          # (B, D) always fp32 for EMA

                    self.ema_cluster_size[c] = (
                        self.ema_decay * self.ema_cluster_size[c]
                        + (1 - self.ema_decay) * oh_c.sum(0)
                    )
                    self.ema_embed_sum[c] = (
                        self.ema_decay * self.ema_embed_sum[c]
                        + (1 - self.ema_decay) * (oh_c.t() @ z_c)
                    )
                    n = self.ema_cluster_size[c].unsqueeze(-1).clamp(min=1e-5)
                    self.codebooks[c] = self.ema_embed_sum[c] / n

                    # Dead-entry restart: if any entry has near-zero usage,
                    # reinitialise it to a random unit vector (NOT an encoder
                    # output from this batch).  Using encoder outputs fails when
                    # the encoder hasn't specialised yet — all outputs cluster in
                    # the same direction, so restarted entries immediately collapse
                    # back to the same winner.  A random unit vector guarantees
                    # the codebook spreads across the embedding sphere regardless
                    # of encoder state, breaking the feedback loop.
                    dead = self.ema_cluster_size[c] < 0.5            # (E,)
                    if dead.any():
                        n_dead = dead.sum().item()
                        rand_z = F.normalize(
                            torch.randn(n_dead, self.embedding_dim,
                                        device=self.codebooks.device,
                                        dtype=torch.float32),
                            dim=-1,
                        )
                        self.codebooks[c][dead]        = rand_z
                        self.ema_embed_sum[c][dead]    = rand_z
                        self.ema_cluster_size[c][dead] = 1.0

        commit_loss = self.commitment_cost * (z_cb - quantized.detach()).pow(2).mean()
        action_code = one_hot.reshape(B, self.num_codebooks * self.num_entries)

        return z_q_st.reshape(B, -1), action_code, commit_loss

    @torch.no_grad()
    def reinit_from_batch(self, z: torch.Tensor) -> None:
        """
        Re-seed codebook entries from encoder outputs.
        Use after a resolution change: the old entries were calibrated for a
        different feature distribution and all inputs collapse to one entry.
        Randomly samples E vectors from the batch for each codebook and
        normalises them to sit on the unit sphere, matching the lookup space.
        """
        B = z.shape[0]
        z_cb = z.reshape(B, self.num_codebooks, self.embedding_dim)
        for c in range(self.num_codebooks):
            z_c = F.normalize(z_cb[:, c, :].float(), dim=-1)   # (B, D) on unit sphere
            idx = torch.randperm(B, device=z_c.device)[:self.num_entries]
            # If batch smaller than num_entries, sample with replacement
            if B < self.num_entries:
                idx = torch.randint(0, B, (self.num_entries,), device=z_c.device)
            seeds = z_c[idx]                                    # (E, D)
            self.codebooks[c]        = seeds
            self.ema_embed_sum[c]    = seeds.clone()
            self.ema_cluster_size[c] = torch.ones(self.num_entries, device=z_c.device)
        print(f"  [VQ] Codebook re-seeded from {B} encoder outputs.")


class LatentActionModel(nn.Module):
    """
    LAM — Latent Action Model (Phase 3).

    Infers a discrete action code from two consecutive frames using
    multi-codebook Vector Quantization (VQ).  Each codebook independently
    selects one of its entries via nearest-neighbour lookup, producing a
    one-hot vector.  The C one-hots are concatenated into the final code.

    The bottleneck is structural (codebook capacity) rather than soft
    (sparsity penalty), which avoids the sparsity-vs-reconstruction tension
    and produces cleaner, more interpretable action codes.

    Architecture:
        cat(frame_t, frame_{t+1})            # (B, 6, H, W)
        → CNN encoder (4 stride-2 stages)    # (B, 256, H/16, W/16)
        → Global Average Pooling             # (B, 256)
        → FC → LayerNorm → GELU → FC        # (B, C*D)
        → VQCodebook (C codebooks, E entries) # (B, C*E) one-hot action code

    With C=4, E=4, D=32:
        action code is (B, 16) — same shape as the old 16-bit binary code,
        so SlotTransition's action_proj requires no changes.
        The number of representable action combinations is 4^4 = 256.

    Output:
        code:        (B, C*E) — one-hot action code (4 ones, one per codebook).
        commit_loss: scalar   — encoder commitment loss (replaces sparsity loss).
    """

    def __init__(
        self,
        num_codebooks:   int   = config.VQ_NUM_CODEBOOKS,
        num_entries:     int   = config.VQ_NUM_ENTRIES,
        embedding_dim:   int   = config.VQ_EMBEDDING_DIM,
        commitment_cost: float = config.VQ_COMMITMENT_COST,
        ema_decay:       float = config.VQ_EMA_DECAY,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.num_entries   = num_entries
        n_actions          = num_codebooks * num_entries  # = N_ACTIONS

        self.encoder = nn.Sequential(
            _ConvBlock(6,   32),
            _ConvBlock(32,  32,  stride=2),
            _ConvBlock(32,  64),
            _ConvBlock(64,  64,  stride=2),
            _ConvBlock(64,  128),
            _ConvBlock(128, 128, stride=2),
            _ConvBlock(128, 256),
            _ConvBlock(256, 256, stride=2),
        )
        # MLP → C*D embeddings (one D-dim vector per codebook)
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_codebooks * embedding_dim),
        )
        self.vq = VQCodebook(
            num_codebooks   = num_codebooks,
            num_entries     = num_entries,
            embedding_dim   = embedding_dim,
            commitment_cost = commitment_cost,
            ema_decay       = ema_decay,
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Do NOT zero-init the final projection for VQ — zero outputs cause all
        # inputs to map to the same codebook entry (collapse).  Let kaiming init
        # spread the embeddings so the codebook has diverse entries to update.

    def entry0_code(self, device: torch.device) -> torch.Tensor:
        """
        Entry-0 code: selects entry 0 from every codebook.
        This is an arbitrary default used in the engine when no key is pressed
        for a given codebook axis.  It is NOT the semantic null (no-action) code
        — use lam(frame, frame) to get the true null after training.
        Returns (1, C*E) one-hot tensor.
        """
        oh = torch.zeros(1, self.num_codebooks, self.num_entries, device=device)
        oh[:, :, 0] = 1.0
        return oh.reshape(1, self.num_codebooks * self.num_entries)

    def encode_z(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> torch.Tensor:
        """Return raw encoder output z (before VQ) for codebook re-seeding."""
        x     = torch.cat([frame_t, frame_t1], dim=1)
        feats = self.encoder(x).mean(dim=(-2, -1))    # (B, 256)
        return self.head(feats)                        # (B, C*D)

    def forward(
        self,
        frame_t:  torch.Tensor,
        frame_t1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frame_t:  (B, 3, H, W) current frame in [0, 1].
            frame_t1: (B, 3, H, W) next frame in [0, 1].
        Returns:
            z_q_st:      (B, C*D) quantized embedding with STE gradient.
                         Pass this to N — gradients from reconstruction loss flow
                         back to the LAM encoder through the STE, which is the
                         whole point of VQ-VAE.  Do NOT pass action_code to N;
                         one-hot has no gradient and breaks the training signal.
            action_code: (B, C*E) one-hot code (C ones, one per codebook).
                         Use for logging, display, and engine manual selection only.
            commit_loss: scalar commitment + entropy loss.
        """
        z     = self.encode_z(frame_t, frame_t1)      # (B, C*D)
        z_q_st, action_code, commit_loss = self.vq(z)
        return z_q_st, action_code, commit_loss

    def code_to_embedding(self, action_code: torch.Tensor) -> torch.Tensor:
        """
        Convert a one-hot action code to its quantized embedding vector.

        Used at inference (engine) when manually selecting an action code:
        look up the codebook entries for each codebook axis and concatenate.

        Args:
            action_code: (B, C*E) one-hot code (same format as LAM output).
        Returns:
            z_q: (B, C*D) the corresponding quantized embedding (no gradient).
        """
        B = action_code.shape[0]
        C = self.vq.num_codebooks
        E = self.vq.num_entries
        D = self.vq.embedding_dim
        one_hot = action_code.reshape(B, C, E).float()
        z_q = torch.einsum(
            "bce,ced->bcd",
            one_hot,
            self.vq.codebooks.to(action_code.device),
        )                                                   # (B, C, D)
        return z_q.reshape(B, C * D)


# ── I model (Phase 2) ─────────────────────────────────────────────────────────

def _warp_frame(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Backward-warp frame_t using a forward flow field.

    flow[:, 0] = Δx per pixel (rightward, normalised [0,1] image coords)
    flow[:, 1] = Δy per pixel (downward,  normalised [0,1] image coords)

    For each output pixel (x, y) the warp samples frame_t at approximately
    (x − Δx, y − Δy) — valid approximation for small per-frame displacements.
    Pixels sampled outside the frame boundary repeat the border value.
    """
    B, _, H, W = frame.shape
    device = frame.device

    ys = torch.linspace(0.0, 1.0, H, device=device)
    xs = torch.linspace(0.0, 1.0, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # (H, W)

    # Backward warp: subtract forward flow to find source position, then
    # convert normalised [0,1] → grid_sample's [-1, 1].
    sx = (grid_x.unsqueeze(0) - flow[:, 0]) * 2.0 - 1.0   # (B, H, W)
    sy = (grid_y.unsqueeze(0) - flow[:, 1]) * 2.0 - 1.0

    grid = torch.stack([sx, sy], dim=-1)                    # (B, H, W, 2)
    return F.grid_sample(frame, grid,
                         mode="bilinear",
                         padding_mode="border",
                         align_corners=True)


class ImageReconstructorP2(nn.Module):
    """
    I for Phase 2.

    Explicit warp-then-refine architecture.

        warped  = Warp(frame_t, flow)                        # bilinear backward warp
        input   = cat([flow(2), warped(3)], dim=1)           # (B, 5, H, W)
        delta   = UNet(input)                                 # (B, 3, H, W) unconstrained
        output  = clamp(warped + delta, 0, 1)

    The flow is structurally forced into the prediction: the residual base is
    already the warped frame, so I cannot ignore the motion signal.  I only
    needs to fix warp artifacts — occlusion boundaries, newly revealed
    regions, texture changes — rather than handling motion itself.

    Flow canvas cannot carry appearance information (2D vectors only), so no
    colour shortcut is possible even though warped contains real texture.

    Initialised from Phase 1 I weights via load_from_p1():
      enc1's first conv grows from (32, 3, 3, 3) → (32, 5, 3, 3):
        channels 0-2 (warped side) ← Phase 1 weights  (same geometric role)
        channels 3-4 (flow side)   ← zero-initialised (new pathway)
      At init the flow channels contribute nothing, so I_P2 starts as a
      pure frame-copy corrector identical to I_P1.

    Architecture: identical to Phase 1 ImageReconstructor except enc1 takes
    5 input channels instead of 3, and the output is a residual added to
    the warped frame rather than a direct sigmoid prediction.
    """

    def __init__(self, world_dim: int = 0):
        super().__init__()

        self.enc1 = nn.Sequential(          # 5ch input: [flow(2), frame_t(3)]
            _ConvBlock(5,  32),
            _ConvBlock(32, 32, stride=2),
            _ResBlock(32),
        )
        self.enc2 = nn.Sequential(
            _ConvBlock(32, 64),
            _ConvBlock(64, 64, stride=2),
            _ResBlock(64),
        )
        self.enc3 = nn.Sequential(
            _ConvBlock(64,  128),
            _ConvBlock(128, 128, stride=2),
            _ResBlock(128),
        )
        self.bottleneck = nn.Sequential(
            _ResBlock(128),
            _ResBlock(128),
        )
        self.dec3 = _UpBlock(128, 64, 64)
        self.dec2 = _UpBlock( 64, 32, 32)
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            _ConvBlock(32, 16),
            _ResBlock(16),
        )
        self.out_conv = nn.Sequential(
            _ConvBlock(16, 16),
            nn.Conv2d(16, 3, kernel_size=1),
            # No Sigmoid here — output is an unconstrained residual Δ.
            # forward() adds frame_t and clamps to [0, 1].
        )

        # Optional action-embedding conditioning (Phase 3+).
        # z_q_st (B, N_ACTIONS) is projected to a (B, 128) bias added to the
        # bottleneck feature map, letting I know which action was taken.
        # This handles non-spatial effects (colour change, appear/disappear, UI)
        # that the flow canvas cannot encode.
        # Zero-initialised → no-op when loading P2 checkpoints.
        if config.VQ_NUM_CODEBOOKS > 0:
            n_act = config.VQ_NUM_CODEBOOKS * config.VQ_EMBEDDING_DIM
            self.action_proj = nn.Linear(n_act, 128)
            nn.init.zeros_(self.action_proj.weight)
            nn.init.zeros_(self.action_proj.bias)

        # Optional world-embedding conditioning via AdaLN (Phase 4).
        # Outputs (scale, shift) for the 128-channel bottleneck feature map.
        # Zero-initialised → identity transform at P3b checkpoint load time.
        if world_dim > 0:
            self.world_adaLN = nn.Linear(world_dim, 256)  # 256 = 128 scale + 128 shift
            nn.init.zeros_(self.world_adaLN.weight)
            nn.init.zeros_(self.world_adaLN.bias)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init the final projection so the network starts as a pure
        # frame_t copy (Δ=0 everywhere) and learns to deviate from there.
        nn.init.zeros_(self.out_conv[-1].weight)
        nn.init.zeros_(self.out_conv[-1].bias)

    def load_from_p1(self, p1_state_dict: dict):
        """
        Transfer weights from a Phase 1 ImageReconstructor checkpoint.

        All layers are copied directly.  enc1's first Conv2d grows from
        3 → 5 input channels:
          channels 0-1 (flow side)   ← zero-initialised  (new; learns gradually)
          channels 2-4 (warped side) ← Phase 1 weights   (same role as canvas)
        At initialisation the flow channels contribute nothing, so I_P2 starts
        identical to I_P1 and learns to exploit the flow on top.
        """
        own = self.state_dict()
        for name, param in p1_state_dict.items():
            if name not in own:
                continue
            if own[name].shape == param.shape:
                own[name].copy_(param)
            elif name == "enc1.0.net.0.weight":
                # Phase 1: (32, 3, 3, 3) → Phase 2: (32, 5, 3, 3)
                own[name].zero_()
                own[name][:, 2:5].copy_(param)   # warped ← P1 canvas weights
        self.load_state_dict(own)

    def forward(self, flow:        torch.Tensor,
                frame_t:      torch.Tensor,
                action_emb:   torch.Tensor | None = None,
                world_emb:    torch.Tensor | None = None,
                noise_std:    float = 0.0,
                frame_drop_p: float = 0.0) -> torch.Tensor:
        """
        Args:
            flow:        (B, 2, H, W) optical-flow canvas — per-pixel (Δcx, Δcy)
                         from rasterize_flow_from_params. Encodes motion only.
            frame_t:     (B, 3, H, W) current source frame in [0, 1].
            action_emb:  (B, C*D) z_q_st from LAM — lets I condition on which
                         action was taken to handle non-spatial effects (colour
                         changes, appear/disappear, UI). None = disabled (P2).
            world_emb:   (B, LATENT_DIM) global world embedding (Phase 4), or None.
            noise_std:   std of Gaussian noise injected at the bottleneck during
                         training. 0.0 = deterministic (default / inference).
            frame_drop_p: probability of zeroing the *warped* encoder input channels
                         (0-2) per sample during training. Forces I to derive the
                         refinement delta purely from the flow signal when the
                         warped-texture shortcut is masked. The residual base
                         (warped + Δ) always uses the full warped frame.
                         0.0 = disabled (default / inference).
        Returns:
            recon: (B, 3, H, W) reconstructed next frame in [0, 1].
        """
        # Explicit warp: every pixel is displaced by the slot-motion flow.
        # This makes the flow structurally unavoidable — I only refines the result.
        warped = _warp_frame(frame_t, flow)             # (B, 3, H, W)

        x = torch.cat([flow, warped], dim=1)            # (B, 5, H, W): [flow(2)|warped(3)]

        # Frame dropout: zero the warped *encoder input* channels per sample.
        # Forces I to compute the refinement delta from the flow alone.
        # The residual skip (warped + Δ) always uses the full warped frame.
        if self.training and frame_drop_p > 0.0:
            mask = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) > frame_drop_p).float()
            x = x.clone()
            x[:, 2:] = x[:, 2:] * mask

        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x  = self.bottleneck(s3)

        if self.training and noise_std > 0.0:
            x = x + torch.randn_like(x) * noise_std

        if action_emb is not None and hasattr(self, "action_proj"):
            x = x + self.action_proj(action_emb.to(x.dtype))[:, :, None, None]

        if world_emb is not None and hasattr(self, "world_adaLN"):
            scale, shift = self.world_adaLN(world_emb).chunk(2, dim=-1)
            x = x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

        x  = self.dec3(x, s2)
        x  = self.dec2(x, s1)
        x  = self.dec1(x)
        delta = self.out_conv(x)                        # unconstrained Δ (B, 3, H, W)
        return torch.clamp(warped + delta, 0.0, 1.0)   # base = warped frame + refinement


# ── Phase 4: multi-agent communication models ─────────────────────────────────

class LocalSceneEncoder(nn.Module):
    """
    Encodes one agent's current observation into a compact local latent z_local.

    Three parallel branches are summed and projected:
      image branch  : lightweight CNN → global avg pool → Linear
      slot branch   : mean-pool over K slots → Linear
      action branch : Linear(N_ACTIONS → latent_dim)

    The image branch uses the same three-stride-2 CNN as _ImagePatchEncoder but
    with global average pooling instead of flattening, giving a fixed-size (B, D)
    vector regardless of resolution.

    Args:
        latent_dim : output dimension (= LATENT_DIM in config_p4).
        n_actions  : number of action bits (= N_ACTIONS).
        slot_dim   : dimensionality of one activated slot (= SLOT_DIM).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        n_actions:  int = 16,
        slot_dim:   int = config.SLOT_DIM,
    ):
        super().__init__()

        # Image branch: 3-stage stride-2 CNN + global avg pool → latent_dim
        self.img_cnn = nn.Sequential(
            _ConvBlock(3,   32),
            _ConvBlock(32,  32,  stride=2),
            _ConvBlock(32,  64),
            _ConvBlock(64,  64,  stride=2),
            _ConvBlock(64,  128),
            _ConvBlock(128, 128, stride=2),
        )
        self.img_proj    = nn.Linear(128, latent_dim)

        # Slot branch: mean over K slots → latent_dim
        self.slot_proj   = nn.Linear(slot_dim, latent_dim)

        # Action branch: N_ACTIONS → latent_dim
        self.action_proj = nn.Linear(n_actions, latent_dim)

        # Output normalisation after summing the three branches
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        frame:        torch.Tensor,
        params_tensor: torch.Tensor,
        action_code:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            frame:         (B, 3, H, W) agent's current imagined / real frame.
            params_tensor: (B, K, SLOT_DIM) activated slot params as a tensor
                           (use SlotTransition.params_to_tensor to obtain this).
            action_code:   (B, N_ACTIONS) binary action code for the current step.
        Returns:
            z_local: (B, latent_dim)
        """
        # Image branch
        img_feat = self.img_cnn(frame)                # (B, 128, H/8, W/8)
        img_feat = img_feat.mean(dim=(-2, -1))        # (B, 128)  global avg pool
        z_img    = self.img_proj(img_feat)            # (B, latent_dim)

        # Slot branch: mean over the K slot dimension
        z_slot   = self.slot_proj(params_tensor.mean(dim=1))  # (B, latent_dim)

        # Action branch
        z_act    = self.action_proj(action_code)      # (B, latent_dim)

        return self.norm(z_img + z_slot + z_act)      # (B, latent_dim)


class GlobalAggregator(nn.Module):
    """
    Combines local latents from P agents into a single global world embedding.

    Architecture:
        mean([z_1, ..., z_P])  →  2-layer MLP with LayerNorm  →  z_global

    Permutation-invariant by construction (mean pooling over agents).
    Designed for P=2 but works for any P ≥ 1.

    Args:
        latent_dim : must match LocalSceneEncoder.latent_dim.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, z_locals: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            z_locals: list of P tensors each (B, latent_dim).
        Returns:
            z_global: (B, latent_dim)
        """
        z_mean = torch.stack(z_locals, dim=0).mean(dim=0)  # (B, latent_dim)
        return self.mlp(z_mean)                             # (B, latent_dim)
