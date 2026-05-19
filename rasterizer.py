"""
Differentiable primitive rasterizer.

Supports two primitive modes, controlled by config.PRIMITIVE:

  "circle"  —  each slot renders a soft-edged filled circle.
               Slot layout (10 dims):
                 [exists, cx, cy, radius, r, g, b, alpha, sharpness, depth]

  "mixed"   —  each slot blends a circle AND a line via a soft 'type' weight.
               type ≈ 0 → circle  (p1=radius)
               type ≈ 1 → line    (cx,cy=centre, p1=half-length, p2=angle, p3=half-width)
               Slot layout (13 dims):
                 [exists, type, cx, cy, p1, p2, p3, r, g, b, alpha, sharpness, depth]
               coverage = (1 - type) * circle_cov + type * line_cov
               This is fully differentiable; the model learns which primitive
               to use for each slot through gradient descent.

Per-slot learnable sharpness:
  Each slot has a dedicated 'sharpness' parameter (raw logit) that the model
  learns independently.  It activates to a per-slot edge sigma:
      sigma = EDGE_SIGMA_MIN + sigmoid(sharpness) * (EDGE_SIGMA_MAX - EDGE_SIGMA_MIN)
  This lets large background blobs learn soft edges while small detail
  primitives learn sharp edges — without any manual tuning.

Rendering is fully vectorised — no Python loops.

Depth ordering:
  Primitives are sorted front-to-back by descending depth before compositing.

Alpha compositing (Porter-Duff "over", front-to-back):
  weight_k   = eff_alpha_k  *  product_{j<k}(1 − eff_alpha_j)
  colour_out = Σ_k  weight_k * colour_k
  background fills the remaining fraction.
"""

import math

import torch

import config


# ── Activation helpers ────────────────────────────────────────────────────────

def decode_slots(slots: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Apply activations to raw slot logits from O.

    Args:
        slots: (B, K, SLOT_DIM) raw logits.

    Returns:
        dict of activated tensors, each (B, K).

    Circle mode (SLOT_DIM=10):
        exists, r, g, b, alpha  ∈ [0, 1]
        cx, cy                  ∈ [POSITION_MIN, POSITION_MAX]  (off-screen allowed)
        radius                  ∈ [0, MAX_RADIUS]
        sharpness               ∈ [EDGE_SIGMA_MIN, EDGE_SIGMA_MAX]
        depth                   ∈ ℝ  (sorting only)

    Mixed mode (SLOT_DIM=13):
        exists, type, r, g, b, alpha  ∈ [0, 1]
        cx, cy                         ∈ [POSITION_MIN, POSITION_MAX]
        p1                             ∈ [0, MAX_P1]
        p2  (angle)                    ∈ [−π, π]
        p3  (half-width)               ∈ [0, MAX_LINE_WIDTH]
        sharpness                      ∈ [EDGE_SIGMA_MIN, EDGE_SIGMA_MAX]
        depth                          ∈ ℝ

    cx/cy use tanh scaled to [POSITION_MIN, POSITION_MAX] so O can predict
    object centres outside the visible [0,1] area for partially-clipped
    objects at frame edges.  The logit=0 initialisation maps to screen centre
    (0.5) and gradients are stronger at the edges than sigmoid would give.
    """
    sig = torch.sigmoid
    _sigma_range = config.EDGE_SIGMA_MAX - config.EDGE_SIGMA_MIN
    # tanh maps ℝ → (-1, 1); scale+shift to (POSITION_MIN, POSITION_MAX).
    # midpoint = (MAX+MIN)/2 = 0.5, half-range = (MAX-MIN)/2 = 1.5.
    _pos_mid   = (config.POSITION_MAX + config.POSITION_MIN) * 0.5   # 0.5
    _pos_half  = (config.POSITION_MAX - config.POSITION_MIN) * 0.5   # 1.5

    def _pos(x):
        return torch.tanh(x) * _pos_half + _pos_mid

    def _ste(x):
        """Straight-Through Estimator: hard binary forward, soft sigmoid gradient backward."""
        soft = sig(x)
        hard = (x > 0).float()          # equivalent to soft > 0.5
        return soft + (hard - soft).detach()

    if config.PRIMITIVE == "circle":
        return {
            "exists":    _ste(slots[..., 0]),
            "cx":        _pos(slots[..., 1]),
            "cy":        _pos(slots[..., 2]),
            "radius":    sig(slots[..., 3]) * config.MAX_RADIUS,
            "r":         sig(slots[..., 4]),
            "g":         sig(slots[..., 5]),
            "b":         sig(slots[..., 6]),
            "alpha":     sig(slots[..., 7]),
            "sharpness": config.EDGE_SIGMA_MIN + sig(slots[..., 8]) * _sigma_range,
            "depth":     slots[..., 9],
        }
    else:  # "mixed"
        return {
            "exists":    _ste(slots[..., 0]),
            "type":      _ste(slots[..., 1]),
            "cx":        _pos(slots[..., 2]),
            "cy":        _pos(slots[..., 3]),
            "p1":        sig(slots[..., 4]) * config.MAX_P1,
            "p2":        torch.tanh(slots[..., 5]) * math.pi,
            "p3":        sig(slots[..., 6]) * config.MAX_LINE_WIDTH,
            "r":         sig(slots[..., 7]),
            "g":         sig(slots[..., 8]),
            "b":         sig(slots[..., 9]),
            "alpha":     sig(slots[..., 10]),
            "sharpness": config.EDGE_SIGMA_MIN + sig(slots[..., 11]) * _sigma_range,
            "depth":     slots[..., 12],
        }


# ── Coverage helpers ──────────────────────────────────────────────────────────

def _circle_coverage(
    s_cx:      torch.Tensor,   # (B, K)
    s_cy:      torch.Tensor,
    s_radius:  torch.Tensor,
    s_sigma:   torch.Tensor,   # (B, K) per-slot edge sigma
    grid_x:    torch.Tensor,   # (1, 1, H, W)
    grid_y:    torch.Tensor,
) -> torch.Tensor:
    """Soft filled-circle coverage.  Returns (B, K, H, W) in [0, 1]."""
    dx   = grid_x - s_cx.unsqueeze(-1).unsqueeze(-1)
    dy   = grid_y - s_cy.unsqueeze(-1).unsqueeze(-1)
    dist = (dx.square() + dy.square()).add(1e-12).sqrt()
    return torch.sigmoid(
        (s_radius.unsqueeze(-1).unsqueeze(-1) - dist)
        / s_sigma.unsqueeze(-1).unsqueeze(-1)
    )


def _line_coverage(
    s_cx:     torch.Tensor,   # (B, K)  line centre x
    s_cy:     torch.Tensor,
    s_p1:     torch.Tensor,   # (B, K)  half-length
    s_p2:     torch.Tensor,   # (B, K)  angle (radians)
    s_p3:     torch.Tensor,   # (B, K)  half-width
    s_sigma:  torch.Tensor,   # (B, K)  per-slot edge sigma
    grid_x:   torch.Tensor,   # (1, 1, H, W)
    grid_y:   torch.Tensor,
) -> torch.Tensor:
    """
    Soft capsule (rounded-end line) coverage.  Returns (B, K, H, W) in [0, 1].

    Algorithm:
      1. Translate pixel coords to line-centre frame.
      2. Rotate so the line axis is horizontal.
      3. Clamp the along-axis coord to [−half_len, +half_len]  (capsule ends).
      4. Distance to the closest point on the segment.
      5. Sigmoid with per-slot half-width and sharpness sigma.
    """
    dx    = grid_x - s_cx.unsqueeze(-1).unsqueeze(-1)   # (B, K, H, W)
    dy    = grid_y - s_cy.unsqueeze(-1).unsqueeze(-1)

    cos_a = torch.cos(s_p2).unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
    sin_a = torch.sin(s_p2).unsqueeze(-1).unsqueeze(-1)

    along = dx * cos_a + dy * sin_a
    perp  = -dx * sin_a + dy * cos_a

    hl            = s_p1.unsqueeze(-1).unsqueeze(-1)
    along_clamped = torch.minimum(torch.maximum(along, -hl), hl)

    dist = ((along - along_clamped).square() + perp.square()).add(1e-12).sqrt()
    return torch.sigmoid(
        (s_p3.unsqueeze(-1).unsqueeze(-1) - dist)
        / s_sigma.unsqueeze(-1).unsqueeze(-1)
    )


# ── Core compositing ──────────────────────────────────────────────────────────

def _composite(
    coverage:  torch.Tensor,   # (B, K, H, W) — already depth-sorted
    s_exists:  torch.Tensor,   # (B, K)
    s_r:       torch.Tensor,
    s_g:       torch.Tensor,
    s_b:       torch.Tensor,
    s_alpha:   torch.Tensor,
    B: int, K: int, H: int, W: int,
    device,
    bg_color: float,
) -> torch.Tensor:
    """
    Porter-Duff front-to-back alpha compositing over a background.
    Returns (B, 3, H, W) canvas in [0, 1].
    """
    eff_alpha  = (
        s_exists.unsqueeze(-1).unsqueeze(-1)
        * s_alpha.unsqueeze(-1).unsqueeze(-1)
        * coverage
    )

    one_minus  = 1.0 - eff_alpha
    incl_prod  = torch.cumprod(one_minus, dim=1)
    ones       = torch.ones(B, 1, H, W, device=device)
    excl_prod  = torch.cat([ones, incl_prod[:, :-1]], dim=1)
    weight     = eff_alpha * excl_prod

    colours    = torch.stack([s_r, s_g, s_b], dim=2).unsqueeze(-1).unsqueeze(-1)
    colour_out = (colours * weight.unsqueeze(2)).sum(dim=1)

    remaining  = incl_prod[:, -1]
    canvas     = colour_out + bg_color * remaining.unsqueeze(1)
    return canvas.clamp(0.0, 1.0)


def _make_grids(H: int, W: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.linspace(0.0, 1.0, H, device=device)
    xs = torch.linspace(0.0, 1.0, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x[None, None], grid_y[None, None]   # each (1, 1, H, W)


def _coverage_from_params(
    p:       dict[str, torch.Tensor],   # already depth-sorted activated params
    grid_x:  torch.Tensor,
    grid_y:  torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-pixel coverage for every slot, handling circle or mixed mode.
    Uses per-slot sharpness sigma from p["sharpness"].
    Returns (B, K, H, W).
    """
    sigma = p["sharpness"]   # (B, K)
    if config.PRIMITIVE == "circle":
        return _circle_coverage(p["cx"], p["cy"], p["radius"], sigma,
                                grid_x, grid_y)
    else:  # "mixed"
        circ = _circle_coverage(p["cx"], p["cy"], p["p1"], sigma, grid_x, grid_y)
        line = _line_coverage(p["cx"], p["cy"], p["p1"], p["p2"], p["p3"], sigma,
                              grid_x, grid_y)
        t    = p["type"].unsqueeze(-1).unsqueeze(-1)
        return (1.0 - t) * circ + t * line


# ── Flow compositing ──────────────────────────────────────────────────────────

def _composite_flow(
    coverage:   torch.Tensor,   # (B, K, H, W) depth-sorted
    s_exists:   torch.Tensor,   # (B, K)
    s_flow_x:   torch.Tensor,   # (B, K)  Δcx per slot (normalised coords)
    s_flow_y:   torch.Tensor,   # (B, K)  Δcy per slot
    s_alpha:    torch.Tensor,   # (B, K)
    B: int, K: int, H: int, W: int,
    device,
) -> torch.Tensor:
    """
    Porter-Duff front-to-back compositing of per-slot flow vectors.
    Background flow = 0 (unoccluded background is assumed stationary).
    Returns (B, 2, H, W).
    """
    eff_alpha = (
        s_exists.unsqueeze(-1).unsqueeze(-1)
        * s_alpha.unsqueeze(-1).unsqueeze(-1)
        * coverage
    )
    one_minus  = 1.0 - eff_alpha
    incl_prod  = torch.cumprod(one_minus, dim=1)
    ones       = torch.ones(B, 1, H, W, device=device)
    excl_prod  = torch.cat([ones, incl_prod[:, :-1]], dim=1)
    weight     = eff_alpha * excl_prod                             # (B, K, H, W)

    flows      = torch.stack([s_flow_x, s_flow_y], dim=2)         # (B, K, 2)
    flows      = flows.unsqueeze(-1).unsqueeze(-1)                 # (B, K, 2, 1, 1)
    flow_out   = (flows * weight.unsqueeze(2)).sum(dim=1)          # (B, 2, H, W)
    return flow_out   # background contributes 0 automatically


# ── Public API ────────────────────────────────────────────────────────────────

def rasterize(
    slots:      torch.Tensor,
    image_size: int   = config.IMAGE_SIZE,
    bg_color:   float = config.BG_COLOR,
) -> torch.Tensor:
    """
    Differentiably render primitive slots onto a canvas.

    Args:
        slots:      (B, K, SLOT_DIM) raw slot logits from O.
        image_size: height = width of the output canvas (pixels).
        bg_color:   scalar background fill in [0, 1].

    Returns:
        canvas: (B, 3, H, W) float32 in [0, 1].
    """
    B, K, _ = slots.shape
    H = W    = image_size
    device   = slots.device

    depth_idx  = config.SLOT_DIM - 1
    _, order   = slots[..., depth_idx].sort(dim=1, descending=True)
    idx_expand = order.unsqueeze(-1).expand(B, K, config.SLOT_DIM)
    sorted_s   = slots.gather(1, idx_expand)

    params = decode_slots(sorted_s)

    grid_x, grid_y = _make_grids(H, W, device)
    coverage        = _coverage_from_params(params, grid_x, grid_y)

    return _composite(
        coverage = coverage,
        s_exists = params["exists"],
        s_r      = params["r"],
        s_g      = params["g"],
        s_b      = params["b"],
        s_alpha  = params["alpha"],
        B=B, K=K, H=H, W=W, device=device, bg_color=bg_color,
    )


def rasterize_from_params(
    params:     dict[str, torch.Tensor],
    image_size: int   = config.IMAGE_SIZE,
    bg_color:   float = config.BG_COLOR,
) -> torch.Tensor:
    """
    Render from a dict of *already-activated* slot parameters.

    Used in Phase 2 after SlotTransition outputs next-step params — avoids
    converting back to logit space before rendering.

    Args:
        params:     dict matching config.SLOT_KEYS.  All values are (B, K)
                    in their activated ranges (including sharpness ∈
                    [EDGE_SIGMA_MIN, EDGE_SIGMA_MAX]).
        image_size, bg_color: same as rasterize().

    Returns:
        canvas: (B, 3, H, W) float32 in [0, 1].
    """
    B = params["exists"].shape[0]
    K = params["exists"].shape[1]
    H = W   = image_size
    device  = params["exists"].device

    _, order = params["depth"].sort(dim=1, descending=True)

    def _sort(t: torch.Tensor) -> torch.Tensor:
        return t.gather(1, order)

    sorted_p = {k: _sort(v) for k, v in params.items()}

    grid_x, grid_y = _make_grids(H, W, device)
    coverage        = _coverage_from_params(sorted_p, grid_x, grid_y)

    return _composite(
        coverage = coverage,
        s_exists = sorted_p["exists"],
        s_r      = sorted_p["r"],
        s_g      = sorted_p["g"],
        s_b      = sorted_p["b"],
        s_alpha  = sorted_p["alpha"],
        B=B, K=K, H=H, W=W, device=device, bg_color=bg_color,
    )


def rasterize_flow_from_params(
    params_prev: dict[str, torch.Tensor],
    params_next: dict[str, torch.Tensor],
    image_size:  int = config.IMAGE_SIZE,
) -> torch.Tensor:
    """
    Render a 2-channel optical-flow canvas from two consecutive slot states.

    Each pixel gets the (Δcx, Δcy) of whichever slot covers it in params_prev,
    composited front-to-back using the same depth ordering and alpha as the RGB
    canvas.  Pixels covered by no slot get flow = 0 (stationary background).

    Flow values are in normalised image coordinates (1.0 = full image width/height).
    Typical per-step magnitudes at 6 fps are well within [-0.5, 0.5].

    Args:
        params_prev: dict of activated slot params at time t   (B, K each).
        params_next: dict of activated slot params at time t+1 (B, K each).
        image_size:  spatial resolution.

    Returns:
        flow: (B, 2, H, W) — channel 0 = Δx (rightward), channel 1 = Δy (downward).
    """
    B      = params_prev["exists"].shape[0]
    K      = params_prev["exists"].shape[1]
    H = W  = image_size
    device = params_prev["exists"].device

    # Depth sort by params_prev so coverage and flow are consistently ordered.
    _, order = params_prev["depth"].sort(dim=1, descending=True)

    def _sort(t: torch.Tensor) -> torch.Tensor:
        return t.gather(1, order)

    sorted_prev = {k: _sort(v) for k, v in params_prev.items()}
    sorted_next = {k: _sort(v) for k, v in params_next.items()}

    grid_x, grid_y = _make_grids(H, W, device)
    coverage        = _coverage_from_params(sorted_prev, grid_x, grid_y)

    flow_x = sorted_next["cx"] - sorted_prev["cx"]   # (B, K) normalised
    flow_y = sorted_next["cy"] - sorted_prev["cy"]

    return _composite_flow(
        coverage  = coverage,
        s_exists  = sorted_prev["exists"],
        s_flow_x  = flow_x,
        s_flow_y  = flow_y,
        s_alpha   = sorted_prev["alpha"],
        B=B, K=K, H=H, W=W, device=device,
    )
