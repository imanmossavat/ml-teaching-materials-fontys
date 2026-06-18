"""
=========================================================
DIFFUSION DOODLE LAB — CORE LIBRARY  
=========================================================
Iman Mosavat — 2026-06-18
PURPOSE
-------
This file implements a minimal diffusion system designed for
controlled deep learning experimentation.

It is a deterministic experimental substrate for studying:

  - optimization stability
  - gradient behavior (per-stage grad norms)
  - normalization effects
  - architectural depth sensitivity
  - residual connection utility
  - mismatch between loss and perceptual quality

DESIGN PRINCIPLES
-----------------
1. Minimal but correct diffusion implementation
2. All tensors stay on the correct device throughout
3. One variable changes per experiment (see lab script)
4. No experiment logic, logging, or analysis lives here
5. Only stable, tested APIs are exposed

ARCHITECTURE NOTES
------------------
Channel count doubles at each downsampling stage, following
the standard U-Net convention. base_channels is the channel
count at the first encoder stage. The bottleneck has
base_channels * 2^depth channels.

  Example: base_channels=8, depth=3
    64x64  →   8 channels
    32x32  →  16 channels
    16x16  →  32 channels
     8x8   →  64 channels  ← bottleneck (8*8*64 = 4096 values,
                               same total as 64*64*1 input)

This is the architecturally honest pattern: as spatial
resolution shrinks, channel capacity grows to compensate.
The bottleneck is the richest representational point in the
network, not the poorest.

RESIDUAL CONNECTIONS
--------------------
All residual connections are pure identity shortcuts — no
learned projection anywhere. This is enforced by design:

  - input_proj lifts 1 → base_channels before any residual
    block sees the signal, so the first ConvBlock always
    sees ch_in == ch_out.
  - GatedSkipFusion handles skip concatenation, meaning
    StageBlocks always see ch_in == ch_out.
  - Every ConvBlock therefore always has ch_in == ch_out
    and the residual is x + f(x) with no parameters in x.

This makes use_residual a clean single-variable switch:
toggling it adds or removes the identity shortcut and
nothing else. No confounding from learned projections.

TIME EMBEDDING INJECTION
------------------------
The timestep is injected at every encoder/decoder stage, not
only the bottleneck. Each stage has a small linear layer that
projects the shared time embedding vector (from TimeEmbedding)
into the stage's channel width. This means every conv knows
the current noise level directly, which is how production
diffusion models work.

GATED SKIP FUSION
-----------------
Skip connections are no longer simple additions. At each
decoder stage:
  1. The upsampled feature map and the skip are concatenated
     along the channel axis.
  2. A two-layer pointwise MLP (1×1 convolutions) fuses them.
  3. A sigmoid gate, modulated by the time embedding, controls
     how much of the fused skip information passes through.
This replaces the previous skip_scale scalar and lets the
network learn when (and at which timesteps) to trust skip
features vs bottleneck features.

T CONSISTENCY
-------------
ModelConfig.T is the single source of truth for the diffusion
timestep count. Both NoiseSchedule and TimeEmbedding receive
T from the same ModelConfig, making it impossible for them to
diverge.

KNOWN SIMPLIFICATIONS (intentional)
-------------------------------------
- No multi-head attention (unnecessary at 64x64)
- Time embedding is a learned lookup table, not sinusoidal.
  This is simpler to understand and sufficient for small T.
- Single-channel (grayscale) images only
=========================================================
"""


"""
=========================================================
CHANGELOG
=========================================================

2026-06-18  — v1.3.0

  T consistency
    ModelConfig.T is now the single source of truth for the
    diffusion timestep count. Use get_schedule(cfg) to build
    a NoiseSchedule guaranteed to match the model embedding
    table. Direct NoiseSchedule(T=...) construction with a
    hand-typed value is discouraged.

  Time embedding at every stage
    TimeEmbedding now outputs a shared vector (dim =
    bottleneck channels). Per-stage TimeProject modules map
    it to each stage's channel width and inject it before
    every StageBlock in both encoder and decoder, not only
    at the bottleneck.

  Gated skip fusion (GatedSkipFusion)
    Skip connections are no longer scaled additions. Each
    decoder stage concatenates the upsampled map and the
    skip, fuses them through a two-layer pointwise MLP, then
    applies a sigmoid gate whose bias is driven by the time
    embedding. The network learns to suppress skip features
    at high noise levels and amplify them at low noise levels.
    ModelConfig.skip_scale removed.

  register_hook guard
    Bottleneck gradient hook now checks h.requires_grad
    before attaching, preventing a RuntimeError during
    no_grad eval and sampling forwards.
=========================================================
"""


import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict

from scipy.fftpack import shift
from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


# =========================================================
# 1. MODEL CONFIG
# =========================================================

@dataclass
class ModelConfig:
    """
    Controls all architectural variation.

    Each field maps to exactly one architectural decision.
    Change ONE field per experiment to isolate its effect.

    Fields
    ------
    T : int
        Number of diffusion timesteps.  This is the single
        source of truth shared by both the model's
        TimeEmbedding table and NoiseSchedule.  Pass a
        ModelConfig to get_schedule() to build a matching
        NoiseSchedule — never construct NoiseSchedule(T=...)
        directly with a different value.

    depth : int
        Number of down/up stages. Each stage halves spatial
        resolution and doubles channel count.
        Must satisfy: 64 % (2**depth) == 0.
        Safe values: 1, 2, 3, 4.
        depth=5 → 2px bottleneck (very compressed).
        depth=6 → 1px bottleneck (almost certainly broken).

    base_channels : int
        Channel count at the first encoder stage.
        Doubles at each subsequent stage.
        Must be divisible by 8 so GroupNorm works at all stages.
        Bottleneck has base_channels * 2^depth channels.
        Example: base_channels=8, depth=3 → 64ch bottleneck.

    use_residual : bool
        Whether ConvBlocks add a residual (skip) connection.
        All residuals are pure identity — no learned parameters
        in the shortcut path. Toggling this changes exactly
        one thing: whether x is added back to f(x).
        When False, gradients must traverse every conv in a
        stage sequentially — with depth_per_stage >= 2 this
        produces visible gradient decay.

    use_norm : bool
        Whether to apply GroupNorm after each conv.
        When False, activations can grow unboundedly across
        depth, destabilizing training.

    depth_per_stage : int
        Number of ConvBlocks stacked within each encoder/
        decoder stage. Spatial resolution does not change
        within a stage.

        At 1: block-level residuals are redundant because
        the U-Net skip connections already provide gradient
        highways across stages. Removing residuals has no
        visible effect (EXP2 is a non-experiment).

        At 2+: the intra-stage path is genuinely deep. The
        block-level residual becomes the only gradient highway
        within each stage. Removing it has measurable and
        visible consequences.

        Do not set this to 1 unless you are deliberately
        demonstrating the U-Net skip confounder.
    """
    T: int                = 1000
    depth: int            = 3
    base_channels: int    = 8
    use_residual: bool    = True
    use_norm:     bool    = True
    depth_per_stage: int  = 2,

DEBUG = False

def debug(x, DEBUG, name):
    if DEBUG:
        print(f"{name:>18}: {tuple(x.shape)}")
# =========================================================
# 2. BUILDING BLOCKS
# =========================================================

class ConvBlock(nn.Module):
    """
    Single conv → (optional norm) → activation block.

    IMPORTANT: ch_in always equals ch_out by construction.
    The caller (StageBlock, SmallUNet) is responsible for
    ensuring this. See input_proj and GatedSkipFusion in
    SmallUNet.

    The residual is always a pure identity: out = f(x) + x.
    There is no learned projection in the shortcut path.
    This makes use_residual a clean single-variable switch.

    Observations this enables
    -------------------------
    use_norm=False       → activation magnitude grows with depth;
                           watch per-stage grad norms diverge.
    use_residual=False   → gradient must traverse every conv in
                           the stage sequentially; with
                           depth_per_stage >= 2 watch grad norms
                           decay toward zero in deep stages.
    """

    def __init__(self, ch: int, use_norm: bool, use_residual: bool):
        super().__init__()

        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

        # GroupNorm requires num_channels divisible by num_groups.
        # We guard here so students can freely vary base_channels.
        if use_norm:
            num_groups = min(8, ch)
            while ch % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            self.norm = nn.GroupNorm(num_groups, ch)
        else:
            self.norm = nn.Identity()

        self.act          = nn.SiLU()
        self.use_residual = use_residual
        # No projection ever needed: ch_in == ch_out by construction.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm(self.conv(x)))
        if self.use_residual:
            out = out + x   # pure identity shortcut, zero parameters
        return out


class StageBlock(nn.Module):
    """
    A stack of depth_per_stage ConvBlocks forming one encoder/
    decoder stage. Spatial resolution does not change here —
    pooling and upsampling happen outside this block.

    WHY THIS EXISTS
    ---------------
    A single ConvBlock per stage means each stage is only one
    conv deep. The U-Net skip connections already provide
    gradient highways across stages, so block-level residuals
    have almost nothing left to do within a stage.

    With depth_per_stage >= 2 the intra-stage path becomes
    genuinely deep. The block-level residual is now the only
    gradient highway within each stage. Removing use_residual
    has real consequences: gradients must traverse
    depth_per_stage convs sequentially, and without the
    shortcut they will visibly decay (or explode without norm).

    depth_per_stage=1  →  residual effect invisible
    depth_per_stage=2  →  residual effect starts to appear
    depth_per_stage=3  →  residual effect clearly visible
    """

    def __init__(
        self,
        ch: int,
        use_norm: bool,
        use_residual: bool,
        depth_per_stage: int,
    ):
        super().__init__()
        # All blocks are ch → ch. Channel changes happen outside
        # via input_proj and GatedSkipFusion in SmallUNet.
        self.blocks = nn.Sequential(*[
            ConvBlock(ch, use_norm, use_residual)
            for _ in range(depth_per_stage)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# =========================================================
# 3. TIME EMBEDDING
# =========================================================

class TimeEmbedding(nn.Module):
    """
    Learned timestep embedding table.

    WHY THIS MATTERS
    ----------------
    Without timestep conditioning the model must learn a single
    function that denoises every noise level simultaneously.
    That is a much harder problem and produces worse samples.

    With conditioning the model learns: "at timestep t the image
    looks like X, so I should predict noise Y."

    Implementation: a simple lookup table (nn.Embedding) followed
    by a linear projection into a shared embedding dimension.
    Each stage then has a dedicated linear layer that projects
    the shared embedding into that stage's channel width, so
    every layer in the network knows the noise level directly.

    WHY A SHARED EMBEDDING DIMENSION?
    ----------------------------------
    All stages share the same embedding table but have different
    channel counts. A single linear per stage projects from the
    shared dim to the stage's ch. This is cheaper than having a
    separate embedding table per stage.

    T IS FIXED BY ModelConfig
    -------------------------
    The embedding table has exactly T rows.  T comes from
    ModelConfig.T and must match the NoiseSchedule used during
    training.  Use get_schedule(cfg) to build a guaranteed-
    consistent NoiseSchedule — never pass a different T.
    """

    def __init__(self, T: int, emb_dim: int):
        """
        Parameters
        ----------
        T       : number of diffusion timesteps (from ModelConfig.T)
        emb_dim : shared embedding dimension (set to bottleneck ch
                  in SmallUNet so zero extra parameters are needed
                  at the bottleneck stage)
        """
        super().__init__()
        self.table = nn.Embedding(T, emb_dim)
        self.proj  = nn.Linear(emb_dim, emb_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : (B,) integer timestep indices, range [0, T)

        Returns
        -------
        emb : (B, emb_dim) shared time embedding
        """
        return self.proj(self.table(t))   # (B, emb_dim)


class TimeProject(nn.Module):
    """
    Produces FiLM parameters (scale, shift) for a stage.
    """

    def __init__(self, emb_dim: int, ch: int):
        super().__init__()
        self.linear = nn.Linear(emb_dim, 2 * ch)

    def forward(self, emb: torch.Tensor):
        gate_scale, gate_shift = self.linear(emb).chunk(2, dim=1)

        gate_scale = gate_scale[:, :, None, None]
        gate_shift = gate_shift[:, :, None, None]

        return gate_scale, gate_shift


# =========================================================
# 4. GATED SKIP FUSION
# =========================================================

class GatedSkipFusion(nn.Module):
    """
    Fuses an upsampled feature map with its U-Net skip connection
    via a learned MLP and a time-gated sigmoid.

    WHY THIS REPLACES SIMPLE ADDITION
    -----------------------------------
    A fixed scalar (skip_scale) applies the same weight to every
    skip, at every timestep, at every spatial position.  A gated
    MLP can learn:
      - at high noise levels (large t): trust the bottleneck more,
        suppress skip features (they carry mostly noise).
      - at low noise levels (small t): trust skip features more,
        they carry fine structural detail.
    The time gate makes this noise-level-aware automatically.

    ARCHITECTURE
    ------------
    Inputs:
      h    : (B, ch, H, W)  upsampled + projected feature map
      skip : (B, ch, H, W)  encoder skip at this resolution
      gate_bias : (B, ch)   time-projected gate bias from TimeProject

    Step 1 — concatenate:
      fused = cat([h, skip], dim=1)   # (B, 2ch, H, W)

    Step 2 — pointwise MLP (two 1×1 convolutions):
      fused = act(norm(conv1(fused)))  # (B, ch, H, W)
      fused = norm(conv2(fused))       # (B, ch, H, W)  (no act yet)

    Step 3 — time gate:
      gate = sigmoid(fused + gate_bias[:, :, None, None])
      out  = gate * fused

    The gate broadcasts over spatial dims.  Its bias comes from
    the shared time embedding projected to ch, so the network
    can learn to suppress or amplify skip information as a
    function of the current noise level.

    RESIDUAL INTEGRITY
    ------------------
    This module sits outside all ConvBlocks, so it does not
    affect the use_residual single-variable switch.
    """

    def __init__(self, ch: int, use_norm: bool, emb_dim: int):
        """
        Parameters
        ----------
        ch       : channel count of both h and skip (they must match)
        use_norm : whether to apply GroupNorm (mirrors cfg.use_norm)
        emb_dim  : shared time embedding dimension (for gate projection)
        """
        super().__init__()

        # MLP: 2ch → ch → ch  (pointwise, no spatial mixing)
        self.conv1 = nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(ch,     ch, kernel_size=1, bias=True)

        if use_norm:
            num_groups = min(8, ch)
            while ch % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            self.norm1 = nn.GroupNorm(num_groups, ch)
            self.norm2 = nn.GroupNorm(num_groups, ch)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.act = nn.SiLU()

        # Gate bias: shared time emb → ch
        self.gate_proj = TimeProject(emb_dim, ch)

    def forward(
        self,
        h:   torch.Tensor,
        skip: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h     : (B, ch, H, W)
        skip  : (B, ch, H, W)
        t_emb : (B, emb_dim)  shared time embedding

        Returns
        -------
        (B, ch, H, W)
        """
        # --- Step 1: concatenate ---
        # concatenate decoder feature and encoder skip
        
        fused = torch.cat([h, skip], dim=1)
        debug(fused, DEBUG, "fused skip cat")

        # fuse them
        fused = self.act(self.norm1(self.conv1(fused)))
        fused = self.norm2(self.conv2(fused))
        debug(fused, DEBUG, "fused skip after MLP")
        # time conditioning (same FiLM pattern used everywhere else)
        scale, shift = self.gate_proj(t_emb)

        fused = fused * (1.0 + scale) + shift

        # gate
        gate = torch.sigmoid(fused)

        out = gate * fused
        debug(out, DEBUG, "output after gate")  

        return out


# =========================================================
# 5. SMALL U-NET
# =========================================================

class SmallUNet(nn.Module):
    """
    U-Net for diffusion on doodles, with channel doubling.

    Architecture
    ------------
    input_proj → in_stage → [down × depth] → bottleneck
               → [up × depth] → out_conv

    Channel count at each stage:
        stage 0 (input):    base_channels
        stage 1:            base_channels * 2
        ...
        stage depth (bottleneck): base_channels * 2^depth

    At each down stage:
      - Time embedding injected (projected to stage's ch)
      - StageBlock processes at current resolution
      - skip connection saved
      - avg_pool2d halves spatial dims
      - down_proj doubles channel count (ch → ch*2)

    At each up stage:
      - bilinear upsample doubles spatial dims
      - up_proj halves channel count to match skip (ch → ch//2)
      - GatedSkipFusion merges upsampled map and skip,
        conditioned on time embedding
      - Time embedding injected (projected to stage's ch)
      - StageBlock processes the fused feature map

    Residual design
    ---------------
    All ConvBlocks see ch_in == ch_out. Channel changes happen
    only via dedicated pointwise convs that live outside the
    residual path:

      input_proj  : nn.Conv2d(1, base_channels, 1)  — lifts input
                    channels before any StageBlock sees the signal.
      down_projs  : nn.Conv2d(ch, ch*2, 1) per stage — doubles
                    channels after pooling, outside residual path.
      up_projs    : nn.Conv2d(ch, ch//2, 1) per stage — halves
                    channels before GatedSkipFusion.

    Time conditioning
    -----------------
    A single TimeEmbedding table produces a shared embedding
    vector (B, emb_dim) from the integer timestep.  Per-stage
    TimeProject layers map this to each stage's channel width.
    The embedding is injected:
      - At every encoder stage (before StageBlock)
      - At the bottleneck
      - Inside every GatedSkipFusion gate (decoder)
      - At every decoder stage (before StageBlock)

    T consistency
    -------------
    ModelConfig.T is passed directly into TimeEmbedding(T=...).
    Use get_schedule(cfg) to build a matching NoiseSchedule.
    Never construct NoiseSchedule(T=X) with a different X.

    Spatial resolution constraint
    -----------------------------
    Input must be 64×64. Each down stage halves resolution.
    After depth stages: 64 / 2^depth must be a whole number.
    Asserted at construction time.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        assert 64 % (2 ** cfg.depth) == 0, (
            f"depth={cfg.depth} incompatible with 64x64 input: "
            f"64 / 2^{cfg.depth} = {64 / 2**cfg.depth:.1f} (must be integer). "
            f"Safe values: depth ∈ {{1, 2, 3, 4, 5, 6}}."
        )

        # Bottleneck diagnostics buffer
        self._bottleneck_grad       = None
        self._bottleneck_activation = None

        self.cfg = cfg
        ch       = cfg.base_channels
        dps      = cfg.depth_per_stage

        # ── input projection ──────────────────────────────────────
        # Lifts 1 grayscale channel to base_channels before any
        # StageBlock sees the signal.
        self.input_proj = nn.Conv2d(1, ch, kernel_size=1, bias=False)

        # ── time embedding ────────────────────────────────────────
        # emb_dim = bottleneck channel count so we can inject at
        # the bottleneck without any extra projection.
        bottleneck_ch = cfg.base_channels * (2 ** cfg.depth)
        self.time_emb = TimeEmbedding(T=cfg.T, emb_dim=bottleneck_ch)

        # ── encoder ───────────────────────────────────────────────
        # Per-stage time projections: emb_dim → stage ch
        self.in_stage = StageBlock(ch, cfg.use_norm, cfg.use_residual, dps)
        self.in_time  = TimeProject(bottleneck_ch, ch)

        self.down_stages    = nn.ModuleList()
        self.down_projs     = nn.ModuleList()   # ch → ch*2 after pool
        self.down_time_proj = nn.ModuleList()   # emb_dim → stage ch

        ch_now = ch
        for _ in range(cfg.depth):
            # Channel count at the next (deeper) stage
            ch_next = ch_now * 2
            self.down_projs.append(
                nn.Conv2d(ch_now, ch_next, kernel_size=1, bias=False)
            )
            self.down_stages.append(
                StageBlock(ch_next, cfg.use_norm, cfg.use_residual, dps)
            )
            self.down_time_proj.append(
                TimeProject(bottleneck_ch, ch_next)
            )
            ch_now = ch_next

        # ch_now is now base_channels * 2^depth — the bottleneck width
        # (equals bottleneck_ch; emb_dim already matches so no extra proj)

        # ── decoder ───────────────────────────────────────────────
        self.up_projs     = nn.ModuleList()   # ch → ch//2
        self.skip_fusions = nn.ModuleList()   # GatedSkipFusion per stage
        self.up_stages    = nn.ModuleList()   # StageBlock at ch//2
        self.up_time_proj = nn.ModuleList()   # emb_dim → stage ch

        for _ in range(cfg.depth):
            ch_out = ch_now // 2
            self.up_projs.append(
                nn.Conv2d(ch_now, ch_out, kernel_size=1, bias=False)
            )
            # GatedSkipFusion fuses h (ch_out) + skip (ch_out) → ch_out
            self.skip_fusions.append(
                GatedSkipFusion(ch_out, cfg.use_norm, emb_dim=bottleneck_ch)
            )
            self.up_stages.append(
                StageBlock(ch_out, cfg.use_norm, cfg.use_residual, dps)
            )
            self.up_time_proj.append(
                TimeProject(bottleneck_ch, ch_out)
            )
            ch_now = ch_out

        # ch_now is back to base_channels

        # ── output ────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(ch_now, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, 64, 64)   noisy image at timestep t
        t : (B,)              integer timestep indices in [0, cfg.T)

        Returns
        -------
        pred_noise : (B, 1, 64, 64)
        """

        # Shared time embedding — computed once, reused at every stage
        t_emb = self.time_emb(t)   # (B, emb_dim)

        # ── Lift input channels: 1 → base_channels ──
        debug(x, DEBUG, "input x")

        h = self.input_proj(x)   # (B, ch, 64, 64)
        debug(h, DEBUG, "after input proj")

        # ── Encoder input stage ──
        # Inject time, then process
        
        scale, shift = self.in_time(t_emb)

        h = h * (1.0 + scale) + shift
        h = self.in_stage(h)

        # ── Encoder down stages ──
        skips = []
        for down_proj, down_stage, t_proj in zip(
            self.down_projs, self.down_stages, self.down_time_proj
        ):
            debug(h, DEBUG, "before down stage")
            skips.append(h)                    # save before pool+proj
            h = F.avg_pool2d(h, 2)            # halve spatial
            h = down_proj(h)
            debug(h, DEBUG, "after down proj")
            # double channels
            scale, shift = t_proj(t_emb)
            debug(scale, DEBUG, "scale after down proj")
            debug(shift, DEBUG, "shift after down proj")

            h = h * (1.0 + scale) + shift            
            h = down_stage(h)                  # process

        # ── Bottleneck: inject timestep ──
        # emb_dim == bottleneck_ch, so we add directly
        h = h + t_emb[:, :, None, None]

        if self.training and h.requires_grad:
            self._bottleneck_activation = h.detach()
            self._bottleneck_grad       = None

            def _save_grad(grad):
                self._bottleneck_grad = grad

            h.register_hook(_save_grad)

        # ── Decoder up stages ──
        for up_proj, skip_fusion, up_stage, t_proj in zip(
            self.up_projs, self.skip_fusions, self.up_stages, self.up_time_proj
        ):
            # Upsample: (B, C, H, W) → (B, C, 2H, 2W)
            h = F.interpolate(
                h,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
            debug(h, DEBUG, "decoder after upsample")
            # Project channels: bottleneck_ch//k → skip_ch
            h = up_proj(h)   # (B, ch_out, 2H, 2W)
            debug(h, DEBUG, "decoder after up proj")

            # Gated skip fusion (replaces h + skip_scale * skip)
            skip = skips.pop()                        # (B, ch_out, 2H, 2W)
            h    = skip_fusion(h, skip, t_emb)        # (B, ch_out, 2H, 2W)
            debug(h, DEBUG, "decoder after skip fusion")
            # Inject time, then process
            scale, shift = t_proj(t_emb)
            debug(scale, DEBUG, "scale after up proj")
            debug(shift, DEBUG, "shift after up proj")  

            h = h * (1.0 + scale) + shift            
            h = up_stage(h)
            debug(h, DEBUG, "decoder after up stage")

        return self.out_conv(h)


def build_model(cfg: ModelConfig) -> nn.Module:
    """Construct a SmallUNet from a ModelConfig."""
    return SmallUNet(cfg)


# =========================================================
# 6. NOISE SCHEDULE
# =========================================================

class NoiseSchedule:
    """
    Fixed linear DDPM noise schedule.

    DO NOT CONSTRUCT THIS DIRECTLY with an arbitrary T.
    Use get_schedule(cfg) to guarantee T matches the model's
    TimeEmbedding table.

    All tensors are stored on CPU and moved to the correct
    device inside q_sample / sample via the input tensor's
    device.

    Attributes (all CPU tensors, shape (T,))
    -----------------------------------------
    betas     : noise added at each step
    alphas    : 1 - betas
    alpha_bar : cumulative product of alphas (signal retention)
    """

    def __init__(self, T: int):
        self.T = T

        betas     = torch.linspace(1e-4, 0.02, T)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas     = betas
        self.alphas    = alphas
        self.alpha_bar = alpha_bar

    def to(self, device: torch.device) -> "NoiseSchedule":
        """Return a copy of this schedule with tensors on `device`."""
        s           = NoiseSchedule.__new__(NoiseSchedule)
        s.T         = self.T
        s.betas     = self.betas.to(device)
        s.alphas    = self.alphas.to(device)
        s.alpha_bar = self.alpha_bar.to(device)
        return s


def get_schedule(cfg: ModelConfig) -> NoiseSchedule:
    """
    Build a NoiseSchedule that is guaranteed to match the model.

    This is the ONLY correct way to construct a NoiseSchedule.
    It reads T from ModelConfig so the schedule and the model's
    TimeEmbedding table cannot diverge.

    Usage
    -----
        cfg      = ModelConfig(T=1000, ...)
        model    = build_model(cfg)
        schedule = get_schedule(cfg).to(device)
    """
    return NoiseSchedule(T=cfg.T)


# =========================================================
# 7. FORWARD DIFFUSION
# =========================================================

def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward diffusion: add noise to x0 according to timestep t.

    All tensors (x0, t, schedule) must be on the same device.
    Call schedule.to(device) before passing here.

    Returns
    -------
    xt    : noisy image at timestep t, same shape as x0
    noise : the Gaussian noise that was added (training target)
    """
    noise = torch.randn_like(x0)
    a_bar = schedule.alpha_bar[t].view(-1, 1, 1, 1)
    xt    = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise
    return xt, noise


# =========================================================
# 8. TRAINING STEP
# =========================================================

def train_step(
    model: nn.Module,
    batch: torch.Tensor,
    schedule: NoiseSchedule,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, Dict[str, float], torch.Tensor]:

    model._bottleneck_grad = None
    model.train()
    optimizer.zero_grad()

    B      = batch.size(0)
    device = batch.device

    # sample timesteps
    t = torch.randint(0, schedule.T, (B,), device=device)

    # forward diffusion
    xt, noise = q_sample(batch, t, schedule)

    # predict noise
    pred = model(xt, t)
    loss = F.mse_loss(pred, noise)

    loss.backward()

    # -------------------------------------------------
    # bottleneck activation gradient (TRUE signal)
    # -------------------------------------------------
    bottleneck_grad_norm = 0.0
    if (
        hasattr(model, "_bottleneck_grad")
        and model._bottleneck_grad is not None
    ):
        bottleneck_grad_norm = model._bottleneck_grad.norm().item()

    # -------------------------------------------------
    # helper: parameter grad norm
    # -------------------------------------------------
    def _norm(params):
        sq = 0.0
        for p in params:
            if p.grad is None:
                continue
            sq += p.grad.detach().pow(2).sum().item()
        return math.sqrt(sq)

    # -------------------------------------------------
    # group-wise gradient norms
    # -------------------------------------------------
    if hasattr(model, "input_proj"):

        grad_norms = {
            "input_proj": _norm(model.input_proj.parameters()),

            "encoder": _norm(
                list(model.in_stage.parameters())
                + list(model.in_time.parameters())
                + list(model.down_stages.parameters())
                + list(model.down_projs.parameters())
                + list(model.down_time_proj.parameters())
            ),

            # THIS is the real bottleneck signal (activation grad)
            "bottleneck": bottleneck_grad_norm,

            "decoder": _norm(
                list(model.up_stages.parameters())
                + list(model.up_projs.parameters())
                + list(model.skip_fusions.parameters())
                + list(model.up_time_proj.parameters())
            ),

            "output": _norm(model.out_conv.parameters()),
        }

        total_sq        = sum(v ** 2 for v in grad_norms.values())
        grad_norms["global"] = math.sqrt(total_sq)

    else:
        grad_norms = {"global": _norm(model.parameters())}

    optimizer.step()

    # bucket timestep for diagnostics
    t_bucket = (t // max(1, (schedule.T // 10))).detach().cpu()

    return loss.item(), grad_norms, t_bucket


# =========================================================
# 9. SAMPLING (REVERSE DIFFUSION)
# =========================================================

@torch.no_grad()
def sample(
    model: nn.Module,
    schedule: NoiseSchedule,
    device: torch.device,
    num_samples: int = 16,
) -> torch.Tensor:
    """
    Generate images by reversing the diffusion process.

    Starts from pure Gaussian noise and iteratively denoises.

    Parameters
    ----------
    model       : trained SmallUNet
    schedule    : NoiseSchedule — use get_schedule(cfg).to(device)
    device      : torch.device — where to run inference
    num_samples : how many images to generate

    Returns
    -------
    x : (num_samples, 1, 64, 64) tensor, pixel values in [-1, 1]

    Teaching note
    -------------
    This is the primary perceptual evaluation signal.
    A model can have a low MSE loss but still produce incoherent
    samples — watch for this in EXP3 (deep network). Loss and
    sample quality can diverge. When they do, trust the samples.
    """
    model.eval()
    sched = schedule.to(device)

    x = torch.randn(num_samples, 1, 64, 64, device=device)

    for t_val in reversed(range(sched.T)):
        t_batch = torch.full(
            (num_samples,), t_val, device=device, dtype=torch.long
        )

        pred  = model(x, t_batch)

        beta  = sched.betas[t_val]
        alpha = sched.alphas[t_val]
        a_bar = sched.alpha_bar[t_val]

        # DDPM reverse step
        x = (1.0 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1.0 - a_bar)) * pred
        )

        if t_val > 0:
            x = x + torch.randn_like(x) * beta.sqrt()

    return x


# =========================================================
# 10. MNIST DATASET CACHED TO DISK
# =========================================================

class CachedMNIST64(torch.utils.data.Dataset):
    """
    MNIST resized to 64x64 and cached on disk.

    First run:
      - download MNIST if needed
      - resize every image to 64x64
      - normalize to [-1, 1]
      - save the processed tensor to disk

    Later runs:
      - load the cached tensor directly

    This keeps the training code fast and ensures every run sees
    the exact same resized dataset.
    """

    def __init__(self, images: torch.Tensor):
        self.images = images

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.images[idx]


def _build_mnist_cache(
    cache_dir: str | Path = "data/mnist_64",
    train: bool = True,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    split = "train" if train else "test"
    return cache_dir / f"mnist64_{split}.pt"


def _load_or_create_mnist64(
    cache_dir: str | Path = "data/mnist_64",
    train: bool = True,
) -> CachedMNIST64:
    cache_path = _build_mnist_cache(cache_dir=cache_dir, train=train)

    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        images  = payload["images"]
        return CachedMNIST64(images)

    resize    = transforms.Resize((64, 64), antialias=True)
    to_tensor = transforms.ToTensor()

    raw = datasets.MNIST(
        root=str(Path(cache_dir).parent),
        train=train,
        download=True,
    )

    images = []
    for img, _label in raw:
        x = to_tensor(img)     # (1, 28, 28), [0, 1]
        x = resize(x)          # (1, 64, 64)
        x = x * 2.0 - 1.0      # [-1, 1]
        images.append(x)

    images = torch.stack(images, dim=0).contiguous()
    torch.save({"images": images}, cache_path)
    return CachedMNIST64(images)


def get_loader(batch_size: int = 32) -> torch.utils.data.DataLoader:
    """Return a DataLoader over cached 64x64 MNIST images."""
    dataset = _load_or_create_mnist64(cache_dir="data/mnist_64", train=True)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
