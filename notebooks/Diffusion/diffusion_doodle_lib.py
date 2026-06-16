"""
=========================================================
DIFFUSION DOODLE LAB — CORE LIBRARY (DO NOT MODIFY)
=========================================================

PURPOSE
-------
This file implements a minimal diffusion system designed for
controlled deep learning experimentation.

It is NOT a research framework.
It is NOT production code.

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
  - up_proj collapses ch*2 → ch after skip concatenation,
    again before any residual block sees the signal.
  - Every ConvBlock therefore always has ch_in == ch_out
    and the residual is x + f(x) with no parameters in x.

This makes use_residual a clean single-variable switch:
toggling it adds or removes the identity shortcut and
nothing else. No confounding from learned projections.

KNOWN SIMPLIFICATIONS (intentional)
-------------------------------------
- No multi-head attention (unnecessary at 64x64)
- Time embedding is a learned lookup table, not sinusoidal.
  This is simpler to understand and sufficient for T=100.
- Single-channel (grayscale) images only
- Timestep injected at bottleneck only. Real diffusion models
  inject at every stage. Sufficient here given T=100 and
  simple dataset, but not how production models work.
=========================================================
"""

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

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
    skip_scale : float
        Multiplier for skip connections before fusion in the decoder.
        Larger values give more weight to the skip features, smaller values give more weight to the upsampled features. 
        If you want the model to use the bottleneck features more and rely less on the skip connections, you can set this to a value less than 1 (e.g., 0.3). 
    """
    depth: int            = 3
    base_channels: int    = 8
    use_residual: bool    = True
    use_norm: bool        = True
    depth_per_stage: int  = 2
    skip_scale: float       = 0.5   # multiplier for skip connections before fusion


# =========================================================
# 2. BUILDING BLOCKS
# =========================================================

class ConvBlock(nn.Module):
    """
    Single conv → (optional norm) → activation block.

    IMPORTANT: ch_in always equals ch_out by construction.
    The caller (StageBlock, SmallUNet) is responsible for
    ensuring this. See input_proj and up_proj in SmallUNet.

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

        self.act = nn.SiLU()
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
        # via input_proj and up_proj in SmallUNet.
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
    by a linear projection into the channel dimension. This is
    simpler than sinusoidal embeddings and sufficient for T=100.

    SIMPLIFICATION NOTE
    -------------------
    This lab injects the timestep embedding only at the bottleneck.
    Real diffusion models (DDPM, latent diffusion) inject it at
    every stage so every layer knows the noise level directly.
    Single-point injection is sufficient here given T=100 and
    simple doodle data, but is not how production models work.
    """

    def __init__(self, T: int, ch: int):
        super().__init__()
        self.table = nn.Embedding(T, ch)
        self.proj  = nn.Linear(ch, ch)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer timestep indices
        # returns: (B, ch)
        return self.proj(self.table(t))


# =========================================================
# 4. SMALL U-NET
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
      - StageBlock processes at current resolution
      - skip connection saved
      - avg_pool2d halves spatial dims
      - up_proj doubles channel count (ch → ch*2)

    At each up stage:
      - bilinear upsample doubles spatial dims
      - up_proj halves channel count to match skip (ch → ch//2)
      - skip concatenated on channel axis (ch//2 + ch//2 = ch)
      - StageBlock processes the combined feature map

    Residual design
    ---------------
    All ConvBlocks see ch_in == ch_out. Channel changes happen
    only via parameter-free operations or dedicated pointwise
    convs that live outside the residual path:

      input_proj  : nn.Conv2d(1, base_channels, 1)  — lifts input
                    channels before any StageBlock sees the signal.
      down_projs  : nn.Conv2d(ch, ch*2, 1) per stage — doubles
                    channels after pooling, outside residual path.
      up_projs    : nn.Conv2d(ch, ch//2, 1) per stage — halves
                    channels before skip concat, outside residual.

    This ensures use_residual is a clean single-variable switch.

    Timestep conditioning
    ---------------------
    A learned embedding for t is added to the bottleneck
    feature map (broadcast over spatial dims). This lets the
    model behave differently at different noise levels.

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

        self.cfg   = cfg
        ch         = cfg.base_channels
        dps        = cfg.depth_per_stage
        skip_scale        = cfg.skip_scale
        
        
        self.skip_scale = skip_scale   # multiplier for skip connections before fusion 

        # ── input projection ──────────────────────────────────────
        # Lifts 1 grayscale channel to base_channels before any
        # StageBlock sees the signal. This ensures all ConvBlocks
        # have ch_in == ch_out and residuals are pure identities.
        self.input_proj = nn.Conv2d(1, ch, kernel_size=1, bias=False)

        # ── encoder ───────────────────────────────────────────────
        # Stage i processes at channel count ch * 2^i.
        # After each stage: pool spatially, double channels.
        self.in_stage = StageBlock(ch, cfg.use_norm, cfg.use_residual, dps)

        self.down_stages = nn.ModuleList()
        self.down_projs  = nn.ModuleList()   # ch → ch*2 after each pool

        ch_now = ch
        for _ in range(cfg.depth):
            self.down_projs.append(
                nn.Conv2d(ch_now, ch_now * 2, kernel_size=1, bias=False)
            )
            ch_now *= 2
            self.down_stages.append(
                StageBlock(ch_now, cfg.use_norm, cfg.use_residual, dps)
            )

        # ch_now is now base_channels * 2^depth — the bottleneck width

        # ── timestep embedding ────────────────────────────────────
        # Projected into bottleneck channel dimension.
        self.time_emb = TimeEmbedding(T=100, ch=ch_now)

        # ── decoder ───────────────────────────────────────────────
        # Each up stage: halve channels via up_proj, concat skip
        # (restores ch), then StageBlock at that ch.
        self.up_stages = nn.ModuleList()
        self.up_projs  = nn.ModuleList()   # ch → ch//2 before skip concat

        for _ in range(cfg.depth):
            self.up_projs.append(
                nn.Conv2d(ch_now, ch_now // 2, kernel_size=1, bias=False)
            )
            ch_now //= 2
            # after concat with skip: ch_now + ch_now = ch_now * 2
            # but we process at ch_now * 2 then the next up_proj halves again.
            # StageBlock sees ch_now channels (after up_proj, before concat
            # and after concat is handled by the *next* up_proj).
            #
            # Actually: up_proj gives ch_now, skip is also ch_now,
            # concat gives ch_now*2, StageBlock must handle ch_now*2.
            # So StageBlock here takes ch_now * 2.
            self.up_stages.append(
                StageBlock(ch_now * 2, cfg.use_norm, cfg.use_residual, dps)
            )

        # Wait — this means StageBlock sees ch_in != ch_out again.
        # Fix: add a post-concat proj that collapses ch*2 → ch
        # before StageBlock, keeping all StageBlocks at fixed ch.
        # Rebuild properly below.

        # ── rebuild decoder correctly ─────────────────────────────
        # Pattern per up stage:
        #   upsample → up_proj (ch_now → ch_now//2) → concat skip
        #   → post_concat_proj (ch_now → ch_now//2) → StageBlock(ch_now//2)
        #
        # This ensures StageBlock always sees ch_in == ch_out.

        del self.up_stages, self.up_projs

        ch_now = cfg.base_channels * (2 ** cfg.depth)   # reset to bottleneck

        self.up_projs        = nn.ModuleList()   # ch → ch//2
        self.up_stages       = nn.ModuleList()   # StageBlock at ch//2

        for _ in range(cfg.depth):
            ch_out = ch_now // 2
            self.up_projs.append(
                nn.Conv2d(ch_now, ch_out, kernel_size=1, bias=False)
            )
            # after concat: ch_out (upsampled) + ch_out (skip) = ch_now
 
            self.up_stages.append(
                StageBlock(ch_out, cfg.use_norm, cfg.use_residual, dps)
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
        t : (B,)              integer timestep indices

        Returns
        -------
        pred_noise : (B, 1, 64, 64)
        """
        # Lift input channels: 1 → base_channels
        h = self.input_proj(x)          # (B, ch, 64, 64)

        # Encoder
        h = self.in_stage(h)

        skips = []
        for down_proj, down_stage in zip(self.down_projs, self.down_stages):
            skips.append(h)                           # save before pool+proj
            h = F.avg_pool2d(h, 2)                   # halve spatial
            h = down_proj(h)                          # double channels
            h = down_stage(h)                         # process

        # Bottleneck: inject timestep
        # t_emb: (B, ch_bottleneck) → (B, ch_bottleneck, 1, 1)
        t_emb = self.time_emb(t)[:, :, None, None]
        h = h + t_emb

        # Decoder
        for up_proj, up_stage in zip(self.up_projs, self.up_stages):

            # (B, C, H, W) → (B, 2C, 2H, 2W)
            h = F.interpolate(
                h,
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            )

            # (B, 2C, 2H, 2W) → (B, C, 2H, 2W)
            # bottleneck channels → skip channels
            h = up_proj(h)

            # skip from encoder at same resolution
            # (B, C, 2H, 2W)
            skip = skips.pop()

            # scaled skip fusion
            # (B, C, 2H, 2W) + (B, C, 2H, 2W) → (B, C, 2H, 2W)
            h = h + self.skip_scale * skip

            # StageBlock keeps channel size fixed
            # (B, C, 2H, 2W) → (B, C, 2H, 2W)
            h = up_stage(h)

        return self.out_conv(h)


def build_model(cfg: ModelConfig) -> nn.Module:
    """Construct a SmallUNet from a ModelConfig."""
    return SmallUNet(cfg)


# =========================================================
# 5. DIFFUSION PROCESS
# =========================================================

class NoiseSchedule:
    """
    Fixed linear DDPM noise schedule.

    All tensors are stored on CPU and moved to the correct
    device inside q_sample / sample via the input tensor's device.

    Attributes (all CPU tensors, shape (T,))
    -----------------------------------------
    betas     : noise added at each step
    alphas    : 1 - betas
    alpha_bar : cumulative product of alphas (signal retention)
    """

    def __init__(self, T: int = 100):
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
# 6. TRAINING STEP
# =========================================================

def train_step(
    model: nn.Module,
    batch: torch.Tensor,
    schedule: NoiseSchedule,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, Dict[str, float]]:
    """
    Single gradient update step.

    The schedule must already be on the same device as batch/model.
    Use schedule.to(device) once before the training loop.

    Returns
    -------
    loss      : float
        MSE between predicted and actual noise.

    grad_norms : dict[str, float]
        Per-module-group gradient L2 norms, plus 'global'.
        Keys: 'input_proj', 'encoder', 'bottleneck',
              'decoder', 'output', 'global'.

        Watching individual keys reveals which parts of the
        network are learning, stalling, or exploding — a single
        global scalar hides this entirely.

    Teaching notes
    --------------
    - Stable training → all norms steady across steps
    - use_norm=False  → encoder/decoder norms may spike or drift
    - use_residual=False + depth_per_stage>=2 → encoder norms
      decay toward zero while decoder norms stay healthy
    - Near-zero encoder norms for many steps then a sudden jump
      = frozen early layers (visible in EXP4)
    - Large global norm spikes often precede loss divergence
    """
    model.train()
    optimizer.zero_grad()

    B = batch.size(0)
    t = torch.randint(0, schedule.T, (B,), device=batch.device)
    t_bucket = (t // 100) # bucket timesteps into groups of 100 for logging stability

    xt, noise = q_sample(batch, t, schedule)
    pred      = model(xt, t)
    loss      = F.mse_loss(pred, noise)

    loss.backward()

    # ── per-group gradient norms ──────────────────────────────────
    def _norm(params):
        sq = sum(
            p.grad.data.norm(2).item() ** 2
            for p in params if p.grad is not None
        )
        return math.sqrt(sq)

    # Only SmallUNet exposes named groups; fall back to global only.
    if hasattr(model, 'input_proj'):
        grad_norms = {
            'input_proj' : _norm(model.input_proj.parameters()),
            'encoder'    : _norm(
                list(model.in_stage.parameters()) +
                list(model.down_stages.parameters()) +
                list(model.down_projs.parameters())
            ),
            'bottleneck' : _norm(model.time_emb.parameters()),
            'decoder'    : _norm(
                list(model.up_stages.parameters()) +
                list(model.up_projs.parameters())
            )   ,
            'output'     : _norm(model.out_conv.parameters()),
        }
        total_sq = sum(v ** 2 for v in grad_norms.values())
        grad_norms['global'] = math.sqrt(total_sq)
    else:
        # Fallback for non-SmallUNet models
        global_norm = _norm(model.parameters())
        grad_norms  = {'global': global_norm}

    optimizer.step()

    return loss.item(), grad_norms, t_bucket.detach().cpu()


# =========================================================
# 7. SAMPLING (REVERSE DIFFUSION)
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
    schedule    : NoiseSchedule (will be moved to device here)
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
# 8. MNIST DATASET CACHED TO DISK
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
        images = payload["images"]
        return CachedMNIST64(images)

    resize = transforms.Resize((64, 64), antialias=True)
    to_tensor = transforms.ToTensor()

    raw = datasets.MNIST(
        root=str(Path(cache_dir).parent),
        train=train,
        download=True,
    )

    images = []
    for img, _label in raw:
        x = to_tensor(img)          # (1, 28, 28), [0, 1]
        x = resize(x)               # (1, 64, 64)
        x = x * 2.0 - 1.0           # [-1, 1]
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
