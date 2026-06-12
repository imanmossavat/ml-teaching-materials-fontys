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
  - gradient behavior (per-step grad norms)
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

KNOWN SIMPLIFICATIONS (intentional)
-------------------------------------
- No multi-head attention (unnecessary at 64x64)
- Time embedding is a learned lookup table, not sinusoidal.
  This is simpler to understand and sufficient for T=100.
- Single-channel (grayscale) images only
=========================================================
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        Number of down/up stages. Each stage halves/doubles
        spatial resolution. Must satisfy: 64 % (2**depth) == 0.
        Safe values: 1, 2, 3.  depth=4 → 4px bottleneck (unstable).

    base_channels : int
        Feature channels throughout the network.
        Must be divisible by 8 (GroupNorm requirement).

    use_residual : bool
        Whether ConvBlocks add a residual (skip) connection.
        When False, gradients must flow through every conv
        sequentially — expect vanishing gradients at depth >= 3.

    use_norm : bool
        Whether to apply GroupNorm after each conv.
        When False, activations can grow unboundedly across depth.
    """
    depth: int = 3
    base_channels: int = 32
    use_residual: bool = True
    use_norm: bool = True


# =========================================================
# 2. BUILDING BLOCKS
# =========================================================

class ConvBlock(nn.Module):
    """
    Single conv → (optional norm) → activation block.

    Optionally adds a residual connection. When ch_in != ch_out
    the residual is projected via a 1x1 conv so the shapes match.

    Observations this enables
    -------------------------
    use_norm=False  → activation magnitude grows with depth;
                       watch grad norms explode or vanish.
    use_residual=False → gradient must traverse every layer
                          sequentially; watch grad norms decay.
    """

    def __init__(self, ch_in: int, ch_out: int, use_norm: bool, use_residual: bool):
        super().__init__()

        self.conv = nn.Conv2d(ch_in, ch_out, 3, padding=1)

        # GroupNorm requires num_channels divisible by num_groups.
        # We guard here so students can freely vary base_channels.
        if use_norm:
            num_groups = min(8, ch_out)
            while ch_out % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            self.norm = nn.GroupNorm(num_groups, ch_out)
        else:
            self.norm = nn.Identity()

        self.act = nn.SiLU()

        self.use_residual = use_residual
        # Project residual if channel count changes
        if use_residual and ch_in != ch_out:
            self.proj = nn.Conv2d(ch_in, ch_out, 1, bias=False)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm(self.conv(x)))
        if self.use_residual:
            res = self.proj(x) if self.proj is not None else x
            out = out + res
        return out


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
    by a linear projection into the channel dimension.  This is
    simpler than sinusoidal embeddings and sufficient for T=100.
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
    Minimal U-Net for diffusion on doodles.

    Architecture
    ------------
    in_conv → [down × depth] → bottleneck → [up × depth] → out_conv

    At each down stage:
      - ConvBlock (may have residual + norm per cfg)
      - avg_pool2d halves spatial dims
      - skip connection saved

    At each up stage:
      - bilinear upsample doubles spatial dims
      - skip concatenated on channel axis
      - ConvBlock processes the combined feature map

    Timestep conditioning
    ---------------------
    A learned embedding for t is added to the bottleneck
    feature map (broadcast over spatial dims).  This lets the
    model behave differently at different noise levels.

    Spatial resolution constraint
    -----------------------------
    Input must be 64×64.  Each down stage halves resolution.
    After `depth` stages: 64 / 2^depth must be a whole number.
    Asserted at construction time.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        assert 64 % (2 ** cfg.depth) == 0, (
            f"depth={cfg.depth} incompatible with 64x64 input: "
            f"64 / 2^{cfg.depth} = {64 / 2**cfg.depth:.1f} (must be integer). "
            f"Safe values: depth ∈ {{1, 2, 3, 4, 5, 6}}."
        )

        self.cfg = cfg
        ch = cfg.base_channels

        self.in_conv = ConvBlock(1, ch, cfg.use_norm, cfg.use_residual)

        self.down = nn.ModuleList([
            ConvBlock(ch, ch, cfg.use_norm, cfg.use_residual)
            for _ in range(cfg.depth)
        ])

        # Time embedding: T=100, projected to ch dims, added at bottleneck
        self.time_emb = TimeEmbedding(T=100, ch=ch)

        self.up = nn.ModuleList([
            # ch*2 because skip is concatenated
            ConvBlock(ch * 2, ch, cfg.use_norm, cfg.use_residual)
            for _ in range(cfg.depth)
        ])

        self.out_conv = nn.Conv2d(ch, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, 64, 64)  noisy image at timestep t
        t : (B,)             integer timestep indices

        Returns
        -------
        pred_noise : (B, 1, 64, 64)
        """
        h = self.in_conv(x)

        skips = []
        for layer in self.down:
            h = layer(h)
            skips.append(h)
            h = F.avg_pool2d(h, 2)

        # Inject timestep: (B, ch) → (B, ch, 1, 1) broadcast over spatial
        t_emb = self.time_emb(t)[:, :, None, None]
        h = h + t_emb

        for layer in self.up:
            h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = layer(h)

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
        s = NoiseSchedule.__new__(NoiseSchedule)
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
) -> tuple[float, float]:
    """
    Single gradient update step.

    The schedule must already be on the same device as batch/model.
    Use schedule.to(device) once before the training loop.

    Returns
    -------
    loss      : float, MSE between predicted and actual noise
    grad_norm : float, L2 norm of all gradients (useful diagnostic)

    Teaching notes
    --------------
    - Watch grad_norm over time: stable training → steady norm
    - use_norm=False  → grad_norm may spike or collapse
    - use_residual=False + deep → grad_norm decays toward zero
    - Large grad_norm spikes often precede loss divergence
    """
    model.train()
    optimizer.zero_grad()

    B  = batch.size(0)
    t  = torch.randint(0, schedule.T, (B,), device=batch.device)

    xt, noise = q_sample(batch, t, schedule)

    pred = model(xt, t)
    loss = F.mse_loss(pred, noise)

    loss.backward()

    # Compute global gradient L2 norm across all parameters
    sq_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq_sum += p.grad.data.norm(2).item() ** 2
    grad_norm = math.sqrt(sq_sum)

    optimizer.step()

    return loss.item(), grad_norm


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
    x : (num_samples, 1, 64, 64) tensor, pixel values roughly in [-1, 1]

    Teaching note
    -------------
    This is the primary perceptual evaluation signal.
    A model can have a reasonable MSE loss but still produce
    incoherent samples — watch for this in the ablation experiments.
    """
    model.eval()
    sched = schedule.to(device)

    x = torch.randn(num_samples, 1, 64, 64, device=device)

    for t_val in reversed(range(sched.T)):
        t_batch = torch.full((num_samples,), t_val, device=device, dtype=torch.long)

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
# 8. SYNTHETIC DOODLE DATASET
# =========================================================

class ToyDoodles(torch.utils.data.Dataset):
    """
    Minimal synthetic dataset: white circles on black backgrounds.

    WHY THIS INSTEAD OF MNIST / CIFAR
    -----------------------------------
    - Zero external dependencies, works in any environment
    - Structured enough that coherent generation is visually obvious
    - Simple enough that failures are attributable to the model,
      not to dataset difficulty

    Each image is a single circle with random center and radius,
    normalized to [-1, 1] (black=-1, white=+1).
    """

    def __init__(self, size: int = 2000):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = torch.zeros(1, 64, 64)

        cx, cy = torch.randint(10, 54, (2,)).tolist()
        r      = torch.randint(3, 10, (1,)).item()

        y, x = torch.meshgrid(
            torch.arange(64), torch.arange(64), indexing="ij"
        )
        mask = (x - cx) ** 2 + (y - cy) ** 2 < r ** 2
        img[0][mask] = 1.0

        return img * 2.0 - 1.0   # [0,1] → [-1,1]


def get_loader(batch_size: int = 32) -> torch.utils.data.DataLoader:
    """Return an infinite-friendly DataLoader over ToyDoodles."""
    return torch.utils.data.DataLoader(
        ToyDoodles(),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
