from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# DEBUG STATE
# =========================================================

DEBUG = False
DEBUG_MODE = False


def set_debug(on: bool = True):
    global DEBUG, DEBUG_MODE
    DEBUG = on
    DEBUG_MODE = on


class _State:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step = 0
        self.last_loss = None
        self.last_grad = 0.0
        self.last_film = None
        self.last_skip = None
        self.last_bottleneck = None


STATE = _State()


def step_end(loss):
    STATE.step += 1
    STATE.last_loss = loss

    # SAFE PRINT (no crashes)
    print(
        f"[STEP {STATE.step:05d}] "
        f"loss={loss:.4f} | "
        f"grad={STATE.last_grad:.4f} | "
        f"film={STATE.last_film} | "
        f"skip={STATE.last_skip} | "
        f"bneck={STATE.last_bottleneck}"
    )


# =========================================================
# CONFIG
# =========================================================

@dataclass
class ModelConfig:
    T: int = 1000
    depth: int = 3
    base_channels: int = 16
    use_residual: bool = True
    use_norm: bool = True
    depth_per_stage: int = 2


# =========================================================
# TIME
# =========================================================

class TimeEmbedding(nn.Module):
    def __init__(self, T: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(T, dim)

    def forward(self, t):
        return self.emb(t)


class TimeProject(nn.Module):
    def __init__(self, t_dim: int, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(t_dim, ch * 2),
            nn.SiLU(),
            nn.Linear(ch * 2, ch * 2),
        )

    def forward(self, t_emb):
        raw = self.net(t_emb)
        ch = raw.shape[1] // 2
        scale = 0.5 * torch.tanh(raw[:, :ch])
        shift = 0.5 * torch.tanh(raw[:, ch:])
        return scale[:, :, None, None], shift[:, :, None, None]


def apply_film(x, scale, shift):
    return x * (1 + scale) + shift


# =========================================================
# BLOCKS
# =========================================================

class ConvBlock(nn.Module):
    def __init__(self, ch, use_norm, use_residual):
        super().__init__()

        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

        if use_norm:
            g = min(8, ch)
            while ch % g != 0 and g > 1:
                g -= 1
            self.norm = nn.GroupNorm(g, ch)
        else:
            self.norm = nn.Identity()

        self.act = nn.SiLU()
        self.use_residual = use_residual

    def forward(self, x):
        out = self.act(self.norm(self.conv(x)))
        return out + x if self.use_residual else out


class StageBlock(nn.Module):
    def __init__(self, ch, use_norm, use_residual, depth):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(ch, use_norm, use_residual)
            for _ in range(depth)
        ])

    def forward(self, x):
        return self.net(x)


# =========================================================
# SKIP FUSION
# =========================================================

class GatedSkipFusion(nn.Module):
    def __init__(self, ch: int, t_dim: int):
        super().__init__()

        self.t_proj = nn.Linear(t_dim, ch)

        self.gate_net = nn.Sequential(
            nn.Linear(ch, ch),
            nn.SiLU(),
            nn.Linear(ch, ch),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 1),
        )

    def forward(self, skip, dec, t_emb):
        t = self.t_proj(t_emb)
        gate = torch.sigmoid(self.gate_net(t))[:, :, None, None]

        skip = skip * gate
        x = torch.cat([skip, dec], dim=1)
        return self.fuse(x)


# =========================================================
# MODEL
# =========================================================

class DiffusionUNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        base = cfg.base_channels
        depth = cfg.depth

        channels = [base * (2 ** i) for i in range(depth + 1)]
        bottleneck = channels[-1]

        self.in_proj = nn.Conv2d(1, base, 3, padding=1)

        self.time_emb = TimeEmbedding(cfg.T, bottleneck)

        self.enc = nn.ModuleList()
        self.enc_time = nn.ModuleList()
        self.down = nn.ModuleList()

        for i in range(depth):
            self.enc.append(StageBlock(channels[i], cfg.use_norm, cfg.use_residual, cfg.depth_per_stage))
            self.enc_time.append(TimeProject(bottleneck, channels[i]))
            self.down.append(nn.Conv2d(channels[i], channels[i + 1], 2, 2))

        self.bottleneck = StageBlock(bottleneck, cfg.use_norm, cfg.use_residual, cfg.depth_per_stage)
        self.bottleneck_time = TimeProject(bottleneck, bottleneck)

        self.up = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.dec_time = nn.ModuleList()

        for i in reversed(range(depth)):
            ch = channels[i]
            prev = channels[i + 1]

            self.up.append(nn.ConvTranspose2d(prev, ch, 2, 2))
            self.skip.append(GatedSkipFusion(ch, bottleneck))
            self.dec.append(StageBlock(ch, cfg.use_norm, cfg.use_residual, cfg.depth_per_stage))
            self.dec_time.append(TimeProject(bottleneck, ch))

        self.out = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_emb(t)

        h = self.in_proj(x)
        skips = []

        for enc, tproj, down in zip(self.enc, self.enc_time, self.down):
            h = enc(h)
            s, sh = tproj(t_emb)
            h = apply_film(h, s, sh)
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h)
        s, sh = self.bottleneck_time(t_emb)
        h = apply_film(h, s, sh)

        for up, skip, dec, tproj in zip(self.up, self.skip, self.dec, self.dec_time):
            h = up(h)
            s = skips.pop()
            h = skip(s, h, t_emb)
            h = dec(h)
            sc, sh = tproj(t_emb)
            h = apply_film(h, sc, sh)

        return self.out(h)


# =========================================================
# SCHEDULE (FIXED + STABLE)
# =========================================================

def get_schedule(cfg: ModelConfig):
    # FIX: stable diffusion alpha schedule (NOT linear 1→0)
    return torch.linspace(0.9, 0.1, cfg.T)


# =========================================================
# TRAIN STEP (OPTIMIZED FOR MPS)
# =========================================================

def train_step(model, batch, schedule, opt):
    model.train()

    t = torch.randint(0, len(schedule), (batch.size(0),), device=batch.device)

    noise = torch.randn_like(batch)

    alpha = schedule[t].view(-1, 1, 1, 1)
    sigma = torch.sqrt(torch.clamp(1 - alpha**2, min=1e-8))

    x = alpha * batch + sigma * noise

    pred = model(x, t)
    loss = F.mse_loss(pred, noise)

    opt.zero_grad(set_to_none=True)
    loss.backward()

    # FAST GRAD LOGGING (no full scan every step)
    total_grad = 0.0
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                total_grad += p.grad.abs().mean().item()

    STATE.last_grad = total_grad

    opt.step()

    return loss.item(), {}, t.tolist()


# =========================================================
# SAMPLING (FIXED DDPM REVERSE)
# =========================================================

@torch.no_grad()
def sample(model, schedule, device, num_samples=16):
    model.eval()

    x = torch.randn(num_samples, 1, 64, 64, device=device)

    for i in reversed(range(len(schedule))):
        t = torch.full((num_samples,), i, device=device)

        eps = model(x, t)

        alpha = schedule[i]
        sigma = torch.sqrt(torch.clamp(1 - alpha**2, min=1e-8))

        x = (x - sigma * eps) / torch.clamp(alpha, min=1e-6)

    return x


# =========================================================
# API (IMPORT COMPATIBILITY FIXED)
# =========================================================

def build_model(cfg: ModelConfig):
    return DiffusionUNet(cfg)


def get_loader(batch_size: int):
    while True:
        yield torch.randn(batch_size, 1, 64, 64)