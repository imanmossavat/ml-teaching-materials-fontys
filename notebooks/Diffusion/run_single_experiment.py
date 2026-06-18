"""
SINGLE EXPERIMENT RUNNER — Diffusion Doodle Lab

Purpose:
- Run ONE experiment
- Train quickly
- Save sample grid at the end
- Designed for M4 / CPU / MPS / small GPU

T CONSISTENCY
-------------
ModelConfig.T is the single source of truth.  Use get_schedule(cfg)
to build the NoiseSchedule — it reads T from the config so the
model embedding table and the schedule can never diverge.
"""

import torch
from torchvision.utils import make_grid, save_image

from diffusion_doodle_lib import (
    ModelConfig,
    build_model,
    get_schedule,          # ← replaces NoiseSchedule(T=...) directly
    train_step,
    sample,
    get_loader,
)

# ----------------------------
# DEVICE (MPS-friendly)
# ----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# ----------------------------
# EXPERIMENT CONFIG
# T lives here — one place, no divergence possible.
# ----------------------------
EXPERIMENT_CFG = ModelConfig(
    T=1000,              # ← diffusion timesteps; shared by model + schedule
    depth=3,
    base_channels=16,
    use_residual=True,
    use_norm=True,
    depth_per_stage=3,
)

# ----------------------------
# TRAIN SETTINGS (FAST MODE)
# ----------------------------
STEPS      = 20000
BATCH_SIZE = 8
LR         = 3e-4
LOG_EVERY  = 25

# get_schedule reads EXPERIMENT_CFG.T — guaranteed consistent.
schedule = get_schedule(EXPERIMENT_CFG).to(device)

# ----------------------------
# MODEL + OPT
# ----------------------------
model = build_model(EXPERIMENT_CFG).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=LR)

loader    = get_loader(BATCH_SIZE)
data_iter = iter(loader)

print("\nModel ready. Starting training...\n")

# Column guide printed once so the per-step rows are self-explanatory.
#
#   step    training iteration index
#   loss    MSE between predicted noise and actual noise (lower = better)
#
#   Gradient norm columns — L2 norm of parameter gradients per model region.
#   These diagnose where learning is (or isn't) happening each step:
#
#   enc     encoder (input_proj + all down stages)
#   dec     decoder (all up stages + skip fusion modules)
#   bneck   bottleneck activation gradient (hook-captured, not a param norm);
#           near-zero signals a vanishing-gradient problem upstream of here
#   global  combined L2 norm across the entire model
#
# Timestep-bucket rows (every 200 steps) break the running average loss by
# noise level.  t≈0 is nearly clean signal (easy for the model); t≈900 is
# near-pure Gaussian noise (hard).  A healthy run sees loss fall in all
# buckets.  A bucket that stays high reveals where the model still struggles.

BUCKET_WIDTH = EXPERIMENT_CFG.T // 10

HEADER = (
    f"\n{'step':>6}  {'loss':>8}  "
    f"{'enc':>8}  {'dec':>8}  {'bneck':>8}  {'global':>8}"
    f"     ← loss | grad norms →"
)
DIVIDER = "-" * 70
print(HEADER)
print(DIVIDER)

bucket_loss = {i: [] for i in range(10)}

# ----------------------------
# TRAIN LOOP
# ----------------------------
for step in range(STEPS):
    try:
        batch = next(data_iter).to(device)
    except StopIteration:
        data_iter = iter(loader)
        batch     = next(data_iter).to(device)

    loss, grads, buckets = train_step(model, batch, schedule, opt)
    for b in buckets:
        bucket_loss[int(b)].append(loss)

    if step % LOG_EVERY == 0 or step == STEPS - 1:
        print(
            f"{step:>6d}  {loss:>8.4f}  "
            f"{grads.get('encoder',    0):>8.3f}  "
            f"{grads.get('decoder',    0):>8.3f}  "
            f"{grads.get('bottleneck', 0):>8.4f}  "
            f"{grads.get('global',     0):>8.3f}"
        )

    if step % 200 == 0:
        # Per-bucket average loss since the last reset.
        # Buckets 0, 4, 9 sample low / mid / high noise levels.
        rows = [
            f"    t≈{k * BUCKET_WIDTH:>4d}  avg_loss = {sum(bucket_loss[k]) / len(bucket_loss[k]):.4f}"
            for k in [0, 4, 9]
            if bucket_loss[k]
        ]
        if rows:
            print(f"  [loss by noise level — step {step}]")
            print("\n".join(rows))
            print()

# ----------------------------
# SAMPLING
# ----------------------------
print("\nGenerating samples...")

imgs = sample(model, schedule, device, num_samples=16)
grid = make_grid(imgs, nrow=4, normalize=True, value_range=(-1, 1))

out_file = "single_experiment_samples.png"
save_image(grid, out_file)

print("Saved:", out_file)
print("Done.")
