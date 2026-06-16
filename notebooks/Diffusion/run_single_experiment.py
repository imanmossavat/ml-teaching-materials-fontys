
"""
SINGLE EXPERIMENT RUNNER — Diffusion Doodle Lab

Purpose:
- Run ONE experiment
- Train quickly
- Save sample grid at the end
- Designed for M4 / CPU / MPS / small GPU
"""

import torch
from torchvision.utils import make_grid, save_image

from diffusion_doodle_lib import (
    ModelConfig,
    NoiseSchedule,
    build_model,
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
# ----------------------------
EXPERIMENT_CFG = ModelConfig(
    depth=3,
    base_channels=8,
    use_residual=True,
    use_norm=True,
    depth_per_stage=2,
)

# ----------------------------
# TRAIN SETTINGS (FAST MODE)
# ----------------------------
STEPS = 3000
BATCH_SIZE = 8
LR = 3e-4
LOG_EVERY = 25

schedule = NoiseSchedule(T=100).to(device)

# ----------------------------
# MODEL + OPT
# ----------------------------
model = build_model(EXPERIMENT_CFG).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

loader = get_loader(BATCH_SIZE)
data_iter = iter(loader)

print("\nModel ready. Starting training...\n")

# ----------------------------
# TRAIN LOOP
# ----------------------------
for step in range(STEPS):
    try:
        batch = next(data_iter).to(device)
    except StopIteration:
        data_iter = iter(loader)
        batch = next(data_iter).to(device)

    loss, grads = train_step(model, batch, schedule, opt)

    if step % LOG_EVERY == 0 or step == STEPS - 1:
        print(
            f"step {step:4d} | loss {loss:.4f} | "
            f"enc {grads.get('encoder',0):.3f} | "
            f"dec {grads.get('decoder',0):.3f} | "
            f"glob {grads.get('global',0):.3f} | "
            f"bottleneck {grads.get('bottleneck',0):.3f}"
        )

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
