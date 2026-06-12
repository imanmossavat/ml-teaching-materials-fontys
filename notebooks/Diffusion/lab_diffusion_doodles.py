"""
=========================================================
DIFFUSION DOODLE LAB — EXPERIMENT SCRIPT
=========================================================

PURPOSE
-------
This is where all scientific reasoning happens.

You will run a structured ablation study: starting from a
working baseline, you disable ONE variable at a time and
observe what breaks, and why.

RULE: Do not modify diffusion_doodle_lib.py.
      Only change ModelConfig fields and experiment parameters here.

WHAT TO OBSERVE
---------------
For each experiment, watch:

  loss      — does it decrease steadily, plateau, or diverge?
  grad_norm — is it stable, decaying, or exploding?
  samples   — do the generated images look like circles?

The gap between loss behavior and sample quality is itself
an important lesson.

EXPERIMENT STRUCTURE
--------------------
  EXP 0 — Baseline (everything on, depth=3)
  EXP 1 — Remove normalization only
  EXP 2 — Remove residuals only
  EXP 3 — Increase depth only (to 5 — approaches instability)
  EXP 4 — Worst case: no norm, no residuals, deep

Each experiment changes exactly one variable from baseline.
Experiment 4 combines all bad choices to show the collapse.
=========================================================
"""

import itertools
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

from diffusion_doodle_lib import (
    ModelConfig,
    NoiseSchedule,
    build_model,
    train_step,
    sample,
    get_loader,
)


# =========================================================
# REPRODUCIBILITY
# =========================================================

torch.manual_seed(42)

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
schedule = NoiseSchedule(T=100)

# Move schedule tensors to device once, reuse everywhere.
# This avoids the silent CPU/GPU tensor mismatch bug.
schedule_dev = schedule.to(device)

print(f"Running on: {device}")


# =========================================================
# EXPERIMENT DEFINITIONS
# =========================================================
# Each entry: (name, ModelConfig)
# ONE field differs from baseline per experiment.
# =========================================================

EXPERIMENTS = [
    (
        "EXP0_baseline",
        ModelConfig(depth=3, base_channels=32, use_residual=True,  use_norm=True),
    ),
    (
        "EXP1_no_norm",
        ModelConfig(depth=3, base_channels=32, use_residual=True,  use_norm=False),
        # What to watch: activation magnitudes grow unchecked across layers.
        # Expect: grad_norm spikes, loss may plateau or oscillate.
    ),
    (
        "EXP2_no_residual",
        ModelConfig(depth=3, base_channels=32, use_residual=False, use_norm=True),
        # What to watch: gradients must flow through every conv sequentially.
        # At depth=3 this is marginal; the effect is clearer at depth=5.
        # Expect: slower convergence, possibly higher final loss.
    ),
    (
        "EXP3_deep",
        ModelConfig(depth=5, base_channels=32, use_residual=True,  use_norm=True),
        # What to watch: bottleneck is now 64/2^5 = 2px.
        # Spatial information is severely compressed.
        # Expect: loss may still decrease, but samples degrade.
        # This demonstrates loss/perceptual-quality mismatch.
    ),
    (
        "EXP4_worst_case",
        ModelConfig(depth=5, base_channels=32, use_residual=False, use_norm=False),
        # All bad choices combined.
        # Expect: unstable training, incoherent or collapsed samples.
        # Useful contrast against EXP0 to motivate each design choice.
    ),
]


# =========================================================
# TRAINING LOOP
# =========================================================

STEPS       = 300    # gradient steps per experiment
LOG_EVERY   = 25     # print interval
BATCH_SIZE  = 32
LR          = 1e-4


def run_experiment(name: str, cfg: ModelConfig) -> nn.Module:
    """
    Train a model for STEPS gradient steps.

    Uses itertools.cycle so the DataLoader never exhausts
    before we reach STEPS — each step is a genuine gradient update.

    Prints loss and grad_norm every LOG_EVERY steps so you can
    watch training dynamics unfold.
    """
    model = build_model(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"  depth={cfg.depth}  base_ch={cfg.base_channels}  "
          f"residual={cfg.use_residual}  norm={cfg.use_norm}")
    print(f"  Parameters: {param_count:,}")
    print(f"{'='*55}")

    loader       = get_loader(BATCH_SIZE)
    data_stream  = itertools.cycle(loader)   # never exhausts

    for step in range(STEPS):
        batch = next(data_stream).to(device)
        loss, grad_norm = train_step(model, batch, schedule_dev, opt)

        if step % LOG_EVERY == 0 or step == STEPS - 1:
            print(f"  step={step:>3d}  loss={loss:.4f}  grad_norm={grad_norm:.3f}")

    return model


# =========================================================
# RUN ALL EXPERIMENTS
# =========================================================

trained_models = {}

for entry in EXPERIMENTS:
    name, cfg = entry[0], entry[1]
    model = run_experiment(name, cfg)
    trained_models[name] = model


# =========================================================
# EVALUATION: GENERATE SAMPLES
# =========================================================
# For each experiment, generate 16 images and save a grid.
# Normalize from model output range (roughly [-1,1]) to [0,1]
# for display.  Without this, images may appear all-white or
# all-black depending on the model's output scale.
# =========================================================

print("\n" + "="*55)
print("  Generating samples...")
print("="*55)

for name, model in trained_models.items():
    samples = sample(model, schedule, device, num_samples=16)
    grid    = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
    fname   = f"{name}_samples.png"
    save_image(grid, fname)
    print(f"  Saved: {fname}")


# =========================================================
# GUIDED OBSERVATIONS
# =========================================================

print("""
=========================================================
 WHAT TO LOOK FOR
=========================================================

EXP0 (baseline) vs EXP1 (no norm)
  - Are grad_norms stable across 300 steps in EXP0?
  - Do they spike or drift in EXP1?
  - Does removing norm change when the loss starts decreasing?

EXP0 (baseline) vs EXP2 (no residual)
  - Does convergence slow down without residuals?
  - At depth=3 the effect is subtle — note it carefully.
  - Question: why might residuals matter more at greater depth?

EXP0 (baseline) vs EXP3 (deep, norm+residual intact)
  - Does loss still decrease in EXP3? (It probably does.)
  - Do the samples still look like circles?
  - THIS is the loss/perceptual-quality mismatch in action.
  - A model can fit the noise prediction task while losing
    the ability to generate coherent structure.

EXP4 (worst case) vs EXP0 (baseline)
  - How different are the sample grids?
  - Can you now attribute each part of the degradation to
    the right variable using EXP1, EXP2, EXP3 as evidence?

KEY QUESTIONS
  1. In which experiment did grad_norm behave most erratically?
  2. Which single change hurt sample quality the most?
  3. Did any experiment show decreasing loss but bad samples?
     What does that tell you about using loss as a metric?
  4. Why does adding norm stabilize gradients?
     (Hint: what does norm do to activation scale layer-to-layer?)
  5. Why does a 2px bottleneck hurt generation even with
     otherwise stable training?
=========================================================
""")
