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
For each experiment, watch three things in this order:

  1. loss curve shape  — decreasing steadily, plateauing, or
                         diverging? The shape matters more than
                         the final value.

  2. grad_norms        — look at encoder and decoder separately,
                         not just global. A healthy network has
                         similar norms across groups. Dying encoder
                         norms with healthy decoder norms = gradients
                         are not reaching the early layers.

  3. samples           — this is the ground truth. A model with
                         acceptable loss but incoherent samples has
                         learned to predict noise but lost the ability
                         to generate structure. Loss and perceptual
                         quality can diverge. When they do, trust
                         the samples.

EXPERIMENT STRUCTURE
--------------------
  EXP0 — Baseline: everything on, depth=3, depth_per_stage=2
  EXP1 — Remove normalization only
  EXP2 — Remove residuals only
  EXP3a — Increase depth to 4 (4×4 bottleneck)
  EXP3b — Increase depth to 5 (2×2 bottleneck)
  EXP4 — Worst case: no norm, no residuals, deep

Each experiment (except EXP4) changes exactly ONE variable
from baseline. EXP3 is split into two sub-experiments to
separate the depth effect from the bottleneck compression
effect — these are different phenomena that happen to be
coupled when you increase depth.

EXP4 changes three variables simultaneously. It is useful
as a failure illustration but CANNOT be used to attribute
the failure to any individual component. Use EXP1, EXP2,
EXP3 as the causal evidence; EXP4 as the dramatic contrast.

ARCHITECTURE NOTE
-----------------
base_channels= base_channels with depth=3 gives this channel progression:

  64×64  →   8 ch   (stage 0)
  32×32  →  16 ch   (stage 1)
  16×16  →  32 ch   (stage 2)
   8×8   →  64 ch   (bottleneck)

Total bottleneck values: 8×8×64 = 4096, matching the input
volume 64×64×1 = 4096. The bottleneck is the richest point
in the network, not the most compressed — which is the correct
design philosophy.
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

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("=="*10)
print(f"Running on: {device}")
print("=="*10)

schedule    = NoiseSchedule(T=100)
schedule_dev = schedule.to(device)

base_channels= 4  # base_channels=4 is a good starting point, it gives a reasonable channel progression and bottleneck size without being too expensive.

# =========================================================
# EXPERIMENT DEFINITIONS
# =========================================================
# Each entry: (name, ModelConfig, prediction_prompt)
#
# Read the prediction prompt BEFORE running each experiment.
# Write down your prediction first. The gap between what you
# expected and what happened is where the learning is.
# =========================================================

EXPERIMENTS = [
    (
        "EXP0_baseline",
        ModelConfig(depth=3, base_channels= base_channels, use_residual=True, use_norm=True,
                    depth_per_stage=2),
        # PREDICT BEFORE RUNNING:
        # - What do you expect the loss curve to look like?
        # - Do you expect grad norms to be higher in the encoder
        #   or decoder, or similar?
        # - What should the generated samples look like?
        # This is your reference. All other experiments are
        # compared against this one.
    ),
    (
        "EXP1_no_norm",
        ModelConfig(depth=3, base_channels= base_channels, use_residual=True, use_norm=False,
                    depth_per_stage=2),
        # ONE CHANGE FROM BASELINE: use_norm=False
        #
        # PREDICT BEFORE RUNNING:
        # Without GroupNorm, activation magnitudes are not
        # rescaled between layers. What do you expect happens
        # to activations as they pass through multiple convs?
        # How should this affect grad norms — encoder vs decoder?
        # Will the loss still decrease, or will it plateau?
        #
        # WHAT TO WATCH:
        # - Per-stage grad norms: do encoder and decoder diverge?
        # - Loss curve shape: does it plateau above the baseline?
        # - Samples: do they look noisier or more blurry?
    ),
    (
        "EXP2_no_residual",
        ModelConfig(depth=3, base_channels= base_channels, use_residual=False, use_norm=True,
                    depth_per_stage=2),
        # ONE CHANGE FROM BASELINE: use_residual=False
        #
        # PREDICT BEFORE RUNNING:
        # With depth_per_stage=2 each stage has 2 conv blocks.
        # Removing residuals means gradients must traverse both
        # convs sequentially with no shortcut. The U-Net skip
        # connections help across stages but NOT within a stage.
        # What do you expect happens to encoder grad norms?
        #
        # WHAT TO WATCH:
        # - Encoder grad norms: do they decay over training?
        # - Compare to EXP1: different failure mode, same loss?
        # - Do samples degrade even if loss looks similar?
        #
        # NOTE ON INTERPRETATION:
        # If you had run this with depth_per_stage=1, removing
        # residuals would have had almost no effect — because with
        # only one conv per stage, the U-Net skips already provide
        # all the gradient flow needed within each stage. The
        # depth_per_stage=2 setting is what makes this experiment
        # meaningful. This is a real architectural subtlety:
        # residuals matter most when they are the only gradient
        # highway available.
    ),
    (
        "EXP3a_deeper_depth4",
        ModelConfig(depth=4, base_channels= base_channels, use_residual=True, use_norm=True,
                    depth_per_stage=2),
        # ONE CHANGE FROM BASELINE: depth=4 (bottleneck: 4×4, 128ch)
        #
        # PREDICT BEFORE RUNNING:
        # Increasing depth to 4 does two things simultaneously:
        #   (a) the network has more stages to process features
        #   (b) the bottleneck shrinks from 8×8 to 4×4
        # Can you separate these effects from the loss curve alone?
        #
        # WHAT TO WATCH:
        # - Does loss still decrease at the same rate as baseline?
        # - Do samples look as sharp as baseline, or blurrier?
        # - Compare to EXP3b: is the degradation worse at depth=5?
        #   If yes, the bottleneck size is the likely cause.
    ),
    (
        "EXP3b_deeper_depth5",
        ModelConfig(depth=5, base_channels= base_channels, use_residual=True, use_norm=True,
                    depth_per_stage=2),
        # ONE CHANGE FROM BASELINE: depth=5 (bottleneck: 2×2, 256ch)
        #
        # PREDICT BEFORE RUNNING:
        # The bottleneck is now 2×2 — four spatial positions to
        # represent the entire image structure. Even with 256
        # channels, can four positions encode where a circle is
        # and how big it is well enough to reconstruct it?
        #
        # WHAT TO WATCH:
        # - THIS is the loss/perceptual quality mismatch experiment.
        # - Loss may still decrease (noise prediction is learnable).
        # - Samples may degrade (structure is lost at 2×2).
        # - Compare EXP3a vs EXP3b: the difference between them
        #   isolates the bottleneck compression effect from depth.
        #
        # KEY QUESTION: If loss looks similar to baseline but
        # samples look worse, what does that tell you about using
        # loss as your primary evaluation metric?
    ),
    (
        "EXP4_worst_case",
        ModelConfig(depth=5, base_channels= base_channels, use_residual=False, use_norm=False,
                    depth_per_stage=2),
        # THREE CHANGES FROM BASELINE: depth=5, no norm, no residual
        #
        # !! THIS IS NOT A CONTROLLED EXPERIMENT !!
        # Three variables change simultaneously. You cannot
        # attribute any observed failure to a single cause.
        # Use EXP1, EXP2, EXP3 as your causal evidence.
        # Use EXP4 only as a dramatic contrast against EXP0.
        #
        # PREDICT BEFORE RUNNING:
        # Based on EXP1, EXP2, EXP3 — what do you expect the
        # combined failure to look like? Which effect do you
        # expect to dominate?
        #
        # WHAT TO WATCH:
        # - Encoder grad norms in the first 100 steps: are they
        #   near zero? This is the frozen early-layer pattern.
        #   A sudden jump after many near-zero steps means the
        #   network was barely learning until the later layers
        #   accidentally built enough signal to push gradients
        #   backward. A global norm scalar hides this — the
        #   per-stage breakdown is what makes it visible.
        # - Loss curve: does it stay flat for many steps before
        #   suddenly dropping? This is the same pattern.
    ),
]


# =========================================================
# TRAINING LOOP
# =========================================================

STEPS      = 3000    # gradient steps per experiment
LOG_EVERY  = 100      # print interval
BATCH_SIZE = 32
LR         = 3e-4


def format_norms(grad_norms: dict) -> str:
    """Format per-stage grad norms for compact printing."""
    keys = ['encoder', 'bottleneck', 'decoder', 'global']
    parts = [
        f"{k}={grad_norms[k]:.3f}"
        for k in keys if k in grad_norms
    ]
    return "  ".join(parts)


def run_experiment(name: str, cfg: ModelConfig) -> nn.Module:
    """
    Train a model for STEPS gradient steps.

    Uses itertools.cycle so the DataLoader never exhausts
    before STEPS — each step is a genuine gradient update.

    Prints loss and per-stage grad norms every LOG_EVERY steps.
    Watch encoder and decoder norms separately — they tell
    different parts of the gradient flow story.
    """
    model = build_model(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)

    ch_bottleneck = cfg.base_channels * (2 ** cfg.depth)
    param_count   = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"  depth={cfg.depth}  base_ch={cfg.base_channels}  "
          f"dps={cfg.depth_per_stage}  "
          f"residual={cfg.use_residual}  norm={cfg.use_norm}")
    print(f"  Bottleneck: {64 // 2**cfg.depth}×{64 // 2**cfg.depth}×{ch_bottleneck}"
          f"  ({(64 // 2**cfg.depth)**2 * ch_bottleneck:,} values)")
    print(f"  Parameters: {param_count:,}")
    print(f"{'='*65}")
    print(f"  {'step':>5}  {'loss':>8}  grad norms by group")
    print(f"  {'-'*60}")

    loader      = get_loader(BATCH_SIZE)
    data_stream = itertools.cycle(loader)

    for step in range(STEPS):
        batch              = next(data_stream).to(device)
        loss, grad_norms   = train_step(model, batch, schedule_dev, opt)

        if step % LOG_EVERY == 0 or step == STEPS - 1:
            norm_str = format_norms(grad_norms)
            print(f"  step={step:>4d}  loss={loss:.4f}  {norm_str}",
                  flush=True)

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

print("\n" + "="*65)
print("  Generating samples...")
print("="*65)

for name, model in trained_models.items():
    imgs  = sample(model, schedule, device, num_samples=16)
    grid  = make_grid(imgs, nrow=4, normalize=True, value_range=(-1, 1))
    fname = f"{name}_samples.png"
    save_image(grid, fname)
    print(f"  Saved: {fname}", flush=True)


# =========================================================
# GUIDED OBSERVATIONS
# =========================================================

print("""
=========================================================
 WHAT TO LOOK FOR — POST-RUN QUESTIONS
=========================================================

For each question, compare your prediction to the result.
If they differ, that gap is the most important thing to
explain.

EXP0 (baseline)
  - Did loss converge cleanly? Is it still noisy at the end?
  - Are encoder and decoder grad norms similar, or is one
    group systematically higher?
  - Do the samples look like circles? This is your reference.

EXP0 vs EXP1 (no norm)
  - Did removing norm cause grad norms to spike, drift, or
    stay similar?
  - Did the loss plateau above baseline? At what value?
  - Were encoder and decoder norms affected equally?
    (If not: which layers lost stability first?)

EXP0 vs EXP2 (no residual)
  - Did encoder grad norms decay over training?
  - Did the loss converge more slowly, or to a worse value?
  - Is the failure mode the same as EXP1 or different?
    (EXP1 fails through activation instability.
     EXP2 fails through gradient starvation. These look
     different in the per-stage norm logs.)

EXP3a vs EXP3b (depth=4 vs depth=5)
  - Did loss decrease in both? How does it compare to baseline?
  - Do samples degrade more in EXP3b than EXP3a?
  - If loss looks similar but samples look worse in EXP3b,
    what is the bottleneck size doing that loss cannot show?
  - This comparison isolates bottleneck compression from depth.
    EXP3a has more depth than baseline; EXP3b has more depth
    AND a much smaller bottleneck. The difference between them
    tells you what the 2×2 bottleneck costs.

EXP4 (worst case)
  - Look at encoder grad norms in steps 0–150. Are they
    near zero? When do they jump?
  - Can you explain the failure using EXP1+EXP2+EXP3b as
    separate evidence for each component?
  - Do NOT use EXP4 alone to conclude anything about norm,
    residuals, or depth individually.

KEY QUESTIONS
  1. Which experiment showed the clearest difference between
     encoder and decoder grad norms?
  2. Did any experiment show acceptable loss but bad samples?
     What does that tell you about MSE as an evaluation metric
     for generative models?
  3. In EXP4, if encoder grad norms are near zero for the
     first 100+ steps, what is the network actually doing
     during that time? Is it learning at all?
  4. Why does GroupNorm stabilize gradient norms?
     (Hint: what does it do to activation scale at each layer?)
  5. Why does a 2×2 bottleneck hurt generation even when
     training loss still decreases?
  6. In EXP2, would removing residuals have mattered if
     depth_per_stage were 1 instead of 2? Why or why not?
=========================================================
""")
