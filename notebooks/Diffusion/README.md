# Diffusion Doodle Lab

A minimal diffusion model designed as a **deep learning laboratory**, not a production system.

The goal is to study how neural networks behave under a challenging optimization problem where:
- gradients are noisy
- supervision is indirect
- learning is multi-step and unstable
- architecture choices strongly affect convergence

---

## Why Diffusion?

Diffusion models define a learning problem where the network must predict noise added at different time steps.

This creates a difficult optimization landscape:

- The target signal depends on noise level (time step)
- Small gradient errors accumulate across denoising steps
- Training can appear stable while generation quality fails
- Architecture and normalization strongly affect stability

This makes diffusion a useful case study for:
> optimization, gradient flow, and representation learning in deep networks

---

## Dataset

The dataset consists of **synthetic doodles** (simple geometric shapes).

This is intentional:

- removes data complexity as a confounder
- keeps the focus on optimization and architecture
- ensures failure modes come from the model, not the data

The task is NOT visual realism, but:
> learning whether the model can recover structure from noisy supervision

---

## Learning Goals

This lab investigates:

- Optimization stability in deep networks
- Gradient explosion / vanishing behavior
- Effect of residual connections
- Role of normalization layers
- Impact of network depth
- Why loss ≠ sample quality in generative models

---

## Files

### `diffusion_doodle_lib.py`

Core implementation:
- U-Net model
- diffusion process (forward + sampling)
- training step
- synthetic doodle dataset

Do not modify this file.

---

### `lab_diffusion_doodles.py`

Experiment script:
- defines controlled ablations
- runs training loops
- generates samples
- guides analysis and interpretation

This is the only file you should modify.

---

## Experiments

| Experiment | Change |
|------------|--------|
| EXP0 | Baseline |
| EXP1 | Remove normalization |
| EXP2 | Remove residuals |
| EXP3 | Increase depth |
| EXP4 | No norm + no residuals + deep |

Compare across:
- loss curves
- gradient norms
- generated samples

---

## Run

```bash
pip install torch torchvision
python lab_diffusion_doodles.py