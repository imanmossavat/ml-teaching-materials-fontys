# Diffusion Doodle Lab

A minimal diffusion model designed to study **deep learning behavior**, not to achieve state-of-the-art results.

## Learning Goals

This lab investigates:

- Optimization stability
- Gradient flow
- Residual connections
- Normalization
- Network depth
- Diffusion training dynamics
- Why loss and sample quality are not the same thing

## Files

### `diffusion_doodle_lib.py`

Core implementation:

- U-Net
- Diffusion process
- Training step
- Sampling
- Synthetic doodle dataset

Students should not modify this file.

### `lab_diffusion_doodles.py`

Experiment script:

- Defines ablations
- Trains models
- Generates samples
- Guides analysis

This is the file students work in.

## Experiments

| Experiment | Change |
|------------|---------|
| EXP0 | Baseline |
| EXP1 | Remove normalization |
| EXP2 | Remove residuals |
| EXP3 | Increase depth |
| EXP4 | No norm + no residuals + deep |

Run all experiments and compare:

- Loss
- Gradient norms
- Generated samples

## Run

```bash
pip install torch torchvision
python lab_diffusion_doodles.py

