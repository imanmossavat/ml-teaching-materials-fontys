# Orientation – Technical Workshops

Hands-on notebooks for the orientation phase, usable in class or for self-study. They are grouped in two tracks, not tied to week numbers.

## How to read the names

`A1_vision_yolo` = **track letter** + **step** + **topic**.

* **A** = Applied AI: use existing models and test what they can and cannot do.
* **M** = ML Modeling: understand how models learn.
* The number is the suggested order within a track.
* `_optional` marks material that can be skipped.

## Overview

### Track A: Applied AI ([1_applied_ai](1_applied_ai/))

| Notebook | Topic | What you do | Builds on | Level |
|---|---|---|---|---|
| [A1_vision_yolo](1_applied_ai/A1_vision_yolo.ipynb) | Computer vision | Detect objects with a pretrained YOLO model and try to break it. High-level introduction. | – | Core |
| [A2_language_hugging_face](1_applied_ai/A2_language_hugging_face.ipynb) | Natural language processing | Use Hugging Face pipelines, build a small tool and test it. | – | Core |
| [A3_agents_atari_rl_optional](1_applied_ai/A3_agents_atari_rl_optional.ipynb) | Reinforcement learning agents | Watch a pretrained agent play Atari Breakout. For those who like games, AI and a challenge. | – | Optional |

### Track M: ML Modeling ([2_ml_modeling](2_ml_modeling/))

| Notebook | Topic | What you do | Builds on | Level |
|---|---|---|---|---|
| [M1_ml_foundations_decision_trees](2_ml_modeling/M1_ml_foundations_decision_trees.ipynb) | ML foundations | Train and test a decision tree; study model complexity, overfitting and underfitting. Covers most of what you need to understand ML. | – | Core, start here |
| [M2_classical_ml_models](2_ml_modeling/M2_classical_ml_models.ipynb) | Other models | Compare logistic regression, SVMs, k-nearest neighbours and AdaBoost. | M1 | Core |
| [M3_neural_networks_playground_pytorch](2_ml_modeling/M3_neural_networks_playground_pytorch.ipynb) | Neural networks | Explore the interactive playground, then build the same thing in PyTorch. | M1 | Core, can be two sessions |
| [M4_inside_a_neuron_optional](2_ml_modeling/M4_inside_a_neuron_optional.ipynb) | The math inside a neuron | Vectors, linear maps, activations and decision boundaries. | M3 | Optional |

M3 can be split into two sessions:

1. **Interactive:** experiment with [TensorFlow Playground](https://playground.tensorflow.org/) (the link with the starting settings is at the top of the notebook).
2. **Code:** work through the PyTorch notebook.

## Teacher's note

Start with **Track A (Applied AI)**, then move on to **Track M (ML Modeling)**: students first see what AI can do, then learn how it works.

## Roadmap

**Under development**

* Gesture recognition
* Voice recognition
* MCP (Model Context Protocol)

**New track: Agentic AI**

* How to use agentic frameworks
* Memory management
* Token economy
* Documentation
* Best practices and advanced prompting methods

## Getting started

See [GettingStartedWithColab.md](GettingStartedWithColab.md) to run the notebooks in Google Colab.
