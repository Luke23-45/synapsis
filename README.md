# NTH-Attention: Neuro-Topological Hysteresis Attention Architecture

A next-generation robotic memory system that combats contextual amnesia and geometric blindness through differentiable temporal anchoring and topological feature extraction.

## Installation

```bash
cd nth_attention
pip install -e .
```

## Project Structure

```
nth_attention/
├── src/
│   └── nth/
│       ├── __init__.py
│       ├── config.py              # Configuration dataclasses
│       ├── temporal_sampler.py    # Module 1: Adaptive Temporal Sampler
│       ├── geometric_projector.py # Module 2: Geometric Manifold Projector
│       ├── nth_attention.py       # Module 3: NTH-Attention Transformer
│       ├── action_head.py         # Flow Matching Action Head
│       ├── planner.py             # Unified NTH Diffusion Planner
│       ├── losses.py              # Loss functions
│       └── utils.py               # Utility functions
├── tests/
│   └── test_modules.py
├── scripts/
│   └── train.py
├── new/                           # Design documentation
│   └── agent/
├── pyproject.toml
└── README.md
```

## Quick Start

```python
from nth import NTHConfig, NTHDiffusionPlanner

config = NTHConfig()
model = NTHDiffusionPlanner(config)

# Inference
actions = model.predict(batch)
```

## Key Features

- **Differentiable Top-K Selection**: Gumbel-Softmax based temporal anchoring
- **Self-Similarity Matrix (SSM)**: TDA proxy for topological features
- **Geometry-Biased Attention**: AdaLN-Zero conditioned transformer with geometric bias
- **Flow Matching**: SOTA action generation replacing DDIM

## References

See `new/agent/08-REFERENCE.md` for full citations.
