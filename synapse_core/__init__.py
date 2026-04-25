"""
SYNAPSE Core: Pure Mathematical Memory Operator
================================================

This package implements the exact mathematical object defined in:
  docs/formal_math/z2/02_rigorous_architecture.md  (Z2 — primary)
  docs/formal_math/01_main_definition.md             (Z1 — legacy)

Every function is a faithful transcription of the formal definition.
No neural networks, no approximations, no heuristics.
No experiment infrastructure, no config, no I/O.

Z2 Usage
--------
::

    from synapse_core import compute_memory
    result = compute_memory(trajectory, K=10, r=2, lam=0.5, W_Theta=W, Q=1)

Z1 Usage (legacy)
-----------------
::

    from synapse_core import M
    result = M(trajectory, K=10, r=2, tau=0.5, weights=(1,1,1,1), Q=1)
"""

from .event_encoder import sharp_event_score, hysteretic_event_score
from .anchor_selector import (
    select_anchors, admissible_index_sets,
    solve_relaxed_selector, hard_projection, build_anchors, Anchor,
)
from .geometric_lift import (
    lift_anchors, lift_single_anchor,
    anchor_vectors, normalize_anchors, apply_lift,
)
from .saliency_normalizer import causal_running_stats, normalize_saliency
from .training_readout import (
    RelaxedReadoutState,
    candidate_anchor_vectors,
    relaxed_weighted_cloud,
    compute_relaxed_readout,
)
from .topological_summary import compute_persistence_diagrams
from .memory_operator import M, compute_memory, MemoryState, Z2MemoryState

__all__ = [
    # Z2 primary API
    "compute_memory",
    "Z2MemoryState",
    "solve_relaxed_selector",
    "hard_projection",
    "build_anchors",
    "anchor_vectors",
    "normalize_anchors",
    "apply_lift",
    "normalize_saliency",
    "causal_running_stats",
    "compute_relaxed_readout",
    "candidate_anchor_vectors",
    "relaxed_weighted_cloud",
    "RelaxedReadoutState",
    "Anchor",
    # Shared (Z1 + Z2)
    "sharp_event_score",
    "hysteretic_event_score",
    "compute_persistence_diagrams",
    # Z1 legacy
    "M",
    "MemoryState",
    "select_anchors",
    "admissible_index_sets",
    "lift_anchors",
    "lift_single_anchor",
]
