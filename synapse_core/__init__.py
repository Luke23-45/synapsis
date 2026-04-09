"""
SYNAPSE Core: Pure Mathematical Memory Operator
================================================

This package implements the exact mathematical object defined in:
  docs/formal_math/01_main_definition.md
  docs/formal_math/02_rigorous_architecture.md

Every function is a faithful transcription of the formal definition.
No neural networks, no approximations, no heuristics.
No experiment infrastructure, no config, no I/O.

Usage
-----
::

    from synapse_core import M
    result = M(trajectory, K=10, r=2, tau=0.5, weights=(1,1,1,1), Q=1)
"""

from .event_encoder import sharp_event_score, hysteretic_event_score
from .anchor_selector import select_anchors, admissible_index_sets
from .geometric_lift import lift_anchors, lift_single_anchor
from .topological_summary import compute_persistence_diagrams
from .memory_operator import M

__all__ = [
    "sharp_event_score",
    "hysteretic_event_score",
    "select_anchors",
    "admissible_index_sets",
    "lift_anchors",
    "lift_single_anchor",
    "compute_persistence_diagrams",
    "M",
]
