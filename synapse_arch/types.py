from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np
import torch


@dataclass
class ExactMemoryState:
    anchor_indices: List[int]
    y_star: np.ndarray
    event_scores: np.ndarray
    saliency_scores: np.ndarray
    anchor_vectors: np.ndarray
    normalized_anchor_vectors: np.ndarray
    point_cloud: np.ndarray
    persistence_diagrams: List[Any]
    topology_summary: np.ndarray


@dataclass
class TrainForwardOutput:
    pred_actions: torch.Tensor
    event_scores: torch.Tensor
    saliency_scores: torch.Tensor
    y_star: torch.Tensor
    dense_lifted_tokens: torch.Tensor
    topology_token: torch.Tensor
    transformer_tokens: torch.Tensor


@dataclass
class DeployForwardOutput:
    pred_actions: torch.Tensor
    exact_memory_states: List[ExactMemoryState]
    transformer_tokens: torch.Tensor
