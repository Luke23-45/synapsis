from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from synapse_arch.model import SynapseArchitectureConfig, SynapseEndToEndModel
from synapse_core.anchor_selector import hard_projection as core_hard_projection


def _config() -> SynapseArchitectureConfig:
    return SynapseArchitectureConfig(
        input_dim=39,
        action_dim=8,
        action_chunk_size=4,
        hidden_dim=32,
        d_model=32,
        num_heads=4,
        num_layers=1,
        ffn_ratio=2,
        dropout=0.0,
        K=4,
        r=2,
        lam=0.5,
        Q=1,
        k=16,
        max_history_tokens=64,
    )


def _batch(batch_size: int = 2, steps: int = 16) -> dict:
    return {
        "structured_state": torch.randn(batch_size, 39),
        "structured_history": torch.randn(batch_size, steps, 39),
        "action_chunk": torch.randn(batch_size, 4, 8),
    }


def test_forward_train_backward():
    model = SynapseEndToEndModel(_config())
    batch = _batch()
    out = model.forward_train(batch)
    loss = out.pred_actions.square().mean()
    loss.backward()
    assert out.pred_actions.shape == (2, 4, 8)
    assert model.normalized_lift.W_theta.grad is not None


def test_forward_deploy_returns_exact_memory_state():
    model = SynapseEndToEndModel(_config())
    batch = _batch(batch_size=1, steps=12)
    out = model.forward_deploy(batch)
    assert out.pred_actions.shape == (1, 4, 8)
    assert len(out.exact_memory_states) == 1
    state = out.exact_memory_states[0]
    assert state.y_star.shape == (12,)
    assert state.point_cloud.ndim == 2
    assert len(state.persistence_diagrams) == 2


def test_hard_projection_parity():
    model = SynapseEndToEndModel(_config())
    y_star = np.array([0.0, 0.8, 0.4, 0.7, 0.2, 0.9], dtype=np.float64)
    ours = model.hard_projector.project(y_star)
    core = core_hard_projection(y_star, K=model.config.K, r=model.config.r)
    assert ours == core


def test_train_deploy_interfaces_are_separate():
    model = SynapseEndToEndModel(_config())
    batch = _batch(batch_size=1, steps=10)
    train_out = model.forward_train(batch)
    deploy_out = model.forward_deploy(batch)
    assert hasattr(train_out, "y_star")
    assert hasattr(deploy_out, "exact_memory_states")
    assert deploy_out.exact_memory_states[0].anchor_indices == sorted(deploy_out.exact_memory_states[0].anchor_indices)
