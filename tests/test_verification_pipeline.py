from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from synapse_arch.model import SynapseArchitectureConfig, SynapseEndToEndModel
from synapse_core import anchor_selector
from synapse_core.memory_operator import compute_memory
from synapse_core.topological_summary import compute_persistence_diagrams
from synapse_core.training_readout import compute_relaxed_readout


def _config(**overrides: int | float) -> SynapseArchitectureConfig:
    base = dict(
        input_dim=6,
        action_dim=3,
        action_chunk_size=2,
        hidden_dim=16,
        d_model=16,
        num_heads=4,
        num_layers=1,
        ffn_ratio=2,
        dropout=0.0,
        K=4,
        r=2,
        lam=0.5,
        Q=1,
        k=8,
        max_history_tokens=32,
    )
    base.update(overrides)
    return SynapseArchitectureConfig(**base)


def _trajectory(steps: int = 12, dims: int = 6) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    cols = [
        np.sin(2.0 * np.pi * t),
        np.cos(2.0 * np.pi * t),
        t,
        t ** 2,
        np.sign(np.sin(4.0 * np.pi * t)),
        np.linspace(-1.0, 1.0, steps, dtype=np.float64),
    ]
    return np.stack(cols[:dims], axis=1)


def _batch(batch_size: int = 1, steps: int = 12, dims: int = 6) -> dict:
    sequence = torch.from_numpy(_trajectory(steps=steps, dims=dims)).float()
    history = sequence.unsqueeze(0).repeat(batch_size, 1, 1)
    return {
        "structured_state": history[:, -1, :].clone(),
        "structured_history": history,
        "action_chunk": torch.zeros(batch_size, 2, 3),
    }


def _lift_matrix(dims: int, lift_dim: int = 8) -> np.ndarray:
    return np.linspace(
        -0.3,
        0.3,
        lift_dim * (dims + 3),
        dtype=np.float64,
    ).reshape(lift_dim, dims + 3)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("K", 0, "K must be >= 1"),
        ("r", -1, "r must be >= 0"),
        ("lam", 0.0, "lam must be > 0"),
        ("max_history_tokens", 0, "max_history_tokens must be >= 1"),
    ],
)
def test_model_validates_invalid_selector_config(field: str, value: int | float, message: str):
    kwargs = {field: value}
    with pytest.raises(ValueError, match=message):
        SynapseEndToEndModel(_config(**kwargs))


def test_train_and_deploy_agree_on_event_saliency_and_selector():
    model = SynapseEndToEndModel(_config())
    batch = _batch(batch_size=1)

    train_out = model.forward_train(batch)
    deploy_out = model.forward_deploy(batch)
    exact = deploy_out.exact_memory_states[0]

    np.testing.assert_allclose(
        train_out.event_scores[0].detach().cpu().numpy(),
        exact.event_scores,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        train_out.saliency_scores[0].detach().cpu().numpy(),
        exact.saliency_scores,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        train_out.y_star[0].detach().cpu().numpy(),
        exact.y_star,
        rtol=1e-6,
        atol=1e-6,
    )


def test_relaxed_readout_matches_model_training_lift():
    model = SynapseEndToEndModel(_config())
    sequence = _trajectory(steps=12, dims=model.config.input_dim)

    state = compute_relaxed_readout(
        sequence,
        K=model.config.K,
        r=model.config.r,
        lam=model.config.lam,
        W_Theta=model.normalized_lift.W_theta.detach().cpu().numpy().astype(np.float64),
        mu=model.normalized_lift.mu.detach().cpu().numpy().astype(np.float64),
        sigma=model.normalized_lift.sigma.detach().cpu().numpy().astype(np.float64),
        solver=model.relaxed_selector.solver,
        saliency_mode="temperature",
        saliency_temperature=float(torch.exp(model.saliency_normalizer.log_temperature).item()),
        saliency_eps=model.saliency_normalizer.eps,
    )

    with torch.no_grad():
        candidate_vectors = torch.from_numpy(state.candidate_vectors).unsqueeze(0).float()
        normalized_vectors, lifted_candidates = model.normalized_lift(candidate_vectors)

    np.testing.assert_allclose(
        state.y_star,
        anchor_selector.solve_relaxed_selector(
            state.saliency_scores,
            model.config.K,
            model.config.r,
            model.config.lam,
            solver=model.relaxed_selector.solver,
        ),
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        state.normalized_vectors,
        normalized_vectors[0].detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        state.lifted_candidates,
        lifted_candidates[0].detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        state.weighted_cloud,
        state.lifted_candidates * state.y_star[:, None],
        rtol=1e-6,
        atol=1e-6,
    )


def test_compute_memory_handles_single_timestep_boundary_case():
    trajectory = np.array([[0.25, -0.5, 0.75]], dtype=np.float64)
    state = compute_memory(
        trajectory,
        K=3,
        r=2,
        lam=0.5,
        W_Theta=_lift_matrix(dims=trajectory.shape[1], lift_dim=4),
        Q=1,
        solver="osqp",
    )

    assert state.anchor_indices == []
    assert state.y_star.shape == (1,)
    assert state.point_cloud.shape == (0, 4)
    assert len(state.persistence_diagrams) == 2
    assert state.persistence_diagrams == [[], []]


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_compute_memory_rejects_non_finite_inputs(bad_value: float):
    trajectory = _trajectory()
    W_theta = _lift_matrix(dims=trajectory.shape[1], lift_dim=6)

    bad_trajectory = trajectory.copy()
    bad_trajectory[3, 1] = bad_value
    with pytest.raises(ValueError, match="trajectory must contain only finite values"):
        compute_memory(bad_trajectory, K=3, r=1, lam=0.5, W_Theta=W_theta, Q=0, solver="osqp")

    bad_W = W_theta.copy()
    bad_W[0, 0] = bad_value
    with pytest.raises(ValueError, match="W_Theta must contain only finite values"):
        compute_memory(trajectory, K=3, r=1, lam=0.5, W_Theta=bad_W, Q=0, solver="osqp")


def test_solver_failure_propagates_runtime_error(monkeypatch: pytest.MonkeyPatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("forced solver failure")

    monkeypatch.setattr(anchor_selector, "_solve_osqp", _boom)

    with pytest.raises(RuntimeError, match="forced solver failure"):
        anchor_selector.solve_relaxed_selector(
            np.array([0.0, 0.4, 0.7, 0.2], dtype=np.float64),
            K=2,
            r=1,
            lam=0.5,
            solver="osqp",
        )


def test_scipy_solver_produces_feasible_solution():
    y_star = anchor_selector.solve_relaxed_selector(
        np.array([0.0, 0.4, 0.7, 0.2], dtype=np.float64),
        K=2,
        r=1,
        lam=0.5,
        solver="scipy",
    )

    assert y_star.shape == (4,)
    assert np.all(np.isfinite(y_star))
    assert y_star[0] == 0.0
    assert np.all((0.0 <= y_star) & (y_star <= 1.0))
    assert y_star.sum() <= 2.0 + 1e-6
    assert y_star[1] + y_star[2] <= 1.0 + 1e-6
    assert y_star[2] + y_star[3] <= 1.0 + 1e-6


def test_osqp_path_uses_scipy_when_cvxpy_fallback_fails(monkeypatch: pytest.MonkeyPatch):
    class _FakeInfo:
        status_val = 7
        status = "maximum iterations reached"

    class _FakeResult:
        info = _FakeInfo()

    class _FakeProb:
        def setup(self, *args, **kwargs):
            return None

        def update(self, **kwargs):
            return None

        def solve(self):
            return _FakeResult()

    class _FakeOSQPModule:
        class OSQP(_FakeProb):
            pass

    def _cvxpy_boom(*args, **kwargs):
        raise RuntimeError("cvxpy user_limit")

    monkeypatch.setitem(sys.modules, "osqp", _FakeOSQPModule())
    monkeypatch.setattr(anchor_selector, "_solve_cvxpy", _cvxpy_boom)

    y_star = anchor_selector.solve_relaxed_selector(
        np.array([0.0, 0.4, 0.7, 0.2], dtype=np.float64),
        K=2,
        r=1,
        lam=0.5,
        solver="osqp",
    )

    assert y_star.shape == (4,)
    assert np.all(np.isfinite(y_star))


def test_compute_memory_respects_budget_under_extreme_refractory():
    trajectory = _trajectory(steps=14, dims=5)
    state = compute_memory(
        trajectory,
        K=5,
        r=trajectory.shape[0],
        lam=1e-4,
        W_Theta=_lift_matrix(dims=trajectory.shape[1], lift_dim=6),
        Q=0,
        solver="osqp",
    )

    assert len(state.anchor_indices) <= 1
    assert state.anchor_indices == sorted(state.anchor_indices)


def test_persistence_diagrams_handle_degenerate_cloud_without_nan():
    cloud = np.zeros((4, 3), dtype=np.float64)
    diagrams = compute_persistence_diagrams(cloud, Q=1)

    assert len(diagrams) == 2
    finite_values = []
    for dgm in diagrams:
        for birth, death in dgm:
            if np.isfinite(birth):
                finite_values.append(birth)
            if np.isfinite(death):
                finite_values.append(death)
    assert np.all(np.isfinite(np.asarray(finite_values, dtype=np.float64)))
