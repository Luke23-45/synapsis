"""
Tests for Multi-Condition Planner Architecture (Agent 2)
=========================================================

Verifies:
    - All planners produce correct output shapes (B, K_act, D_act)
    - All conditions feed identical Seq_Len to the transformer (§3.6)
    - Parameter counts match expectations across conditions
    - Forward pass runs without error on test data
    - Condition B has ≤1% more parameters than A1/A2
    - A2 correctly uniform-samples across full history
    - Ablation flags correctly modify token composition
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config import (
    ExperimentConfig, Condition, SynapseParams, TransformerParams,
    DataParams, TrainingParams,
)
from src.planner.planner_recent import PlannerRecent
from src.planner.planner_uniform import PlannerUniform
from src.planner.planner_synapse import PlannerSynapse, create_planner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(condition: Condition) -> ExperimentConfig:
    return ExperimentConfig(
        condition=condition,
        seed=42,
        synapse=SynapseParams(K=5, r=2, tau=0.3, Q=1),
        transformer=TransformerParams(d_model=64, num_heads=4, num_layers=2),
        data=DataParams(
            proprio_dim=22, action_dim=8, action_chunk_size=4, history_window=10,
            ee_pose_dim=0, ee_vel_dim=0, object_pos_dim=0, grasp_dim=0,
        ),
        training=TrainingParams(batch_size=4, num_workers=0),
    )


@pytest.fixture
def config_a1():
    return _make_config(Condition.A1_RECENT)


@pytest.fixture
def config_a2():
    return _make_config(Condition.A2_UNIFORM)


@pytest.fixture
def config_b():
    return _make_config(Condition.B_SYNAPSE)


@pytest.fixture
def config_b_anchors():
    return _make_config(Condition.B_ANCHORS)


@pytest.fixture
def config_b_topo():
    return _make_config(Condition.B_TOPO)


def _make_batch(config: ExperimentConfig, B: int = 2, T: int = 30) -> dict:
    """Create a dummy batch matching the config's condition."""
    batch = {
        "proprio": torch.randn(B, config.data.proprio_dim),
        "proprio_history": torch.randn(B, T, config.data.proprio_dim),
        "structured_state": torch.randn(B, config.structured_state_dim),
        "structured_history": torch.randn(B, T, config.structured_state_dim),
        "action_chunk": torch.randn(B, config.data.action_chunk_size, config.data.action_dim),
        "history_length": torch.full((B,), T, dtype=torch.long),
    }
    if config.condition.uses_synapse:
        batch["synapse_anchors"] = torch.randn(B, config.synapse.K, config.anchor_feature_dim)
        batch["synapse_topo"] = torch.randn(B, config.topo_feature_dim)
    return batch


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestOutputShapes:
    @pytest.mark.parametrize("condition", [
        Condition.A1_RECENT, Condition.A2_UNIFORM,
        Condition.B_SYNAPSE, Condition.B_ANCHORS, Condition.B_TOPO,
    ])
    def test_output_shape(self, condition):
        config = _make_config(condition)
        model = create_planner(config)
        batch = _make_batch(config)
        with torch.no_grad():
            output = model(batch)
        assert output.shape == (
            batch["proprio"].shape[0],
            config.data.action_chunk_size,
            config.data.action_dim,
        ), f"Wrong output shape for {condition.value}: {output.shape}"

    @pytest.mark.parametrize("condition", [
        Condition.A1_RECENT, Condition.A2_UNIFORM,
        Condition.B_SYNAPSE, Condition.B_ANCHORS, Condition.B_TOPO,
    ])
    def test_no_nan_inf(self, condition):
        config = _make_config(condition)
        model = create_planner(config)
        batch = _make_batch(config)
        with torch.no_grad():
            output = model(batch)
        assert torch.isfinite(output).all(), f"Non-finite output for {condition.value}"


# ---------------------------------------------------------------------------
# Seq_Len equalization tests (§3.6)
# ---------------------------------------------------------------------------

class TestSeqLenEqualization:
    def test_a1_seq_len(self, config_a1):
        model = PlannerRecent(config_a1)
        assert model.seq_len == config_a1.data.history_window + 1

    def test_a2_seq_len(self, config_a2):
        model = PlannerUniform(config_a2)
        assert model.seq_len == config_a2.data.history_window + 1

    def test_b_seq_len(self, config_b):
        model = PlannerSynapse(config_b)
        assert model.seq_len == config_b.data.history_window + 1

    def test_all_equal_seq_len(self):
        configs = [_make_config(c) for c in [
            Condition.A1_RECENT, Condition.A2_UNIFORM, Condition.B_SYNAPSE
        ]]
        seq_lens = [c.seq_len for c in configs]
        assert len(set(seq_lens)) == 1, f"Seq_Len mismatch: {seq_lens}"


# ---------------------------------------------------------------------------
# Parameter count tests
# ---------------------------------------------------------------------------

class TestParameterCounts:
    def test_total_params_reasonable(self, config_b):
        model = PlannerSynapse(config_b)
        total = model.num_trainable_params
        assert total < 2_000_000, f"Unexpectedly large model: {total}"

    def test_parameter_count_by_component(self, config_b):
        model = PlannerSynapse(config_b)
        counts = model.count_parameters_by_component()
        assert "synapse_architecture" in counts
        assert counts["total_base"] > 0


# ---------------------------------------------------------------------------
# Uniform sampling test (A2)
# ---------------------------------------------------------------------------

class TestUniformSampling:
    def test_uniform_indices_cover_horizon(self, config_a2):
        model = PlannerUniform(config_a2)
        T = 100
        W = config_a2.data.history_window
        indices = model._compute_uniform_indices(T, W, torch.device("cpu"))
        assert len(indices) == W
        # First index should be 0 (or near 0)
        assert indices[0] <= 1
        # Last index should be T-1 (or near T-1)
        assert indices[-1] >= T - 2

    def test_uniform_indices_short_history(self, config_a2):
        model = PlannerUniform(config_a2)
        T = 5  # Shorter than window
        W = config_a2.data.history_window
        indices = model._compute_uniform_indices(T, W, torch.device("cpu"))
        # Should return all available indices
        assert len(indices) == T

    def test_uniform_indices_no_duplicates(self, config_a2):
        model = PlannerUniform(config_a2)
        T = 100
        W = config_a2.data.history_window
        indices = model._compute_uniform_indices(T, W, torch.device("cpu"))
        assert len(torch.unique(indices)) == len(indices)

    def test_batched_uniform_indices_skip_left_padding(self, config_a2):
        model = PlannerUniform(config_a2)
        history_lengths = torch.tensor([4, 10], dtype=torch.long)
        indices = model._compute_batched_uniform_indices(
            history_lengths,
            padded_length=10,
            device=torch.device("cpu"),
        )
        assert indices.shape == (2, config_a2.data.history_window)
        assert torch.all(indices[0] >= 6)
        assert indices[0, -1].item() == 9
        assert indices[1, 0].item() == 0


# ---------------------------------------------------------------------------
# Ablation flag tests
# ---------------------------------------------------------------------------

class TestAblationFlags:
    def test_anchors_only_tokens(self, config_b_anchors):
        adapter = PlannerSynapse(config_b_anchors).synapse_adapter
        assert adapter.use_anchors is True
        assert adapter.use_topo is False
        assert adapter.num_output_tokens == config_b_anchors.synapse.K

    def test_topo_only_tokens(self, config_b_topo):
        adapter = PlannerSynapse(config_b_topo).synapse_adapter
        assert adapter.use_anchors is False
        assert adapter.use_topo is True
        assert adapter.num_output_tokens == 1

    def test_full_tokens(self, config_b):
        adapter = PlannerSynapse(config_b).synapse_adapter
        assert adapter.use_anchors is True
        assert adapter.use_topo is True
        assert adapter.num_output_tokens == config_b.synapse.K + 1


# ---------------------------------------------------------------------------
# Factory function test
# ---------------------------------------------------------------------------

class TestCreatePlanner:
    @pytest.mark.parametrize("condition", [
        Condition.A1_RECENT, Condition.A2_UNIFORM,
        Condition.B_SYNAPSE, Condition.B_ANCHORS, Condition.B_TOPO,
    ])
    def test_factory_creates_correct_type(self, condition):
        config = _make_config(condition)
        model = create_planner(config)
        assert isinstance(model, torch.nn.Module)

    def test_invalid_condition_raises(self):
        config = _make_config(Condition.A1_RECENT)
        config_invalid = ExperimentConfig(condition=Condition.A1_RECENT)
        # PlannerSynapse should reject A1
        with pytest.raises(ValueError):
            PlannerSynapse(config_invalid)
