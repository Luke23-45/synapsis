"""
Tests for SYNAPSE Adapter + Feature Cache + Normalization (Agent 1)
====================================================================

Verifies:
    - Correct projection of pre-computed features
    - Output shapes match planner expectations for all conditions
    - Ablation flags correctly zero/remove tokens
    - Empty cloud / single anchor edge cases
    - Normalization stats computed from training split only
    - Cached features match on-the-fly M() output (numerical parity)
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config import ExperimentConfig, Condition, SynapseParams, DataParams, TransformerParams
from src.core.normalization import NormalizationStats, compute_normalization_stats
from src.synapse.synapse_adapter import SynapseAdapter
from src.synapse.synapse_cache import (
    pad_anchor_cloud,
    summarize_persistence_diagrams,
    compute_synapse_features,
    cache_synapse_features,
    verify_cache_consistency,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _smoke_config(condition: Condition = Condition.B_SYNAPSE) -> ExperimentConfig:
    return ExperimentConfig(
        condition=condition,
        seed=42,
        synapse=SynapseParams(K=5, r=2, tau=0.3, Q=1),
        transformer=TransformerParams(d_model=64, num_heads=4, num_layers=2),
        data=DataParams(proprio_dim=22, action_dim=8, action_chunk_size=4, history_window=10),
    )


@pytest.fixture
def config_b():
    return _smoke_config(Condition.B_SYNAPSE)


@pytest.fixture
def config_anchors():
    return _smoke_config(Condition.B_ANCHORS)


@pytest.fixture
def config_topo():
    return _smoke_config(Condition.B_TOPO)


@pytest.fixture
def norm_stats():
    rng = np.random.default_rng(42)
    arrays = [rng.normal(0, 1, size=(50, 22)) for _ in range(5)]
    return compute_normalization_stats(arrays)


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_shape_preservation(self, norm_stats):
        x = np.random.randn(10, 22)
        x_norm = norm_stats.normalize(x)
        assert x_norm.shape == x.shape

    def test_zero_mean_unit_var(self, norm_stats):
        x = np.random.randn(1000, 22)
        x_norm = norm_stats.normalize(x)
        # After normalization, should be approximately zero-mean, unit-var
        np.testing.assert_allclose(x_norm.mean(axis=0), 0.0, atol=0.15)
        np.testing.assert_allclose(x_norm.std(axis=0), 1.0, atol=0.15)

    def test_roundtrip(self, norm_stats):
        x = np.random.randn(10, 22)
        x_norm = norm_stats.normalize(x)
        x_denorm = norm_stats.denormalize(x_norm)
        np.testing.assert_allclose(x_denorm, x, atol=1e-10)

    def test_torch_normalize(self, norm_stats):
        x = torch.randn(10, 22)
        x_norm = norm_stats.normalize_torch(x)
        assert x_norm.shape == x.shape
        assert x_norm.dtype == x.dtype

    def test_dim_mismatch_raises(self, norm_stats):
        x = np.random.randn(10, 15)  # Wrong dimension
        with pytest.raises(ValueError, match="dimension mismatch"):
            norm_stats.normalize(x)

    def test_zero_std_guard(self):
        arrays = [np.ones((10, 3)) * 5.0]  # Constant dimension
        stats = compute_normalization_stats(arrays)
        # Should not crash; std should be set to 1.0
        assert stats.std[0] == 1.0

    def test_save_load_roundtrip(self, norm_stats, tmp_path):
        path = tmp_path / "norm_stats.pt"
        norm_stats.save(path)
        loaded = NormalizationStats.load(path)
        np.testing.assert_allclose(loaded.mean, norm_stats.mean)
        np.testing.assert_allclose(loaded.std, norm_stats.std)

    def test_compute_from_episodes(self):
        episodes = [
            {"proprio_history": np.random.randn(50, 22)},
            {"proprio_history": np.random.randn(30, 22)},
        ]
        from src.core.normalization import compute_normalization_stats_from_episodes
        stats = compute_normalization_stats_from_episodes(episodes)
        assert stats.dim == 22


# ---------------------------------------------------------------------------
# SynapseAdapter tests
# ---------------------------------------------------------------------------

class TestSynapseAdapter:
    def test_output_shape_b_full(self, config_b):
        adapter = SynapseAdapter(config_b)
        B = 4
        anchors = torch.randn(B, config_b.synapse.K, config_b.anchor_feature_dim)
        topo = torch.randn(B, config_b.topo_feature_dim)
        tokens, n_real = adapter(anchors, topo)
        assert tokens.shape == (B, config_b.synapse.K + 1, config_b.transformer.d_model)
        assert n_real == config_b.synapse.K + 1

    def test_output_shape_anchors_only(self, config_anchors):
        adapter = SynapseAdapter(config_anchors)
        B = 4
        anchors = torch.randn(B, config_anchors.synapse.K, config_anchors.anchor_feature_dim)
        topo = torch.randn(B, config_anchors.topo_feature_dim)
        tokens, n_real = adapter(anchors, topo)
        assert tokens.shape == (B, config_anchors.synapse.K, config_anchors.transformer.d_model)
        assert n_real == config_anchors.synapse.K

    def test_output_shape_topo_only(self, config_topo):
        adapter = SynapseAdapter(config_topo)
        B = 4
        anchors = torch.randn(B, config_topo.synapse.K, config_topo.anchor_feature_dim)
        topo = torch.randn(B, config_topo.topo_feature_dim)
        tokens, n_real = adapter(anchors, topo)
        assert tokens.shape == (B, 1, config_topo.transformer.d_model)
        assert n_real == 1

    def test_both_false_raises(self):
        config = ExperimentConfig(condition=Condition.A1_RECENT)
        with pytest.raises(ValueError, match="At least one"):
            SynapseAdapter(config)

    def test_num_output_tokens(self, config_b, config_anchors, config_topo):
        assert SynapseAdapter(config_b).num_output_tokens == config_b.synapse.K + 1
        assert SynapseAdapter(config_anchors).num_output_tokens == config_anchors.synapse.K
        assert SynapseAdapter(config_topo).num_output_tokens == 1

    def test_trainable_params(self, config_b):
        adapter = SynapseAdapter(config_b)
        params = adapter.num_trainable_params
        expected = (
            config_b.anchor_feature_dim * config_b.transformer.d_model
            + config_b.transformer.d_model  # bias
            + config_b.topo_feature_dim * config_b.transformer.d_model
            + config_b.transformer.d_model  # bias
        )
        assert params == expected

    def test_dummy_output(self, config_b):
        adapter = SynapseAdapter(config_b)
        tokens, n_real = adapter.forward_with_dummy(4, torch.device("cpu"))
        assert tokens.shape == (4, 0, config_b.transformer.d_model)
        assert n_real == 0

    def test_gradient_flows(self, config_b):
        adapter = SynapseAdapter(config_b)
        anchors = torch.randn(2, config_b.synapse.K, config_b.anchor_feature_dim)
        topo = torch.randn(2, config_b.topo_feature_dim)
        tokens, _ = adapter(anchors, topo)
        loss = tokens.sum()
        loss.backward()
        # Adapter parameters should have gradients
        assert adapter.anchor_proj.weight.grad is not None
        assert adapter.topo_proj.weight.grad is not None


# ---------------------------------------------------------------------------
# Feature cache tests
# ---------------------------------------------------------------------------

class TestFeatureCache:
    def test_pad_anchor_cloud_empty(self):
        padded, mask = pad_anchor_cloud(np.empty((0, 0)), K=5)
        assert padded.shape[0] == 5
        assert mask.shape[0] == 5
        assert not mask.any()

    def test_pad_anchor_cloud_fewer_than_K(self):
        cloud = np.random.randn(3, 25).astype(np.float32)
        padded, mask = pad_anchor_cloud(cloud, K=5)
        assert padded.shape == (5, 25)
        assert mask.sum() == 3
        np.testing.assert_array_equal(padded[:3], cloud)
        np.testing.assert_array_equal(padded[3:], 0.0)

    def test_pad_anchor_cloud_equal_K(self):
        cloud = np.random.randn(5, 25).astype(np.float32)
        padded, mask = pad_anchor_cloud(cloud, K=5)
        assert padded.shape == (5, 25)
        assert mask.all()

    def test_summarize_persistence_diagrams(self):
        diagrams = [
            [(0.0, 1.0), (0.0, 2.0), (0.0, float("inf"))],  # H_0
            [(0.5, 1.5)],  # H_1
        ]
        summary = summarize_persistence_diagrams(diagrams, Q=1)
        assert summary.shape == (8,)  # 4 * (Q+1) = 4 * 2 = 8
        # H_0: 2 finite bars
        assert summary[0] == 2.0  # count
        assert summary[1] == 1.5  # mean persistence: (1.0 + 2.0) / 2
        assert summary[2] == 2.0  # max persistence
        assert summary[3] == 3.0  # total persistence: 1.0 + 2.0
        # H_1: 1 finite bar
        assert summary[4] == 1.0  # count
        assert summary[5] == 1.0  # mean persistence: 1.5 - 0.5
        assert summary[6] == 1.0  # max persistence
        assert summary[7] == 1.0  # total persistence

    def test_compute_synapse_features(self, config_b, norm_stats):
        proprio = np.random.randn(50, 22)
        features = compute_synapse_features(proprio, config_b, norm_stats)
        assert features.anchors.shape == (config_b.synapse.K, config_b.anchor_feature_dim)
        assert features.anchor_mask.shape == (config_b.synapse.K,)
        assert features.topo.shape == (config_b.topo_feature_dim,)

    def test_cache_and_load(self, config_b, norm_stats, tmp_path):
        episodes = [
            {"episode_id": f"ep_{i}", "proprio_history": np.random.randn(30, 22)}
            for i in range(5)
        ]
        output_path = tmp_path / "test_cache.pt"
        metadata = cache_synapse_features(episodes, config_b, norm_stats, output_path)
        assert metadata["num_episodes"] == 5
        assert metadata["num_errors"] == 0

        from src.synapse.synapse_cache import load_cached_features
        cached = load_cached_features(output_path)
        assert cached["anchors"].shape[0] == 5
        assert cached["topo"].shape[0] == 5

    def test_cache_consistency(self, config_b, norm_stats, tmp_path):
        episodes = [
            {"episode_id": f"ep_{i}", "proprio_history": np.random.default_rng(i).normal(size=(30, 22))}
            for i in range(3)
        ]
        output_path = tmp_path / "consistency_cache.pt"
        cache_synapse_features(episodes, config_b, norm_stats, output_path)

        from src.synapse.synapse_cache import load_cached_features
        cached = load_cached_features(output_path)
        issues = verify_cache_consistency(episodes, cached, config_b, norm_stats, max_check=3)
        assert len(issues) == 0
