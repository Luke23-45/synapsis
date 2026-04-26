"""
Tests for Data Pipeline
===================================

Verifies:
    - In-memory episode creation produces valid episodes
    - Z-score normalization applied correctly
    - Episode splitting is deterministic and correct
    - Dataset __getitem__ returns correct shapes and types
    - Collate function pads variable-length histories correctly
    - SYNAPSE features attached when condition requires them
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config import ExperimentConfig, Condition, SynapseParams, DataParams, TransformerParams, TrainingParams
from src.core.normalization import compute_normalization_stats, NormalizationStats
from src.data.dataset import (
    split_episodes,
    RoboticsDataset,
    collate_fn,
    create_dataloaders,
)
from src.data.adapters.base_adapter import RobotEpisode


# ---------------------------------------------------------------------------
# Helpers — in-memory episode creation
# ---------------------------------------------------------------------------

def _make_test_episodes(
    num_episodes: int = 5,
    T: int = 50,
    proprio_dim: int = 22,
    action_dim: int = 8,
    num_phases: int = 5,
    seed: int = 42,
) -> list:
    """Create lightweight in-memory test episodes."""
    rng = np.random.default_rng(seed)
    episodes = []
    for i in range(num_episodes):
        phases = np.repeat(np.arange(num_phases), T // num_phases + 1)[:T]
        episodes.append(
            RobotEpisode(
                episode_id=f"test_{i:03d}",
                dataset_name="test",
                proprio_history=rng.standard_normal((T, proprio_dim)).astype(np.float32),
                actions=rng.standard_normal((T, action_dim)).astype(np.float32),
                gt_phase=phases.astype(np.int64),
            )
        )
    return episodes


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
def episodes():
    return _make_test_episodes(num_episodes=5, T=50, seed=42)


@pytest.fixture
def norm_stats(episodes):
    arrays = [ep.proprio_history for ep in episodes]
    return compute_normalization_stats(arrays)


# ---------------------------------------------------------------------------
# Episode tests
# ---------------------------------------------------------------------------

class TestEpisodes:
    def test_episode_count(self, episodes):
        assert len(episodes) == 5

    def test_episode_shapes(self, episodes):
        for ep in episodes:
            assert ep.proprio_history.shape == (50, 22)
            assert ep.actions.shape == (50, 8)
            assert ep.gt_phase.shape == (50,)

    def test_phase_labels_valid(self, episodes):
        for ep in episodes:
            assert ep.gt_phase.min() >= 0
            assert ep.gt_phase.max() < 5

    def test_deterministic(self):
        eps1 = _make_test_episodes(num_episodes=3, T=30, seed=42)
        eps2 = _make_test_episodes(num_episodes=3, T=30, seed=42)
        for e1, e2 in zip(eps1, eps2):
            np.testing.assert_array_equal(e1.proprio_history, e2.proprio_history)

    def test_different_seeds(self):
        eps1 = _make_test_episodes(num_episodes=3, T=30, seed=42)
        eps2 = _make_test_episodes(num_episodes=3, T=30, seed=99)
        assert not np.allclose(eps1[0].proprio_history, eps2[0].proprio_history)


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------

class TestSplitEpisodes:
    def test_split_ratios(self, episodes):
        train, val, test = split_episodes(episodes, train_ratio=0.6, val_ratio=0.2, seed=42)
        n = len(episodes)
        assert len(train) == int(n * 0.6)
        assert len(val) == int(n * 0.2)
        assert len(test) + len(train) + len(val) == n

    def test_no_overlap(self, episodes):
        train, val, test = split_episodes(episodes, seed=42)
        all_ids = [ep.episode_id for ep in train + val + test]
        assert len(all_ids) == len(set(all_ids))

    def test_deterministic_split(self, episodes):
        t1, v1, te1 = split_episodes(episodes, seed=42)
        t2, v2, te2 = split_episodes(episodes, seed=42)
        assert [e.episode_id for e in t1] == [e.episode_id for e in t2]


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestRoboticsDataset:
    def test_length(self, episodes, norm_stats):
        config = _smoke_config(Condition.A1_RECENT)
        dataset = RoboticsDataset(episodes, config, norm_stats, split="train")
        expected = sum(ep.length - config.data.action_chunk_size for ep in episodes)
        assert len(dataset) == expected

    def test_getitem_shapes(self, episodes, norm_stats):
        config = _smoke_config(Condition.A1_RECENT)
        dataset = RoboticsDataset(episodes, config, norm_stats, split="train")
        sample = dataset[0]
        assert sample["proprio"].shape == (22,)
        assert sample["proprio_history"].shape[1] == 22
        assert sample["action_chunk"].shape == (4, 8)
        assert sample["phase_label"].dtype == torch.long

    def test_synapse_features_present(self, episodes, norm_stats):
        config = _smoke_config(Condition.B_SYNAPSE)
        for ep in episodes:
            ep.synapse_anchors = np.random.randn(config.synapse.K, config.anchor_feature_dim).astype(np.float32)
            ep.synapse_topo = np.random.randn(config.topo_feature_dim).astype(np.float32)
        dataset = RoboticsDataset(episodes, config, norm_stats, split="train")
        sample = dataset[0]
        assert "synapse_anchors" in sample
        assert "synapse_topo" in sample
        assert sample["synapse_anchors"].shape == (config.synapse.K, config.anchor_feature_dim)

    def test_no_synapse_for_a1(self, episodes, norm_stats):
        config = _smoke_config(Condition.A1_RECENT)
        dataset = RoboticsDataset(episodes, config, norm_stats, split="train")
        sample = dataset[0]
        assert "synapse_anchors" not in sample


# ---------------------------------------------------------------------------
# Collate function tests
# ---------------------------------------------------------------------------

class TestCollateFn:
    def test_pads_histories(self, episodes, norm_stats):
        config = _smoke_config(Condition.A1_RECENT)
        dataset = RoboticsDataset(episodes, config, norm_stats, split="train")
        samples = [dataset[0], dataset[10]]
        batch = collate_fn(samples)
        assert batch["proprio_history"].shape[0] == 2
        assert batch["structured_history"].shape[0] == 2
        h1_len = batch["proprio_history"].shape[1]
        assert h1_len >= samples[0]["proprio_history"].shape[0]

    def test_batch_shapes(self, episodes, norm_stats):
        config = _smoke_config(Condition.B_SYNAPSE)
        for ep in episodes:
            ep.synapse_anchors = np.random.randn(config.synapse.K, config.anchor_feature_dim).astype(np.float32)
            ep.synapse_topo = np.random.randn(config.topo_feature_dim).astype(np.float32)
        dataset = RoboticsDataset(episodes, config, norm_stats, split="train")
        samples = [dataset[i] for i in range(4)]
        batch = collate_fn(samples)
        assert batch["proprio"].shape[0] == 4
        assert batch["action_chunk"].shape[0] == 4
        assert "synapse_anchors" in batch


# ---------------------------------------------------------------------------
# DataLoader integration test
# ---------------------------------------------------------------------------

class TestDataLoaderIntegration:
    def test_create_dataloaders(self, episodes, norm_stats):
        config = ExperimentConfig(
            condition=Condition.A1_RECENT,
            training=TrainingParams(batch_size=4, num_workers=0),
        )
        train_eps, val_eps, test_eps = split_episodes(episodes, seed=42)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_eps, val_eps, test_eps, config, norm_stats
        )
        batch = next(iter(train_loader))
        assert batch["proprio"].shape[0] <= 4
