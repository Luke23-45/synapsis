"""
Integration Tests for Dataset Adapters
========================================

Tests that adapters produce valid RobotEpisode instances with correct shapes,
dtypes, and data integrity. Uses LeRobot adapters for all production tests
and lightweight in-memory episodes for unit-level validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.adapters.base_adapter import RobotEpisode
from src.data.adapters.lerobot_adapter import LeRobotAdapter, LEROBOT_SPECS
from src.data.adapters.lmdb_adapter import LMDBAdapter, LMDB_SPECS
from src.data.registry import create_adapter, list_available_datasets
from src.data.dataset import (
    RoboticsDataset,
    collate_fn,
    split_episodes,
)
from src.core.config import ExperimentConfig, Condition, DatasetSpec


# ---------------------------------------------------------------------------
# Helpers — create lightweight in-memory episodes for unit tests
# (no external adapter needed; just raw RobotEpisode construction)
# ---------------------------------------------------------------------------

def _make_episodes(
    n: int = 5,
    T: int = 50,
    proprio_dim: int = 10,
    action_dim: int = 4,
    num_phases: int = 3,
    dataset_name: str = "test",
) -> list:
    """Create lightweight in-memory RobotEpisode instances for testing."""
    rng = np.random.default_rng(42)
    episodes = []
    for i in range(n):
        ep_len = T
        phases = np.repeat(np.arange(num_phases), ep_len // num_phases + 1)[:ep_len]
        episodes.append(
            RobotEpisode(
                episode_id=f"{dataset_name}_{i:03d}",
                dataset_name=dataset_name,
                proprio_history=rng.standard_normal((ep_len, proprio_dim)).astype(np.float32),
                actions=rng.standard_normal((ep_len, action_dim)).astype(np.float32),
                gt_phase=phases.astype(np.int64),
            )
        )
    return episodes


# ---------------------------------------------------------------------------
# RobotEpisode Unit Tests
# ---------------------------------------------------------------------------

class TestRobotEpisode:
    """Test the unified RobotEpisode dataclass."""

    def test_basic_properties(self):
        ep = RobotEpisode(
            episode_id="test_001",
            dataset_name="test",
            proprio_history=np.random.randn(100, 10).astype(np.float32),
            actions=np.random.randn(100, 5).astype(np.float32),
            gt_phase=np.zeros(100, dtype=np.int64),
        )
        assert ep.length == 100
        assert ep.structured_state_dim == 10  # Only proprio, no enrichments
        assert ep.structured_history.shape == (100, 10)

    def test_with_enrichments(self):
        ep = RobotEpisode(
            episode_id="test_002",
            dataset_name="test",
            proprio_history=np.random.randn(50, 22).astype(np.float32),
            actions=np.random.randn(50, 8).astype(np.float32),
            gt_phase=np.zeros(50, dtype=np.int64),
            ee_pose_history=np.random.randn(50, 7).astype(np.float32),
            ee_vel_history=np.random.randn(50, 6).astype(np.float32),
            object_pos_history=np.random.randn(50, 3).astype(np.float32),
            is_grasped_history=np.zeros((50, 1), dtype=np.float32),
        )
        assert ep.structured_state_dim == 22 + 7 + 6 + 3 + 1
        assert ep.structured_history.shape == (50, 39)

    def test_normalization_roundtrip(self):
        ep = RobotEpisode(
            episode_id="test_003",
            dataset_name="test",
            proprio_history=np.ones((20, 5), dtype=np.float32) * 3.0,
            actions=np.random.randn(20, 3).astype(np.float32),
            gt_phase=np.zeros(20, dtype=np.int64),
        )
        normalized = ep.structured_history * 0.5  # Fake normalization
        ep.apply_structured_normalization(normalized)
        assert ep.proprio_history.shape == (20, 5)
        np.testing.assert_allclose(ep.proprio_history, 1.5, atol=1e-6)


# ---------------------------------------------------------------------------
# LeRobot Adapter Tests
# ---------------------------------------------------------------------------

class TestLeRobotAdapter:
    """Test the LeRobot adapter specs are correctly defined."""

    def test_all_specs_registered(self):
        assert "pusht" in LEROBOT_SPECS
        assert "aloha_transfer" in LEROBOT_SPECS
        assert "xarm_lift" in LEROBOT_SPECS

    def test_pusht_spec(self):
        spec = LEROBOT_SPECS["pusht"]
        assert spec.proprio_dim == 2
        assert spec.action_dim == 2
        assert spec.num_phases == 3
        assert spec.hf_repo == "lerobot/pusht"

    def test_aloha_spec(self):
        spec = LEROBOT_SPECS["aloha_transfer"]
        assert spec.proprio_dim == 14
        assert spec.action_dim == 14
        assert spec.num_phases == 4
        assert spec.hf_repo == "lerobot/aloha_sim_transfer_cube_human"

    def test_xarm_spec(self):
        spec = LEROBOT_SPECS["xarm_lift"]
        assert spec.proprio_dim == 4
        assert spec.action_dim == 3
        assert spec.num_phases == 3

    def test_adapter_construction(self):
        """Test adapter can be constructed (doesn't load data)."""
        adapter = LeRobotAdapter("pusht")
        assert adapter.proprio_dim == 2
        assert adapter.action_dim == 2
        assert adapter.dataset_name == "pusht"
        assert adapter.num_phases == 3

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown LeRobot dataset"):
            LeRobotAdapter("nonexistent")


class TestLMDBAdapter:
    def test_spec_registered(self):
        spec = LMDB_SPECS["pick_place"]
        assert spec.proprio_dim == 22
        assert spec.action_dim == 8
        assert spec.num_phases == 5

    def test_adapter_construction(self, tmp_path):
        lmdb_path = tmp_path / "pick_place.lmdb"
        lmdb_path.write_bytes(b"")
        (tmp_path / "pick_place_index.json").write_text('{"episodes": []}', encoding="utf-8")
        adapter = LMDBAdapter("pick_place", local_path=lmdb_path)
        assert adapter.dataset_name == "pick_place"
        assert adapter.structured_state_dim == 39

    def test_load_episodes_maps_modalities(self, tmp_path, monkeypatch):
        lmdb_path = tmp_path / "pick_place.lmdb"
        lmdb_path.write_bytes(b"")
        (tmp_path / "pick_place_index.json").write_text('{"episodes": []}', encoding="utf-8")

        class FakeReader:
            def __init__(self, lmdb_path: str):
                self.lmdb_path = lmdb_path

            def to_applied_dataset(self, max_episodes: int = 0):
                episode = {
                    "episode_id": "ep_0001",
                    "length": 4,
                    "states": np.ones((4, 22), dtype=np.float32),
                    "actions": np.ones((4, 8), dtype=np.float32) * 2.0,
                    "phase_labels": np.array([0, 1, 2, 3], dtype=np.int64),
                    "ee_pose": np.ones((4, 7), dtype=np.float32),
                    "ee_vel": np.ones((4, 6), dtype=np.float32),
                    "object_pos": np.ones((4, 3), dtype=np.float32),
                    "is_grasped": np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
                    "success": True,
                }
                return {"episodes": [episode]}

            def close_env(self):
                return None

        import src.data.adapters.lmdb_adapter as lmdb_adapter_module

        monkeypatch.setattr(
            lmdb_adapter_module,
            "StandaloneLMDBReader",
            FakeReader,
            raising=False,
        )

        adapter = LMDBAdapter("pick_place", local_path=lmdb_path)
        episodes = adapter.load_episodes()
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.proprio_history.shape == (4, 22)
        assert ep.actions.shape == (4, 8)
        assert ep.gt_phase.tolist() == [0, 1, 2, 3]
        assert ep.structured_history.shape == (4, 39)
        assert ep.is_grasped_history.shape == (4, 1)
        assert ep.success is True


# ---------------------------------------------------------------------------
# Registry Tests
# ---------------------------------------------------------------------------

class TestRegistry:
    """Test the dataset registry."""

    def test_list_available(self):
        available = list_available_datasets()
        assert "pusht" in available
        assert "aloha_transfer" in available
        assert "xarm_lift" in available
        assert "pick_place" in available
        assert len(available) == 4

    def test_create_lerobot_adapter(self):
        adapter = create_adapter("pusht")
        assert adapter.dataset_name == "pusht"
        assert adapter.proprio_dim == 2

    def test_create_lmdb_adapter(self, tmp_path):
        lmdb_path = tmp_path / "pick_place.lmdb"
        lmdb_path.write_bytes(b"")
        (tmp_path / "pick_place_index.json").write_text('{"episodes": []}', encoding="utf-8")
        adapter = create_adapter("pick_place", local_path=lmdb_path)
        assert adapter.dataset_name == "pick_place"
        assert adapter.proprio_dim == 22

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            create_adapter("nonexistent_dataset")


class TestLMDBDatasetSpecConfig:
    def test_for_dataset_enables_rich_structured_state(self):
        config = ExperimentConfig(condition=Condition.B_SYNAPSE)
        ds = DatasetSpec(
            name="pick_place",
            source="lmdb",
            proprio_dim=22,
            action_dim=8,
            max_episode_length=300,
            num_phases=5,
            ee_pose_dim=7,
            ee_vel_dim=6,
            object_pos_dim=3,
            grasp_dim=1,
            use_rich_structured_state=True,
        )
        resolved = config.for_dataset(ds)
        assert resolved.data.use_rich_structured_state is True
        assert resolved.structured_state_dim == 39


# ---------------------------------------------------------------------------
# RoboticsDataset Tests (using in-memory episodes)
# ---------------------------------------------------------------------------

class TestRoboticsDataset:
    """Test the unified RoboticsDataset with in-memory episodes."""

    def test_dataset_creation(self):
        episodes = _make_episodes(n=5, T=50, proprio_dim=10, action_dim=4)
        config = ExperimentConfig(
            condition=Condition.A1_RECENT,
        )
        from src.core.normalization import NormalizationStats
        norm = NormalizationStats(
            mean=np.zeros(10, dtype=np.float64),
            std=np.ones(10, dtype=np.float64),
        )
        ds = RoboticsDataset(episodes, config, norm, split="train")
        assert len(ds) > 0
        assert ds.proprio_dim == 10
        assert ds.action_dim == 4

    def test_split_ratios(self):
        episodes = _make_episodes(n=20)
        train, val, test = split_episodes(
            episodes, train_ratio=0.6, val_ratio=0.2, seed=42,
        )
        assert len(train) == 12
        assert len(val) == 4
        assert len(test) == 4

    def test_getitem_returns_correct_keys(self):
        episodes = _make_episodes(n=3, T=30, proprio_dim=4, action_dim=2)
        config = ExperimentConfig(condition=Condition.A1_RECENT)
        from src.core.normalization import NormalizationStats
        norm = NormalizationStats(
            mean=np.zeros(4, dtype=np.float64),
            std=np.ones(4, dtype=np.float64),
        )
        ds = RoboticsDataset(episodes, config, norm, split="train")
        if len(ds) > 0:
            sample = ds[0]
            assert "proprio" in sample
            assert "action_chunk" in sample
            assert "structured_state" in sample
            assert "phase_label" in sample


# ---------------------------------------------------------------------------
# Collation Tests
# ---------------------------------------------------------------------------

class TestCollation:
    """Test the collate function for variable-length batching."""

    def test_collate_variable_history(self):
        batch = []
        for t in [10, 20, 30]:
            batch.append({
                "proprio": torch.randn(5),
                "proprio_history": torch.randn(t, 5),
                "structured_state": torch.randn(5),
                "structured_history": torch.randn(t, 5),
                "action_chunk": torch.randn(8, 3),
                "phase_label": torch.tensor(0),
                "episode_length": torch.tensor(100),
                "timestep": torch.tensor(t),
            })

        collated = collate_fn(batch)
        assert collated["proprio_history"].shape == (3, 30, 5)  # Padded to max
        assert collated["action_chunk"].shape == (3, 8, 3)

    def test_collate_preserves_dtypes(self):
        batch = [{
            "proprio": torch.randn(3),
            "proprio_history": torch.randn(5, 3),
            "structured_state": torch.randn(3),
            "structured_history": torch.randn(5, 3),
            "action_chunk": torch.randn(4, 2),
            "phase_label": torch.tensor(1, dtype=torch.long),
            "episode_length": torch.tensor(50, dtype=torch.long),
            "timestep": torch.tensor(3, dtype=torch.long),
        }]
        collated = collate_fn(batch)
        assert collated["phase_label"].dtype == torch.long
        assert collated["proprio"].dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
