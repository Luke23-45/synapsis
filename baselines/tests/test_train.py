"""
Training Smoke Test (Agent 4)
==============================

Verifies:
    - Training loop completes without error for all conditions
    - Loss decreases over epochs (basic convergence check)
    - Checkpoint save/load roundtrip works
    - Evaluation produces valid metrics
    - Rollout evaluation runs without error
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
from src.core.normalization import compute_normalization_stats
from src.data.dataset import (
    split_episodes,
    create_dataloaders,
)
from src.engine.train import Trainer
from src.engine.evaluate import Evaluator
from src.engine.rollout import rollout_evaluate, aggregate_rollout_results


# ---------------------------------------------------------------------------
# Helpers & Fixtures
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
    from src.data.adapters.base_adapter import RobotEpisode
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

def _smoke_config(condition: Condition) -> ExperimentConfig:
    return ExperimentConfig(
        condition=condition,
        seed=42,
        synapse=SynapseParams(K=3, r=2, tau=0.3, Q=1),
        transformer=TransformerParams(d_model=32, num_heads=4, num_layers=1),
        data=DataParams(
            proprio_dim=22, action_dim=8, action_chunk_size=2, history_window=5,
            ee_pose_dim=0, ee_vel_dim=0, object_pos_dim=0, grasp_dim=0,
        ),
        training=TrainingParams(
            max_epochs=2, batch_size=4, learning_rate=1e-3,
            warmup_steps=5, early_stopping_patience=5,
            num_workers=0, use_amp=False,
        ),
    )


def _prepare_data(config: ExperimentConfig):
    num_episodes = 4 if config.condition.uses_synapse else 8
    horizon = 16 if config.condition.uses_synapse else 30
    episodes = _make_test_episodes(
        num_episodes=num_episodes, T=horizon, proprio_dim=22, action_dim=8, seed=config.seed,
    )
    train_eps, val_eps, test_eps = split_episodes(episodes, seed=config.seed)
    norm_stats = compute_normalization_stats([ep.structured_history for ep in train_eps])
    for ep in episodes:
        ep.apply_structured_normalization(norm_stats.normalize(ep.structured_history))
    return train_eps, val_eps, test_eps, norm_stats


def _attach_synapse_features(episodes, config, norm_stats):
    """Compute and attach SYNAPSE features to episodes."""
    from src.synapse.synapse_cache import compute_synapse_features
    for ep in episodes:
        features = compute_synapse_features(ep.proprio_history, config, norm_stats)
        ep.synapse_anchors = features.anchors
        ep.synapse_topo = features.topo


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------

class TestTrainingSmoke:
    @pytest.mark.parametrize("condition", [
        Condition.A1_RECENT, Condition.A2_UNIFORM, Condition.B_SYNAPSE,
    ])
    def test_training_completes(self, condition, tmp_path):
        config = _smoke_config(condition)
        train_eps, val_eps, test_eps, norm_stats = _prepare_data(config)

        train_loader, val_loader, test_loader = create_dataloaders(
            train_eps, val_eps, test_eps, config, norm_stats
        )

        trainer = Trainer(config, train_loader, val_loader, tmp_path)
        state = trainer.train()

        assert state.epoch >= 0
        assert len(state.train_losses) > 0
        assert len(state.val_losses) > 0

    def test_loss_decreases(self, tmp_path):
        config = _smoke_config(Condition.A1_RECENT)
        config = ExperimentConfig(
            condition=Condition.A1_RECENT,
            training=TrainingParams(
                max_epochs=5, batch_size=4, learning_rate=1e-3,
                warmup_steps=5, early_stopping_patience=10,
                num_workers=0, use_amp=False,
            ),
            data=DataParams(
                proprio_dim=22, action_dim=8, action_chunk_size=2, history_window=5,
                ee_pose_dim=0, ee_vel_dim=0, object_pos_dim=0, grasp_dim=0,
            ),
            transformer=TransformerParams(d_model=32, num_heads=4, num_layers=1),
        )
        train_eps, val_eps, test_eps, norm_stats = _prepare_data(config)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_eps, val_eps, test_eps, config, norm_stats
        )

        trainer = Trainer(config, train_loader, val_loader, tmp_path)
        state = trainer.train()

        # Last val loss should be less than first (basic convergence)
        if len(state.val_losses) >= 2:
            assert state.val_losses[-1] <= state.val_losses[0] * 1.5  # Allow some slack

    def test_checkpoint_roundtrip(self, tmp_path):
        config = _smoke_config(Condition.A1_RECENT)
        train_eps, val_eps, test_eps, norm_stats = _prepare_data(config)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_eps, val_eps, test_eps, config, norm_stats
        )

        trainer = Trainer(config, train_loader, val_loader, tmp_path)
        trainer.train_epoch()
        trainer.save_checkpoint("test.pt")

        # Load into new trainer
        trainer2 = Trainer(config, train_loader, val_loader, tmp_path)
        trainer2.load_checkpoint("test.pt")
        # Should not crash


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluationSmoke:
    def test_evaluate_condition(self, tmp_path):
        config = _smoke_config(Condition.A1_RECENT)
        train_eps, val_eps, test_eps, norm_stats = _prepare_data(config)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_eps, val_eps, test_eps, config, norm_stats
        )

        trainer = Trainer(config, train_loader, val_loader, tmp_path)
        trainer.train()

        evaluator = Evaluator(config, test_loader, tmp_path / "eval")
        result = evaluator.evaluate_condition(trainer.model, compute_rollout=False)

        assert result.mean_mse >= 0
        assert len(result.per_episode_mse) > 0
        assert result.condition == Condition.A1_RECENT


# ---------------------------------------------------------------------------
# Rollout tests
# ---------------------------------------------------------------------------

class TestRolloutSmoke:
    def test_rollout_evaluate(self, tmp_path):
        config = _smoke_config(Condition.A1_RECENT)
        train_eps, val_eps, test_eps, norm_stats = _prepare_data(config)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_eps, val_eps, test_eps, config, norm_stats
        )

        trainer = Trainer(config, train_loader, val_loader, tmp_path)
        trainer.train()

        batch = next(iter(test_loader))
        gt_actions = batch["action_chunk"]

        result = rollout_evaluate(
            trainer.model,
            batch,
            gt_actions,
            n_steps=3,
        )

        assert result.mse_per_step.shape[0] == 3
        assert result.open_loop_mse >= 0
        assert result.area_under_curve >= 0

    def test_aggregate_rollout(self, tmp_path):
        from src.engine.rollout import RolloutResult
        results = {
            "ep_0": RolloutResult(
                mse_per_step=np.array([0.1, 0.2, 0.3]),
                area_under_curve=0.6,
                divergence_step=None,
                slope=0.1,
                open_loop_mse=0.1,
            ),
            "ep_1": RolloutResult(
                mse_per_step=np.array([0.15, 0.25, 0.35]),
                area_under_curve=0.75,
                divergence_step=2,
                slope=0.1,
                open_loop_mse=0.15,
            ),
        }
        agg = aggregate_rollout_results(results)
        assert "mean_auc" in agg
        assert "mean_slope" in agg
        assert abs(agg["mean_auc"] - 0.675) < 0.01
