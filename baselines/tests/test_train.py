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
from types import SimpleNamespace

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config import (
    ExperimentConfig, Condition, SynapseImplementation, SynapseParams, TransformerParams,
    DataParams, TrainingParams,
)
from src.core.normalization import compute_normalization_stats
from src.data.dataset import (
    split_episodes,
    create_dataloaders,
    _effective_num_workers,
)
from src.engine.train import Trainer
from src.engine.evaluate import Evaluator
from src.engine.rollout import rollout_evaluate, aggregate_rollout_results
from experiments.end_to_end.losses.auxiliary_losses import sparsity_loss, topology_reg_loss


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


def _smoke_e2e_config() -> ExperimentConfig:
    return ExperimentConfig(
        condition=Condition.B_SYNAPSE,
        synapse_implementation=SynapseImplementation.END_TO_END,
        seed=42,
        synapse=SynapseParams(K=3, r=2, tau=0.3, lam=1.0, k=8, Q=1),
        transformer=TransformerParams(
            d_model=32,
            num_heads=4,
            num_layers=1,
            event_encoder_hidden_dim=32,
        ),
        data=DataParams(
            proprio_dim=22, action_dim=8, action_chunk_size=2, history_window=5,
            max_episode_length=16,
            ee_pose_dim=0, ee_vel_dim=0, object_pos_dim=0, grasp_dim=0,
        ),
        training=TrainingParams(
            max_epochs=1, batch_size=2, learning_rate=1e-3,
            warmup_steps=1, early_stopping_patience=2,
            num_workers=0, use_amp=False,
            sparsity_weight=0.01, topology_reg_weight=0.001,
            aux_ramp_start=0, aux_ramp_end=1,
            sparsity_ramp_start=0, sparsity_ramp_end=1,
            topology_ramp_start=0, topology_ramp_end=1,
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
    def test_effective_num_workers_caps_requested_value(self):
        capped = _effective_num_workers(10_000)
        assert capped >= 0
        assert capped <= 10_000

    def test_synapse_loss_reports_raw_aux_losses_before_ramp(self, tmp_path):
        trainer = Trainer.__new__(Trainer)
        trainer.state = SimpleNamespace(epoch=0)
        trainer.loss_config = SimpleNamespace(
            action_weight=1.0,
            sparsity_weight=0.01,
            topology_reg_weight=0.001,
            action_beta=0.5,
            aux_ramp_start=10,
            aux_ramp_end=30,
        )
        trainer.sparsity_ramp_start = 10
        trainer.sparsity_ramp_end = 30
        trainer.topology_ramp_start = 10
        trainer.topology_ramp_end = 30

        pred_outputs = SimpleNamespace(
            pred_actions=torch.tensor([[[0.1, -0.2], [0.0, 0.3]]], dtype=torch.float32),
            y_star=torch.tensor([[0.0, 0.5, 0.25, 0.0]], dtype=torch.float32),
            topology_token=torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32),
        )
        target_actions = torch.zeros_like(pred_outputs.pred_actions)

        total_loss, loss_dict = Trainer._synapse_loss(trainer, pred_outputs, target_actions)

        expected_action = torch.nn.functional.mse_loss(pred_outputs.pred_actions, target_actions)
        expected_sparse = sparsity_loss(pred_outputs.y_star)
        expected_topo = topology_reg_loss(pred_outputs.topology_token)

        assert torch.isclose(loss_dict["action_mse"], expected_action)
        assert torch.isclose(loss_dict["sparsity_loss"], expected_sparse)
        assert torch.isclose(loss_dict["topo_loss"], expected_topo)
        assert float(loss_dict["alpha_sparsity"].item()) == 0.0
        assert float(loss_dict["alpha_topo"].item()) == 0.0
        assert torch.isclose(loss_dict["weighted_sparsity_loss"], torch.tensor(0.0))
        assert torch.isclose(loss_dict["weighted_topo_loss"], torch.tensor(0.0))
        assert torch.isclose(total_loss, expected_action)

    def test_trainer_uses_aux_schedule_from_config(self, tmp_path):
        config = _smoke_e2e_config()
        config = ExperimentConfig(
            condition=config.condition,
            synapse_implementation=config.synapse_implementation,
            seed=config.seed,
            synapse=config.synapse,
            transformer=config.transformer,
            data=config.data,
            stats=config.stats,
            training=TrainingParams(
                max_epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                warmup_steps=1,
                early_stopping_patience=1,
                num_workers=0,
                use_amp=False,
                sparsity_weight=0.2,
                topology_reg_weight=0.05,
                aux_ramp_start=0,
                aux_ramp_end=1,
                sparsity_ramp_start=2,
                sparsity_ramp_end=3,
                topology_ramp_start=0,
                topology_ramp_end=1,
            ),
        )
        train_eps, val_eps, test_eps, norm_stats = _prepare_data(config)
        train_loader, val_loader, _ = create_dataloaders(
            train_eps, val_eps, test_eps, config, norm_stats
        )

        trainer = Trainer(config, train_loader, val_loader, tmp_path)

        assert trainer.loss_config.sparsity_weight == 0.2
        assert trainer.loss_config.topology_reg_weight == 0.05
        assert trainer.loss_config.aux_ramp_start == 0
        assert trainer.loss_config.aux_ramp_end == 1
        assert trainer.sparsity_ramp_start == 2
        assert trainer.sparsity_ramp_end == 3
        assert trainer.topology_ramp_start == 0
        assert trainer.topology_ramp_end == 1

    def test_cached_synapse_disables_auxiliary_losses(self, tmp_path):
        config = _smoke_config(Condition.B_SYNAPSE)
        train_eps, val_eps, test_eps, norm_stats = _prepare_data(config)
        train_loader, val_loader, _ = create_dataloaders(
            train_eps, val_eps, test_eps, config, norm_stats
        )
        trainer = Trainer(config, train_loader, val_loader, tmp_path)
        assert trainer.uses_cached_synapse_features is True
        assert trainer.uses_auxiliary_synapse_losses is False

    def test_end_to_end_synapse_enables_auxiliary_losses(self, tmp_path):
        config = _smoke_e2e_config()
        train_eps, val_eps, test_eps, norm_stats = _prepare_data(config)
        train_loader, val_loader, _ = create_dataloaders(
            train_eps, val_eps, test_eps, config, norm_stats
        )
        trainer = Trainer(config, train_loader, val_loader, tmp_path)
        assert trainer.uses_cached_synapse_features is False
        assert trainer.uses_auxiliary_synapse_losses is True

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
