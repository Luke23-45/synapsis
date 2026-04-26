from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_BASELINES_ROOT = _PROJECT_ROOT / "baselines"
if str(_BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINES_ROOT))

from baselines.src.core.config import Condition, ExperimentConfig, load_config
from baselines.src.core.normalization import compute_normalization_stats
from baselines.src.data.adapters.base_adapter import RobotEpisode
from baselines.src.data.dataset import create_dataloaders, split_episodes
from baselines.src.engine.train import Trainer


def _make_episode(episode_id: str, horizon: int, state_dim: int, action_dim: int) -> RobotEpisode:
    proprio = torch.randn(horizon, state_dim, dtype=torch.float32).numpy()
    actions = torch.randn(horizon, action_dim, dtype=torch.float32).numpy()
    phases = torch.zeros(horizon, dtype=torch.int64).numpy()
    return RobotEpisode(
        episode_id=episode_id,
        dataset_name="verify",
        proprio_history=proprio,
        actions=actions,
        gt_phase=phases,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify which SYNAPSE pipeline a config activates.")
    parser.add_argument("--config", required=True, help="Path to a baseline experiment config YAML.")
    parser.add_argument("--condition", default="B_synapse", help="Condition to verify.")
    args = parser.parse_args()

    loaded = load_config(args.config)
    condition = Condition(args.condition)
    config = ExperimentConfig(
        condition=condition,
        synapse_implementation=loaded.synapse_implementation,
        seed=loaded.seed,
        synapse=loaded.synapse,
        transformer=loaded.transformer,
        training=loaded.training,
        data=loaded.data,
        stats=loaded.stats,
        output_dir=loaded.output_dir,
        experiment_name=loaded.experiment_name,
        datasets=loaded.datasets,
    )

    episodes = [
        _make_episode(f"verify_{idx}", horizon=12, state_dim=config.data.proprio_dim, action_dim=config.data.action_dim)
        for idx in range(6)
    ]
    train_eps, val_eps, test_eps = split_episodes(episodes, seed=config.seed)
    norm_stats = compute_normalization_stats([ep.structured_history for ep in train_eps])
    train_loader, val_loader, _ = create_dataloaders(
        train_eps,
        val_eps,
        test_eps,
        config,
        norm_stats,
    )
    trainer = Trainer(config, train_loader, val_loader, Path(config.output_dir) / "_verify_pipeline")

    print(f"condition={config.condition.value}")
    print(f"synapse_implementation={config.synapse_implementation.value}")
    print(f"uses_cached_synapse_features={config.uses_cached_synapse_features}")
    print(f"uses_end_to_end_synapse={config.uses_end_to_end_synapse}")
    print(f"model_class={trainer.model.__class__.__name__}")
    print(f"has_forward_train={hasattr(trainer.model, 'forward_train')}")
    print(f"has_forward_deploy={hasattr(trainer.model, 'forward_deploy')}")
    print(f"uses_auxiliary_synapse_losses={trainer.uses_auxiliary_synapse_losses}")

    if config.uses_end_to_end_synapse and not trainer.uses_auxiliary_synapse_losses:
        print("ERROR: end-to-end SYNAPSE was requested but auxiliary losses are inactive.")
        return 1
    if config.uses_cached_synapse_features and trainer.uses_auxiliary_synapse_losses:
        print("ERROR: cached SYNAPSE should not activate end-to-end auxiliary losses.")
        return 1
    print("verification=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
