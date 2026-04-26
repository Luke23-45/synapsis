"""
Smoke test for the end-to-end data pipeline (Step 1).

Validates:
  1. LMDB dataset loads via existing infrastructure
  2. Episode-level splitting works correctly
  3. Normalization statistics are computed
  4. TrajectoryDataset produces samples with correct shapes
  5. Collation handles variable-length histories
  6. Batches are compatible with SynapseEndToEndModel.forward_train
"""

from __future__ import annotations

import sys
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("smoke_test_data")


def main() -> int:
    log.info("=" * 60)
    log.info("Phase 3 Data Pipeline Smoke Test")
    log.info("=" * 60)

    # ----------------------------------------------------------------
    # 1. Load dataset via existing LMDB infrastructure
    # ----------------------------------------------------------------
    log.info("\n[1/6] Loading LMDB dataset...")
    from experiments.empirical.common.data import load_applied_dataset_from_lmdb

    lmdb_path = (
        "outputs/dataset/data/val/"
        "expert_expert_pick_place_val_50_episodes/"
        "expert_expert_pick_place_val_50_episodes.lmdb"
    )
    dataset = load_applied_dataset_from_lmdb(lmdb_path)
    log.info("  Loaded %d episodes", dataset.num_episodes)
    log.info("  Summary: %s", dataset.summary())

    # Convert RobotEpisode objects to dicts for TrajectoryDataset
    episode_dicts = []
    for ep in dataset.episodes:
        episode_dicts.append({
            "states": ep.proprio,          # (T, 22) float32
            "actions": ep.actions,         # (T, 8) float32
            "phase_labels": ep.gt_phase,   # (T,) int64
        })
    log.info("  Converted %d episodes to dicts", len(episode_dicts))
    log.info("  ✅ Dataset loading PASSED")

    # ----------------------------------------------------------------
    # 2. Test episode-level splitting (REMOVED)
    # ----------------------------------------------------------------
    # As of the latest architecture update, train and val datasets are 
    # strictly separated into different LMDB files prior to training,
    # so in-memory dataset splitting is no longer required.

    # ----------------------------------------------------------------
    # 3. Test normalization statistics
    # ----------------------------------------------------------------
    log.info("\n[3/6] Computing normalization statistics...")
    from experiments.end_to_end.data.normalization import compute_normalization_stats

    train_episodes = episode_dicts
    norm_stats = compute_normalization_stats(train_episodes)

    log.info("  state_mean shape: %s", norm_stats.state_mean.shape)
    log.info("  state_std shape:  %s", norm_stats.state_std.shape)
    log.info("  anchor_mu shape:  %s", norm_stats.anchor_mu.shape)
    log.info("  anchor_sigma shape: %s", norm_stats.anchor_sigma.shape)
    log.info("  state_mean[:5]: %s", norm_stats.state_mean[:5])
    log.info("  state_std[:5]:  %s", norm_stats.state_std[:5])

    assert norm_stats.state_mean.shape == (22,), f"Expected (22,), got {norm_stats.state_mean.shape}"
    assert norm_stats.anchor_mu.shape == (25,), f"Expected (25,), got {norm_stats.anchor_mu.shape}"
    assert np.all(norm_stats.state_std > 0), "state_std must be positive"
    assert np.all(norm_stats.anchor_sigma > 0), "anchor_sigma must be positive"
    log.info("  ✅ Normalization PASSED")

    # ----------------------------------------------------------------
    # 4. Test TrajectoryDataset
    # ----------------------------------------------------------------
    log.info("\n[4/6] Testing TrajectoryDataset...")
    from experiments.end_to_end.data.trajectory_dataset import TrajectoryDataset

    action_chunk_size = 10
    train_ds = TrajectoryDataset(
        episodes=train_episodes,
        action_chunk_size=action_chunk_size,
        norm_stats=norm_stats,
    )
    log.info("  Dataset: %d samples from %d episodes", len(train_ds), train_ds.num_episodes)
    log.info("  state_dim=%d, action_dim=%d", train_ds.state_dim, train_ds.action_dim)

    # Check a single sample
    sample = train_ds[0]
    log.info("  Sample keys: %s", list(sample.keys()))
    log.info("  structured_history shape: %s", sample["structured_history"].shape)
    log.info("  structured_state shape:   %s", sample["structured_state"].shape)
    log.info("  ground_truth_actions shape: %s", sample["ground_truth_actions"].shape)

    assert sample["structured_history"].ndim == 2, "History should be 2D"
    assert sample["structured_history"].shape[1] == 22, "State dim should be 22"
    assert sample["structured_state"].shape == (22,), f"State shape mismatch"
    assert sample["ground_truth_actions"].shape == (action_chunk_size, 8), "Action shape mismatch"
    assert sample["structured_history"].shape[0] >= 2, "History must have ≥2 frames"
    log.info("  ✅ TrajectoryDataset PASSED")

    # ----------------------------------------------------------------
    # 5. Test collation
    # ----------------------------------------------------------------
    log.info("\n[5/6] Testing collation...")
    from experiments.end_to_end.data.collate import trajectory_collate_fn

    loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=trajectory_collate_fn,
    )

    batch = next(iter(loader))
    log.info("  Batch keys: %s", list(batch.keys()))
    log.info("  structured_history: %s", batch["structured_history"].shape)
    log.info("  history_mask:       %s", batch["history_mask"].shape)
    log.info("  history_lengths:    %s", batch["history_lengths"].tolist())
    log.info("  structured_state:   %s", batch["structured_state"].shape)
    log.info("  ground_truth_actions: %s", batch["ground_truth_actions"].shape)

    B, T_max, D = batch["structured_history"].shape
    assert B == 4, f"Batch size should be 4, got {B}"
    assert D == 22, f"State dim should be 22, got {D}"
    assert batch["history_mask"].shape == (B, T_max), "Mask shape mismatch"
    assert batch["ground_truth_actions"].shape == (B, action_chunk_size, 8), "Actions shape mismatch"

    # Verify mask correctness
    for i in range(B):
        L = batch["history_lengths"][i].item()
        assert batch["history_mask"][i, :L].all(), f"Real data should be True in mask"
        if L < T_max:
            assert not batch["history_mask"][i, L:].any(), f"Padding should be False in mask"
    log.info("  ✅ Collation PASSED")

    # ----------------------------------------------------------------
    # 6. Test batch compatibility with model forward_train
    # ----------------------------------------------------------------
    log.info("\n[6/6] Testing model forward compatibility...")
    from synapse_arch.model import SynapseArchitectureConfig, SynapseEndToEndModel

    # max_history_tokens must be set to the max sequence length in the dataset,
    # NOT K. During training, the Z2 relaxed path (§4.2 of z2_architectures.md)
    # processes T dense tokens gated by y*. The transformer receives
    # 1 (current) + T (anchor tokens) + 1 (topology) tokens.
    max_T = max(len(ep["states"]) for ep in train_episodes)
    log.info("  Max sequence length in training set: %d", max_T)

    config = SynapseArchitectureConfig(
        input_dim=22,
        action_dim=8,
        action_chunk_size=action_chunk_size,
        hidden_dim=64,
        d_model=128,
        num_heads=4,
        num_layers=3,
        ffn_ratio=4,
        dropout=0.1,
        K=10,
        r=2,
        lam=1.0,
        Q=1,
        k=8,
        max_history_tokens=max_T,
    )
    model = SynapseEndToEndModel(config)

    # Set normalization from computed stats
    model.normalized_lift.set_normalization(
        norm_stats.anchor_mu_tensor(),
        norm_stats.anchor_sigma_tensor(),
    )

    model.eval()
    with torch.no_grad():
        output = model.forward_train(batch)

    log.info("  pred_actions shape:  %s", output.pred_actions.shape)
    log.info("  event_scores shape:  %s", output.event_scores.shape)
    log.info("  y_star shape:        %s", output.y_star.shape)
    log.info("  topology_token shape: %s", output.topology_token.shape)

    assert output.pred_actions.shape == (B, action_chunk_size, 8), "Action prediction shape mismatch"
    assert not torch.isnan(output.pred_actions).any(), "NaN in predictions!"
    assert not torch.isinf(output.pred_actions).any(), "Inf in predictions!"
    log.info("  ✅ Model forward PASSED")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("✅ ALL DATA PIPELINE SMOKE TESTS PASSED")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
