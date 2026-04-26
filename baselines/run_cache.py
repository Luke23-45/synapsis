#!/usr/bin/env python3
"""
SYNAPSE Offline Feature Caching
===============================

Pre-computes and caches the outputs of the non-parametric memory operator M()
for all episodes in a dataset. This avoids re-computing pairwise distance
matrices and topological persistence diagrams during the inner training loop.

Usage:
    python run_cache.py --dataset pusht --config configs/experiment/smoke.yaml
    python run_cache.py --dataset aloha_transfer --config configs/experiment/full.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config import load_config
from src.core.normalization import compute_normalization_stats_from_episodes
from src.data.dataset import split_episodes
from src.data.registry import create_adapter
from src.synapse.synapse_cache import cache_synapse_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Cache SYNAPSE features offline.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. pusht, xarm_lift)")
    parser.add_argument("--output_dir", type=str, default="data/cache", help="Where to save cache")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Find dataset spec
    ds_spec = next((ds for ds in config.datasets if ds.name == args.dataset), None)
    if ds_spec is None:
        log.error(f"Dataset '{args.dataset}' not found in config datasets list.")
        sys.exit(1)
        
    ds_config = config.for_dataset(ds_spec)

    log.info(f"Loading dataset: {args.dataset}")
    local_path = Path(ds_spec.local_path) if ds_spec.local_path else None
    adapter = create_adapter(
        ds_spec.name,
        local_path=local_path,
        max_episodes=ds_spec.max_episodes,
    )
    episodes = adapter.load_episodes()

    if not episodes:
        log.error("No episodes loaded.")
        sys.exit(1)

    train_eps, val_eps, test_eps = split_episodes(
        episodes,
        train_ratio=ds_config.data.train_ratio,
        val_ratio=ds_config.data.val_ratio,
        seed=ds_config.seed,
    )

    # Compute normalization from train
    log.info("Computing normalization stats from train split...")
    norm_stats = compute_normalization_stats_from_episodes(
        [{"proprio_history": ep.structured_history} for ep in train_eps]
    )

    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_stats.save(out_dir / "normalization_stats.pt")

    for split_name, eps in [("train", train_eps), ("val", val_eps), ("test", test_eps)]:
        if not eps:
            continue
        log.info(f"Caching {split_name} split ({len(eps)} episodes)...")
        cache_path = out_dir / f"{split_name}.pt"
        
        episode_dicts = [
            {"episode_id": ep.episode_id, "proprio_history": ep.proprio_history}
            for ep in eps
        ]
        
        cache_synapse_features(episode_dicts, ds_config, norm_stats, cache_path)
        log.info(f"Saved {split_name} cache to {cache_path}")


if __name__ == "__main__":
    main()
