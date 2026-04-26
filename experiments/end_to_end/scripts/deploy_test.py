"""
Deployment-path validation script using Hydra.

Usage:
  python -m experiments.end_to_end.scripts.deploy_test checkpoint=path/to/best.pt
"""

from __future__ import annotations

import logging
import sys

import hydra
from omegaconf import DictConfig

import numpy as np
import torch

from experiments.end_to_end.data.collate import trajectory_collate_fn
from experiments.end_to_end.runtime import resolve_project_path, save_json
from experiments.end_to_end.scripts.common import build_eval_stack, load_checkpoint_bundle

log = logging.getLogger("e2e.deploy_test")


@hydra.main(config_path="../config", config_name="default", version_base="1.2")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint")
    if not checkpoint:
        log.error("Please provide a checkpoint: checkpoint=path/to/best.pt")
        sys.exit(1)

    device = torch.device("cpu")
    log.info("Device: %s", device)

    checkpoint_bundle = load_checkpoint_bundle(checkpoint, cfg, device)
    model, _, _, val_ds, _ = build_eval_stack(cfg, checkpoint_bundle, device)

    num_samples = int(cfg.get("num_samples", cfg.scripts.deploy_test.num_samples))
    gap_threshold = float(cfg.scripts.deploy_test.gap_threshold)
    log.info("Comparing train and deploy paths on %d samples.", num_samples)

    n = min(num_samples, len(val_ds))
    train_mses = []
    deploy_mses = []
    gaps = []
    failures = 0

    for index in range(n):
        sample = val_ds[index]
        if sample is None:
            failures += 1
            continue

        batch = trajectory_collate_fn([sample])
        if not batch:
            failures += 1
            continue

        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        gt = batch["ground_truth_actions"]

        with torch.no_grad():
            train_out = model.forward_train(
                batch,
                use_anchors=checkpoint_bundle["use_anchors"],
                use_topology=checkpoint_bundle["use_topology"],
            )
            train_mse = torch.nn.functional.mse_loss(train_out.pred_actions, gt).item()

            try:
                deploy_out = model.forward_deploy(
                    batch,
                    use_anchors=checkpoint_bundle["use_anchors"],
                    use_topology=checkpoint_bundle["use_topology"],
                )
                deploy_mse = torch.nn.functional.mse_loss(deploy_out.pred_actions, gt).item()
            except Exception as exc:
                log.warning("Sample %d deploy path failed: %s", index, exc)
                failures += 1
                continue

        gap = abs(train_mse - deploy_mse) / max(train_mse, 1e-8)
        train_mses.append(train_mse)
        deploy_mses.append(deploy_mse)
        gaps.append(gap)

        log.info(
            "[%02d] train_mse=%.6f | deploy_mse=%.6f | gap=%.2f%%",
            index,
            train_mse,
            deploy_mse,
            gap * 100,
        )

    if gaps:
        mean_gap = float(np.mean(gaps))
        status = "passed" if mean_gap < gap_threshold else "failed"
        results = {
            "status": status,
            "num_requested_samples": num_samples,
            "num_evaluated_samples": len(gaps),
            "num_failed_samples": failures,
            "mean_train_mse": float(np.mean(train_mses)),
            "mean_deploy_mse": float(np.mean(deploy_mses)),
            "mean_gap": mean_gap,
            "max_gap": float(np.max(gaps)),
            "fraction_below_threshold": float(np.mean(np.asarray(gaps) < gap_threshold)),
            "gap_threshold": gap_threshold,
        }
    else:
        results = {
            "status": "failed",
            "num_requested_samples": num_samples,
            "num_evaluated_samples": 0,
            "num_failed_samples": failures,
            "gap_threshold": gap_threshold,
            "error": "No valid samples were evaluated.",
        }

    log.info("=" * 60)
    log.info("Deploy path validation summary")
    for key, value in results.items():
        log.info("  %s: %s", key, value)
    log.info("=" * 60)

    output_path = cfg.get(
        "output",
        str(
            resolve_project_path(checkpoint_bundle["checkpoint_path"]).parent.parent
            / cfg.scripts.deploy_test.output_filename
        ),
    )
    save_json(output_path, results)
    log.info("Results saved to %s", resolve_project_path(output_path))


if __name__ == "__main__":
    main()
