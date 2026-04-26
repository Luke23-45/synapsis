"""
Publication-ready evaluation entrypoint for end-to-end checkpoints.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict

import hydra
from omegaconf import DictConfig
import torch

from experiments.end_to_end.runtime import resolve_project_path, save_json
from experiments.end_to_end.scripts.common import build_eval_stack, load_checkpoint_bundle
from experiments.end_to_end.training.evaluator import full_evaluation

log = logging.getLogger("e2e.evaluate")


def _build_output_payload(
    checkpoint_bundle: Dict[str, Any],
    metrics: Dict[str, float],
    device: torch.device,
    cfg: DictConfig,
) -> Dict[str, Any]:
    gap_threshold = float(cfg.scripts.evaluate.get("gap_threshold", cfg.scripts.deploy_test.gap_threshold))
    gap = metrics.get("gap_mse_relative", float("nan"))
    status = "passed" if not torch.isnan(torch.tensor(gap)) and gap < gap_threshold else "failed"
    return {
        "status": status,
        "checkpoint_path": str(checkpoint_bundle["checkpoint_path"]),
        "checkpoint_epoch": int(checkpoint_bundle["epoch"]),
        "device": str(device),
        "use_anchors": bool(checkpoint_bundle["use_anchors"]),
        "use_topology": bool(checkpoint_bundle["use_topology"]),
        "train_lmdb": str(resolve_project_path(cfg.data.train_lmdb)),
        "val_lmdb": str(resolve_project_path(cfg.data.val_lmdb)),
        "gap_threshold": gap_threshold,
        "metrics": metrics,
    }


@hydra.main(config_path="../config", config_name="default", version_base="1.2")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint")
    if not checkpoint:
        log.error("Please provide a checkpoint: checkpoint=path/to/best.pt")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_bundle = load_checkpoint_bundle(checkpoint, cfg, device)
    model, val_loader, train_ds, val_ds, loss_config = build_eval_stack(cfg, checkpoint_bundle, device)

    log.info("=" * 70)
    log.info("Evaluating checkpoint %s", checkpoint_bundle["checkpoint_path"])
    log.info("Device: %s", device)
    log.info(
        "Datasets: train=%d samples, val=%d samples",
        len(train_ds),
        len(val_ds),
    )
    log.info("Model params: %d", sum(parameter.numel() for parameter in model.parameters()))
    log.info("=" * 70)

    metrics = full_evaluation(
        model,
        val_loader,
        loss_config,
        checkpoint_bundle["epoch"],
        device,
        amp_enabled=False,
        use_anchors=checkpoint_bundle["use_anchors"],
        use_topology=checkpoint_bundle["use_topology"],
        max_batches=cfg.training.get("max_eval_batches", None),
    )

    for key, value in sorted(metrics.items()):
        log.info("  %-28s %s", key, f"{value:.6f}" if isinstance(value, float) else value)

    payload = _build_output_payload(checkpoint_bundle, metrics, device, cfg)
    log.info("Evaluation status: %s", payload["status"])
    output_path = cfg.get(
        "output",
        str(resolve_project_path(checkpoint_bundle["checkpoint_path"]).parent.parent / cfg.scripts.evaluate.output_filename),
    )
    save_json(output_path, payload)
    log.info("Evaluation report saved to %s", resolve_project_path(output_path))


if __name__ == "__main__":
    main()
