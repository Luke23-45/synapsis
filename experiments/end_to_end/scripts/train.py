"""
CLI entry point for Z2 end-to-end training using Hydra.

Usage:
  python -m experiments.end_to_end.scripts.train
  python -m experiments.end_to_end.scripts.train training.epochs=50 logging.seed=123
  python -m experiments.end_to_end.scripts.train smoke.enabled=true
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from experiments.end_to_end.runtime import resolve_project_path
from experiments.end_to_end.training.trainer import TrainingConfig, train

log = logging.getLogger("e2e.train")


def _apply_smoke_overrides(cfg: DictConfig) -> None:
    if not cfg.smoke.enabled:
        return

    log.info("Smoke mode enabled.")
    cfg.training.epochs = cfg.smoke.epochs
    cfg.training.batch_size = cfg.smoke.batch_size
    cfg.training.early_stopping_patience = cfg.smoke.early_stopping_patience
    cfg.training.val_check_interval = cfg.smoke.val_check_interval
    cfg.logging.log_interval = cfg.smoke.log_interval
    cfg.data.use_aug = cfg.smoke.use_aug
    cfg.training.max_train_batches = cfg.smoke.max_train_batches
    cfg.training.max_eval_batches = cfg.smoke.max_eval_batches
    cfg.training.max_analysis_batches = cfg.smoke.max_analysis_batches
    if cfg.smoke.reuse_validation_for_training:
        cfg.data.train_lmdb = cfg.data.val_lmdb


def _validate_ablation_contract(cfg: DictConfig) -> None:
    unsupported = []
    if cfg.ablation.encoder_mode != "learned":
        unsupported.append(f"encoder_mode={cfg.ablation.encoder_mode}")
    if cfg.ablation.selector_mode != "relaxed":
        unsupported.append(f"selector_mode={cfg.ablation.selector_mode}")
    if cfg.ablation.lift_mode != "learned":
        unsupported.append(f"lift_mode={cfg.ablation.lift_mode}")
    if not cfg.ablation.normalization:
        unsupported.append("normalization=false")
    if unsupported:
        raise ValueError(
            "The current end-to-end training stack does not implement these ablation modes: "
            + ", ".join(unsupported)
        )


@hydra.main(config_path="../config", config_name="default", version_base="1.2")
def main(cfg: DictConfig) -> None:
    _apply_smoke_overrides(cfg)
    _validate_ablation_contract(cfg)

    config = TrainingConfig(
        input_dim=cfg.model.input_dim,
        action_dim=cfg.model.action_dim,
        action_chunk_size=cfg.model.action_chunk_size,
        hidden_dim=cfg.model.hidden_dim,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        ffn_ratio=cfg.model.ffn_ratio,
        dropout=cfg.model.dropout,
        K=cfg.model.K,
        r=cfg.model.r,
        lam=cfg.model.lam,
        Q=cfg.model.Q,
        k=cfg.model.k,
        batch_size=cfg.training.batch_size,
        epochs=cfg.training.epochs,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas),
        warmup_epochs=cfg.training.warmup_epochs,
        min_lr=cfg.training.min_lr,
        gradient_clip=cfg.training.gradient_clip,
        ema_decay=cfg.training.ema_decay,
        early_stopping_patience=cfg.training.early_stopping_patience,
        val_check_interval=cfg.training.val_check_interval,
        use_amp=cfg.training.get("use_amp", False),
        compile_model=cfg.training.get("compile_model", False),
        num_workers=cfg.training.get("num_workers", 0),
        pin_memory=cfg.training.get("pin_memory", True),
        max_train_batches=cfg.training.get("max_train_batches", None),
        max_eval_batches=cfg.training.get("max_eval_batches", None),
        max_analysis_batches=cfg.training.get("max_analysis_batches", None),
        action_weight=cfg.losses.action_weight,
        sparsity_weight=cfg.losses.sparsity_weight,
        topology_reg_weight=cfg.losses.topology_reg_weight,
        action_beta=cfg.losses.action_beta,
        aux_ramp_start=cfg.losses.aux_ramp_start,
        aux_ramp_end=cfg.losses.aux_ramp_end,
        proprio_noise=cfg.data.proprio_noise,
        use_aug=cfg.data.use_aug,
        min_history=cfg.data.min_history,
        use_anchors=cfg.ablation.use_anchors,
        use_topology=cfg.ablation.use_topology,
        output_dir=str(resolve_project_path(cfg.logging.output_dir)),
        log_interval=cfg.logging.log_interval,
        save_interval=cfg.logging.save_interval,
        manifest_name=cfg.logging.get("manifest_name", "run_manifest.json"),
        resolved_config_name=cfg.logging.get("resolved_config_name", "resolved_config.yaml"),
        seed=cfg.logging.get("seed", 42),
        use_wandb=cfg.logging.get("use_wandb", False),
        wandb_project=cfg.logging.get("wandb_project", "SYNAPSE_E2E"),
        wandb_mode=cfg.logging.get("wandb_mode", "offline"),
        wandb_run_name=cfg.logging.get("wandb_run_name", None),
        analysis_enabled=cfg.analysis.get("enabled", True),
        anchor_heatmap_samples=cfg.analysis.get("anchor_heatmap_samples", 20),
        topology_tsne_max_samples=cfg.analysis.get("topology_tsne_max_samples", 2000),
        topology_tsne_perplexity=cfg.analysis.get("topology_tsne_perplexity", 30),
        topology_tsne_random_seed=cfg.analysis.get("topology_tsne_random_seed", 42),
    )

    results = train(
        config,
        str(resolve_project_path(cfg.data.train_lmdb)),
        str(resolve_project_path(cfg.data.val_lmdb)),
        raw_cfg=cfg,
    )

    log.info("=" * 60)
    log.info("Training complete.")
    if "final_metrics" in results:
        for key, value in sorted(results["final_metrics"].items()):
            log.info("  %s: %.6f", key, value)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
