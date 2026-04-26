"""
Robust ablation orchestrator for publication-facing end-to-end runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf

from experiments.end_to_end.runtime import END_TO_END_ROOT, build_run_artifacts, load_json, resolve_project_path, save_json


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("e2e.run_ablations")

CONFIG_PATH = END_TO_END_ROOT / "config" / "default.yaml"


@dataclass
class AblationRunResult:
    name: str
    status: str
    return_code: int
    duration_s: float
    output_root: str
    run_dir: str
    manifest_path: str
    metrics_path: str
    best_checkpoint: str
    val_action_mse: float | None = None
    val_loss: float | None = None
    gap_mse_relative: float | None = None
    topology_phase_separability: float | None = None
    error: str | None = None


def load_script_config() -> Dict[str, Any]:
    cfg = OmegaConf.load(CONFIG_PATH)
    return OmegaConf.to_container(cfg.scripts.ablations, resolve=True)


def _run_training_process(command: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=resolve_project_path("."),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return process.returncode


def _metric_or_none(metrics: Dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, (float, int)) else None


def _load_result_from_manifest(name: str, output_root: Path, seed: int) -> AblationRunResult:
    artifacts = build_run_artifacts(output_root, seed)
    manifest_path = artifacts["manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    metrics = load_json(manifest["final_metrics"])
    return AblationRunResult(
        name=name,
        status="ok",
        return_code=0,
        duration_s=0.0,
        output_root=str(output_root),
        run_dir=str(manifest["output_dir"]),
        manifest_path=str(manifest_path),
        metrics_path=str(manifest["final_metrics"]),
        best_checkpoint=str(manifest["best_checkpoint"]),
        val_action_mse=_metric_or_none(metrics, "val_action_mse"),
        val_loss=_metric_or_none(metrics, "val_loss"),
        gap_mse_relative=_metric_or_none(metrics, "gap_mse_relative"),
        topology_phase_separability=_metric_or_none(metrics, "topology/phase_separability"),
    )


def run_single_ablation(
    name: str,
    output_root: Path,
    seed: int,
    smoke: bool,
    run_name_prefix: str,
) -> AblationRunResult:
    run_output_root = output_root / name
    command = [
        sys.executable,
        "-m",
        "experiments.end_to_end.scripts.train",
        f"logging.output_dir={run_output_root.as_posix()}",
        f"logging.seed={seed}",
        f"logging.wandb_run_name={run_name_prefix}_{name}_seed_{seed}",
    ]
    if smoke:
        command.extend(
            [
                "smoke.enabled=true",
                "losses.aux_ramp_start=0",
                "losses.aux_ramp_end=1",
            ]
        )
    if name != "baseline":
        command.append(f"ablation={name}")

    train_log_path = build_run_artifacts(run_output_root, seed)["run_dir"] / "ablation_train.log"
    started = time.perf_counter()
    return_code = _run_training_process(command, train_log_path)
    duration_s = time.perf_counter() - started

    if return_code != 0:
        return AblationRunResult(
            name=name,
            status="failed",
            return_code=return_code,
            duration_s=duration_s,
            output_root=str(run_output_root),
            run_dir=str(build_run_artifacts(run_output_root, seed)["run_dir"]),
            manifest_path=str(build_run_artifacts(run_output_root, seed)["manifest"]),
            metrics_path=str(build_run_artifacts(run_output_root, seed)["final_metrics"]),
            best_checkpoint=str(build_run_artifacts(run_output_root, seed)["best_checkpoint"]),
            error=f"Training exited with code {return_code}. See {train_log_path}",
        )

    try:
        result = _load_result_from_manifest(name, run_output_root, seed)
        result.duration_s = duration_s
        return result
    except Exception as exc:
        return AblationRunResult(
            name=name,
            status="failed",
            return_code=return_code,
            duration_s=duration_s,
            output_root=str(run_output_root),
            run_dir=str(build_run_artifacts(run_output_root, seed)["run_dir"]),
            manifest_path=str(build_run_artifacts(run_output_root, seed)["manifest"]),
            metrics_path=str(build_run_artifacts(run_output_root, seed)["final_metrics"]),
            best_checkpoint=str(build_run_artifacts(run_output_root, seed)["best_checkpoint"]),
            error=str(exc),
        )


def write_summary_json(results: List[AblationRunResult], path: Path) -> None:
    save_json(path, {"results": [asdict(result) for result in results]})


def write_summary_csv(results: List[AblationRunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_markdown_report(results: List[AblationRunResult], path: Path) -> None:
    baseline = next((result for result in results if result.name == "baseline" and result.status == "ok"), None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Z2 End-to-End Ablation Study\n\n")
        handle.write("| Ablation | Status | Val Loss | Val Action MSE | Gap Rel | Delta MSE vs Baseline | Topology Sep | Duration (s) |\n")
        handle.write("|----------|--------|---------:|---------------:|--------:|----------------------:|-------------:|-------------:|\n")
        for result in results:
            if result.status != "ok":
                handle.write(
                    f"| `{result.name}` | failed | ERROR | ERROR | ERROR | ERROR | ERROR | {result.duration_s:.1f} |\n"
                )
                continue

            baseline_delta = (
                result.val_action_mse - baseline.val_action_mse
                if baseline and baseline.val_action_mse is not None and result.val_action_mse is not None
                else None
            )

            def fmt(value: float | None) -> str:
                return f"{value:.6f}" if value is not None else "N/A"

            handle.write(
                f"| `{result.name}` | ok | {fmt(result.val_loss)} | {fmt(result.val_action_mse)} | "
                f"{fmt(result.gap_mse_relative)} | {fmt(baseline_delta)} | "
                f"{fmt(result.topology_phase_separability)} | {result.duration_s:.1f} |\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run configured end-to-end ablations.")
    parser.add_argument("--smoke", action="store_true", help="Run smoke-mode ablations.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for all runs.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed ablations when manifests already exist.")
    args = parser.parse_args()

    script_cfg = load_script_config()
    ablations = list(script_cfg["configs"])
    output_root = resolve_project_path(script_cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Starting ablation study")
    log.info("Output root: %s", output_root)
    log.info("Seed: %d", args.seed)
    log.info("Ablations: %s", ", ".join(ablations))
    log.info("=" * 70)

    results: List[AblationRunResult] = []
    for name in ablations:
        ablation_output = output_root / name
        manifest_path = build_run_artifacts(ablation_output, args.seed)["manifest"]
        if args.resume and manifest_path.exists():
            log.info("Reusing completed ablation: %s", name)
            result = _load_result_from_manifest(name, ablation_output, args.seed)
            results.append(result)
            continue

        log.info("Running ablation: %s", name)
        result = run_single_ablation(
            name=name,
            output_root=output_root,
            seed=args.seed,
            smoke=args.smoke,
            run_name_prefix=script_cfg["run_name_prefix"],
        )
        if result.status != "ok":
            log.error("Ablation failed: %s | %s", name, result.error)
        else:
            log.info(
                "Completed %s | val_mse=%s | gap=%s",
                name,
                f"{result.val_action_mse:.6f}" if result.val_action_mse is not None else "N/A",
                f"{result.gap_mse_relative:.6f}" if result.gap_mse_relative is not None else "N/A",
            )
        results.append(result)

    summary_json = output_root / "ablation_summary.json"
    summary_csv = output_root / "ablation_summary.csv"
    summary_md = output_root / script_cfg["report_filename_template"].format(seed=args.seed)
    write_summary_json(results, summary_json)
    write_summary_csv(results, summary_csv)
    write_markdown_report(results, summary_md)
    log.info("Wrote ablation summaries to %s", output_root)

    return 0 if all(result.status == "ok" for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
