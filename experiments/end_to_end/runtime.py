from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
END_TO_END_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def relativize_to_project(path_like: str | Path) -> str:
    path = resolve_project_path(path_like)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_run_dir(output_dir: str | Path, seed: int) -> Path:
    return resolve_project_path(output_dir) / f"seed_{seed}"


def build_run_artifacts(
    output_dir: str | Path,
    seed: int,
    manifest_name: str = "run_manifest.json",
    resolved_config_name: str = "resolved_config.yaml",
) -> Dict[str, Path]:
    run_dir = build_run_dir(output_dir, seed)
    checkpoint_dir = run_dir / "checkpoints"
    return {
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_dir,
        "best_checkpoint": checkpoint_dir / "best.pt",
        "history": run_dir / "training_history.json",
        "final_metrics": run_dir / "final_metrics.json",
        "analysis_dir": run_dir / "analysis",
        "manifest": run_dir / manifest_name,
        "resolved_config": run_dir / resolved_config_name,
    }


def load_json(path_like: str | Path) -> Dict[str, Any]:
    with open(resolve_project_path(path_like), "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path_like: str | Path, payload: Dict[str, Any]) -> None:
    path = resolve_project_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
