#!/usr/bin/env python3
"""
SYNAPSE Phase 4 — HuggingFace Dataset Downloader
=================================================

Downloads and caches all LeRobot datasets required for the baselines experiment.
Each dataset is stored under a canonical root:

    baselines/data/datasets/<dataset_name>/
        train.parquet
        info.json
        download_manifest.json

Usage:
    python download_datasets.py                  # Download all datasets
    python download_datasets.py --datasets pusht  # Download specific dataset
    python download_datasets.py --verify          # Verify existing downloads
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

BASELINES_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = BASELINES_ROOT / "data" / "datasets"

# ---------------------------------------------------------------------------
# Dataset Registry — all HuggingFace datasets used in Phase 4
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, dict] = {
    "pusht": {
        "hf_repo": "lerobot/pusht",
        "description": "PushT: 2D non-prehensile pushing (Diffusion Policy benchmark)",
        "expected_episodes": 206,
        "expected_frames": 25650,
        "proprio_dim": 2,
        "action_dim": 2,
        "fps": 10,
        "license": "MIT",
        "citation": "chi2024diffusionpolicy",
    },
    "aloha_transfer": {
        "hf_repo": "lerobot/aloha_sim_transfer_cube_human",
        "description": "ALOHA Transfer Cube: bimanual cube handover (50 episodes)",
        "expected_episodes": 50,
        "expected_frames": 20000,
        "proprio_dim": 14,
        "action_dim": 14,
        "fps": 50,
        "license": "MIT",
        "citation": "Zhao2023LearningFB",
    },
    "xarm_lift": {
        "hf_repo": "lerobot/xarm_lift_medium",
        "description": "xArm Lift Medium: single-arm object lifting",
        "expected_episodes": 100,
        "expected_frames": 12500,
        "proprio_dim": 4,
        "action_dim": 3,
        "fps": 15,
        "license": "MIT",
        "citation": "hansen2022tdmpc",
    },
}


def dataset_dir_for(name: str, root_dir: Path | str | None = None) -> Path:
    return Path(root_dir) if root_dir is not None else DEFAULT_DATASET_ROOT / name


def canonical_dataset_dir(name: str, root_dir: Path | str | None = None) -> Path:
    base = Path(root_dir) if root_dir is not None else DEFAULT_DATASET_ROOT
    return base / name


def canonical_dataset_file(name: str, root_dir: Path | str | None = None) -> Path:
    return canonical_dataset_dir(name, root_dir) / "train.parquet"


def _manifest_path(dataset_dir: Path) -> Path:
    return dataset_dir / "download_manifest.json"


def _info_path(dataset_dir: Path) -> Path:
    return dataset_dir / "info.json"


def resolve_dataset_file(
    name: str,
    dataset_root: Path | str | None = None,
    explicit_path: Path | str | None = None,
) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return canonical_dataset_file(name, dataset_root)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def download_dataset(
    name: str,
    output_dir: Path,
    force: bool = False,
) -> Path:
    """Download a single LeRobot dataset from HuggingFace Hub.

    Parameters
    ----------
    name : str
        Dataset name (key in DATASET_REGISTRY).
    output_dir : Path
        Base output directory (e.g., data/external/).
    force : bool
        If True, re-download even if data already exists.

    Returns
    -------
    Path to the downloaded dataset directory.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")

    meta = DATASET_REGISTRY[name]
    dataset_dir = canonical_dataset_dir(name, output_dir)

    if dataset_dir.exists() and not force:
        # Check if download is complete
        manifest_path = _manifest_path(dataset_dir)
        if manifest_path.exists():
            if verify_dataset(name, output_dir):
                log.info("Dataset '%s' already downloaded at %s (use --force to re-download)", name, dataset_dir)
                return dataset_dir
            log.warning("Existing dataset '%s' failed verification. Re-downloading.", name)
        else:
            log.warning("Dataset directory exists but no manifest found. Re-downloading '%s'.", name)

    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        log.error(
            "The 'datasets' library is required. Install with: pip install datasets"
        )
        sys.exit(1)

    log.info("Downloading '%s' from '%s'...", name, meta["hf_repo"])

    try:
        ds = load_dataset(meta["hf_repo"], split="train")

        # Save to Parquet for efficient loading
        parquet_path = dataset_dir / "train.parquet"
        ds.to_parquet(str(parquet_path))
        log.info("Saved %d rows to %s", len(ds), parquet_path)

        # Save metadata
        info = {
            "hf_repo": meta["hf_repo"],
            "total_episodes": meta["expected_episodes"],
            "total_frames": len(ds),
            "proprio_dim": meta["proprio_dim"],
            "action_dim": meta["action_dim"],
            "fps": meta["fps"],
            "features": {col: str(ds.features[col]) for col in ds.column_names},
            "description": meta["description"],
        }
        info["dataset_dir"] = str(dataset_dir)
        info["parquet_path"] = "train.parquet"
        _write_json(_info_path(dataset_dir), info)

        # Write download manifest
        manifest = {
            "dataset_name": name,
            "hf_repo": meta["hf_repo"],
            "total_rows": len(ds),
            "total_episodes": meta["expected_episodes"],
            "parquet_path": "train.parquet",
            "dataset_dir": str(dataset_dir),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
        }
        _write_json(_manifest_path(dataset_dir), manifest)

        log.info("✅ Dataset '%s' downloaded successfully (%d rows)", name, len(ds))
        return dataset_dir

    except Exception as e:
        log.error("❌ Failed to download '%s': %s", name, e)
        raise


def verify_dataset(name: str, output_dir: Path) -> bool:
    """Verify a downloaded dataset's integrity.

    Returns True if the dataset passes all checks.
    """
    meta = DATASET_REGISTRY[name]
    dataset_dir = canonical_dataset_dir(name, output_dir)

    if not dataset_dir.exists():
        log.error("❌ Dataset directory not found: %s", dataset_dir)
        return False

    manifest_path = _manifest_path(dataset_dir)
    if not manifest_path.exists():
        log.error("❌ Download manifest not found for '%s'", name)
        return False

    manifest = _load_json(manifest_path)

    if manifest.get("status") != "complete":
        log.error("❌ Download incomplete for '%s'", name)
        return False

    parquet_path = dataset_dir / manifest["parquet_path"]
    if not parquet_path.exists():
        log.error("❌ Parquet file missing: %s", parquet_path)
        return False

    try:
        import pyarrow.parquet as pq
        table = pq.read_table(str(parquet_path))
        n_rows = table.num_rows
        log.info("✅ '%s': %d rows in Parquet (expected ~%d frames)", name, n_rows, meta["expected_frames"])

        # Check for required columns
        columns = set(table.column_names)
        for required in ["episode_index", "frame_index"]:
            if required not in columns:
                log.warning("⚠️  Missing expected column '%s' in '%s'", required, name)

        return True

    except Exception as e:
        log.error("❌ Failed to read Parquet for '%s': %s", name, e)
        return False


def ensure_datasets_available(
    dataset_names: Iterable[str],
    output_dir: Path | str | None = None,
    force_download: bool = False,
) -> Dict[str, Path]:
    root = Path(output_dir) if output_dir is not None else DEFAULT_DATASET_ROOT
    resolved: Dict[str, Path] = {}
    for name in dataset_names:
        if name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {name}")
        if not verify_dataset(name, root):
            download_dataset(name, root, force=force_download)
            if not verify_dataset(name, root):
                raise RuntimeError(f"Dataset '{name}' is still invalid after download.")
        resolved[name] = canonical_dataset_file(name, root)
    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets for SYNAPSE Phase 4 baselines"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help=f"Datasets to download. Available: {list(DATASET_REGISTRY.keys())}. Default: all."
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_DATASET_ROOT),
        help="Base directory for downloaded datasets"
    )
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--verify", action="store_true", help="Verify existing downloads only")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    datasets_to_process = args.datasets or list(DATASET_REGISTRY.keys())

    if args.verify:
        log.info("=== Verifying %d datasets ===", len(datasets_to_process))
        all_ok = True
        for name in datasets_to_process:
            if name not in DATASET_REGISTRY:
                log.error("Unknown dataset: %s", name)
                all_ok = False
                continue
            if not verify_dataset(name, output_dir):
                all_ok = False
        sys.exit(0 if all_ok else 1)

    log.info("=== Downloading %d datasets to %s ===", len(datasets_to_process), output_dir)
    for name in datasets_to_process:
        if name not in DATASET_REGISTRY:
            log.error("Unknown dataset: %s — skipping", name)
            continue
        try:
            download_dataset(name, output_dir, force=args.force)
        except Exception:
            log.error("Skipping '%s' due to download failure.", name)

    log.info("=== Download complete ===")


if __name__ == "__main__":
    main()
