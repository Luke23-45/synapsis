"""
Experiment utilities -- shared infrastructure adapted from GibbsQ.

Modules
-------
config        : Typed dataclass config loaded from YAML.
model_io      : Run capsule management and artifact I/O.
run_artifacts : Standardized per-run path helpers.
logging       : File-based per-run logging setup.
exporter      : JSONL and CSV data export.
progress      : Terminal progress bars (tqdm wrapper).
"""

from experiments.utils.config import (
    load_config, validate, ExperimentConfig, get_experiment_overrides,
)
from experiments.utils.model_io import (
    create_run_capsule, RunCapsule, save_config_snapshot,
    save_metrics, save_run_pointer, save_experiment_artifact,
)
from experiments.utils.run_artifacts import (
    logs_dir, figures_dir, metrics_dir, artifacts_dir,
    metadata_dir, config_path, metrics_path, figure_path,
)
from experiments.utils.exporter import append_metrics_jsonl, save_results_csv
from experiments.utils.progress import create_progress, iter_progress
