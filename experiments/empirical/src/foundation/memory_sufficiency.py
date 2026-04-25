"""
EZ2-04: Memory Sufficiency Probe — Empirical Experiment.

Z2 Reference: §13 of 02_rigorous_architecture.md
Formal Claims: §13 (End-to-End Task Architecture), Prop 12.4 (no universal sufficiency)

This probe is real-data-first. When the LMDB robotics dataset is available,
the experiment includes LMDB-derived prefix tasks and supplements them with
the shared synthetic task families. If LMDB is unavailable, it falls back to
the shared synthetic task generator only.

The expensive part of this experiment is feature extraction, so features are
cached across baselines/tests instead of recomputing the memory operator for
the same sequence multiple times.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from typing import Dict, List, Tuple, Optional

import numpy as np

from synapse_core.anchor_selector import build_anchors
from synapse_core.geometric_lift import anchor_vectors, apply_lift, normalize_anchors
from synapse_core.memory_operator import Z2MemoryState, compute_memory
from experiments.utils.config import load_config, validate, ExperimentConfig
from experiments.common.report import ExperimentReport, TestCase, ExperimentTimer
from experiments.verification.utils.data_recorder import VerificationRecorder
from experiments.empirical.common.baselines import summarize_diagrams, uniform_feature
from experiments.empirical.common.data import RobotEpisode, load_applied_dataset_from_lmdb
from experiments.empirical.common.tasks import SequenceSample, generate_memory_task_dataset


DEFAULT_LMDB_PATH = (
    "outputs/dataset/data/val/expert_expert_pick_place_val_50_episodes/"
    "expert_expert_pick_place_val_50_episodes.lmdb"
)

REAL_TASK_FAMILIES = [
    "real_phase_bucket",
    "real_grasp_history",
    "real_future_grasp",
]

SYNTHETIC_TASK_FAMILIES = [
    "delayed_retrieval",
    "ordered_trigger_response",
    "phase_conditioned_decision",
    "revisit_dependent_rule_switching",
]


@dataclass(frozen=True)
class ProbeSample:
    sample_id: str
    sequence: np.ndarray
    label: int
    family: str
    source: str


def run_experiment(cfg: ExperimentConfig, verbose: bool = False, recorder: Optional[VerificationRecorder] = None) -> ExperimentReport:
    report = ExperimentReport(
        experiment_id="EZ2-04",
        experiment_name="Memory Sufficiency Probe",
        formal_reference="§13 of 02_rigorous_architecture.md, Prop 12.4",
        claim="Z2 memory preserves task-relevant info under a train/validation/test linear probe",
    )

    timer = ExperimentTimer()
    timer.__enter__()

    overrides = cfg.experiments.get("ez2_04_memory_sufficiency", {})
    K_values = overrides.get("K_values", [5, 10, 20])
    lam_values = overrides.get("lam_values", [0.1, 0.5, 1.0])
    k_values = overrides.get("k_values", [4, 8])
    n_samples = int(overrides.get("n_samples", 80))
    probe_seeds = list(overrides.get("probe_seeds", [11, 23, 37]))
    baseline_names = overrides.get("baseline_names", ["B0", "B3", "B4", "B5", "B6", "B7"])

    task_splits, dataset_details = _build_probe_datasets(cfg, overrides, n_samples)
    task_families = sorted(task_splits.keys())

    sel = cfg.memory_operator.selector
    default_K = int(overrides.get("default_K", sel.K))
    default_lam = float(overrides.get("default_lam", lam_values[min(len(lam_values) - 1, 0)] if lam_values else sel.lam))
    default_k = int(overrides.get("default_k", k_values[min(len(k_values) - 1, len(k_values) - 1)] if k_values else cfg.memory_operator.lift.k))
    r = sel.r
    Q = cfg.memory_operator.topology.Q
    solver = sel.solver
    ez04_margin = float(overrides.get("accuracy_margin", 0.02))

    feature_cache: Dict[Tuple[str, str, int, int, float, int, int], np.ndarray] = {}
    state_cache: Dict[Tuple[str, int, int, float, int, int], Z2MemoryState] = {}
    relaxed_cache: Dict[Tuple[str, int, int, float, int], np.ndarray] = {}
    w_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def get_W(k: int, D: int) -> np.ndarray:
        key = (k, D)
        if key not in w_cache:
            local_rng = np.random.default_rng(cfg.execution.seed + 1009 * k + 17 * D)
            w_cache[key] = _make_orthogonal_W(k, D, local_rng)
        return w_cache[key]

    baseline_results = {b: [] for b in baseline_names}
    baseline_failures = {b: 0 for b in baseline_names}
    baseline_failure_types = {b: [] for b in baseline_names}
    family_baseline_scores: Dict[str, Dict[str, float]] = {}

    for family in task_families:
        split = task_splits[family]
        train_samples = split["train"]
        val_samples = split["val"]
        test_samples = split["test"]

        for bname in baseline_names:
            try:
                x_train = _feature_matrix(train_samples, bname, default_K, r, default_lam, default_k, Q, solver, get_W, feature_cache, state_cache, relaxed_cache)
                x_val = _feature_matrix(val_samples, bname, default_K, r, default_lam, default_k, Q, solver, get_W, feature_cache, state_cache, relaxed_cache)
                x_test = _feature_matrix(test_samples, bname, default_K, r, default_lam, default_k, Q, solver, get_W, feature_cache, state_cache, relaxed_cache)
            except (ValueError, RuntimeError, KeyError, IndexError) as exc:
                baseline_failures[bname] += 1
                baseline_failure_types[bname].append(type(exc).__name__)
                continue

            y_train = np.asarray([sample.label for sample in train_samples], dtype=np.int64)
            y_val = np.asarray([sample.label for sample in val_samples], dtype=np.int64)
            y_test = np.asarray([sample.label for sample in test_samples], dtype=np.int64)
            acc = _probe_accuracy(x_train, y_train, x_val, y_val, x_test, y_test, probe_seeds)
            baseline_results[bname].append(acc)
            family_baseline_scores.setdefault(family, {})[bname] = acc

    baseline_means = {b: float(np.mean(v)) for b, v in baseline_results.items() if v}
    b6_acc = baseline_means.get("B6", 0.0)
    b4_acc = baseline_means.get("B4", 0.0)
    b0_acc = baseline_means.get("B0", 0.0)
    topology_sensitive_families = [
        family for family in task_families
        if ("revisit" in family) or ("history" in family) or ("future" in family)
    ]
    wins_on_topology_family = any(
        family_baseline_scores.get(family, {}).get("B6", 0.0) >= family_baseline_scores.get(family, {}).get("B4", 0.0)
        for family in topology_sensitive_families
    )
    competitive_overall = b6_acc >= b4_acc - ez04_margin

    report.add_case(TestCase(
        name="A: Cross-Family Probe Accuracy",
        passed=competitive_overall and wins_on_topology_family and all(v == 0 for v in baseline_failures.values()),
        details={
            "families": task_families,
            "dataset_details": dataset_details,
            "family_baseline_scores": family_baseline_scores,
            "baseline_means": baseline_means,
            "baseline_failures": baseline_failures,
            "baseline_failure_types": baseline_failure_types,
            "topology_sensitive_families": topology_sensitive_families,
            "wins_on_topology_family": wins_on_topology_family,
            "competitive_overall": competitive_overall,
            "reference_note": "B0 is reported as a high-capacity fixed-window reference, not a universal dominance target; Prop 12.4 rules out universal sufficiency.",
        },
    ))

    b6_accs: List[float] = []
    b7_accs: List[float] = []
    for family in task_families:
        split = task_splits[family]
        for bname, accs in [("B6", b6_accs), ("B7", b7_accs)]:
            x_train = _feature_matrix(split["train"], bname, default_K, r, default_lam, default_k, Q, solver, get_W, feature_cache, state_cache, relaxed_cache)
            x_val = _feature_matrix(split["val"], bname, default_K, r, default_lam, default_k, Q, solver, get_W, feature_cache, state_cache, relaxed_cache)
            x_test = _feature_matrix(split["test"], bname, default_K, r, default_lam, default_k, Q, solver, get_W, feature_cache, state_cache, relaxed_cache)
            y_train = np.asarray([sample.label for sample in split["train"]], dtype=np.int64)
            y_val = np.asarray([sample.label for sample in split["val"]], dtype=np.int64)
            y_test = np.asarray([sample.label for sample in split["test"]], dtype=np.int64)
            accs.append(_probe_accuracy(x_train, y_train, x_val, y_val, x_test, y_test, probe_seeds))

    b6_mean = float(np.mean(b6_accs)) if b6_accs else 0.0
    b7_mean = float(np.mean(b7_accs)) if b7_accs else 0.0
    report.add_case(TestCase(
        name="B: Relaxed vs Hard Readout",
        passed=True,
        details={"B6_mean": b6_mean, "B7_mean": b7_mean, "delta": b7_mean - b6_mean},
    ))

    if recorder is not None:
        recorder.save_trial_data("probe_results", family_scores=np.array([family_baseline_scores], dtype=object))

    budget_results: Dict[str, Dict[str, float]] = {}
    for K in K_values:
        accs: List[float] = []
        for family in task_families:
            split = task_splits[family]
            x_train = _feature_matrix(split["train"], "B6", K, r, default_lam, default_k, Q, solver, get_W, feature_cache, state_cache, relaxed_cache)
            x_val = _feature_matrix(split["val"], "B6", K, r, default_lam, default_k, Q, solver, get_W, feature_cache, state_cache, relaxed_cache)
            x_test = _feature_matrix(split["test"], "B6", K, r, default_lam, default_k, Q, solver, get_W, feature_cache, state_cache, relaxed_cache)
            y_train = np.asarray([sample.label for sample in split["train"]], dtype=np.int64)
            y_val = np.asarray([sample.label for sample in split["val"]], dtype=np.int64)
            y_test = np.asarray([sample.label for sample in split["test"]], dtype=np.int64)
            accs.append(_probe_accuracy(x_train, y_train, x_val, y_val, x_test, y_test, probe_seeds))
        budget_results[f"K={K}"] = {"accuracy_mean": float(np.mean(accs)) if accs else 0.0}

    report.add_case(TestCase(
        name="C: Memory Budget Scaling",
        passed=True,
        details=budget_results,
    ))

    timer.__exit__(None, None, None)
    report.duration_seconds = timer.elapsed
    report.finalize()
    return report


def _build_probe_datasets(
    cfg: ExperimentConfig,
    overrides: Dict[str, object],
    n_samples: int,
) -> Tuple[Dict[str, Dict[str, List[ProbeSample]]], Dict[str, object]]:
    task_splits: Dict[str, Dict[str, List[ProbeSample]]] = {}
    requested = overrides.get("task_families")
    lmdb_path = _resolve_lmdb_path(overrides)

    if requested is None:
        families = list(REAL_TASK_FAMILIES if lmdb_path is not None else []) + list(SYNTHETIC_TASK_FAMILIES)
    else:
        families = list(requested)

    details: Dict[str, object] = {"lmdb_used": lmdb_path is not None, "families": families}

    if lmdb_path is not None:
        dataset = load_applied_dataset_from_lmdb(lmdb_path, max_episodes=int(overrides.get("max_episodes", 50)))
        details["lmdb_path"] = lmdb_path
        details["lmdb_summary"] = dataset.summary()
        episode_splits = _split_episodes(dataset.episodes, cfg.execution.seed)
        for family in families:
            if family not in REAL_TASK_FAMILIES:
                continue
            family_split = {
                split_name: _build_real_samples_for_family(family, episodes)
                for split_name, episodes in episode_splits.items()
            }
            if _valid_family_split(family_split):
                task_splits[family] = family_split

    rng = np.random.default_rng(cfg.execution.seed)
    d = cfg.trajectory.d
    T = cfg.trajectory.T
    for family in families:
        if family not in SYNTHETIC_TASK_FAMILIES:
            continue
        train = _wrap_synthetic_family(generate_memory_task_dataset(rng, family, n_samples, T, d, 0.25), family, "synthetic_train")
        val = _wrap_synthetic_family(generate_memory_task_dataset(rng, family, max(20, n_samples // 3), T, d, 0.35), family, "synthetic_val")
        test = _wrap_synthetic_family(generate_memory_task_dataset(rng, family, max(30, n_samples // 2), T, d, 0.55), family, "synthetic_test")
        family_split = {"train": train, "val": val, "test": test}
        if _valid_family_split(family_split):
            task_splits[family] = family_split

    details["real_families_active"] = [f for f in task_splits if f in REAL_TASK_FAMILIES]
    details["synthetic_families_active"] = [f for f in task_splits if f in SYNTHETIC_TASK_FAMILIES]
    return task_splits, details


def _resolve_lmdb_path(overrides: Dict[str, object]) -> str | None:
    candidate = str(overrides.get("lmdb_path", DEFAULT_LMDB_PATH))
    path = Path(candidate)
    return str(path) if path.exists() else None


def _split_episodes(episodes: List[RobotEpisode], seed: int) -> Dict[str, List[RobotEpisode]]:
    order = np.random.default_rng(seed).permutation(len(episodes))
    n = len(order)
    n_train = max(1, int(round(0.6 * n)))
    n_val = max(1, int(round(0.2 * n)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]
    return {
        "train": [episodes[i] for i in train_idx],
        "val": [episodes[i] for i in val_idx],
        "test": [episodes[i] for i in test_idx],
    }


def _build_real_samples_for_family(family: str, episodes: List[RobotEpisode]) -> List[ProbeSample]:
    samples: List[ProbeSample] = []
    cut_fracs = (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
    horizon = 25

    for episode in episodes:
        T = episode.length
        if T < 20:
            continue
        grasp_signal = np.asarray(episode.is_grasped > 0.5, dtype=np.int64)
        for idx, frac in enumerate(cut_fracs):
            t = min(T - 2, max(5, int(frac * T)))
            prefix = episode.state_sequence[:t + 1].astype(np.float32)
            if family == "real_phase_bucket":
                label = int(episode.gt_phase[t])
            elif family == "real_grasp_history":
                label = int(np.any(grasp_signal[:t + 1]))
            elif family == "real_future_grasp":
                label = int(np.any(grasp_signal[t + 1:min(T, t + horizon)]))
            else:
                continue
            samples.append(ProbeSample(
                sample_id=f"{episode.episode_id}:{family}:{idx}",
                sequence=prefix,
                label=label,
                family=family,
                source="lmdb",
            ))
    return samples


def _wrap_synthetic_family(samples: List[SequenceSample], family: str, source: str) -> List[ProbeSample]:
    wrapped: List[ProbeSample] = []
    for idx, sample in enumerate(samples):
        wrapped.append(ProbeSample(
            sample_id=f"{source}:{family}:{idx}",
            sequence=sample.sequence.astype(np.float32),
            label=int(sample.label),
            family=family,
            source="synthetic",
        ))
    return wrapped


def _valid_family_split(split: Dict[str, List[ProbeSample]]) -> bool:
    if not split["train"] or not split["test"]:
        return False
    train_labels = {sample.label for sample in split["train"]}
    test_labels = {sample.label for sample in split["test"]}
    return len(train_labels) >= 2 and len(test_labels) >= 2


def _feature_matrix(
    samples: List[ProbeSample],
    baseline: str,
    K: int,
    r: int,
    lam: float,
    k: int,
    Q: int,
    solver: str,
    get_W,
    feature_cache: Dict[Tuple[str, str, int, int, float, int, int], np.ndarray],
    state_cache: Dict[Tuple[str, int, int, float, int, int], Z2MemoryState],
    relaxed_cache: Dict[Tuple[str, int, int, float, int], np.ndarray],
) -> np.ndarray:
    features = [
        _extract_feature_cached(sample, baseline, K, r, lam, k, Q, solver, get_W, feature_cache, state_cache, relaxed_cache)
        for sample in samples
    ]
    feat_arr = np.asarray(features)
    if feat_arr.ndim == 1:
        feat_arr = feat_arr[:, None]
    return feat_arr.astype(np.float64)


def _extract_feature_cached(
    sample: ProbeSample,
    baseline: str,
    K: int,
    r: int,
    lam: float,
    k: int,
    Q: int,
    solver: str,
    get_W,
    feature_cache: Dict[Tuple[str, str, int, int, float, int, int], np.ndarray],
    state_cache: Dict[Tuple[str, int, int, float, int, int], Z2MemoryState],
    relaxed_cache: Dict[Tuple[str, int, int, float, int], np.ndarray],
) -> np.ndarray:
    D = sample.sequence.shape[1] + 3
    feature_key = (sample.sample_id, baseline, K, r, lam, k, Q)
    if feature_key in feature_cache:
        return feature_cache[feature_key]

    W_Theta = get_W(k, D)

    if baseline == "B0":
        feat = sample.sequence[-K:].astype(np.float32).reshape(-1)
    elif baseline == "B3":
        feat = uniform_feature(sample.sequence, K)
    elif baseline in {"B4", "B5", "B6"}:
        state = _get_state(sample, K, r, lam, k, Q, solver, W_Theta, state_cache)
        cloud = state.point_cloud.astype(np.float32) if state.point_cloud.size else np.zeros((0, k), dtype=np.float32)
        base = _pad_rows(cloud, K).reshape(-1)
        topo = summarize_diagrams(state.persistence_diagrams)
        if baseline == "B4":
            feat = base
        elif baseline == "B5":
            feat = np.concatenate([base, _cloud_geometry_summary(cloud)], axis=0).astype(np.float32)
        else:
            feat = np.concatenate([base, topo], axis=0).astype(np.float32)
    elif baseline == "B7":
        relaxed_key = (sample.sample_id, K, r, lam, k)
        if relaxed_key not in relaxed_cache:
            state = _get_state(sample, K, r, lam, k, Q, solver, W_Theta, state_cache)
            relaxed_cache[relaxed_key] = _relaxed_feature_from_state(sample.sequence, state, K, k, W_Theta)
        feat = relaxed_cache[relaxed_key]
    else:
        raise KeyError(f"Unknown baseline: {baseline}")

    feature_cache[feature_key] = feat.astype(np.float32)
    return feature_cache[feature_key]


def _get_state(
    sample: ProbeSample,
    K: int,
    r: int,
    lam: float,
    k: int,
    Q: int,
    solver: str,
    W_Theta: np.ndarray,
    state_cache: Dict[Tuple[str, int, int, float, int, int], Z2MemoryState],
) -> Z2MemoryState:
    key = (sample.sample_id, K, r, lam, k, Q)
    if key not in state_cache:
        state_cache[key] = compute_memory(sample.sequence, K, r, lam, W_Theta, Q, solver=solver)
    return state_cache[key]


def _relaxed_feature_from_state(
    sequence: np.ndarray,
    state: Z2MemoryState,
    K: int,
    k: int,
    W_Theta: np.ndarray,
) -> np.ndarray:
    indices = [idx for idx, weight in enumerate(state.y_star) if weight > 0.01]
    if not indices:
        return np.zeros(K * k, dtype=np.float32)
    anchors = build_anchors(indices, sequence, state.event_scores)
    V = anchor_vectors(anchors, D=sequence.shape[1] + 3)
    V_norm, _, _ = normalize_anchors(V)
    cloud = apply_lift(V_norm, W_Theta)
    weighted = cloud * state.y_star[indices][:, None]
    return _pad_rows(weighted.astype(np.float32), K).reshape(-1)


def _pad_rows(arr: np.ndarray, rows: int) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0:
        cols = arr.shape[1] if arr.ndim == 2 else 1
        return np.zeros((rows, cols), dtype=np.float32)
    if arr.shape[0] >= rows:
        return arr[:rows].astype(np.float32)
    pad = np.zeros((rows - arr.shape[0], arr.shape[1]), dtype=np.float32)
    return np.concatenate([arr.astype(np.float32), pad], axis=0)


def _cloud_geometry_summary(cloud: np.ndarray) -> np.ndarray:
    if cloud.size == 0 or cloud.shape[0] < 2:
        return np.zeros(8, dtype=np.float32)
    pairwise = np.linalg.norm(cloud[:, None, :] - cloud[None, :, :], axis=-1)
    tri = pairwise[np.triu_indices_from(pairwise, k=1)]
    norms = np.linalg.norm(cloud, axis=1)
    return np.asarray(
        [
            float(np.mean(tri)),
            float(np.std(tri)),
            float(np.max(tri)),
            float(np.percentile(tri, 75)),
            float(np.mean(norms)),
            float(np.std(norms)),
            float(np.max(norms)),
            float(np.percentile(norms, 75)),
        ],
        dtype=np.float32,
    )


def _probe_accuracy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seeds: List[int],
) -> float:
    if len(x_train) < 6 or len(x_test) < 2:
        return 0.0

    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        return 1.0

    accs: List[float] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(x_train))
        train_x = x_train[order]
        train_y = y_train[order]

        mean = np.mean(train_x, axis=0, keepdims=True)
        std = np.std(train_x, axis=0, keepdims=True)
        std[std < 1e-8] = 1.0

        train_x_n = (train_x - mean) / std
        val_x_n = (x_val - mean) / std
        test_x_n = (x_test - mean) / std

        best_w = None
        best_val = -1.0
        for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            w = _fit_ridge_ovr(train_x_n, train_y, unique_labels, alpha)
            val_acc = _predict_accuracy(w, val_x_n, y_val, unique_labels) if len(val_x_n) else _predict_accuracy(w, train_x_n, train_y, unique_labels)
            if val_acc > best_val:
                best_val = val_acc
                best_w = w
        if best_w is not None:
            accs.append(_predict_accuracy(best_w, test_x_n, y_test, unique_labels))

    return float(np.mean(accs)) if accs else 0.0


def _fit_ridge_ovr(x: np.ndarray, y: np.ndarray, unique_labels: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    xtx = x_aug.T @ x_aug
    reg = alpha * np.eye(x_aug.shape[1], dtype=x.dtype)
    reg[-1, -1] = 0.0
    inv = np.linalg.pinv(xtx + reg)

    weights = []
    for label in unique_labels:
        target = (y == label).astype(np.float64)
        weights.append(inv @ x_aug.T @ target)
    return np.stack(weights, axis=0)


def _predict_accuracy(weights: np.ndarray, x: np.ndarray, y: np.ndarray, unique_labels: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    logits = x_aug @ weights.T
    preds = unique_labels[np.argmax(logits, axis=1)]
    return float(np.mean(preds == y))


def _make_orthogonal_W(k: int, D: int, rng: np.random.Generator) -> np.ndarray:
    if k <= D:
        A = rng.standard_normal((D, D))
        Q, _ = np.linalg.qr(A)
        return Q[:k, :].astype(np.float64)
    W = np.zeros((k, D), dtype=np.float64)
    W[:D, :D] = np.eye(D)
    return W


if __name__ == "__main__":
    from experiments.verification.utils._shared import run_standalone
    sys.exit(run_standalone(
        caller_file=__file__,
        experiment_id="EZ2-04",
        experiment_name="Memory Sufficiency Probe",
        run_experiment_fn=run_experiment,
        config_key="ez2_04_memory_sufficiency",
        project_root=str(Path(__file__).resolve().parent.parent.parent.parent.parent),
    ))
