"""
EMP-04: Topology Branch Value Validation

Demonstrates that the topological summary (persistence diagrams) provides
information that is NOT redundant with the geometric summary of the point
cloud. Cloud + topology features must outperform cloud-only features on
downstream classification.

Spec: docs/implementation/phase2_empirical_validation/04_topology_value.md
Z2 Reference: §10 of 02_rigorous_architecture.md
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
try:
    import pytorch_lightning as pl
except ImportError:
    pl = None

from synapse_core.memory_operator import compute_memory
from experiments.empirical.common.baselines import summarize_diagrams, proxy_topology_features
from experiments.empirical.common.tasks import generate_topology_dataset
from experiments.empirical.common.seed_runner import run_multi_seed
from experiments.empirical.common.emp_config import load_emp_config
from experiments.empirical.common.data_saver import save_experiment_npz
from experiments.empirical.common.math_utils import (
    make_orthogonal_W,
    cloud_geometry_summary,
    ridge_probe_accuracy,
)

log = logging.getLogger(__name__)


# ── Orthogonal W_Theta ─────────────────────────────────────────────────────

# Helpers imported from experiments.empirical.common.math_utils:
#   make_orthogonal_W, cloud_geometry_summary, ridge_probe_accuracy


def _extract_all_features(
    trajectory: np.ndarray,
    K: int, r: int, lam: float, k: int, Q: int,
    W_Theta: np.ndarray,
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract 5 feature sets from Z2 memory operator output:
      cloud_only, topo_only, cloud_topo, proxy_only, cloud_proxy
    """
    traj_f64 = trajectory.astype(np.float64)
    state = compute_memory(
        traj_f64, K=K, r=r, lam=lam,
        W_Theta=W_Theta, Q=Q, solver="osqp",
        mu=mu, sigma=sigma,
    )
    cloud = state.point_cloud
    cloud_feat = cloud_geometry_summary(cloud)
    topo_feat = summarize_diagrams(state.persistence_diagrams)
    # H_0 features (first 4) are redundant with cloud_feat distances.
    # We append only the H_1 features (last 4) to provide strictly non-redundant topology.
    combined_feat = np.concatenate([cloud_feat, topo_feat[4:]])
    proxy_feat = proxy_topology_features(trajectory, k)
    cloud_proxy_feat = np.concatenate([cloud_feat, proxy_feat])

    return {
        "cloud_only": cloud_feat,
        "topo_only": topo_feat,
        "cloud_topo": combined_feat,
        "proxy_only": proxy_feat,
        "cloud_proxy": cloud_proxy_feat,
    }


# ── Ridge Probe ────────────────────────────────────────────────────────────

def _ridge_probe(X_train, y_train, X_test, y_test) -> float:
    """Ridge regression probe — delegates to consolidated implementation."""
    return ridge_probe_accuracy(
        X_train.astype(np.float64), y_train,
        X_test.astype(np.float64), y_test,
    )


# ── Single-Seed Experiment ─────────────────────────────────────────────────

def run_single_seed(config: Any, seed: int) -> Dict[str, float]:
    """Run EMP-04 for a single seed."""
    if pl is not None:
        pl.seed_everything(seed, workers=True)
    cfg = config
    rng = np.random.default_rng(seed)

    K = cfg.memory.K
    r = cfg.memory.r
    lam = cfg.memory.lam
    k = cfg.memory.k
    Q = cfg.memory.Q
    d = cfg.trajectory.d
    T = cfg.trajectory.T
    n_total = cfg.data.n_train + cfg.data.n_val + cfg.data.n_test
    noise_std = cfg.data.noise_std

    all_samples = generate_topology_dataset(rng, n_total, T, d, noise_std)
    D = d + 3
    W_Theta = make_orthogonal_W(k, D, rng)
    
    # Downweight time (col 0) and metadata (cols d+1, d+2) to prevent
    # the monotonically increasing time coordinate from unrolling 
    # the state-space loops into unclosed helices.
    W_Theta[:, 0] *= 0.01
    W_Theta[:, d+1:] *= 0.01
    n = len(all_samples)
    perm = rng.permutation(n)
    n_train = cfg.data.n_train
    n_val = cfg.data.n_val
    train_samples = [all_samples[i] for i in perm[:n_train]]
    test_samples = [all_samples[i] for i in perm[n_train + n_val:]]

    # Compute global normalization statistics from train samples
    all_V = []
    for sample in train_samples:
        state = compute_memory(sample.sequence.astype(np.float64), K, r, lam, W_Theta, Q, solver="osqp")
        all_V.append(state.V)
    if len(all_V) > 0:
        V_concat = np.concatenate(all_V, axis=0)
        global_mu = np.mean(V_concat, axis=0)
        global_sigma = np.ones_like(global_mu)
    else:
        global_mu = None
        global_sigma = None

    feature_sets = ["cloud_only", "topo_only", "cloud_topo", "proxy_only", "cloud_proxy"]
    results: Dict[str, float] = {}

    train_features: Dict[str, List[np.ndarray]] = {fs: [] for fs in feature_sets}
    test_features: Dict[str, List[np.ndarray]] = {fs: [] for fs in feature_sets}

    for sample in train_samples:
        feats = _extract_all_features(sample.sequence, K, r, lam, k, Q, W_Theta, global_mu, global_sigma)
        for fs in feature_sets:
            train_features[fs].append(feats[fs])

    for sample in test_samples:
        feats = _extract_all_features(sample.sequence, K, r, lam, k, Q, W_Theta, global_mu, global_sigma)
        for fs in feature_sets:
            test_features[fs].append(feats[fs])

    y_train = np.array([s.label for s in train_samples])
    y_test = np.array([s.label for s in test_samples])

    for fs in feature_sets:
        X_train = np.stack(train_features[fs]).astype(np.float64)
        X_test = np.stack(test_features[fs]).astype(np.float64)
        results[f"{fs}_accuracy"] = _ridge_probe(X_train, y_train, X_test, y_test)

    return results


# ── Entry Point ────────────────────────────────────────────────────────────

def run_experiment(config: Any = None) -> Dict[str, Any]:
    """Run EMP-04 across all seeds."""
    if config is None:
        config = load_emp_config("EMP-04")

    report = run_multi_seed(
        experiment_fn=run_single_seed,
        config=config,
        seeds=config.training.seeds,
        experiment_id="EMP-04",
        output_dir=config.output_dir,
    )

    agg = report["aggregated"]
    cloud_topo_acc = agg.get("cloud_topo_accuracy", {}).get("mean", 0.0)
    cloud_only_acc = agg.get("cloud_only_accuracy", {}).get("mean", 0.0)
    passed = cloud_topo_acc > cloud_only_acc

    report["experiment_name"] = "Topology Branch Value"
    report["pass_criterion"] = "cloud_topo > cloud_only"
    report["passed"] = passed

    log.info("[EMP-04] cloud_topo=%.4f  cloud_only=%.4f  passed=%s",
             cloud_topo_acc, cloud_only_acc, passed)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if str(Path(__file__).resolve().parent.parent.parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
    result = run_experiment()
    print(f"\nEMP-04 PASSED: {result['passed']}")
