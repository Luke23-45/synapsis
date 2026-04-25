"""
EMP-09: Empirical Metric Structure Validation

Verifies that the learned lift space W_Θ produces distances that empirically
satisfy the triangle inequality and other metric axioms on real point clouds.
VZ2-06 tests this analytically; this experiment validates on trained models.

Z2 Reference: §9 of 02_rigorous_architecture.md (Learned Metric Lift)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn

from synapse_core.event_encoder import sharp_event_score
from synapse_core.anchor_selector import solve_relaxed_selector, hard_projection, build_anchors
from synapse_core.geometric_lift import anchor_vectors, normalize_anchors, apply_lift
from synapse_arch.normalized_lift import NormalizedLift
from experiments.empirical.common.tasks import generate_topology_dataset
from experiments.empirical.common.seed_runner import run_multi_seed
from experiments.empirical.common.emp_config import load_emp_config
from experiments.empirical.common.math_utils import (
    make_orthogonal_W,
    check_triangle_inequality,
    check_symmetry,
    check_non_negativity,
    check_identity,
)
from experiments.empirical.common.data_saver import save_experiment_npz

log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_anchors_numpy(
    trajectory: np.ndarray, K: int, r: int, lam: float,
) -> np.ndarray:
    """Extract anchor vectors via Z2 core operator."""
    traj_f64 = trajectory.astype(np.float64)
    scores = sharp_event_score(traj_f64)
    y_star = solve_relaxed_selector(scores, K, r, lam, solver="osqp")
    indices = hard_projection(y_star, K, r)
    if not indices:
        return np.zeros((0, traj_f64.shape[1] + 3), dtype=np.float64)
    anchors = build_anchors(indices, traj_f64, scores)
    return anchor_vectors(anchors)


# ── Single-Seed Experiment ─────────────────────────────────────────────────

def run_single_seed(config: Any, seed: int) -> Dict[str, float]:
    """Run EMP-09 for a single seed."""
    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed, workers=True)
    except ImportError:
        pass
    rng = np.random.default_rng(seed)
    cfg = config

    K = cfg.memory.K
    r = cfg.memory.r
    lam = cfg.memory.lam
    k = cfg.memory.k
    d = cfg.trajectory.d
    T = cfg.trajectory.T
    noise_std = cfg.data.noise_std
    n_samples = getattr(cfg.data, "n_samples", 100)
    n_triples = getattr(cfg, "n_triples", 5000)
    n_pairs = getattr(cfg, "n_pairs", 2000)

    anchor_dim = d + 3
    lift_dim = k

    results: Dict[str, float] = {}

    # ── Generate point clouds via Z2 operator ─────────────────────────
    all_samples = generate_topology_dataset(rng, n_samples, T, d, noise_std)

    raw_clouds: List[np.ndarray] = []
    for sample in all_samples:
        V = _extract_anchors_numpy(sample.sequence, K, r, lam)
        if len(V) >= 3:
            raw_clouds.append(V.astype(np.float32))

    if not raw_clouds:
        log.warning("[EMP-09] No valid point clouds extracted.")
        return {"error": 1.0}

    # ── Test A: Raw anchor space metric properties ────────────────────
    all_raw_points = np.concatenate(raw_clouds, axis=0)

    tri_raw = check_triangle_inequality(all_raw_points, n_triples, rng)
    sym_raw = check_symmetry(all_raw_points, n_pairs, rng)
    nn_raw = check_non_negativity(all_raw_points)
    id_raw = check_identity(all_raw_points)

    results["n_points"] = float(len(all_raw_points))
    save_experiment_npz("EMP-09", seed, {
        "sequences": np.stack([s.sequence for s in all_samples]),
        "cloud_points": all_raw_points
    }, cfg.output_dir)

    results["A_raw_triangle_violation_rate"] = tri_raw["violation_rate"]
    results["A_raw_triangle_max_violation"] = tri_raw["max_violation"]
    results["A_raw_symmetry_max_asym"] = sym_raw
    results["A_raw_non_negative"] = 1.0 if nn_raw else 0.0
    results["A_raw_identity_max_self_dist"] = id_raw

    # ── Test B: Random lift W_Θ — metric properties ──────────────────
    W_random = make_orthogonal_W(lift_dim, anchor_dim, rng)
    lifted_random_clouds: List[np.ndarray] = []
    for cloud in raw_clouds:
        V_norm, _, _ = normalize_anchors(cloud.astype(np.float64))
        lifted = apply_lift(V_norm, W_random)
        lifted_random_clouds.append(lifted.astype(np.float32))

    all_lifted_random = np.concatenate(lifted_random_clouds, axis=0)
    tri_rand = check_triangle_inequality(all_lifted_random, n_triples, rng)
    sym_rand = check_symmetry(all_lifted_random, n_pairs, rng)

    results["B_random_lift_triangle_violation_rate"] = tri_rand["violation_rate"]
    results["B_random_lift_triangle_max_violation"] = tri_rand["max_violation"]
    results["B_random_lift_symmetry_max_asym"] = sym_rand

    # ── Test C: Trained lift W_Θ — metric properties ─────────────────
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F

    # Prepare training data
    all_vectors = np.zeros((len(all_samples), K, anchor_dim), dtype=np.float32)
    all_labels = np.zeros(len(all_samples), dtype=np.int64)
    for i, sample in enumerate(all_samples):
        V = _extract_anchors_numpy(sample.sequence, K, r, lam)
        m = min(len(V), K)
        if m > 0:
            all_vectors[i, :m, :] = V[:m].astype(np.float32)
        all_labels[i] = sample.label

    # Compute normalization stats
    flat = all_vectors.reshape(-1, anchor_dim)
    nonzero_mask = np.any(flat != 0, axis=1)

    lift_model = NormalizedLift(input_dim=anchor_dim, lift_dim=lift_dim)
    if nonzero_mask.sum() > 1:
        mu_np = flat[nonzero_mask].mean(axis=0)
        sigma_np = flat[nonzero_mask].std(axis=0)
        sigma_np[sigma_np < 1e-6] = 1.0
        lift_model.set_normalization(
            torch.from_numpy(mu_np).float(),
            torch.from_numpy(sigma_np).float(),
        )

    # Quick contrastive training
    optimizer = torch.optim.Adam(lift_model.parameters(), lr=5e-3)
    train_t = torch.from_numpy(all_vectors).float()
    train_l = torch.from_numpy(all_labels).long()
    loader = DataLoader(TensorDataset(train_t, train_l), batch_size=32, shuffle=True)

    lift_model.train()
    epochs = getattr(cfg.training, "metric_epochs", 30)
    for _epoch in range(epochs):
        for batch_v, batch_l in loader:
            _, lifted = lift_model(batch_v)
            centroids = lifted.mean(dim=1)
            centroids = F.normalize(centroids, dim=-1)
            B = centroids.shape[0]
            if B < 2:
                continue
            sim = torch.mm(centroids, centroids.T) / 0.1
            label_eq = (batch_l.unsqueeze(0) == batch_l.unsqueeze(1)).float()
            # Out-of-place mask instead of fill_diagonal_
            diag_mask = (1.0 - torch.eye(B, device=label_eq.device))
            label_eq = label_eq * diag_mask
            pos_count = label_eq.sum(dim=1)
            valid = pos_count > 0
            if not valid.any():
                continue
            exp_sim = torch.exp(sim)
            exp_sim = exp_sim * diag_mask
            pos_sum = (exp_sim * label_eq).sum(dim=1)
            all_sum = exp_sim.sum(dim=1)
            loss = -torch.log(pos_sum / all_sum.clamp_min(1e-8) + 1e-8)
            loss = loss[valid].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Extract lifted clouds from trained model
    lift_model.eval()
    lifted_trained_clouds: List[np.ndarray] = []
    with torch.no_grad():
        for cloud in raw_clouds:
            cloud_t = torch.from_numpy(cloud).float().unsqueeze(0)
            _, lifted = lift_model(cloud_t)
            lifted_trained_clouds.append(lifted.squeeze(0).numpy())

    all_lifted_trained = np.concatenate(lifted_trained_clouds, axis=0)
    tri_trained = check_triangle_inequality(all_lifted_trained, n_triples, rng)
    sym_trained = check_symmetry(all_lifted_trained, n_pairs, rng)
    nn_trained = check_non_negativity(all_lifted_trained)

    results["C_trained_lift_triangle_violation_rate"] = tri_trained["violation_rate"]
    results["C_trained_lift_triangle_max_violation"] = tri_trained["max_violation"]
    results["C_trained_lift_symmetry_max_asym"] = sym_trained
    results["C_trained_lift_non_negative"] = 1.0 if nn_trained else 0.0
    results["C_trained_lift_n_triples"] = float(tri_trained["n_triples_tested"])

    # ── Test D: Distortion ratio ──────────────────────────────────────
    # Measure how much the lift distorts pairwise distances relative to raw space
    if len(all_raw_points) >= 2 and len(all_lifted_trained) >= 2:
        n_dist = min(500, len(all_raw_points), len(all_lifted_trained))
        # Vectorised pairwise distance sampling
        n_sample_pairs = min(1000, n_dist * (n_dist - 1) // 2)
        idx_i = rng.choice(n_dist, size=n_sample_pairs, replace=True)
        idx_j = rng.choice(n_dist, size=n_sample_pairs, replace=True)
        # Filter out self-pairs
        valid_mask = idx_i != idx_j
        idx_i, idx_j = idx_i[valid_mask], idx_j[valid_mask]

        if len(idx_i) > 0:
            raw_dists = np.linalg.norm(
                all_raw_points[idx_i] - all_raw_points[idx_j], axis=1,
            )
            lift_dists = np.linalg.norm(
                all_lifted_trained[idx_i] - all_lifted_trained[idx_j], axis=1,
            )

            # Distortion for non-degenerate pairs
            mask = (raw_dists > 1e-8) & (lift_dists > 1e-8)
            if mask.sum() > 10:
                ratios = lift_dists[mask] / raw_dists[mask]
                results["D_distortion_max_expansion"] = float(np.max(ratios))
                results["D_distortion_max_contraction"] = float(1.0 / np.min(ratios))
                results["D_distortion_mean_ratio"] = float(np.mean(ratios))
                corr = float(np.corrcoef(raw_dists[mask], lift_dists[mask])[0, 1])
                results["D_distance_correlation"] = corr if np.isfinite(corr) else 0.0

    return results


# ── Entry Point ────────────────────────────────────────────────────────────

def run_experiment(config: Any = None) -> Dict[str, Any]:
    """Run EMP-09 across all seeds and return aggregated report."""
    if config is None:
        config = load_emp_config("EMP-09")

    report = run_multi_seed(
        experiment_fn=run_single_seed,
        config=config,
        seeds=config.training.seeds,
        experiment_id="EMP-09",
        output_dir=config.output_dir,
    )

    agg = report["aggregated"]

    # Pass criterion: triangle inequality violation rate < 1e-6 across all spaces
    raw_viol = agg.get("A_raw_triangle_violation_rate", {}).get("mean", 1.0)
    rand_viol = agg.get("B_random_lift_triangle_violation_rate", {}).get("mean", 1.0)
    trained_viol = agg.get("C_trained_lift_triangle_violation_rate", {}).get("mean", 1.0)

    # L2 distance in Euclidean space always satisfies triangle inequality exactly
    # Any violations indicate numerical issues, not mathematical ones
    max_viol = max(raw_viol, rand_viol, trained_viol)
    passed = max_viol < 1e-4  # generous tolerance for floating point

    report["experiment_name"] = "Metric Structure Validation"
    report["pass_criterion"] = f"triangle_violation_rate < 1e-4 (observed max: {max_viol:.2e})"
    report["passed"] = passed

    log.info(
        "[EMP-09] raw_viol=%.2e  rand_viol=%.2e  trained_viol=%.2e  passed=%s",
        raw_viol, rand_viol, trained_viol, passed,
    )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if str(Path(__file__).resolve().parent.parent.parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
    result = run_experiment()
    print(f"\nEMP-09 PASSED: {result['passed']}")
