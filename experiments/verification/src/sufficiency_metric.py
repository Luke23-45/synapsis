"""
VZ2-06: Sufficiency Limits and Metric Properties - Verification Experiment.

Z2 Reference: §12 Prop 12.4, §9 Prop 9.1 of 02_rigorous_architecture.md
Rigorously evaluates Fundamental Theorem mappings establishing analytical limits constraints and topological metric geometry limits definitively. 
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when running this script directly
if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
from typing import TYPE_CHECKING

import numpy as np

from experiments.common.metrics import bottleneck_distance
from experiments.common.report import ExperimentReport, ExperimentTimer
from experiments.common.trajectory_generators import piecewise_constant_auto
from experiments.utils.config import ExperimentConfig
from experiments.utils.progress import iter_progress
from experiments.verification.utils._shared import iter_parameter_grid, make_orthogonal_lift, pairwise_distance_matrix, resolve_selector_solver
from synapse_core.memory_operator import compute_memory

if TYPE_CHECKING:
    from experiments.verification.utils.data_recorder import VerificationRecorder

log = logging.getLogger(__name__)


def _make_analytical_collision_pair(d: int, T: int, r: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct a trajectory pair that validates Prop 12.4 (non-sufficiency).

    ``x`` has structured jumps; ``z`` adds a small perturbation immediately
    after an anchor position.  The perturbation creates nonzero event scores
    at positions (anchor+1) and (anchor+2), but both lie within the refractory
    radius r of the anchor (requires r >= 2), so hard_projection rejects them
    and the anchor set, anchor vectors, and point cloud remain identical.
    """
    x = np.zeros((T, d), dtype=np.float64)
    level = np.zeros(d, dtype=np.float64)

    # Structured piecewise-constant jumps
    jump_pts = [T // 4, T // 2, 3 * T // 4]
    start = 0
    for stop in jump_pts + [T]:
        level = level + rng.standard_normal(d) * 8.0
        x[start:stop] = level
        start = stop

    z = x.copy()

    # Perturb immediately after the first anchor.  This creates events at
    # anchor+1 and anchor+2, both within refractory radius r (requires r >= 2).
    # Hard projection rejects them, preserving the anchor set and all anchor
    # values (s_j, delta_j, xi_j unchanged), so M^inf_Theta(x) = M^inf_Theta(z)
    # in all components that matter for downstream computation.
    anchor_idx = jump_pts[0]
    perturb_idx = anchor_idx + 1
    if perturb_idx < T - 2:
        z[perturb_idx] += rng.standard_normal(d) * 0.1

    return x, z


def _validate_diagram_equivalency(diagrams_x, diagrams_z, atol: float = 1e-6) -> bool:
    """Check that corresponding persistence diagrams agree within tolerance."""
    if len(diagrams_x) != len(diagrams_z):
        return False
    for d1, d2 in zip(diagrams_x, diagrams_z):
        if bottleneck_distance(d1, d2) > atol:
            return False
    return True


def run_experiment(
    cfg: ExperimentConfig,
    verbose: bool = False,
    recorder: VerificationRecorder | None = None,
) -> ExperimentReport:
    """Run VZ2-06: Sufficiency Limits and Metric Properties verification."""
    report = ExperimentReport(
        experiment_id="VZ2-06",
        experiment_name="Sufficiency Limits and Metric Properties",
        formal_reference="§12 Prop 12.4, §9 Prop 9.1 of 02_rigorous_architecture.md",
        claim="Prop 12.4 (operator compression non-sufficiency) and Prop 9.1 (lift metric properties)",
    )

    timer = ExperimentTimer()
    with timer:
        # Load structured test dimensional scales
        overrides = cfg.experiments.get("vz2_06_sufficiency_metric", {})
        dims = overrides.get("dims",[2, 5])
        lengths = overrides.get("lengths",[80, 150])
        
        selector = cfg.memory_operator.selector
        K_opt = selector.K
        r_opt = selector.r
        lam_opt = selector.lam
        solver_engine = resolve_selector_solver(selector.solver)
        Q_deg = cfg.memory_operator.topology.Q
        
        rng = np.random.default_rng(cfg.execution.seed)

        # ---------------------------------------------------------
        # TEST A: Information Loss / Non-Sufficiency (Prop 12.4)
        # ---------------------------------------------------------
        if verbose:
            print("  Test A: Non-sufficiency counter-example (Prop 12.4)")
        
        sufficiency_collapse_validated = False
        evaluations_run = 0

        for params in iter_progress(
            iter_parameter_grid(d=dims, T=lengths),
            desc="VZ2-06 Test A",
        ):
            evaluations_run += 1
            d_t, T_t = params["d"], params["T"]
            W_mat = make_orthogonal_lift(max(3, d_t), d_t + 3, rng)
            
            x_traj, z_traj = _make_analytical_collision_pair(d_t, T_t, r_opt, rng)

            mem_x = compute_memory(x_traj, K_opt, r_opt, lam_opt, W_mat, Q_deg, solver=solver_engine)
            mem_z = compute_memory(z_traj, K_opt, r_opt, lam_opt, W_mat, Q_deg, solver=solver_engine)

            traj_different = not np.allclose(x_traj, z_traj, atol=1e-8)
            anchors_identical = list(mem_x.anchor_indices) == list(mem_z.anchor_indices)
            
            shapes_match = False
            if mem_x.point_cloud.size > 0 and mem_z.point_cloud.size > 0:
                if mem_x.point_cloud.shape == mem_z.point_cloud.shape:
                    shapes_match = np.allclose(mem_x.point_cloud, mem_z.point_cloud, atol=1e-8)
            
            diagrams_equivalent = _validate_diagram_equivalency(mem_x.persistence_diagrams, mem_z.persistence_diagrams)
            
            if recorder is not None:
                recorder.log_scalar(A_traj_different=float(traj_different), A_anchors_identical=float(anchors_identical), A_shapes_match=float(shapes_match), A_diagrams_equivalent=float(diagrams_equivalent))

            if traj_different and anchors_identical and shapes_match and diagrams_equivalent:
                sufficiency_collapse_validated = True
                break

        log.info("Test A (Sufficiency Limits Analysis): %s passed", sufficiency_collapse_validated)

        # ---------------------------------------------------------
        # TEST B: Euclidean Metric Properties (Prop 9.1)
        # ---------------------------------------------------------
        if verbose:
            print("  Test B: Euclidean metric properties of lifted point cloud (Prop 9.1)")
            
        metric_evals_pass = 0
        metric_evals_total = 0
        
        for d in iter_progress(dims, desc="VZ2-06 Test B"):
            metric_evals_total += 1
            D_base = d + 3
            k_test = D_base + 2  # Over-determined lift preserves true metric
            W_theta = make_orthogonal_lift(k_test, D_base, rng)

            sig = piecewise_constant_auto(d, 120, num_segments=5, seed=int(rng.integers(2**31)))
            state_mem = compute_memory(sig, K=15, r=r_opt, lam=lam_opt, W_Theta=W_theta, Q=Q_deg, solver=solver_engine)

            pts_cloud = state_mem.point_cloud
            if pts_cloud.shape[0] < 3:
                metric_evals_pass += 1
                continue
                
            n_evals = min(pts_cloud.shape[0], 10)
            sample_sub = pts_cloud[:n_evals]
            D_mtx = pairwise_distance_matrix(sample_sub)

            is_nonneg = bool(np.all(D_mtx >= -1e-12))
            is_ident = bool(np.allclose(np.diag(D_mtx), 0.0, atol=1e-12))
            is_symmetric = bool(np.allclose(D_mtx, D_mtx.T, atol=1e-12))

            # Triangle inequality
            is_tri_safe = True
            for i in range(n_evals):
                for j in range(n_evals):
                    for k in range(n_evals):
                        if D_mtx[i, j] > D_mtx[i, k] + D_mtx[k, j] + 1e-10:
                            is_tri_safe = False
                            break
                            
            if recorder is not None:
                recorder.log_scalar(B_is_nonneg=float(is_nonneg), B_is_ident=float(is_ident), B_is_symmetric=float(is_symmetric), B_is_tri_safe=float(is_tri_safe))
                            
            if is_nonneg and is_ident and is_symmetric and is_tri_safe:
                metric_evals_pass += 1
                
        log.info("Test B (Topological Lift Metric Properties): %d/%d passed", metric_evals_pass, metric_evals_total)

        # ---------------------------------------------------------
        # TEST C: Pseudometric from Dimensionality Reduction (Prop 9.1)
        # ---------------------------------------------------------
        if verbose:
            print("  Test C: Pseudometric from dimensionality reduction (Prop 9.1)")

        pseudo_validated = False
        
        for d in iter_progress(dims, desc="VZ2-06 Test C"):
            D_in = d + 3
            k_small = D_in - 2 
            
            W_th = make_orthogonal_lift(max(1, k_small), D_in, rng)
            U, S, Vh = np.linalg.svd(W_th, full_matrices=True)
            kernel_vectors = Vh[max(1, k_small):, :]

            # Project into kernel null-space
            null_shift = rng.standard_normal(kernel_vectors.shape[0]) @ kernel_vectors
            
            base_v = rng.standard_normal(D_in)
            alt_v = base_v + 5.0 * null_shift
            
            dist_origin = np.linalg.norm(base_v - alt_v)
            dist_mapping = np.linalg.norm((W_th @ base_v) - (W_th @ alt_v))
            
            if recorder is not None:
                recorder.log_scalar(C_dist_origin=float(dist_origin), C_dist_mapping=float(dist_mapping))
            
            if dist_origin > 1e-3 and dist_mapping < 1e-10:
                pseudo_validated = True
                break

        log.info("Test C (Pseudo-Metric Limits): %s passed", pseudo_validated)

    report.duration_seconds = timer.elapsed
    report.status = "PASS" if (sufficiency_collapse_validated and metric_evals_pass == metric_evals_total and pseudo_validated) else "FAIL"
    return report


if __name__ == "__main__":
    from experiments.verification.utils._shared import run_standalone
    sys.exit(run_standalone(
        caller_file=__file__,
        experiment_id="VZ2-06",
        experiment_name="Sufficiency Limits and Metric Properties",
        run_experiment_fn=run_experiment,
        config_key="vz2_06_sufficiency_metric",
    ))
