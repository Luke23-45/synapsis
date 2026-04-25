"""
VZ2-05: Topological Stability - Verification Experiment.

Z2 Reference: §12 of 02_rigorous_architecture.md, Theorem 12.3
Rigorously maps foundational point-cloud displacement constraints dynamically asserting formal geometric topology limitation boundaries flawlessly under extreme and applied perturbation fields. 
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

from experiments.common.metrics import bottleneck_distance, hausdorff_distance, max_pointwise_distance
from experiments.common.report import ExperimentReport, ExperimentTimer
from experiments.common.trajectory_generators import random_walk
from experiments.utils.config import ExperimentConfig
from experiments.utils.progress import iter_progress
from experiments.verification.utils._shared import iter_parameter_grid, make_orthogonal_lift, resolve_selector_solver
from synapse_core.memory_operator import compute_memory
from synapse_core.topological_summary import compute_persistence_diagrams

if TYPE_CHECKING:
    from experiments.verification.utils.data_recorder import VerificationRecorder

log = logging.getLogger(__name__)


def _exact_spherical_perturbation(cloud: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """Perturb each point by exactly ε in a random direction."""
    noise = rng.standard_normal(cloud.shape)
    norms = np.linalg.norm(noise, axis=1, keepdims=True)
    unit_directions = noise / np.maximum(norms, 1e-15)
    return cloud + unit_directions * epsilon


def run_experiment(
    cfg: ExperimentConfig,
    verbose: bool = False,
    recorder: VerificationRecorder | None = None,
) -> ExperimentReport:
    """Run VZ2-05: Topological Stability verification."""
    report = ExperimentReport(
        experiment_id="VZ2-05",
        experiment_name="Topological Stability",
        formal_reference="§12 of 02_rigorous_architecture.md, Theorem 12.3",
        claim="d_H <= epsilon implies d_B <= 2*epsilon for persistence diagrams under bounded perturbation",
    )

    timer = ExperimentTimer()
    with timer:
        overrides = cfg.experiments.get("vz2_05_topological_stability", {})
        m_values = overrides.get("m_values", [4, 12, 25])
        D_values = overrides.get("D_values", [3, 6, 10])
        epsilon_values = overrides.get("epsilon_values", [1e-3, 5e-2, 0.5, 2.0])
        num_perturbations = overrides.get("num_perturbations", 30)

        Q_deg = cfg.memory_operator.topology.Q
        rtol = max(cfg.verification.bottleneck_rtol, 1e-5)
        rng = np.random.default_rng(cfg.execution.seed)

        # ---------------------------------------------------------
        # TEST A: Spherical Displacement Stability (Thm 12.3)
        # ---------------------------------------------------------
        if verbose:
            print("  Test A: Hausdorff and Bottleneck bounds under spherical displacement...")

        stability_eval_passed = 0
        stability_eval_total = 0
        extreme_bound_diagnostics = {}

        for params in iter_progress(
            iter_parameter_grid(m=m_values, D=D_values, epsilon=epsilon_values),
            desc="VZ2-05 Test A",
        ):
            m_s, D_s, eps_targ = params["m"], params["D"], params["epsilon"]
            d_H_measured_history = []
            d_B_measured_history =[]

            for _ in range(num_perturbations):
                stability_eval_total += 1
                base_points = rng.standard_normal((m_s, D_s))
                
                # Assure absolute precision structural bounding
                pert_points = _exact_spherical_perturbation(base_points, eps_targ, rng)
                eps_actual = max_pointwise_distance(base_points, pert_points)

                d_H_dist = hausdorff_distance(base_points, pert_points)
                d_H_measured_history.append(d_H_dist / max(eps_actual, 1e-15))

                dgm_base = compute_persistence_diagrams(base_points, Q_deg)
                dgm_pert = compute_persistence_diagrams(pert_points, Q_deg)

                bottleneck_secure = True
                local_b_dist_ratio =[]
                for b_base, p_pert in zip(dgm_base, dgm_pert):
                    d_b_val = bottleneck_distance(b_base, p_pert)
                    local_b_dist_ratio.append(d_b_val / max(eps_actual, 1e-15))
                    if d_b_val > (2.0 * eps_actual + rtol):
                        bottleneck_secure = False

                d_B_measured_history.append(max(local_b_dist_ratio) if local_b_dist_ratio else 0.0)

                # Thm 12.3: d_H <= ε and d_B <= 2ε
                hausdorff_secure = d_H_dist <= (eps_actual + 1e-8)

                if recorder is not None:
                    recorder.log_scalar(A_eps_actual=float(eps_actual), A_d_H_dist=float(d_H_dist), A_d_B_max=float(max(local_b_dist_ratio) * eps_actual if local_b_dist_ratio else 0.0), A_hausdorff_secure=float(hausdorff_secure), A_bottleneck_secure=float(bottleneck_secure))
                    if _ == 0:  # Save representative sample per parameter combo
                        recorder.save_diagrams(f"A_m{m_s}_D{D_s}_eps{eps_targ}", diagrams=dgm_pert, extra={"cloud_base": base_points, "cloud_pert": pert_points, "dgm_base": dgm_base[0] if dgm_base else np.zeros((0,2))})

                if hausdorff_secure and bottleneck_secure:
                    stability_eval_passed += 1

            extreme_bound_diagnostics[f"m={m_s}_eps={eps_targ}"] = {
                "avg_hausdorff_saturation_rate": float(np.mean(d_H_measured_history)),
                "max_bottleneck_saturation_rate": float(np.max(d_B_measured_history))
            }

        log.info("Test A (Topological Stability Evaluation): %d/%d passed", stability_eval_passed, stability_eval_total)

        # ---------------------------------------------------------
        # TEST B: End-to-End Operator Stability Under Trajectory Perturbation
        # ---------------------------------------------------------
        if verbose:
            print("  Test B: End-to-end operator stability under trajectory perturbation...")

        # Trajectory perturbation propagates through the full operator pipeline
        lift_match_passed = 0
        lift_checked_evals = 0
        rejection_from_drift_counts = 0
        
        selector = cfg.memory_operator.selector
        sys_solver = resolve_selector_solver(selector.solver)

        for params in iter_progress(
            iter_parameter_grid(d=[2, 5], k=[4, 8]),
            desc="VZ2-05 Test B",
        ):
            d_p = params["d"]
            k_p = params["k"]
            W_th_map = make_orthogonal_lift(k_p, d_p + 3, rng)

            for _ in range(num_perturbations * 2):
                T_p = 100
                traj_org = random_walk(d_p, T_p, step_std=2.0, seed=int(rng.integers(2**31)))
                
                # Small trajectory perturbation
                drift_noise = rng.standard_normal(traj_org.shape) * 1e-4
                traj_alt = traj_org + drift_noise
                
                mem_org = compute_memory(traj_org, K=8, r=3, lam=1.0, W_Theta=W_th_map, Q=Q_deg, solver=sys_solver)
                mem_alt = compute_memory(traj_alt, K=8, r=3, lam=1.0, W_Theta=W_th_map, Q=Q_deg, solver=sys_solver)

                # Skip cases where perturbation changes anchor selection
                if list(mem_org.anchor_indices) != list(mem_alt.anchor_indices):
                    rejection_from_drift_counts += 1
                    continue
                
                if mem_org.point_cloud.shape[0] < 2:
                    continue  # Trivial cloud, skip
                
                lift_checked_evals += 1
                e_real = max_pointwise_distance(mem_org.point_cloud, mem_alt.point_cloud)
                h_distance = hausdorff_distance(mem_org.point_cloud, mem_alt.point_cloud)
                
                topology_steady = True
                for b1, b2 in zip(mem_org.persistence_diagrams, mem_alt.persistence_diagrams):
                    dist_b = bottleneck_distance(b1, b2)
                    if dist_b > 2.0 * e_real + rtol:
                        topology_steady = False

                if recorder is not None:
                    recorder.log_scalar(B_e_real=float(e_real), B_h_distance=float(h_distance), B_topology_steady=float(topology_steady))
                    if lift_checked_evals == 1: # save a representative sample
                        recorder.save_diagrams(f"B_d{d_p}_k{k_p}_T100", diagrams=mem_alt.persistence_diagrams, extra={"traj_org": traj_org, "traj_alt": traj_alt, "cloud_org": mem_org.point_cloud, "cloud_alt": mem_alt.point_cloud})

                if h_distance <= e_real + 1e-10 and topology_steady:
                    lift_match_passed += 1

        log.info("Test B (Lift Metric Isometry Under Perturbation): %d/%d passed", lift_match_passed, lift_checked_evals)

    report.duration_seconds = timer.elapsed
    report.status = "PASS" if (stability_eval_passed == stability_eval_total and lift_match_passed == lift_checked_evals and lift_checked_evals > 0) else "FAIL"
    return report


if __name__ == "__main__":
    from experiments.verification.utils._shared import run_standalone
    sys.exit(run_standalone(
        caller_file=__file__,
        experiment_id="VZ2-05",
        experiment_name="Topological Stability",
        run_experiment_fn=run_experiment,
        config_key="vz2_05_topological_stability",
    ))
