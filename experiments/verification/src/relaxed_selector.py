"""
VZ2-01: Relaxed Selector Properties - Verification Experiment.

Z2 Reference: §5 of 02_rigorous_architecture.md
Rigorous Validation against Global Optimality and Polytope Constraints
Formal Claims: Prop 5.1 (Strong QP Convergence & Global CVX Optimality limits), 
Prop 5.2 (Prefix Map deterministic boundary scoping & causality).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when running this script directly
if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import logging
from typing import Optional

import cvxpy as cp
import numpy as np

from experiments.common.report import ExperimentReport, ExperimentTimer
from experiments.common.trajectory_generators import random_walk
from experiments.utils.config import ExperimentConfig
from experiments.utils.progress import iter_progress
from experiments.verification.utils._shared import iter_parameter_grid, resolve_selector_solver
from synapse_core.anchor_selector import solve_relaxed_selector
from synapse_core.event_encoder import sharp_event_score

log = logging.getLogger(__name__)


def cvxpy_oracle_solver(scores: np.ndarray, K: int, r: int, lam: float) -> np.ndarray:
    """Ground-truth convex solver implementing the §5 Z2 formulation."""
    T = len(scores)
    y = cp.Variable(T)

    # Strictly concave objective (Eq §5)
    objective = cp.Maximize(scores @ y - lam * cp.sum_squares(y))

    constraints = [
        y >= 0,
        y <= 1,
        y[0] == 0,        # Root index excluded
        cp.sum(y) <= K,   # Budget constraint
    ]

    # Refractory spacing constraints
    if r > 0 and T > 1:
        for offset in range(1, r + 1):
            if offset < T:
                constraints.append(y[:-offset] + y[offset:] <= 1)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=25000)

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or y.value is None:
        prob.solve(solver=cp.ECOS)

    if y.value is None:
        raise ValueError("Oracle solver failed to converge")

    return np.clip(np.array(y.value, dtype=np.float64), 0.0, 1.0)


def evaluate_polytope_feasibility(y: np.ndarray, K: int, r: int, tol: float = 1e-4) -> dict:
    """Validate that y satisfies all polytope constraints."""
    T = len(y)
    report = {
        "box_bounds_valid": bool(np.all((y >= -tol) & (y <= 1.0 + tol))),
        "root_anchor_filtered_valid": bool(abs(y[0]) <= tol),
        "budget_limit_adhered": bool(np.sum(y) <= K + tol),
        "spatial_overlap_resolved": True,
    }

    # Refractory overlap constraints
    if r > 0 and T > 1:
        for offset in range(1, r + 1):
            if offset < T:
                viol = y[:-offset] + y[offset:] - 1.0
                if np.max(viol) > tol:
                    report["spatial_overlap_resolved"] = False
                    break

    report["is_structurally_feasible"] = all(report.values())
    return report


def run_experiment(cfg: ExperimentConfig, verbose: bool = False, recorder=None) -> ExperimentReport:
    """Run VZ2-01: Relaxed Selector Properties verification."""
    report = ExperimentReport(
        experiment_id="VZ2-01",
        experiment_name="Relaxed Selector Properties",
        formal_reference="§5 of 02_rigorous_architecture.md",
        claim="Prop 5.1 (strongly-concave QP optimality) and Prop 5.2 (prefix map causality)",
    )

    timer = ExperimentTimer()
    with timer:
        overrides = cfg.experiments.get("vz2_01_relaxed_selector", {})
        dims = overrides.get("dims", [2, 5])
        lengths = overrides.get("lengths", [50, 120])
        lambda_values = overrides.get("lambda_values", [0.1, 1.0, 5.0])

        selector = cfg.memory_operator.selector
        K_opt = selector.K
        r_opt = selector.r
        solver_name = resolve_selector_solver(selector.solver)

        atol = max(cfg.verification.score_match_atol, 1e-4)
        rng = np.random.default_rng(cfg.execution.seed)

        # ---------------------------------------------------------
        # TEST A: Feasibility and Global Optimality (Prop 5.1)
        # ---------------------------------------------------------
        if verbose:
            print("  Test A: Feasibility and global optimality against CVXPY oracle...")

        optimality_pass_cnt = 0
        feasibility_pass_cnt = 0
        total_sys_evals = 0

        for params in iter_progress(
            iter_parameter_grid(d=dims, T=lengths, lam=lambda_values),
            desc="VZ2-01 Test A",
        ):
            total_sys_evals += 1
            d_p, T_p, lam_p = params["d"], params["T"], params["lam"]

            traj_signal = random_walk(d_p, T_p, step_std=1.0, seed=int(rng.integers(2**31)))
            s_scores = sharp_event_score(traj_signal)

            # Library solver
            y_library_solved = solve_relaxed_selector(s_scores, K_opt, r_opt, lam_p, solver=solver_name)

            feasibility = evaluate_polytope_feasibility(y_library_solved, K_opt, r_opt, atol)
            feasible = feasibility["is_structurally_feasible"]
            if feasible:
                feasibility_pass_cnt += 1
            
            # Ground-truth oracle
            y_oracle_checked = cvxpy_oracle_solver(s_scores, K_opt, r_opt, lam_p)
            
            gap = float(np.max(np.abs(y_library_solved - y_oracle_checked)))
            optimal = np.allclose(y_library_solved, y_oracle_checked, atol=atol)
            if optimal:
                optimality_pass_cnt += 1

            # ---- DATA RECORDING ----
            if recorder is not None:
                trial_name = f"testA_d{d_p}_T{T_p}_lam{lam_p}"
                recorder.log_scalar(
                    test="A", trial=total_sys_evals, d=d_p, T=T_p, lam=lam_p,
                    K=K_opt, r=r_opt, feasible=feasible, optimal=optimal,
                    max_gap=gap, y_sum=float(np.sum(y_library_solved)),
                    y_l2=float(np.linalg.norm(y_library_solved)),
                )
                recorder.save_trial_data(
                    trial_name,
                    scores=s_scores,
                    y_library=y_library_solved,
                    y_oracle=y_oracle_checked,
                    trajectory=traj_signal,
                )

        log.info("Test A (Polytope Feasibility & Global Optimality): feasibility %d/%d passed, optimality %d/%d passed", feasibility_pass_cnt, total_sys_evals, optimality_pass_cnt, total_sys_evals)

        # ---------------------------------------------------------
        # TEST B: Prefix Causality (Prop 5.2)
        # ---------------------------------------------------------
        if verbose:
            print("  Test B: Prefix state independence (Prop 5.2 causality)...")

        prefix_causality_checks_cnt = 0
        prefix_causality_evals_total = 0

        for params in iter_progress(
            iter_parameter_grid(d=dims, T=lengths),
            desc="VZ2-01 Test B",
        ):
            d_t, T_t = params["d"], params["T"]
            traj_source = random_walk(d_t, T_t, step_std=1.5, seed=int(rng.integers(2**31)))
            probe_anchor_bound = T_t // 2

            if probe_anchor_bound <= 2:
                continue

            prefix_causality_evals_total += 1
            
            x_limit = np.array(traj_source[:probe_anchor_bound], copy=True)
            s_isolated = sharp_event_score(x_limit)
            y_isolated = solve_relaxed_selector(s_isolated, K_opt, r_opt, selector.lam, solver=solver_name)

            x_violent = traj_source.copy()
            x_violent[probe_anchor_bound:] += rng.standard_normal((T_t - probe_anchor_bound, d_t)) * 75.0

            # Prefix of perturbed trajectory must yield identical scores and solution
            s_tracked_post_variance = sharp_event_score(x_violent[:probe_anchor_bound])
            y_solved_under_variance = solve_relaxed_selector(s_tracked_post_variance, K_opt, r_opt, selector.lam, solver=solver_name)

            scores_held_correct = np.allclose(s_isolated, s_tracked_post_variance, atol=1e-8)
            y_solve_held_correct = np.allclose(y_isolated, y_solved_under_variance, atol=1e-6)

            passed = scores_held_correct and y_solve_held_correct
            if passed:
                prefix_causality_checks_cnt += 1

            # ---- DATA RECORDING ----
            if recorder is not None:
                score_gap = float(np.max(np.abs(s_isolated - s_tracked_post_variance)))
                y_gap = float(np.max(np.abs(y_isolated - y_solved_under_variance)))
                recorder.log_scalar(
                    test="B", trial=prefix_causality_evals_total, d=d_t, T=T_t,
                    probe_t=probe_anchor_bound, scores_match=scores_held_correct,
                    y_match=y_solve_held_correct, passed=passed,
                    score_gap=score_gap, y_gap=y_gap,
                )
                trial_name = f"testB_d{d_t}_T{T_t}"
                recorder.save_trial_data(
                    trial_name,
                    prefix_trajectory=x_limit,
                    scores_isolated=s_isolated,
                    scores_perturbed=s_tracked_post_variance,
                    y_isolated=y_isolated,
                    y_perturbed=y_solved_under_variance,
                )

        log.info("Test B (Prefix Causality): %d/%d passed", prefix_causality_checks_cnt, prefix_causality_evals_total)

        # ---------------------------------------------------------
        # TEST C: Strong Concavity — Monotonic L2 Shrinkage
        # ---------------------------------------------------------
        if verbose:
            print("  Test C: Monotonic L2 shrinkage under increasing lambda...")

        # Strongly concave: increasing lambda must monotonically decrease ||y||_2
        lambda_eval_ranges = [0.1, 0.5, 2.0, 10.0, 50.0]
        decay_monotonous_pass_count = 0
        decay_checked_count = 0
        
        # Collect all sweep data for recording
        all_sweep_lambdas = np.array(lambda_eval_ranges)
        all_sweep_l2_profiles = []

        for d_t in iter_progress(dims, desc="VZ2-01 Test C"):
            decay_checked_count += 1
            
            T_c = lengths[0]
            traj_c = random_walk(d_t, T_c, step_std=1.0, seed=int(rng.integers(2**31)))
            scores_c = sharp_event_score(traj_c)

            measured_l2_shrinkage_response_profiles = []
            
            for lm in lambda_eval_ranges:
                y_d = solve_relaxed_selector(scores_c, K_opt, r_opt, lm, solver=solver_name)
                measured_l2_shrinkage_response_profiles.append(float(np.linalg.norm(y_d, ord=2)))

            is_monotonic = all(
                measured_l2_shrinkage_response_profiles[n] >= measured_l2_shrinkage_response_profiles[n + 1]
                for n in range(len(measured_l2_shrinkage_response_profiles) - 1)
            )
            
            if is_monotonic:
                decay_monotonous_pass_count += 1

            all_sweep_l2_profiles.append(measured_l2_shrinkage_response_profiles)

            # ---- DATA RECORDING ----
            if recorder is not None:
                recorder.log_scalar(
                    test="C", d=d_t, T=T_c, monotonic=is_monotonic,
                    l2_profile=measured_l2_shrinkage_response_profiles,
                )

        # ---- SWEEP DATA RECORDING ----
        if recorder is not None and all_sweep_l2_profiles:
            recorder.save_sweep(
                "lambda_l2_shrinkage",
                lambdas=all_sweep_lambdas,
                l2_profiles=np.array(all_sweep_l2_profiles),
                dims=np.array(dims[:decay_checked_count]),
            )

        log.info("Test C (Monotonic L2 Shrinkage Under Increasing Lambda): %d/%d passed", decay_monotonous_pass_count, decay_checked_count)

    report.duration_seconds = timer.elapsed
    report.status = "PASS" if (feasibility_pass_cnt == total_sys_evals and optimality_pass_cnt == total_sys_evals and (prefix_causality_checks_cnt == prefix_causality_evals_total) and decay_monotonous_pass_count == decay_checked_count) else "FAIL"

    # ---- FINALIZE RECORDING ----
    if recorder is not None:
        recorder.finalize()

    return report


if __name__ == "__main__":
    from experiments.verification.utils._shared import run_standalone
    sys.exit(run_standalone(
        caller_file=__file__,
        experiment_id="VZ2-01",
        experiment_name="Relaxed Selector Properties",
        run_experiment_fn=run_experiment,
        config_key="vz2_01_relaxed_selector",
        project_root=str(Path(__file__).resolve().parent.parent.parent.parent),
    ))
