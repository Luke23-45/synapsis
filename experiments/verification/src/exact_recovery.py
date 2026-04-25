"""
VZ2-03: Exact Recovery Under Dominance - Verification Experiment.

Z2 Reference: §6 of 02_rigorous_architecture.md, Theorem 6.3
Strictly respects the positive-support constraints for exact algorithmic subset bounding.
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

from experiments.common.report import ExperimentReport, ExperimentTimer
from experiments.utils.config import ExperimentConfig
from experiments.utils.progress import iter_progress
from experiments.verification.utils._shared import (
    iter_parameter_grid,
    resolve_selector_solver,
    spaced_indices,
)
from synapse_core.anchor_selector import hard_projection, solve_relaxed_selector

if TYPE_CHECKING:
    from experiments.verification.utils.data_recorder import VerificationRecorder

log = logging.getLogger(__name__)


def verify_thm_6_3_conditions(y_star: np.ndarray, target_indices: list[int], K: int, tol: float = 1e-6) -> bool:
    """
    Rigorously tests whether a candidate selection precisely meets the updated Z2 Theorem 6.3 logic.
    Evaluates Strict Dominance + Boundary constraints.
    """
    if not target_indices:
        return False
        
    m = len(target_indices)
    C = np.array(target_indices, dtype=int)
    
    # Identify non-target elements.
    mask_out = np.ones(len(y_star), dtype=bool)
    mask_out[C] = False
    mask_out[0] = False # Exclude index 0 (t=1, the starting state never included as an anchor event).

    min_in_C = np.min(y_star[C])
    max_out_C = np.max(y_star[mask_out]) if np.any(mask_out) else 0.0

    # 1. Strict Dominance Constraint
    strict_dominance = min_in_C > (max_out_C + tol)

    # 2. Safety/Boundary Constraint (Target matches Budget exactly OR unbudgeted signals have collapsed entirely)
    # y^* in opt net space might have 1e-8 artifacting above 0. 
    boundary_safe = (m == K) or (max_out_C <= tol)

    return bool(strict_dominance and boundary_safe)


def run_experiment(
    cfg: ExperimentConfig,
    verbose: bool = False,
    recorder: VerificationRecorder | None = None,
) -> ExperimentReport:
    """Run VZ2-03: Exact Recovery Under Dominance Verification."""
    report = ExperimentReport(
        experiment_id="VZ2-03",
        experiment_name="Exact Recovery Under Dominance",
        formal_reference="§6 of 02_rigorous_architecture.md, Theorem 6.3",
        claim="Hard projection uniquely and strictly recovers exactly C under the stated Theorem boundaries.",
    )

    timer = ExperimentTimer()
    with timer:
        # Load configuration ranges
        overrides = cfg.experiments.get("vz2_03_exact_recovery", {})
        dims = overrides.get("dims", [3, 5])
        lengths = overrides.get("lengths", [100, 200])
        K_values = overrides.get("K_values", [3, 5, 10])
        r_values = overrides.get("r_values", [2, 3])
        num_trials = overrides.get("num_trials", 100)
        margin_trials = overrides.get("margin_trials", 50)

        selector = cfg.memory_operator.selector
        lam = selector.lam
        solver = resolve_selector_solver(selector.solver)
        rng = np.random.default_rng(cfg.execution.seed)

        # ---------------------------------------------------------
        # TEST A: Constructive Theorem Check (General Coverage)
        # ---------------------------------------------------------
        if verbose:
            print("  Test A: Evaluating strict algorithmic theorem recovery...")

        eligible_total = 0
        recovery_pass = 0
        recovery_total = 0

        for params in iter_progress(
            iter_parameter_grid(d=dims, T=lengths, K=K_values, r=r_values),
            desc="VZ2-03 Test A",
        ):
            T, K, r = params["T"], params["K"], params["r"]
            m_max = min(K, (T - 1) // (r + 1))
            
            # Use valid m up to budget limit K
            for _ in range(num_trials):
                m = int(rng.integers(1, m_max + 1))
                indices = spaced_indices(T, m, r, rng)
                C_idx = np.array(indices, dtype=int)

                # Heavily partition signal generation
                saliency = rng.uniform(0.0, 0.05, size=T)
                saliency[0] = 0.0
                saliency[C_idx] = rng.uniform(0.85, 1.0, size=m)

                # Mathematical forward pass
                y_star = solve_relaxed_selector(saliency, K, r, lam, solver=solver)
                projected = hard_projection(y_star, K, r)
                
                safe = verify_thm_6_3_conditions(y_star, indices, K)

                if recorder is not None:
                    recorder.log_scalar(A_num_targets_m=float(m), A_budget_K=float(K), A_is_safe=float(safe), A_perfect_recovery=float(set(projected) == set(indices)))
                    recorder.save_trial_data(f"A_K{K}_m{m}_r{r}", saliency=saliency, y_star=y_star, projected=np.array(projected), target_indices=C_idx)

                if safe:
                    eligible_total += 1
                    recovery_total += 1
                    # Ensure deterministic recovery (order agnostic but content identical)
                    if set(projected) == set(indices):
                        recovery_pass += 1

        log.info("Test A (Strict Recovery Analysis): %d/%d passed (eligible: %d)", recovery_pass, recovery_total, eligible_total)

        # ---------------------------------------------------------
        # TEST B: Margin Breakdown and Boundary Sweep
        # ---------------------------------------------------------
        if verbose:
            print("  Test B: Stress-testing structural separation degradation...")

        margin_results = {}
        T_margin, K_margin, r_margin = lengths[0], K_values[0], r_values[0]
        m_margin = min(K_margin, (T_margin - 1) // (r_margin + 1))

        if m_margin >= 1:
            # We decay the isolation. Once theorem breaks, tracking halts incrementing `eligible`.
            for margin in iter_progress([0.05, 0.1, 0.2, 0.4, 0.6, 1.0], desc="VZ2-03 Test B"):
                m_eligible = 0
                m_pass = 0
                
                for _ in range(margin_trials):
                    indices = spaced_indices(T_margin, m_margin, r_margin, rng)
                    C_idx = np.array(indices, dtype=int)

                    # Provide highly volatile structured base logic
                    saliency = np.full(T_margin, 0.4, dtype=np.float64)
                    saliency[0] = 0.0
                    saliency[C_idx] = 0.4 + margin
                    
                    y_star = solve_relaxed_selector(saliency, K_margin, r_margin, lam, solver=solver)
                    
                    safe = verify_thm_6_3_conditions(y_star, indices, K_margin)
                    projected = hard_projection(y_star, K_margin, r_margin)
                    
                    if recorder is not None:
                        recorder.log_scalar(B_margin=margin, B_is_safe=float(safe), B_perfect_recovery=float(set(projected) == set(indices)))
                        if m_eligible == 0:  # Save first eligible or tested case per margin
                            recorder.save_trial_data(f"B_margin_{margin}", saliency=saliency, y_star=y_star, projected=np.array(projected))
                    
                    if safe:
                        m_eligible += 1
                        if set(projected) == set(indices):
                            m_pass += 1

                margin_results[f"margin={margin}"] = {"eligible": m_eligible, "matches": m_pass}

        valid_margins =[v for v in margin_results.values() if v["eligible"] > 0]
        log.info("Test B (Margin Analysis vs Support Set): passed=%s", all(stats["matches"] == stats["eligible"] for stats in valid_margins) if valid_margins else True)

        # ---------------------------------------------------------
        # TEST C: Vulnerability / Edge Constraints
        # ---------------------------------------------------------
        if verbose:
            print("  Test C: Probing algorithmic bounding (m = K vs. Greedy Spillovers)...")

        C_diagnostics = {"C1_passed": False, "C2_passed": False, "C3_prevented": False}

        T_edge, K_edge, r_edge = 100, 5, 2
        
        # Scenario C1: Budget completely utilized (m = K)
        # -----------------------------------------------
        indices_C1 = spaced_indices(T_edge, K_edge, r_edge, rng)
        saliency_C1 = np.zeros(T_edge, dtype=np.float64)
        saliency_C1[np.array(indices_C1, dtype=int)] = 1.0
        
        y_star_C1 = solve_relaxed_selector(saliency_C1, K_edge, r_edge, lam, solver=solver)
        # Even if background noise pushes y^out>0, length is restricted naturally.
        if verify_thm_6_3_conditions(y_star_C1, indices_C1, K_edge):
            proj_C1 = hard_projection(y_star_C1, K_edge, r_edge)
            if set(proj_C1) == set(indices_C1):
                C_diagnostics["C1_passed"] = True

        # Scenario C2: Under-Budget but perfectly isolated (m < K, Noise == 0)
        # --------------------------------------------------------------------
        m_under = K_edge - 2
        indices_C2 = spaced_indices(T_edge, m_under, r_edge, rng)
        saliency_C2 = np.zeros(T_edge, dtype=np.float64)
        saliency_C2[np.array(indices_C2, dtype=int)] = 1.0
        
        y_star_C2 = solve_relaxed_selector(saliency_C2, K_edge, r_edge, lam, solver=solver)
        # Background should perfectly collapse -> Condition 2 passes, exact recovery.
        if verify_thm_6_3_conditions(y_star_C2, indices_C2, K_edge):
            proj_C2 = hard_projection(y_star_C2, K_edge, r_edge)
            if set(proj_C2) == set(indices_C2):
                C_diagnostics["C2_passed"] = True

        # Scenario C3: Strict Dominance HOLDING but condition bounding FAILING.
        # Ensure our verification theorem properly *rejects* mathematically weak geometries
        # where max_not_C > 0 preventing a "greedy spillover false claim" 
        # ---------------------------------------------------------------------------------
        saliency_C3 = rng.uniform(0.1, 0.2, size=T_edge) # Add structural noise preventing zeroes
        saliency_C3[0] = 0.0
        indices_C3 = spaced_indices(T_edge, m_under, r_edge, rng)
        saliency_C3[np.array(indices_C3, dtype=int)] = 1.0
        
        y_star_C3 = solve_relaxed_selector(saliency_C3, K_edge, r_edge, lam, solver=solver)
        is_safe = verify_thm_6_3_conditions(y_star_C3, indices_C3, K_edge)
        
        # It MUST return False, successfully protecting Theorem 6.3 from greedily sweeping garbage.
        if not is_safe:
            C_diagnostics["C3_prevented"] = True
            
        edge_pass = all(C_diagnostics.values())

        if recorder is not None:
            recorder.log_scalar(**{k: float(v) for k, v in C_diagnostics.items()})
            recorder.save_trial_data("C1_budget_exact", saliency=saliency_C1, y_star=y_star_C1)
            recorder.save_trial_data("C2_underbudget", saliency=saliency_C2, y_star=y_star_C2)
            recorder.save_trial_data("C3_spillover", saliency=saliency_C3, y_star=y_star_C3)

        log.info("Test C (Bound Vulnerability Isolation Tests): passed=%s", edge_pass)

    report.duration_seconds = timer.elapsed
    report.status = "PASS" if ((recovery_pass == recovery_total if recovery_total > 0 else True) and (all(stats["matches"] == stats["eligible"] for stats in valid_margins) if valid_margins else True) and edge_pass) else "FAIL"
    return report


if __name__ == "__main__":
    from experiments.verification.utils._shared import run_standalone
    sys.exit(run_standalone(
        caller_file=__file__,
        experiment_id="VZ2-03",
        experiment_name="Exact Recovery Under Dominance",
        run_experiment_fn=run_experiment,
        config_key="vz2_03_exact_recovery",
    ))
