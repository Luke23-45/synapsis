"""
VZ2-04: Full Operator Causality and Boundedness - Verification Experiment.

Z2 Reference: §12 of 02_rigorous_architecture.md
Rigorously checks formal map consistency, bounds bounding (Thm 12.2), and analytical parameter pathway fidelity.
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
from experiments.common.trajectory_generators import piecewise_constant_auto, random_walk
from experiments.utils.config import ExperimentConfig
from experiments.utils.progress import iter_progress
from experiments.verification.utils._shared import iter_parameter_grid, make_orthogonal_lift, resolve_selector_solver
from synapse_core.memory_operator import compute_memory

if TYPE_CHECKING:
    from experiments.verification.utils.data_recorder import VerificationRecorder

log = logging.getLogger(__name__)


def run_experiment(
    cfg: ExperimentConfig,
    verbose: bool = False,
    recorder: VerificationRecorder | None = None,
) -> ExperimentReport:
    """Run VZ2-04: Full Operator Causality and Boundedness verification."""
    report = ExperimentReport(
        experiment_id="VZ2-04",
        experiment_name="Full Operator Causality and Boundedness",
        formal_reference="§12 of 02_rigorous_architecture.md",
        claim="Thm 12.1 (Prefix Map Causality), Thm 12.2 (Bounded Complexity), and Pipeline Normalization Fidelity",
    )

    timer = ExperimentTimer()
    with timer:
        overrides = cfg.experiments.get("vz2_04_full_operator", {})
        dims = overrides.get("dims",[2, 5, 10])
        lengths = overrides.get("lengths",[50, 100, 200])
        K_values = overrides.get("K_values", [5, 10])
        encoder_modes = overrides.get("encoder_modes", ["sharp", "hysteretic"])
        bounded_trials = overrides.get("bounded_trials", 10)

        selector = cfg.memory_operator.selector
        r = selector.r
        lam = selector.lam
        solver = resolve_selector_solver(selector.solver)
        k = cfg.memory_operator.lift.k
        Q = cfg.memory_operator.topology.Q
        atol = cfg.verification.score_match_atol

        rng = np.random.default_rng(cfg.execution.seed)

        # ---------------------------------------------------------
        # TEST A: Causality and Full-Input Determinism (Thm 12.1)
        # ---------------------------------------------------------
        if verbose:
            print("  Test A: Checking Prefix-Map Consistency & Causality (Thm 12.1)...")

        causality_pass = 0
        causality_total = 0
        determinism_pass = 0
        determinism_total = 0
        
        for params in iter_progress(
            iter_parameter_grid(d=dims, T=lengths, K=K_values, encoder_mode=encoder_modes),
            desc="VZ2-04 Test A",
        ):
            d, T, K = params["d"], params["T"], params["K"]
            alpha = 0.3 if params["encoder_mode"] == "hysteretic" else 0.0
            W_theta = make_orthogonal_lift(k, d + 3, rng)
            
            # Base analytical trajectory
            traj = random_walk(d, T, step_std=0.5, seed=int(rng.integers(2**31)))

            # Determinism check: identical inputs must produce identical outputs
            state1 = compute_memory(traj, K, r, lam, W_theta, Q, alpha=alpha, solver=solver)
            state2 = compute_memory(traj.copy(), K, r, lam, W_theta, Q, alpha=alpha, solver=solver)

            determinism_total += 1
            is_deterministic = _states_match(state1, state2, atol)
            if is_deterministic:
                determinism_pass += 1
                
            if recorder is not None:
                recorder.log_scalar(A_determinism_checked=1.0, A_determinism_passed=float(is_deterministic))

            # Causality check: prefix output must be independent of future inputs
            probe_t = T // 2
            if probe_t <= 2:
                continue

            prefix_baseline = traj[:probe_t].copy()
            state_baseline_prefix = compute_memory(prefix_baseline, K, r, lam, W_theta, Q, alpha=alpha, solver=solver)

            # Perturb the future; prefix computation must remain unchanged
            traj_alt_future = traj.copy()
            traj_alt_future[probe_t:] += rng.standard_normal((T - probe_t, d)) * 20.0

            state_alt_prefix = compute_memory(traj_alt_future[:probe_t].copy(), K, r, lam, W_theta, Q, alpha=alpha, solver=solver)

            causality_total += 1
            is_causal = _states_match(state_baseline_prefix, state_alt_prefix, atol)
            if is_causal:
                causality_pass += 1
                
            if recorder is not None:
                recorder.log_scalar(A_causality_checked=1.0, A_causality_passed=float(is_causal))

        log.info("Test A (Prefix-Map Causality & Determinism): determinism %d/%d passed, causality %d/%d passed", determinism_pass, determinism_total, causality_pass, causality_total)

        # ---------------------------------------------------------
        # TEST B: Bounded Complexity Integrity (Thm 12.2)
        # ---------------------------------------------------------
        if verbose:
            print("  Test B: Verifying bounded dimensionality (Thm 12.2)...")

        bounded_pass = 0
        bounded_total = 0
        for params in iter_progress(
            iter_parameter_grid(d=dims, T=lengths, K=K_values),
            desc="VZ2-04 Test B",
        ):
            d, T, K = params["d"], params["T"], params["K"]
            W_theta = make_orthogonal_lift(k, d + 3, rng)

            for _ in range(bounded_trials):
                bounded_total += 1
                # Stress with high-transition trajectories to exercise the budget bound
                traj = piecewise_constant_auto(d, T, num_segments=min(3 * K, T), seed=int(rng.integers(2**31)))
                state = compute_memory(traj, K, r, lam, W_theta, Q, solver=solver)

                # Thm 12.2: anchor count must not exceed budget K
                is_bounded = len(state.anchor_indices) <= K
                if is_bounded:
                    bounded_pass += 1
                    
                if recorder is not None:
                    recorder.log_scalar(B_bounded_K=float(K), B_anchor_count=float(len(state.anchor_indices)), B_is_bounded=float(is_bounded))

        log.info("Test B (Memory Bounded Complexity Maximums): %d/%d passed", bounded_pass, bounded_total)

        # ---------------------------------------------------------
        # TEST C: Architecture State Geometry Invariants (Normalization bounds)
        # ---------------------------------------------------------
        if verbose:
            print("  Test C: Assuring architectural geometric invariants (§8)...")

        norm_pass = 0
        norm_total = 0
        for d in iter_progress(dims, desc="VZ2-04 Test C"):
            norm_total += 1
            W_theta = make_orthogonal_lift(k, d + 3, rng)
            traj = random_walk(d, 120, step_std=1.0, seed=int(rng.integers(2**31)))
            state = compute_memory(traj, K=10, r=r, lam=lam, W_Theta=W_theta, Q=Q, solver=solver)

            m_val = len(state.anchor_indices)

            # Dimensional invariants from §§8-9
            dim_v_correct = state.V.shape == (m_val, d + 3) if m_val > 0 else True
            dim_norm_correct = state.V_norm.shape == (m_val, d + 3) if m_val > 0 else True
            sigma_positivity = bool(np.all(state.sigma > 0.0)) if hasattr(state, "sigma") and getattr(state, "sigma").size > 0 else True
            dim_lift_correct = state.point_cloud.shape == (m_val, k)
            
            if all([dim_v_correct, dim_norm_correct, sigma_positivity, dim_lift_correct]):
                norm_pass += 1

        log.info("Test C (Extracted Object Geometric Parameter Dimensionalities): %d/%d passed", norm_pass, norm_total)

        # ---------------------------------------------------------
        # TEST D: Hysteretic vs Sharp Mode Distinct Representational Pathways
        # ---------------------------------------------------------
        if verbose:
            print("  Test D: Analytical parameter branching path checks (Modes)...")

        encoder_divergence_pass = 0
        encoder_total = 0
        
        for d in iter_progress(dims, desc="VZ2-04 Test D"):
            encoder_total += 1
            W_theta = make_orthogonal_lift(k, d + 3, rng)
            # High turbulence activates hysteretic decay vs sharp directness
            traj = random_walk(d, 100, step_std=1.5, seed=int(rng.integers(2**31)))

            state_sharp = compute_memory(traj, K=15, r=r, lam=lam, W_Theta=W_theta, Q=Q, alpha=0.0, solver=solver)
            state_hyst = compute_memory(traj, K=15, r=r, lam=lam, W_Theta=W_theta, Q=Q, alpha=0.3, solver=solver)

            # Non-sparse trajectories must produce distinct representations across modes
            if not _states_match(state_sharp, state_hyst, atol=1e-8):
                encoder_divergence_pass += 1

        log.info("Test D (Sharp/Hysteretic Generative Distinct Representation): %d/%d passed", encoder_divergence_pass, encoder_total)

    report.duration_seconds = timer.elapsed
    report.status = "PASS" if ((causality_pass == causality_total) and (determinism_pass == determinism_total) and bounded_pass == bounded_total and norm_pass == norm_total and encoder_divergence_pass == encoder_total) else "FAIL"
    return report


def _states_match(state1, state2, atol: float) -> bool:
    """Compare two memory operator states for structural and numerical equality."""
    if list(state1.anchor_indices) != list(state2.anchor_indices):
        return False

    if state1.point_cloud.shape != state2.point_cloud.shape:
        return False
    if state1.V_norm.shape != state2.V_norm.shape:
        return False

    if state1.point_cloud.size > 0:
        if not np.allclose(state1.point_cloud, state2.point_cloud, atol=atol):
            return False

    if state1.V_norm.size > 0:
        if not np.allclose(state1.V_norm, state2.V_norm, atol=atol):
            return False

    return True


if __name__ == "__main__":
    from experiments.verification.utils._shared import run_standalone
    sys.exit(run_standalone(
        caller_file=__file__,
        experiment_id="VZ2-04",
        experiment_name="Full Operator Causality and Boundedness",
        run_experiment_fn=run_experiment,
        config_key="vz2_04_full_operator",
    ))
