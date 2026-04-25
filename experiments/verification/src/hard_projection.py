"""
VZ2-02: Hard Projection and Separation - Verification Experiment.

Z2 Reference: §6 of 02_rigorous_architecture.md
Formal Claims: Prop 6.1 (Determinism and strict bound budgeting), Prop 6.2 (Metric temporal separation),
alongside exact mathematical tracking against pure algorithmic definitions.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when running this script directly
if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import logging

import numpy as np

from experiments.common.report import ExperimentReport, ExperimentTimer
from experiments.utils.config import ExperimentConfig
from experiments.utils.progress import iter_progress
from experiments.verification.utils._shared import iter_parameter_grid
from synapse_core.anchor_selector import hard_projection

log = logging.getLogger(__name__)


def exact_hard_projection_oracle(y: np.ndarray, K: int, r: int) -> list[int]:
    """
    Mathematical oracle implementing Z2 §6 hard projection specification.

    1. Identify strictly positive elements.
    2. Sort descending by ``y`` value, with ascending index as tiebreaker.
    3. Greedily select up to K elements subject to ``|u - t| > r`` spacing.
    """
    support = np.where(y > 0)[0]
    if support.size == 0:
        return []

    # Dual-key stable sort: primary descending by y, secondary ascending by index
    y_sup = y[support]
    sort_order = np.lexsort((support, -y_sup))
    greedy_order = support[sort_order]

    # Constrained greedy packing with refractory spacing
    retained: list[int] = []
    for idx in greedy_order:
        if len(retained) == K:
            break
        if all(abs(idx - retained_u) > r for retained_u in retained):
            retained.append(int(idx))

    return sorted(retained)


def generate_adversarial_patterns(T: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Generate stressed signal patterns targeting known failure modes."""
    patterns = []

    # 1. Standard noise field
    patterns.append(rng.uniform(-1.0, 1.0, size=T))

    # 2. All-negative (empty support)
    patterns.append(rng.uniform(-5.0, 0.0, size=T))

    # 3. Dense tie saturation (forces index tiebreaker logic)
    tie_matrix = np.ones(T, dtype=np.float64)
    tie_matrix[: T // 2] = -1.0
    rng.shuffle(tie_matrix)
    patterns.append(tie_matrix)

    # 4. Alternating comb (directly contests radius r)
    comb = np.zeros(T, dtype=np.float64)
    comb[::2] = 0.5
    comb[::3] = 0.8
    patterns.append(comb)

    # Root index t_0 must always be zero
    for p in patterns:
        p[0] = 0.0

    return patterns


def run_experiment(cfg: ExperimentConfig, verbose: bool = False, recorder=None) -> ExperimentReport:
    """Run VZ2-02: Algorithmic Verification against Theoretical Strict bounds (Hard Projection)."""
    report = ExperimentReport(
        experiment_id="VZ2-02",
        experiment_name="Hard Projection Strict Optimization Integrity",
        formal_reference="§6 of 02_rigorous_architecture.md",
        claim="Prop 6.1 (Pure mapping function determinism under budget limits K), Prop 6.2 (Uncompromised positional resolution metrics `r`)",
    )

    overrides = cfg.experiments.get("vz2_02_hard_projection", {})
    K_values = overrides.get("K_values",[1, 3, 5, 10, 50])
    r_values = overrides.get("r_values",[0, 1, 2, 5, 10])
    T_values = overrides.get("T_values", [15, 100, 1000])

    rng = np.random.default_rng(cfg.execution.seed)

    timer = ExperimentTimer()
    with timer:

        # Variables tracking robust condition invariants over global bounds parameters.
        determinism_pass = 0
        budget_pass = 0
        separation_pass = 0
        positive_support_pass = 0
        oracle_match_pass = 0
        total_evaluations = 0

        if verbose:
            print("  Conducting High-Density Adversarial Oracle Consistency Checks...")

        for params in iter_progress(
            iter_parameter_grid(K=K_values, r=r_values, T=T_values),
            desc="VZ2-02 sweep",
        ):
            K_t, r_t, T_len = params["K"], params["r"], params["T"]
            test_signals = generate_adversarial_patterns(T_len, rng)

            for sig_idx, signal in enumerate(test_signals):
                total_evaluations += 1
                
                # Executions generated using module library and pure Python mathematically restricted sequence
                y_signal = np.array(signal, copy=True, dtype=np.float64)
                lib_projected_1 = hard_projection(y_signal, K_t, r_t)
                lib_projected_2 = hard_projection(y_signal, K_t, r_t) # Assure temporal parity

                oracle_result = exact_hard_projection_oracle(y_signal, K_t, r_t)

                det_ok = list(lib_projected_1) == list(lib_projected_2)
                oracle_ok = list(lib_projected_1) == oracle_result
                budget_ok = len(lib_projected_1) <= K_t

                p_arr = np.array(lib_projected_1, dtype=int)
                if len(p_arr) < 2:
                    sep_ok = True
                else:
                    sep_ok = bool(np.all(np.diff(np.sort(p_arr)) > r_t))

                if len(p_arr) == 0:
                    pos_ok = True
                else:
                    pos_ok = bool(np.all(y_signal[p_arr] > 0.0))

                if det_ok:
                    determinism_pass += 1
                if oracle_ok:
                    oracle_match_pass += 1
                if budget_ok:
                    budget_pass += 1
                if sep_ok:
                    separation_pass += 1
                if pos_ok:
                    positive_support_pass += 1

                # ---- DATA RECORDING ----
                if recorder is not None:
                    min_spacing = int(np.min(np.diff(np.sort(p_arr)))) if len(p_arr) >= 2 else -1
                    recorder.log_scalar(
                        trial=total_evaluations, K=K_t, r=r_t, T=T_len,
                        pattern_idx=sig_idx, n_selected=len(lib_projected_1),
                        deterministic=det_ok, oracle_match=oracle_ok,
                        budget_ok=budget_ok, separation_ok=sep_ok,
                        positive_support_ok=pos_ok, min_spacing=min_spacing,
                    )
                    recorder.save_trial_data(
                        f"K{K_t}_r{r_t}_T{T_len}_pat{sig_idx}",
                        y_signal=y_signal,
                        projected_indices=p_arr,
                        oracle_indices=np.array(oracle_result, dtype=int),
                    )

        log.info("Test A (Complete Oracle Convergence): %d/%d passed", oracle_match_pass, total_evaluations)
        log.info("Test B (Method Level Functional Determinism): %d/%d passed", determinism_pass, total_evaluations)
        log.info("Test C (Bound Extraction Complexity Checks): %d/%d passed", budget_pass, total_evaluations)
        log.info("Test D (Sequence Minimum Resolution Interval Strict Isolation): %d/%d passed", separation_pass, total_evaluations)
        log.info("Test E (Subgradient Origin Support Validity Constraints): %d/%d passed", positive_support_pass, total_evaluations)

    report.duration_seconds = timer.elapsed
    report.status = "PASS" if (oracle_match_pass == total_evaluations and determinism_pass == total_evaluations and budget_pass == total_evaluations and separation_pass == total_evaluations and positive_support_pass == total_evaluations) else "FAIL"
    if recorder is not None:
        recorder.finalize()
    return report


if __name__ == "__main__":
    from experiments.verification.utils._shared import run_standalone
    sys.exit(run_standalone(
        caller_file=__file__,
        experiment_id="VZ2-02",
        experiment_name="Hard Projection Strict Optimization Integrity",
        run_experiment_fn=run_experiment,
        config_key="vz2_02_hard_projection",
        project_root=str(Path(__file__).resolve().parent.parent.parent.parent),
    ))
