"""
VZ2-07: Event Encoder and Saliency Normalization - Verification Experiment.

Z2 Reference: §§3-4 of 02_rigorous_architecture.md
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
from experiments.common.trajectory_generators import random_walk
from experiments.utils.config import ExperimentConfig
from experiments.utils.progress import iter_progress
from experiments.verification.utils._shared import iter_parameter_grid, make_orthogonal_lift, resolve_selector_solver
from synapse_core.anchor_selector import solve_relaxed_selector
from synapse_core.event_encoder import hysteretic_event_score, sharp_event_score
from synapse_core.memory_operator import compute_memory
from synapse_core.saliency_normalizer import causal_running_stats, normalize_saliency

if TYPE_CHECKING:
    from experiments.verification.utils.data_recorder import VerificationRecorder

log = logging.getLogger(__name__)


def run_experiment(
    cfg: ExperimentConfig,
    verbose: bool = False,
    recorder: VerificationRecorder | None = None,
) -> ExperimentReport:
    """Run VZ2-07: Event Encoder and Saliency Normalization verification."""
    report = ExperimentReport(
        experiment_id="VZ2-07",
        experiment_name="Event Encoder and Saliency Normalization",
        formal_reference="§§3-4 of 02_rigorous_architecture.md",
        claim="Encoder recursion and saliency normalization are causal, deterministic, and formula-correct",
    )

    timer = ExperimentTimer()
    timer.__enter__()

    overrides = cfg.experiments.get("vz2_07_encoder_saliency", {})
    dims = overrides.get("dims", [2, 5])
    lengths = overrides.get("lengths", [50, 100])
    alpha_values = overrides.get("alpha_values", [0.2, 0.6])
    num_perturbations = overrides.get("num_perturbations", 20)
    saliency_modes = overrides.get("saliency_modes", ["identity", "z_score", "temperature"])
    temperatures = overrides.get("temperatures", [0.5, 1.0, 2.0])

    rng = np.random.default_rng(cfg.execution.seed)
    atol = cfg.verification.score_match_atol

    if verbose:
        print("  Test A: Sharp Encoder Exactness...")

    sharp_pass = 0
    sharp_total = 0
    for params in iter_progress(
        iter_parameter_grid(d=dims, T=lengths),
        desc="VZ2-07 Test A",
    ):
        sharp_total += 1
        traj = random_walk(params["d"], params["T"], step_std=0.5, seed=int(rng.integers(2**31)))
        scores = sharp_event_score(traj)
        manual = np.zeros(params["T"], dtype=np.float64)
        manual[1:] = np.linalg.norm(traj[1:] - traj[:-1], axis=1)
        is_sharp_ok = np.allclose(scores, manual, atol=atol)
        
        if recorder is not None:
            recorder.log_scalar(A_is_sharp_ok=float(is_sharp_ok))
            recorder.save_trial_data(f"A_d{params['d']}_T{params['T']}", traj=traj, scores=scores)
            
        if is_sharp_ok:
            sharp_pass += 1

    log.info("Test A (Sharp Special Case): %d/%d passed", sharp_pass, sharp_total)

    if verbose:
        print("  Test B: Hysteretic Recursion and Prefix Causality...")

    hysteretic_pass = 0
    hysteretic_total = 0
    for params in iter_progress(
        iter_parameter_grid(d=dims, T=lengths, alpha=alpha_values),
        desc="VZ2-07 Test B",
    ):
        d = params["d"]
        T = params["T"]
        alpha = params["alpha"]
        traj = random_walk(d, T, step_std=0.5, seed=int(rng.integers(2**31)))
        scores, latent = hysteretic_event_score(traj, alpha=_causal_alpha(alpha))
        manual_scores, manual_latent = _manual_hysteretic(traj, alpha)

        formula_ok = np.allclose(scores, manual_scores, atol=atol) and np.allclose(latent, manual_latent, atol=atol)
        prefix_ok = True
        probe_t = T // 2
        for _ in range(num_perturbations):
            perturbed = traj.copy()
            perturbed[probe_t:] += rng.standard_normal((T - probe_t, d)) * 0.5
            pert_scores, pert_latent = hysteretic_event_score(perturbed, alpha=_causal_alpha(alpha))
            if not (
                np.allclose(scores[:probe_t], pert_scores[:probe_t], atol=atol)
                and np.allclose(latent[:probe_t], pert_latent[:probe_t], atol=atol)
            ):
                prefix_ok = False
                break

        hysteretic_total += 1
        
        if recorder is not None:
            recorder.log_scalar(B_formula_ok=float(formula_ok), B_prefix_ok=float(prefix_ok))
            recorder.save_trial_data(f"B_d{d}_T{T}_alpha{alpha}", traj=traj, scores=scores, latent=latent)
            
        if formula_ok and prefix_ok:
            hysteretic_pass += 1

    log.info("Test B (Hysteretic Recursion): %d/%d passed", hysteretic_pass, hysteretic_total)

    if verbose:
        print("  Test C: Saliency Normalization Causality...")

    saliency_pass = 0
    saliency_total = 0
    for mode in saliency_modes:
        mode_temperatures = temperatures if mode == "temperature" else [1.0]
        for params in iter_progress(
            iter_parameter_grid(T=lengths, temperature=mode_temperatures),
            desc="VZ2-07 Test C",
        ):
            T = params["T"]
            event_scores = np.zeros(T, dtype=np.float64)
            event_scores[1:] = np.abs(rng.standard_normal(T - 1))

            saliency = normalize_saliency(event_scores, mode=mode, temperature=params["temperature"])
            expected = _expected_saliency(event_scores, mode, params["temperature"])
            deterministic = np.allclose(
                saliency,
                normalize_saliency(event_scores, mode=mode, temperature=params["temperature"]),
                atol=atol,
            )

            prefix_ok = True
            probe_t = T // 2
            for _ in range(num_perturbations):
                perturbed = event_scores.copy()
                perturbed[probe_t:] = np.abs(rng.standard_normal(T - probe_t))
                pert_saliency = normalize_saliency(perturbed, mode=mode, temperature=params["temperature"])
                if not np.allclose(saliency[:probe_t], pert_saliency[:probe_t], atol=atol):
                    prefix_ok = False
                    break

            saliency_total += 1
            is_saliency_ok = np.allclose(saliency, expected, atol=atol) and deterministic and prefix_ok and saliency[0] == 0.0
            
            if recorder is not None:
                recorder.log_scalar(**{"C_mode_" + mode: 1.0, "C_is_saliency_ok": float(is_saliency_ok)})
                recorder.save_trial_data(f"C_{mode}_T{T}_temp{params['temperature']}", scores=event_scores, saliency=saliency)
                
            if is_saliency_ok:
                saliency_pass += 1

    log.info("Test C (Causal Saliency Normalization): %d/%d passed", saliency_pass, saliency_total)

    if verbose:
        print("  Test D: Operator Integration with Saliency Modes...")

    integration_pass = 0
    integration_total = 0
    selector = cfg.memory_operator.selector
    solver = resolve_selector_solver(selector.solver)
    for mode in iter_progress(saliency_modes, desc="VZ2-07 Test D"):
        integration_total += 1
        d = dims[0]
        T = lengths[0]
        traj = random_walk(d, T, step_std=0.5, seed=int(rng.integers(2**31)))
        W_theta = make_orthogonal_lift(cfg.memory_operator.lift.k, d + 3, rng)
        temperature = temperatures[0] if mode == "temperature" else 1.0

        state = compute_memory(
            traj,
            selector.K,
            selector.r,
            selector.lam,
            W_theta,
            cfg.memory_operator.topology.Q,
            solver=solver,
            saliency_mode=mode,
            saliency_temperature=temperature,
        )

        manual_scores = sharp_event_score(traj)
        manual_saliency = normalize_saliency(manual_scores, mode=mode, temperature=temperature)
        manual_y = solve_relaxed_selector(manual_saliency, selector.K, selector.r, selector.lam, solver=solver)

        is_integration_ok = np.allclose(state.event_scores, manual_scores, atol=atol) and np.allclose(state.y_star, manual_y, atol=atol)
        
        if recorder is not None:
            recorder.log_scalar(**{"D_mode_" + mode: 1.0, "D_is_integration_ok": float(is_integration_ok)})
            recorder.save_trial_data(f"D_{mode}_T{T}", traj=traj, state_scores=state.event_scores, state_y=state.y_star)
            
        if is_integration_ok:
            integration_pass += 1

    log.info("Test D (Selector Input Integration): %d/%d passed", integration_pass, integration_total)

    timer.__exit__(None, None, None)
    report.duration_seconds = timer.elapsed
    report.status = "PASS" if (sharp_pass == sharp_total and hysteretic_pass == hysteretic_total and saliency_pass == saliency_total and integration_pass == integration_total) else "FAIL"
    return report


def _causal_alpha(base_alpha: float):
    def alpha_fn(t: int, prefix: np.ndarray) -> float:
        del t
        prefix_energy = np.linalg.norm(prefix[-1] - prefix[0]) if prefix.shape[0] > 1 else 0.0
        return float(np.clip(base_alpha + 0.05 * np.tanh(prefix_energy), 0.0, 0.95))

    return alpha_fn


def _manual_hysteretic(trajectory: np.ndarray, base_alpha: float) -> tuple[np.ndarray, np.ndarray]:
    T, d = trajectory.shape
    latent = np.zeros((T, d), dtype=np.float64)
    scores = np.zeros(T, dtype=np.float64)
    alpha_fn = _causal_alpha(base_alpha)

    for t in range(1, T):
        alpha_t = alpha_fn(t, trajectory[: t + 1])
        phi_val = trajectory[t] - trajectory[t - 1]
        latent[t] = alpha_t * latent[t - 1] + (1.0 - alpha_t) * phi_val
        scores[t] = np.linalg.norm(latent[t])
    return scores, latent


def _expected_saliency(event_scores: np.ndarray, mode: str, temperature: float) -> np.ndarray:
    if mode == "identity":
        expected = np.array(event_scores, copy=True)
        expected[0] = 0.0
        return expected

    mean, std = causal_running_stats(event_scores)
    z = (event_scores - mean) / std
    if mode == "z_score":
        expected = z
    else:
        expected = event_scores * (1.0 / (1.0 + np.exp(-z / temperature)))
    expected[0] = 0.0
    return expected


if __name__ == "__main__":
    from experiments.verification.utils._shared import run_standalone
    sys.exit(run_standalone(
        caller_file=__file__,
        experiment_id="VZ2-07",
        experiment_name="Event Encoder and Saliency Normalization",
        run_experiment_fn=run_experiment,
        config_key="vz2_07_encoder_saliency",
    ))
