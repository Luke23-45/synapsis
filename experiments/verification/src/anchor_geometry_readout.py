"""
VZ2-08: Anchor Geometry and Relaxed Readout - Verification Experiment.

Z2 Reference: §§7-8 and §13 of 02_rigorous_architecture.md
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
from experiments.common.trajectory_generators import piecewise_constant, random_walk
from experiments.utils.config import ExperimentConfig
from experiments.utils.progress import iter_progress
from experiments.verification.utils._shared import iter_parameter_grid, make_orthogonal_lift, resolve_selector_solver
from synapse_core.anchor_selector import build_anchors
from synapse_core.event_encoder import sharp_event_score
from synapse_core.geometric_lift import anchor_vectors, apply_lift, normalize_anchors
from synapse_core.memory_operator import compute_memory
from synapse_core.training_readout import candidate_anchor_vectors, compute_relaxed_readout, relaxed_weighted_cloud

if TYPE_CHECKING:
    from experiments.verification.utils.data_recorder import VerificationRecorder

log = logging.getLogger(__name__)


def run_experiment(
    cfg: ExperimentConfig,
    verbose: bool = False,
    recorder: VerificationRecorder | None = None,
) -> ExperimentReport:
    """Run VZ2-08: Anchor Geometry and Relaxed Readout verification."""
    report = ExperimentReport(
        experiment_id="VZ2-08",
        experiment_name="Anchor Geometry and Relaxed Readout",
        formal_reference="§§7-8 and §13 of 02_rigorous_architecture.md",
        claim="Anchor construction is exact, normalization is bijective, and relaxed readout is typed and continuous",
    )

    timer = ExperimentTimer()
    timer.__enter__()

    overrides = cfg.experiments.get("vz2_08_anchor_geometry_readout", {})
    dims = overrides.get("dims", [2, 4])
    lengths = overrides.get("lengths", [40, 80])
    num_trials = overrides.get("num_trials", 10)

    rng = np.random.default_rng(cfg.execution.seed)
    atol = cfg.verification.score_match_atol

    if verbose:
        print("  Test A: Anchor Construction Exactness...")

    anchor_pass = 0
    anchor_total = 0
    for params in iter_progress(
        iter_parameter_grid(d=dims, T=lengths),
        desc="VZ2-08 Test A",
    ):
        anchor_total += 1
        T = params["T"]
        change_points = [max(2, T // 4), max(3, T // 2), max(4, 3 * T // 4)]
        change_points = sorted(set(cp for cp in change_points if cp < T))
        traj, _ = piecewise_constant(params["d"], T, change_points, seed=int(rng.integers(2**31)))
        scores = sharp_event_score(traj)
        anchors = build_anchors(change_points, traj, scores)

        exact = True
        for j, anchor in enumerate(anchors):
            idx = change_points[j]
            expected_t = (idx + 1) / T
            expected_delta = idx if j == 0 else idx - change_points[j - 1]
            if not (
                np.isclose(anchor.t, expected_t, atol=atol)
                and np.allclose(anchor.s, traj[idx], atol=atol)
                and anchor.delta == expected_delta
                and np.isclose(anchor.xi, scores[idx], atol=atol)
                and anchor.index == idx
            ):
                exact = False
                break

        if recorder is not None:
            recorder.log_scalar(A_exact=float(exact))
            
        if exact:
            anchor_pass += 1

    log.info("Test A (Anchor Sequence Exactness): %d/%d passed", anchor_pass, anchor_total)

    if verbose:
        print("  Test B: Normalized Geometry Invertibility...")

    norm_pass = 0
    norm_total = 0
    for params in iter_progress(
        iter_parameter_grid(d=dims, T=lengths),
        desc="VZ2-08 Test B",
    ):
        norm_total += 1
        T = params["T"]
        traj = random_walk(params["d"], T, step_std=0.5, seed=int(rng.integers(2**31)))
        scores = sharp_event_score(traj)
        indices = [idx for idx in range(1, T, max(2, T // 5))]
        anchors = build_anchors(indices, traj, scores)
        V = anchor_vectors(anchors)
        V_norm, mu, sigma = normalize_anchors(V)
        reconstructed = V_norm * sigma + mu

        mean_ok = np.allclose(np.mean(V_norm, axis=0), 0.0, atol=1e-10)
        std_ok = np.allclose(np.std(V_norm, axis=0), 1.0, atol=1e-10)
        invertible = np.all(sigma > 0.0) and np.allclose(reconstructed, V, atol=atol)
        if recorder is not None:
            recorder.log_scalar(B_mean_ok=float(mean_ok), B_std_ok=float(std_ok), B_invertible=float(invertible))
            
        if mean_ok and std_ok and invertible:
            norm_pass += 1

    log.info("Test B (Normalization Bijectivity): %d/%d passed", norm_pass, norm_total)

    if verbose:
        print("  Test C: Deploy Geometry Integration...")

    integration_pass = 0
    integration_total = 0
    selector = cfg.memory_operator.selector
    solver = resolve_selector_solver(selector.solver)
    for params in iter_progress(
        iter_parameter_grid(d=dims, T=lengths),
        desc="VZ2-08 Test C",
    ):
        integration_total += 1
        traj = random_walk(params["d"], params["T"], step_std=0.5, seed=int(rng.integers(2**31)))
        W_theta = make_orthogonal_lift(cfg.memory_operator.lift.k, params["d"] + 3, rng)
        state = compute_memory(
            traj,
            selector.K,
            selector.r,
            selector.lam,
            W_theta,
            cfg.memory_operator.topology.Q,
            solver=solver,
        )

        V_manual = anchor_vectors(state.anchors, D=params["d"] + 3)
        V_norm_manual, _, _ = normalize_anchors(V_manual, mu=state.mu, sigma=state.sigma)
        cloud_manual = apply_lift(V_norm_manual, W_theta)

        is_deploy_ok = (
            np.allclose(state.V, V_manual, atol=atol)
            and np.allclose(state.V_norm, V_norm_manual, atol=atol)
            and np.allclose(state.point_cloud, cloud_manual, atol=atol)
        )
        
        if recorder is not None:
            recorder.log_scalar(C_is_deploy_ok=float(is_deploy_ok))
            
        if is_deploy_ok:
            integration_pass += 1

    log.info("Test C (Deploy Geometry Integration): %d/%d passed", integration_pass, integration_total)

    if verbose:
        print("  Test D: Relaxed Readout Typing and Continuity...")

    readout_pass = 0
    readout_total = 0
    for params in iter_progress(
        iter_parameter_grid(d=dims, T=lengths),
        desc="VZ2-08 Test D",
    ):
        for _ in iter_progress(range(num_trials), desc="VZ2-08 Test D trials", total=num_trials):
            readout_total += 1
            traj = random_walk(params["d"], params["T"], step_std=0.5, seed=int(rng.integers(2**31)))
            W_theta = make_orthogonal_lift(cfg.memory_operator.lift.k, params["d"] + 3, rng)

            deploy_state = compute_memory(
                traj,
                selector.K,
                selector.r,
                selector.lam,
                W_theta,
                cfg.memory_operator.topology.Q,
                solver=solver,
            )
            relaxed_state = compute_relaxed_readout(
                traj,
                selector.K,
                selector.r,
                selector.lam,
                W_theta,
                solver=solver,
            )

            candidate_manual = candidate_anchor_vectors(traj, relaxed_state.event_scores)
            lifted_manual, weighted_manual = relaxed_weighted_cloud(
                relaxed_state.normalized_vectors,
                W_theta,
                relaxed_state.y_star,
            )

            dy = rng.standard_normal(relaxed_state.y_star.shape[0]) * 1e-4
            dy[0] = 0.0
            _, weighted_perturbed = relaxed_weighted_cloud(
                relaxed_state.normalized_vectors,
                W_theta,
                relaxed_state.y_star + dy,
            )
            y_linearity_ok = np.allclose(
                weighted_perturbed - relaxed_state.weighted_cloud,
                lifted_manual * dy[:, None],
                atol=1e-10,
            )

            delta_W = rng.standard_normal(W_theta.shape) * 1e-4
            lifted_shifted, weighted_shifted = relaxed_weighted_cloud(
                relaxed_state.normalized_vectors,
                W_theta + delta_W,
                relaxed_state.y_star,
            )
            w_linearity_ok = np.allclose(
                weighted_shifted - relaxed_state.weighted_cloud,
                (lifted_shifted - lifted_manual) * relaxed_state.y_star[:, None],
                atol=1e-10,
            )

            typed_ok = deploy_state.point_cloud.shape[0] <= selector.K and relaxed_state.weighted_cloud.shape == (
                params["T"],
                cfg.memory_operator.lift.k,
            )

            if recorder is not None:
                recorder.log_scalar(D_typed_ok=float(typed_ok), D_y_linearity_ok=float(y_linearity_ok), D_w_linearity_ok=float(w_linearity_ok))

            if (
                np.allclose(relaxed_state.candidate_vectors, candidate_manual, atol=atol)
                and np.allclose(relaxed_state.lifted_candidates, lifted_manual, atol=atol)
                and np.allclose(relaxed_state.weighted_cloud, weighted_manual, atol=atol)
                and typed_ok
                and y_linearity_ok
                and w_linearity_ok
            ):
                readout_pass += 1

    log.info("Test D (Relaxed Readout Typing): %d/%d passed", readout_pass, readout_total)

    timer.__exit__(None, None, None)
    report.duration_seconds = timer.elapsed
    report.status = "PASS" if (anchor_pass == anchor_total and norm_pass == norm_total and integration_pass == integration_total and readout_pass == readout_total) else "FAIL"
    return report


if __name__ == "__main__":
    from experiments.verification.utils._shared import run_standalone
    sys.exit(run_standalone(
        caller_file=__file__,
        experiment_id="VZ2-08",
        experiment_name="Anchor Geometry and Relaxed Readout",
        run_experiment_fn=run_experiment,
        config_key="vz2_08_anchor_geometry_readout",
    ))
