"""
Experiment 05 -- Topological Stability
========================================

Formal claim
    Theorem 8.5  (docs/formal_math/02_rigorous_architecture.md)
    d_H(P, P') <= eps  and  d_B(Dgm_q(P), Dgm_q(P')) <= 2*eps
    when max_j ||p_j - p'_j||_2 <= eps.

Outputs
    figures/stability_bounds.{png,pdf}
    metrics/metrics.jsonl, results.csv
    artifacts/report.json
    logs/run.log
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synapse_core.topological_summary import (
    compute_persistence_diagrams, has_full_persistence_backend,
)
from experiments.utils.config import load_config, validate, ExperimentConfig, get_experiment_overrides
from experiments.utils.model_io import create_run_capsule, save_config_snapshot, save_run_pointer
from experiments.utils.logging import setup_run_logging
from experiments.utils.exporter import append_metrics_jsonl, save_results_csv
from experiments.utils.progress import iter_progress
from experiments.common.metrics import hausdorff_distance, bottleneck_distance, max_pointwise_distance
from experiments.common.report import (
    ExperimentReport, TestCase, ExperimentTimer, print_report, save_report_json,
)
from experiments.common.theme import apply_theme
from experiments.common.plotting import plot_topological_stability

log = logging.getLogger(__name__)

EXPERIMENT_ID = "EXP-05"
EXPERIMENT_NAME = "Topological Stability"
FORMAL_REF = "Theorem 8.5 (02_rigorous_architecture.md)"
CLAIM = "d_H <= eps and d_B <= 2*eps under pointwise perturbation"

CSV_FIELDS = [
    "m", "D", "epsilon", "num_perturbations",
    "max_hausdorff_ratio", "max_bottleneck_ratio", "passed",
]


def run_experiment(
    cfg: ExperimentConfig | None = None,
    verbose: bool = False,
) -> ExperimentReport:
    if cfg is None:
        cfg = load_config("experiments/configs/default.yaml")
        validate(cfg)

    ov = get_experiment_overrides(cfg, "exp05_topological_stability")
    m_values          = ov.get("m_values", [5, 10, 20])
    D_values          = ov.get("D_values", [5, 8])
    epsilon_values    = ov.get("epsilon_values", [0.001, 0.01, 0.05, 0.1, 0.5, 1.0])
    num_perturbations = ov.get("num_perturbations", 50)
    Q = cfg.memory_operator.Q

    capsule = create_run_capsule(cfg.output_dir, "topological_stability")
    save_config_snapshot(capsule, cfg)
    apply_theme(cfg.plotting.theme)
    setup_run_logging(capsule, verbose=verbose, experiment_id=EXPERIMENT_ID)

    log.info("[%s] %s", EXPERIMENT_ID, EXPERIMENT_NAME)
    log.info("  m=%s  D=%s  eps=%s  perturbations=%d  Q=%d",
             m_values, D_values, epsilon_values, num_perturbations, Q)

    rng = np.random.default_rng(cfg.execution.seed)

    report = ExperimentReport(
        experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
        formal_reference=FORMAL_REF, claim=CLAIM,
    )
    csv_rows: list[dict] = []
    full_backend = has_full_persistence_backend()
    report.metadata["full_persistence_backend"] = full_backend
    report.metadata["requested_Q"] = Q
    if Q > 0 and not full_backend:
        report.add_case(TestCase(
            name="full_topology_backend_available",
            passed=False,
            details={"requested_Q": Q},
            error="Neither gudhi nor ripser is installed; only H_0 can be verified in this environment.",
        ))
        csv_rows.append({
            "m": -1,
            "D": -1,
            "epsilon": -1.0,
            "num_perturbations": 0,
            "max_hausdorff_ratio": float("nan"),
            "max_bottleneck_ratio": float("nan"),
            "passed": False,
        })
        append_metrics_jsonl(
            {
                "case": "full_topology_backend_available",
                "requested_Q": Q,
                "passed": False,
            },
            capsule.metrics / "metrics.jsonl",
        )
    # For the summary figure
    eps_for_fig: list[float] = []
    h_ratios_fig: list[float] = []
    b_ratios_fig: list[float] = []

    with ExperimentTimer() as timer:
        sweep = [(m, D, eps)
                 for m in m_values for D in D_values for eps in epsilon_values]

        for m, D, eps in iter_progress(sweep, desc="topological stability"):
            all_bounded = True
            error_msg = None
            max_h_ratio = 0.0
            max_b_ratio = 0.0

            base_cloud = rng.standard_normal((m, D)) * 5.0

            for p in range(num_perturbations):
                direction = rng.standard_normal((m, D))
                norms = np.linalg.norm(direction, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-15)
                perturbation = direction / norms * eps
                perturbed = base_cloud + perturbation

                pw_dist = max_pointwise_distance(base_cloud, perturbed)
                assert pw_dist <= eps + 1e-10, f"Construction bug: pw={pw_dist} > eps={eps}"

                d_H = hausdorff_distance(base_cloud, perturbed)
                if d_H > eps + 1e-10:
                    all_bounded = False
                    error_msg = f"d_H={d_H:.6e} > eps={eps}"
                    break

                h_ratio = d_H / eps if eps > 0 else 0.0
                max_h_ratio = max(max_h_ratio, h_ratio)

                dgms_base = compute_persistence_diagrams(base_cloud, Q=Q)
                dgms_pert = compute_persistence_diagrams(perturbed, Q=Q)
                for q in range(min(len(dgms_base), len(dgms_pert))):
                    d_B = bottleneck_distance(dgms_base[q], dgms_pert[q])
                    if d_B > 2 * eps + 1e-10:
                        all_bounded = False
                        error_msg = f"d_B={d_B:.6e} > 2*eps={2*eps} at H_{q}"
                        break
                    b_ratio = d_B / (2 * eps) if eps > 0 else 0.0
                    max_b_ratio = max(max_b_ratio, b_ratio)

                if not all_bounded:
                    break

            case_name = f"m{m}_D{D}_eps{eps}"
            report.add_case(TestCase(
                name=case_name, passed=all_bounded,
                details={"m": m, "D": D, "eps": eps,
                         "h_ratio": round(max_h_ratio, 6),
                         "b_ratio": round(max_b_ratio, 6)},
                error=error_msg,
            ))

            row = {"m": m, "D": D, "epsilon": eps,
                   "num_perturbations": num_perturbations,
                   "max_hausdorff_ratio": round(max_h_ratio, 6),
                   "max_bottleneck_ratio": round(max_b_ratio, 6),
                   "passed": all_bounded}
            csv_rows.append(row)
            append_metrics_jsonl(
                {"case": case_name, **row}, capsule.metrics / "metrics.jsonl",
            )

            eps_for_fig.append(eps)
            h_ratios_fig.append(max_h_ratio)
            b_ratios_fig.append(max_b_ratio)

            if verbose:
                icon = "PASS" if all_bounded else "FAIL"
                print(f"  [{icon}] {case_name}  h={max_h_ratio:.3f}  b={max_b_ratio:.3f}")

    report.duration_seconds = timer.elapsed
    report.finalize()

    save_report_json(report, str(capsule.artifacts / "report.json"))
    save_results_csv(csv_rows, CSV_FIELDS, capsule.metrics / "results.csv")

    # Use first m and D for the figure
    n_eps = len(epsilon_values)
    plot_topological_stability(
        epsilon_values,
        h_ratios_fig[:n_eps],
        b_ratios_fig[:n_eps],
        save_path=capsule.figures / "stability_bounds",
        formats=cfg.plotting.formats, theme=cfg.plotting.theme,
    )
    log.info("Figure saved -> %s/stability_bounds", capsule.figures)

    save_run_pointer(capsule, cfg.output_dir)
    log.info("[%s] %s  (%d/%d)", EXPERIMENT_ID, report.status,
             report.passed_cases, report.total_cases)
    return report


if __name__ == "__main__":
    config_name = "experiments/configs/default.yaml"
    for arg in sys.argv:
        if arg.startswith("--config="):
            config_name = arg.split("=", 1)[1]
    cfg = load_config(config_name)
    validate(cfg)
    v = "--verbose" in sys.argv or "-v" in sys.argv
    report = run_experiment(cfg=cfg, verbose=v)
    print_report(report)
    sys.exit(0 if report.status == "PASS" else 1)
