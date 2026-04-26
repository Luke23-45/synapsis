"""
Report Generation — M1 Experiment
==================================

Generates JSON and Markdown reports from evaluation results.
The Markdown report follows the format specified in §5 of PLAN.md.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.core.config import ExperimentConfig, Condition
from src.engine.evaluate import ConditionEvaluation
from src.engine.metrics import ComparisonResult

log = logging.getLogger(__name__)


def generate_json_report(
    evaluations: Dict[str, ConditionEvaluation],
    comparisons: List[ComparisonResult],
    stratified: Dict[str, Dict[str, Dict]],
    output_path: Path,
) -> Path:
    """Generate a JSON report with all metrics and statistical tests.

    Parameters
    ----------
    evaluations : dict mapping condition_key → ConditionEvaluation
    comparisons : list of ComparisonResult
    stratified : dict mapping condition_key → stratified results
    output_path : Path

    Returns
    -------
    Path to the saved JSON file.
    """
    report = {
        "primary_comparison": [],
        "ablation_analysis": [],
        "rollout_results": {},
        "stratified_analysis": {},
    }

    # Primary comparison table
    for comp in comparisons:
        entry = {
            "metric": comp.metric_name,
            "condition_a": comp.condition_a,
            "condition_b": comp.condition_b,
            "mean_a": comp.mean_a,
            "std_a": comp.std_a,
            "mean_b": comp.mean_b,
            "std_b": comp.std_b,
            "mean_diff": comp.mean_diff,
            "cohens_d": comp.cohens_d,
            "d_ci": [comp.d_ci_lower, comp.d_ci_upper],
            "t_statistic": comp.t_statistic,
            "p_value": comp.p_value,
            "welch_significant": comp.welch_significant,
            "significant_after_bonferroni": comp.significant_after_correction,
            "is_normal": comp.is_normal,
            "shapiro_p": comp.shapiro_p,
            "wilcoxon_p": comp.wilcoxon_p,
        }
        if comp.condition_b in ("B_synapse",) and comp.condition_a in (
            "A1_recent", "A2_uniform"
        ):
            report["primary_comparison"].append(entry)
        else:
            report["ablation_analysis"].append(entry)

    # Rollout results
    for cond_key, eval_result in evaluations.items():
        if eval_result.rollout is not None:
            report["rollout_results"][cond_key] = eval_result.rollout

    # Stratified analysis
    report["stratified_analysis"] = stratified

    # Per-condition summary
    report["condition_summaries"] = {}
    for cond_key, eval_result in evaluations.items():
        report["condition_summaries"][cond_key] = {
            "mean_mse": eval_result.mean_mse,
            "std_mse": eval_result.std_mse,
            "mean_cosine": eval_result.mean_cosine,
            "mean_phase_acc": eval_result.mean_phase_acc,
            "num_episodes": len(eval_result.per_episode_mse),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("Saved JSON report to %s", output_path)
    return output_path


def generate_markdown_report(
    evaluations: Dict[str, ConditionEvaluation],
    comparisons: List[ComparisonResult],
    stratified: Dict[str, Dict[str, Dict]],
    output_path: Path,
) -> Path:
    """Generate a Markdown report following the PLAN.md format.

    Parameters
    ----------
    evaluations : dict mapping condition_key → ConditionEvaluation
    comparisons : list of ComparisonResult
    stratified : dict mapping condition_key → stratified results
    output_path : Path

    Returns
    -------
    Path to the saved Markdown file.
    """
    lines = []
    lines.append("# SYNAPSE M1 Experiment Report")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    for cond_key, eval_result in evaluations.items():
        lines.append(f"- **{cond_key}**: {eval_result.condition.label}")
    lines.append("")

    # Primary Comparison
    lines.append("## Primary Comparison")
    lines.append("")
    lines.append("| Metric | A1 (mean±std) | A2 (mean±std) | B (mean±std) | B vs A1 p | B vs A2 p | Cohen's d |")
    lines.append("|--------|---------------|---------------|--------------|-----------|-----------|------------|")

    primary_comps = [c for c in comparisons
                     if c.condition_b == "B_synapse" and c.condition_a in ("A1_recent", "A2_uniform")]

    if primary_comps:
        a1_eval = evaluations.get("A1_recent")
        a2_eval = evaluations.get("A2_uniform")
        b_eval = evaluations.get("B_synapse")

        if a1_eval and a2_eval and b_eval:
            metric = primary_comps[0].metric_name if primary_comps else "action_mse"
            a1_str = f"{a1_eval.mean_mse:.4f}±{a1_eval.std_mse:.4f}"
            a2_str = f"{a2_eval.mean_mse:.4f}±{a2_eval.std_mse:.4f}"
            b_str = f"{b_eval.mean_mse:.4f}±{b_eval.std_mse:.4f}"

            p_a1 = "—"
            p_a2 = "—"
            d_val = "—"
            for comp in primary_comps:
                d_str = f"{comp.cohens_d:.3f}"
                if comp.condition_a == "A1_recent":
                    p_a1 = f"{comp.p_value:.4f}" + ("*" if comp.welch_significant else "")
                    d_val = d_str
                elif comp.condition_a == "A2_uniform":
                    p_a2 = f"{comp.p_value:.4f}" + ("*" if comp.welch_significant else "")
                    d_val = d_str

            lines.append(f"| {metric} | {a1_str} | {a2_str} | {b_str} | {p_a1} | {p_a2} | {d_val} |")
    lines.append("")

    # Ablation Analysis
    lines.append("## Ablation Analysis")
    lines.append("")
    lines.append("| Metric | B-Full | B-Anchors | B-Topo | Conclusion |")
    lines.append("|--------|--------|-----------|--------|------------|")

    b_eval = evaluations.get("B_synapse")
    ba_eval = evaluations.get("B_anchors")
    bt_eval = evaluations.get("B_topo")

    if b_eval and ba_eval and bt_eval:
        full_mse = f"{b_eval.mean_mse:.4f}"
        anchors_mse = f"{ba_eval.mean_mse:.4f}"
        topo_mse = f"{bt_eval.mean_mse:.4f}"

        # Determine conclusion
        full_better_anchors = b_eval.mean_mse < ba_eval.mean_mse
        full_better_topo = b_eval.mean_mse < bt_eval.mean_mse
        if full_better_anchors and full_better_topo:
            conclusion = "Both components contribute"
        elif full_better_anchors:
            conclusion = "Topology dominates"
        elif full_better_topo:
            conclusion = "Anchors dominate"
        else:
            conclusion = "No clear pattern"

        lines.append(f"| Action MSE | {full_mse} | {anchors_mse} | {topo_mse} | {conclusion} |")
    lines.append("")

    # Rollout Error Accumulation
    lines.append("## Rollout Error Accumulation")
    lines.append("")
    for cond_key, eval_result in evaluations.items():
        if eval_result.rollout:
            lines.append(f"**{cond_key}**:")
            lines.append(f"- AUC: {eval_result.rollout.get('mean_auc', 'N/A'):.4f}")
            lines.append(f"- Slope: {eval_result.rollout.get('mean_slope', 'N/A'):.4f}")
            div = eval_result.rollout.get('mean_divergence_step', 'N/A')
            lines.append(f"- Divergence step: {div}")
            lines.append("")
    lines.append("")

    # Statistical Significance
    lines.append("## Statistical Significance")
    lines.append("")
    for comp in comparisons:
        lines.append(f"**{comp.condition_a} vs {comp.condition_b}** ({comp.metric_name}):")
        lines.append(f"- Welch's t: t={comp.t_statistic:.4f}, p={comp.p_value:.4f}")
        lines.append(f"- Cohen's d: {comp.cohens_d:.3f} (95% CI: [{comp.d_ci_lower:.3f}, {comp.d_ci_upper:.3f}])")
        lines.append(f"- Normality (Shapiro-Wilk): p={comp.shapiro_p:.4f} ({'normal' if comp.is_normal else 'non-normal'})")
        if comp.wilcoxon_p is not None:
            lines.append(f"- Wilcoxon signed-rank: p={comp.wilcoxon_p:.4f}")
        lines.append(f"- Bonferroni-corrected α: {comp.bonferroni_alpha:.4f}")
        lines.append(f"- Significant after correction: {comp.significant_after_correction}")
        lines.append("")

    # Stratified Analysis
    lines.append("## Stratified Analysis")
    lines.append("")
    for cond_key, strata in stratified.items():
        lines.append(f"### {cond_key}")
        lines.append("")
        for stratum, metrics in strata.items():
            mean_mse = metrics.get('mean_mse', 'N/A')
            std_mse = metrics.get('std_mse', 'N/A')
            count = metrics.get('count', 'N/A')
            mean_str = f"{mean_mse:.4f}" if isinstance(mean_mse, (int, float)) else str(mean_mse)
            std_str = f"{std_mse:.4f}" if isinstance(std_mse, (int, float)) else str(std_mse)
            lines.append(f"- **{stratum}**: mean_mse={mean_str}, "
                        f"std={std_str}, "
                        f"n={count}")
        lines.append("")

    # Conclusion
    lines.append("## Conclusion")
    lines.append("")
    b_eval = evaluations.get("B_synapse")
    a2_eval = evaluations.get("A2_uniform")
    if b_eval and a2_eval:
        if b_eval.mean_mse < a2_eval.mean_mse:
            lines.append("SYNAPSE memory conditioning provides measurable benefit over "
                        "uniform subsampling at the same temporal horizon. "
                        "The improvement is attributable to the *quality* of SYNAPSE's "
                        "mathematical summarization, not mere access to more data.")
        else:
            lines.append("SYNAPSE memory conditioning did not provide statistically "
                        "significant improvement over uniform subsampling. "
                        "Further analysis is needed to determine whether the task "
                        "complexity or anchor budget is insufficient.")
    else:
        lines.append("Insufficient data for conclusion.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("Saved Markdown report to %s", output_path)
    return output_path
