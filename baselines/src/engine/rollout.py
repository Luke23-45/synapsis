"""
Autoregressive Rollout Evaluation — Professor Feedback #6
=========================================================

Open-loop MSE measures prediction quality from perfect ground-truth
history. Real robots suffer from compounding errors: a tiny mistake at
step 1 puts the robot in an unfamiliar state at step 2, leading to
escalating errors. A superior memory system should slow this compounding.

This module implements autoregressive rollout evaluation:
    1. Predict action from current (potentially corrupted) history
    2. Append predicted action to history (simulating execution)
    3. Measure MSE against ground-truth action at that step
    4. Return MSE at each rollout step → error accumulation curve

Rollout metrics:
    - rollout_mse_curve: MSE at each rollout step (1..N) per condition
    - rollout_area_under_curve: Integral of MSE curve (single summary)
    - rollout_divergence_step: First step where MSE exceeds 2× open-loop MSE
    - rollout_slope: Linear regression slope of MSE curve (compounding rate)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.planner.planner import RoboticsPlannerBase
from src.core.config import ExperimentConfig

log = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """Results from autoregressive rollout evaluation.

    Attributes
    ----------
    mse_per_step : np.ndarray, shape (N,)
        MSE at each rollout step.
    area_under_curve : float
        Integral of the MSE curve.
    divergence_step : int or None
        First step where MSE exceeds 2× open-loop MSE.
        None if divergence never occurs.
    slope : float
        Linear regression slope of the MSE curve (compounding rate).
    open_loop_mse : float
        Baseline MSE from ground-truth history (for reference).
    """

    mse_per_step: np.ndarray
    area_under_curve: float
    divergence_step: Optional[int]
    slope: float
    open_loop_mse: float


def plot_persistence_diagram(diagrams, output_path: Path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    colors = ['blue', 'red', 'green', 'orange']
    for q, dgm in enumerate(diagrams):
        if len(dgm) == 0: continue
        dgm = np.array(dgm)
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        finite_deaths = deaths[np.isfinite(deaths)]
        max_death = np.max(finite_deaths) if len(finite_deaths) > 0 else np.max(births) + 1.0
        inf_replacement = max_death * 1.1
        plot_deaths = np.where(np.isfinite(deaths), deaths, inf_replacement)
        plt.scatter(births, plot_deaths, color=colors[q % len(colors)], label=f'H{q}', alpha=0.7)
    ax = plt.gca()
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    plt.plot(lims, lims, 'k-', alpha=0.3, zorder=0)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title("Persistence Diagram")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def plot_point_cloud_3d(point_cloud: np.ndarray, anchor_times: np.ndarray, output_path: Path):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    if point_cloud.shape[0] < 3: return
    pca = PCA(n_components=min(3, point_cloud.shape[1]))
    points_3d = pca.fit_transform(point_cloud)
    fig = plt.figure(figsize=(8, 6))
    if points_3d.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=anchor_times, cmap='viridis', s=50)
        ax.plot(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 'gray', alpha=0.5)
    else:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], c=anchor_times, cmap='viridis', s=50)
        ax.plot(points_3d[:, 0], points_3d[:, 1], 'gray', alpha=0.5)
    fig.colorbar(scatter, label="Normalized Time")
    plt.savefig(output_path)
    plt.close()

def rollout_evaluate(
    model: RoboticsPlannerBase,
    initial_batch: Dict[str, torch.Tensor],
    ground_truth_actions: torch.Tensor,
    n_steps: int = 10,
    action_to_proprio_projection: Optional[torch.nn.Linear] = None,
    dump_topology_dir: Optional[Path] = None,
) -> RolloutResult:
    """Autoregressive rollout: feed predictions back as input.

    At each step:
        1. Predict action from current (potentially corrupted) history
        2. Append predicted action to history (simulating execution)
        3. Measure MSE against ground-truth action at that step

    Parameters
    ----------
    model : RoboticsPlannerBase
        Trained planner model.
    initial_batch : dict
        Initial batch dictionary with keys:
            proprio, proprio_history, (synapse_anchors, synapse_topo if SYNAPSE)
    ground_truth_actions : torch.Tensor, shape (N, action_chunk_size, action_dim)
        Ground-truth actions for each rollout step.
    n_steps : int
        Number of rollout steps.
    action_to_proprio_projection : nn.Linear, optional
        Projection from action space to proprio space for history update.
        If None, a simple truncation/padding is used.
    dump_topology_dir : Path, optional
        If provided, dumps topology geometry for each step to this directory.

    Returns
    -------
    RolloutResult
    """
    device = next(model.parameters()).device
    model.eval()
    config = getattr(model, "config", None)

    uses_synapse = bool(
        config is not None
        and hasattr(model, "forward_deploy")
        and config.uses_end_to_end_synapse
    )

    history = initial_batch.get("proprio_history")
    if history is not None:
        history = history.clone().to(device)
    structured_history = initial_batch.get("structured_history")
    if structured_history is not None:
        structured_history = structured_history.clone().to(device)
    structured_state = initial_batch.get("structured_state")
    if structured_state is not None:
        structured_state = structured_state.clone().to(device)

    if uses_synapse:
        if structured_history is None or structured_state is None:
            raise KeyError("SYNAPSE rollout requires 'structured_history' and 'structured_state'.")
        B, _, D = structured_history.shape
        history_lengths = initial_batch.get("history_lengths")
        if history_lengths is None:
            history_lengths = initial_batch.get("history_length")
        if history_lengths is None:
            history_lengths = torch.full(
                (B,),
                structured_history.shape[1],
                dtype=torch.long,
                device=device,
            )
        else:
            history_lengths = history_lengths.clone().to(device)
    else:
        if history is None:
            raise KeyError("Rollout requires 'proprio_history'.")
        B, _, D = history.shape

    # SYNAPSE features (if applicable)
    synapse_anchors = initial_batch.get("synapse_anchors")
    synapse_topo = initial_batch.get("synapse_topo")

    # Compute open-loop MSE (baseline from perfect history)
    with torch.no_grad():
        open_loop_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in initial_batch.items()}
        if uses_synapse:
            open_loop_pred = model.forward_deploy(open_loop_batch).pred_actions
        else:
            open_loop_pred = model(open_loop_batch)
        open_loop_mse = F.mse_loss(
            open_loop_pred[:, 0, :], ground_truth_actions[:, 0, :].to(device)
        ).item()

    # Create action-to-proprio projection if not provided
    if action_to_proprio_projection is None:
        action_dim = ground_truth_actions.shape[-1]
        action_to_proprio_projection = torch.nn.Linear(
            action_dim, D, bias=False
        ).to(device)
        # Initialize as approximate identity on overlapping dimensions
        with torch.no_grad():
            min_dim = min(action_dim, D)
            action_to_proprio_projection.weight.data[:min_dim, :min_dim] = (
                torch.eye(min_dim, device=device) * 0.1
            )

    mse_per_step = []

    for step in range(n_steps):
        with torch.no_grad():
            # Construct batch from current (potentially corrupted) history
            if uses_synapse:
                max_len = structured_history.shape[1]
                history_mask = (
                    torch.arange(max_len, device=device).unsqueeze(0)
                    < history_lengths.unsqueeze(1)
                )
                batch = {
                    "structured_state": structured_state,
                    "structured_history": structured_history,
                    "history_lengths": history_lengths,
                    "history_mask": history_mask,
                }
            else:
                batch = {
                    "proprio": history[:, -1, :],
                    "proprio_history": history,
                    "structured_state": history[:, -1, :],
                    "structured_history": history,
                }
            if synapse_anchors is not None:
                batch["synapse_anchors"] = synapse_anchors.to(device)
                batch["synapse_topo"] = synapse_topo.to(device)

            # Predict action
            if uses_synapse:
                deploy_out = model.forward_deploy(batch)
                pred_action = deploy_out.pred_actions
                
                if dump_topology_dir is not None and len(deploy_out.exact_memory_states) > 0:
                    try:
                        step_dir = Path(dump_topology_dir) / f"step_{step:02d}"
                        step_dir.mkdir(parents=True, exist_ok=True)
                        memory_state = deploy_out.exact_memory_states[0]
                        np.savez(
                            step_dir / "topology_data.npz",
                            point_cloud=memory_state.point_cloud,
                            anchor_indices=np.array(memory_state.anchor_indices),
                            y_star=memory_state.y_star,
                            topology_summary=memory_state.topology_summary,
                        )
                        plot_persistence_diagram(memory_state.persistence_diagrams, step_dir / "persistence_diagram.png")
                        seq_len = int(history_lengths[0].item())
                        anchor_times = np.array(memory_state.anchor_indices) / max(seq_len, 1)
                        plot_point_cloud_3d(memory_state.point_cloud, anchor_times, step_dir / "point_cloud_pca.png")
                    except Exception as e:
                        log.warning(f"Failed to dump topology: {e}")
            else:
                pred_action = model(batch)

            # Measure MSE against ground truth for first action in chunk
            if step < ground_truth_actions.shape[1]:
                gt = ground_truth_actions[:, step, :].to(device)
                mse = F.mse_loss(pred_action[:, 0, :], gt).item()
            else:
                # Beyond GT horizon: use self-comparison as proxy
                # (MSE of prediction against itself at this step)
                mse = mse_per_step[-1] if mse_per_step else 0.0

            mse_per_step.append(mse)

            # Update history: append predicted proprioception
            # Use the first predicted action to simulate next proprio state
            next_state = action_to_proprio_projection(pred_action[:, 0, :]).unsqueeze(1)
            if uses_synapse:
                structured_history = torch.cat([structured_history, next_state], dim=1)
                structured_state = next_state[:, 0, :]
                history_lengths = history_lengths + 1
            else:
                history = torch.cat([history, next_state], dim=1)

    mse_curve = np.array(mse_per_step, dtype=np.float64)

    # Replace NaN values (from steps beyond GT horizon) with last valid MSE
    nan_mask = np.isnan(mse_curve)
    if nan_mask.any() and not nan_mask.all():
        last_valid = mse_curve[~nan_mask][-1]
        mse_curve[nan_mask] = last_valid
    elif nan_mask.all():
        mse_curve = np.zeros(len(mse_curve), dtype=np.float64)

    # Compute summary metrics
    area_under_curve = float(np.trapezoid(mse_curve)) if len(mse_curve) > 0 else 0.0

    # Divergence step: first step where MSE > 2× open-loop MSE
    divergence_step = None
    if open_loop_mse > 0:
        threshold = 2.0 * open_loop_mse
        for i, mse_val in enumerate(mse_per_step):
            if mse_val > threshold:
                divergence_step = i
                break

    # Slope: linear regression on MSE curve
    slope = 0.0
    if len(mse_curve) >= 2:
        x = np.arange(len(mse_curve), dtype=np.float64)
        slope = float(np.polyfit(x, mse_curve, 1)[0])

    return RolloutResult(
        mse_per_step=mse_curve,
        area_under_curve=area_under_curve,
        divergence_step=divergence_step,
        slope=slope,
        open_loop_mse=open_loop_mse,
    )

def rollout_evaluate_dataset(
    model: RoboticsPlannerBase,
    dataloader: torch.utils.data.DataLoader,
    config: ExperimentConfig,
    n_steps: int = 10,
    max_episodes: int = 50,
    dump_topology_dir: Optional[Path] = None,
) -> Dict[str, RolloutResult]:
    """Run rollout evaluation across a dataset, aggregating per-episode results.

    Parameters
    ----------
    model : RoboticsPlannerBase
    dataloader : DataLoader
    config : ExperimentConfig
    n_steps : int
    max_episodes : int
        Maximum episodes to evaluate (for compute budget).
    dump_topology_dir : Path, optional
        If provided, dumps topology geometry for the *first* episode to this directory.

    Returns
    -------
    dict mapping episode_id → RolloutResult
    """
    device = next(model.parameters()).device
    model.eval()

    results = {}

    with torch.no_grad():
        for batch in dataloader:
            # We track valid unique episodes inside `results` directly
            if len(results) >= max_episodes:
                break

            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Get ground truth actions for rollout comparison
            gt_actions = batch_device["action_chunk"]  # (B, K_act, D_act)

            # Run rollout for each sample in batch
            B = gt_actions.shape[0]
            for b in range(B):
                if len(results) >= max_episodes:
                    break
                
                # Check for existing episode id to prevent evaluating/overwriting same-episode samples
                if "episode_idx" in batch_device:
                    # Safely handles both scalar items and 1D tensors safely
                    ep_val = batch_device["episode_idx"][b].item() if batch_device["episode_idx"].dim() > 0 else batch_device["episode_idx"].item()
                    ep_id = f"episode_{int(ep_val):04d}"
                else:
                    # Fallback to counts if the dataset pipeline strips episode context
                    ep_id = f"sample_{len(results):04d}"

                # CRITICAL FIX: Only evaluate one rollout sequence per unique episode
                if ep_id in results:
                    continue

                # Create single-sample batch
                single_batch = {
                    k: v[b:b+1] if isinstance(v, torch.Tensor) else v
                    for k, v in batch_device.items()
                }
                
                # Only dump topology for the very first episode to save space/time
                current_dump_dir = None
                if dump_topology_dir is not None and len(results) == 0:
                    current_dump_dir = dump_topology_dir / ep_id

                result = rollout_evaluate(
                    model,
                    single_batch,
                    gt_actions[b:b+1],
                    n_steps=n_steps,
                    dump_topology_dir=current_dump_dir,
                )

                results[ep_id] = result

    log.info(
        "Rollout evaluation: %d episodes, %d steps, condition=%s",
        len(results),
        n_steps,
        config.condition.value,
    )

    return results


def aggregate_rollout_results(
    results: Dict[str, RolloutResult],
) -> Dict[str, float]:
    """Aggregate rollout results across episodes.

    Returns
    -------
    dict with keys:
        mean_auc, std_auc,
        mean_slope, std_slope,
        mean_divergence_step, std_divergence_step,
        mean_open_loop_mse, std_open_loop_mse,
        mean_mse_per_step (array)
    """
    if not results:
        return {}

    aucs = [r.area_under_curve for r in results.values()]
    slopes = [r.slope for r in results.values()]
    div_steps = [r.divergence_step for r in results.values()
                 if r.divergence_step is not None]
    open_loop = [r.open_loop_mse for r in results.values()]

    # Average MSE curves (may have different lengths)
    curves = [r.mse_per_step for r in results.values()]
    max_len = max(len(c) for c in curves)
    padded = np.zeros((len(curves), max_len))
    for i, c in enumerate(curves):
        padded[i, :len(c)] = c
    mean_curve = padded.mean(axis=0)
    std_curve = padded.std(axis=0)

    return {
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "mean_slope": float(np.mean(slopes)),
        "std_slope": float(np.std(slopes)),
        "mean_divergence_step": float(np.mean(div_steps)) if div_steps else float("inf"),
        "std_divergence_step": float(np.std(div_steps)) if len(div_steps) > 1 else 0.0,
        "mean_open_loop_mse": float(np.mean(open_loop)),
        "std_open_loop_mse": float(np.std(open_loop)),
        "mean_mse_per_step": mean_curve.tolist(),
        "std_mse_per_step": std_curve.tolist(),
    }
