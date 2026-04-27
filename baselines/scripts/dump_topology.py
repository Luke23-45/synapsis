import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.core.config import ExperimentConfig, load_config
from src.data.dataset import RoboticsDataset
from src.planner.planner_synapse import create_planner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def plot_persistence_diagram(diagrams, output_path: Path):
    """Plot persistence diagrams using matplotlib."""
    plt.figure(figsize=(6, 6))
    
    colors = ['blue', 'red', 'green', 'orange']
    for q, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue
        dgm = np.array(dgm)
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        
        # Replace inf with a slightly larger max value for plotting
        finite_deaths = deaths[np.isfinite(deaths)]
        max_death = np.max(finite_deaths) if len(finite_deaths) > 0 else np.max(births) + 1.0
        inf_replacement = max_death * 1.1
        
        plot_deaths = np.where(np.isfinite(deaths), deaths, inf_replacement)
        
        plt.scatter(births, plot_deaths, color=colors[q % len(colors)], label=f'H{q}', alpha=0.7)
    
    # Plot the diagonal
    ax = plt.gca()
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  
        np.max([ax.get_xlim(), ax.get_ylim()]),  
    ]
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
    """Perform PCA to 3D and plot the lifted point cloud with time-based coloring."""
    if point_cloud.shape[0] < 3:
        log.warning("Not enough points for 3D PCA plot.")
        return

    pca = PCA(n_components=min(3, point_cloud.shape[1]))
    points_3d = pca.fit_transform(point_cloud)
    
    fig = plt.figure(figsize=(8, 6))
    
    if points_3d.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=anchor_times, cmap='viridis', s=50)
        ax.plot(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 'gray', alpha=0.5) # Connect the path
        ax.set_title(f"Lifted Point Cloud (PCA 3D)\nExplained Var: {pca.explained_variance_ratio_.sum():.2f}")
    else:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], c=anchor_times, cmap='viridis', s=50)
        ax.plot(points_3d[:, 0], points_3d[:, 1], 'gray', alpha=0.5)
        ax.set_title(f"Lifted Point Cloud (PCA 2D)\nExplained Var: {pca.explained_variance_ratio_.sum():.2f}")

    fig.colorbar(scatter, label="Normalized Time")
    plt.savefig(output_path)
    plt.close()


import sys
from baselines.src.core.config import DatasetSpec
from baselines.src.data.registry import create_adapter
from baselines.src.data.dataset import split_episodes, create_dataloaders
from baselines.src.core.normalization import compute_normalization_stats_from_episodes

def main():
    parser = argparse.ArgumentParser(description="Dump SYNAPSE topological geometry for analysis.")
    parser.add_argument("--config", type=str, default="baselines/configs/experiment/full.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--dataset", type=str, default="xarm_lift")
    parser.add_argument("--output_dir", type=str, default="topo_dump")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Creating model...")
    model = create_planner(config)
    
    log.info(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    log.info(f"Loading dataset {args.dataset}")
    adapter = create_adapter(args.dataset, max_episodes=50) # Just load a few episodes
    episodes = adapter.load_episodes()
    
    train_eps, val_eps, test_eps = split_episodes(
        episodes,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        seed=config.seed,
    )
    
    norm_stats = compute_normalization_stats_from_episodes(
        [{"proprio_history": ep.structured_history} for ep in train_eps]
    )
    
    _, val_loader, _ = create_dataloaders(
        train_eps, val_eps, test_eps, config, norm_stats
    )
    
    # Get one batch
    batch = next(iter(val_loader))
    
    # Move to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    log.info("Running autoregressive rollout on the first validation sequence...")
    
    out_data = {}
    rollout_steps = 10
    
    # Take the first sequence in the batch
    single_batch = {k: v[0:1] for k, v in batch.items()}
    
    # We need a simple action-to-proprio projection just like in rollout.py
    D = single_batch["structured_history"].shape[-1]
    action_dim = config.data.action_dim
    action_to_proprio = torch.nn.Linear(action_dim, D, bias=False).to(device)
    with torch.no_grad():
        min_dim = min(action_dim, D)
        action_to_proprio.weight.data[:min_dim, :min_dim] = (torch.eye(min_dim, device=device) * 0.1)

    with torch.no_grad():
        if hasattr(model, "architecture"):
            arch = model.architecture
        else:
            arch = model
            
        for step in range(rollout_steps):
            log.info(f"--- Rollout Step {step} ---")
            out = arch.forward_deploy(single_batch)
            memory_state = out.exact_memory_states[0]
            
            log.info(f"Number of anchors: {len(memory_state.anchor_indices)}")
            
            # Save data for this step
            step_dir = out_dir / f"step_{step:02d}"
            step_dir.mkdir(exist_ok=True)
            
            npz_path = step_dir / "topology_data.npz"
            np.savez(
                npz_path,
                point_cloud=memory_state.point_cloud,
                anchor_indices=np.array(memory_state.anchor_indices),
                y_star=memory_state.y_star,
                topology_summary=memory_state.topology_summary,
            )
            
            pd_path = step_dir / "persistence_diagram.png"
            plot_persistence_diagram(memory_state.persistence_diagrams, pd_path)
            
            pc_path = step_dir / "point_cloud_pca.png"
            seq_len = int(single_batch["history_length"][0].item())
            anchor_times = np.array(memory_state.anchor_indices) / seq_len
            plot_point_cloud_3d(memory_state.point_cloud, anchor_times, pc_path)
            
            # Update history for next step (Autoregressive)
            pred_action = out.pred_actions[:, 0, :]
            next_state = action_to_proprio(pred_action).unsqueeze(1)
            
            single_batch["structured_history"] = torch.cat([single_batch["structured_history"], next_state], dim=1)
            single_batch["structured_state"] = next_state[:, 0, :]
            single_batch["history_length"] += 1
            single_batch["history_lengths"] = single_batch["history_length"]
            
            max_len = single_batch["structured_history"].shape[1]
            single_batch["history_mask"] = (
                torch.arange(max_len, device=device).unsqueeze(0)
                < single_batch["history_length"].unsqueeze(1)
            )

    log.info("Analysis dump complete.")

if __name__ == "__main__":
    main()
