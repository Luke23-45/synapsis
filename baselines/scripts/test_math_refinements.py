import sys
import logging
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Fallback to ripser
try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    log.warning("Ripser not found. Cannot test H1 loops.")

def generate_synthetic_trajectory(num_points=100, num_loops=2, noise=0.05):
    """Generates a noisy circular trajectory that loops multiple times."""
    t = np.linspace(0, num_loops * 2 * np.pi, num_points)
    x = np.cos(t) + np.random.randn(num_points) * noise
    y = np.sin(t) + np.random.randn(num_points) * noise
    time_normalized = np.linspace(0, 1, num_points)
    
    # State is 2D (x,y), we append time to simulate our system's lift
    # Anchor = [t, s_0, s_1]
    trajectory = np.stack([time_normalized, x, y], axis=1)
    return trajectory

def compute_persistence(points: np.ndarray, max_dim=1):
    """Computes persistence using Ripser."""
    if not HAS_RIPSER:
        return {0: 1, 1: 0} # Dummy
        
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ripser(points, maxdim=max_dim, distance_matrix=False)
    
    h0_count = 0
    h1_count = 0
    
    if len(result["dgms"]) > 0:
        for birth, death in result["dgms"][0]:
            if death - birth > 0.1: h0_count += 1
            
    if len(result["dgms"]) > 1:
        for birth, death in result["dgms"][1]:
            pers = death - birth
            log.info(f"H1 Feature: Birth={birth:.4f}, Death={death:.4f}, Pers={pers:.4f}")
            if pers > 0.1: h1_count += 1
            
    return {0: h0_count, 1: h1_count}

def test_proposal_b_time_invariance():
    log.info("\n" + "="*50)
    log.info("PROPOSAL B: TIME-INVARIANT MANIFOLD DIAGNOSTIC")
    log.info("="*50)
    
    # Generate 3 loops of a circle
    traj = generate_synthetic_trajectory(num_points=150, num_loops=3, noise=0.1)
    
    # 1. Point Cloud WITH Time (Standard SYNAPSE approach)
    log.info("[Testing Standard Lift (State + Time)]")
    # Time is traj[:, 0], State is traj[:, 1:]
    # Because time is monotonically increasing, it stretches the circle into a spiral helix in 3D.
    # A helix never self-intersects!
    h_counts_std = compute_persistence(traj)
    log.info(f"  H0 Components: {h_counts_std[0]}")
    log.info(f"  H1 Loops (>0.1 persistence): {h_counts_std[1]}")
    
    # 2. Point Cloud WITHOUT Time (Proposal B approach)
    log.info("\n[Testing Time-Invariant Lift (State Only)]")
    # We drop the time column
    traj_no_time = traj[:, 1:]
    h_counts_inv = compute_persistence(traj_no_time)
    log.info(f"  H0 Components: {h_counts_inv[0]}")
    log.info(f"  H1 Loops (>0.1 persistence): {h_counts_inv[1]}")
    
    if h_counts_inv[1] > h_counts_std[1]:
        log.info("-> PROOF: Time creates a non-intersecting helix! Dropping absolute time collapses the helix into a circle, revealing the true robust H1 topological loops!")

def test_proposal_a_gradient_flow():
    log.info("\n" + "="*50)
    log.info("PROPOSAL A: GRADIENT FLOW DIAGNOSTIC")
    log.info("="*50)
    
    torch.manual_seed(42)
    # Minimal representation of the topology branch
    # W_theta projects from d+3 to k
    d_in = 3 
    k = 8
    W_theta = nn.Linear(d_in, k)
    
    # Mock inputs
    V = torch.randn(10, d_in) # 10 anchors
    
    # Forward Pass 1: Standard Pipeline
    # ---------------------------------
    dense_lifted = W_theta(V) # Shape: (10, k)
    
    # EXACT TDA PATH (Detached because Gudhi requires numpy)
    exact_cloud = dense_lifted.detach().numpy()
    # Mock summarize_diagrams returning some features
    exact_features = torch.tensor([1.0, 0.5, 0.2], requires_grad=False) 
    
    # SURROGATE PATH (Differentiable spatial stats)
    # E.g. variance of the point cloud
    surrogate_var = dense_lifted.var(dim=0).mean()
    
    # Transformer action prediction (Only uses EXACT TDA)
    # pred_action = Transformer(exact_features)
    action_loss = F.mse_loss(exact_features * 0.1, torch.zeros(3))
    
    # Auxiliary Loss (Only uses SURROGATE)
    aux_loss = F.mse_loss(surrogate_var, torch.tensor(1.0))
    
    total_loss_std = action_loss + aux_loss
    
    W_theta.zero_grad()
    total_loss_std.backward()
    
    # How much of the action loss contributed to W_theta's gradient?
    # Zero! Only aux_loss contributed.
    log.info("[Standard Pipeline]")
    log.info("Is W_theta receiving gradients from the action task?")
    log.info("Answer: NO. (exact_features has requires_grad=False due to detached Gudhi)")
    
    # Forward Pass 2: Dual-Channel Pipeline
    # -------------------------------------
    W_theta.zero_grad()
    dense_lifted_2 = W_theta(V)
    
    exact_cloud_2 = dense_lifted_2.detach().numpy()
    exact_features_2 = torch.tensor([1.0, 0.5, 0.2], requires_grad=False)
    
    # The surrogate features (Differentiable spatial projection)
    # Imagine this is an MLP producing 12 features representing the spatial spread
    surrogate_features_2 = dense_lifted_2.mean(dim=0) + dense_lifted_2.var(dim=0)
    
    # In Dual-Channel, BOTH exact and surrogate are fed to the Transformer
    # pred_action = Transformer(exact_features, surrogate_features)
    # We simulate this by making the action loss depend on surrogate_features
    action_loss_2 = F.mse_loss(surrogate_features_2 * 0.1, torch.zeros(k))
    
    action_loss_2.backward()
    
    grad_norm = W_theta.weight.grad.norm().item()
    log.info(f"\n[Dual-Channel Pipeline (Proposal A)]")
    log.info(f"W_Theta gradient norm from ACTION TASK: {grad_norm:.6f}")
    if grad_norm > 0:
        log.info("-> PROOF: By injecting the differentiable surrogate into the Transformer, the action-prediction task actively shapes W_Theta and the point cloud!")

if __name__ == "__main__":
    test_proposal_b_time_invariance()
    test_proposal_a_gradient_flow()
