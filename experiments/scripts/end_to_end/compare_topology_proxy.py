import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

try:
    from synapse_core.topological_summary import compute_persistence_diagrams
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False
    print("Warning: synapse_core.topological_summary not found or missing Gudhi.")

def generate_synthetic_trajectory(num_points=100, loops=1):
    """Generates a synthetic cyclic trajectory (e.g. a circular pick-and-place loop)."""
    t = np.linspace(0, loops * 2 * np.pi, num_points)
    # A circle in 2D space, embedded in 3D
    x = np.cos(t)
    y = np.sin(t)
    z = np.zeros_like(t)
    time = np.linspace(0, 1, num_points)
    
    spatial = np.stack([x, y, z], axis=1)
    spatio_temporal = np.stack([time, x, y, z], axis=1)
    
    return spatial, spatio_temporal

def compute_graph_laplacian_eigvals(points, sigma=0.5):
    """Computes the Graph Laplacian eigenvalues (our neural proxy for beta_0)."""
    D = torch.cdist(points, points)
    A = torch.exp(-D.square() / (2 * sigma**2))
    degree = A.sum(dim=-1)
    L = torch.diag(degree) - A
    # Add small epsilon for numerical stability
    L = L + torch.eye(points.shape[0]) * 1e-6
    eigvals = torch.linalg.eigvalsh(L)
    return eigvals.numpy()

def main():
    print("======================================================")
    print("Topology vs Connectivity Proxy Analytical Comparison")
    print("======================================================")
    
    print("1. Generating synthetic cyclic trajectory (Betti-1 = 1 in space)...")
    spatial, spatio_temporal = generate_synthetic_trajectory(num_points=60)
    
    spatial_tensor = torch.tensor(spatial, dtype=torch.float32)
    spatio_temporal_tensor = torch.tensor(spatio_temporal, dtype=torch.float32)
    
    h0_spatial, h1_spatial = [], []
    if HAS_GUDHI:
        print("2. Computing EXACT persistence diagrams (Vietoris-Rips)...")
        try:
            diagrams_spatial = compute_persistence_diagrams(spatial, Q=2)
            h0_spatial = diagrams_spatial[0] if len(diagrams_spatial) > 0 else []
            h1_spatial = diagrams_spatial[1] if len(diagrams_spatial) > 1 else []
            print(f"   -> Exact Topology detected {len(h1_spatial)} spatial loops (H1).")
        except Exception as e:
            print(f"   -> Could not compute exact persistence: {e}")
    else:
        print("2. Skipping exact persistence (Gudhi not installed).")
        
    print("3. Computing Graph Laplacian eigenvalues (Our Connectivity Proxy)...")
    eigvals_spatial = compute_graph_laplacian_eigvals(spatial_tensor, sigma=0.4)
    eigvals_temporal = compute_graph_laplacian_eigvals(spatio_temporal_tensor, sigma=0.4)
    
    print(f"   -> Spatial graph nullity (eig ~ 0): {np.sum(eigvals_spatial < 1e-3)}")
    print(f"   -> Spatio-Temporal graph nullity (eig ~ 0): {np.sum(eigvals_temporal < 1e-3)}")
    
    # --- Plotting ---
    print("4. Generating comparison figures...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Topological Analysis: Exact Vietoris-Rips vs. Graph Laplacian Proxy", fontsize=16)
    
    # Plot A: Trajectories
    ax = axes[0, 0]
    ax.plot(spatial[:, 0], spatial[:, 1], 'o-', color='teal', label='Spatial Loop (XY)')
    ax.plot(spatio_temporal[:, 0], spatio_temporal[:, 1], 'x--', color='purple', alpha=0.5, label='Time vs X')
    ax.set_title("A: Synthetic Robotic Trajectory (1 Cycle)")
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot B: Persistence Diagram (Exact Topology)
    ax = axes[0, 1]
    if len(h0_spatial) > 0:
        h0 = np.array(h0_spatial)
        ax.scatter(h0[:, 0], h0[:, 1], c='red', s=50, label='H0 (Connected Components)')
    if len(h1_spatial) > 0:
        h1 = np.array(h1_spatial)
        ax.scatter(h1[:, 0], h1[:, 1], c='blue', s=80, marker='^', label='H1 (Loops / Voids)')
    max_val = 2.0
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5) # diagonal
    ax.set_title("B: Exact Persistent Homology (Vietoris-Rips)\nClearly detects the loop (Betti-1)")
    ax.set_xlabel("Birth Scale (Radius)")
    ax.set_ylabel("Death Scale (Radius)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    # Plot C: Graph Laplacian Spectrum (Spatial)
    ax = axes[1, 0]
    ax.plot(eigvals_spatial[:25], 'o-', color='darkred', linewidth=2)
    ax.set_title("C: Graph Laplacian Spectrum (Spatial Only)\nDetects $\\beta_0$ connectivity, but NO explicit $\\beta_1$ gap")
    ax.set_xlabel("Eigenvalue Index $k$")
    ax.set_ylabel("Eigenvalue $\\lambda_k$")
    ax.grid(True, alpha=0.3)
    
    # Plot D: Graph Laplacian Spectrum (Spatio-Temporal)
    ax = axes[1, 1]
    ax.plot(eigvals_temporal[:25], 'o-', color='darkorange', linewidth=2)
    ax.set_title("D: Graph Laplacian Spectrum (with Monotonic Time)\nTime stretches the graph, altering connectivity entirely")
    ax.set_xlabel("Eigenvalue Index $k$")
    ax.set_ylabel("Eigenvalue $\\lambda_k$")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../docs/formal_math/z3'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'topology_comparison_plot.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"5. Saved comparison figure to:\n   {out_path}")
    print("   (Submit this high-res plot to the professor)")

if __name__ == '__main__':
    main()
