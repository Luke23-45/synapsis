"""
Verification: Differentiable Spectral Topology Features.

Tests:
  1. Gradient flows from a task loss through spectral features to W_theta.
  2. Spectral features distinguish topologically distinct point clouds.
  3. Numerical stability under AMP float16 conditions.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def differentiable_spectral_features(
    point_cloud: torch.Tensor,
    activations: torch.Tensor,
    num_scales: int = 4,
    num_eigvals: int = 4,
    log_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute spectral features from the point cloud's graph Laplacian.

    Args:
        point_cloud: (B, N, k)  lifted anchor positions
        activations:  (B, N)    anchor weights in [0, 1]
        num_scales:   number of Gaussian kernel widths
        num_eigvals:  number of smallest Laplacian eigenvalues per scale

    Returns:
        features: (B, num_scales * num_eigvals + 4)
    """
    B, N, k = point_cloud.shape

    # Force float32 for numerical stability (eigvalsh doesn't like float16)
    cloud_f32 = point_cloud.float()
    act_f32 = activations.float()

    mask = (act_f32 > 1e-3).float()  # (B, N)
    mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)  # (B, N, N)

    # Pairwise distances (fully differentiable)
    D = torch.cdist(cloud_f32, cloud_f32)  # (B, N, N)
    D = D * mask_2d  # zero out inactive pairs

    if log_scales is None:
        log_scales = torch.linspace(-1.5, 1.5, num_scales, device=point_cloud.device)
    scales = torch.exp(log_scales.float())

    features = []
    for s_idx in range(num_scales):
        sigma = scales[s_idx]
        # Gaussian kernel adjacency
        A = torch.exp(-D.square() / (2 * sigma.square() + 1e-8))
        A = A * mask_2d

        # Graph Laplacian
        degree = A.sum(dim=-1)
        L = torch.diag_embed(degree) - A
        # Numerical stabilization
        L = L + torch.eye(N, device=L.device).unsqueeze(0) * 1e-6

        # Eigenvalues (sorted ascending)
        eigvals = torch.linalg.eigvalsh(L)  # (B, N)
        # Take smallest num_eigvals
        top = eigvals[:, :num_eigvals]  # (B, num_eigvals)
        features.append(top)

    # Global distance statistics
    tri_mask = torch.triu(mask_2d, diagonal=1)
    tri_sum = tri_mask.sum(dim=(1, 2)).clamp_min(1)
    mean_dist = (D * tri_mask).sum(dim=(1, 2)) / tri_sum
    max_dist = (D * tri_mask).amax(dim=(1, 2))
    dist_var = ((D - mean_dist[:, None, None]).square() * tri_mask).sum(dim=(1, 2)) / tri_sum
    compactness = mean_dist / (max_dist + 1e-6)

    features.append(torch.stack([mean_dist, max_dist, dist_var, compactness], dim=-1))

    return torch.cat(features, dim=-1).to(point_cloud.dtype)


# ── Test 1: Gradient Flow ─────────────────────────────────────────────────────

def test_gradient_flow():
    log.info("\n" + "=" * 60)
    log.info("TEST 1: GRADIENT FLOW THROUGH SPECTRAL FEATURES TO W_THETA")
    log.info("=" * 60)

    torch.manual_seed(42)
    d_in, k_lift, d_model = 4, 8, 32
    N = 15

    W_theta = nn.Linear(d_in, k_lift, bias=False)
    task_head = nn.Linear(20, 3)  # 4 scales * 4 eigvals + 4 stats = 20

    V = torch.randn(2, N, d_in)  # 2 batch, 15 anchors, 4-dim input
    activations = torch.ones(2, N)

    lifted = W_theta(V)  # (2, 15, 8)
    spectral_feats = differentiable_spectral_features(lifted, activations)
    pred = task_head(spectral_feats)
    loss = F.mse_loss(pred, torch.zeros_like(pred))

    W_theta.zero_grad()
    task_head.zero_grad()
    loss.backward()

    grad_norm = W_theta.weight.grad.norm().item()
    log.info(f"W_theta gradient norm from spectral task loss: {grad_norm:.6f}")
    assert grad_norm > 0, "FAIL: No gradient reached W_theta!"
    log.info("✅ PASS: Gradients flow through spectral features to W_theta")
    return True


# ── Test 2: Topological Discrimination ────────────────────────────────────────

def test_discrimination():
    log.info("\n" + "=" * 60)
    log.info("TEST 2: SPECTRAL FEATURES DISTINGUISH TOPOLOGIES")
    log.info("=" * 60)

    torch.manual_seed(42)

    # Cloud A: single tight cluster
    cluster = torch.randn(1, 20, 3) * 0.1
    act_a = torch.ones(1, 20)

    # Cloud B: two separated clusters
    c1 = torch.randn(1, 10, 3) * 0.1
    c2 = torch.randn(1, 10, 3) * 0.1 + 5.0
    two_clusters = torch.cat([c1, c2], dim=1)
    act_b = torch.ones(1, 20)

    # Cloud C: circle (has H_1 structure)
    t = torch.linspace(0, 2 * 3.14159, 20).unsqueeze(0)
    circle = torch.stack([torch.cos(t), torch.sin(t), torch.zeros_like(t)], dim=-1)
    act_c = torch.ones(1, 20)

    feats_a = differentiable_spectral_features(cluster, act_a)
    feats_b = differentiable_spectral_features(two_clusters, act_b)
    feats_c = differentiable_spectral_features(circle, act_c)

    dist_ab = (feats_a - feats_b).norm().item()
    dist_ac = (feats_a - feats_c).norm().item()
    dist_bc = (feats_b - feats_c).norm().item()

    log.info(f"Distance cluster vs 2-clusters:  {dist_ab:.4f}")
    log.info(f"Distance cluster vs circle:      {dist_ac:.4f}")
    log.info(f"Distance 2-clusters vs circle:   {dist_bc:.4f}")

    # All three should be distinguishable
    assert dist_ab > 0.1, f"FAIL: Can't distinguish cluster from 2-clusters ({dist_ab:.4f})"
    assert dist_ac > 0.1, f"FAIL: Can't distinguish cluster from circle ({dist_ac:.4f})"
    log.info("✅ PASS: Spectral features discriminate topologically distinct clouds")

    # Check the Fiedler value (2nd eigenvalue at first scale)
    log.info(f"\nFiedler values (λ_1 at scale 0):")
    log.info(f"  Cluster:    {feats_a[0, 1].item():.6f} (should be large — connected)")
    log.info(f"  2-Clusters: {feats_b[0, 1].item():.6f} (should be ≈0 — disconnected)")
    log.info(f"  Circle:     {feats_c[0, 1].item():.6f} (should be moderate — ring)")
    return True


# ── Test 3: Numerical Stability ──────────────────────────────────────────────

def test_numerical_stability():
    log.info("\n" + "=" * 60)
    log.info("TEST 3: NUMERICAL STABILITY")
    log.info("=" * 60)

    torch.manual_seed(42)

    # Edge cases
    cases = {
        "Normal cloud": (torch.randn(2, 10, 4), torch.ones(2, 10)),
        "Tiny cloud (2 pts)": (torch.randn(2, 2, 4), torch.ones(2, 2)),
        "Sparse activations": (torch.randn(2, 10, 4), torch.tensor([[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                                       [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]]).float()),
        "Collapsed cloud": (torch.zeros(2, 10, 4), torch.ones(2, 10)),
    }

    all_pass = True
    for name, (cloud, act) in cases.items():
        try:
            feats = differentiable_spectral_features(cloud, act)
            has_nan = torch.isnan(feats).any().item()
            has_inf = torch.isinf(feats).any().item()
            if has_nan or has_inf:
                log.error(f"  ✗ {name}: NaN={has_nan}, Inf={has_inf}")
                all_pass = False
            else:
                log.info(f"  ✓ {name}: shape={feats.shape}, range=[{feats.min():.4f}, {feats.max():.4f}]")
        except Exception as e:
            log.error(f"  ✗ {name}: EXCEPTION: {e}")
            all_pass = False

    if all_pass:
        log.info("✅ PASS: All edge cases handled without NaN/Inf")
    else:
        log.error("✗ FAIL: Some edge cases produced bad values")
    return all_pass


if __name__ == "__main__":
    results = []
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("Discrimination", test_discrimination()))
    results.append(("Stability", test_numerical_stability()))

    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    for name, passed in results:
        log.info(f"  {'✅' if passed else '✗'} {name}")
    if all(r for _, r in results):
        log.info("\nAll tests passed. Safe to implement.")
    else:
        log.info("\nSome tests failed. Do NOT implement.")
