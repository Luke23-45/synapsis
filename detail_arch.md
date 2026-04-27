# SYNAPSE: End-to-End Visuomotor Policy with Differentiable Topological Data Analysis (TDA)

This document formalizes the mathematical and architectural foundations of the **SYNAPSE** (Spatial-Temporal Anchoring and Persistent Shape Encoding) model. This architecture bridges the gap between continuous geometric priors and discrete action-chunked behavioral cloning by explicitly modeling the topological structure of the robot's proprioceptive history.

---

## 1. High-Level Architecture Overview

SYNAPSE operates on a sequence of historical proprioceptive states $X_{1:t} \in \mathbb{R}^{t \times d_{in}}$ to predict a future action chunk $A_{t:t+k} \in \mathbb{R}^{k \times d_{act}}$. 

Instead of passing the entire dense history to a transformer (which suffers from $O(T^2)$ complexity and temporal dilution), SYNAPSE uses a differentiable subset-selection mechanism to extract $K$ salient "Anchor" events. These events are projected into a metric space to form a point cloud, whose geometric and topological properties are explicitly computed via a differentiable Graph Laplacian spectrum. The final policy conditions on the current state, the selected anchors, and the topological shape of the history.

The pipeline consists of five primary stages:
1. **Event Saliency & Normalization**
2. **Relaxed Selection (Differentiable Top-K)**
3. **Geometric Lift & Manifold Projection**
4. **Spectral Topology Encoding**
5. **Transformer Integration & Action Decoding**

---

## 2. Event Saliency & Normalization

The first module evaluates the importance (saliency) of every historical frame.
Let $X \in \mathbb{R}^{t \times d_{in}}$ be the input history. To capture dynamic motion effectively, the **Event Encoder** explicitly computes finite differences. For each time step, it concatenates the current state, the previous state, and the differential:
$$ X^{(diff)}_i = [X_i, X_{i-1}, X_i - X_{i-1}] \in \mathbb{R}^{3 d_{in}} $$
This composite feature vector is passed through a multi-layer perceptron (Linear $\to$ GELU $\to$ Linear $\to$ GELU) and a final linear projection to produce raw saliency logits $S \in \mathbb{R}^t$.

Because downstream optimization requires strict non-negativity and stable gradients, we apply a custom **Saliency Normalizer**. It computes a running mean $\mu_S$ and standard deviation $\sigma_S$ to produce $z$-scores:
$$ z_i = \frac{S_i - \mu_S}{\sigma_S + \epsilon} $$

The final continuous saliency score $\tilde{S}_i$ applies a gated softplus to ensure non-negativity while using a learnable temperature $\tau$:
$$ \tilde{S}_i = \sigma\left(\frac{z_i}{\tau}\right) \odot \text{Softplus}(S_i) $$
where $\sigma(\cdot)$ is the sigmoid function.

---

## 3. Relaxed Selection (Continuous Budget & Refractory Constraints)

Selecting exactly $K$ discrete events from $T$ frames is a non-differentiable combinatorial problem. We formulate it as a convex Quadratic Program (QP) and solve its continuous relaxation on the GPU using Projected Gradient Descent (PGD).

We define an activation vector $y^* \in [0, 1]^t$. The objective is to maximize the gathered saliency minus a quadratic regularization (controlled by $\lambda$):
$$ y^* = \arg\min_{y \in [0, 1]^t} \left( \frac{1}{2} \|y\|^2_2 - \frac{1}{2\lambda} \tilde{S}^T y \right) $$

**Constraints:**
1. **Budget Constraint:** $\sum_{i=1}^t y_i \leq K$ (Select at most $K$ anchors).
2. **Refractory Period Constraint:** $y_i + y_{i+d} \leq 1$ for $d \in [1, r]$. This guarantees that selected anchors are temporally spaced out by at least $r$ frames, preventing the collapse of all $K$ anchors into a single dense bottleneck.
3. **Initial State Constraint:** $y_1 = 0$ (handled via masking).

The PGD solver computes the forward pass in $\sim30$ iterations. During the backward pass, we employ a dampened Straight-Through Estimator (STE) to pass gradients through the saturated constraints back to the Event Encoder.

---

## 4. Geometric Lift & Point Cloud Generation

The selected states must be placed into a metric space to evaluate their topology. We append a normalized temporal coordinate $t/T$ to each state to preserve temporal ordering, forming $V \in \mathbb{R}^{t \times (d_{in}+3)}$.

A parameterized **Normalized Lift** layer shifts and scales the space based on training statistics ($\mu_V, \sigma_V$), followed by a learned affine projection $W_\theta$:
$$ \tilde{v}_i = \frac{v_i - \mu_V}{\sigma_V} $$
$$ p_i = W_\theta \tilde{v}_i \quad \text{for } i \in \{1, \dots, t\} $$

The resulting set $\mathcal{P} = \{p_i \mid y^*_i > \epsilon\}$ forms a dynamic point cloud in $\mathbb{R}^{k}$. 

*Crucial Architecture Detail (Modeling Choice)*: For the topological evaluation, we explicitly zero out the temporal coordinate before the projection ($v_{i, \text{time}} = 0$). While appending time changes the embedding and therefore the filtration geometry, keeping time separate or explicitly excluding it from the point cloud allows us to isolate the purely spatial topology. This is a modeling choice designed to analyze spatial paths without temporal distortion.

---

## 5. Connectivity Regularization & Topological Proxies (The B-Condition)

Traditional exact Topological Data Analysis (TDA) relies on building a **Vietoris–Rips filtration** on the selected anchor cloud and computing persistent homology to obtain a scale-aware topological summary. However, exact simplex reduction (e.g., via the Gudhi library) is non-differentiable and computationally hostile to GPU batching.

If full end-to-end differentiability of higher-order homology ($\beta_1$, $\beta_2$) were required, the mathematically proper proxy would be a **Hodge Laplacian on a simplicial complex**, where the dimension of the kernel of the combinatorial Laplacian $\Delta_k$ equals the $k$-th Betti number $\beta_k = \dim \ker(\Delta_k)$.

To maintain computational feasibility at $O(K^3)$ rather than exponential simplicial complexity, SYNAPSE utilizes a standard **Graph Laplacian** as a connectivity regularizer. We construct a weighted adjacency matrix $A$ based on the Gaussian kernel at multiple learned spatial scales $\sigma_s$:
$$ A_{ij}^{(s)} = \exp\left(-\frac{\|p_i - p_j\|^2}{2\sigma_s^2}\right) \cdot (y^*_i y^*_j) $$

We construct the Graph Laplacian $L^{(s)} = \text{diag}(A^{(s)}\mathbf{1}) - A^{(s)}$. 
The eigenvalues of the graph Laplacian ($\lambda_1 \leq \lambda_2 \leq \dots \leq \lambda_K$) capture essential connectivity and smoothness properties:
- The dimension of the nullspace (the multiplicity of the $0$-eigenvalue) equals the 0th Betti number $\beta_0$ (the number of connected components).
- The spectral gap and distribution of the eigenvalues heavily regularize the spatial dispersion and connectivity of the anchors.

*Note:* A graph Laplacian alone does not recover loops or voids ($\beta_1$, $\beta_2$). Therefore, it is strictly interpreted as a connectivity and spatial regularizer ($\beta_0$ proxy), not a full topological invariant.

We extract the first $m$ eigenvalues using `torch.linalg.eigvalsh(L)`. This decomposition is performed strictly on the compact subset of $K$ activated anchors. These spectral features are passed through an MLP to generate the **Topology Token** $h_{topo} \in \mathbb{R}^{d_{model}}$.

---

## 6. Task Transformer & Action Decoding

The policy conditions the action on three sources of information:
1. $h_{curr} = W_{curr} X_t$ (The instantaneous current state token).
2. $h_{anchor}^{(i)} = W_{anchor} \, p_i \cdot y^*_i$ (The weighted spatial anchors).
3. $h_{topo} = W_{topo} \, (\text{spectral\_features})$ (The global topological summary).

These tokens are concatenated into a sequence $[h_{curr}, h_{anchor}^{(1\dots K)}, h_{topo}]$ and injected with **learnable absolute positional embeddings**. During sparse deployment, because we dynamically collapse a history of length $T$ down to $K$ anchors, the positional embeddings are gathered dynamically using original temporal indices to preserve true temporal spacing. 

This sequence is fed into the `TaskTransformer` (a standard multi-head self-attention encoder with LayerNorm-first architecture and GELU activations). The output token corresponding to $h_{curr}$ (index 0) is routed to the `ActionHead`, which applies a single linear projection mapping $\mathbb{R}^{d_{model}} \to \mathbb{R}^{k \times d_{act}}$ to directly predict the flattened future action chunk $\hat{A}_{t:t+k}$.

---

## 7. Objective Function & Optimization Dynamics

The network is trained end-to-end using a scheduled multi-task objective:
$$ \mathcal{L}_{total} = \mathcal{L}_{action} + \alpha_{sparsity}(e) \mathcal{L}_{sparsity} + \alpha_{topo}(e) \mathcal{L}_{topo} $$

- **$\mathcal{L}_{action}$**: Smooth L1 loss ($\beta=0.5$) between predicted and ground-truth expert action chunks.
- **$\mathcal{L}_{sparsity}$**: An $L_1$ penalty on $y^*$ to force the selector to make decisive, binary-like choices (minimizing entropy).
- **$\mathcal{L}_{topo}$**: A surrogate auxiliary regression loss. Since gradients from the primary action loss can struggle to reshape the manifold entirely on their own, we analytically compute exact geometric moments of the point cloud $\mathcal{P}$ (e.g., mean pairwise distance, maximum spatial dispersion, point-cloud compactness ratio, and spatial variance). The spectral projection branch is penalized via MSE to predict these tangible statistics, anchoring the abstract Laplacian eigenvalues to grounded spatial metrics.

**Auxiliary Ramping:**
To prevent the topological gradients from dominating the action learning early in training, $\alpha_{sparsity}$ and $\alpha_{topo}$ follow a linear schedule:
$$
\alpha(e) = \begin{cases} 
0 & e < E_{start} \\
\alpha_{target} \frac{e - E_{start}}{E_{end} - E_{start}} & E_{start} \leq e \leq E_{end} \\
\alpha_{target} & e > E_{end}
\end{cases}
$$
where $E_{start} = 10$ and $E_{end} = 30$.

## Conclusion

By forcing the network to dynamically select salient events and project them into a manifold whose topological invariants are explicitly optimized, SYNAPSE explicitly disentangles "what happened" (Anchors) from "what shape the task took" (Topology), providing a rigorous geometric foundation for end-to-end visuomotor policies.
