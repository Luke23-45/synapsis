from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn

from synapse_core.geometric_lift import anchor_vectors, normalize_anchors, apply_lift

from .action_head import ActionHead
from .anchor_builder import AnchorBuilder
from .event_encoder import EventEncoder
from .hard_projector import HardProjector
from .memory_readout import MemoryReadout
from .normalized_lift import NormalizedLift
from .relaxed_selector_layer import RelaxedSelectorLayer
from .saliency_normalizer import SaliencyNormalizer
from .task_transformer import TaskTransformer
from .hodge_branch import HodgeTopologyBranch
from .types import DeployForwardOutput, ExactMemoryState, TrainForwardOutput


@dataclass
class SynapseArchitectureConfig:
    input_dim: int
    action_dim: int
    action_chunk_size: int
    hidden_dim: int
    d_model: int
    num_heads: int
    num_layers: int
    ffn_ratio: int
    dropout: float
    K: int
    r: int
    lam: float
    Q: int
    k: int
    max_history_tokens: int
    bypass_anchor_selection: bool = False


class SynapseEndToEndModel(nn.Module):
    def __init__(self, config: SynapseArchitectureConfig) -> None:
        super().__init__()
        if config.K < 1:
            raise ValueError(f"K must be >= 1, got {config.K}")
        if config.r < 0:
            raise ValueError(f"r must be >= 0, got {config.r}")
        if config.lam <= 0:
            raise ValueError(f"lam must be > 0, got {config.lam}")
        if config.max_history_tokens < 1:
            raise ValueError(
                f"max_history_tokens must be >= 1, got {config.max_history_tokens}"
            )
        self.config = config
        anchor_dim = config.input_dim + 3
        self.current_proj = nn.Linear(config.input_dim, config.d_model)
        self.event_encoder = EventEncoder(config.input_dim, config.hidden_dim)
        self.saliency_normalizer = SaliencyNormalizer()
        self.relaxed_selector = RelaxedSelectorLayer(config.K, config.r, config.lam)
        self.hard_projector = HardProjector(config.K, config.r)
        self.anchor_builder = AnchorBuilder()
        self.normalized_lift = NormalizedLift(anchor_dim, config.k)
        topo_summary_dim = 4 * (config.Q + 1)
        self.topology_branch = HodgeTopologyBranch(config.k, summary_dim=topo_summary_dim, hidden_dim=config.d_model)
        self.memory_readout = MemoryReadout(config.k, config.d_model, topology_dim=config.d_model)
        self.task_transformer = TaskTransformer(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            ffn_ratio=config.ffn_ratio,
            max_tokens=config.max_history_tokens + 2,  # +2: current + anchors + spectral_topo
        )
        self.action_head = ActionHead(config.d_model, config.action_chunk_size, config.action_dim)

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _soft_anchor_vectors(self, structured_history: torch.Tensor, saliency_scores: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = structured_history.shape
        t = torch.linspace(1.0 / max(steps, 1), 1.0, steps, device=structured_history.device, dtype=structured_history.dtype)
        t = t.unsqueeze(0).expand(batch, -1)
        delta = torch.ones_like(t)
        delta[:, 0] = 0.0
        return torch.cat([t.unsqueeze(-1), structured_history, delta.unsqueeze(-1), saliency_scores.unsqueeze(-1)], dim=-1)

    def _build_padding_mask(self, activations: torch.Tensor) -> torch.Tensor:
        B, device = activations.shape[0], activations.device
        current_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        anchor_mask = activations <= 0
        topo_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        return torch.cat([current_mask, anchor_mask, topo_mask], dim=1)

    def _mask_saliency(self, saliency_scores: torch.Tensor, batch: dict) -> torch.Tensor:
        history_mask = batch.get("history_mask")
        if history_mask is None:
            return saliency_scores
        return saliency_scores.masked_fill(~history_mask.to(device=saliency_scores.device), -10.0)

    def forward_train(self, batch: dict, use_anchors: bool = True, use_topology: bool = True) -> TrainForwardOutput:
        structured_history = batch["structured_history"]
        structured_state = batch["structured_state"]
        _, event_scores = self.event_encoder(structured_history)
        history_mask = batch.get("history_mask")
        if self.config.bypass_anchor_selection:
            saliency_scores = torch.ones_like(event_scores)
            y_star = torch.ones_like(event_scores)
            if history_mask is not None:
                y_star = y_star * history_mask.to(dtype=y_star.dtype)
        else:
            saliency_scores = self.saliency_normalizer(event_scores, history_mask)
            saliency_scores = self._mask_saliency(saliency_scores, batch)
            y_star = self.relaxed_selector(saliency_scores)
        soft_vectors = self._soft_anchor_vectors(structured_history, saliency_scores)
        _, dense_lifted = self.normalized_lift(soft_vectors)  # Full lift for anchor tokens

        # Proposal B: Time-invariant lift for topology — zeroing the time column
        # prevents monotonic time from stretching the manifold into a non-intersecting helix
        # PERF: Instead of running normalized_lift twice, zero the time column on the
        # normalized vectors and re-apply the same W_theta matmul.
        soft_vectors_spatial = soft_vectors.clone()
        soft_vectors_spatial[:, :, 0] = 0.0
        normed_spatial = self.normalized_lift.normalize(soft_vectors_spatial)
        dense_lifted_spatial = torch.matmul(normed_spatial, self.normalized_lift.W_theta.t())

        # Surrogate (kept for auxiliary topology_reg_loss only)
        topo_features_surrogate = self.topology_branch.surrogate(dense_lifted_spatial, y_star)

        # Differentiable spectral topology features (replaces detached Gudhi)
        # Uses graph Laplacian eigenvalues — fully differentiable, action loss shapes geometry
        with torch.autocast(device_type=dense_lifted.device.type, enabled=False):
            spectral_topo = self.topology_branch.spectral_features(dense_lifted_spatial, y_star)

        # Anchor tokens use the FULL lift (with time) for temporal ordering
        anchor_tokens = self.memory_readout.anchor_proj(dense_lifted) * y_star.unsqueeze(-1)
        topo_token = self.memory_readout.topology_proj(spectral_topo).unsqueeze(1)

        if not use_anchors:
            anchor_tokens = torch.zeros_like(anchor_tokens)
        if not use_topology:
            topo_token = torch.zeros_like(topo_token)
            topo_features_surrogate = torch.zeros_like(topo_features_surrogate)

        current_token = self.current_proj(structured_state).unsqueeze(1)
        transformer_tokens = torch.cat([current_token, anchor_tokens, topo_token], dim=1)
        key_padding_mask = self._build_padding_mask(y_star)
        encoded = self.task_transformer(transformer_tokens, key_padding_mask=key_padding_mask)
        pred_actions = self.action_head(encoded[:, 0, :])
        return TrainForwardOutput(
            pred_actions=pred_actions,
            event_scores=event_scores,
            saliency_scores=saliency_scores,
            y_star=y_star,
            dense_lifted_tokens=dense_lifted,
            topology_token=topo_features_surrogate,
            transformer_tokens=transformer_tokens,
        )

    def _deploy_single(self, sequence: np.ndarray) -> ExactMemoryState:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        sequence_t = torch.from_numpy(sequence).to(device=device, dtype=dtype).unsqueeze(0)
        if self.config.bypass_anchor_selection:
            y_star = np.ones(sequence.shape[0], dtype=np.float64)
            saliency_scores = np.ones(sequence.shape[0], dtype=np.float64)
            event_scores = np.zeros(sequence.shape[0], dtype=np.float64)
            indices = list(range(sequence.shape[0]))
        else:
            _, event_scores_t = self.event_encoder(sequence_t)
            saliency_scores_t = self.saliency_normalizer(event_scores_t)
            y_star_t = self.relaxed_selector(saliency_scores_t)
            event_scores = event_scores_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
            saliency_scores = saliency_scores_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
            y_star = y_star_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
            indices = self.hard_projector.project(y_star)
        anchors = self.anchor_builder.build(indices, sequence.astype(np.float64), event_scores)
        V = anchor_vectors(anchors, D=sequence.shape[1] + 3)
        # Proposal B: Zero out time column for time-invariant TDA
        V[:, 0] = 0.0
        mu = self.normalized_lift.mu.detach().cpu().numpy().astype(np.float64)
        sigma = self.normalized_lift.sigma.detach().cpu().numpy().astype(np.float64)
        V_norm, _, _ = normalize_anchors(V, mu=mu, sigma=sigma)
        W_theta = self.normalized_lift.W_theta.detach().cpu().numpy().astype(np.float64)
        point_cloud = apply_lift(V_norm, W_theta)
        diagrams, topo_summary = self.topology_branch.exact(point_cloud, self.config.Q)
        return ExactMemoryState(
            anchor_indices=indices,
            y_star=y_star,
            event_scores=event_scores,
            saliency_scores=saliency_scores,
            anchor_vectors=V,
            normalized_anchor_vectors=V_norm,
            point_cloud=point_cloud,
            persistence_diagrams=diagrams,
            topology_summary=topo_summary,
        )

    def forward_deploy(self, batch: dict, use_anchors: bool = True, use_topology: bool = True) -> DeployForwardOutput:
        structured_history = batch["structured_history"]
        structured_state = batch["structured_state"]
        history_lengths = batch.get("history_lengths")
        if history_lengths is None:
            history_lengths = batch.get("history_length")
        history_mask = batch.get("history_mask")

        # Differentiable spectral topology (same path as forward_train)
        _, event_scores_d = self.event_encoder(structured_history)
        if self.config.bypass_anchor_selection:
            saliency_d = torch.ones_like(event_scores_d)
            y_star_d = torch.ones_like(event_scores_d)
            if history_mask is not None:
                y_star_d = y_star_d * history_mask.to(dtype=y_star_d.dtype)
        else:
            saliency_d = self.saliency_normalizer(event_scores_d, history_mask)
            saliency_d = self._mask_saliency(saliency_d, batch)
            y_star_d = self.relaxed_selector(saliency_d)
        soft_vectors = self._soft_anchor_vectors(structured_history, saliency_d)
        # Proposal B: Time-invariant spatial lift
        soft_vectors_spatial = soft_vectors.clone()
        soft_vectors_spatial[:, :, 0] = 0.0
        _, dense_lifted_spatial = self.normalized_lift(soft_vectors_spatial)

        with torch.autocast(device_type=structured_history.device.type, enabled=False):
            spectral_topo = self.topology_branch.spectral_features(dense_lifted_spatial, y_star_d)
        topo_token = self.memory_readout.topology_proj(spectral_topo).unsqueeze(1)

        # Per-sample exact TDA (for diagnostic dumps only — NOT fed to Transformer)
        exact_states = []
        anchor_clouds = []
        deploy_activations_list = []
        pos_indices_list = []
        for batch_index, sequence in enumerate(structured_history.detach().cpu().numpy()):
            if history_mask is not None:
                sample_mask = history_mask[batch_index].detach().cpu().numpy().astype(bool)
                trimmed_sequence = np.asarray(sequence[sample_mask], dtype=np.float64)
                actual_length = int(sample_mask.sum())
            else:
                actual_length = int(history_lengths[batch_index].item()) if history_lengths is not None else sequence.shape[0]
                trimmed_sequence = np.asarray(sequence[:actual_length], dtype=np.float64)
            if actual_length < 1:
                trimmed_sequence = np.asarray(sequence[:1], dtype=np.float64)
                actual_length = 1
            exact_state = self._deploy_single(trimmed_sequence)
            exact_states.append(exact_state)
            cloud = exact_state.point_cloud.astype(np.float32)
            if self.config.bypass_anchor_selection:
                num_valid = len(exact_state.y_star)
                deploy_act_size = self.config.max_history_tokens
            else:
                num_valid = min(cloud.shape[0], self.config.K)
                deploy_act_size = self.config.K
            
            deploy_act = np.zeros(deploy_act_size, dtype=np.float32)
            deploy_act[:num_valid] = 1.0
            deploy_activations_list.append(torch.from_numpy(deploy_act).to(structured_history.device))
            
            pos = [0]
            for idx in exact_state.anchor_indices[:num_valid]:
                pos.append(int(idx) + 1)
            for _ in range(deploy_act_size - len(exact_state.anchor_indices[:num_valid])):
                pos.append(0)
            pos.append(actual_length + 1)  # spectral topo position
            pos_indices_list.append(pos)
            
            if self.config.bypass_anchor_selection:
                if cloud.shape[0] < self.config.max_history_tokens:
                    pad = np.zeros((self.config.max_history_tokens - cloud.shape[0], self.config.k), dtype=np.float32)
                    cloud = np.concatenate([cloud, pad], axis=0)
                else:
                    cloud = cloud[: self.config.max_history_tokens]
            else:
                if cloud.shape[0] < self.config.K:
                    pad = np.zeros((self.config.K - cloud.shape[0], self.config.k), dtype=np.float32)
                    cloud = np.concatenate([cloud, pad], axis=0)
                else:
                    cloud = cloud[: self.config.K]
            
            anchor_clouds.append(torch.from_numpy(cloud).to(structured_history.device, dtype=structured_history.dtype))
            
        anchor_cloud_tensor = torch.stack(anchor_clouds, dim=0)
        deploy_activations = torch.stack(deploy_activations_list, dim=0)
        pos_indices_tensor = torch.tensor(pos_indices_list, dtype=torch.long, device=structured_history.device)
        
        anchor_tokens = self.memory_readout.anchor_proj(anchor_cloud_tensor)
        anchor_tokens = anchor_tokens * deploy_activations.unsqueeze(-1)
        
        if not use_anchors:
            anchor_tokens = torch.zeros_like(anchor_tokens)
        if not use_topology:
            topo_token = torch.zeros_like(topo_token)

        current_token = self.current_proj(structured_state).unsqueeze(1)
        transformer_tokens = torch.cat([current_token, anchor_tokens, topo_token], dim=1)
        
        key_padding_mask = self._build_padding_mask(deploy_activations)
        encoded = self.task_transformer(transformer_tokens, key_padding_mask=key_padding_mask, pos_indices=pos_indices_tensor)
        
        pred_actions = self.action_head(encoded[:, 0, :])
        return DeployForwardOutput(
            pred_actions=pred_actions,
            exact_memory_states=exact_states,
            transformer_tokens=transformer_tokens,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.forward_train(batch).pred_actions
