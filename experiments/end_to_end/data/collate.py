"""
Custom collation for variable-length trajectory batches.

The Z2 model's event encoder processes full trajectory prefixes, which
have different lengths across samples in a batch. This collator:

1. Filters out None samples (from loading errors — mirrors semantic_planner_collate_fn)
2. Pads structured_history to the maximum length in the batch
3. Creates a boolean padding mask (True = real data, False = padding)
4. Stacks all fixed-size tensors normally

The padding is handled naturally by the Z2 pipeline:
- Zero-padded suffix → near-zero event scores from EventEncoder
- SaliencyNormalizer suppresses low scores → zero saliency
- RelaxedSelector ignores zero-saliency timesteps

No special masking logic is needed inside the model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

log = logging.getLogger(__name__)


def trajectory_collate_fn(
    batch: List[Optional[Dict[str, Any]]],
) -> Dict[str, torch.Tensor]:
    """Collate variable-length trajectory samples into a padded batch.

    Handles None samples gracefully (from loading errors in SynapseE2EDataset),
    matching the robust collation pattern from SemanticPlannerDataset.

    Parameters
    ----------
    batch : List[Optional[Dict[str, torch.Tensor]]]
        List of samples from SynapseE2EDataset.__getitem__.
        May contain None values from failed loading attempts.

    Returns
    -------
    Dict[str, torch.Tensor]
        Collated batch. Empty dict if all samples failed.
    """
    # --- 1. Filter out failed samples (None) ---
    valid_samples = [s for s in batch if s is not None]

    if not valid_samples:
        log.warning("Batch collation failed: all samples were None.")
        return {}

    # --- 2. Extract variable-length histories ---
    histories = [s["structured_history"] for s in valid_samples]

    # Use history_length field if available, otherwise infer from tensor shape
    if "history_length" in valid_samples[0]:
        lengths = torch.stack([s["history_length"] for s in valid_samples])
    else:
        lengths = torch.tensor([h.shape[0] for h in histories], dtype=torch.long)

    # --- 3. Pad histories to max length in this batch ---
    padded_histories = pad_sequence(histories, batch_first=True, padding_value=0.0)
    # padded_histories shape: (B, T_max, state_dim)

    # --- 4. Create mask: True for real timesteps, False for padding ---
    T_max = padded_histories.shape[1]
    history_mask = torch.arange(T_max).unsqueeze(0) < lengths.unsqueeze(1)
    # history_mask shape: (B, T_max)

    # --- 5. Stack all fixed-size tensors ---
    return {
        "structured_history": padded_histories,
        "history_mask": history_mask,
        "history_lengths": lengths,
        "structured_state": torch.stack([s["structured_state"] for s in valid_samples]),
        "ground_truth_actions": torch.stack([s["ground_truth_actions"] for s in valid_samples]),
        "phase_label": torch.stack([s["phase_label"] for s in valid_samples]),
        "episode_idx": torch.stack([s["episode_idx"] for s in valid_samples]),
        "timestep": torch.stack([s["timestep"] for s in valid_samples]),
    }
