"""
Synthetic sequence and task generators for broader-paper experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SequenceSample:
    sequence: np.ndarray
    label: int
    event_indices: List[int]
    metadata: Dict[str, float]


def generate_event_sparse_sequence(
    rng: np.random.Generator,
    T: int,
    d: int,
    num_events: int,
    noise_std: float,
    distractor_scale: float,
    near_threshold_prob: float = 0.25,
    smooth: bool = False,
) -> Tuple[np.ndarray, List[int]]:
    sequence = np.zeros((T, d), dtype=np.float32)
    state = rng.normal(scale=0.1, size=d)
    event_positions = sorted(rng.choice(np.arange(5, T - 5), size=num_events, replace=False).tolist())
    prev = state.copy()
    for t in range(T):
        if t in event_positions:
            jump_scale = distractor_scale * (0.6 if rng.random() < near_threshold_prob else 1.5)
            jump = rng.normal(scale=jump_scale, size=d)
            state = state + jump
        elif smooth:
            state = 0.97 * state + 0.03 * prev + rng.normal(scale=noise_std * 0.5, size=d)
        else:
            state = state + rng.normal(scale=noise_std, size=d)
        prev = state.copy()
        sequence[t] = state + rng.normal(scale=noise_std, size=d)
    return sequence.astype(np.float32), event_positions


def _delayed_retrieval(
    rng: np.random.Generator,
    T: int,
    d: int,
    distractor_scale: float,
) -> SequenceSample:
    seq = rng.normal(scale=0.15, size=(T, d)).astype(np.float32)
    event_t = int(rng.integers(low=4, high=max(5, T // 4)))
    label = int(rng.integers(0, 3))
    token = np.zeros(d, dtype=np.float32)
    token[label % d] = 3.5
    seq[event_t] += token
    tail_noise = rng.normal(scale=distractor_scale, size=(T - event_t - 1, d))
    seq[event_t + 1:] += tail_noise
    return SequenceSample(seq, label, [event_t], {"family": 0, "distractor_scale": distractor_scale})


def _ordered_trigger_response(
    rng: np.random.Generator,
    T: int,
    d: int,
    distractor_scale: float,
) -> SequenceSample:
    seq = rng.normal(scale=0.12, size=(T, d)).astype(np.float32)
    t1 = int(rng.integers(4, max(5, T // 3)))
    t2 = int(rng.integers(t1 + 3, max(t1 + 4, 2 * T // 3)))
    label = int(rng.integers(0, 2))
    seq[t1, 0] += 2.5 if label == 0 else -2.5
    seq[t2, 1] += 2.5
    seq[t2 + 1:] += rng.normal(scale=distractor_scale, size=(T - t2 - 1, d))
    return SequenceSample(seq, label, [t1, t2], {"family": 1, "distractor_scale": distractor_scale})


def _phase_conditioned(
    rng: np.random.Generator,
    T: int,
    d: int,
    distractor_scale: float,
) -> SequenceSample:
    seq = rng.normal(scale=0.1, size=(T, d)).astype(np.float32)
    cp1 = int(rng.integers(5, max(6, T // 3)))
    cp2 = int(rng.integers(cp1 + 5, max(cp1 + 6, 2 * T // 3)))
    label = int(rng.integers(0, 4))
    seq[:cp1, 0] += label * 0.2
    seq[cp1:cp2, 2] += 1.5
    seq[cp2:, 3] += (-1.0 if label % 2 else 1.0) * 1.8
    seq += rng.normal(scale=distractor_scale, size=(T, d))
    return SequenceSample(seq, label, [cp1, cp2], {"family": 2, "distractor_scale": distractor_scale})


def _revisit_switching(
    rng: np.random.Generator,
    T: int,
    d: int,
    distractor_scale: float,
) -> SequenceSample:
    seq = rng.normal(scale=0.1, size=(T, d)).astype(np.float32)
    t1 = int(rng.integers(5, max(6, T // 4)))
    t2 = int(rng.integers(t1 + 6, max(t1 + 7, T // 2)))
    t3 = int(rng.integers(t2 + 6, max(t2 + 7, T - 5)))
    label = int(rng.integers(0, 2))
    base = rng.normal(scale=1.0, size=d)
    seq[t1] += base
    seq[t2] += base + rng.normal(scale=0.05, size=d)
    seq[t3, 4 % d] += 2.0 if label == 1 else -2.0
    seq += rng.normal(scale=distractor_scale, size=(T, d))
    return SequenceSample(seq, label, [t1, t2, t3], {"family": 3, "distractor_scale": distractor_scale})


TASK_GENERATORS = {
    "delayed_retrieval": _delayed_retrieval,
    "ordered_trigger_response": _ordered_trigger_response,
    "phase_conditioned_decision": _phase_conditioned,
    "revisit_dependent_rule_switching": _revisit_switching,
}


def generate_memory_task_dataset(
    rng: np.random.Generator,
    family: str,
    n_samples: int,
    T: int,
    d: int,
    distractor_scale: float,
) -> List[SequenceSample]:
    gen = TASK_GENERATORS[family]
    return [gen(rng, T, d, distractor_scale) for _ in range(n_samples)]


def generate_topology_sequence(
    rng: np.random.Generator,
    T: int,
    d: int,
    family: str,
    noise_std: float,
) -> SequenceSample:
    if d < 2:
        d = 2
    t = np.linspace(0.0, 1.0, T, dtype=np.float32)
    if family == "loop":
        x = np.stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)], axis=1)
        label = 0
    elif family == "line":
        x = np.stack([2 * t - 1, np.zeros_like(t)], axis=1)
        label = 1
    elif family == "revisit":
        x = np.stack([np.sin(4 * np.pi * t), np.sin(2 * np.pi * t)], axis=1)
        label = 2
    else:
        x = np.stack([t, np.sign(np.sin(4 * np.pi * t)) * 0.6], axis=1)
        label = 3
    if d > 2:
        tail = rng.normal(scale=0.05, size=(T, d - 2))
        x = np.concatenate([x, tail], axis=1)
    x = x + rng.normal(scale=noise_std, size=x.shape)
    diffs = np.linalg.norm(np.diff(x[:, :2], axis=0), axis=1)
    event_indices = np.where(diffs > np.quantile(diffs, 0.75))[0].tolist()
    return SequenceSample(x.astype(np.float32), label, event_indices, {"family": float(label), "noise_std": noise_std})


def generate_topology_dataset(
    rng: np.random.Generator,
    n_samples: int,
    T: int,
    d: int,
    noise_std: float,
) -> List[SequenceSample]:
    families = ["loop", "line", "revisit", "zigzag"]
    samples: List[SequenceSample] = []
    for idx in range(n_samples):
        samples.append(generate_topology_sequence(rng, T, d, families[idx % len(families)], noise_std))
    return samples


def generate_control_dataset(
    rng: np.random.Generator,
    n_episodes: int,
    T: int,
    state_dim: int,
    action_dim: int,
) -> List[Dict[str, np.ndarray]]:
    episodes = []
    for _ in range(n_episodes):
        state = rng.normal(scale=0.1, size=(T, state_dim)).astype(np.float32)
        target = rng.normal(scale=0.5, size=action_dim).astype(np.float32)
        action = np.zeros((T, action_dim), dtype=np.float32)
        for t in range(T):
            src = state[t, :action_dim]
            action[t] = 0.6 * src + 0.3 * target + 0.1 * rng.normal(size=action_dim)
        episodes.append(
            {
                "states": state,
                "actions": action.astype(np.float32),
                "phase_labels": np.clip((np.arange(T) / max(1, T / 4)).astype(int), 0, 3),
            }
        )
    return episodes

