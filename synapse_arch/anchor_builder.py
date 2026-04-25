from __future__ import annotations

from typing import List

import numpy as np

from synapse_core.anchor_selector import Anchor, build_anchors


class AnchorBuilder:
    def build(self, indices: List[int], trajectory: np.ndarray, event_scores: np.ndarray) -> List[Anchor]:
        return build_anchors(indices, trajectory, event_scores)
