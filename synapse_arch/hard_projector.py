from __future__ import annotations

from typing import List

import numpy as np

from synapse_core.anchor_selector import hard_projection


class HardProjector:
    def __init__(self, K: int, r: int) -> None:
        self.K = K
        self.r = r

    def project(self, y_star: np.ndarray) -> List[int]:
        return hard_projection(y_star, self.K, self.r)
