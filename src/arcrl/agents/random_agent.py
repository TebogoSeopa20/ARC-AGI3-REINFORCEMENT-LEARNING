"""Uniform random policy over legal abstracted actions (sanity-check reference)."""
from __future__ import annotations

import numpy as np


class RandomLearner:
    def __init__(self, cfg, n_actions: int, device: str = "cpu", seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def select(self, s, mask) -> int:
        self.t += 1
        return int(self.rng.choice(np.flatnonzero(mask)))

    def observe(self, *args, **kwargs) -> None:
        return None

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, sd: dict) -> None:
        pass
