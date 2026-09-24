"""CNN over one-hot 64x64 grids with a spatial head for ACTION6 clicks.

Output is one vector of size 6 + k*k: global-pooled features score the 6 simple actions,
a 1x1 conv over a 16x16 feature map (pooled to k x k) scores each click cell.
Used as Q-values by DQN and as policy logits (plus a value head) by PPO.
Optional effect head (improvement 4): same layout, logits of P(action changes the clock-masked frame),
trained as a supervised auxiliary task on every transition (cf. Jaderberg et al., 2017; Smit, 2025).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_COLOURS = 16


def encode(grids: torch.Tensor) -> torch.Tensor:
    """(B, S, 64, 64) uint8/long -> (B, 16*S, 64, 64) float one-hot."""
    b, s, h, w = grids.shape
    x = F.one_hot(grids.long(), N_COLOURS).permute(0, 1, 4, 2, 3)
    return x.reshape(b, s * N_COLOURS, h, w).float()


class ArcNet(nn.Module):
    def __init__(self, frame_stack: int, click_grid: int, n_simple: int = 6, value_head: bool = False,
                 effect_head: bool = False):
        super().__init__()
        self.k = click_grid
        self.trunk = nn.Sequential(
            nn.Conv2d(N_COLOURS * frame_stack, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.click = nn.Conv2d(64, 1, 1)
        self.mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        self.simple = nn.Linear(128, n_simple)
        self.value = nn.Linear(128, 1) if value_head else None
        self.effect_click = nn.Conv2d(64, 1, 1) if effect_head else None
        self.effect_simple = nn.Linear(128, n_simple) if effect_head else None

    def _heads(self, grids: torch.Tensor):
        f = self.trunk(encode(grids))
        g = self.mlp(f.mean((2, 3)))
        out = torch.cat([self.simple(g), F.adaptive_avg_pool2d(self.click(f), self.k).flatten(1)], 1)
        return f, g, out

    def forward(self, grids: torch.Tensor):
        _, g, out = self._heads(grids)
        if self.value is None:
            return out
        return out, self.value(g).squeeze(1)

    def forward_all(self, grids: torch.Tensor):
        """(policy/Q logits, value or None, effect logits)."""
        f, g, out = self._heads(grids)
        eff = torch.cat([self.effect_simple(g), F.adaptive_avg_pool2d(self.effect_click(f), self.k).flatten(1)], 1)
        v = self.value(g).squeeze(1) if self.value is not None else None
        return out, v, eff
