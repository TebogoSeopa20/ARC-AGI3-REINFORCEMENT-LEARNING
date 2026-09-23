"""Frame -> grid conversion, hashing, frame stacking."""
from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field

import numpy as np

GRID = 64
N_COLOURS = 16


def to_grid(frame) -> np.ndarray:
    """Last animation frame of a FrameData/FrameDataRaw `frame` field as a 64x64 uint8 grid."""
    if frame is None or len(frame) == 0:
        return np.zeros((GRID, GRID), dtype=np.uint8)
    last = np.asarray(frame[-1], dtype=np.int64)
    if last.ndim == 3:
        last = last[-1]
    out = np.zeros((GRID, GRID), dtype=np.uint8)
    h, w = min(GRID, last.shape[0]), min(GRID, last.shape[1])
    out[:h, :w] = np.clip(last[:h, :w], 0, N_COLOURS - 1)
    return out


def grid_hash(grid: np.ndarray) -> bytes:
    return hashlib.blake2b(grid.tobytes(), digest_size=8).digest()


@dataclass
class Obs:
    grid: np.ndarray
    state: str
    levels: int
    available: list[int] = field(default_factory=list)
    win_levels: int = 0

    @classmethod
    def from_frame_data(cls, fd) -> "Obs":
        state = fd.state.value if hasattr(fd.state, "value") else str(fd.state)
        return cls(
            grid=to_grid(fd.frame),
            state=state,
            levels=int(fd.levels_completed),
            available=list(fd.available_actions or []),
            win_levels=int(getattr(fd, "win_levels", 0) or 0),
        )


class FrameStack:
    def __init__(self, k: int):
        self.k = k
        self.buf: deque[np.ndarray] = deque(maxlen=k)

    def clear(self) -> None:
        self.buf.clear()

    def push(self, grid: np.ndarray) -> np.ndarray:
        if not self.buf:
            for _ in range(self.k):
                self.buf.append(grid)
        else:
            self.buf.append(grid)
        return np.stack(self.buf, 0)
