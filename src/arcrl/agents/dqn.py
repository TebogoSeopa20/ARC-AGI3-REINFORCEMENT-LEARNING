"""DQN (Mnih et al., 2015) with action masking; optional Double-DQN target.

Improvement 4 (cfg.effect_model): an effect head on the shared trunk predicts, per action, whether it will
change the clock-masked frame. It is trained with binary cross-entropy on every replayed transition
(dense supervision, unlike the sparse level reward), and replaces uniform ε-exploration with sampling
proportional to the predicted effect probability.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from arcrl.models.nets import ArcNet
from arcrl.utils import Config

NEG = -1e9
P_FLOOR = 0.05


class Replay:
    def __init__(self, capacity: int, stack: int, n_actions: int):
        self.cap, self.i, self.full = capacity, 0, False
        self.s = np.zeros((capacity, stack, 64, 64), np.uint8)
        self.s2 = np.zeros_like(self.s)
        self.m2 = np.zeros((capacity, n_actions), bool)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.y = np.zeros(capacity, np.float32)

    def add(self, s, a, r, s2, m2, d, y=0.0) -> None:
        j = self.i
        self.s[j], self.a[j], self.r[j], self.s2[j], self.m2[j], self.d[j], self.y[j] = s, a, r, s2, m2, d, y
        self.i = (j + 1) % self.cap
        self.full |= self.i == 0

    def __len__(self) -> int:
        return self.cap if self.full else self.i

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self), n)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.m2[idx], self.d[idx], self.y[idx]


class DQNLearner:
    def __init__(self, cfg: Config, n_actions: int, device: str, seed: int = 0):
        self.cfg, self.device, self.n = cfg, device, n_actions
        self.rng = np.random.default_rng(seed)
        self.effect = cfg.effect_model
        self.q = ArcNet(cfg.frame_stack, cfg.click_grid, effect_head=self.effect).to(device)
        self.q_t = ArcNet(cfg.frame_stack, cfg.click_grid, effect_head=self.effect).to(device)
        self.q_t.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buf = Replay(cfg.replay_capacity, cfg.frame_stack, n_actions)
        self.t = 0
        self.updates = 0

    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.t / max(1, c.eps_decay_steps))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def select(self, s: np.ndarray, mask: np.ndarray) -> int:
        self.t += 1
        explore = self.rng.random() < self.epsilon()
        if explore and not self.effect:
            return int(self.rng.choice(np.flatnonzero(mask)))
        with torch.no_grad():
            x = torch.as_tensor(s[None], device=self.device)
            if self.effect:
                q, _, eff = self.q.forward_all(x)
                p = torch.sigmoid(eff[0]).cpu().numpy()
            else:
                q = self.q(x)
            q = q[0].cpu().numpy()
        if explore:
            w = np.where(mask, np.maximum(p, P_FLOOR), 0.0)
            return int(self.rng.choice(self.n, p=w / w.sum()))
        q[~mask] = NEG
        return int(q.argmax())

    def observe(self, s, mask, a, r, s2, m2, done, changed: bool = False) -> dict | None:
        self.buf.add(s, a, r, s2, m2, float(done), float(changed))
        c = self.cfg
        if len(self.buf) < max(c.warmup_steps, c.batch_size) or self.t % c.train_freq:
            return None
        return self._update()

    def _update(self) -> dict:
        c, dev = self.cfg, self.device
        s, a, r, s2, m2, d, y = self.buf.sample(c.batch_size, self.rng)
        s, s2 = torch.as_tensor(s, device=dev), torch.as_tensor(s2, device=dev)
        a, r = torch.as_tensor(a, device=dev), torch.as_tensor(r, device=dev)
        d, m2 = torch.as_tensor(d, device=dev), torch.as_tensor(m2, device=dev)
        stats = {}
        if self.effect:
            qa, _, eff = self.q.forward_all(s)
            aux = F.binary_cross_entropy_with_logits(eff.gather(1, a[:, None]).squeeze(1),
                                                     torch.as_tensor(y, device=dev))
            stats["effect_loss"] = float(aux.detach())
        else:
            qa, aux = self.q(s), 0.0
        q = qa.gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            qt = self.q_t(s2).masked_fill(~m2, NEG)
            if c.double_dqn:
                a2 = self.q(s2).masked_fill(~m2, NEG).argmax(1, keepdim=True)
                nxt = qt.gather(1, a2).squeeze(1)
            else:
                nxt = qt.max(1).values
            target = r + c.gamma * (1 - d) * nxt
        td = F.smooth_l1_loss(q, target)
        loss = td + c.effect_coef * aux
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), c.grad_clip)
        self.opt.step()
        self.updates += 1
        if self.updates % c.target_sync == 0:
            self.q_t.load_state_dict(self.q.state_dict())
        return {"loss": float(td.detach()), "q_mean": float(q.detach().mean()), **stats}

    def state_dict(self) -> dict:
        return {"net": self.q.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.q.load_state_dict(sd["net"])
        self.q_t.load_state_dict(sd["net"])
