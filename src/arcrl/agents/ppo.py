"""PPO (Schulman et al., 2017): clipped surrogate, GAE(lambda), masked categorical policy.

Learning happens every `rollout_len` environment steps: advantages are computed with GAE,
then `ppo_epochs` passes of minibatch SGD maximise
    E[min(rho*A, clip(rho, 1-eps, 1+eps)*A)] - vf_coef*(V - R)^2 + ent_coef*H(pi)
where rho = pi_new(a|s) / pi_old(a|s). The rollout is then discarded (on-policy).

Improvement 4 (cfg.effect_model): an effect head predicts, per action, whether it changes the clock-masked
frame (BCE auxiliary loss on every transition). Its detached log-probability, scaled by effect_bias, is added
to the policy logits. The bias used when sampling is stored with the transition and reused in the update, so
the importance ratio compares identical behaviour distributions.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from arcrl.models.nets import ArcNet
from arcrl.utils import Config

NEG = -1e9


class PPOLearner:
    def __init__(self, cfg: Config, n_actions: int, device: str, seed: int = 0):
        self.cfg, self.device, self.n = cfg, device, n_actions
        self.rng = np.random.default_rng(seed)
        self.effect = cfg.effect_model
        self.net = ArcNet(cfg.frame_stack, cfg.click_grid, value_head=True, effect_head=self.effect).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.t = 0
        self._pending: tuple[float, float] | None = None
        self._clear()

    def _clear(self) -> None:
        self.S, self.M, self.A, self.LP, self.V, self.R, self.D = [], [], [], [], [], [], []
        self.B, self.Y = [], []

    def _dist(self, s: torch.Tensor, m: torch.Tensor, bias: torch.Tensor | None = None):
        if not self.effect:
            logits, v = self.net(s)
            return Categorical(logits=logits.masked_fill(~m, NEG)), v, None
        logits, v, eff = self.net.forward_all(s)
        if bias is None:
            bias = self.cfg.effect_bias * F.logsigmoid(eff.detach())
        return Categorical(logits=(logits + bias).masked_fill(~m, NEG)), v, (eff, bias)

    def select(self, s: np.ndarray, mask: np.ndarray) -> int:
        self.t += 1
        with torch.no_grad():
            dist, v, extra = self._dist(
                torch.as_tensor(s[None], device=self.device),
                torch.as_tensor(mask[None], device=self.device),
            )
            a = dist.sample()
        bias = extra[1][0].cpu().numpy() if extra else None
        self._pending = (float(dist.log_prob(a)), float(v), bias)
        return int(a)

    def observe(self, s, mask, a, r, s2, m2, done, changed: bool = False) -> dict | None:
        lp, v, bias = self._pending
        self.S.append(s); self.M.append(mask); self.A.append(a)
        self.LP.append(lp); self.V.append(v); self.R.append(r); self.D.append(float(done))
        self.B.append(bias); self.Y.append(float(changed))
        if len(self.S) < self.cfg.rollout_len:
            return None
        last_v = 0.0
        if not done:
            with torch.no_grad():
                _, lv = self.net(torch.as_tensor(s2[None], device=self.device))
            last_v = float(lv)
        stats = self._update(last_v)
        self._clear()
        return stats

    def _gae(self, last_v: float):
        c = self.cfg
        adv = np.zeros(len(self.R), np.float32)
        gae, nxt = 0.0, last_v
        for i in reversed(range(len(self.R))):
            nonterm = 1.0 - self.D[i]
            delta = self.R[i] + c.gamma * nxt * nonterm - self.V[i]
            gae = delta + c.gamma * c.gae_lambda * nonterm * gae
            adv[i] = gae
            nxt = self.V[i]
        return adv, adv + np.asarray(self.V, np.float32)

    def _update(self, last_v: float) -> dict:
        c, dev = self.cfg, self.device
        adv, ret = self._gae(last_v)
        S = torch.as_tensor(np.stack(self.S), device=dev)
        M = torch.as_tensor(np.stack(self.M), device=dev)
        A = torch.as_tensor(self.A, device=dev)
        LP = torch.as_tensor(self.LP, device=dev)
        ADV = torch.as_tensor(adv, device=dev)
        RET = torch.as_tensor(ret, device=dev)
        B = torch.as_tensor(np.stack(self.B), device=dev) if self.effect else None
        Y = torch.as_tensor(self.Y, device=dev)
        n, stats = len(A), {}
        for _ in range(c.ppo_epochs):
            for idx in np.array_split(self.rng.permutation(n), max(1, n // c.minibatch_size)):
                idx = torch.as_tensor(idx, device=dev)
                dist, v, extra = self._dist(S[idx], M[idx], B[idx] if B is not None else None)
                ratio = torch.exp(dist.log_prob(A[idx]) - LP[idx])
                a_n = ADV[idx]
                a_n = (a_n - a_n.mean()) / (a_n.std() + 1e-8) if len(idx) > 1 else a_n
                pg = -torch.min(ratio * a_n, ratio.clamp(1 - c.clip_eps, 1 + c.clip_eps) * a_n).mean()
                vf = (v - RET[idx]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg + c.vf_coef * vf - c.ent_coef * ent
                if extra is not None:
                    aux = F.binary_cross_entropy_with_logits(extra[0].gather(1, A[idx][:, None]).squeeze(1), Y[idx])
                    loss = loss + c.effect_coef * aux
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), c.grad_clip)
                self.opt.step()
                stats = {"loss": loss.item(), "pg": pg.item(), "vf": vf.item(), "entropy": ent.item(),
                         "clip_frac": float(((ratio - 1).abs() > c.clip_eps).float().mean())}
                if extra is not None:
                    stats["effect_loss"] = aux.item()
        return stats

    def state_dict(self) -> dict:
        return {"net": self.net.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.net.load_state_dict(sd["net"])
