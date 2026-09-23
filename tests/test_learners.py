import numpy as np
import pytest
import torch

from arcrl.agents.dqn import DQNLearner
from arcrl.agents.ppo import PPOLearner
from arcrl.models.nets import ArcNet
from arcrl.utils import Config


def test_net_shapes():
    net = ArcNet(frame_stack=2, click_grid=16, value_head=True)
    out, v = net(torch.zeros(3, 2, 64, 64, dtype=torch.uint8))
    assert out.shape == (3, 6 + 256) and v.shape == (3,)


@pytest.mark.parametrize("cls", [DQNLearner, PPOLearner])
def test_learner_updates_and_respects_mask(cls):
    cfg = Config(batch_size=4, warmup_steps=4, rollout_len=8, minibatch_size=4, ppo_epochs=1)
    n = 6 + 64
    L = cls(cfg, n, "cpu", seed=0)
    s = np.zeros((1, 64, 64), np.uint8)
    mask = np.zeros(n, bool)
    mask[[0, 10]] = True
    stats = None
    for i in range(16):
        a = L.select(s, mask)
        assert a in (0, 10)
        stats = L.observe(s, mask, a, 1.0 if a == 10 else 0.0, s, mask, i % 7 == 6) or stats
    assert stats and np.isfinite(stats["loss"])


def test_dqn_state_dict_roundtrip():
    cfg = Config()
    a, b = DQNLearner(cfg, 70, "cpu"), DQNLearner(cfg, 70, "cpu", seed=1)
    b.load_state_dict(a.state_dict())
    x = torch.zeros(1, 1, 64, 64, dtype=torch.uint8)
    assert torch.allclose(a.q(x), b.q(x))
