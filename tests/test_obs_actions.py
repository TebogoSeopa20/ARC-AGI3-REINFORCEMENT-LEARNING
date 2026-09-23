import numpy as np
from arcengine import GameAction

from arcrl.env.actions import ActionSpace
from arcrl.env.obs import FrameStack, grid_hash, to_grid


def test_to_grid_takes_last_frame_and_pads():
    a, b = np.zeros((64, 64)), np.full((10, 20), 7)
    g = to_grid([a, b])
    assert g.shape == (64, 64) and g[0, 0] == 7 and g[20, 30] == 0


def test_hash_and_stack():
    g = np.zeros((64, 64), np.uint8)
    assert grid_hash(g) == grid_hash(g.copy())
    fs = FrameStack(3)
    assert fs.push(g).shape == (3, 64, 64)


def test_mask_respects_available_actions():
    sp = ActionSpace(8)
    m = sp.mask([1, 2], np.zeros((64, 64), np.uint8))
    assert m[:2].all() and not m[2:].any()


def test_click_decode_centre_and_object_mask():
    sp = ActionSpace(8, object_clicks=True)
    g = np.zeros((64, 64), np.uint8)
    g[40:42, 9:11] = 5
    m = sp.mask([6], g)
    click = np.flatnonzero(m[6:])
    assert len(click) == 1
    ga, data = sp.decode(6 + click[0])
    assert ga is GameAction.ACTION6 and g[data["y"], data["x"]] == 5
    ga, data = ActionSpace(8).decode(6)
    assert (data["x"], data["y"]) == (4, 4)
