import numpy as np
from helpers import MAP, cell, make_cfg

from flypac.core import directions as d
from flypac.core.rng import make_rng
from flypac.engine.fake import FakeGame
from flypac.pacman.rule_based import RuleBasedPacman


def setup():
    cfg = make_cfg()
    g = FakeGame(MAP, cfg, make_rng(0, 0, "engine"))
    return g, RuleBasedPacman(MAP, cfg, make_rng(0, 0, "pacman"))


def test_heads_to_nearest_dot():
    g, p = setup()
    g.dots[:] = False
    g.energizers[:] = False
    g.dots[cell(5, 8)] = True
    g.pac_cell = cell(5, 3)
    assert p.act([g.view()]).tolist() == [d.EAST]


def test_flees_close_danger_and_ignores_far_one():
    g, p = setup()
    g.dots[:] = False
    g.energizers[:] = False
    g.dots[cell(5, 20)] = True  # 점은 동쪽에 있다
    g.pac_cell = cell(5, 10)
    g.released[:] = True
    g.fly_cell[:] = cell(5, 12)  # 위험한 초파리가 동쪽 2칸
    assert p.act([g.view()]).tolist() == [d.WEST]
    g.fly_cell[:] = cell(26, 2)  # 멀다 (기준 거리 밖)
    assert p.act([g.view()]).tolist() == [d.EAST]


def test_frightened_fly_is_chased():
    g, p = setup()
    g.dots[:] = False
    g.energizers[:] = False
    g.dots[cell(5, 1)] = True  # 점은 서쪽
    g.pac_cell = cell(5, 10)
    g.fly_cell[:] = cell(26, 2)
    g.fly_cell[0] = cell(5, 14)
    v = g.view()
    z = np.zeros(4, bool)
    z[0] = True
    from dataclasses import replace

    assert p.act([replace(v, fly_z=z, fly_in_house=np.zeros(4, bool))]).tolist() == [d.EAST]


def test_act_shape_and_reset_noop():
    g, p = setup()
    out = p.act([g.view(), g.view()])
    assert out.shape == (2,) and ((0 <= out) & (out < 4)).all()
    p.reset(np.ones(2, bool))
