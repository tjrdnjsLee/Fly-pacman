import numpy as np

from flypac.core import directions as d
from flypac.core.types import GameView
from flypac.maps.original import load_original
from flypac.senses.sense import sense

M = load_original()


def _view(cell, heading, z=0):
    f = 4
    return GameView(
        pac_cell=M.pacman_start, pac_dir=0, lives=3, power=False, power_ticks=0, ghost_streak=0,
        fly_cell=np.full(f, cell), fly_dir=np.full(f, heading), fly_z=np.full(f, z, bool),
        fly_in_house=np.zeros(f, bool), dots=M.dots.copy(), energizers=M.energizers.copy(),
        fruit=False, tick=0, score=0, learning_active=False, elroy_stage=0, frozen=False,
    )


INFO = (np.ones((1, 4, 4), bool), np.zeros((1, 4), bool))
C0 = 0.01


def _sense(odor, cell, heading, phero=None):
    phero = np.zeros_like(odor) if phero is None else phero
    return sense(odor, phero, [_view(cell, heading)], INFO, C0, nbr=M.nbr)


def test_shapes_and_passthrough():
    obs = _sense(np.zeros((1, M.nbr.shape[0])), M.cell_id[5, 1], d.SOUTH)
    obs.validate()
    assert obs.odor.shape == obs.phero.shape == obs.legal.shape == (1, 4, 4)
    assert obs.z.shape == obs.decide.shape == (1, 4)
    assert not obs.odor.any()


def test_only_front_cell_scented():
    cell = M.cell_id[5, 10]  # 가로 복도 가운데
    for heading in (d.EAST, d.WEST):  # 복도 가운데: 정면이 벽이 아닌 방향
        odor = np.zeros((1, 320))
        front = M.nbr[cell, heading]
        odor[0, front] = 1.0
        obs = _sense(odor, cell, heading)
        assert obs.odor[0, 0, 1] > 0
        assert (obs.odor[0, 0, [0, 2, 3]] == 0).all()
        assert np.isclose(obs.odor[0, 0, 1], np.log1p(1.0 / C0))


def test_left_cell_scent_is_odor0():
    cell = M.cell_id[5, 6]  # 위아래로 통로가 있는 갈림길
    odor = np.zeros((1, 320))
    odor[0, M.nbr[cell, d.NORTH]] = 1.0  # 동쪽을 보면 북쪽이 왼쪽
    obs = _sense(odor, cell, d.EAST)
    assert obs.odor[0, 0, 0] > 0 and (obs.odor[0, 0, 1:] == 0).all()


def test_wall_is_zero_even_with_scent_everywhere():
    cell = M.cell_id[5, 10]
    obs = _sense(np.ones((1, 320)), cell, d.EAST)
    for e in range(4):
        a = int(d.ego_to_abs(d.EAST, e))
        wall = M.nbr[cell, a] < 0
        assert (obs.odor[0, 0, e] == 0) == wall


def test_phero_read_same_way_and_tunnel_neighbor():
    left, right = M.cell_id[14, 0], M.cell_id[14, 27]
    phero = np.zeros((1, 320))
    phero[0, right] = 1.0
    obs = _sense(np.zeros((1, 320)), left, d.WEST, phero)  # 서쪽을 보면 터널 건너 오른쪽 끝칸이 앞
    assert obs.phero[0, 0, 1] > 0 and not obs.odor.any()
