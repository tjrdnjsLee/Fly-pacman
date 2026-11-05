import numpy as np
import pytest

from flypac.core import directions as d
from flypac.maps.original import HOUSE, load_original


@pytest.fixture(scope="module")
def m():
    return load_original()


def test_walkable_count_is_320(m):
    assert m.nbr.shape == (320, 4)
    assert (m.cell_id >= 0).sum() == 320
    assert m.grid.shape == (31, 28)


def test_cell_table_roundtrip(m):
    for i, (r, c) in enumerate(m.cell_rc):
        assert m.cell_id[r, c] == i
    assert m.cell_id.max() == 319 and (m.cell_id[m.grid == 0] == -1).all()


def test_nbr_is_symmetric_and_matches_grid(m):
    for i in range(320):
        for k in range(4):
            j = m.nbr[i, k]
            if j >= 0:
                assert m.nbr[j, (k + 2) % 4] == i
    r, c = m.cell_rc[m.pacman_start]
    for k, (dr, dc) in enumerate(d.STEP_RC):
        want = m.cell_id[r + dr, (c + dc) % 28]
        assert m.nbr[m.pacman_start, k] == want


def test_tunnel_connects_both_sides(m):
    left, right = m.cell_id[14, 0], m.cell_id[14, 27]
    assert set(m.tunnel_cells) == {left, right}
    assert m.nbr[left, d.WEST] == right and m.nbr[right, d.EAST] == left


def test_pacman_cannot_enter_house_or_door(m):
    forbidden = set(m.house_cells) | set(m.door_cells)
    targets = {int(m.nbr[i, k]) for i in range(320) for k in range(4) if m.move_mask["pacman"][i, k]}
    assert not (targets & forbidden)
    assert (m.move_mask["fly"] == (m.nbr >= 0)).all()
    # 문 칸은 팩맨의 마스크에서 막히지만 초파리에게는 열려 있다
    door = m.door_cells[0]
    above = m.nbr[door, d.NORTH]
    assert not m.move_mask["pacman"][above, d.SOUTH] and m.move_mask["fly"][above, d.SOUTH]


def test_starts_dots_and_fruit(m):
    assert len(set(m.fly_starts)) == 4
    for c in [m.pacman_start, m.fruit_cell, *m.fly_starts]:
        assert 0 <= c < 320
    assert m.dots.shape == m.energizers.shape == (320,)
    assert m.energizers.sum() == 4 and not (m.dots & m.energizers).any()
    assert m.dots.sum() > 200
    assert not m.dots[list(m.house_cells)].any()
    assert len(m.house_cells) > 0 and HOUSE == 4


def test_graph_is_connected(m):
    seen, stack = {0}, [0]
    while stack:
        for j in m.nbr[stack.pop()]:
            if j >= 0 and j not in seen:
                seen.add(int(j))
                stack.append(int(j))
    assert len(seen) == 320
