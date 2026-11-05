import numpy as np

from flypac.core.events import Event, Kind
from flypac.engine.fake import FakeGame, FakeVecGame
from flypac.engine.game import Game, VecGame


class StubGame(Game):
    """VecGame이 규칙 없이 Game에 위임만 하는지 보는 시험용 Game."""

    def __init__(self, map, cfg, rng):
        self.rng_id = id(rng)
        self.resets = []

    def step_pacman(self, action):
        return [Event(Kind.DOT, 0, 0, -1, True)] * action

    def step_flies(self, actions):
        return [], np.asarray(actions) >= 0, np.ones(4, bool)

    def decision_info(self):
        return np.ones((4, 4), bool), np.zeros(4, bool)

    def reset_life(self):
        self.resets.append("life")

    def reset_game(self):
        self.resets.append("game")


class StubVec(VecGame):
    game_cls = StubGame


def test_vecgame_is_thin_wrapper_of_games():
    rngs = [np.random.default_rng(i) for i in range(3)]
    v = StubVec(3, None, None, rngs)
    assert len(v.games) == 3 and all(isinstance(g, StubGame) for g in v.games)
    assert [len(e) for e in v.step_pacman(np.array([0, 1, 2]))] == [0, 1, 2]
    ev, ex, ac = v.step_flies(np.array([[0, -1, 1, 2]] * 3))
    assert ex.shape == ac.shape == (3, 4) and not ex[0, 1]
    legal, decide = v.decision_info()
    assert legal.shape == (3, 4, 4) and decide.shape == (3, 4)


def test_vecgame_reset_only_masked_and_no_auto_reset():
    v = StubVec(3, None, None, [np.random.default_rng(i) for i in range(3)])
    v.step_pacman(np.zeros(3, int))
    assert all(g.resets == [] for g in v.games)
    v.reset_life(np.array([True, False, True]))
    v.reset_game(np.array([False, True, False]))
    assert [g.resets for g in v.games] == [["life"], ["game"], ["life"]]


def test_fake_vec_uses_fake_game():
    assert FakeVecGame.game_cls is FakeGame and VecGame.game_cls is Game
