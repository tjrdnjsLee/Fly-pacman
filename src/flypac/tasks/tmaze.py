"""T자 미로 작은 과제 환경 (TRD 10.8, DECISIONS 17.4): 과제 A · B · C · D.

같은 FlyBrain 코드와 학습 규칙을 쓰고 환경만 작다. 갈림길 하나, 선택지는 좌 · 우 둘이다.

한 시행 = 복도 (corridor_ticks틱, 결정 없음) + 갈림길 1틱 (결정). 입력 odor = [좌, 앞, 우, 뒤]는 복도에서도
갈림길에서도 같다: 갈림길의 두 길에서 오는 냄새가 복도로도 들어와 증거 x, h가 쌓인다. 합법 방향은 복도에서 앞
하나, 갈림길에서 좌 · 우 둘이다 (뒤 · 앞 없음).

  A. 방향: 양쪽에 같은 세기(amp)의 입력, 한쪽(reward_side)에만 보상. reverse면 반대쪽.
  B. 냄새: 냄새는 좌 · 우 중 한쪽(무작위 균등)에만 amp, 보상은 냄새 쪽. reverse면 반대쪽.
  C. 상태 전환: B와 같은 입력에 z ∈ {0, 1}. z = 0이면 냄새 쪽, z = 1이면 반대쪽에 보상.
     (냄새 방향 × z) 네 조합 균등. 두 상태의 좌 · 우 선택지는 같다. reverse면 z 대응을 맞바꾼다.
  D. 지연: B에서 첫 결정 뒤에 결정이 n번 더 있고 보상은 마지막 결정 직후에 온다. 추가 결정(필러)은 보상과 무관한
     임의 냄새 단서(좌 · 우 균등, z = 0)가 있는 갈림길이다 (복도 filler_corridor_ticks틱).

보상은 정답 +reward_correct, 오답 reward_wrong을 마지막 결정 직후 학습자에게 준다 (give_reward로 넘길 R).
FlyBrain은 이 보상을 다음 결정의 확정 때 학습한다 (TRD 10.4: R_n은 직전 결정의 결과).

난수는 core.rng의 (시드, 판 번호, "task") 칸별 생성기만 쓴다. 한 학습 시드가 게임 한 칸이고, 칸의 시행 열은 그 칸의
시드와 판 번호로만 정해진다 (묶음 크기나 칸 순서와 무관).
"""

from __future__ import annotations

import numpy as np

from flypac.core import directions as d
from flypac.core.config import Config
from flypac.core.rng import game_rngs
from flypac.core.types import FlyObs

F = 4
LEFT, FWD, RIGHT = (d.EGO_NAMES.index(n) for n in ("left", "forward", "right"))
TASKS = ("A", "B", "C", "D")
LEARNER_BASE = {"D": "B"}   # 과제 D는 B를 지연시킨 것


def correct_action(task: str, cue: int, z: int, reward_side: str = "L", reverse: bool = False) -> int:
    """정답 방향 (좌 = LEFT, 우 = RIGHT). cue: 냄새 쪽 (0 = 좌, 1 = 우). 정의가 한 곳에 있다."""
    if task == "A":
        a = LEFT if reward_side == "L" else RIGHT
        if reverse:
            a = RIGHT if a == LEFT else LEFT
        return a
    smell = LEFT if cue == 0 else RIGHT
    other = RIGHT if smell == LEFT else LEFT
    if task in ("B", "D"):
        return other if reverse else smell
    if task == "C":
        ans = smell if int(z) == 0 else other
        return (RIGHT if ans == LEFT else LEFT) if reverse else ans
    raise ValueError(f"알 수 없는 과제: {task!r}")


def combos(task: str):
    """균등하게 섞을 조합 목록 [(cue, z), ...]. A는 하나."""
    if task == "A":
        return [(0, 0)]
    if task in ("B", "D"):
        return [(0, 0), (1, 0)]
    if task == "C":
        return [(c, z) for c in (0, 1) for z in (0, 1)]
    raise ValueError(f"알 수 없는 과제: {task!r}")


class TMaze:
    """B개 칸(학습 시드마다 하나)이 같은 틱 구조로 나란히 돈다. 배열의 앞 축은 B, 초파리 축은 F = 4."""

    def __init__(self, cfg: Config, task: str, seeds, *, n_delay: int = 0, reward_side: str = "L",
                 reverse: bool = False, game_no: int = 1):
        if task not in TASKS:
            raise ValueError(f"알 수 없는 과제: {task!r}")
        if reward_side not in ("L", "R"):
            raise ValueError("reward_side는 'L' 또는 'R'")
        if task != "D" and n_delay != 0:
            raise ValueError("지연값 n은 과제 D에만 있다")
        e = cfg.env
        self.cfg = cfg
        self.task, self.n_delay, self.reward_side, self.reverse = task, int(n_delay), reward_side, bool(reverse)
        self.seeds = [int(s) for s in seeds]
        self.B, self.game_no = len(self.seeds), int(game_no)
        self.learner = int(e.learner_fly)
        self.L, self.Lf = int(e.corridor_ticks), int(e.filler_corridor_ticks)
        self.amp, self.ratio = float(e.amp), float(e.other_side_ratio)
        self.r_ok, self.r_bad = float(e.reward_correct), float(e.reward_wrong)
        self.balanced = bool(e.balanced)
        self.rngs = game_rngs(self.seeds, "task", self.game_no)
        self._combos = combos(task)
        self._queue: list[list[tuple[int, int]]] = [[] for _ in range(self.B)]
        self.trial_log: list[dict] = []      # 끝난 시행 요약 (시행 순서, 항목마다 (B,) 배열)
        self.trials_done = 0
        self.ticks = 0
        self._new_trial()

    # ---- 시행 구성 ----
    def fresh(self, game_no: int) -> "TMaze":
        """같은 설정 · 같은 시드, 새 판 번호의 환경 (평가용: 학습과 시행 열이 독립)."""
        return TMaze(self.cfg, self.task, self.seeds, n_delay=self.n_delay, reward_side=self.reward_side,
                     reverse=self.reverse, game_no=game_no)

    def _draw(self, b: int) -> tuple[int, int]:
        if not self.balanced:
            return self._combos[int(self.rngs[b].integers(len(self._combos)))]
        if not self._queue[b]:
            order = self.rngs[b].permutation(len(self._combos))
            self._queue[b] = [self._combos[int(i)] for i in order]
        return self._queue[b].pop(0)

    def _new_trial(self) -> None:
        draws = [self._draw(b) for b in range(self.B)]
        self.cue = np.array([c for c, _ in draws], dtype=np.int64)       # 냄새 쪽 (0 좌, 1 우)
        self.z = np.array([z for _, z in draws], dtype=np.int64)
        self.correct = np.array([correct_action(self.task, int(c), int(z), self.reward_side, self.reverse)
                                 for c, z in draws], dtype=np.int64)
        self.dec = 0                       # 지금 몇 번째 결정을 향해 가는 중인가
        self.t = 0                         # 이 결정 앞의 복도에서 지난 틱
        self.filler_cue = np.zeros(self.B, dtype=np.int64)
        self.choice = np.full(self.B, -1, dtype=np.int64)

    def _corridor_len(self) -> int:
        return self.L if self.dec == 0 else self.Lf

    @property
    def at_junction(self) -> bool:
        return self.t >= self._corridor_len()

    @property
    def ticks_per_trial(self) -> int:
        return self.L + 1 + self.n_delay * (self.Lf + 1)

    def _stimulus(self) -> np.ndarray:
        """(B, 4) 지금 결정 구간의 냄새 입력 [좌, 앞, 우, 뒤]."""
        o = np.zeros((self.B, 4))
        if self.task == "A":
            o[:, LEFT] = o[:, RIGHT] = self.amp
            return o
        cue = self.cue if self.dec == 0 else self.filler_cue
        side = np.where(cue == 0, LEFT, RIGHT)
        other = np.where(cue == 0, RIGHT, LEFT)
        o[np.arange(self.B), side] = self.amp
        o[np.arange(self.B), other] = self.amp * self.ratio
        return o

    # ---- 관측 · 진행 ----
    def obs(self) -> FlyObs:
        B, f = self.B, self.learner
        odor = np.zeros((B, F, 4))
        odor[:, f] = self._stimulus()
        z = np.zeros((B, F))
        if self.dec == 0:
            z[:, f] = self.z
        legal = np.zeros((B, F, 4), dtype=bool)
        decide = np.zeros((B, F), dtype=bool)
        if self.at_junction:
            legal[:, f, LEFT] = legal[:, f, RIGHT] = True
            decide[:, f] = True
        else:
            legal[:, f, FWD] = True
        return FlyObs(odor=odor, phero=np.zeros((B, F, 4)), z=z, legal=legal, decide=decide)

    def step(self, actions: np.ndarray) -> np.ndarray:
        """한 틱 진행. actions (B, F) int (propose의 반환). 반환 R (B, F): 보상은 마지막 결정 직후 학습자에게만."""
        R = np.zeros((self.B, F))
        self.ticks += 1
        if not self.at_junction:
            self.t += 1
            return R
        a = np.asarray(actions)[:, self.learner]
        if not np.isin(a, (LEFT, RIGHT)).all():
            raise ValueError(f"갈림길의 행동은 좌 · 우여야 한다: {a.tolist()}")
        if self.dec == 0:
            self.choice = a.astype(np.int64)
        if self.dec < self.n_delay:
            self.dec += 1
            self.t = 0
            self.filler_cue = np.array([int(self.rngs[b].integers(2)) for b in range(self.B)], dtype=np.int64)
            return R
        ok = self.choice == self.correct
        R[:, self.learner] = np.where(ok, self.r_ok, self.r_bad)
        self.trial_log.append({"cue": self.cue.copy(), "z": self.z.copy(), "correct_action": self.correct.copy(),
                               "choice": self.choice.copy(), "ok": ok.copy()})
        self.trials_done += 1
        self._new_trial()
        return R
