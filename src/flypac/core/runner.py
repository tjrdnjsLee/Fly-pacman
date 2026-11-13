"""실행기 (TRD 6). 한 틱 순서(6.1)와 종료 처리(6.2)는 core/tick.py 한 곳에만 있다. 실행기는 그것을 쓴다.

B개 게임 칸을 나란히 돌린다. 칸마다 자기 시드로 `n_games`판을 연속으로 하고,
판 초기화는 기록 뒤에 core/tick.py에서 한다. 엔진은 스스로 초기화하지 않는다.
"""

from __future__ import annotations

import numpy as np

from flypac.core.rng import game_rngs
from flypac.core.tick import TickCore
from flypac.engine.fake import FakeVecGame
from flypac.engine.game import VecGame
from flypac.field.odor import Field
from flypac.ghosts.random_ghost import RandomGhost
from flypac.maps.original import load_original
from flypac.pacman.rule_based import RuleBasedPacman
from flypac.regimes import make_regime


class NullRecorder:
    """아무것도 기록하지 않는 기록기 (시험 · 연기 시험용). Recorder와 같은 메서드를 가진다."""

    def on_events(self, game, events, view): ...
    def on_life_end(self, game, view, events): ...
    def on_game_end(self, game, view, events): ...
    def replay_frame(self, game, views, obs, odor, phero, brain): ...
    def replay_end(self, game, keep): ...
    def close(self): ...


def check_loop_limit(cfg) -> None:
    """무한 루프 한도는 판 수 × (감시 장치 틱 + 1) 이상이어야 한다 (DECISIONS 28.2, 33.3).

    엔진은 `tick + 1 > watchdog_ticks`에서 판을 중단하므로 한 판은 최대 watchdog_ticks + 1틱이다.
    두 키 중 하나라도 없으면 검사하지 않는다."""
    engine = cfg.get("engine", None)
    if "n_games" not in cfg or engine is None or "watchdog_ticks" not in engine:
        return
    n_games, watchdog = int(cfg.n_games), int(engine.watchdog_ticks)
    limit = int(cfg.run.max_loop_ticks)
    need = n_games * (watchdog + 1)
    if limit < need:
        raise ValueError(f"run.max_loop_ticks({limit})가 n_games({n_games}) × (engine.watchdog_ticks({watchdog}) + 1) = "
                         f"{need}보다 작다 (DECISIONS 33.3: 한 판은 최대 watchdog_ticks + 1틱)")


class Runner(TickCore):
    """실험 실행기: TickCore에 부품 조립 · views 규칙 검사 · 무한 루프 감시 · 기록기 닫기를 더한다."""

    def __init__(self, cfg, map, game: VecGame, engine_rngs, odor: Field, phero: Field | None, pacman, ghosts,
                 regime, recorder):
        replay = cfg.get("replay", {})
        super().__init__(map, game, engine_rngs, ghosts, condition=str(cfg.condition), pacman=pacman,
                         odor=odor, phero=phero, regime=regime, recorder=recorder,
                         c0=float(cfg.odor.c0), s_odor=float(cfg.odor.s),
                         reset_odor_on_life_loss=bool(cfg.odor.reset_on_life_loss),
                         n_games=int(cfg.n_games),
                         keep_games=replay.get("keep_games", []) if replay else [])
        self.cfg = cfg

    # ---- 초기화 ----
    def _check_views_rule(self) -> None:
        """실행 시작 때 한 번: 조종기가 views를 쓰는지가 조건과 맞아야 한다 (TRD 5.5, 원칙 2)."""
        if bool(self.ghosts.uses_views) != self.pass_views:
            raise ValueError(
                f"조건 {self.condition!r}에서 조종기 {type(self.ghosts).__name__}의 uses_views="
                f"{self.ghosts.uses_views}: views를 쓰는 조종기는 조건 'b1'에서만 허용한다")

    def _reset_all(self) -> None:
        self._check_views_rule()
        self.start()

    # ---- 한 틱 ----
    def tick(self) -> None:
        views = self.begin()                  # 1
        self.finish(self.pacman.act(views))   # 2~13

    def run(self) -> dict:
        check_loop_limit(self.cfg)
        self._reset_all()
        guard = int(self.cfg.run.max_loop_ticks)
        while not self.finished.all():
            if self.total_ticks >= guard:
                raise RuntimeError(f"무한 루프 감시: {guard}틱 안에 끝나지 않았다")
            self.tick()
        close = getattr(self.recorder, "close", None)
        if close is not None:
            close()
        return {"ticks": self.total_ticks, "games": list(self.results)}

LEARNING_CONDITIONS = ("regime1", "regime2", "regime3")  # 가중치를 저장하는 조건 (B0 · B1은 학습이 없다)


def build_runner(cfg, recorder=None) -> Runner:
    """설정에서 부품을 만들어 Runner를 돌려준다. 기록기를 주지 않으면 NullRecorder."""
    amap = load_original()
    seeds = [int(s) for s in cfg.seeds]
    B = len(seeds)
    # 엔진 난수: (시드, 판 번호, "engine"). 판 1부터, 판마다 다시 만들어 reset_game(rng)으로 준다 (DECISIONS 11절)
    engine_rngs = game_rngs(seeds, "engine")
    kind = cfg.engine.kind
    if kind == "fake":
        game = FakeVecGame(B, amap, cfg, [engine_rngs[b] for b in range(B)])
    elif kind == "real":
        game = VecGame(B, amap, cfg, [engine_rngs[b] for b in range(B)])
    else:
        raise ValueError(f"알 수 없는 engine.kind: {kind}")
    odor = Field(amap, float(cfg.odor.D), float(cfg.odor.lam))
    condition = cfg.condition
    phero = Field(amap, float(cfg.phero.D), float(cfg.phero.lam)) if condition == "regime3" else None
    pacman_kind = cfg.pacman.kind
    if pacman_kind == "rule_based":
        pacman = RuleBasedPacman(amap, cfg, game_rngs(seeds, "pacman"))
    elif pacman_kind == "ppo":
        # 학습한 체크포인트로 행동한다 (torch 필요: uv sync --extra pacman). 값은 모두 설정에 적는다.
        from flypac.pacman.ppo import PPOPacman
        pc = cfg.pacman
        missing = [k for k in ("checkpoint", "device", "eps") if k not in pc]
        if missing:
            raise ValueError(f"pacman.kind: ppo에는 pacman.{', pacman.'.join(missing)} 설정이 필요하다")
        if "checkpoint_sha256" in pc:  # DECISIONS 28.3: 체크포인트가 설정의 해시와 같아야 한다
            from flypac.recording.envinfo import checkpoint_sha256
            actual = checkpoint_sha256(pc.checkpoint)
            if actual != str(pc.checkpoint_sha256):
                raise ValueError(f"체크포인트 sha256이 설정과 다르다: {pc.checkpoint}의 {actual}, 설정 {pc.checkpoint_sha256}")
        pacman = PPOPacman(str(pc.checkpoint), str(pc.device), float(pc.eps), game_rngs(seeds, "pacman"),
                           run_cfg=cfg)
    else:
        raise NotImplementedError(f"pacman.kind={pacman_kind}")
    if cfg.ghosts.kind == "random":
        ghosts = RandomGhost(cfg, B, game_rngs(seeds, "ghost"))
    elif cfg.ghosts.kind == "flybrain":
        from flypac.brain.flybrain import FlyBrain, load_brain_config
        ghosts = FlyBrain(load_brain_config(cfg.ghosts.config), B, seeds, condition=str(condition),
                          evaluate=bool(cfg.get("evaluate", False)))
    elif cfg.ghosts.kind == "original":
        from flypac.ghosts.original import OriginalGhosts
        ghosts = OriginalGhosts(cfg, B, game_rngs(seeds, "ghost"))
        game.configure_condition(cfg, str(condition))
    else:
        raise NotImplementedError(f"ghosts.kind={cfg.ghosts.kind}")
    regime = make_regime(cfg)
    runner = Runner(cfg, amap, game, engine_rngs, odor, phero, pacman, ghosts, regime, recorder or NullRecorder())
    if "weights" in cfg and cfg.ghosts.kind == "flybrain" and str(condition) in LEARNING_CONDITIONS:
        # 가중치 저장 (DECISIONS 26.1): 학습 조건에서만. B0 · B1은 저장할 학습이 없다
        from flypac.recording.weights import WeightWriter
        if getattr(runner.recorder, "run_dir", None) is None:
            raise ValueError("weights 설정에는 결과 폴더(기록기)가 필요하다")
        runner.weight_writer = WeightWriter.from_config(cfg, runner.recorder)
    return runner
