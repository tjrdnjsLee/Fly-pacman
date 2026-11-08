"""학습 설정 해석과 시작 검사 (torch 없이 돈다).

- 커리큘럼 단계마다 상대 유령과 **조건 이름**을 설정에 직접 적는다. 이름을 유령 종류에서 추론하지 않는다.
  평가 상대도 `training.eval_opponent`에 같은 모양으로 적는다 (DECISIONS 13.3절).
- 시작 검사(`require_ready`): 학습에 쓸 모든 부품(엔진 숫자, 엔진, 단계마다의 상대 유령, 조건 규칙, 평가 상대)을
  학습을 시작하기 전에 한 번씩 만들어 보고, 만들 수 없는 것을 모아서 한꺼번에 알린다. 본 학습 설정(진짜 엔진 ·
  원작 유령)이 부품이 모자란 채 몇 시간을 돌다 죽는 일을 막는다.
"""

from __future__ import annotations

import numpy as np

from flypac.core.rng import game_rngs
from flypac.maps.original import load_original
from flypac.pacman.obs import engine_constants
from flypac.pacman.train_env import PacmanTrainEnv, check_condition, make_game, make_ghosts


def _why(e: Exception) -> str:
    """오류 종류와 설명 (설명이 비어 있으면 종류만: 자리 파일의 NotImplementedError)."""
    return f"{type(e).__name__}: {e}" if str(e) else f"{type(e).__name__} (아직 구현되지 않았다)"


class PreflightError(RuntimeError):
    """시작 검사 실패. 메시지에 만들 수 없는 부품이 모두 적혀 있다."""


def curriculum_phases(p) -> list[tuple[int, str, str]]:
    """[(from_update, 유령 종류, 조건 이름)]. 조건 이름이 빠진 단계는 오류다."""
    phases = []
    for i, ph in enumerate(p.curriculum):
        if "condition" not in ph:
            raise ValueError(f"pacman_train.curriculum[{i}]에 condition을 직접 적는다 "
                             "(예: random 유령은 regime1, original 유령은 b1)")
        phases.append((int(ph["from_update"]), str(ph["ghosts"]), str(ph["condition"])))
    if not phases or phases[0][0] != 0:
        raise ValueError("pacman_train.curriculum의 첫 단계는 from_update: 0 이어야 한다")
    return phases


def eval_opponent(p) -> tuple[str, str]:
    """(유령 종류, 조건 이름): 평가 상대."""
    try:
        op = p.training.eval_opponent
        return str(op["ghosts"]), str(op["condition"])
    except (AttributeError, KeyError):
        raise ValueError("pacman_train.training.eval_opponent에 {ghosts: ..., condition: ...}를 적는다") from None


def check_problems(cfg) -> list[str]:
    """학습을 시작하기 전에 만들어 볼 수 없는 부품의 목록. 비어 있으면 준비된 것이다."""
    problems: list[str] = []
    p = cfg.pacman_train
    seed = int(p.seed)
    try:
        engine_constants(cfg)
    except Exception as e:  # noqa: BLE001 - 무엇이든 부품 실패로 모은다
        problems.append(f"엔진 숫자(engine.power_s · engine.lives · time.v_max_tiles_per_s): {e}")
    combos: list[tuple[str, str, str]] = []  # (출처, 유령, 조건)
    try:
        combos += [(f"커리큘럼 단계 {i}", g, c) for i, (_, g, c) in enumerate(curriculum_phases(p))]
        g, c = eval_opponent(p)
        combos.append(("평가 상대", g, c))
    except ValueError as e:
        problems.append(f"설정: {e}")
    amap = load_original()
    try:
        make_game(cfg, 1, amap, game_rngs([seed], "engine"))
    except Exception as e:  # noqa: BLE001
        problems.append(f"엔진 (engine.kind={cfg.engine.get('kind')}): {_why(e)}")
    for src, g, c in combos:
        try:
            check_condition(make_ghosts(g, cfg, 1, [seed]), c)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{src}의 유령 {g!r} (조건 {c!r}): {_why(e)}")
    if not problems:  # 부품이 모두 만들어지면 조합마다 한 스텝을 실제로 돌려 본다
        unique: dict[tuple[str, str], str] = {}
        for src, g, c in combos:
            unique.setdefault((g, c), src)
        for (g, c), src in unique.items():
            try:
                env = PacmanTrainEnv(cfg, 1, seed, g, c)
                env.reset()
                env.step(np.array([int(np.flatnonzero(env.legal_masks()[0])[0])]))
            except Exception as e:  # noqa: BLE001
                problems.append(f"{src} (유령 {g!r}, 조건 {c!r})의 환경 한 스텝: {_why(e)}")
    return problems


def require_ready(cfg) -> None:
    problems = check_problems(cfg)
    if problems:
        raise PreflightError("시작 검사 실패: 만들 수 없는 부품이 있다. 학습을 시작하지 않는다.\n"
                             + "\n".join(f" - {m}" for m in problems))
