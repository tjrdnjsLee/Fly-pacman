"""기록기 (TRD 5.8, 13.3, 19.5).

결과 폴더 `runs/<날짜_시각>_<실험명>/`에 config.yaml, git_commit.txt, games.csv, lives.csv, flies.csv,
replay/*.npz를 만든다. CSV는 utf-8-sig로 쓰고 utf-8-sig로 읽는다 (엑셀에서 한글이 깨지지 않게).
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from flypac.core.config import Config
from flypac.core.directions import BACK
from flypac.core.events import Event, Kind, ZCause
from flypac.core.types import FlyObs, GameView
from flypac.recording.envinfo import environment_info
from flypac.regimes import make_regime
from flypac.replay import schema as s
from flypac.replay.writer import ReplayBuffer, replay_filename, write_replay

CSV_ENCODING = "utf-8-sig"
F = 4

GAMES_COLS = ["조건", "시드", "판", "점수", "승자", "틱 수", "누적 틱", "누적 결정 수", "남은 점",
              "포획 수", "활성 후 포획 수", "먹힘 수", "활성 후 먹힘 수"]
LIVES_COLS = ["조건", "시드", "판", "목숨 번호", "종료 원인", "포획까지 틱", "잡은 초파리", "출발 전 포획 여부"] + [
    f"먹힘 수_초파리{k}" for k in range(F)]
FLIES_COLS = ["조건", "시드", "판", "초파리", "합계 R", "보상 사건 수", "결정에서 보상까지 지연(결정 수)", "포획",
              "먹힘", "상태 전환 시 반전 기회", "상태 전환 시 반전 횟수",
              "파워 시작 반전 기회", "파워 시작 반전 횟수", "파워 종료 반전 기회", "파워 종료 반전 횟수"]

_CAUSE = {"CATCH": "포획", "CLEAR": "클리어", "CENSORED": "중단"}


def make_run_dir(cfg, root: Path = Path("runs")) -> Path:
    """`runs/<날짜_시각>_<실험명>/` 경로 (같은 초에 겹치면 번호를 붙인다). 폴더는 만들지 않는다."""
    base = Path(root) / f"{datetime.now():%Y%m%d_%H%M%S}_{cfg.experiment}"
    path, n = base, 1
    while path.exists():
        n += 1
        path = Path(f"{base}_{n}")
    return path


def _git_commit() -> str:
    repo = Path(__file__).resolve().parents[3]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True,
                               timeout=10, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class _Csv:
    def __init__(self, path: Path, columns: list[str]):
        self._fh = open(path, "w", encoding=CSV_ENCODING, newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(columns)

    def row(self, values) -> None:
        self._w.writerow(values)

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class Recorder:
    def __init__(self, run_dir: Path, cfg):
        """결과 폴더 생성, 설정 · 커밋 번호 저장."""
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        self.condition = str(cfg.condition)
        self.regime = make_regime(cfg)
        saved_cfg = cfg.to_dict()
        if hasattr(self.regime, "p") and hasattr(self.regime, "q"):
            reward = saved_cfg.get("reward")
            if not isinstance(reward, dict):
                reward = {}
            reward.update(p=self.regime.p, q=self.regime.q)
            saved_cfg["reward"] = reward

        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(saved_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (self.run_dir / "environment.json").write_text(
            json.dumps(environment_info(cfg), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.git_commit = _git_commit()
        (self.run_dir / "git_commit.txt").write_text(self.git_commit + "\n", encoding="utf-8")
        self.seeds = [int(x) for x in cfg.seeds]
        B = len(self.seeds)
        self.games_csv = _Csv(self.run_dir / "games.csv", GAMES_COLS)
        self.lives_csv = _Csv(self.run_dir / "lives.csv", LIVES_COLS)
        self.flies_csv = _Csv(self.run_dir / "flies.csv", FLIES_COLS)
        rep = cfg.get("replay", {})
        self.replay_brain = bool(rep.get("brain", False)) if rep else False
        self.replay_max_bytes = int(rep.get("max_bytes", 0)) if rep else 0  # TODO(미결 17): 저장 용량 상한
        self.replay_bytes = 0
        self.replay_with_phero = self.condition == "regime3"
        self.config_hash = Config(saved_cfg).hash()
        self.replay_files: list[Path] = []
        # 칸(게임)별 상태
        self.game_no = np.ones(B, dtype=np.int64)
        self.life_no = np.ones(B, dtype=np.int64)
        self.ticks_total = np.zeros(B, dtype=np.int64)
        self.decisions_total = np.zeros(B, dtype=np.int64)
        self.life_ticks = np.zeros(B, dtype=np.int64)
        self.game_stats = [self._new_game_stats() for _ in range(B)]
        self.life_stats = [self._new_life_stats() for _ in range(B)]
        self.pending: list[list[Event]] = [[] for _ in range(B)]
        self.buffers: list[ReplayBuffer | None] = [None] * B

    # ---- 통계 묶음 ----
    @staticmethod
    def _new_game_stats() -> dict:
        return {"catch": 0, "catch_active": 0, "eaten": 0, "eaten_active": 0,
                "sumR": np.zeros(F), "reward_events": np.zeros(F, dtype=np.int64),
                "delay_sum": np.zeros(F), "delay_n": np.zeros(F, dtype=np.int64),
                "dec_since": np.zeros(F, dtype=np.int64),
                "catches_by": np.zeros(F, dtype=np.int64), "eaten_by": np.zeros(F, dtype=np.int64),
                "rev_chances": np.zeros(F, dtype=np.int64), "rev_count": np.zeros(F, dtype=np.int64),
                # 원인별 (0 시작, 1 종료). 원인을 모르는 기회는 합계에서 두 원인을 뺀 값이다 (DECISIONS 23)
                "rev_chances_by": np.zeros((2, F), dtype=np.int64), "rev_count_by": np.zeros((2, F), dtype=np.int64),
                # 반전 기회를 연 Z_CHANGE의 원인: -1 없음, 0 POWER_START, 1 POWER_END, 그 사건의 틱 번호
                "open_cause": np.full(F, -1, dtype=np.int64), "open_tick": np.zeros(F, dtype=np.int64)}

    @staticmethod
    def _new_life_stats() -> dict:
        return {"eaten_by": np.zeros(F, dtype=np.int64), "catch_fly": -1, "catch_inactive": False,
                "catch_tick": None}

    def _buffer(self, b: int) -> ReplayBuffer:
        if self.buffers[b] is None:
            self.buffers[b] = ReplayBuffer(self.replay_with_phero, self.replay_brain)
        return self.buffers[b]

    # ---- 매 틱 ----
    def on_events(self, game: int, events: list[Event], view: GameView) -> None:
        """사건 누적."""
        b = game
        self.ticks_total[b] += 1
        self.life_ticks[b] += 1
        gs, ls, buf = self.game_stats[b], self.life_stats[b], self._buffer(b)
        for e in events:
            buf.add_event(e.tick, e.kind, e.fly)
            if e.kind is Kind.CATCH:
                gs["catch"] += 1
                gs["catch_active"] += int(e.active)
                gs["catches_by"][e.fly] += 1
                ls["catch_fly"], ls["catch_inactive"], ls["catch_tick"] = e.fly, not e.active, int(self.life_ticks[b])
            elif e.kind is Kind.Z_CHANGE:
                # 파워 시작 · 종료가 반전 기회를 연다. 부활은 열지 않는다 (엔진이 예약을 비운다)
                gs["open_cause"][e.fly] = {ZCause.POWER_START: 0, ZCause.POWER_END: 1}.get(e.cause, -1)
                gs["open_tick"][e.fly] = self.ticks_total[b]
            elif e.kind is Kind.EATEN:
                gs["open_cause"][e.fly] = -1
                gs["eaten"] += 1
                gs["eaten_active"] += int(e.active)
                gs["eaten_by"][e.fly] += 1
                ls["eaten_by"][e.fly] += 1
        self.pending[b] = list(events)

    def replay_frame(self, game: int, views: list[GameView], obs: FlyObs, odor: np.ndarray, phero: np.ndarray,
                     brain: dict | None) -> None:
        """매 틱 현재 판의 재생 정보를 메모리에 쌓음 (TRD 19.4, 19.5).

        brain: 조종기가 낸 {"action": (B,F), "pi": (B,F,4)}, 실행기 7번의 {"committed", "moved"} (B,F) bool
        (없으면 거짓), 그리고 replay.brain이 켜진 경우
        {"kc_active", "pn_input", "mbon_out", "dan", "brain_on"} (모두 앞 축이 B).
        뇌가 아닌 조종기는 뇌 필드를 주지 않는다: 그 로그에는 뇌 필드 묶음이 없고 brain_on은 모두 거짓이다.
        """
        b = game
        v = views[b]
        decided = obs.decide[b] & (not v.frozen)
        self.decisions_total[b] += int(decided.sum())
        gs = self.game_stats[b]
        gs["dec_since"] += decided.astype(np.int64)
        chance = decided & obs.legal[b, :, BACK]
        gs["rev_chances"] += chance
        brain = brain or {}
        action = np.asarray(brain.get("action", np.full((len(views), F), -1)))[b]
        no = np.zeros((len(views), F), dtype=bool)
        committed = np.asarray(brain.get("committed", no), dtype=bool)[b]
        # 반전 횟수: 반전 기회에서 확정된 행동이 BACK인 횟수. 지연 반전은 저장한 틱(committed)에 센다 (DECISIONS 23)
        reversed_ = chance & committed & (action == BACK)
        gs["rev_count"] += reversed_
        # 원인별: 기회를 연 Z_CHANGE는 같은 틱(파워 시작) 또는 바로 앞 틱(파워 종료)에 있다. 더 오래된 것은 버린다
        fresh = (gs["open_cause"] >= 0) & (self.ticks_total[b] - gs["open_tick"] <= 1)
        for c in (0, 1):
            m = chance & fresh & (gs["open_cause"] == c)
            gs["rev_chances_by"][c] += m
            gs["rev_count_by"][c] += m & reversed_
        gs["open_cause"][chance] = -1  # 반전 기회는 한 번 쓰이면 닫힌다 (BACK이 합법이 아닌 보통 결정은 닫지 않는다)
        # 이번 틱의 보상 사건 (결정은 위에서 먼저 센다)
        for e in self.pending[b]:
            r = self.regime.rewards([e])
            for k in np.flatnonzero(r):
                gs["sumR"][k] += r[k]
                gs["reward_events"][k] += 1
                gs["delay_sum"][k] += gs["dec_since"][k]
                gs["delay_n"][k] += 1
                gs["dec_since"][k] = 0
        self.pending[b] = []

        pi = np.asarray(brain.get("pi", np.zeros((len(views), F, 4))))[b]
        moved = np.asarray(brain.get("moved", no), dtype=bool)[b]
        brain_on = np.asarray(brain.get("brain_on", no), dtype=bool)[b]
        frame = dict(
            pac_cell=v.pac_cell, pac_dir=v.pac_dir, pac_acc=v.pac_acc,
            fly_cell=v.fly_cell, fly_dir=v.fly_dir,
            fly_acc=np.zeros(F) if v.fly_acc is None else v.fly_acc,
            fly_z=v.fly_z, fly_in_house=v.fly_in_house, lives=v.lives, power_ticks=v.power_ticks,
            score=v.score, fruit=int(v.fruit), odor_field=odor[b],
            sense=np.concatenate([obs.odor[b], obs.phero[b]], axis=1), action=action, pi=pi,
            committed=committed, moved=moved, brain_on=brain_on,
        )
        if self.replay_with_phero:
            frame["phero_field"] = phero[b]
        buf = self._buffer(b)
        given = [n for n in s.BRAIN_FIELDS if n in brain]
        if self.replay_brain and given:
            if len(given) != len(s.BRAIN_FIELDS):
                missing = [n for n in s.BRAIN_FIELDS if n not in brain]
                raise ValueError(f"조종기가 뇌 필드를 일부만 주었다: 빠진 것 {missing}")
            for n in s.BRAIN_FIELDS:
                frame[n] = np.asarray(brain[n])[b]
            buf.with_brain = True
        else:  # 뇌가 아닌 조종기(무작위 유령, B1): 뇌 필드 묶음을 두지 않는다 (TRD 19.13)
            frame["brain_on"] = np.zeros(F, dtype=bool)  # 뇌 필드가 없으면 "있음" 표시도 없다
            buf.with_brain = False
        buf.add_frame(**frame)

    # ---- 종료 (초기화 전 상태) ----
    def on_life_end(self, game: int, view: GameView, events: list[Event]) -> None:
        """초기화 전의 종료 상태로 lives.csv 한 줄."""
        b = game
        ls = self.life_stats[b]
        self.game_stats[b]["open_cause"][:] = -1  # 목숨이 끝나면 열린 반전 기회는 없어진다
        winner = next((e.winner for e in events if e.kind is Kind.GAME_OVER), None)
        if any(e.kind is Kind.CATCH for e in events):
            cause = _CAUSE["CATCH"]
        elif winner is not None and str(winner) == "PACMAN":
            cause = _CAUSE["CLEAR"]
        else:
            cause = _CAUSE["CENSORED"]
        caught = cause == _CAUSE["CATCH"]
        self.lives_csv.row([self.condition, self.seeds[b], int(self.game_no[b]), int(self.life_no[b]), cause,
                            ls["catch_tick"] if caught else "", ls["catch_fly"] if caught else "",
                            int(ls["catch_inactive"]) if caught else "", *map(int, ls["eaten_by"])])
        self.life_no[b] += 1
        self.life_ticks[b] = 0
        self.life_stats[b] = self._new_life_stats()
        self.game_stats[b]["dec_since"][:] = 0

    def on_game_end(self, game: int, view: GameView, events: list[Event]) -> None:
        """초기화 전의 종료 상태로 games.csv 한 줄, flies.csv 네 줄."""
        b = game
        gs = self.game_stats[b]
        winner = next(e.winner for e in events if e.kind is Kind.GAME_OVER)
        left = int(view.dots.sum() + view.energizers.sum())
        base = [self.condition, self.seeds[b], int(self.game_no[b])]
        self.games_csv.row(base + [int(view.score), str(winner), int(view.tick), int(self.ticks_total[b]),
                                   int(self.decisions_total[b]), left, gs["catch"], gs["catch_active"], gs["eaten"], gs["eaten_active"]])
        for k in range(F):
            n = gs["delay_n"][k]
            self.flies_csv.row(base + [k, repr(float(gs["sumR"][k])), int(gs["reward_events"][k]),
                                       f"{gs['delay_sum'][k] / n:.6g}" if n else "",
                                       int(gs["catches_by"][k]), int(gs["eaten_by"][k]),
                                       int(gs["rev_chances"][k]), int(gs["rev_count"][k]),
                                       int(gs["rev_chances_by"][0][k]), int(gs["rev_count_by"][0][k]),
                                       int(gs["rev_chances_by"][1][k]), int(gs["rev_count_by"][1][k])])
        for c in (self.games_csv, self.lives_csv, self.flies_csv):
            c.flush()
        self.game_stats[b] = self._new_game_stats()
        self.life_no[b] = 1

    def replay_end(self, game: int, keep: bool) -> None:
        """판이 끝나면 남길 판(keep)만 파일로 저장. 판 번호는 on_game_end 직후에 올라간다."""
        b = game
        buf, self.buffers[b] = self.buffers[b], None
        if keep and buf is not None and len(buf) > 0:
            arrays = buf.arrays(s.make_meta(self.condition, self.seeds[b], int(self.game_no[b]),
                                            self.git_commit, self.config_hash, "original"))
            path = self.run_dir / "replay" / replay_filename(self.condition, self.seeds[b], int(self.game_no[b]))
            write_replay(path, arrays)
            size = path.stat().st_size
            if self.replay_max_bytes and self.replay_bytes + size > self.replay_max_bytes:
                path.unlink()  # 저장 용량 상한을 넘으면 버린다
            else:
                self.replay_bytes += size
                self.replay_files.append(path)
        self.game_no[b] += 1

    def close(self) -> None:
        for c in (self.games_csv, self.lives_csv, self.flies_csv):
            c.close()
