"""M4 최종 통합 검증의 확인 (DECISIONS 34.3의 C1 ~ C9).

  check_m4.py <학습 묶음 폴더 T> <평가 묶음 폴더 E> [--train-config <yaml>] [--eval-config <yaml>] [--circuit <npz>]

T는 `run_parallel.py configs/m4_check_v1.yaml ...`의 묶음 폴더, E는 `final_eval.py run configs/m4_eval_v1.yaml ...`의
묶음 폴더(runs/m4_eval_v1)다. 항목마다 통과 · 실패와 그 근거가 되는 **개수만** 화면과 `T/m4_check.json`에 낸다.
쓰는 파일은 `T/m4_check.json` 하나다. 하나라도 실패면 종료 코드 1.

**성능 값(점수, 승패 수, W, 보상 합, 회피 비율 값, 신뢰구간)은 내지 않는다** (34.3: M4에서 행동의 근거는 버그
신호뿐이다). `final_eval.build_results`의 반환값은 "오류 없이 돌았다"만 쓰고 버린다. 회피 지표는 "오류 없이
계산됐다"와 상태별 표본 수만 쓴다.

기대하는 값(시드, 판 수, 남길 판, 짝, 중단 비율 문턱)은 두 설정 파일에서 읽는다. 기대하는 해시는 코드에 적지 않는다:
체크포인트는 결과 폴더 config.yaml의 `pacman.checkpoint_sha256`, 회로는 회로 파일을 직접 해시한 값과 대조한다.
항목 하나의 확인이 예외로 끝나도 나머지 항목은 계속 확인한다 (그 항목만 실패, 예외 글을 근거로 남긴다).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import final_eval  # noqa: E402

from flypac.recording.recorder import FLIES_COLS  # noqa: E402
from flypac.recording.weights import weight_filename  # noqa: E402
from flypac.replay.decision_metrics import strongest_avoidance  # noqa: E402
from flypac.replay.reader import read_replay  # noqa: E402
from flypac.replay.writer import replay_filename  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = ROOT / "configs" / "m4_check_v1.yaml"
EVAL_CONFIG = ROOT / "configs" / "m4_eval_v1.yaml"
CIRCUIT = ROOT / "data" / "circuit" / "mb_v1.npz"
OUT_NAME = "m4_check.json"
CSV_ENCODING = "utf-8-sig"

REPLAY_FORMAT = 5                               # 형식 5 (DECISIONS 33.4, 34.3의 C2)
FORMAT5_FIELDS = ("legal", "strongest")
B1 = "b1"                                       # 뇌가 아닌 조건 (원작 유령). 그 밖의 조건이 뇌 조건이다
PHERO_CONDITION = "regime3"                     # 페로몬 장을 남기는 조건
CHANCE_COL = "상태 전환 시 반전 기회"            # 반전 기회의 합계 열 (DECISIONS 23)
CAUSE_COLS = tuple(FLIES_COLS[-4:])             # 원인별 네 열 (DECISIONS 23)
CENSORED = "CENSORED"

ITEMS = (("C1", "실행"), ("C2", "재생 형식"), ("C3", "회피 지표"), ("C4", "반전 기록"), ("C5", "페로몬"),
         ("C6", "가중치"), ("C7", "평가 경로"), ("C8", "해시"), ("C9", "커밋"))


# ---------------------------------------------------------------- 기대하는 값 (설정에서)

@dataclass(frozen=True)
class Spec:
    conditions: tuple[str, ...]       # 다섯 조건 (평가 설정의 final_eval.conditions)
    train_seeds: tuple[int, ...]      # 학습 설정의 seeds
    train_n_games: int                # 학습 설정의 n_games (가중치 파일의 판 번호)
    train_keep: tuple[int, ...]       # 학습 설정의 replay.keep_games
    eval_seeds: tuple[int, ...]       # 평가 설정의 seeds
    eval_n_games: int
    eval_keep: tuple[int, ...]
    pair_train_seeds: tuple[int, ...]  # 평가 설정의 final_eval.train_seeds (칸 i의 학습 시드)
    manifest_game_no: int             # 평가 설정의 final_eval.train_n_games
    censored_max: float               # 평가 설정의 final_eval.censored_warn_fraction (폴더마다 중단 비율의 상한)

    @property
    def brain(self) -> tuple[str, ...]:
        return tuple(c for c in self.conditions if c != B1)

    @property
    def learning(self) -> tuple[str, ...]:
        return tuple(c for c in self.conditions if c in final_eval.LEARNING)

    @property
    def no_learning(self) -> tuple[str, ...]:
        return tuple(c for c in self.conditions if c not in final_eval.LEARNING)


def make_spec(train_cfg: dict, eval_cfg: dict) -> Spec:
    fe = eval_cfg["final_eval"]
    return Spec(conditions=tuple(str(c) for c in fe["conditions"]),
                train_seeds=tuple(int(s) for s in train_cfg["seeds"]), train_n_games=int(train_cfg["n_games"]),
                train_keep=tuple(int(g) for g in train_cfg["replay"]["keep_games"]),
                eval_seeds=tuple(int(s) for s in eval_cfg["seeds"]), eval_n_games=int(eval_cfg["n_games"]),
                eval_keep=tuple(int(g) for g in eval_cfg["replay"]["keep_games"]),
                pair_train_seeds=tuple(int(s) for s in fe["train_seeds"]),
                manifest_game_no=int(fe["train_n_games"]), censored_max=float(fe["censored_warn_fraction"]))


# ---------------------------------------------------------------- 읽기

def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _yaml(path: Path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, encoding=CSV_ENCODING, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(Path(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_dirs(T: Path, spec: Spec) -> dict[str, Path]:
    """학습 결과 폴더: `T/<조건>_p1` (배치 A, 조건당 프로세스 하나)."""
    return {c: Path(T) / f"{c}_p1" for c in spec.conditions}


def eval_dirs(E: Path, spec: Spec, r: "Result") -> dict[str, Path]:
    """평가 결과 폴더: E 안이 아니라 `E.parent / run_dir` (E/index.json의 runs[].run_dir).
    index.json에 정확히 한 번 있는 조건만 돌려주고, 그렇지 않은 조건은 r에 실패 이유로 남긴다."""
    runs = _json(Path(E) / "index.json")["runs"]
    out = {}
    for c in spec.conditions:
        hits = [x for x in runs if str(x["condition"]) == c]
        if r.need(len(hits) == 1, f"평가 묶음의 index.json에 조건 {c}가 {len(hits)}번 있다 (정확히 1번이어야 한다)"):
            out[c] = Path(E).parent / hits[0]["run_dir"]
    return out


def all_dirs(T: Path, E: Path, spec: Spec, r: "Result") -> list[tuple[str, str, Path]]:
    """열 폴더: [(종류 "학습" | "평가", 조건, 폴더)]. 평가 index.json에 없는 조건은 r에 실패 이유로 남기고 뺀다."""
    return ([("학습", c, d) for c, d in train_dirs(T, spec).items()]
            + [("평가", c, d) for c, d in eval_dirs(E, spec, r).items()])


def replay_path(run_dir: Path, cond: str, seed: int, game: int) -> Path:
    return Path(run_dir) / "replay" / replay_filename(cond, seed, game)


class Result:
    """항목 하나의 결과: 근거 개수(counts)와 실패 이유(reasons). 이유가 없으면 통과."""

    def __init__(self):
        self.counts: dict = {}
        self.reasons: list[str] = []

    def need(self, ok: bool, reason: str) -> bool:
        if not ok:
            self.reasons.append(reason)
        return bool(ok)


# ---------------------------------------------------------------- C1 ~ C9

def check_c1(T: Path, E: Path, spec: Spec) -> Result:
    """실행: 학습 index.json의 failed · exit_code, 평가 index.json의 조건, 열 폴더의 games.csv 완결과 중단 비율."""
    r = Result()
    index = _json(Path(T) / "index.json")
    r.counts["학습 failed 수"] = len(index["failed"])
    r.need(not index["failed"], f"학습 묶음의 failed가 비어 있지 않다: {index['failed']}")
    zero = 0
    for c in spec.conditions:
        hits = [x for x in index["runs"] if str(x["condition"]) == c]
        if r.need(len(hits) == 1, f"학습 묶음의 index.json에 조건 {c}가 {len(hits)}번 있다"):
            if r.need(int(hits[0]["exit_code"]) == 0, f"학습 {c}: exit_code가 {hits[0]['exit_code']}이다"):
                zero += 1
    r.counts["학습 exit_code 0인 조건 수"] = zero
    dirs = all_dirs(T, E, spec, r)   # 평가 index.json에 한 번씩 있지 않은 조건은 여기서 실패 이유가 된다
    r.counts["평가 index.json의 조건 수"] = sum(kind == "평가" for kind, _, _ in dirs)
    complete, games, censored = 0, {}, {}
    for kind, c, d in dirs:
        seeds, n = (spec.train_seeds, spec.train_n_games) if kind == "학습" else (spec.eval_seeds, spec.eval_n_games)
        key = f"{kind} {c}"
        try:
            _, rows = _csv(d / "games.csv")
            final_eval.check_rows(c, rows, list(seeds), n)   # (시드, 판) 겹침 없음, 시드마다 판 1 ~ n
        except Exception as e:   # noqa: BLE001  (폴더 하나의 실패를 근거로 남기고 나머지를 계속 본다)
            r.need(False, f"{key}: games.csv가 완결이 아니다 ({type(e).__name__}: {e})")
            continue
        complete += 1
        games[key] = len(rows)
        censored[key] = sum(x["승자"] == CENSORED for x in rows)
        r.need(len(rows) > 0 and censored[key] <= spec.censored_max * len(rows),
               f"{key}: 중단 판 {censored[key]} / {len(rows)}판이 비율 {spec.censored_max}를 넘는다")
    r.counts.update({"games.csv가 완결인 폴더 수": complete, "폴더별 판 수": games, "폴더별 중단 판 수": censored})
    return r


def check_c2(T: Path, E: Path, spec: Spec) -> Result:
    """재생 형식: 학습의 모든 칸에서 남길 판의 재생 파일이 있고 형식 5이며 legal · strongest가 있다."""
    r = Result()
    want = ok = 0
    for c, d in train_dirs(T, spec).items():
        for s in spec.train_seeds:
            for g in spec.train_keep:
                want += 1
                p = replay_path(d, c, s, g)
                if not r.need(p.exists(), f"재생 파일이 없다: {p.name} ({c})"):
                    continue
                try:
                    meta, arrays = read_replay(p)
                except Exception as e:   # noqa: BLE001
                    r.need(False, f"{p.name}: 읽지 못했다 ({type(e).__name__}: {e})")
                    continue
                good = r.need(meta["format_version"] == REPLAY_FORMAT,
                              f"{p.name}: 형식 {meta['format_version']}이다 ({REPLAY_FORMAT}이어야 한다)")
                missing = [k for k in FORMAT5_FIELDS if k not in arrays]
                good &= r.need(not missing, f"{p.name}: {missing}이 없다")
                ok += int(good)
    r.counts.update({"기대한 재생 파일 수": want, "형식이 맞는 파일 수": ok})
    return r


def check_c3(T: Path, E: Path, spec: Spec) -> Result:
    """회피 지표: 뇌 조건의 학습 · 평가 재생 모두에서 strongest_avoidance가 오류 없이 돌고, 합쳐서 z = 0, 1의
    표본이 각각 1 이상이다. 회피 비율 값은 쓰지 않는다 (표본 수만)."""
    r = Result()
    n = {0: 0, 1: 0}
    tdirs, edirs = train_dirs(T, spec), eval_dirs(E, spec, r)
    for kind, dirs, seeds, keep in (("학습", tdirs, spec.train_seeds, spec.train_keep),
                                    ("평가", edirs, spec.eval_seeds, spec.eval_keep)):
        want = ok = 0
        for c in spec.brain:
            for s in seeds:
                for g in keep:
                    want += 1
                    if c not in dirs:   # 평가 index.json에 없는 조건 (이유는 eval_dirs가 남겼다)
                        continue
                    p = replay_path(dirs[c], c, s, g)
                    try:
                        _, arrays = read_replay(p)
                        got = strongest_avoidance(arrays)
                    except Exception as e:   # noqa: BLE001
                        r.need(False, f"{kind} {p.name}: 회피 지표를 계산하지 못했다 ({type(e).__name__}: {e})")
                        continue
                    ok += 1
                    for z in n:
                        n[z] += int(got[z]["n"])
        r.counts[f"{kind} 재생: 기대한 파일 수"] = want
        r.counts[f"{kind} 재생: 오류 없이 계산된 파일 수"] = ok
    r.counts["표본 수 n (z = 0)"] = n[0]
    r.counts["표본 수 n (z = 1)"] = n[1]
    for z in n:
        r.need(n[z] >= 1, f"z = {z}의 표본이 0이다")
    return r


def check_c4(T: Path, E: Path, spec: Spec) -> Result:
    """반전 기록: 학습의 flies.csv에 원인별 네 열이 있다. 반전 기회의 합이 뇌 조건마다 0보다 크고 B1에서는 0이다."""
    r = Result()
    sums, with_cols = {}, 0
    for c, d in train_dirs(T, spec).items():
        try:
            cols, rows = _csv(d / "flies.csv")
        except Exception as e:   # noqa: BLE001
            r.need(False, f"{c}: flies.csv를 읽지 못했다 ({type(e).__name__}: {e})")
            continue
        missing = [k for k in CAUSE_COLS if k not in cols]
        with_cols += int(r.need(not missing, f"{c}: flies.csv에 원인별 열 {missing}이 없다"))
        if not r.need(CHANCE_COL in cols, f"{c}: flies.csv에 '{CHANCE_COL}' 열이 없다"):
            continue
        sums[c] = sum(int(x[CHANCE_COL]) for x in rows)
        if c == B1:
            r.need(sums[c] == 0, f"{c}: 반전 기회의 합이 {sums[c]}이다 (0이어야 한다)")
        else:
            r.need(sums[c] > 0, f"{c}: 반전 기회의 합이 0이다 (0보다 커야 한다)")
    r.counts.update({"원인별 네 열이 있는 폴더 수": with_cols, "조건별 반전 기회의 합": sums})
    return r


def check_c5(T: Path, E: Path, spec: Spec) -> Result:
    """페로몬: 체제 3의 학습 재생 모두에 페로몬 장이 있고 최댓값이 0보다 크다. 다른 조건의 학습 재생에는 없다."""
    r = Result()
    want_with = ok_with = want_without = ok_without = 0
    for c, d in train_dirs(T, spec).items():
        for s in spec.train_seeds:
            for g in spec.train_keep:
                p = replay_path(d, c, s, g)
                with_phero = c == PHERO_CONDITION
                want_with += int(with_phero)
                want_without += int(not with_phero)
                try:
                    _, arrays = read_replay(p)
                except Exception as e:   # noqa: BLE001
                    r.need(False, f"{p.name}: 읽지 못했다 ({type(e).__name__}: {e})")
                    continue
                has = "phero_field" in arrays
                if with_phero:
                    if r.need(has, f"{p.name}: 페로몬 장이 없다"):
                        ok_with += int(r.need(bool(np.max(arrays["phero_field"], initial=0) > 0),
                                              f"{p.name}: 페로몬 장의 최댓값이 0보다 크지 않다"))
                else:
                    ok_without += int(r.need(not has, f"{p.name}: 페로몬 장이 있다 ({c}에는 없어야 한다)"))
    r.counts.update({"체제 3: 기대한 재생 파일 수": want_with, "체제 3: 페로몬 장이 있고 최댓값 > 0인 파일 수": ok_with,
                     "다른 조건: 기대한 재생 파일 수": want_without, "다른 조건: 페로몬 장이 없는 파일 수": ok_without})
    return r


def check_c6(T: Path, E: Path, spec: Spec) -> Result:
    """가중치: 학습 조건은 시드마다 마지막 판 파일이 있고 파일 안 판 번호가 같다. B0 · B1은 가중치 파일이 없다."""
    r = Result()
    dirs = train_dirs(T, spec)
    want = ok = 0
    for c in spec.learning:
        for s in spec.train_seeds:
            want += 1
            p = dirs[c] / "weights" / weight_filename(s, spec.train_n_games)
            if not r.need(p.exists(), f"{c}: 가중치 파일이 없다: {p.name}"):
                continue
            try:
                inner = final_eval.read_weight_meta(p)["game_no"]
            except Exception as e:   # noqa: BLE001
                r.need(False, f"{c} {p.name}: 읽지 못했다 ({type(e).__name__}: {e})")
                continue
            ok += int(r.need(inner == spec.train_n_games,
                             f"{c} {p.name}: 파일 안 판 번호가 {inner}이다 ({spec.train_n_games}이어야 한다)"))
    stray = {}
    for c in spec.no_learning:
        stray[c] = len(list((dirs[c] / "weights").glob("*.npz"))) if (dirs[c] / "weights").exists() else 0
        r.need(dirs[c].is_dir(), f"{c}: 학습 결과 폴더가 없다: {dirs[c].name}")
        r.need(stray[c] == 0, f"{c}: 가중치 파일이 {stray[c]}개 있다 (없어야 한다)")
    r.counts.update({"기대한 가중치 파일 수": want, "판 번호가 맞는 파일 수": ok, "학습 없는 조건의 가중치 파일 수": stray})
    return r


def check_c7(T: Path, E: Path, spec: Spec) -> Result:
    """평가 경로: report.md · results.json이 있고 보고의 입력 검사가 오류 없이 돈다. 매니페스트의 칸 · 판 번호 · 짝."""
    r = Result()
    E = Path(E)
    for name in ("report.md", "results.json"):
        r.need((E / name).exists(), f"평가 묶음에 {name}이 없다")
    r.counts["report.md · results.json 중 있는 수"] = sum((E / n).exists() for n in ("report.md", "results.json"))
    try:
        final_eval.build_results(E)   # 반환값은 버린다 (성능 값이다). "오류 없이 돌았다"만 쓴다
        r.counts["입력 검사 (build_results)"] = "오류 없음"
    except Exception as e:   # noqa: BLE001
        r.counts["입력 검사 (build_results)"] = "오류"
        r.need(False, f"보고의 입력 검사가 멈췄다 ({type(e).__name__}: {e})")
    dirs = eval_dirs(E, spec, r)
    pairs = list(zip(spec.eval_seeds, spec.pair_train_seeds))
    slots, good_pairs, good_games, empty = {}, 0, 0, 0
    for c in spec.conditions:
        if c not in dirs:   # 평가 index.json에 없는 조건 (이유는 eval_dirs가 남겼다)
            continue
        try:
            files = _json(dirs[c] / "eval_manifest.json")["files"]
        except Exception as e:   # noqa: BLE001
            r.need(False, f"{c}: eval_manifest.json을 읽지 못했다 ({type(e).__name__}: {e})")
            continue
        if c not in spec.learning:
            empty += int(r.need(files == [], f"{c}: 매니페스트에 파일이 {len(files)}개 있다 (비어 있어야 한다)"))
            continue
        slots[c] = len(files)
        r.need(len(files) == len(pairs), f"{c}: 매니페스트가 칸 {len(files)}개다 ({len(pairs)}개여야 한다)")
        by_slot = {int(f["slot"]): f for f in files}
        r.need(sorted(by_slot) == list(range(len(pairs))) and len(by_slot) == len(files),
               f"{c}: 매니페스트의 칸 번호가 0 ~ {len(pairs) - 1}이 한 번씩이 아니다")
        for f in files:
            good_games += int(r.need(int(f["game_no"]) == spec.manifest_game_no,
                                     f"{c} 칸 {f['slot']}: 판 번호가 {f['game_no']}이다 ({spec.manifest_game_no}이어야 한다)"))
        for i, (e_seed, s_seed) in enumerate(pairs):
            f = by_slot.get(i)
            if f is None:
                continue
            got = (int(f["eval_seed"]), int(f["train_seed"]))
            good_pairs += int(r.need(got == (e_seed, s_seed),
                                     f"{c} 칸 {i}: 짝이 {got}이다 ({(e_seed, s_seed)}이어야 한다)"))
    r.counts.update({"학습 조건별 매니페스트 칸 수": slots, "판 번호가 맞는 칸 수": good_games, "짝이 맞는 칸 수": good_pairs,
                     "매니페스트가 빈 조건 수 (학습 없는 조건)": empty})
    return r


def check_c8(T: Path, E: Path, spec: Spec, circuit: Path = CIRCUIT) -> Result:
    """해시: 열 폴더의 체크포인트 sha256이 그 폴더 config.yaml의 값과 같다. 회로 sha256은 뇌 조건에서 회로 파일의
    값이고 B1에서는 비어 있다(null)."""
    r = Result()
    circuit_sha = sha256_file(circuit)
    ckpt_ok = circuit_ok = circuit_null = 0
    expected = set()
    for kind, c, d in all_dirs(T, E, spec, r):
        key = f"{kind} {c}"
        try:
            env = _json(d / "environment.json")
            want = (_yaml(d / "config.yaml").get("pacman") or {}).get("checkpoint_sha256")
        except Exception as e:   # noqa: BLE001
            r.need(False, f"{key}: environment.json 또는 config.yaml을 읽지 못했다 ({type(e).__name__}: {e})")
            continue
        got = env.get("pacman_checkpoint_sha256")
        if r.need(isinstance(want, str) and bool(want), f"{key}: config.yaml에 pacman.checkpoint_sha256이 없다"):
            expected.add(want)
            ckpt_ok += int(r.need(got == want, f"{key}: 체크포인트 sha256이 {got}이다 (설정은 {want})"))
        got = env.get("circuit_sha256")
        if c == B1:
            circuit_null += int(r.need(got is None, f"{key}: 회로 sha256이 {got}이다 (B1은 비어 있어야 한다)"))
        else:
            circuit_ok += int(r.need(got == circuit_sha, f"{key}: 회로 sha256이 {got}이다 (회로 파일은 {circuit_sha})"))
    r.need(len(expected) <= 1, f"폴더들의 config.yaml에 적힌 체크포인트 sha256이 하나가 아니다: {sorted(expected)}")
    r.counts.update({"체크포인트 sha256이 맞는 폴더 수": ckpt_ok, "회로 sha256이 맞는 폴더 수 (뇌 조건)": circuit_ok,
                     "회로 sha256이 빈 폴더 수 (B1)": circuit_null})
    return r


def check_c9(T: Path, E: Path, spec: Spec) -> Result:
    """커밋: 열 폴더 모두 git_commit.txt에 -dirty가 없고 같은 커밋이다."""
    r = Result()
    commits, dirty, read = set(), 0, 0
    for kind, c, d in all_dirs(T, E, spec, r):
        try:
            text = (d / "git_commit.txt").read_text(encoding="utf-8").strip()
        except Exception as e:   # noqa: BLE001
            r.need(False, f"{kind} {c}: git_commit.txt를 읽지 못했다 ({type(e).__name__}: {e})")
            continue
        read += 1
        commits.add(text)
        if not r.need("-dirty" not in text, f"{kind} {c}: git_commit.txt가 {text}이다"):
            dirty += 1
    r.need(len(commits) == 1, f"커밋이 하나가 아니다: {sorted(commits)}")
    r.counts.update({"읽은 폴더 수": read, "-dirty인 폴더 수": dirty, "서로 다른 커밋 수": len(commits),
                     "커밋": sorted(commits)})
    return r


CHECKS = {"C1": check_c1, "C2": check_c2, "C3": check_c3, "C4": check_c4, "C5": check_c5, "C6": check_c6,
          "C7": check_c7, "C8": check_c8, "C9": check_c9}


# ---------------------------------------------------------------- 모으기 · 출력

def run_checks(T: Path, E: Path, spec: Spec, circuit: Path = CIRCUIT) -> dict:
    """C1 ~ C9를 모두 돈다. 한 항목이 예외로 끝나면 그 항목만 실패로 하고 예외 글을 근거로 남긴다."""
    items = []
    for cid, name in ITEMS:
        try:
            res = CHECKS[cid](T, E, spec, circuit) if cid == "C8" else CHECKS[cid](T, E, spec)
            counts, reasons = res.counts, res.reasons
        except Exception as e:   # noqa: BLE001
            counts, reasons = {}, [f"확인이 예외로 끝났다 ({type(e).__name__}: {e})"]
        items.append({"id": cid, "name": name, "passed": not reasons, "counts": counts, "reasons": reasons})
    return {"train_bundle": Path(T).name, "eval_bundle": Path(E).name, "passed": all(i["passed"] for i in items),
            "items": items}


def render(result: dict) -> str:
    L = [f"M4 확인 (DECISIONS 34.3): 학습 묶음 {result['train_bundle']}, 평가 묶음 {result['eval_bundle']}", ""]
    for it in result["items"]:
        L.append(f"[{'통과' if it['passed'] else '실패'}] {it['id']} {it['name']}")
        L += [f"    {k}: {json.dumps(v, ensure_ascii=False)}" for k, v in it["counts"].items()]
        L += [f"    ! {x}" for x in it["reasons"]]
    n = sum(i["passed"] for i in result["items"])
    L += ["", f"{len(result['items'])}개 항목 중 {n}개 통과" + ("" if result["passed"] else " (실패 있음)")]
    return "\n".join(L)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("train_bundle", type=Path, help="학습 묶음 폴더 T")
    ap.add_argument("eval_bundle", type=Path, help="평가 묶음 폴더 E")
    ap.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    ap.add_argument("--eval-config", type=Path, default=EVAL_CONFIG)
    ap.add_argument("--circuit", type=Path, default=CIRCUIT)
    a = ap.parse_args(argv[1:])
    if not a.train_bundle.is_dir():
        print(f"멈춤: 학습 묶음 폴더가 없다: {a.train_bundle}", file=sys.stderr)
        return 2
    spec = make_spec(_yaml(a.train_config), _yaml(a.eval_config))
    result = run_checks(a.train_bundle, a.eval_bundle, spec, a.circuit)
    (a.train_bundle / OUT_NAME).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(render(result))
    print(f"기록: {a.train_bundle / OUT_NAME}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
