"""보상 장부 검사: 결과 폴더의 CSV만 읽어 판마다 초파리 보상 합이 사건 수와 맞는지 본다.

  체제 1: Σ R = (1 − 3p) · N_c(활성) − q · N_e(활성)    p = 1/3이면 −q · N_e (포획의 팀 합 0, DECISIONS 20)
  체제 2 · 3: Σ R = N_c(활성) − q · N_e(활성)        B0 · B1: Σ R = 0
p, q는 결과 폴더 config.yaml의 reward 값이다 (기록기가 적용값을 남긴다). 허용 오차는 1e-12 (합계 R을 repr로 써서 반올림이 없다).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import yaml

TOL = 1e-12


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check_run(run_dir) -> tuple[int, list[str]]:
    """(검사한 판 수, 어긋난 판 설명 목록)."""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    reward = cfg.get("reward") or {}
    p, q = float(reward.get("p", 1 / 3)), float(reward.get("q", 1.0))
    sum_r: dict[tuple, float] = defaultdict(float)
    for row in _read(run_dir / "flies.csv"):
        sum_r[(row["조건"], row["시드"], row["판"])] += float(row["합계 R"])
    bad, n = [], 0
    for g in _read(run_dir / "games.csv"):
        key = (g["조건"], g["시드"], g["판"])
        n_c, n_e = int(g["활성 후 포획 수"]), int(g["활성 후 먹힘 수"])
        cond = g["조건"]
        if cond == "regime1":
            expected = (1 - 3 * p) * n_c - q * n_e
        elif cond in ("regime2", "regime3"):
            expected = n_c - q * n_e
        elif cond in ("b0", "b1"):
            expected = 0.0
        else:
            raise ValueError(f"알 수 없는 조건: {cond!r}")
        n += 1
        if abs(sum_r[key] - expected) > TOL:
            bad.append(f"{cond} 시드 {key[1]} 판 {key[2]}: Σ R = {sum_r[key]!r}, 기대 {expected!r} "
                       f"(N_c {n_c}, N_e {n_e}, p {p}, q {q})")
    return n, bad
