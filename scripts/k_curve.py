"""본능 세기 k의 곡선 (DECISIONS 38.4 (가)).

  k_curve.py <학습 묶음>     예: runs/<날짜_시각>_kdev_v1 (run_parallel.py가 만든 폴더)

묶음의 조건 폴더마다 가중치 파일(`weights/s<시드>_g<판>.npz`)의 `k_gain`만 읽는다 (`games.csv`는 읽지 않는다).
조건은 폴더의 config.yaml, 시드와 판은 파일 이름에서 얻는다. 시드마다 초파리 4마리의 평균 k̄_s를 먼저 내고,
저장 판마다 시드 평균 · 표준편차를, 마지막 판에는 k̄ − 1의 평균과 t 95% 신뢰구간(단위 = 시드)을 낸다.
가중치 파일이 없는 조건(B0는 저장하지 않는다, 26.1)은 그 사실만 적는다. 결과는 표준 출력과 묶음의
`k_curve.md` · `k_curve.json`에 쓴다.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import yaml

from flypac.recording.weights import parse_weight_filename

K_REF = 1.0       # DECISIONS 38.4 · 38.5: "마지막 판의 k̄ − 1" (k_init = 1)
CI_LEVEL = 0.95   # DECISIONS 38.4 · 38.5: t 95% 신뢰구간


class CurveInputError(ValueError):
    """입력이 맞지 않는다: 멈춘다."""


def read_k_gain(path: Path) -> np.ndarray:
    """가중치 파일에서 k_gain (F,)만 읽는다. 없으면 CurveInputError."""
    with np.load(Path(path), allow_pickle=False) as z:
        if "k_gain" not in z.files:
            raise CurveInputError(f"{path}: k_gain이 없다 (본능 세기를 배운 실행의 가중치 파일이 아니다)")
        return np.asarray(z["k_gain"], dtype=np.float64)


def collect(bundle: Path) -> dict:
    """{조건: {시드: {판: 초파리 평균 k}}}. 가중치 파일이 없는 조건은 빈 사전."""
    out: dict[str, dict[int, dict[int, float]]] = {}
    dirs = sorted(p.parent for p in Path(bundle).glob("*/config.yaml"))
    if not dirs:
        raise CurveInputError(f"{bundle}: 조건 폴더(config.yaml이 있는 폴더)가 없다")
    for d in dirs:
        cond = str(yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))["condition"])
        per_seed = out.setdefault(cond, {})
        for p in sorted((d / "weights").glob("*.npz")):
            parsed = parse_weight_filename(p.name)
            if not parsed:
                continue
            seed, game = parsed
            if game in per_seed.setdefault(seed, {}):
                raise CurveInputError(f"{cond}: 시드 {seed} 판 {game}의 가중치 파일이 두 번 있다")
            per_seed[seed][game] = float(read_k_gain(p).mean())
    return out


def mean_ci(values: list[float], level: float) -> dict:
    """한 표본 평균과 level 신뢰구간 (t 분포, 자유도 n − 1)."""
    from scipy import stats

    n = len(values)
    if n < 2:
        raise CurveInputError("신뢰구간에는 시드가 2개 이상 필요하다")
    mean, sd = statistics.fmean(values), statistics.stdev(values)
    se = sd / n ** 0.5
    tcrit = float(stats.t.ppf(1 - (1 - level) / 2, n - 1))
    lo, hi = mean - tcrit * se, mean + tcrit * se
    return {"n": n, "df": n - 1, "mean": mean, "sd": sd, "se": se, "t": tcrit, "lo": lo, "hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def curve(per_seed: dict[int, dict[int, float]], *, k_ref: float = K_REF, level: float = CI_LEVEL) -> dict:
    """조건 하나의 곡선. 모든 시드가 같은 판들을 저장했어야 한다 (아니면 CurveInputError)."""
    seeds = sorted(per_seed)
    games = sorted(per_seed[seeds[0]])
    for s in seeds:
        if sorted(per_seed[s]) != games:
            raise CurveInputError(f"시드 {s}의 저장 판이 시드 {seeds[0]}와 다르다")
    rows = []
    for g in games:
        vals = [per_seed[s][g] for s in seeds]
        rows.append({"game": g, "mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else None})
    last = games[-1]
    final = [per_seed[s][last] for s in seeds]
    return {"seeds": seeds, "games": games, "rows": rows, "last_game": last, "last_k_by_seed": final,
            "last_minus_ref": mean_ci([v - k_ref for v in final], level)}


def build(bundle: Path) -> dict:
    data = collect(bundle)
    return {"k_ref": K_REF, "ci_level": CI_LEVEL,
            "conditions": {c: (curve(ps) if ps else None) for c, ps in data.items()}}


def render(res: dict) -> str:
    lvl = int(round(res["ci_level"] * 100))
    L = ["# 본능 세기 k의 곡선 (DECISIONS 38.4)", "",
         "- 가중치 파일의 `k_gain`만 읽었다. 시드마다 초파리 4마리의 평균을 먼저 내고, 시드 사이의 평균 · 표준편차를 낸다.",
         f"- 마지막 판의 k̄ − {res['k_ref']:g}: 평균과 {lvl}% 신뢰구간 (t 분포, 단위 = 시드).", ""]
    L += ["## 마지막 판", "", f"| 조건 | 판 | 시드 수 | k̄ − {res['k_ref']:g} 평균 | {lvl}% 신뢰구간 | 0 제외 |", "|---|---|---|---|---|---|"]
    for c, cv in res["conditions"].items():
        if cv is None:
            L.append(f"| {c} | - | - | - | 가중치 파일 없음 (k를 배우지 않는 조건) | - |")
            continue
        ci = cv["last_minus_ref"]
        L.append(f"| {c} | {cv['last_game']} | {ci['n']} | {ci['mean']:+.4f} | [{ci['lo']:+.4f}, {ci['hi']:+.4f}] | "
                 f"{'예' if ci['excludes_zero'] else '아니오'} |")
    for c, cv in res["conditions"].items():
        if cv is None:
            continue
        L += ["", f"## 곡선: {c}", "", "| 판 | k 시드 평균 | 표준편차 |", "|---|---|---|"]
        L += [f"| {r['game']} | {r['mean']:.4f} | {'-' if r['sd'] is None else format(r['sd'], '.4f')} |" for r in cv["rows"]]
        L += ["", f"마지막 판의 시드별 k̄ (시드 {cv['seeds']}): " + ", ".join(f"{v:.4f}" for v in cv["last_k_by_seed"])]
    return "\n".join(L) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("사용법: k_curve.py <학습 묶음>", file=sys.stderr)
        return 2
    bundle = Path(argv[1])
    try:
        res = build(bundle)
    except CurveInputError as e:
        print(f"멈춤: {e}", file=sys.stderr)
        return 1
    (bundle / "k_curve.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = render(res)
    (bundle / "k_curve.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
