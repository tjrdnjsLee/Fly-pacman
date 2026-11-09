"""진단 v1 표 만들기 (숫자만). brain_repr_diag.py report 가 부른다."""

from __future__ import annotations

import json

import numpy as np

import brain_repr_diag as D
from flypac.brain import representation as R

NAMES = ["좌", "앞", "우", "뒤"]


def load():
    rows = [json.loads(ln) for ln in (D.OUT / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    by = {}
    for t in rows:
        for c in t["combos"]:
            by.setdefault((c["rho"], c["c"]), {})[c["seed"]] = c
    return by


def ms(vals, nd=3):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return "—"
    return f"{np.mean(v):.{nd}f} [{min(v):.{nd}f}–{max(v):.{nd}f}]"


def grids(by):
    return sorted(by, key=lambda k: (R.RHOS.index(k[0]), k[1]))


def table(by, getter, title, note=""):
    out = [f"### {title}", note, "", "| ρ | c | " + " | ".join(h for h, _ in getter) + " |",
           "|---|---|" + "---|" * len(getter)]
    for g in grids(by):
        cs = by[g]
        out.append(f"| {g[0]} | {g[1]} | " + " | ".join(ms([f(cs[s]) for s in sorted(cs)]) for _, f in getter) + " |")
    return out


def get(path):
    def f(c):
        x = c
        for k in path:
            x = x[k]
        return x
    return f


def main() -> int:
    by = load()
    n_seed = {len(v) for v in by.values()}
    legal = json.loads((D.OUT / "legal_count.json").read_text(encoding="utf-8"))
    L = ["# 표현력 진단 v1 (판정에 쓰지 않음, 규칙 변경 아님)", "",
         f"- 배정 시드 0~4, 격자 {len(by)}개 (ρ × c), 격자마다 시드 수 {sorted(n_seed)}. 표의 값은 시드 평균 [최소–최대].",
         "- 분류기 설정, 학습/평가 분할, 특징 생성은 기존 측정(DECISIONS 13절)과 같다. 평가 표본만 나눠서 셌다.",
         "- 균형 정확도 = 방향별 재현율의 평균 (기존 시험 4 판정과 같은 지표). 일반 = 맞힌 비율.", ""]
    L += ["## (a) 합법 방향 수", "", "결정 표본·시험별 표본의 합법 방향 수 분포 (개수):", "", "```json",
          json.dumps(legal, ensure_ascii=False, indent=1), "```", ""]
    for z in (0, 1):
        L += table(by, [("전체 균형", get(["t4", f"z{z}", "all", "balanced"])),
                        ("|L|=2 균형", get(["t4", f"z{z}", "L2", "balanced"])),
                        ("|L|=3 균형", get(["t4", f"z{z}", "L3", "balanced"])),
                        ("|L|=2 일반", get(["t4", f"z{z}", "L2", "general"])),
                        ("|L|=3 일반", get(["t4", f"z{z}", "L3", "general"]))],
                   f"시험 4, z = {z}, 선택지 수별 정확도 (평가 표본)")
        c0 = next(iter(by[grids(by)[0]].values()))
        L += ["", f"평가 표본 수 (z={z}): 전체 {c0['t4'][f'z{z}']['all']['n']}, |L|=2 {c0['t4'][f'z{z}']['L2']['n']}, "
              f"|L|=3 {c0['t4'][f'z{z}']['L3']['n']}", ""]
    L += ["## (b) 시험 4, z = 1: 방향별 재현율과 혼동 행렬", ""]
    L += table(by, [(f"재현율 {NAMES[d]}", get(["t4", "z1", "all", "per_dir", str(d)])) for d in R.DIRS],
               "방향별 재현율 (정답 방향 기준)")
    L += ["", "혼동 행렬: 행 = 정답, 열 = 예측 (좌·앞·우·뒤), 시드 5개를 합친 개수 → 행 비율.", ""]
    for g in grids(by):
        m = sum(np.array(by[g][s]["t4"]["z1"]["conf"]) for s in by[g])
        L += [f"**ρ {g[0]}, c {g[1]}**", "", "| 정답＼예측 | 좌 | 앞 | 우 | 뒤 | 정답 수 |", "|---|---|---|---|---|---|"]
        for d in R.DIRS:
            tot = m[d].sum()
            L.append(f"| {NAMES[d]} | " + " | ".join(f"{m[d][j] / tot:.3f}" if tot else "—" for j in range(4)) + f" | {int(tot)} |")
        L.append("")
    L += ["## (c) 시험 1, 구배별 정확도", "",
          "r = (o₁ − o₂)/o₁ (합법 방향 중 최댓값과 두 번째). 약함 r < 0.1, 강함 r ≥ 0.3.", ""]
    L += table(by, [("전체 균형", get(["t1", "all", "balanced"])), ("약함 균형", get(["t1", "weak", "balanced"])),
                    ("강함 균형", get(["t1", "strong", "balanced"])), ("약함 일반", get(["t1", "weak", "general"])),
                    ("강함 일반", get(["t1", "strong", "general"]))], "시험 1 (평가 표본)")
    c0 = next(iter(by[grids(by)[0]].values()))
    L += ["", f"표본 수: 전체 {c0['t1']['all']['n']}, 약함 {c0['t1']['weak']['n']}, 중간 {c0['t1']['mid']['n']}, 강함 {c0['t1']['strong']['n']}", ""]
    L += ["## (d) 참고: 시험 4 z = 1 집합 정답 (정답 = 합법 방향 중 가장 센 방향이 아닌 것)", "",
          "기존 판정과 별개의 진단. 우연 수준 = 합법 방향 중 임의로 하나를 고를 때의 정답 확률.", ""]
    L += table(by, [("전체 정확도", get(["t4", "z1_set", "all", "acc"])), ("우연(전체)", get(["t4", "z1_set", "all", "chance"])),
                    ("|L|=2", get(["t4", "z1_set", "L2", "acc"])), ("|L|=3", get(["t4", "z1_set", "L3", "acc"])),
                    ("우연 |L|=3", get(["t4", "z1_set", "L3", "chance"]))], "집합 정답 정확도")
    L += ["", "## 기존 측정과의 일치 확인", ""]
    old = [json.loads(ln) for ln in (D.SRC / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    oldmap = {(r["seed"], r["rho"], c["c"]): c for r in old for c in r["combos"]}
    d1 = d4 = n = 0.0
    for (rho, c), cs in by.items():
        for s, v in cs.items():
            o = oldmap[(s, rho, c)]
            d1 = max(d1, abs(o["t1"]["acc"] - v["t1"]["all"]["balanced"]))
            d4 = max(d4, abs(o["t4"]["acc_z0"] - v["t4"]["z0"]["all"]["balanced"]),
                     abs(o["t4"]["acc_z1"] - v["t4"]["z1"]["all"]["balanced"]))
            n += 1
    L.append(f"조합 {int(n)}개에서 기존 results의 시험 1 균형 정확도와의 최대 차이 {d1:.2e}, 시험 4 z=0·z=1과의 최대 차이 {d4:.2e}.")
    (D.OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (D.OUT / "summary.json").write_text(json.dumps({f"rho{g[0]}_c{g[1]}": by[g] for g in grids(by)}, ensure_ascii=False),
                                        encoding="utf-8")
    print("\n".join(L))
    return 0
