"""작은 과제 D (지연) 한 조각 실행 (DECISIONS 17.4, 19절의 설정). 지연값 n과 η 모드마다 프로세스를 나눠 돌린다.

  uv run python scripts/run_small_task_d.py N [eta0]     # 학습 시드 6000 ~ 6004. 결과: runs/small_tasks/D_n{N}[_eta0].json
  uv run python scripts/run_small_task_d.py report       # 조각을 모아 표로 출력

합격: n = 0, 5에서 최초 선택 정답률 70% 이상 (5개 중 4개 시드). n = 10, 30, 50은 성능 저하 곡선으로 기록만 한다.
최초 선택 = 시행의 첫 갈림길 결정 (보상은 n번 더 결정한 뒤에 그 선택의 정답 여부로 온다).
"""

from __future__ import annotations

import json
import sys

import numpy as np

import run_small_tasks as S
from flypac.tasks import runner
from flypac.tasks.tmaze import TMaze

PASS_D = 0.70
NS = [int(n) for n in S.TASKS.delays]


def run_piece(n: int, eta0: bool) -> dict:
    circuit = S.FlyBrain(S.load_brain_config(S.BRAIN), 1, [0]).circuit
    seeds = S.SEEDS
    env = TMaze(S.TASKS, "D", seeds, n_delay=n)
    brain = runner.build_brain(S.BRAIN, seeds, circuit=circuit, **({"eta": 0.0} if eta0 else {}))
    runner.start(brain)
    budget, block = S.BUDGET["D"], S.BUDGET["D"] // 10
    curve, ent = [], []
    for _ in range(budget // block):
        rec = runner.run_trials(env, brain, block)
        curve.append(rec["ok"].mean(0))
        ent.append(rec["entropy"].mean(0))
    runner.finish(brain)
    ev = runner.evaluate(env, brain, S.EVAL_N)
    rows = S.task_kc_rows(circuit, "B")
    loc, glo = S.weight_ratios(brain, rows)
    return {"n": n, "eta0": eta0, "seeds": seeds, "budget": budget, "eval_trials": S.EVAL_N,
            "curve": np.stack(curve), "entropy_blocks": np.stack(ent), "acc": ev["ok"].mean(0),
            "p_correct": ev["p_correct"].mean(0), "pass_seed": ev["ok"].mean(0) >= PASS_D,
            "local_ratio": loc, "global_ratio": glo}


def report() -> None:
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    for eta0 in (False, True):
        print("η = 0 대조" if eta0 else "학습")
        for n in NS:
            f = S.OUT / f"D_n{n}{'_eta0' if eta0 else ''}.json"
            if not f.exists():
                print(f"  n={n}: 없음")
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            acc = np.array(d["acc"])
            print(f"  n={n:<2} 평가 {acc} 합격 {int((acc >= PASS_D).sum())}/5 | 학습 첫→마지막 구간 "
                  f"{np.array(d['curve'])[0].round(2)} → {np.array(d['curve'])[-1].round(2)} | 후기 H/ln2 "
                  f"{(np.array(d['entropy_blocks'])[-1] / np.log(2)).round(3)} | 로컬 {np.round(d['local_ratio'], 3)} "
                  f"글로벌 {np.round(d['global_ratio'], 3)}")


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
    else:
        n, eta0 = int(sys.argv[1]), len(sys.argv) > 2 and sys.argv[2] == "eta0"
        S.OUT.mkdir(parents=True, exist_ok=True)
        r = run_piece(n, eta0)
        (S.OUT / f"D_n{n}{'_eta0' if eta0 else ''}.json").write_text(
            json.dumps(S.jsonable(r), ensure_ascii=False), encoding="utf-8")
        print(n, eta0, np.array(r["acc"]).round(3).tolist(), flush=True)
