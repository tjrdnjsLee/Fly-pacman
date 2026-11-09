"""작은 과제 A · B · C 실행 (DECISIONS 17.4, 18.7의 값). 과제 D는 따로 돌린다.

  uv run python scripts/run_small_tasks.py [A B C]     # 학습 시드 6000 ~ 6004. C가 통과하면 확인용 4000 ~ 4004로 C를 한 번 더.

회로는 configs/brain_v1.yaml (배정 시드 0, ρ = 0.05, θ = 0, 18.7의 하이퍼파라미터). 설정을 바꿔 다시 시도하지 않는다.
각 과제: 학습(η = 설정값) → 가중치 고정 평가 200회, 같은 시드 · 같은 초기 회로의 η = 0 대조. η = 0도 통과하면 보상 방향을
뒤집어 새 방향을 배우는지 본다. A는 좌 · 우 보상 버전 모두, B는 g = 0 평가를 진단으로 더한다. 결과는 runs/small_tasks/.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from flypac.brain.flybrain import FlyBrain, kc_drive, kc_select, load_brain_config, pn_input
from flypac.core.config import load_config
from flypac.tasks import runner
from flypac.tasks.tmaze import TMaze, combos

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "small_tasks"
BRAIN = "configs/brain_v1.yaml"
TASKS = load_config(REPO / "configs" / "tasks_v1.yaml")
SEEDS = [6000, 6001, 6002, 6003, 6004]
CONFIRM_SEEDS = [4000, 4001, 4002, 4003, 4004]
BUDGET = {t: int(TASKS.budget[t]) for t in "ABCD"}
EVAL_N = int(TASKS.budget.eval_trials)
# TRD 10.8 합격 기준 (5개 중 4개 시드)
PASS = {"A": 0.80, "B": 0.80, "C": 0.75}
SEEDS_NEEDED = 4


class _Rng:
    def __init__(self):
        self.g = np.random.default_rng(0)

    def __getitem__(self, r):
        return self.g


def task_kc_rows(circuit, task):
    """과제 입력들이 켜는 KC의 합집합 (로컬 ‖Δw‖/‖w‖의 행)."""
    cfg = load_brain_config(BRAIN)
    k = int(math.ceil(float(cfg.kc.rho) * circuit.n_kc - 1e-9))
    a = float(TASKS.env.amp)
    inputs = [([a, 0, a, 0], 0.0)] if task == "A" else [([a, 0, 0, 0], 0.0), ([0, 0, a, 0], 0.0)]
    if task == "C":
        inputs += [([a, 0, 0, 0], 1.0), ([0, 0, a, 0], 1.0)]
    rows = np.zeros(circuit.n_kc, dtype=bool)
    for odor, z in inputs:
        u = kc_drive(pn_input(np.asarray(odor, float)[None], np.array([z]), np.zeros((1, 4))), circuit)
        rows |= kc_select(u, float(cfg.kc.theta), k, _Rng())[0]
    return rows


def weight_ratios(brain, rows):
    w0, dw = brain.circuit.w0, brain.dw[:, 0]
    loc = [float(np.linalg.norm(dw[b][rows]) / np.linalg.norm((w0 + dw[b])[rows])) for b in range(brain.B)]
    glo = [float(np.linalg.norm(dw[b]) / np.linalg.norm(w0 + dw[b])) for b in range(brain.B)]
    return loc, glo


def per_combo(rec, task):
    s = runner.summarize(rec, task)
    return s["rate"], s["p_correct"], s["combos"]


def train(circuit, task, seeds, *, eta0=False, reward_side="L", reverse=False, brain=None, game_no=1, block=None):
    env = TMaze(TASKS, task, seeds, reward_side=reward_side, reverse=reverse, game_no=game_no)
    if brain is None:
        brain = runner.build_brain(BRAIN, seeds, circuit=circuit, **({"eta": 0.0} if eta0 else {}))
        runner.start(brain)
    n = BUDGET[task]
    block = block or n // 10
    curve, curve_combo, ent = [], [], []
    for _ in range(n // block):
        rec = runner.run_trials(env, brain, block)
        curve.append(rec["ok"].mean(0))                      # (B,)
        if task == "C":
            curve_combo.append(per_combo(rec, task)[0])      # (B, 4)
        ent.append(rec["entropy"].mean(0))
    runner.finish(brain)
    return brain, env, {"curve": np.stack(curve), "curve_combo": np.stack(curve_combo) if curve_combo else None,
                        "entropy_last_block": ent[-1], "entropy_blocks": np.stack(ent)}


def evaluate(env, brain, task, g=None, reverse_env=None):
    e = reverse_env if reverse_env is not None else env
    rec = runner.evaluate(e, brain, EVAL_N, g=g)
    rate, pc, cs = per_combo(rec, task)
    return {"acc": rec["ok"].mean(0), "p_correct": rec["p_correct"].mean(0), "combo_rate": rate, "combo_p": pc,
            "combos": cs}


def passes(task, ev):
    """시드별 합격 (B,) bool."""
    if task == "C":
        return (ev["combo_rate"] >= PASS["C"]).all(1)
    return ev["acc"] >= PASS[task]


def run_task(circuit, task, seeds, tag):
    out = {"task": task, "seeds": seeds, "budget": BUDGET[task], "eval_trials": EVAL_N}
    rows = task_kc_rows(circuit, task)
    sides = ["L", "R"] if task == "A" else ["L"]
    versions = {}
    for side in sides:
        v = {}
        brain, env, tr = train(circuit, task, seeds, reward_side=side)
        ev = evaluate(env, brain, task)
        loc, glo = weight_ratios(brain, rows)
        v.update({"train": tr, "eval": ev, "pass_seed": passes(task, ev), "local_ratio": loc, "global_ratio": glo})
        if task == "B":
            v["eval_g0"] = evaluate(env, brain, task, g=0.0)       # 진단 (관문 아님)
        b0, env0, tr0 = train(circuit, task, seeds, eta0=True, reward_side=side)
        ev0 = evaluate(env0, b0, task)
        v.update({"train_eta0": tr0, "eval_eta0": ev0, "pass_seed_eta0": passes(task, ev0)})
        if passes(task, ev0).sum() >= SEEDS_NEEDED:
            # η = 0으로도 통과: 보상 방향을 뒤집어 같은 학습을 이어 새 방향을 배우는지 본다 (η > 0 뇌에서 이어서)
            renv = TMaze(TASKS, task, seeds, reward_side=side, reverse=True, game_no=3)
            _, _, trr = train(circuit, task, seeds, reward_side=side, reverse=True, brain=brain, game_no=3)
            evr = evaluate(None, brain, task, reverse_env=renv)
            v["reversal"] = {"train": trr, "eval": evr, "pass_seed": passes(task, evr)}
        versions[side] = v
    out["versions"] = versions
    if task == "A":
        both = np.all([versions[s]["pass_seed"] for s in sides], axis=0)
    else:
        both = versions["L"]["pass_seed"]
    out["pass_seed"] = both
    out["passed"] = bool(both.sum() >= SEEDS_NEEDED)
    out["n_pass"] = int(both.sum())
    return out


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def main(argv) -> int:
    tasks = [t for t in (argv or ["A", "B", "C"]) if t in "ABC"]
    OUT.mkdir(parents=True, exist_ok=True)
    circuit = FlyBrain(load_brain_config(BRAIN), 1, [0]).circuit
    res = {}
    for t in tasks:
        res[t] = run_task(circuit, t, SEEDS, "main")
        (OUT / f"task_{t}.json").write_text(json.dumps(jsonable(res[t]), ensure_ascii=False), encoding="utf-8")
        print(t, "통과 시드", res[t]["n_pass"], "/5", res[t]["pass_seed"].tolist(), flush=True)
    if "C" in res and res["C"]["passed"]:
        conf = run_task(circuit, "C", CONFIRM_SEEDS, "confirm")
        (OUT / "task_C_confirm.json").write_text(json.dumps(jsonable(conf), ensure_ascii=False), encoding="utf-8")
        print("C 확인(4000~4004) 통과 시드", conf["n_pass"], "/5", conf["pass_seed"].tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
