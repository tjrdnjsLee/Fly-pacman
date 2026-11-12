"""조건 여러 개를 프로세스 여러 개로 동시에 돌리는 실행기 (DECISIONS 27).

  run_parallel.py <기본 설정.yaml> --conditions regime1,regime2,regime3,b0,b1 --split <k> [--sequential] [--out <묶음 폴더>]

조건마다 시드 목록을 앞에서부터 k등분해 프로세스 k개를 띄운다 (k = 1이면 조건당 하나). 각 프로세스는
`run_experiment.py`와 같은 경로(`build_runner` + `Recorder`)로 돈다. 프로세스마다 수치 라이브러리 스레드는 1개다.
묶음 폴더의 index.json에 (조건, 시드 목록, 결과 폴더, 시작 · 끝 시각, 벽시계 초, 종료 코드)를 적는다.
하나라도 실패하면 나머지를 끝까지 기다린 뒤 실패를 보고하고 0이 아닌 코드로 끝난다.
자식 프로세스의 출력은 버린다 (학습 결과를 화면에 내지 않는다). 오류 출력만 묶음 폴더의 logs/에 남는다.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


# ---------------------------------------------------------------- 구성 (순수)

def split_seeds(seeds: list[int], k: int) -> list[list[int]]:
    """시드 목록을 앞에서부터 k등분한다. 나누어떨어지지 않으면 앞 몫이 하나씩 더 가진다."""
    seeds = [int(s) for s in seeds]
    if k < 1 or k > len(seeds):
        raise ValueError(f"--split은 1 이상 시드 수({len(seeds)}) 이하여야 한다: {k}")
    base, extra = divmod(len(seeds), k)
    out, i = [], 0
    for j in range(k):
        n = base + (1 if j < extra else 0)
        out.append(seeds[i:i + n])
        i += n
    return out


def condition_config(base: dict, condition: str, seeds: list[int], part: int) -> dict:
    """프로세스 하나의 설정. select_eps.run_config과 같이 b1이면 원본 유령, 나머지는 기본 설정의 ghosts(뇌)."""
    d = copy.deepcopy(base)
    d["condition"] = condition
    d["ghosts"] = {"kind": "original"} if condition == "b1" else copy.deepcopy(base["ghosts"])
    d["seeds"] = [int(s) for s in seeds]
    d["experiment"] = f"{base['experiment']}_{condition}_p{part}"
    return d


def plan(base: dict, conditions: list[str], k: int) -> list[dict]:
    """[{name, condition, seeds, config}] 조건 순서, 조각 순서대로."""
    jobs = []
    for cond in conditions:
        for i, part in enumerate(split_seeds(base["seeds"], k), start=1):
            jobs.append({"name": f"{cond}_p{i}", "condition": cond, "seeds": part,
                         "config": condition_config(base, cond, part, i)})
    return jobs


def child_env() -> dict:
    env = dict(os.environ)
    for v in THREAD_VARS:
        env[v] = "1"
    return env


# ---------------------------------------------------------------- 실행

def _run_child(job: dict, out: Path) -> dict:
    cfg_path = out / "cfg" / f"{job['name']}.yaml"
    log_path = out / "logs" / f"{job['name']}.err"
    run_dir = out / job["name"]
    started, t0 = datetime.now().isoformat(timespec="seconds"), time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as err:
        code = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(cfg_path), str(run_dir)],
                              env=child_env(), stdout=subprocess.DEVNULL, stderr=err).returncode
    return {"condition": job["condition"], "seeds": job["seeds"], "run_dir": job["name"],
            "start": started, "end": datetime.now().isoformat(timespec="seconds"),
            "wall_seconds": time.perf_counter() - t0, "exit_code": code}


def run_batch(base: dict, conditions: list[str], k: int, sequential: bool, out: Path) -> dict:
    """프로세스들을 돌리고 out/index.json을 쓴다. 반환: index 사전 (runs, wall_seconds, failed)."""
    out = Path(out)
    jobs = plan(base, conditions, k)
    (out / "cfg").mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    for job in jobs:
        (out / "cfg" / f"{job['name']}.yaml").write_text(
            yaml.safe_dump(job["config"], allow_unicode=True, sort_keys=False), encoding="utf-8")
    t0 = time.perf_counter()
    if sequential:
        runs = [_run_child(j, out) for j in jobs]
    else:
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:  # 전부 한꺼번에 띄운다
            runs = list(ex.map(lambda j: _run_child(j, out), jobs))
    index = {"conditions": list(conditions), "split": k, "sequential": bool(sequential),
             "wall_seconds": time.perf_counter() - t0, "runs": runs,
             "failed": [r["run_dir"] for r in runs if r["exit_code"] != 0]}
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index


def child_main(cfg_path: Path, run_dir: Path) -> int:
    from flypac.core.config import load_config
    from flypac.core.runner import build_runner
    from flypac.recording.recorder import Recorder

    cfg = load_config(cfg_path)
    build_runner(cfg, Recorder(run_dir, cfg)).run()
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[1] == "--child":
        return child_main(Path(argv[2]), Path(argv[3]))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv[1:])
    from flypac.core.config import load_config
    from flypac.recording.recorder import make_run_dir

    cfg = load_config(a.config)
    out = a.out or make_run_dir(cfg, Path("runs"))
    conditions = [c for c in a.conditions.split(",") if c]
    index = run_batch(cfg.to_dict(), conditions, a.split, a.sequential, out)
    print(f"묶음 폴더: {out} ({len(index['runs'])}개 프로세스, {index['wall_seconds']:.1f}초)")
    if index["failed"]:
        print(f"실패: {', '.join(index['failed'])} (오류 출력: {out / 'logs'})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
