"""41절 탐색 스크립트의 고정과 확인 실행 스크립트의 시험 (DECISIONS 41.12 ~ 41.15). torch · 체크포인트는 쓰지 않는다.

1 바이트 고정 (sha256), 2 확인 시드, 3 세 팔이 끝까지 돈다, 4 대조군 구현 (ξ = +1이면 learn과 같다), 5 CTRL_SIGN은 시험 전용.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPLORE = ROOT / "scripts" / "explore"
SHA256 = {
    "phero_land.py": "9d63caa0fcb945e21ba87330e62eeb809f14490028e3318171194fdb3b512cfe",
    "phero_dev.py": "6bdd3982408f8801b96b0ddc2ef3baeef5097dc86693583e7f256681d41c4477",
    "phero_ctrl.py": "3b3e14de3ae349e62792a594cdf6f5f179ddec52cac0d43e61a4dd7d3ea4a3a0",
    "phero_confirm.py": "6570d5818b65d0ee3066768d96d399a891df1870866847a7af5d799e400b98f2",
}


def _script(root: Path, **env):
    return subprocess.run([sys.executable, str(EXPLORE / "phero_confirm.py")], cwd=ROOT,
                          env={**os.environ, "COOP_TEST": "1", "COOP_ROOT": str(root), **env},
                          capture_output=True, text=True, encoding="utf-8")


def _results(root: Path, arm: str) -> dict:
    return json.loads((root / f"phero_confirm_v1_{arm}" / "results.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 1. 바이트 고정

@pytest.mark.parametrize("name", sorted(SHA256))
def test_script_bytes_are_the_ones_that_ran(name):
    """실행했거나 실행할 그대로여야 한다 (41.12 ~ 41.15). 고치면 이 시험이 실패한다."""
    assert hashlib.sha256((EXPLORE / name).read_bytes()).hexdigest() == SHA256[name]


# ---------------------------------------------------------------- 2. 시드

def test_confirm_seeds_are_used_nowhere_else():
    """7700 ~ 7729 · 9920 ~ 9949는 phero_confirm.py에만 있다 (docs/ 제외)."""
    pat = re.compile(r"(?<!\d)(?:77[0-2]\d|99[2-4]\d)(?!\d)")
    seeds = set(range(7700, 7730)) | set(range(9920, 9950))
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        used = list(d.get("seeds") or []) + list((d.get("final_eval") or {}).get("train_seeds") or [])
        assert not {s for s in used if isinstance(s, int)} & seeds, path.name
        assert not pat.search(path.read_text(encoding="utf-8")), path.name
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        if path.name != "phero_confirm.py":
            assert not pat.search(path.read_text(encoding="utf-8")), path
    text = (EXPLORE / "phero_confirm.py").read_text(encoding="utf-8")
    assert "range(7700, 7730)" in text and "range(9920, 9950)" in text


# ---------------------------------------------------------------- 3 · 4. 스크립트

@pytest.fixture(scope="module")
def confirm_runs(tmp_path_factory):
    """한 뿌리에 learn · ctrl · report, 다른 뿌리에 CTRL_SIGN=plus의 ctrl."""
    main, plus = tmp_path_factory.mktemp("confirm"), tmp_path_factory.mktemp("confirm_plus")
    done = {arm: _script(main, PH_ARM=arm) for arm in ("learn", "ctrl", "report")}
    done["ctrl_plus"] = _script(plus, PH_ARM="ctrl", CTRL_SIGN="plus")
    return main, plus, done


def test_learn_ctrl_report_run_to_the_end(confirm_runs):
    main, _, done = confirm_runs
    for arm in ("learn", "ctrl", "report"):
        assert done[arm].returncode == 0, (arm, done[arm].stderr)
    assert (main / "phero_confirm_v1" / "report.json").exists()
    rep = json.loads((main / "phero_confirm_v1" / "report.json").read_text(encoding="utf-8"))
    assert set(rep["comparisons"]) == {"learn − ctrl", "learn − solo", "ctrl − solo"} and isinstance(rep["verdict"], bool)
    learn, ctrl = _results(main, "learn"), _results(main, "ctrl")
    assert learn["train_seeds"] == ctrl["train_seeds"] == [0, 1] and learn["eval_seeds"] == ctrl["eval_seeds"] == [2, 3]
    assert set(learn["eval"]) == {"learned", "solo"} and set(ctrl["eval"]) == {"learned"}
    assert ctrl["xi"]["sign"] == "random" and ctrl["xi"]["plus"] + ctrl["xi"]["minus"] > 0
    assert "판정 (41.15)" in done["report"].stdout


def test_ctrl_with_plus_sign_equals_learn(confirm_runs):
    main, plus, done = confirm_runs
    assert done["ctrl_plus"].returncode == 0, done["ctrl_plus"].stderr
    learn, ctrl = _results(main, "learn"), _results(plus, "ctrl")
    assert ctrl["xi"]["sign"] == "plus" and ctrl["xi"]["minus"] == 0 and ctrl["xi"]["plus"] > 0
    assert ctrl["kph_final"] == learn["kph_final"]
    assert any(v != 0.0 for row in learn["kph_final"] for v in row)          # k_ph가 실제로 움직였다
    assert ctrl["eval"]["learned"]["W"] == learn["eval"]["learned"]["W"]


# ---------------------------------------------------------------- 5. 시험 전용 선택

def test_ctrl_sign_outside_test_mode_stops(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "COOP_TEST"}
    done = subprocess.run([sys.executable, str(EXPLORE / "phero_confirm.py")], cwd=ROOT,
                          env={**env, "COOP_ROOT": str(tmp_path), "PH_ARM": "ctrl", "CTRL_SIGN": "plus"},
                          capture_output=True, text=True, encoding="utf-8")
    assert done.returncode != 0 and "CTRL_SIGN" in done.stderr
    assert not any(tmp_path.iterdir())
