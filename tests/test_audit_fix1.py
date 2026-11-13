"""전체 점검 처리 시험 (DECISIONS 28): 무한 루프 한도, 체크포인트 대조, 분할 수 대조, 장부 폴백."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from helpers import make_cfg

from flypac.core.config import Config
from flypac.core.runner import build_runner, check_loop_limit

ROOT = Path(__file__).resolve().parents[1]


def cfg(n_games, limit, watchdog=100):
    return Config({"n_games": n_games, "engine": {"watchdog_ticks": watchdog}, "run": {"max_loop_ticks": limit}})


def test_loop_limit_boundary_equal_passes_and_below_fails():
    # DECISIONS 33.3: 한도 ≥ 판 수 × (watchdog_ticks + 1) (28.2의 판 수 × watchdog_ticks를 고침)
    check_loop_limit(cfg(30, 3030))                                    # 같으면 통과
    with pytest.raises(ValueError, match="33.3") as e:
        check_loop_limit(cfg(30, 3029))
    assert all(x in str(e.value) for x in ("3029", "30", "100", "3030"))  # 값들이 메시지에 있다
    with pytest.raises(ValueError, match="33.3"):
        check_loop_limit(cfg(30, 3000))                                # 옛 규칙(판 수 × watchdog)의 경계는 이제 모자란다


def test_loop_limit_not_checked_without_keys():
    check_loop_limit(Config({"engine": {"watchdog_ticks": 100}, "run": {"max_loop_ticks": 1}}))        # n_games 없음
    check_loop_limit(Config({"n_games": 30, "engine": {}, "run": {"max_loop_ticks": 1}}))              # watchdog 없음
    check_loop_limit(Config({"n_games": 30, "run": {"max_loop_ticks": 1}}))                            # engine 없음


def test_run_raises_before_running_when_limit_too_small():
    r = build_runner(make_cfg(n_games=2, run={"max_loop_ticks": 40001}))   # 2 × 20001 = 40002 미만
    with pytest.raises(ValueError, match="33.3"):
        r.run()
    build_runner(make_cfg(n_games=1, seeds=[0], run={"max_loop_ticks": 20001})).run()   # 같으면 돈다


# DECISIONS 33.3의 한도 규칙(판 수 × (watchdog_ticks + 1))에서 빼는 닫힌 실험의 설정. 0절 · D§30.4에 따라 닫힌 실험의
# 설정은 바꾸지 않고, 그 실험의 재현은 그 커밋(결과 폴더의 git_commit.txt)에서 한다. 지금 main으로 돌리면 시작 검사에서
# 멈추는 것이 맞다.
#   m3_curve_v1: D§30으로 닫힘 (중단 판 0으로 끝났다, D§33.3이 명시).
#   eps_select_v1: D§24로 닫힘 (100판 × 20001 = 2,000,100 > 2,000,000). 33.3에는 이름이 없으나 같은 이유다.
CLOSED_BEFORE_33_3 = {"m3_curve_v1.yaml": 6_000_000, "eps_select_v1.yaml": 2_000_000}


def test_every_run_config_in_the_repository_obeys_the_rule():
    checked = []
    for p in sorted((ROOT / "configs").glob("*.yaml")):
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        if "n_games" in d and "run" in d and "engine" in d and "watchdog_ticks" in d["engine"]:
            if p.name in CLOSED_BEFORE_33_3:
                assert d["run"]["max_loop_ticks"] == CLOSED_BEFORE_33_3[p.name]   # 값은 그대로 둔다
                with pytest.raises(ValueError, match="33.3"):
                    check_loop_limit(Config(d))
                continue
            check_loop_limit(Config(d))
            checked.append(p.name)
    assert {"main_v1.yaml", "final_eval_v1.yaml", "scale_v1.yaml"} <= set(checked)


# ---------------------------------------------------------------- 체크포인트 대조 (28.3)

def ppo_cfg(path, sha=None):
    pacman = {"kind": "ppo", "checkpoint": str(path), "device": "cpu", "eps": 0.0}
    if sha is not None:
        pacman["checkpoint_sha256"] = sha
    return make_cfg(pacman=pacman)


def test_checkpoint_hash_mismatch_stops_before_building_pacman(tmp_path, monkeypatch):
    ck = tmp_path / "best.pt"
    ck.write_bytes(b"fake checkpoint")
    built = []

    class Stub:
        def __init__(self, *a, **k):
            built.append(1)
            raise RuntimeError("stub")

    monkeypatch.setattr("flypac.pacman.ppo.PPOPacman", Stub)
    with pytest.raises(ValueError, match="sha256"):
        build_runner(ppo_cfg(ck, "0" * 64))
    assert built == []                                                  # 팩맨을 만들기 전에 멈췄다
    with pytest.raises(RuntimeError, match="stub"):                     # 해시가 맞으면 팩맨 만들기로 넘어간다
        build_runner(ppo_cfg(ck, hashlib.sha256(b"fake checkpoint").hexdigest()))
    assert built == [1]
    with pytest.raises(RuntimeError, match="stub"):                     # 키가 없으면 대조하지 않는다
        build_runner(ppo_cfg(ck))


def test_environment_json_records_checkpoint_path_and_sha(tmp_path):
    from flypac.recording.envinfo import environment_info

    ck = tmp_path / "best.pt"
    ck.write_bytes(b"abc")
    info = environment_info(Config({"pacman": {"kind": "ppo", "checkpoint": str(ck), "device": "cpu"}}))
    assert info["pacman_checkpoint"] == str(ck)
    assert info["pacman_checkpoint_sha256"] == hashlib.sha256(b"abc").hexdigest()
    none = environment_info(Config({"pacman": {"kind": "rule_based"}}))
    assert none["pacman_checkpoint"] is None and none["pacman_checkpoint_sha256"] is None


# ---------------------------------------------------------------- 분할 수 대조 (28.6)

def test_run_parallel_split_must_match_config(tmp_path):
    spec = importlib.util.spec_from_file_location("run_parallel", ROOT / "scripts" / "run_parallel.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    base = make_cfg().to_dict()
    base.update(seeds=[1, 2], n_games=1, parallel={"split": 1}, ghosts={"kind": "random"})
    with pytest.raises(ValueError, match="parallel.split"):
        rp.run_batch(base, ["b0"], 2, False, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()                              # 아무것도 띄우지 않았다
    idx = rp.run_batch(base, ["b0"], 1, False, tmp_path / "ok")
    assert not idx["failed"]
    cfgfile = tmp_path / "c.yaml"
    cfgfile.write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")
    assert rp.main(["run_parallel.py", str(cfgfile), "--conditions", "b0", "--split", "2",
                    "--out", str(tmp_path / "bad2")]) == 2


# ---------------------------------------------------------------- 장부 폴백 (28.6)

def test_ledger_fallback_reads_reward_defaults(tmp_path):
    from flypac.recording import ledger
    from flypac.regimes.base import REWARD_DEFAULTS

    src = (ROOT / "src" / "flypac" / "recording" / "ledger.py").read_text(encoding="utf-8")
    assert "1 / 3" not in src and "REWARD_DEFAULTS" in src
    d = tmp_path / "r"
    d.mkdir()
    (d / "config.yaml").write_text(yaml.safe_dump({}), encoding="utf-8")   # reward 키 없음 → 기본값
    hdr = "조건,시드,판,초파리,합계 R\n"
    (d / "flies.csv").write_text(hdr + "regime1,1,1,0,%r\n" % (-REWARD_DEFAULTS.q * 2), encoding="utf-8-sig")
    (d / "games.csv").write_text("조건,시드,판,활성 후 포획 수,활성 후 먹힘 수\nregime1,1,1,0,2\n", encoding="utf-8-sig")
    n, bad = ledger.check_run(d)
    assert n == 1 and bad == []                                         # (1 − 3p)·0 − q·2 = −2q


def test_gitignore_has_audit_files():
    assert "AUDIT_*.md" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
