"""보상 장부 검사 (recording/ledger.py): 실제 짧은 실행의 CSV와, 손으로 만든 CSV (먹힘 · 비활성 사건 포함)."""

import csv

import pytest
import yaml
from helpers import make_cfg

from flypac.core.runner import build_runner
from flypac.recording.ledger import check_run
from flypac.recording.recorder import FLIES_COLS, GAMES_COLS, Recorder


@pytest.mark.parametrize("condition", ["regime1", "regime2", "regime3", "b0"])
def test_ledger_holds_on_short_run(tmp_path, condition):
    cfg = make_cfg(condition=condition, seeds=[0, 1], n_games=2, reward={"p": 1 / 3, "q": 1.0})
    build_runner(cfg, Recorder(tmp_path / "run", cfg)).run()
    n, bad = check_run(tmp_path / "run")
    assert n == 4 and bad == []


def _write(path, cols, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def _fake_run(tmp_path, condition, n_c, n_e, n_e_all, sum_r, q=1.0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"reward": {"p": 1 / 3, "q": q}}), encoding="utf-8")
    g = {c: 0 for c in GAMES_COLS} | {"조건": condition, "시드": 0, "판": 1, "활성 후 포획 수": n_c,
                                      "먹힘 수": n_e_all, "활성 후 먹힘 수": n_e}
    _write(tmp_path / "games.csv", GAMES_COLS, [[g[c] for c in GAMES_COLS]])
    rows = []
    for k, r in enumerate(sum_r):
        f = {c: 0 for c in FLIES_COLS} | {"조건": condition, "시드": 0, "판": 1, "초파리": k, "합계 R": repr(float(r))}
        rows.append([f[c] for c in FLIES_COLS])
    _write(tmp_path / "flies.csv", FLIES_COLS, rows)
    return tmp_path


def test_ledger_formulas_with_inactive_events(tmp_path):
    # 체제 1: 먹힘 전체 3건 중 활성 2건 → Σ R = −2 (포획 1건은 팀 합 0)
    assert check_run(_fake_run(tmp_path / "a", "regime1", 1, 2, 3, [-1.0, -0.5, -0.5, 0.0])) == (1, [])
    # 체제 2: Σ R = N_c − N_e
    assert check_run(_fake_run(tmp_path / "b", "regime2", 3, 2, 5, [0.25, 0.25, 0.25, 0.25])) == (1, [])
    # B0: 0
    assert check_run(_fake_run(tmp_path / "c", "b0", 2, 2, 2, [0.0] * 4)) == (1, [])


def test_ledger_detects_mismatch(tmp_path):
    n, bad = check_run(_fake_run(tmp_path / "d", "regime2", 3, 2, 5, [0.25, 0.25, 0.25, 0.2500001]))
    assert n == 1 and len(bad) == 1 and "regime2" in bad[0]
    # 전체 먹힘이 아니라 활성 먹힘으로 대조한다: 전체 5건으로 계산한 값은 어긋난다
    assert check_run(_fake_run(tmp_path / "e", "regime1", 0, 2, 5, [-5.0, 0, 0, 0]))[1]
