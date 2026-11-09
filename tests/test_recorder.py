import csv
import re

import numpy as np
import pytest
from helpers import Scripted, cell, make_cfg, place

from flypac.core import directions as d
from flypac.core.runner import build_runner
from flypac.recording.recorder import (FLIES_COLS, GAMES_COLS, LIVES_COLS, Recorder, make_run_dir)
from flypac.replay.reader import read_replay


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def run(tmp_path, **over):
    over.setdefault("replay", {"keep_games": [1]})
    cfg = make_cfg(seeds=[0, 1], n_games=2, **over)
    rec = Recorder(tmp_path / "run", cfg)
    out = build_runner(cfg, rec).run()
    return cfg, rec, out


def test_run_folder_files_and_csv_encoding(tmp_path):
    cfg, rec, out = run(tmp_path)
    root = tmp_path / "run"
    for name in ("config.yaml", "git_commit.txt", "games.csv", "lives.csv", "flies.csv"):
        assert (root / name).is_file(), name
    assert (root / "git_commit.txt").read_text(encoding="utf-8").strip()
    for name in ("games.csv", "lives.csv", "flies.csv"):
        assert (root / name).read_bytes().startswith(b"\xef\xbb\xbf")  # utf-8-sig
    assert list(read_csv(root / "games.csv")[0]) == GAMES_COLS
    assert list(read_csv(root / "lives.csv")[0]) == LIVES_COLS
    assert list(read_csv(root / "flies.csv")[0]) == FLIES_COLS


def test_csv_rows_match_results(tmp_path):
    cfg, rec, out = run(tmp_path)
    games = read_csv(tmp_path / "run" / "games.csv")
    assert len(games) == len(out["games"]) == 4
    assert sorted((int(g["시드"]), int(g["판"])) for g in games) == [(0, 1), (0, 2), (1, 1), (1, 2)]
    flies = read_csv(tmp_path / "run" / "flies.csv")
    assert len(flies) == 4 * 4  # 판마다 초파리 4줄
    lives = read_csv(tmp_path / "run" / "lives.csv")
    # 판마다 마지막 줄은 판을 끝낸 목숨: 목숨 번호는 1부터
    assert all(int(r["목숨 번호"]) >= 1 for r in lives)
    assert all(g["승자"] in ("FLIES", "PACMAN", "CENSORED") for g in games)
    assert all(int(g["누적 틱"]) >= int(g["틱 수"]) for g in games)


def test_replay_files_only_for_kept_games_and_valid(tmp_path):
    cfg, rec, out = run(tmp_path)
    files = sorted((tmp_path / "run" / "replay").glob("*.npz"))
    assert [f.name for f in files] == ["regime1_s0_g1.npz", "regime1_s1_g1.npz"]  # 판 1만
    meta, arr = read_replay(files[0])
    assert meta["condition"] == "regime1" and meta["seed"] == 0 and meta["game_no"] == 1
    T = len(arr["pac_cell"])
    games = read_csv(tmp_path / "run" / "games.csv")
    g = next(g for g in games if g["시드"] == "0" and g["판"] == "1")
    assert T == int(g["누적 틱"])  # 첫 판이라 누적 틱 = 프레임 수
    assert "phero_field" not in arr and "kc_active" not in arr
    assert arr["odor_field"].shape == (T, 320) and (arr["action"] >= -1).all()
    assert len(arr["ev_kind"]) > 0


def test_regime3_logs_phero_and_brain_fields_only_from_brain_provider(tmp_path):
    cfg = make_cfg(condition="regime3", n_games=1, replay={"keep_games": [1]})
    Recorder(tmp_path / "a", cfg)
    build_runner(cfg, Recorder(tmp_path / "b", cfg)).run()
    _, arr = read_replay(next((tmp_path / "b" / "replay").glob("*.npz")))
    assert "phero_field" in arr
    # replay.brain이 켜져도 뇌가 아닌 조종기는 뇌 필드를 주지 않는다: 오류 없이 뇌 필드 묶음이 빠진다 (형식 버전 4)
    cfg = make_cfg(n_games=1, replay={"keep_games": [1], "brain": True})
    build_runner(cfg, Recorder(tmp_path / "c", cfg)).run()
    _, arr = read_replay(next((tmp_path / "c" / "replay").glob("*.npz")))
    assert "kc_active" not in arr and not arr["brain_on"].any()


def test_life_row_for_pre_departure_catch_and_recorded_before_reset(tmp_path):
    cfg = make_cfg(n_games=2, replay={"keep_games": []})
    rec = Recorder(tmp_path / "r", cfg)
    r = build_runner(cfg, rec)
    r.pacman = Scripted(d.EAST)
    r._reset_all()
    place(r.game.games[0], pac=cell(11, 12))
    r.tick()
    rec.games_csv.flush(), rec.lives_csv.flush()
    row = read_csv(tmp_path / "r" / "lives.csv")[0]
    assert row["종료 원인"] == "포획" and row["출발 전 포획 여부"] == "1" and row["잡은 초파리"] == "0"
    assert row["포획까지 틱"] == "1"
    assert r.game.views()[0].lives == 2  # 목숨 초기화 뒤에도 목숨 수는 유지


def test_last_life_game_row_records_pre_reset_state(tmp_path):
    cfg = make_cfg(n_games=2, replay={"keep_games": []})
    rec = Recorder(tmp_path / "r", cfg)
    r = build_runner(cfg, rec)
    r.pacman = Scripted(d.EAST)
    r._reset_all()
    g = r.game.games[0]
    g.lives = 1
    place(g, pac=cell(11, 12), released=True, active=True)
    r.tick()
    rec.games_csv.flush(), rec.flies_csv.flush()
    row = read_csv(tmp_path / "r" / "games.csv")[0]
    assert row["승자"] == "FLIES" and row["포획 수"] == "1" and row["활성 후 포획 수"] == "1"
    assert int(row["남은 점"]) > 200  # 초기화 전 상태 (초기화 뒤에도 같지만 목숨 0 판이 한 줄 기록됨)
    flies = read_csv(tmp_path / "r" / "flies.csv")
    # 체제1: 잡은 0번 +1, 나머지 -p. flies.csv는 초파리 번호 순
    assert [f["합계 R"] for f in flies[:4]] == ["1", "-0.25", "-0.25", "-0.25"]
    assert [f["포획"] for f in flies[:4]] == ["1", "0", "0", "0"]
    assert flies[0]["보상 사건 수"] == "1" and flies[0]["결정에서 보상까지 지연(결정 수)"] == "0"


def test_replay_size_cap_discards(tmp_path):
    cfg, rec, out = run(tmp_path / "x", replay={"keep_games": [1], "max_bytes": 1})
    assert not list((tmp_path / "x" / "run" / "replay").glob("*.npz"))


def test_make_run_dir_name(tmp_path):
    cfg = make_cfg()
    p = make_run_dir(cfg, tmp_path)
    assert re.fullmatch(r"\d{8}_\d{6}_test", p.name) and p.parent == tmp_path
    p.mkdir()
    assert make_run_dir(cfg, tmp_path) != p


@pytest.mark.parametrize("ghosts", [{"kind": "random"}, {"kind": "flybrain", "config": "configs/brain_v1.yaml"}])
def test_replay_records_committed_and_moved(tmp_path, ghosts):
    """재생 기록 형식 3 이후: 실행기 7번의 committed · moved (T,4)를 남긴다 (M1 뇌 ② 3.2)."""
    cfg, rec, out = run(tmp_path, ghosts=ghosts)
    meta, arr = read_replay(sorted((tmp_path / "run" / "replay").glob("*.npz"))[0])
    assert meta["format_version"] == 4
    T = len(arr["pac_cell"])
    assert arr["committed"].shape == arr["moved"].shape == (T, 4)
    assert (arr["action"][arr["committed"]] >= 0).all()
    assert arr["committed"].any() and (arr["moved"] & ~arr["committed"]).any()
    # 위치가 바뀐 프레임 쌍은 moved로 설명된다 (프레임은 이동 전 상태, 다음 프레임이 이동 뒤)
    # 목숨이 바뀌는 둘레의 프레임 쌍은 초기화된 위치라 뺀다. 목숨 수로 판정한다: 팩맨 단계 포획이면 그 틱
    # 프레임에 이미 줄어 있고, 초파리 단계 포획이면 다음 프레임에서 줄어 있다 (ev_tick은 쓰지 않는다)
    changed = arr["fly_cell"][1:] != arr["fly_cell"][:-1]
    lives = arr["lives"].astype(int)
    step = np.diff(lives) != 0
    near_end = step | np.r_[False, step[:-1]]
    assert not (changed & ~near_end[:, None] & ~arr["moved"][:-1]).any()
