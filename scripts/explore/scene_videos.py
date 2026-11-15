"""D§35.5의 남은 두 장면 (D§40.2). 결과를 보기 전에 저장소에 넣고, 병합 뒤 한 번 돌린다.

(가) synapse : 시냅스 학습 전후. 본 실험 최종 평가 runs/final_eval_v1, 평가 시드 9000 판 1, 왼쪽 b0 / 오른쪽 regime1.
      두 재생의 `action` (T x 4) 행에서 어느 초파리든 값이 다른 첫 틱 t* (결정 아닌 틱은 -1).
      끝까지 같으면 9001, 9002, ... 순서. 구간은 t*의 5초 전 ~ 10초 후, 5초 전이 음수면 0부터,
      끝은 짧은 판의 끝을 넘지 않는다 (scripts/explore/pair_videos.py와 같은 구현). 같은 판의 전체 영상도 만든다.
(나) regimes : 세 체제. 본 실험 학습 묶음 runs/20261114_010803_main_v1, 학습 시드 8000,
      체제 1 · 2 · 3마다 왼쪽 판 1 / 오른쪽 판 3000의 처음 30초 (판이 짧으면 그 판의 끝까지). 체제마다 영상 하나.

초 = 틱 / tick_rate (viz/settings.yaml). 재생 파일은 읽기만 한다. 승패 · 점수는 보지 않는다.
결과: runs/videos/synapse_s<시드>_g1_pair.mp4, runs/videos/full/synapse_s<시드>_g1_full_pair.mp4,
      runs/videos/regimes/regime<r>_s8000_g1_vs_g3000.mp4, 기록 runs/videos/scenes_manifest.json.

  저장소 루트에서:
    DRY=1 uv run python - < scripts/explore/scene_videos.py      (t*와 구간만 출력, 영상 · 기록 없음)
    uv run --with pygame==2.6.1 --with imageio-ffmpeg==0.6.0 --with matplotlib python - < scripts/explore/scene_videos.py
"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(os.environ.get("SCENE_ROOT", "runs"))
OUT = ROOT / "videos"
DRY = os.environ.get("DRY") == "1"
RATE = float(yaml.safe_load(Path("viz/settings.yaml").read_text(encoding="utf-8"))["tick_rate"])
BEFORE, AFTER, FIRST = 5.0, 10.0, 30.0
EVAL = ROOT / "final_eval_v1"
TRAIN = ROOT / "20261114_010803_main_v1"
MANIFEST = OUT / "scenes_manifest.json"


def stop(msg):
    sys.exit(f"멈춤: {msg}")


def eval_file(cond, s):
    idx = json.loads((EVAL / "index.json").read_text(encoding="utf-8"))
    d = {r["condition"]: ROOT / r["run_dir"] / "replay" for r in idx["runs"]}
    return d[cond] / f"{cond}_s{s}_g1.npz"


def train_file(cond, s, g):
    idx = json.loads((TRAIN / "index.json").read_text(encoding="utf-8"))
    dirs = [r["run_dir"] for r in idx["runs"] if r["condition"] == cond and s in r["seeds"]]
    if len(dirs) != 1:
        stop(f"학습 묶음에서 {cond}, 시드 {s}의 폴더가 하나가 아니다: {dirs}")
    return TRAIN / dirs[0] / "replay" / f"{cond}_s{s}_g{g}.npz"


def first_diff(a_path, b_path):
    a = np.load(a_path)["action"]; b = np.load(b_path)["action"]
    n = min(len(a), len(b))
    diff = np.flatnonzero((a[:n] != b[:n]).any(axis=1))
    if diff.size:
        return int(diff[0]), len(a), len(b)
    if len(a) != len(b):
        print(f"  주의: action은 같고 길이만 다르다 ({len(a)} / {len(b)})")
        return n, len(a), len(b)
    return None, len(a), len(b)


def render(path, out_dir, start=None, dur=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "viz", str(path), "--view", "game", "--out-dir", str(out_dir)]
    if start is not None:
        cmd += ["--start", f"{start:.4f}", "--duration", f"{dur:.4f}"]
    subprocess.run(cmd, check=True)
    return out_dir / f"{path.stem}_game.mp4"


def hstack(left, right, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = __import__("imageio_ffmpeg").get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(left), "-i", str(right),
                    "-filter_complex", "hstack=inputs=2", "-pix_fmt", "yuv420p", str(out)], check=True)


if MANIFEST.exists():
    stop(f"{MANIFEST}가 이미 있다")
manifest = {"rule": "D§35.5 · 40.2", "tick_rate": RATE, "synapse": None, "regimes": {}}

# (가) 시냅스 학습 전후
for s in range(9000, 9010):
    a, b = eval_file("b0", s), eval_file("regime1", s)
    for p in (a, b):
        if not p.exists():
            stop(f"재생 파일이 없다: {p}")
    t, na, nb = first_diff(a, b)
    if t is None:
        print(f"[synapse] 시드 {s}: 끝까지 같다 → 다음 시드")
        continue
    ra = np.load(a); rb = np.load(b)
    flies = np.flatnonzero(ra["action"][t] != rb["action"][t]).tolist()
    t_s = t / RATE
    start = max(0.0, t_s - BEFORE)
    end = min(t_s + AFTER, min(na, nb) / RATE)
    print(f"[synapse] 시드 {s}, 판 1: t* = 틱 {t} ({t_s:.2f}초), 구간 {start:.2f} ~ {end:.2f}초"
          f"  (왼쪽 b0, 오른쪽 regime1; 판 길이 {na} / {nb} 틱)", flush=True)
    print(f"  t*의 갈라진 초파리(0부터) {flies} | 왼쪽 action {ra['action'][t].tolist()} 오른쪽 {rb['action'][t].tolist()}"
          f" | committed 왼쪽 {ra['committed'][t].tolist()} 오른쪽 {rb['committed'][t].tolist()}", flush=True)
    entry = {"seed": s, "game": 1, "left": "b0", "right": "regime1", "left_file": str(a), "right_file": str(b),
             "len_left": na, "len_right": nb, "t_star_tick": t, "t_star_s": t_s, "start_s": start, "end_s": end,
             "flies_at_t_star": flies, "action_left": ra["action"][t].tolist(), "action_right": rb["action"][t].tolist(),
             "committed_left": ra["committed"][t].tolist(), "committed_right": rb["committed"][t].tolist()}
    if not DRY:
        clip = OUT / f"synapse_s{s}_g1_pair.mp4"
        full = OUT / "full" / f"synapse_s{s}_g1_full_pair.mp4"
        for p in (clip, full):
            if p.exists():
                stop(f"{p}가 이미 있다")
        hstack(render(a, OUT / "synapse" / "b0", start, end - start),
               render(b, OUT / "synapse" / "regime1", start, end - start), clip)
        hstack(render(a, OUT / "full" / "synapse" / "b0"), render(b, OUT / "full" / "synapse" / "regime1"), full)
        entry["video"] = str(clip); entry["video_full"] = str(full)
        print(f"  영상: {clip}\n  영상: {full}", flush=True)
    manifest["synapse"] = entry
    break
else:
    print("[synapse] 모든 시드에서 끝까지 같다. 영상 없음.")

# (나) 세 체제
for cond in ("regime1", "regime2", "regime3"):
    g1, g3000 = train_file(cond, 8000, 1), train_file(cond, 8000, 3000)
    for p in (g1, g3000):
        if not p.exists():
            stop(f"재생 파일이 없다: {p}")
    n1, n3 = len(np.load(g1)["action"]), len(np.load(g3000)["action"])
    d1, d3 = min(FIRST, n1 / RATE), min(FIRST, n3 / RATE)
    print(f"[regimes] {cond}, 학습 시드 8000: 판 1 {n1}틱 → 0 ~ {d1:.2f}초, 판 3000 {n3}틱 → 0 ~ {d3:.2f}초", flush=True)
    entry = {"seed": 8000, "left_game": 1, "right_game": 3000, "left_file": str(g1), "right_file": str(g3000),
             "len_left": n1, "len_right": n3, "dur_left_s": d1, "dur_right_s": d3}
    if not DRY:
        out = OUT / "regimes" / f"{cond}_s8000_g1_vs_g3000.mp4"
        if out.exists():
            stop(f"{out}가 이미 있다")
        hstack(render(g1, OUT / "regimes" / cond, 0.0, d1), render(g3000, OUT / "regimes" / cond, 0.0, d3), out)
        entry["video"] = str(out)
        print(f"  영상: {out}", flush=True)
    manifest["regimes"][cond] = entry

if not DRY:
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"기록: {MANIFEST}")
