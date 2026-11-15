"""나란히 보기 영상: 학습 전후(D§35.5 · 38.10)와 산개(D§39.6)를 같은 규칙으로 만든다.

규칙 (두 짝에 똑같이):
  시드 s0의 판 1에서 두 재생의 `action`이 처음 다른 틱 t*를 찾는다. 끝까지 같으면 다음 시드.
  구간은 t*의 5초 전 ~ 10초 후. 5초 전이 판 시작보다 앞이면 판 시작부터 (끝은 그대로 t* + 10초).
  초 = 틱 / tick_rate (viz/settings.yaml, D§35.5).

  learning : runs/kconf_eval_v1   b0 (왼쪽)   vs regime1  (오른쪽), 시드 9400 ~ 9419
  spread   : runs/spread_v2       solo (왼쪽) vs spread10 (오른쪽), 시드 9700 ~ 9709

결과: runs/videos/<짝>_s<시드>_g1_pair.mp4 (게임 화면 둘을 가로로 붙인 것)와 runs/videos/manifest.json.
재생 파일은 읽기만 한다. 승패 · 점수는 보지 않는다.

  저장소 루트에서:  uv run --with pygame==2.6.1 --with imageio-ffmpeg==0.6.0 --with matplotlib python - < pair_videos.py
  DRY=1 이면 t*와 구간만 출력하고 영상은 만들지 않는다.
"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import yaml

ROOT = Path("runs")
OUT = ROOT / "videos"
DRY = os.environ.get("DRY") == "1"
RATE = float(yaml.safe_load(Path("viz/settings.yaml").read_text(encoding="utf-8"))["tick_rate"])
BEFORE, AFTER = 5.0, 10.0


def kconf_files():
    idx = json.loads((ROOT / "kconf_eval_v1" / "index.json").read_text(encoding="utf-8"))
    d = {r["condition"]: ROOT / r["run_dir"] / "replay" for r in idx["runs"]}
    return lambda cond, s: d[cond] / f"{cond}_s{s}_g1.npz"


def spread_files():
    res = json.loads((ROOT / "spread_v2" / "results.json").read_text(encoding="utf-8"))
    d = {c: ROOT / v["run_dir"] / "replay" for c, v in res["conditions"].items()}
    return lambda cond, s: d[cond] / f"regime3_s{s}_g1.npz"   # 산개 실행은 모두 조건 이름이 regime3


PAIRS = {
    "learning": (kconf_files, "b0", "regime1", range(9400, 9420)),
    "spread": (spread_files, "solo", "spread10", range(9700, 9710)),
}


def first_diff(a_path, b_path):
    a = np.load(a_path)["action"]; b = np.load(b_path)["action"]
    n = min(len(a), len(b))
    diff = np.flatnonzero((a[:n] != b[:n]).any(axis=1))
    if diff.size:
        return int(diff[0]), len(a), len(b)
    if len(a) != len(b):   # 결정이 끝까지 같은데 길이만 다르면 짧은 쪽 끝을 첫 차이로 본다
        print(f"  주의: action은 같고 길이만 다르다 ({len(a)} / {len(b)})")
        return n, len(a), len(b)
    return None, len(a), len(b)


def render(path, label, start, dur, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "viz", str(path), "--view", "game", "--start", f"{start:.4f}",
           "--duration", f"{dur:.4f}", "--out-dir", str(out_dir)]
    subprocess.run(cmd, check=True)
    return out_dir / f"{path.stem}_game.mp4"


if OUT.joinpath("manifest.json").exists():
    sys.exit(f"멈춤: {OUT / 'manifest.json'}가 이미 있다")
manifest = {"rule": "D§35.5 · 38.10 · 39.6", "tick_rate": RATE, "pairs": {}}
for name, (files, left, right, seeds) in PAIRS.items():
    f = files()
    for s in seeds:
        a, b = f(left, s), f(right, s)
        for p in (a, b):
            if not p.exists():
                sys.exit(f"멈춤: 재생 파일이 없다: {p}")
        t, na, nb = first_diff(a, b)
        if t is None:
            print(f"[{name}] 시드 {s}: 끝까지 같다 → 다음 시드")
            continue
        t_s = t / RATE
        start = max(0.0, t_s - BEFORE)
        end = min(t_s + AFTER, min(na, nb) / RATE)
        print(f"[{name}] 시드 {s}, 판 1: t* = 틱 {t} ({t_s:.2f}초), 구간 {start:.2f} ~ {end:.2f}초"
              f"  (왼쪽 {left}, 오른쪽 {right}; 판 길이 {na} / {nb} 틱)", flush=True)
        entry = {"seed": s, "game": 1, "left": left, "right": right, "left_file": str(a), "right_file": str(b),
                 "t_star_tick": t, "t_star_s": t_s, "start_s": start, "end_s": end}
        if not DRY:
            ff = __import__("imageio_ffmpeg").get_ffmpeg_exe()
            la = render(a, left, start, end - start, OUT / name / left)
            rb = render(b, right, start, end - start, OUT / name / right)
            pair = OUT / f"{name}_s{s}_g1_pair.mp4"
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(la), "-i", str(rb),
                            "-filter_complex", "hstack=inputs=2", "-pix_fmt", "yuv420p", str(pair)], check=True)
            entry["video"] = str(pair)
            print(f"  영상: {pair}", flush=True)
        manifest["pairs"][name] = entry
        break
    else:
        print(f"[{name}] 모든 시드에서 끝까지 같다. 영상 없음.")

if not DRY:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"기록: {OUT / 'manifest.json'}")
