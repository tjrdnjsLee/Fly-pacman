"""PPO 팩맨 학습기 (TRD 12.2, 지시서 4.2 체크포인트와 재개). pacman-ai `training/trainer.py`를 옮겼다.

체크포인트에 담는 것: 정책 가중치, 옵티마이저, 갱신 번호, 커리큘럼 단계, 모든 난수 상태(넘파이 · torch ·
환경), 환경 상태(엔진 · 유령 · 프레임 이력), 보상 정규화, RND, 관측 규격 버전, 저장 형식 번호, 엔진 숫자(파워 분모 ·
초기 목숨), 총 갱신 수, 설정 사본, 코드 커밋 해시.
재개하면 끊지 않고 돌린 것과 같은 결과가 나온다 (같은 장치 · 같은 시드, tests/test_pacman_trainer.py).

학습 일정(학습률 · 엔트로피)은 설정의 총 갱신 수(`training.total_updates`)로 계산하고, 총 갱신 수는 체크포인트와
설정 해시에 들어간다. 이번 실행을 어디서 멈출지는 별도 인자(`train(stop_at=...)`, `--stop-at`)라서, 중간에 멈추고
재개해도 일정이 바뀌지 않는다 (학습 종료는 평가 승률이 평탄해지는 시점으로 판단한다).

Ctrl+C: 이번 갱신을 마치고 저장한 뒤 끝낸다. 한 번 더 누르면 저장 없이 바로 끝낸다.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from flypac.pacman.obs import NUM_CHANNELS, NUM_SCALARS, OBS_VERSION
from flypac.pacman.ppo_agent import PPO, RNDModule, RolloutBuffer, RunningMeanStd, build_network, grid_tensor
from flypac.pacman.preflight import curriculum_phases, eval_opponent, require_ready
from flypac.pacman.train_env import PacmanTrainEnv

CKPT_FORMAT = 2
# 재개 때 달라도 되는 설정 (실행 방식만 바꾸고 학습 궤적을 바꾸지 않는 값). total_updates는 일정을 정하므로 여기 없다.
_RUNTIME_KEYS = {"device", "run_dir"}
_RUNTIME_TRAINING_KEYS = {"log_every", "eval_every", "eval_games", "eval_max_ticks",
                          "checkpoint_every", "keep_checkpoints"}


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def learning_config_hash(cfg) -> str:
    """학습 궤적을 정하는 설정의 해시. 재개할 때 대조한다 (기록 주기 · 장치 · 경로는 뺀다. 총 갱신 수는 일정을 정하므로 넣는다)."""
    d = cfg.to_dict()
    p = d.get("pacman_train", {})
    for k in _RUNTIME_KEYS:
        p.pop(k, None)
    for k in _RUNTIME_TRAINING_KEYS:
        p.get("training", {}).pop(k, None)
    d.pop("experiment", None)
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def git_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (head or "unknown") + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def check_checkpoint_header(s: dict, path, warn) -> None:
    """저장 형식 번호 · 관측 규격 버전이 다르면 멈추고, 저장한 커밋이 지금 코드와 다르면 경고한다 (warn(메시지))."""
    if s.get("format") != CKPT_FORMAT:
        raise ValueError(f"저장 형식 번호가 다르다: 체크포인트 {s.get('format')}, 코드 {CKPT_FORMAT} ({path})")
    if s.get("obs_version") != OBS_VERSION:
        raise ValueError(f"관측 규격 버전이 다르다: 체크포인트 {s.get('obs_version')}, 코드 {OBS_VERSION}")
    saved, now = s.get("commit"), git_commit()
    if saved != now:
        warn(f"경고: 체크포인트를 저장한 커밋({saved})이 지금 코드({now})와 다르다")


def _fmt_eta(sec: float) -> str:
    sec = int(max(sec, 0))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


class Trainer:
    def __init__(self, cfg, run_dir: str | Path | None = None, device: str | None = None, quiet: bool = False):
        self.cfg = cfg
        p = cfg.pacman_train
        self.p = p
        self.quiet = quiet
        require_ready(cfg)  # 시작하기 전에 모든 부품을 만들어 본다. 모자라면 여기서 멈춘다 (난수를 쓰지 않는다)
        self.run_dir = Path(run_dir if run_dir is not None else p.run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.device = resolve_device(device or str(p.device))
        self.seed = int(p.seed)
        torch.manual_seed(self.seed)
        self.rng = np.random.default_rng(np.random.SeedSequence([self.seed, 1]))  # 섞기 · RND 표본

        self.phases = curriculum_phases(p)  # [(from_update, 유령, 조건 이름)]
        self.eval_ghosts, self.eval_condition = eval_opponent(p)
        self.total = int(p.training.total_updates)  # 학습 일정의 총 길이. 멈출 지점(stop_at)과 다르다
        self.phase = 0
        self.N = int(p.num_envs)
        self.T = int(p.ppo.rollout_steps)
        self.env = PacmanTrainEnv(cfg, self.N, self.seed, ghosts=self.phases[0][1], condition=self.phases[0][2])
        self.network = build_network(p.network, self.env.grid_shape, NUM_SCALARS)
        self.ppo = PPO(self.network, p.ppo, self.device)
        self.rollout = RolloutBuffer(self.N, self.T, self.env.grid_shape, NUM_SCALARS)
        self.reward_norm = RunningMeanStd(float(p.ppo.reward_norm_clip))
        r = p.rnd
        self.use_rnd = bool(r.enabled)
        self.rnd = self.rnd_norm = None
        if self.use_rnd:
            H, W = self.env.grid_shape[1:]
            self.rnd = RNDModule(NUM_CHANNELS, H, W, int(r.hidden_size), int(r.output_size),
                                 float(r.learning_rate)).to(self.device)
            self.rnd_norm = RunningMeanStd(float(r.clip))
        self.update = 0  # 다음에 할 갱신 번호
        self.entropy_boost_until = -1
        self.best = None  # (승률, 평균 점수)
        self.obs = None
        self.recent: list[dict] = []  # 최근 끝난 판 (진행 출력용)
        self._stop = False
        self._sec_per_update = None

    # ---- 일정 ----
    def _entropy_coef(self, update: int, total: int) -> float:
        c = self.p.ppo
        start, end, frac = float(c.entropy_coef_start), float(c.entropy_coef_end), float(c.entropy_anneal_fraction)
        if update < total * frac:
            coef = start
        else:
            coef = start + (end - start) * min((update - total * frac) / (total * (1 - frac)), 1.0)
        if update < self.entropy_boost_until:
            coef = max(coef, start * 2)
        return coef

    def _phase_for(self, update: int) -> int:
        return max(i for i, (u, _, _) in enumerate(self.phases) if update >= u)

    def _advance_curriculum(self, update: int) -> None:
        target = self._phase_for(update)
        if target != self.phase:
            self.phase = target
            self.obs = self.env.set_ghosts(self.phases[target][1], self.phases[target][2])
            self.entropy_boost_until = update + int(self.p.entropy_boost_updates)
            self._say(f"[갱신 {update}] 커리큘럼 단계 {target} (유령 {self.phases[target][1]}, 조건 {self.phases[target][2]}), "
                      f"엔트로피 올림 {self.entropy_boost_until}까지")

    # ---- 출력 ----
    def _say(self, line: str) -> None:
        if not self.quiet:
            print(line, flush=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "train_log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _csv(self, row: dict) -> None:
        path = self.run_dir / "train_log.csv"
        new = not path.exists()
        with open(path, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    # ---- 한 번의 갱신 ----
    def _one_update(self, update: int) -> dict:
        p = self.p
        self.ppo.anneal_lr(update, self.total)
        self.ppo.entropy_coef = self._entropy_coef(update, self.total)
        self._advance_curriculum(update)
        self.rollout.reset()
        obs = self.obs
        finished = []
        for _ in range(self.T):
            legal = self.env.legal_masks()
            actions, logp, values = self.ppo.select_action(obs["grid"], obs["scalars"], legal)
            nxt, rewards, dones, info = self.env.step(actions)
            finished += info["finished"]
            if self.use_rnd:
                latest = grid_tensor(nxt["grid"][:, -NUM_CHANNELS:], self.device)
                raw = self.rnd.intrinsic_reward(latest).cpu().numpy()
                self.rnd_norm.update(raw)
                intr = np.clip(raw / (np.sqrt(self.rnd_norm.var) + 1e-8), -self.rnd_norm.clip, self.rnd_norm.clip)
                rewards = rewards + float(p.rnd.intrinsic_coef) * intr
            self.reward_norm.update(rewards)
            self.rollout.insert(obs["grid"], obs["scalars"], actions, logp, values,
                                self.reward_norm.normalize(rewards), dones, legal)
            obs = nxt
        self.obs = obs
        last = self.ppo.get_value(obs["grid"], obs["scalars"], self.env.legal_masks())
        self.rollout.compute_gae(last, float(p.ppo.gamma), float(p.ppo.gae_lambda))
        metrics = self.ppo.update(self.rollout, self.rng)
        if self.use_rnd:
            metrics["rnd_loss"] = self._update_rnd()
        metrics["finished"] = finished
        return metrics

    def _update_rnd(self) -> float:
        r = self.p.rnd
        latest = self.rollout.obs_grids[:, :, -NUM_CHANNELS:]
        flat = latest.reshape(-1, *latest.shape[2:])
        total = flat.shape[0]
        n = min(max(int(total * float(r.update_proportion)), int(r.min_samples)), total)
        idx = self.rng.choice(total, size=n, replace=False)
        return self.rnd.update(grid_tensor(flat[idx], self.device))

    # ---- 학습 ----
    def train(self, stop_at: int | None = None) -> None:
        """학습 일정(총 `training.total_updates` 갱신) 안에서 갱신 번호 stop_at에 이를 때까지 하고 저장한 뒤 멈춘다.

        stop_at이 없으면 총 길이까지 한다. stop_at은 이번 실행을 멈출 지점일 뿐 일정을 바꾸지 않는다:
        학습률 · 엔트로피 일정은 언제나 총 길이로 계산한다.
        """
        tr = self.p.training
        total = self.total
        end = total if stop_at is None else int(stop_at)
        if not 0 <= end <= total:
            raise ValueError(f"stop_at {end}은 0 이상 총 갱신 수 {total} 이하여야 한다")
        if self.obs is None:
            self.obs = self.env.reset()
        old = self._install_sigint()
        try:
            while self.update < end:
                u = self.update
                t0 = time.time()
                m = self._one_update(u)
                dt = time.time() - t0
                self.update = u + 1
                self._sec_per_update = dt if self._sec_per_update is None else 0.9 * self._sec_per_update + 0.1 * dt
                self.recent = (self.recent + m["finished"])[-100:]
                if u % int(tr.log_every) == 0:
                    self._log(u, total, m, dt)
                is_best = False
                if int(tr.eval_every) > 0 and u > 0 and u % int(tr.eval_every) == 0:
                    is_best = self._evaluate(u)
                if self.update % int(tr.checkpoint_every) == 0 or self.update == end or is_best:
                    self.save(is_best=is_best)
                if self._stop:
                    self.save()
                    self._say(f"[갱신 {u}] Ctrl+C: 저장하고 끝냅니다. 재개: --resume {self.ckpt_dir / 'latest.pt'}")
                    return
        finally:
            self._restore_sigint(old)

    def _log(self, u: int, total: int, m: dict, dt: float) -> None:
        sps = self.N * self.T / max(dt, 1e-9)
        rec = self.recent
        mean_ret = float(np.mean([g["return"] for g in rec])) if rec else float("nan")
        mean_score = float(np.mean([g["score"] for g in rec])) if rec else float("nan")
        win = float(np.mean([g["winner"] == "PACMAN" for g in rec])) if rec else float("nan")
        eta = (total - self.update) * (self._sec_per_update or dt)
        self._say(f"[갱신 {u}/{total}] 단계 {self.phase} | 스텝/초 {sps:,.0f} | 판 평균 보상 {mean_ret:.1f} "
                  f"점수 {mean_score:.0f} 승률 {win:.0%} (최근 {len(rec)}판) | 남은 시간 {_fmt_eta(eta)}")
        self._csv({"update": u, "phase": self.phase, "steps_per_sec": round(sps, 1), "sec": round(dt, 3),
                   "mean_return": mean_ret, "mean_score": mean_score, "win_rate": win, "games": len(rec),
                   "policy_loss": m["policy_loss"], "value_loss": m["value_loss"], "entropy": m["entropy"],
                   "rnd_loss": m.get("rnd_loss", float("nan")),
                   "lr": self.ppo.optimizer.param_groups[0]["lr"], "entropy_coef": self.ppo.entropy_coef})

    # ---- 평가 (원작 점수 · 승률, TRD 12.2) ----
    @torch.no_grad()
    def evaluate(self, n_games: int, ghosts: str, condition: str, seed: int, max_ticks: int) -> dict:
        """탐욕 행동으로 칸마다 한 판씩. 학습 환경과 별개라 학습 궤적에 영향이 없다.

        CENSORED(감시 장치 중단) 판은 승률 분모와 평균 점수에서 빼고 `censored`로 따로 센다 (best.pt 선택에도
        들어가지 않는다). `games`는 CENSORED를 뺀 끝난 판 수, `unfinished`는 max_ticks 안에 끝나지 않은 판 수.
        """
        env = PacmanTrainEnv(self.cfg, n_games, seed, ghosts=ghosts, condition=condition)
        obs = env.reset()
        results: dict[int, dict] = {}
        self.network.eval()
        for _ in range(max_ticks):
            legal = env.legal_masks()
            logits, _ = self.network(grid_tensor(obs["grid"], self.device), self.ppo._t(obs["scalars"]),
                                     self.ppo._t(legal))
            obs, _, _, info = env.step(logits.argmax(dim=-1).cpu().numpy())
            for g in info["finished"]:
                results.setdefault(g["slot"], g)
            if len(results) == n_games:
                break
        self.network.train()
        done = list(results.values())
        valid = [g for g in done if g["winner"] != "CENSORED"]
        return {"games": len(valid), "censored": len(done) - len(valid), "unfinished": n_games - len(done),
                "win_rate": float(np.mean([g["winner"] == "PACMAN" for g in valid])) if valid else float("nan"),
                "mean_score": float(np.mean([g["score"] for g in valid])) if valid else float("nan")}

    def _evaluate(self, u: int) -> bool:
        tr = self.p.training
        res = self.evaluate(int(tr.eval_games), self.eval_ghosts, self.eval_condition, self.seed + 1_000_000,
                            int(tr.eval_max_ticks))
        if res["games"] == 0:  # 계산할 판이 없다 (모두 중단 · 미완): 최고로 뽑지 않는다
            is_best = False
        else:
            key = (res["win_rate"], res["mean_score"])
            is_best = self.best is None or key > self.best
            if is_best:
                self.best = key
        self._say(f"[갱신 {u}] 평가 ({self.eval_ghosts} 유령, {res['games']}판, 중단 {res['censored']}, "
                  f"미완 {res['unfinished']}): 승률 {res['win_rate']:.0%} 평균 점수 {res['mean_score']:.0f}"
                  + (" (최고)" if is_best else ""))
        path = self.run_dir / "eval_log.csv"
        new = not path.exists()
        with open(path, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["update", "ghosts", "games", "censored", "unfinished", "win_rate", "mean_score"])
            w.writerow([u, self.eval_ghosts, res["games"], res["censored"], res["unfinished"], res["win_rate"],
                        res["mean_score"]])
        return is_best

    # ---- Ctrl+C ----
    def _install_sigint(self):
        def handler(signum, frame):
            if self._stop:
                raise KeyboardInterrupt
            self._stop = True
            print("\nCtrl+C: 이번 갱신을 마치고 저장합니다. 한 번 더 누르면 저장 없이 끝냅니다.", flush=True)
        try:
            return signal.signal(signal.SIGINT, handler)
        except ValueError:  # 주 스레드가 아니면 설치할 수 없다
            return None

    @staticmethod
    def _restore_sigint(old) -> None:
        if old is not None:
            signal.signal(signal.SIGINT, old)

    # ---- 체크포인트 ----
    def state(self) -> dict:
        s = {
            "format": CKPT_FORMAT,
            "obs_version": OBS_VERSION,
            "engine_constants": self.env.constants,  # 파워 분모 · 초기 목숨 (실험 시작 때 대조한다)
            "total_updates": self.total,
            "update": self.update,
            "phase": self.phase,
            "entropy_boost_until": self.entropy_boost_until,
            "best": self.best,
            "recent": self.recent,
            "model": self.network.state_dict(),
            "optimizer": self.ppo.optimizer.state_dict(),
            "reward_norm": self.reward_norm.state_dict(),
            "rng_numpy": self.rng.bit_generator.state,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "env": self.env.state_bytes(),
            "env_obs": self.obs,
            "config": self.cfg.to_dict(),
            "config_hash": learning_config_hash(self.cfg),
            "commit": git_commit(),
            "device": str(self.device),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if self.use_rnd:
            s["rnd"] = {"model": self.rnd.state_dict(), "optimizer": self.rnd.optimizer.state_dict(),
                        "norm": self.rnd_norm.state_dict()}
        return s

    def save(self, is_best: bool = False) -> Path:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        torch.save(self.state(), buf)
        data = buf.getvalue()
        path = self.ckpt_dir / f"update_{self.update:06d}.pt"
        for target in [path, self.ckpt_dir / "latest.pt"] + ([self.ckpt_dir / "best.pt"] if is_best else []):
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)  # 쓰는 도중 끊겨도 이전 파일이 남는다
        keep = int(self.p.training.keep_checkpoints)
        olds = sorted(self.ckpt_dir.glob("update_*.pt"))
        for f in olds[:-keep] if keep > 0 else []:
            f.unlink()
        return path

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if path.is_dir():
            path = path / "checkpoints" / "latest.pt" if (path / "checkpoints").is_dir() else path / "latest.pt"
        s = torch.load(path, map_location="cpu", weights_only=False)
        check_checkpoint_header(s, path, warn=self._say)
        if int(s["total_updates"]) != self.total:
            raise ValueError(f"총 갱신 수가 체크포인트와 다르다: 체크포인트 {s['total_updates']}, 설정 {self.total}. "
                             "학습 일정(학습률 · 엔트로피)이 바뀌므로 허용하지 않는다. 이번 실행을 중간에서 멈추려면 --stop-at을 쓴다")
        if s["engine_constants"] != self.env.constants:
            raise ValueError(f"엔진 숫자가 체크포인트와 다르다: 체크포인트 {s['engine_constants']}, 설정 {self.env.constants}")
        if s["config_hash"] != learning_config_hash(self.cfg):
            raise ValueError("학습 설정이 체크포인트와 다르다 (기록 주기 · 장치 · 경로 외의 값). "
                             f"체크포인트의 설정 사본: {path}의 'config'")
        self.network.load_state_dict(s["model"])
        self.ppo.optimizer.load_state_dict(s["optimizer"])
        self.update, self.phase = int(s["update"]), int(s["phase"])
        self.entropy_boost_until, self.best = int(s["entropy_boost_until"]), s["best"]
        self.recent = list(s["recent"])
        self.reward_norm.load_state_dict(s["reward_norm"])
        self.rng.bit_generator.state = s["rng_numpy"]
        torch.set_rng_state(s["rng_torch"])
        if s["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(s["rng_cuda"])
        self.env.load_state_bytes(s["env"])
        self.obs = s["env_obs"]
        if self.use_rnd:
            self.rnd.load_state_dict(s["rnd"]["model"])
            self.rnd.optimizer.load_state_dict(s["rnd"]["optimizer"])
            self.rnd_norm.load_state_dict(s["rnd"]["norm"])
        self._say(f"재개: {path} (갱신 {self.update}부터, 저장 커밋 {s['commit']}, 지금 커밋 {git_commit()})")
