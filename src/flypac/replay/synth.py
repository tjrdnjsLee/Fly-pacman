"""가짜 재생 로그 생성기 (TRD 19.4, 19.12). 시각화 팀이 M1부터 쓴다.

원작 맵 위에서 무작위로 움직이는 초파리 넷과 냄새 원천(팩맨 자리) 하나로 형식에 맞는 로그를 만든다.
뇌 필드(TRD 19.13)도 만들 수 있다: 무작위 상위 k KC, 가끔 ±값의 DAN.
뇌 회로 크기는 자리표시값이다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flypac.core import directions as d
from flypac.core.events import Kind
from flypac.core.rng import make_rng
from flypac.core.types import GameView
from flypac.field.odor import Field
from flypac.maps.original import MapSpec, load_original
from flypac.replay import schema as s
from flypac.replay.writer import ReplayBuffer, replay_filename, write_replay
from flypac.senses.sense import sense

# TODO(미결 2, 3): 회로 크기와 KC 희소도는 회로 추출 뒤 확정. 아래는 자리표시값이다.
PLACEHOLDER_N_KC = 2000
PLACEHOLDER_N_MBON = 40
PLACEHOLDER_K = 20
PLACEHOLDER_DAN_PROB = 0.3


def _walk(amap: MapSpec, cell: int, heading: int, rng) -> tuple[int, int, bool]:
    """무작위로 한 칸 (반전 금지, 막다른 길이면 반전). 반환: (새 칸, 새 방향, 갈림길 결정이었는지)."""
    nb = amap.nbr[cell]
    ok = [k for k in range(4) if nb[k] >= 0 and not (amap.grid[tuple(amap.cell_rc[nb[k]])] in (4, 5))]
    fwd = [k for k in ok if k != (heading + 2) % 4]
    opts = fwd or ok
    k = int(rng.choice(opts))
    return int(nb[k]), k, len(opts) >= 2


def make_synthetic_arrays(
    T: int = 300,
    seed: int = 0,
    game_no: int = 1,
    brain: bool = False,
    phero: bool = False,
    k: int = PLACEHOLDER_K,
    n_kc: int = PLACEHOLDER_N_KC,
    n_mbon: int = PLACEHOLDER_N_MBON,
    condition: str = "synthetic",
    odor_D: float = 1.0,
    odor_lam: float = 0.05,
    c0: float = 0.01,
) -> dict:
    """형식 버전 s.FORMAT_VERSION에 맞는 가짜 로그 배열 사전. 마지막 틱에 포획 장면 하나를 넣는다."""
    if T < 3:
        raise ValueError("T >= 3 이어야 한다")
    rng = make_rng(seed, game_no, "synth")
    amap = load_original()
    N, F = amap.nbr.shape[0], len(amap.fly_starts)
    field = Field(amap, odor_D, odor_lam)
    field.reset(np.ones(1, bool))
    ph_field = Field(amap, odor_D, odor_lam) if phero else None
    if ph_field is not None:
        ph_field.reset(np.ones(1, bool))

    buf = ReplayBuffer(with_phero=phero, with_brain=brain)
    pac = int(amap.pacman_start)
    pac_dir = d.WEST
    fly = amap.fly_starts.astype(np.int64).copy()
    fly_dir = np.array([d.WEST] + [d.NORTH] * (F - 1))
    dots = amap.dots.copy()
    score = 0
    exit_cell = int(amap.fly_starts[0])
    for t in range(T):
        last = t == T - 1
        # 집 안 초파리는 차례로 나온다 (가짜: 일정 간격)
        for j in range(1, F):
            if t == 10 * j:
                fly[j], fly_dir[j] = exit_cell, d.NORTH
        in_house = np.array([amap.grid[tuple(amap.cell_rc[c])] in (4, 5) for c in fly])
        events: list[tuple[str, int]] = []
        # 감각은 이동 전 상태에서 읽는다 (실행기의 5번과 같은 순서)
        view = GameView(
            pac_cell=pac, pac_dir=pac_dir, lives=3, power=False, power_ticks=0, ghost_streak=0,
            fly_cell=fly.copy(), fly_dir=fly_dir.copy(), fly_z=np.zeros(F, bool), fly_in_house=in_house,
            dots=dots.copy(), energizers=np.zeros(N, bool), fruit=False, tick=t, score=score,
            learning_active=bool(t >= 10 * (F - 1)), elroy_stage=0, frozen=False,
            pac_acc=0.0, fly_acc=np.zeros(F),
        )
        legal = np.zeros((1, F, 4), bool)
        decide = np.zeros((1, F), bool)
        action = np.full(F, -1, dtype=np.int64)
        pi = np.zeros((F, 4))
        new_fly, new_dir = fly.copy(), fly_dir.copy()
        for j in range(F):
            if in_house[j]:
                continue
            c, h = int(fly[j]), int(fly_dir[j])
            nb = amap.nbr[c]
            ok = [q for q in range(4) if nb[q] >= 0 and amap.grid[tuple(amap.cell_rc[nb[q]])] not in (4, 5)]
            opts = [q for q in ok if q != (h + 2) % 4] or ok
            for q in opts:
                legal[0, j, int(d.abs_to_ego(h, q))] = True
            if len(opts) >= 2:
                decide[0, j] = True
                egos = np.flatnonzero(legal[0, j])
                action[j] = int(rng.choice(egos))
                pi[j, egos] = 1.0 / len(egos)
                q = int(d.ego_to_abs(h, action[j]))
            else:
                q = opts[0]
            new_fly[j], new_dir[j] = int(nb[q]), q
        phero_c = ph_field.c if ph_field is not None else np.zeros((1, N))
        obs = sense(field.c, phero_c, [view], (legal, decide), c0, nbr=amap.nbr)

        frame = dict(
            pac_cell=pac, pac_dir=pac_dir, pac_acc=rng.random(), fly_cell=fly.copy(), fly_dir=fly_dir.copy(),
            fly_acc=rng.random(F), fly_z=np.zeros(F, bool), fly_in_house=in_house, lives=3,
            power_ticks=0, score=score, fruit=0, odor_field=field.c[0].copy(), sense=np.concatenate(
                [obs.odor[0], obs.phero[0]], axis=1), action=action, pi=pi,
        )
        if phero:
            frame["phero_field"] = ph_field.c[0].copy()
        if brain:
            kc = np.full((F, k), -1, dtype=np.int16)
            pn = np.concatenate([obs.odor[0], obs.z[0][:, None].astype(float), obs.phero[0]], axis=1)
            dan = np.zeros(F)
            for j in range(F):
                if not in_house[j]:
                    kc[j] = rng.choice(n_kc, size=k, replace=False)
                if decide[0, j] and rng.random() < PLACEHOLDER_DAN_PROB:
                    dan[j] = rng.choice([-1.0, 1.0]) * rng.random()
            frame.update(kc_active=kc, pn_input=pn, mbon_out=rng.random((F, n_mbon)), dan=dan)
        buf.add_frame(**frame)

        # 팩맨(냄새 원천)도 무작위로 걷는다
        if last:  # 마지막 틱: 0번 초파리가 팩맨 칸으로 들어가 포획
            new_fly[0] = pac
            events += [(Kind.CATCH, 0), (Kind.LIFE_LOST, -1)]
        else:
            pac, pac_dir, _ = _walk(amap, pac, pac_dir, rng)
            if dots[pac]:
                dots[pac] = False
                score += 10
                events.append((Kind.DOT, -1))
        for kind, f in events:
            buf.add_event(t, kind, f)
        S = np.zeros((1, N))
        S[0, pac] = 1.0
        field.step(S, np.ones(1, bool))
        if ph_field is not None:
            Sp = np.zeros((1, N))
            Sp[0, new_fly[~in_house]] = 1.0
            ph_field.step(Sp, np.ones(1, bool))
        fly, fly_dir = new_fly, new_dir

    meta = s.make_meta(condition, seed, game_no, "synthetic", "synthetic", "original")
    return buf.arrays(meta)


def write_synthetic(out_dir: Path, **kw) -> Path:
    """가짜 로그를 만들어 `<조건>_s<시드>_g<판>.npz`로 저장한다."""
    arrays = make_synthetic_arrays(**kw)
    meta = s.parse_meta(arrays[s.META_KEY])
    return write_replay(Path(out_dir) / replay_filename(meta["condition"], meta["seed"], meta["game_no"]), arrays)
