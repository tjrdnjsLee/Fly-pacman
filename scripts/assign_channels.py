"""MBON 묶음 · PN 채널 배정과 최종 유효성 검사 (M1 뇌 작업 지시서 ① 6절). 토큰 불필요.

    uv run python scripts/assign_channels.py [--config configs/circuit_assign_v1.yaml]

원본 회로 파일만 읽는다. 배정 결과는 시드로 재현되므로 저장하지 않고, 보고서만 쓴다
(runs/assign/report_v1.md, 저장소에 올리지 않음).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from flypac.brain import assign as A
from flypac.core.config import load_config
from flypac.core.directions import EGO_NAMES

ROOT = Path(__file__).resolve().parents[1]
CLASS_LABEL = {"gamma": "γ", "apbp": "α′β′", "ab": "αβ"}
CH_LABEL = ("냄새 좌", "냄새 앞", "냄새 우", "냄새 뒤", "z", "페로몬 좌", "페로몬 앞", "페로몬 우", "페로몬 뒤")
ACT_LABEL = {"left": "좌", "forward": "직진", "right": "우", "back": "반전"}


def q(x, qs=(0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)):
    x = np.asarray(x, dtype=float)
    return " / ".join(f"{np.quantile(x, v):.0f}" for v in qs)


def glom_line(cfg) -> str:
    """원본 meta에서 θ_PN 적용 전후의 사구체 수를 읽는다."""
    import json

    meta = ROOT / str(cfg["raw"]).replace(".npz", ".meta.json")
    if not meta.exists():
        return "- 사구체 수 (θ_PN 적용 전): meta 없음"
    c = json.loads(meta.read_text(encoding="utf-8"))["counts"]
    return (f"- 사구체 수: 주석 후보 {c['candidates']['glomeruli']} (θ_PN 적용 전) → "
            f"원본 {c['selected']['glomeruli']} (θ_PN 적용 후)")


def run(cfg_path: Path) -> Path:
    cfg = load_config(cfg_path)
    raw = A.load_raw(ROOT / cfg["raw"])
    seeds = list(cfg["seeds"])
    pher = list(cfg["pheromone_glomeruli"])

    cond = A.branch_p_conditions(raw, pher, float(cfg["branch_p_max_ratio"]))
    if cfg["branch"] == "auto":
        branch = "G" if cond["use_g"] else "P"
    elif cfg["branch"] in ("P", "G"):
        branch = cfg["branch"]
    else:
        raise SystemExit(f"알 수 없는 branch: {cfg['branch']}")

    w0 = A.normalized_w0(raw.W0)
    C = A.class_mass(w0, raw.kc_class)
    labels, info = A.assign_mbon_groups(C, int(cfg["mbon_groups"]["seed"]),
                                        int(cfg["mbon_groups"]["max_passes"]))
    diag = A.mbon_diagnostics(raw.W0, raw.kc_class, labels)
    chk = A.circuit_checks(raw)

    L = ["# 단계 4: 배정 보고서 (v1)\n",
         f"- 생성: {datetime.now(timezone.utc).isoformat(timespec='seconds')} (UTC)",
         f"- 설정: `{cfg_path}` (해시 {cfg.hash()}), 원본: `{cfg['raw']}`",
         f"- 원본 크기: PN {len(raw.pn_body)}, KC {len(raw.kc_body)}, MBON {len(raw.mbon_body)}, "
         f"사구체 {len(raw.glom_names)} (θ_PN 적용 후)",
         glom_line(cfg) + "\n",
         "## 분기 P 전환 조건 (6.4)\n",
         "| 사구체 | 원본에 있음 (4.1 통과) | 도달 KC 수 |", "|---|---|---|"]
    L += [f"| {g} | {'예' if cond['present'][g] else '아니오'} | {cond['reach'][g]} |" for g in pher]
    L += ["", f"- 조건 1 (후보 사구체 탈락): **{'발동' if cond['cond1'] else '발동 안 함'}**",
          f"- 조건 2 (도달 KC max/min = {cond['ratio']:.2f} > {cfg['branch_p_max_ratio']}): "
          f"**{'발동' if cond['cond2'] else '발동 안 함'}**",
          f"- 설정 `branch: {cfg['branch']}` → 사용 분기 **{branch}**\n"]

    L += ["## MBON 4묶음 (규칙 2, 시드와 무관)\n",
          f"- 탐욕 배정 시드 {cfg['mbon_groups']['seed']}, 목적값(τ별 상대 편차의 최댓값): "
          f"탐욕 {info['greedy_objective']:.4f} → 국소 탐색 {info['final_objective']:.4f} "
          f"(맞바꾸기 {info['swaps']}회, 상한 {cfg['mbon_groups']['max_passes']})\n",
          "| 묶음 | 크기 | T γ | T α′β′ | T αβ | μ_a | μ_a / 크기 | d_ia 분위수 (0/5/25/50/75/95/100) | MBON |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a, name in enumerate(EGO_NAMES):
        mt = ", ".join(sorted(raw.mbon_type[labels == a]))
        T = diag["T"][a]
        L.append(f"| {ACT_LABEL[name]} | {diag['size'][a]} | {T[0]:.1f} | {T[1]:.1f} | {T[2]:.1f} | "
                 f"{diag['mu'][a]:.4f} | {diag['mu_over_size'][a]:.5f} | {q(diag['d'][:, a])} | {mt} |")
    L += ["", "진단은 기록만 한다 (합격 기준 아님). μ_a = 묶음 a의 연결 밀도 ⟨M_ij⟩, "
          "d_ia = KC i가 묶음 a에서 닿는 MBON 수.\n"]

    kin = chk["kc_input"]
    L += ["## 시드와 무관한 검사 (6.5)\n",
          f"- PN별 최종 KC 연결 수 > 0: 연결 없는 PN {chk['pn_without_kc']}개 "
          f"(PN당 KC 수 분위수 {q(chk['pn_kc_count'])})",
          f"- KC별 선택 PN 입력량 분위수 (0/5/25/50/75/95/100): {q(kin)}. "
          f"{cfg['report']['low_kc_input']} 미만인 KC {int((kin < cfg['report']['low_kc_input']).sum())}개",
          f"- MBON에 닿지 않는 KC: {chk['kc_without_mbon']}개",
          f"- 네 행동 묶음이 비어 있지 않음: {all(diag['size'] > 0)}\n"]

    L += [f"## 시드별 결과 (분기 {branch}, 시드 {seeds[0]} ~ {seeds[-1]})\n",
          "| 시드 | 결과 | 채널별 사구체 수 | 채널별 도달 KC 수 | S_c 최소 / 최대 | α_c 최소 / 최대 | "
          "자카드 평균 / 최대 | 접합 KC 비율 |",
          "|---|---|---|---|---|---|---|---|"]
    n_ok = 0
    for s in seeds:
        gch = A.assign_pn_channels(raw.glom_names, s, branch, pher)
        pch = A.pn_channels(raw.pn_glom, raw.glom_names, gch)
        J = A.channel_matrix(raw.J_pn, pch)
        v = A.validate_seed(raw, J, labels)
        assert np.allclose(v["alpha"] * v["S"], v["S"].mean()) or not v["ok"]
        n_ok += v["ok"]
        n_glom = np.bincount(gch, minlength=A.N_CHANNELS)
        L.append(f"| {s} | {'통과' if v['ok'] else '실패: ' + ', '.join(v['fail_reasons'])} | "
                 f"{' '.join(map(str, n_glom))} | {' '.join(map(str, v['reach_count']))} | "
                 f"{v['S'].min():.0f} / {v['S'].max():.0f} | {np.nanmin(v['alpha']):.2f} / {np.nanmax(v['alpha']):.2f} | "
                 f"{v['jaccard_offdiag_mean']:.3f} / {v['jaccard_offdiag_max']:.3f} | "
                 f"{v['conjunction_fraction']:.1%} |")
    L += ["", f"- 채널 순서: {', '.join(CH_LABEL)}",
          f"- 유효성 통과 시드: {n_ok} / {len(seeds)}",
          f"- 접합 KC 비율 = 냄새 채널 하나 이상과 z 채널에서 모두 입력을 받는 KC의 비율. "
          f"참고값 약 {cfg['report']['conjunction_reference']:.1%}는 동일 확률 · 독립 추출 모형의 값이다."]

    out = ROOT / cfg["report"]["path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"분기 {branch}, 통과 시드 {n_ok}/{len(seeds)}, 보고서: {out.relative_to(ROOT)}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/circuit_assign_v1.yaml"))
    args = ap.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
