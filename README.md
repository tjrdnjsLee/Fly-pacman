# fly-pacman

초파리 버섯체 회로로 움직이는 유령 네 마리가 팩맨을 쫓는 시뮬레이션. 팀원 노트북은 개발과 테스트, 실험 머신은 팩맨 학습과 본 실험을 맡는다.

PRD · TRD는 [`docs/PRD.md`](docs/PRD.md), [`docs/TRD.md`](docs/TRD.md)에 있다 (2026-11-06 공개 전환, 4. 참고). 인터페이스 세부는 [`docs/DECISIONS.md`](docs/DECISIONS.md)에 있다.

## 1. 설치

Python은 시스템 것을 쓰지 않는다. uv가 설치하는 고정 버전(3.12)을 쓴다. 기본 설치에는 torch, manim이 들어가지 않는다 (빠르게 설치된다).

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
cd C:\dev
git clone https://github.com/tjrdnjsLee/Fly-pacman.git
cd Fly-pacman
uv sync
uv run pytest
```

**Ubuntu**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/tjrdnjsLee/Fly-pacman.git
cd Fly-pacman
uv sync
uv run pytest
```

선택 묶음: `uv sync --extra pacman` (torch), `uv sync --extra viz` (pygame, matplotlib, imageio-ffmpeg, manim: 시각화 팀만). 의존성을 바꾸는 일은 관리자에게 요청한다.

## 2. 한국어 Windows 함정

| 함정 | 증상 | 규칙 |
|---|---|---|
| 인코딩 | 기본 cp949라 한글 UTF-8 파일에서 오류 | 모든 파일 입출력에 `encoding="utf-8"`. 환경변수 `PYTHONUTF8=1` |
| 한글 경로 | 사용자 폴더가 한글이면 일부 도구 실패 | 저장소는 `C:\dev\` 같은 영문 경로에 복제 |
| 줄바꿈 | 모든 줄이 바뀐 것처럼 보임 | `.gitattributes`로 LF 고정 (이미 설정됨) |
| 경로 표기 | `\`와 `/` 혼용 오류 | `pathlib.Path`만 사용 |

CSV 결과 파일은 UTF-8 BOM(`utf-8-sig`)으로 저장되어 엑셀에서 한글이 깨지지 않는다. 파이썬으로 읽을 때도 `encoding="utf-8-sig"`를 쓴다.

## 3. 실험 머신 주의

- 노트북은 CPU용 torch, 실험 머신은 **CUDA 12.8 이상 빌드 torch** (RTX 50 세대는 PyTorch 2.7.0의 CUDA 12.8 빌드부터 지원).
- 환경 안에 **torch는 하나만** 설치되어 있어야 한다. 예전 CUDA 빌드가 섞이면 "sm_120 호환되지 않음" 경고가 뜬다.
- `nvidia-smi`의 CUDA 버전이 12.8 이상인지 확인한다. torch 설치 방법(uv 선택 의존성 또는 인덱스 지정)은 관리자가 `pyproject.toml`에 정한다.
- 실험 머신: Ryzen 7 7800X3D, 32 GiB, RTX 5070, Ubuntu 26.04. 포스터에 들어갈 결과는 실험 머신에서 돌린 것만 쓴다.

## 4. docs 폴더와 작업 지시서

PRD · TRD는 저장소의 `docs/`에 있다 (`PRD.md` 첫 줄 v1.3, `TRD.md` 첫 줄 v0.7). 관리자가 팀 공유를 위해 올렸고 공개 문서다 (DECISIONS 9절). PRD · TRD는 관리자만 고친다. `docs/DECISIONS.md`에는 각 담당이 PR로 결정을 추가하고, 관리자가 승인한다.

작업 지시서(`*_TASK.md`)는 비공개다. 받은 지시서는 저장소 루트에 두고 **절대 커밋하지 않는다.** `.gitignore`가 막아 주지만, `git add .`나 `git add -A`를 쓰지 말고 파일을 지정해서 add하는 습관을 들인다. 테스트와 CI는 `docs/`의 PRD · TRD에 의존하지 않는다.

## 5. Git 다섯 동작 (GitHub Desktop)

1. **복제**: File → Clone repository → `Fly-pacman` (영문 경로로)
2. **브랜치 만들기**: Current Branch → New Branch
3. **커밋**: 바뀐 파일을 확인하고 메시지를 적어 Commit
4. **푸시**: Push origin
5. **풀 리퀘스트**: Create Pull Request. main에 직접 올리지 않는다.

브랜치 이름 규칙: `역할/이름-작업` (예: `regime1/kim-reward`). 풀 리퀘스트는 자동 검사(Windows · Ubuntu)가 통과해야 합친다.

## 6. 첫 실행

가짜 부품으로 한 판이 끝까지 돈다 (규칙 기반 팩맨, 무작위 초파리, 체제1).

```bash
uv run python scripts/run_experiment.py configs/m0_smoke.yaml
```

`runs/<날짜_시각>_m0_smoke/`에 `config.yaml`, `git_commit.txt`, `games.csv`, `lives.csv`, `flies.csv`, `replay/*.npz`가 생긴다. `runs/`는 저장소에 올라가지 않는다.

시각화 팀용 가짜 재생 로그:

```bash
uv run python -c "from flypac.replay.synth import write_synthetic; print(write_synthetic('runs/synth', T=300, brain=True))"
```

### 회로 파일과 채널 배정 (뇌)

원본 회로 `data/circuit/raw/mb_raw_v1.npz`와 해부 자료 `data/anatomy/mb_anatomy_v1.npz`는 hemibrain v1.2.1에서 한 번 뽑아 저장소에 넣어 두었다 (출처는 `NOTICE`, 결정은 `docs/DECISIONS.md` 10절). 채널 배정과 유효성 검사는 토큰 없이 누구나 다시 돌릴 수 있다.

```bash
uv run python scripts/assign_channels.py
```

`runs/assign/report_v1.md`에 시드 0 ~ 19의 표가 생긴다. 원본을 다시 뽑는 것(`scripts/extract_circuit.py`, `scripts/export_anatomy.py`)은 neuPrint 토큰이 필요한 뇌 담당의 일이다 (`uv run --extra extract ...`, 토큰은 `.env`의 `NEUPRINT_TOKEN`).

### PPO 팩맨 학습 (실험 머신)

torch가 필요하다 (`--extra pacman`). 명령 하나로 시작하고, 대화형 입력은 없다. 설정은 둘이다.

| 설정 | 용도 |
|---|---|
| `configs/pacman_train_smoke.yaml` | 연기 시험용: 가짜 엔진 · 무작위 유령, 평가도 무작위 유령. 코드가 끝까지 도는지만 본다 |
| `configs/pacman_train_v1.yaml` | 본 학습용: 진짜 엔진 · 무작위 → 원작 유령(B1), 평가는 원작 유령. 진짜 엔진 · B1이 병합되기 전에는 시작 검사에서 멈춘다 |

```bash
uv run --extra pacman python scripts/train_pacman.py --config configs/pacman_train_smoke.yaml
uv run --extra pacman python scripts/train_pacman.py --config configs/pacman_train_v1.yaml
```

- **시작 검사:** 학습에 쓸 모든 부품(엔진 숫자, 엔진, 단계별 유령, 평가 상대)을 먼저 만들어 보고, 모자라면 무엇이 없는지 모두 적고 시작하지 않는다 (종료 코드 3). 커리큘럼 단계마다 `condition`(유령이 views를 쓰면 `b1`)을 설정에 직접 적는다.
- **진행:** 갱신마다 한 줄(갱신 번호, 초당 스텝, 판 평균 보상 · 점수 · 승률, 남은 시간)이 화면과 `runs/<run_dir>/train_log.txt`, `train_log.csv`에 남는다. 평가(원작 점수 · 승률, 중단(CENSORED) 판은 승률에서 빼고 따로 센다)는 `eval_log.csv`. **두 로그의 승률 정의가 다르다:** `train_log.csv`의 `win_rate`는 중단 판을 분모에 포함하고(`censored` 열에 중단 수), `eval_log.csv`의 `win_rate`는 중단 판을 뺀다.
- **일정과 멈출 지점:** 학습 일정의 총 길이는 설정의 `training.total_updates`이고 체크포인트와 설정 해시에 들어간다. 이번 실행만 중간에서 멈추려면 `--stop-at 400`을 붙인다 (일정은 바뀌지 않는다).
- **중단:** Ctrl+C를 한 번 누르면 이번 갱신을 마치고 `checkpoints/latest.pt`에 저장한 뒤 끝난다 (두 번 누르면 저장 없이 끝).
- **재개:** 같은 명령에 `--resume runs/<run_dir>`을 붙인다. 갱신 번호 · 난수 · 게임 상태까지 이어진다. 학습 설정(총 갱신 수 포함)이 저장 때와 다르면 멈춘다 (기록 주기 · 장치 · 경로는 바꿔도 된다). 이미 체크포인트가 있는데 `--resume` 없이 시작하면 덮어쓰지 않고 멈춘다.
- **실험에서 쓰기:** 실험 설정에 `pacman: {kind: ppo, checkpoint: <.pt>, device: cpu, eps: 0.0}`을 적고 `scripts/run_experiment.py`로 돌린다. 실험 설정의 `engine.power_s` · `engine.lives`가 체크포인트에 기록된 값과 다르면 시작 때 멈춘다.
- **속도 실측:** `uv run --extra pacman python scripts/bench_pacman.py --config configs/pacman_train_v1.yaml`

## 7. 폴더 소유권

| 폴더 | 담당 | 비고 |
|---|---|---|
| `core/`, 설정 파일 루트, `configs/` | 관리자 | 규격 변경은 관리자만 |
| `engine/` | 관리자 | 정비 후 관리자 검토로만 수정 |
| `field/`, `senses/`, `brain/`, `recording/`, `tasks/` | 뇌 담당 | 일부 분담 |
| `pacman/`, `ghosts/` | 관리자 | B1 포함 |
| `regimes/regime1.py`, `regimes/regime2.py` | 체제 1 · 2 담당 | 각자 자기 파일만 |
| `regimes/regime3.py` | 관리자 | |
| `maps/` | 맵 담당 | |
| `analysis/` | 포스터 담당 | 기록 파일만 읽음 |
| `replay/` | 뇌 담당 | 형식 변경은 시각화 팀과 합의 |
| `viz/`, `presentation/` | 시각화 팀 | 재생 로그만 읽음 |
| `tests/` | 각 모듈 담당 | 자기 모듈 테스트 |
| `third_party/` | 관리자 | 외부 코드 보관본. 수정하지 않는다 |

의존성(`pyproject.toml`, `uv.lock`) 변경은 관리자에게 요청하거나 그것만 담은 별도 풀 리퀘스트로 한다.

## 8. 역할별로 읽을 곳

TRD 절 번호다.

| 역할 | 꼭 읽을 TRD 절 |
|---|---|
| 모두 | 0, 1, 3, 15, 16 |
| 체제 1 · 2 · 3 | 4.5, 5.6, 6.1, 11, 18 (7.3 · 7.4 해석) |
| 팩맨 AI | 4.3, 5.2, 5.7, 6, 7, 12 |
| 맵 | 4.2 ~ 4.4, 5.1, 8.2 |
| 시각화 · 포스터 | 4.6, 13.3, 18, 19 (19.13 포함) |
| 뇌 | 4, 5.5, 6, 8 ~ 10, 13, 19.13 |

[`docs/DECISIONS.md`](docs/DECISIONS.md)도 함께 따른다. 승인된 DECISIONS 항목이 해당 TRD 조항보다 우선한다. TRD v0.7에 반영했다 (1 ~ 31절). 이후 항목은 다음 판에서 정리한다.

## 9. 관리자용: main 브랜치 보호 체크리스트

공개 저장소에서만 무료로 적용된다 (비공개로 바꾸면 보호 기능이 꺼질 수 있다). Settings → Branches → main, 또는 `gh api`로 설정한다.

- [ ] 풀 리퀘스트 필수 (main 직접 푸시 금지)
- [ ] 필요한 승인 수: **1** (M0 병합 전까지는 0으로 두었다. M0 병합 뒤 1로 올린다)
- [ ] 상태 검사 필수: `test (windows-latest)`, `test (ubuntu-latest)`
- [ ] 최신 main 기준 검사 (strict)
- [ ] 관리자에게도 적용 (enforce admins)
- [ ] 강제 푸시 금지, main 삭제 금지

현재 설정은 `gh api repos/tjrdnjsLee/Fly-pacman/branches/main/protection`으로 읽을 수 있다.

## 10. 결과 백업 규칙

`runs/`는 저장소에 올리지 않는다. **본 실험 기간에는 매일** `runs/`를 외부 저장소(공유 드라이브나 외장 디스크)로 복사한다. 체크포인트(`checkpoints/`)도 같다.

## 11. 문의처

[GitHub Issues](https://github.com/tjrdnjsLee/Fly-pacman/issues). 팀원은 팀 단톡방.
