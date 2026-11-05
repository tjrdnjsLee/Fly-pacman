# fly-pacman

초파리 버섯체 회로로 움직이는 유령 네 마리가 팩맨을 쫓는 시뮬레이션. 팀원 노트북은 개발과 테스트, 실험 머신은 팩맨 학습과 본 실험을 맡는다.

PRD · TRD는 비공개 문서라 이 저장소에 없다 (3. 참고). 인터페이스 세부는 [`docs/DECISIONS.md`](docs/DECISIONS.md)에 있다.

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

## 4. docs 폴더에 문서 넣기

PRD · TRD는 저장소에 없다. 팀에서 받은 `PRD.md`(첫 줄 v1.1), `TRD.md`(첫 줄 v0.5)를 `docs/`에 복사해 쓴다. **절대 커밋하지 않는다.** `.gitignore`가 막아 주지만, `git add .`나 `git add -A`를 쓰지 말고 파일을 지정해서 add하는 습관을 들인다. 테스트와 CI는 `docs/`의 이 파일들에 의존하지 않는다.

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

## 7. 폴더 소유권

| 폴더 | 담당 | 비고 |
|---|---|---|
| `core/`, 설정 파일 루트, `configs/` | 관리자 | 규격 변경은 관리자만 |
| `engine/` | 관리자 (정비: 팩맨 담당) | 정비 후 관리자 검토로만 수정 |
| `field/`, `senses/`, `brain/`, `recording/`, `tasks/` | 뇌 담당 | 일부 분담 |
| `pacman/`, `ghosts/` | 팩맨 담당 | B1 포함 |
| `regimes/regime1·2·3` | 체제 1 · 2 · 3 담당 | 각자 자기 파일만 |
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

TRD 다음으로 [`docs/DECISIONS.md`](docs/DECISIONS.md)를 따른다.

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
