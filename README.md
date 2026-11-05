# fly-pacman

초파리 버섯체 회로로 움직이는 유령 네 마리가 팩맨을 쫓는 시뮬레이션. (이 README는 최소판이다. 완성판은 M0 6단계에서 보완한다.)

## 1. 설치

Python은 시스템 것을 쓰지 않는다. uv가 설치하는 고정 버전(3.12)을 쓴다.

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

## 2. 한국어 Windows 함정

| 함정 | 증상 | 규칙 |
|---|---|---|
| 인코딩 | 기본 cp949라 한글 UTF-8 파일에서 오류 | 모든 파일 입출력에 `encoding="utf-8"`. 환경변수 `PYTHONUTF8=1` |
| 한글 경로 | 사용자 폴더가 한글이면 일부 도구 실패 | 저장소는 `C:\dev\` 같은 영문 경로에 복제 |
| 줄바꿈 | 모든 줄이 바뀐 것처럼 보임 | `.gitattributes`로 LF 고정 (이미 설정됨) |
| 경로 표기 | `\`와 `/` 혼용 오류 | `pathlib.Path`만 사용 |

## 3. docs 폴더에 문서 넣기

PRD · TRD는 비공개라 저장소에 없다. 팀에서 받은 `PRD.md`(첫 줄 v1.1), `TRD.md`(첫 줄 v0.5)를 `docs/`에 복사해 쓴다. **절대 커밋하지 않는다.** (`.gitignore`가 막아 주지만, 파일을 지정해서 add하는 습관을 들인다.)

## 4. Git 다섯 동작 (GitHub Desktop)

1. **복제**: File → Clone repository → `Fly-pacman` (영문 경로로)
2. **브랜치 만들기**: Current Branch → New Branch
3. **커밋**: 바뀐 파일을 확인하고 메시지를 적어 Commit
4. **푸시**: Push origin
5. **풀 리퀘스트**: Create Pull Request. main에 직접 올리지 않는다.

브랜치 이름 규칙: `역할/이름-작업` (예: `regime1/kim-reward`).

## 5. 폴더 소유권 (TRD 3.2)

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

의존성(`pyproject.toml`, `uv.lock`) 변경은 관리자에게 요청하거나 별도 PR로 한다.
