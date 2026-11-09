# 재생 로그 렌더러 v1

저장소 루트에서 실행한다. 첫 명령은 진짜 엔진과 B1으로 시드 하나의 첫 판을
기록하고 로그 경로를 출력한다. 두 번째 명령은 그 로그에서 같은 3초 구간을
게임 화면과 날것 화면으로 각각 저장한다.

```sh
uv run python -m viz.record_demo
uv run --extra viz python -m viz <replay.npz> --start 8 --duration 3
```

영상은 1920×1080, 30 fps, H.264 MP4이며 기본 저장 위치는 로그와 같은
폴더다. `--out-dir`로 출력 폴더를 지정할 수 있고, `--view game` 또는
`--view raw`로 화면 하나를 고를 수 있다. `--no-observations`는 날것 화면의
감각 막대와 결정 표시를 숨긴다. 예시 구간은 고정 시작 시각으로 자른다.

형식 2·3·4의 로그를 읽는다. 형식 3부터는 `committed`로 결정 확정을 표시하고
`moved`는 해당 틱의 칸 이동 여부로 표시한다. 형식 2에서는 기록된 행동을
표시한다. 로그에 시간과 냄새 스케일 설정이 없으므로, 예시와 다른 설정으로
기록했다면 `--tick-rate`에 원래의 `time.v_max_tiles_per_s`를,
`--c0`에 원래의 `odor.c0`를 넘긴다. 원래 값은 실행 폴더의 `config.yaml`에 있다.

가짜 로그도 같은 렌더링 명령에 넘길 수 있다.

```sh
uv run python -c 'from pathlib import Path; from flypac.replay.synth import write_synthetic; print(write_synthetic(Path("runs"), phero=True))'
uv run --extra viz python -m viz runs/synthetic_s0_g1.npz --duration 2
```

## 뇌 화면과 나란히 보기

```sh
uv run python -m viz.record_demo --flybrain
uv run --extra viz python -m viz <replay.npz> --view brain-both --fly 1 --start 8 --duration 3
```

기록 스크립트는 진짜 엔진에서 체제 1 FlyBrain을 초기 가중치로 시작해 첫 판의
형식 4 로그를 만든다. 한 화면만 만들려면 `--view brain` 또는
`--view side-by-side`를 쓴다. 나란히 보기에서는 게임과 뇌가 같은 재생 틱을
읽고, 선택한 초파리에 고리를 그린다.

기록 때 기본 설정과 다른 회로 배정이나 본능 이득을 썼다면 렌더링할 때도
같은 `--brain-config`를 넘긴다. 뇌 화면은 정면 해부 투영 위에 기록된 모델
활동을 표시한다. 외측뿔 윤곽의 세기는 방향별 `g × 냄새 입력`의 최댓값이다.
KC의 잔상은 표시용이다. DAN은 하나의 기록 신호를 대응 구획 전체에 같은
부호로 표시한다.

가짜 뇌 로그는 생성기에 해부 자료와 맞는 회로 크기를 넘긴다.

```sh
uv run python -c 'from pathlib import Path; from flypac.replay.synth import write_synthetic; print(write_synthetic(Path("runs"), brain=True, n_kc=1575, n_mbon=47))'
uv run --extra viz python -m viz runs/synthetic_s0_g1.npz --view brain-both --duration 2
```
