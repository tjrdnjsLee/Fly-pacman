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

형식 2와 3의 로그를 읽는다. 형식 3에서는 `committed`로 결정 확정을 표시하고
`moved`는 해당 틱의 칸 이동 여부로 표시한다. 형식 2에서는 기록된 행동을
표시한다. 로그에 시간과 냄새 스케일 설정이 없으므로, 예시와 다른 설정으로
기록했다면 `--tick-rate`에 원래의 `time.v_max_tiles_per_s`를,
`--c0`에 원래의 `odor.c0`를 넘긴다. 원래 값은 실행 폴더의 `config.yaml`에 있다.

가짜 로그도 같은 렌더링 명령에 넘길 수 있다.

```sh
uv run python -c 'from pathlib import Path; from flypac.replay.synth import write_synthetic; print(write_synthetic(Path("runs"), phero=True))'
uv run --extra viz python -m viz runs/synthetic_s0_g1.npz --duration 2
```
