# 인터페이스 보충 결정 (DECISIONS)

TRD 5절이 모양을 정하지 않은 곳을 M0 인터페이스 자리 단계에서 정한 기록이다. **TRD 다음으로 이 문서를 따른다.** TRD와 충돌하면 TRD가 우선이다. 바꾸려면 관리자에게 제안한다.

## 0. 공통 규칙

- **모든 mask는 참 = 이 게임(초파리)에 적용.** `reset`, `step`, `on_release`, `on_eaten`, `flush_learning`, `alive` 등 어떤 mask든 참인 칸만 처리하고 거짓인 칸은 건드리지 않는다. 모양 `(B,)`는 게임, `(B,F)`는 초파리 단위다.

## 1. 사건 (core/events.py)

- 초파리 번호 `fly`: `CATCH`, `EATEN`, `Z_CHANGE`, `ALL_RELEASED`는 `fly >= 0`. 그 밖의 종류는 `fly == -1`. 어기면 생성 시 `ValueError`.
- 이유: `fly = -1`이 numpy 인덱스로 들어가면 오류 없이 마지막 초파리를 가리킨다. 체제는 `R[ev.fly]` 대신 `R[fly_index(ev, F)]`(`regimes/base.py`)를 쓴다. 이 함수는 `0 <= fly < F`를 검사한다.
- `cause` 사용법:
  - `GAME_OVER`: 승자. `FLIES`, `PACMAN`, `CENSORED` 중 하나.
  - `Z_CHANGE`: 원인. `POWER_START`, `POWER_END`, `REVIVE` 중 하나.
  - 그 밖의 종류: 반드시 `None`.
- `active`: 사건을 만드는 순간의 학습 활성 여부 (TRD 4.5).

## 2. 냄새 장 (field/odor.py)

- `Field(map, D, lam)`의 시그니처는 TRD 그대로다. 게임 수 `B`는 **첫 `reset(mask)`의 mask 길이**로 정해진다.
- `reset` 전에 `step`, `c`, `total`을 부르면 `RuntimeError`. 이후 mask 길이가 처음 `B`와 다르면 `ValueError`. `S`의 모양이 `(B,N)`이 아니어도 `ValueError`.
- `step(S, mask)`의 mask는 `(B,)` bool이고 **참인 게임만 갱신**한다 (거짓 = 동결).


## 2-1. 감각 (senses/sense.py)

- `sense(odor, phero, views, info, c0, *, nbr)`: TRD 5.4의 표가 이웃표 `nbr`(N,4)의 출처를 정하지 않아, 키워드 전용 인자 `nbr`을 더했다. 위치 인자 다섯 개는 TRD 그대로다. 실행기는 `nbr=map.nbr`를 넘긴다. `legal`, `decide`는 `info`(엔진의 `decision_info()`)를 그대로 쓴다.

## 2-2. 맵 (maps/original.py)

- 미로 원본은 `maps/_maze_data.py`(pacman-ai의 `maze_data.py`에 머리말만 붙여 그대로 복사). 칸 번호는 행 우선 순서 0..319.
- 열은 감싸서 터널을 이웃표에 연결한다 (14행 0열 ↔ 27열).
- `fly_starts`: 0번은 집 밖(원본 Blinky 자리), 1~3번은 집 안.
- `move_mask["pacman"]`은 유령 집·문으로 들어가는 이동을 막고, `move_mask["fly"]`는 `nbr >= 0` 전부.

## 3. 엔진 (engine/game.py)

- `Game`은 한 판 단위, `VecGame(B, map, cfg, rngs)`는 `Game` B개를 도는 얇은 묶음이다. 규칙은 `Game`에만 있다 (TRD 원칙 1).
- `VecGame` 메서드와 반환 모양 (게임 인덱스 `b = 0..B-1`):

| 메서드 | 입력 | 반환 |
|---|---|---|
| `step_pacman(actions)` | `(B,)` 절대 방향 | `list[list[Event]]` (게임별) |
| `step_flies(actions)` | `(B,F)` 자기중심 행동, -1 = 유일한 경로 | `(events: list[list[Event]], executed (B,F) bool, active (B,F) bool)` |
| `tick_timers()` | | `list[list[Event]]` |
| `views()` | | `list[GameView]` (길이 B) |
| `decision_info()` | | `(legal (B,F,4) bool, decide (B,F) bool)` |
| `reset_life(mask)`, `reset_game(mask)` | `(B,)` bool | 없음. mask가 참인 게임만 |
| `retire(mask)` | `(B,)` bool | 없음. mask가 참인 게임의 `learning_active`만 거짓으로 내린다. 새 판은 시작하지 않고 동결은 유지한다. **TRD 5.2에 없는 메서드다. 총책임자 결정 — TRD v0.6에 반영할 것** |

- 자동 초기화 없음. 초기화는 실행기만 부른다.
- `retire`는 모든 판을 끝낸 칸을 은퇴시킬 때 실행기가 부른다 (7절). 새 판이 시작되지 않는 칸은 `reset_game`을 거치지 않으므로, 학습 활성이 종료 처리 뒤에도 켜진 채 남지 않게 하려는 것이다.
- `Game`의 같은 이름 메서드는 한 판 모양이다 (`step_flies`는 `(F,)` 입력, `decision_info`는 `(F,4)`, `(F,)` 반환, `view()`).

## 4. 유령 조종기 (brain/base.py)

초파리 뇌, B0, B1, 무작위 초파리가 같은 모양이다. `B` 게임 수, `F` 초파리 수.

| 메서드 | mask / 입력 모양 |
|---|---|
| `propose(obs, alive, views=None)` | `alive` `(B,)` bool, 반환 `(B,F)` int (동결 게임은 -1) |
| `commit(executed, active)` | `(B,F)` bool 둘 다 |
| `give_reward(R)` | `(B,F)` |
| `on_release(mask)` | **`(B,)` bool, 게임 단위.** 학습 활성은 게임 단위로 시작한다 (TRD 4.5). `ALL_RELEASED`의 fly는 기록용이지 초기화 대상이 아니다. 실행기는 `ALL_RELEASED`가 난 게임 전체에 부른다 |
| `on_eaten(mask)` | `(B,F)` bool |
| `flush_learning(mask)` | `(B,)` bool |
| `reset_life(mask)` | `(B,)` bool |

- **6.1의 8번 호출 순서는 `on_release` → `commit` → `on_eaten`이다.** TRD 6.1의 8번은 `commit` 다음에 `on_release`를 적었다. 그 순서면 같은 틱에서 `ALL_RELEASED` 뒤에 실행된 활성 결정의 흔적이 `on_release`(e = 0)로 지워져 TRD 4.5의 의도와 모순된다. `commit`은 결정별 `active`로 학습을 거르므로 순서를 바꿔도 비활성 결정은 흔적에 들어가지 않는다. **총책임자 결정 — TRD v0.6에 반영할 것.**

## 5. 기록기 (recording/recorder.py)

TRD 5.8이 `...`로 비워 둔 인자:

- `Recorder(run_dir, cfg)`
- `on_events(game, events, view)`
- `on_life_end(game, view, events)`, `on_game_end(game, view, events)`: `view`는 초기화 전에 **다시 읽은** 종료 상태 (TRD 6.2). `events`는 그 종료를 일으킨 이번 틱 사건 (`LIFE_LOST`, `GAME_OVER`가 들어 있다).
- `replay_frame(game, views, obs, odor, phero, brain)`: `odor`, `phero`는 `(B,N)`, `brain`은 사전 또는 `None`.
- `replay_end(game, keep)`

`game`은 묶음 안의 게임 칸 번호(0..B-1)다.

## 6. 재생 로그 (replay/schema.py)

- 사건은 세 배열로 저장한다: `ev_tick` `(E,)` int32, `ev_kind` `(E,)` int8, `ev_fly` `(E,)` int8 (초파리 없음 = -1). 사건의 `cause`는 저장하지 않는다 (TRD 19.4는 틱 · 종류 · 초파리만 정함).
- `ev_kind` 정수 코드표 (`schema.KIND_CODES`). **번호를 바꾸거나 지우지 않고, 새 종류는 끝에만 추가한다.**

| 코드 | 종류 | 코드 | 종류 |
|---|---|---|---|
| 0 | CATCH | 6 | GAME_OVER |
| 1 | EATEN | 7 | POWER_START |
| 2 | DOT | 8 | POWER_END |
| 3 | ENERGIZER | 9 | Z_CHANGE |
| 4 | FRUIT | 10 | ALL_RELEASED |
| 5 | LIFE_LOST | | |

- **형식 버전이 2인 이유:** 버전 1은 19.4의 기본 필드, 버전 2는 19.13의 뇌 필드 묶음(`kc_active`, `pn_input`, `mbon_out`, `dan`)을 더한 형식이다. TRD 19.13이 "형식 버전 올림"이라고 했다. 뇌 필드는 전부 있거나 전부 없어야 하므로 뇌 필드가 없는 로그도 버전 2다.

## 7. 실행기와 가짜 엔진 (M0 4단계)

- `Game.index`: 묶음 안의 칸 번호. `VecGame`이 생성 직후 정해 주고, 사건의 `game` 값으로 쓴다.
- `GhostController.uses_views`: 조종기가 `views`를 쓰는지 알리는 표시다. 기본값은 거짓. **`views`를 넘기는 기준은 조건이다: 실행기는 `condition == "b1"`일 때만 넘긴다** (TRD 5.5, 원칙 2). 실행 시작 때 `uses_views != (condition == "b1")`이면 `ValueError`를 낸다. 그래서 `uses_views`만으로 초파리 조건에서 `views`를 받을 수 없다.
- `ALL_RELEASED` 사건 자체의 `active`는 거짓이다 (학습 활성이 켜지기 직전에 만들어진다). 그 뒤 같은 틱에 생긴 사건은 참.
- 판 번호는 1부터. 칸마다 `n_games`판을 한다. 모든 판을 끝낸 칸은 은퇴시킨다: 새 판은 시작하지 않고 `retire`로 `learning_active`만 거짓으로 내리며 동결은 유지한다. **총책임자 결정 — TRD v0.6에 반영할 것** (TRD 6.2의 4번 초기화와 달리 은퇴하는 칸은 판 초기화를 하지 않는다).
- 실행기의 `replay_frame`에는 4번(팩맨 이동 뒤)에 만든 `views`와 5번의 `obs`, 그리고 **5번(관측을 만든 순간)의 냄새 장 · 페로몬 장 복사본**을 넘긴다. 한 프레임의 `views`, `obs`, 냄새 장이 모두 같은 순간이다. 10번의 장 갱신은 이 프레임에 들어가지 않는다. **총책임자 결정 — TRD v0.6에 반영할 것.** 종료 상태는 13번에서 `view`를 다시 읽어 `on_life_end`, `on_game_end`에 넘긴다.
- 조건 `b0`, `b0p`, `b1`은 `NullRegime`(`regimes/null.py`)을 쓴다: 보상 0, 페로몬 방출 0. 이유: 기준선은 학습하지 않으므로 보상이 없다. 설정에 `reward.p`, `reward.q`가 없어도 만들어진다. 알 수 없는 조건 이름은 `ValueError`다 (이전에는 조용히 `Regime1`이 되었다).
- 사건의 틱 번호는 사건이 생긴 실행기 틱의 번호다. 감시 장치 `GAME_OVER`(중단)도 같은 틱의 다른 사건과 같은 번호를 갖는다 (타이머 단계가 틱 번호를 올리기 전에 만든다).
- 가짜 엔진의 `release_ticks`는 순서를 검사하지 않는다. 마지막으로 나오는 초파리가 3번이라는 보장이 없고, `ALL_RELEASED`의 초파리 번호는 마지막으로 나온 초파리의 번호다.
- 가짜 엔진(`engine/fake.py`)은 파워 쿠키 효과, 과일, 엘로이, 터널 감속을 생략한다. 따라서 `z`는 항상 0이고 `EATEN`, `Z_CHANGE`, `POWER_*`는 생기지 않는다 (M1에서 채운다). 불가능한 행동을 받으면 `ValueError`. 행동 -1은 유일한 경로, 갈림길이면 직진을 우선한다.

## 8. 기록과 재생 로그 (M0 5단계)

- `GameView`에 기본값이 있는 필드 `pac_acc: float = 0.0`, `fly_acc: (F,) | None = None`을 끝에 더했다. 재생 로그의 보간(TRD 19.6)이 속도 누적값을 쓰기 때문이다. TRD 4.6의 기존 필드는 그대로다.
- `Recorder.replay_frame(game, views, obs, odor, phero, brain)`의 `brain`은 사전이다: `action (B,F)`, `pi (B,F,4)`는 항상, `replay.brain`이 켜지면 `kc_active`, `pn_input`, `mbon_out`, `dan`도 (모두 앞 축이 B). 실행기가 `{"action": 6번의 반환값, "pi": 조종기.last_pi}`를 만들고, 조종기에 `replay_info()`가 있으면 그 사전을 합친다 (뇌 담당이 뇌 필드를 여기서 낸다). `replay.brain`이 켜졌는데 필드가 없으면 `ValueError`.
- 재생 로그의 `T`는 실행기의 틱 수와 같고, `ev_tick`은 엔진 틱이라 프레임 번호와 같다. 목숨이 바뀌어도 한 판의 프레임은 이어진다.
- `replay_end(game, keep)`는 판 번호를 올리는 곳이기도 하다 (파일 이름의 `g<판>`은 올리기 전 번호). `on_game_end` → `replay_end` 순서로 부른다.
- CSV 열은 TRD 13.3의 한글 이름을 그대로 쓴다. `utf-8-sig`로 쓰고 읽는다. 중단(`CENSORED`)과 미포획은 "포획까지 틱", "잡은 초파리"를 빈칸으로 둔다.
- 해석을 정한 열 (TRD가 정의를 주지 않음):
  - `누적 결정 수`: 결정을 요청받은 횟수 (`obs.decide`가 참이고 게임이 동결이 아닌 초파리 틱). 도중 종료로 실행되지 않은 제안도 센다.
  - `결정에서 보상까지 지연(결정 수)`: 보상 사건마다 "그 초파리의 직전 보상 사건(또는 목숨 시작) 이후 결정 횟수"의 평균. 보상 사건이 없으면 빈칸.
  - `상태 전환 시 반전 기회`: `Z_CHANGE`(원인 파워 시작·종료)의 횟수. `반전 횟수`는 빈칸 (TODO(M1): 파워 쿠키가 생긴 뒤 센다).
- 남길 판은 `replay.keep_games` (고정 판 번호 목록)뿐이다. TODO(미결 17): 협공 판정 거리 `d`, 사건 기준, 저장 용량 상한(`replay.max_bytes`는 임시 구현: 넘으면 그 판을 버린다).
- 가짜 로그 생성기(`replay/synth.py`)의 회로 크기(`PLACEHOLDER_N_KC` 등)는 자리표시값이다 (TODO(미결 2, 3)).

## 9. 문서 공개 범위

- 2026-11-06 PRD·TRD 공개 전환. 관리자가 PR #4로 팀 공유를 위해 올림. 작업 지시서(`*_TASK.md`)는 계속 비공개.
