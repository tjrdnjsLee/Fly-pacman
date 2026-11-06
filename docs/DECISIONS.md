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

## 10. 회로 추출 · 해부 자료 · 채널 배정 (M1 뇌 ①)

미결 2 · 20의 최종 결정이다 (총책임자, 2026-11-06). 관련 파일: `scripts/extract_circuit.py`, `scripts/export_anatomy.py`, `scripts/assign_channels.py`, `src/flypac/brain/assign.py`, `configs/circuit_extract_v1.yaml`, `configs/circuit_assign_v1.yaml`.

### 10.1 미결 2: PN 채널 · MBON 묶음 배정 (해결)

- 배정은 프로젝트가 부여한 대응이다. 생물학적 근거를 주장하지 않는다.
- 학습 전 방향 편향은 **기준선 빼기**(10.3)로 구조적으로 없앤다. 배정은 편향을 최적화하지 않고 **표현력**만 시험한다.
- 규칙 1 (선택): 주석으로 정의하고 연결로 선별한다. 한 번 통과, 반복 제거 없음 (10.6).
- 규칙 2 (MBON 4묶음, 시드와 무관하게 한 번): $w^{(0)}_{ij} = W_{0,ij} / \langle W_0\rangle_{W_0>0}$, $T_{a\tau} = \frac{1}{|a|}\sum_{j\in a}\sum_{i\in\tau} w^{(0)}_{ij}$. 묶음 좌 · 직진 · 우 · 반전, 크기 차이 ≤ 1. 각 KC 종류 $\tau$에서 $T_{a\tau}$의 네 묶음 간 상대 편차의 최댓값을 최소화한다. 고정 시드 탐욕 배정 → 맞바꾸기 국소 탐색, 탐색 횟수는 설정에 고정.
- 규칙 3 (PN 9묶음, 시드마다): 채널 순서 냄새 좌 · 앞 · 우 · 뒤, $z$, 페로몬 좌 · 앞 · 우 · 뒤. 단위는 사구체(같은 사구체의 PN은 같은 채널). 분기 P(기본): DA1, VA1v, VA1d, DL3를 페로몬 4채널에 하나씩, 방향은 시드로 무작위, 나머지 사구체를 냄새 4 + $z$에 무작위 분할. 분기 G(대안): 모든 사구체를 9채널에 무작위 분할. 사구체 수는 채널 간 최대 1 차이. 시드는 0 ~ 19로 고정하고 결과를 본 뒤 바꾸지 않는다. $J_{ic} = \sum_{p\in c} J^{\text{pn}}_{ip}$.
- 규칙 4 (채널 이득): $S_c = \sum_i J_{ic}$, $\alpha_c = \bar S / S_c$. $S_c = 0$이면 그 시드는 실패로 기록한다 (예외로 멈추지 않음).
- 규칙 5 (분기 P 전환, 사전 확정): (1) 후보 사구체 중 하나라도 규칙 1을 통과하지 못함, (2) 네 페로몬 채널의 도달 KC 수 max/min > 1.5, (3) 다음 지시서의 표현력 시험에서 분기 P로 시드 0 ~ 19를 모두 검사한 뒤 통과 시드가 없음. 하나라도 참이면 분기 G.

### 10.2 미결 20: 해부 투영 (해결)

- 시점: **정면** (앞뒤 축을 버림). 이유: 수직엽 · 내측엽이 모두 갈라져 보이는 유일한 시점.
- 영역: 칼릭스, 꽃자루, 엽 구획 15개, 외측뿔. 외측뿔은 **모델 항(본능 $g\,o_a$)이며 뉴런이 아님**을 화면에 표시한다 (`region_model_term`).
- 더듬이엽은 그리지 않는다 (hemibrain에 일부만 있음).
- 대표 좌표: 각 뉴런의 해당 영역 안 **입력 시냅스 위치 중심** (PN은 시냅스 전 위치, 10.8).

### 10.3 기준선 빼기 (**TRD v0.6에 반영할 것**, 구현은 다음 지시서)

$$
m_a = \frac{1}{k\,|a|}\sum_{j\in a}\sum_i \big(w_{ij}-w^{(0)}_{ij}\big)\,k_i
$$

- TRD 10.4의 흔적 · 갱신식은 그대로다 ($\partial x_a/\partial w_{ij} = h_i/(k|a|)$ 유지).
- $w^{(0)}$은 회로 파일에서 읽는 **불변 참조 배열**이다. 네 초파리가 읽기 전용으로 공유한다.
- 체크포인트에 회로 파일 해시를 저장하고, 재개할 때 대조한다.
- 설명 문구: "인위적으로 부여한 방향 대응에서 나오는 초기 정책 기여를 연구 대상에서 제외한다. 연결 구조가 표현과 학습에 미치는 영향은 유지한다." 생물학 설명은 "인공 판독 규칙".

### 10.4 결정 ② 수정: B0′ (**TRD v0.6에 반영할 것**)

- 기준선 빼기로 B0′와 B0는 같은 정책이다. **본 실험에서 B0′를 뺀다. 본 실험은 5조건** (체제 1 · 2 · 3, B0, B1).
- 대신 회귀 테스트로 유지한다: 같은 관측 시퀀스에서 학습 전 뇌와 B0의 $\pi$가 부동소수점 오차 안에서 같다.
- 설정의 `b0p`는 다음 지시서에서 정리한다 (이번 작업에서 건드리지 않음).

### 10.5 채널 이득 기호

채널 이득은 $\alpha_c$다. 본능 이득 $g$와 다른 기호다. $\alpha_c$는 **PN 입력 사본에만** 곱한다.

### 10.6 문턱 · 제외 목록 (총책임자 확정 2026-11-06)

단계 1 목록 조사 보고를 보고, 학습 결과를 보기 전에 데이터 품질만으로 정했다. 값은 `configs/circuit_extract_v1.yaml`에 있다.

| 값 | 결정 | 이유 |
|---|---|---|
| $s_{\min}$ | 3 (연결당 최소 시냅스 수, **맨 먼저** 적용) | 약한 연결은 재구성 오류 가능성이 있고, 남겨도 약화 여유가 작다 |
| $\theta_{\text{PN}}$ | 50 (본 칼릭스 → 후보 KC 시냅스 합) | 실제 연결(대개 11~20 시냅스) 약 3개 이상 |
| $\theta_{\text{KC}}$ | 0.8 (아래 정의 A) | 남는 KC마다 실제 PN 구동의 80% 이상을 모델이 재현 |
| KC 최소 입력 (**새 기준**) | 선택 PN에서 받는 본 칼릭스 시냅스 ≥ 20 | 비율만 높고 입력이 거의 없는 KC를 거름 |
| $\theta_{\text{MBON}}$ | 100 (선택 KC에서 받는 시냅스, 꽃자루 + 15구획) | 조각난 재구성을 거르고 작지만 진짜인 MBON은 남김 |

- **적용 순서 (총책임자 결정 2026-11-06, PR #5 검토):** $s_{\min}$으로 연결 단위 정리를 맨 먼저 하고, $\theta_{\text{PN}}$ · $\theta_{\text{KC}}$ · KC 최소 입력 · $\theta_{\text{MBON}}$은 정리된 시냅스 수로 계산한다. 지시서 4.1이 $s_{\min}$을 6번(마지막)에 적은 순서가 모호했던 것을 바로잡는다. 정리는 후보 전체에 적용한다 (PN 계열 → 후보 KC는 본 칼릭스, 후보 KC → 후보 MBON은 꽃자루 + 15구획 합). 그래서 $\theta_{\text{KC}}$의 분모도 정리된 값이다.

- **제외:** vPN 계열, `Z_`, VP 온습도, 다사구체(VP 포함 다사구체 포함). **포함:** 단일 사구체 V, D.
- **lvPN 계열은 포함하며 흥분성으로 가정한다 (서버 미확인).** hemibrain v1.2.1 서버에는 전달물질 주석이 없어 확인하지 않았다. 지시서 4.1의 "PN은 흥분성"($J \ge 0$) 가정을 따른다.
- KC는 종류 이름으로 빼지 않는다. 문턱이 거른다. 결과: KCg-d 99개 중 5개, KCab-p 60개 중 0개, KCg-s1 ~ s4 각 1개 중 0개가 남았다.
- MBON은 instance의 좌우와 관계없이 $\theta_{\text{MBON}}$으로만 거른다. `MBON15-like`, `MBON17-like`를 포함한다.
- 정의 A (KC 비율): 분자 = 본 칼릭스에서 **선택 PN**에게서 받는 (정리된) 시냅스. 분모 = 본 칼릭스에서 **이름에 PN이 들어간 모든 뉴런**에게서 받는 (정리된) 시냅스 (배제한 온습도 · 다사구체 · vPN 포함, APL · DPM · KC · 주석 불명 제외). WEDPN · LPN은 본 칼릭스 → KC 시냅스가 0이라 분모에 넣어도 값이 같다.
- 결과 (`data/circuit/raw/mb_raw_v1.meta.json`): PN 461 → 후보 145 → 106, KC 1,927 → 1,575, MBON 71 → 47. 사구체는 주석 후보 51 (θ_PN 적용 전) → 원본 51 (θ_PN 적용 후). θ_PN에서 탈락한 39개 PN은 모두 같은 사구체의 다른 PN이 남아 사구체 수는 줄지 않았다.

### 10.7 페로몬: 분기 G로 전환 (규칙 5의 조건 2)

- 단계 1 보고에서 도달 KC 수 max/min ≈ 6.7이었다. 확정 문턱의 원본으로 다시 계산한 값은 DA1 240, VA1d 116, VA1v 87, DL3 35, **max/min = 6.86 > 1.5**다. 조건 1은 발동하지 않았다 (네 사구체 모두 원본에 있음).
- 규칙 5대로 **분기 G**로 간다. 결과를 본 뒤 규칙을 바꾸지 않는다. `configs/circuit_assign_v1.yaml`은 `branch: G`로 고정하고 판정 근거를 주석에 적었다. 보고서는 조건 1 · 2를 계속 다시 계산해 기록한다. $s_{\min}$을 맨 먼저 적용한 뒤에도 PN과 도달 KC 수는 같다 (6.86).

### 10.8 구현에서 정한 해석 (지시서 · TRD 문구에 없거나 다르게 한 곳)

- **ROI 이름** (hemibrain v1.2.1 서버 확인): 본 칼릭스 `CA(R)`, 부속 칼릭스는 `dACA(R)`, `lACA(R)`, `vACA(R)` 세 개(`MB(R)` 밖), 꽃자루 `PED(R)`, 엽 구획 `a1(R)` ~ `g5(R)` 15개(non-primary라 `include_nonprimary=True`로 조회), 외측뿔 `LH(R)`. 15구획 합은 5엽 합과 같다.
- **PN 분류 규칙:** `WEDPN*`, `LPN`은 더듬이엽 밖 PN이라 단일 사구체 후각 PN 후보가 아니다 (제외 목록이 아니라 후보 정의로 빠짐, 칼릭스 입력도 0). 다사구체 = `M_` · `MZ_` 접두 또는 사구체 이름에 `+`. 계열 = `_` 뒤의 `…PN`. vPN 제외는 계열 이름이 정확히 `vPN`인 것만이다 (`lvPN`은 흥분성 가정으로 포함, 서버 미확인 — 10.6). 사구체 = 종류 이름의 첫 `_` 앞.
- **$s_{\min}$은 맨 먼저 적용한다** (10.6). PR #5의 첫 판은 4.1의 문구대로 마지막에 적용해 KC 1,579, MBON 50이었다. 순서를 바꾼 뒤 KCab-c 5개가 빠지고 KC 1개가 새로 들어왔으며, MBON 3개(MBON10 613719036, MBON11 517518166, MBON34 1173814525)가 빠졌다. PN은 그대로다.
- **`mbon_class_profile`**: 선택 KC 기준 (W0의 열을 KC 종류별로 더한 비율). **`mbon_class_profile_candidates`**: 모든 후보 KC에서 받는 시냅스(꽃자루 + 15구획, $s_{\min}$ 정리 뒤) 기준. 열 순서 `gamma`, `apbp`, `ab` (`kc_class`의 값과 같음).
- **파일 이름:** 원본은 `data/circuit/raw/mb_raw_v1.npz`다. TRD 10.1의 `mb_v1.npz`(최종 회로)는 다음 지시서에서 표현력 시험을 통과한 시드로 만든다.
- **KC → MBON 계수는 꽃자루 + 15구획만**이다. 칼릭스 MBON(`MBON22`)의 칼릭스 입력은 들어가지 않는다.
- **정면 투영 축:** hemibrain 좌표에서 `y`가 앞뒤, `z`가 등배(작을수록 등쪽), `x`가 좌우다. 투영 $(u, v) = (+x, -z) \times 0.008\,\mu\text{m}$ (복셀 8 nm, 서버 Meta와 대조). 앞에서 본 모습이라 오른쪽 버섯체의 정중선 쪽이 $+u$다. 가설을 설정에 적고 확인 시험으로 검증했다: (가) α엽 끝 > 뿌리, (나) β엽 끝이 뿌리보다 정중선(EB 중심)에 가까움, (다) 칼릭스 > 뿌리, (라) 칼릭스–뿌리 차이가 가장 큰 축 = 버린 축. 모두 통과.
- **기준점:** 뿌리 = 꽃자루 메시 중 칼릭스 중심에서 가장 먼 5% 꼭짓점의 평균 (엽이 갈라지는 쪽 끝). α · β엽 끝 = 각 엽 메시 중 뿌리에서 가장 먼 5%의 평균. 정중선 = EB 메시 중심.
- **영역 다각형:** shapely 대신 새 의존성 없이 만든다. 삼각형을 0.5 μm 격자에 칠하고 3×3 닫힘 → 경계 추적 → Douglas–Peucker(허용 오차 0.5 μm), 4 μm² 미만 고리는 버린다. 칼릭스 영역은 본 칼릭스 `CA(R)`만이다 (부속 칼릭스는 그리지 않음).
- **DAN:** 이름 규칙(`PAM*`, `PPL1*`)의 317개 전부 (instance 좌우 무관). 좌표 = 15구획 안 시냅스 전 위치 중심, 구획 = 시냅스 전 위치가 가장 많은 구획. 15구획에 시냅스 전 위치가 없는 2개는 좌표 없음(이유 기록).
- **MBON 묶음 탐욕 배정:** 시드로 섞은 순서대로, 자리가 남은 묶음 중 목적값(최종 묶음 크기로 나눈 평균 기준)이 가장 작은 곳에 넣는다. 국소 탐색은 매 회 가장 좋은 쌍 맞바꾸기 하나를 받아들인다. 목적값 1.029 → 0.068 (맞바꾸기 23회).
- **채널 무작위 분할:** 어느 채널이 사구체를 하나 더 받는지도 시드로 무작위다 (앞 채널에 몰리지 않음). 난수는 `np.random.default_rng(seed)`.
- **접합 KC 비율:** 냄새 채널 하나 이상과 $z$ 채널에서 모두 입력을 받는 KC의 비율 (`J_ic > 0` 기준).
