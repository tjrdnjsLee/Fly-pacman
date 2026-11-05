# pacman-ai (upstream 보관본)

- 출처: https://github.com/vikranthreddimasu/pacman-ai
- 커밋: 33612801815026f4ef4bb6a709009e20a929a83d
- 복사한 것: 원 저장소의 `pacman/engine/` 전체를 **수정 없이** `engine/`으로 복사 (TRD 3.1 밖, 총책임자 승인).
- 용도: M1에서 팩맨 담당이 이걸 바탕으로 `src/flypac/engine/`을 정비한다. 이 폴더는 직접 수정하지 않는다.
- pytest는 `testpaths = ["tests"]`라 이 폴더를 수집하지 않는다.

## 라이선스 상태

고지 문구 (`NOTICE`와 같다):

> pacman-ai by vikranthreddimasu (https://github.com/vikranthreddimasu/pacman-ai),
> MIT license as stated in its README; no LICENSE file at commit 33612801815026f4ef4bb6a709009e20a929a83d.

원 저장소의 README는 라이선스를 "MIT"로만 적고, LICENSE 파일도 저작권자 표기도 없다 (복사 시점 확인).
그래서 원 LICENSE를 이 폴더에 둘 수 없고, 저작권 줄도 지어내지 않았다.
