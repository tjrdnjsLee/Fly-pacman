"""초 → 틱 변환 (TRD 7.1). 코드 전체에서 이 함수 하나만 쓴다 (관리자 승인, DECISIONS 13.3절).

게임 규칙의 초 단위 값은 설정 파일에 초로 적고(예: `engine.power_s`), 틱으로 바꾸는 곳은 여기뿐이다.
진짜 엔진도 겁먹는 시간을 이 함수로 바꿔야 관측의 파워 분모(pacman/obs.py)와 엔진의 실제 지속 시간이 같다.
"""

from __future__ import annotations


def seconds_to_ticks(seconds: float, v_max_tiles_per_s: float) -> int:
    """round(초 × v_max). v_max = 설정의 `time.v_max_tiles_per_s` (칸/초, 1틱 = 1/v_max 초). 파이썬 round."""
    return int(round(float(seconds) * float(v_max_tiles_per_s)))


def config_ticks(cfg, seconds: float) -> int:
    """설정 cfg의 time.v_max_tiles_per_s로 초를 틱으로 바꾼다."""
    return seconds_to_ticks(seconds, cfg.time.v_max_tiles_per_s)
