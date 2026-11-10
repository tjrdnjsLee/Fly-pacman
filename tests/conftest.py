"""CI 보조: FLYPAC_REQUIRE_TORCH가 켜져 있으면 torch가 없을 때 시험을 건너뛰지 않고 실패시킨다.

PPO 경로 시험은 torch가 없으면 건너뛴다 (importorskip). CI에서는 그 건너뜀이 조용히 일어나면 안 된다.
"""

import os


def pytest_sessionstart(session):
    if os.environ.get("FLYPAC_REQUIRE_TORCH"):
        import torch  # noqa: F401  (없으면 ImportError로 시험 전체가 실패한다)
