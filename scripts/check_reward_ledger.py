"""보상 장부 검사: uv run python scripts/check_reward_ledger.py <결과 폴더> [...]  (어긋난 판이 있으면 종료 코드 1)"""

from __future__ import annotations

import sys

from flypac.recording.ledger import check_run


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("사용법: check_reward_ledger.py <결과 폴더> [...]", file=sys.stderr)
        return 2
    failed = False
    for d in argv[1:]:
        n, bad = check_run(d)
        print(f"{d}: {n}판 검사, 어긋난 판 {len(bad)}")
        for line in bad:
            print("  ", line)
        failed |= bool(bad)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
