"""Host-side entrypoint for the challenge submission.

This is only a development scaffold so the submission directory already matches
the expected grader layout. Replace the TODO sections with real SDK runtime
logic before running the grader.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Compiled output directory")
    parser.add_argument("--case", required=True, help="Reference test case name")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from reference import ALL_CASES  # type: ignore

    cases = {maker().__getitem__("name"): maker() for maker in ALL_CASES}

    if args.case not in cases:
        valid = ", ".join(sorted(cases))
        raise SystemExit(f"Unknown case '{args.case}'. Valid cases: {valid}")

    case = cases[args.case]
    print(
        f"TODO: implement host runtime for case '{args.case}' "
        f"(N={case['D'].shape[0]}, d={case['D'].shape[1]}, K={case['K']}, P={case['P']}) "
        f"using build output '{args.name}'."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
