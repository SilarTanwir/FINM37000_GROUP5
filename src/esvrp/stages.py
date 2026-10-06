"""Stage runner used by dodo.py (`python -m esvrp.stages <n>`).

Implements PLANNING.md §6 (pipeline stages 1-8). Stub until the stage modules are implemented.
"""

from __future__ import annotations

import sys


def run_stage(n: int) -> None:
    raise NotImplementedError(f"stage {n} is not implemented yet")


if __name__ == "__main__":
    run_stage(int(sys.argv[1]))
