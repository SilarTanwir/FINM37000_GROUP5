"""Stage runner used by dodo.py (`python -m esvrp.stages <n>`).

Implements PLANNING.md §6 (pipeline stages 1-8). Stages are added here as their modules are implemented.
"""

from __future__ import annotations

import logging
import sys

from esvrp.config import load_config


def run_stage(n: int) -> None:
    cfg = load_config()
    if n == 1:
        from esvrp.data.universe import run

        run(cfg)
    elif n == 2:
        from esvrp.data.quotes import run

        run(cfg)
    else:
        raise NotImplementedError(f"stage {n} is not implemented yet")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run_stage(int(sys.argv[1]))
