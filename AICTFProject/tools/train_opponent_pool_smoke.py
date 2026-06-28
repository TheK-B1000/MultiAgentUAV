#!/usr/bin/env python3
"""Smoke-test that elite hardpool config includes and samples OP11/OP12."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    import numpy as np

    from rl.config_presets import v6i8_adapter_balanced_elite_hardpool_config
    from rl.custom_ppo.curriculum_runtime import TrainingOpponentPool

    cfg = v6i8_adapter_balanced_elite_hardpool_config()
    tags = [str(t).upper() for t in cfg.opponent_pool]
    weights = list(cfg.opponent_pool_weights) if cfg.opponent_pool_weights else None
    if weights is not None and len(weights) != len(tags):
        weights = None
    pool = TrainingOpponentPool(
        enabled=True,
        tags=tags,
        weights=weights,
        rng=np.random.default_rng(42),
    )

    counts: Counter[str] = Counter()
    n = 500
    for _ in range(n):
        if pool.weights is not None:
            tag = str(pool.rng.choice(pool.tags, p=pool.weights)).upper()
        else:
            tag = str(pool.rng.choice(pool.tags)).upper()
        counts[tag] += 1

    print(f"Pool tags: {pool.tags}")
    print(f"Samples ({n} episodes):")
    for tag in pool.tags:
        print(f"  {tag}: {counts[tag]} ({100.0 * counts[tag] / n:.1f}%)")

    missing = [t for t in ("OP11", "OP12") if counts[t] == 0]
    if missing:
        print(f"FAIL: never sampled {missing}")
        return 1
    if "OP11" not in tags or "OP12" not in tags:
        print("FAIL: elite hardpool missing OP11 or OP12")
        return 1
    print("PASS: OP11 and OP12 in elite hardpool and sampled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
