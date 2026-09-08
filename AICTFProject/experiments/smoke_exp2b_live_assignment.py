"""Zero-step live-environment proof for EXP2B on development seed 8500001."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_exp2b_specialization_preserving_compression import (
    build_exp2b_config,
    configure_exp2b_live_environment,
)
from rl.training.config_validation import normalize_and_validate_training_config
from rl.training.factories import build_training_env
from rl.training.resolved_config import resolve_training_config


def main() -> int:
    cfg, contract = build_exp2b_config()
    cfg.seed = 8_500_001
    cfg = normalize_and_validate_training_config(cfg)
    resolved = resolve_training_config(cfg)
    env = build_training_env(
        cfg,
        initial_phase=resolved.initial_phase,
        initial_opponent_tag=resolved.initial_opponent_tag,
    )
    try:
        proof = configure_exp2b_live_environment(
            env, cfg, contract=contract, allow_development_seed=True,
        )["exp2b_protocol"]
    finally:
        env.close()
    print(json.dumps({
        "verdict": "EXP2B_LIVE_ASSIGNMENT_PASS",
        "development_seed": cfg.seed,
        "environment_steps": 0,
        "resolved_live_cells": proof["resolved_live_cells"],
        "ruleset_id": proof["ruleset_id"],
        "resolved_opponent_rows": proof["resolved_opponent_rows"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
