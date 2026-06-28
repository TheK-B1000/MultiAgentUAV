#!/usr/bin/env python3
"""Audit OP5..OP12 presence in training/eval pools and presets."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CURRICULUM = ("OP5", "OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12")

# Intentional holdouts documented here (not accidental omissions).
HOLDOUT_TRAINING = {
    "OP4": "eval-only (stripped by config_validation unless --allow-op4-in-training-pool)",
    "OP11": "paper-faithful v5i4/v5i6 pool is OP5/6/7 only; OP11 in elite_hardpool",
    "OP12": "paper-faithful v5i4/v5i6 pool is OP5/6/7 only; OP12 in elite_hardpool",
}


def _pool_from_preset(fn_name: str) -> tuple[str, ...]:
    import importlib

    mod = importlib.import_module("rl.config_presets")
    fn = getattr(mod, fn_name, None)
    if fn is None:
        return ()
    cfg = fn()
    return tuple(str(x).upper() for x in (getattr(cfg, "opponent_pool", ()) or ()))


def _scan_preset_pools() -> dict[str, set[str]]:
    from rl.presets import PRESET_REGISTRY

    out: dict[str, set[str]] = {}
    for name in PRESET_REGISTRY:
        try:
            from rl.config.ppo_config import PPOConfig
            from rl.presets import apply_preset

            cfg = apply_preset(PPOConfig(), name)
            pool = {str(x).upper() for x in (getattr(cfg, "opponent_pool", ()) or ())}
            if pool:
                out[name] = pool
        except Exception:
            continue
    return out


def main() -> int:
    from rl.evaluation.opponent_resolution import SUPPORTED_OPPONENTS
    from rl.training.config_validation import EVAL_ONLY_TRAINING_OPPONENT_TAGS

    preset_pools = _scan_preset_pools()
    training_presets: dict[str, set[str]] = {}
    for name, pool in preset_pools.items():
        training_presets[name] = pool - EVAL_ONLY_TRAINING_OPPONENT_TAGS

    eval_supported = {k.split("_")[0] if k.startswith("OP") else k for k in SUPPORTED_OPPONENTS}
    eval_supported = {x for x in eval_supported if x.startswith("OP") and x[2:].isdigit()}

    hardpool = set(_pool_from_preset("v6i8_adapter_balanced_hardpool_config"))
    elite = set(_pool_from_preset("v6i8_adapter_balanced_elite_hardpool_config"))

    print("# Opponent pool audit\n")
    print("| Opponent | Training pools | Evaluation pools | Holdout status | Notes |")
    print("|----------|----------------|------------------|----------------|-------|")

    for opp in CURRICULUM:
        in_training = [n for n, p in training_presets.items() if opp in p]
        in_eval = "yes" if opp in eval_supported or opp in SUPPORTED_OPPONENTS else "no"
        holdout = ""
        if opp in ("OP11", "OP12") and not in_training:
            holdout = "intentional in paper-faithful; use elite_hardpool"
        elif opp in HOLDOUT_TRAINING:
            holdout = HOLDOUT_TRAINING[opp]
        elif not in_training:
            holdout = "not in any preset pool scanned"
        else:
            holdout = "active"
        notes = []
        if opp in hardpool:
            notes.append("v6i8_hardpool")
        if opp in elite:
            notes.append("v6i8_elite_hardpool")
        print(
            f"| {opp} | {len(in_training)} presets | {in_eval} | {holdout} | {', '.join(notes) or '-'} |"
        )

    print("\n## Key pools")
    print(f"- v6i8 hardpool: {sorted(hardpool)}")
    print(f"- v6i8 elite hardpool: {sorted(elite)}")
    print(f"- eval SUPPORTED_OPPONENTS tags: {len(SUPPORTED_OPPONENTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
