"""Live RASR-PPO assigned-pole persistence smoke across episode resets.

The default S0 path exercises the frozen 16/0/0/16 treatment with the old
scorer. ``--arm R3`` is an optional short lifecycle check that also attaches
the regime scorer, private critic heads, and directed-identity runner.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RASR_DIR = ROOT / "artifacts" / "strategic_demand" / "rasrppo"
EXPECTED = {"0": "SCRIPTED:OP6", "1": "SCRIPTED:OP7"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=24_576)
    parser.add_argument("--arm", choices=("S0", "R3"), default="S0")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    from experiments.run_rasrppo_ladder import (
        build_config,
        configure_rasr_live_environment,
        require_dev_collection_gate,
    )
    from rl.training.orchestrator import orchestrate_training_run

    require_dev_collection_gate()
    arm = args.arm.upper()
    out = RASR_DIR / f"pole_persistence_smoke_{arm.lower()}"
    record = RASR_DIR / f"RASR_POLE_PERSISTENCE_SMOKE_{arm}.json"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cfg, parent_contract = build_config(arm)
    cfg.total_timesteps = int(args.steps)
    cfg.run_tag = f"rasrppo_{arm.lower()}_pole_persistence_smoke"
    cfg.checkpoint_dir = str(out / "ckpts")
    cfg.metrics_csv_path = str(out / "metrics.csv")
    cfg.episode_csv_path = str(out / "episode_rows.csv")
    cfg.periodic_checkpoint_steps = 0

    print("RASR-PPO POLE-PERSISTENCE SMOKE")
    print(f"  arm                  {arm}")
    print(f"  mode                 {cfg.mode}")
    print(f"  opponent_randomize   {cfg.opponent_randomize}")
    print(f"  steps                {cfg.total_timesteps}")
    print(f"  regime scorer        {cfg.rasr_regime_qpsi}")
    print(f"  private critic heads {cfg.rasr_private_critic_heads}")
    print(f"  directed identity    {cfg.rasr_directed_identity}\n", flush=True)

    orchestrate_training_run(
        cfg,
        pre_rollout_env_setup=partial(
            configure_rasr_live_environment, contract=parent_contract
        ),
    )

    with Path(cfg.episode_csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("REFUSING: no completed episodes; the smoke proved nothing")
    counts = collections.Counter((row["latent_z"], row["opponent"]) for row in rows)
    per_env = collections.Counter(row.get("env_index", "?") for row in rows)

    by_latent, mismatches = {}, 0
    for z in ("0", "1"):
        total = sum(value for (observed_z, _), value in counts.items() if observed_z == z)
        if not total:
            raise SystemExit(f"REFUSING: no completed episodes for z={z}")
        good = counts[(z, EXPECTED[z])]
        bad = total - good
        mismatches += bad
        by_latent[f"z{z}"] = {
            "n_episodes": total,
            "expected": EXPECTED[z],
            "pct_expected": good / total,
            "mismatches": bad,
            "observed": {
                opponent: count
                for (observed_z, opponent), count in counts.items()
                if observed_z == z
            },
        }
        print(
            f"  z={z} n={total:4d} {EXPECTED[z]}={good / total:.4f} "
            f"mismatches={bad}"
        )

    crossed_resets = len(rows) > 32
    min_episodes_per_env = (
        min(per_env.values()) if per_env and "?" not in per_env else None
    )
    verdict = "PASS" if mismatches == 0 and crossed_resets else "FAIL"
    result = {
        "record": "RASR-PPO assigned-pole persistence live smoke",
        "status": "FROZEN_RESULT",
        "arm": arm,
        "classification": "DIAGNOSTIC",
        "requirement": (
            "16 z0|A, 0 z0|B, 0 z1|A, 16 z1|B environment assignment; "
            "z0->OP6 and z1->OP7 across episode resets with zero mismatches"
        ),
        "training_seed": int(cfg.seed),
        "steps": int(cfg.total_timesteps),
        "completed_episodes": len(rows),
        "min_episodes_per_env": min_episodes_per_env,
        "crossed_episode_resets": crossed_resets,
        "by_latent": by_latent,
        "total_mismatches": mismatches,
        "scientific_flags": {
            "rasr_regime_qpsi": bool(cfg.rasr_regime_qpsi),
            "rasr_private_critic_heads": bool(cfg.rasr_private_critic_heads),
            "rasr_directed_identity": bool(cfg.rasr_directed_identity),
        },
        "verdict": verdict,
    }
    record.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}\n  -> {record}")
    if not args.keep:
        shutil.rmtree(out, ignore_errors=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
