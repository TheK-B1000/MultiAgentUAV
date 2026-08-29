"""Live rollout-assignment smoke for oracle-gated rehearsal.

Proves the OTHER half of the production treatment. The rehearsal smoke showed the
supervision path exists; this shows the on-policy environment actually delivers:

    z0 -> 16 envs -> OP6 + SDS2_A_payoff_INIT_3 overlay   (Pole A)
    z1 -> 16 envs -> OP7, no overlay                       (Pole B)

with opponent_randomize=False and the mapping surviving every auto-reset.

Verification is LIVE, not configurational. Trusting setup config is exactly how
EXP2B looked correct while running something else:

  * assert_live_opponent_batch reads the resolved opponent AND the behaviour-tree
    profile per environment, so the Pole-A overlay is confirmed active rather than
    merely requested;
  * the pass/fail counts come from completed-episode rows -- observed (latent_z,
    opponent) pairs -- not from the config that requested them;
  * more episodes than environments is required, so the mapping is shown to survive
    resets rather than only episode 1.

Adapted from smoke_rasrppo_pole_persistence.py, which solved the same problem for
the RASR ladder. The RASR arms are explicitly OFF here.

Diagnostic. Authorizes nothing. EVAL is never touched.

Run:  python experiments/smoke_oracle_rollout_assignment.py --steps 24576
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand"
SPEC = SD / "sppo" / "ORACLE_GATED_REHEARSAL_SPEC.json"
OUT_DIR = SD / "sppo" / "rollout_assignment_smoke"
RECORD = SD / "sppo" / "ORACLE_ROLLOUT_ASSIGNMENT_SMOKE.json"

EXPECTED = {"0": "SCRIPTED:OP6", "1": "SCRIPTED:OP7"}
EXPECTED_CELLS = {"z0_A": 16, "z0_B": 0, "z1_A": 0, "z1_B": 16}
# a dedicated smoke block, disjoint from the 10700001..10700160 collection block
SMOKE_SEED = 10_800_001


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_spec() -> dict:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN -- SPEC_FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: spec is not frozen: {spec['status']!r}")
    amd = spec.get("AMENDMENT_1_PERSISTENT_Z_TO_POLE_ROLLOUT")
    if not amd:
        raise SystemExit("REFUSING: the persistent z-to-pole amendment is absent")
    if amd["THE_RULING"]["opponent_randomize"] is not False:
        raise SystemExit("REFUSING: spec does not require opponent_randomize=False")
    if "MUST be installed" not in amd["POLE_IS_NOT_JUST_AN_OPPONENT_TAG"]["required"]:
        raise SystemExit("REFUSING: spec does not require the Pole-A overlay")
    return spec


def build_smoke_config(steps: int):
    """EXP2C-style K=2, fresh, with the RASR arms explicitly off."""
    from experiments.run_exp2_k2_latent_compression import build_exp2_config

    cfg, _exp2_contract = build_exp2_config()   # returns (PPOConfig, contract dict)
    cfg.seed = SMOKE_SEED
    cfg.total_timesteps = int(steps)
    cfg.mode = "FIXED_OPPONENT"
    cfg.opponent_randomize = False
    cfg.latent_assignment_mode = "static_env"
    cfg.forced_latent_env_ids = tuple([0] * 16 + [1] * 16)
    cfg.load_path = None                       # fresh: step 0
    cfg.periodic_checkpoint_steps = 0
    # this experiment is not the RASR ladder
    for flag in ("rasr_regime_qpsi", "rasr_private_critic_heads", "rasr_directed_identity"):
        if hasattr(cfg, flag):
            setattr(cfg, flag, False)
    cfg.run_tag = "oracle_rollout_assignment_smoke"
    cfg.checkpoint_dir = str(OUT_DIR / "ckpts")
    cfg.metrics_csv_path = str(OUT_DIR / "metrics.csv")
    cfg.episode_csv_path = str(OUT_DIR / "episode_rows.csv")
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=24_576)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    _require_spec()
    if RECORD.is_file():
        raise SystemExit(f"REFUSING: {RECORD} exists; this smoke is one-shot")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    from experiments.run_exp2b_specialization_preserving_compression import (
        configure_exp2b_live_environment,
    )
    from rl.launch_gate import check_fresh_training, check_opponent_mode
    from rl.training.orchestrator import orchestrate_training_run

    cfg = build_smoke_config(args.steps)
    opp, fresh = check_opponent_mode(cfg), check_fresh_training(cfg)
    print("ORACLE-GATED ROLLOUT-ASSIGNMENT SMOKE")
    print(f"  [{opp.status}] {opp.detail}")
    print(f"  [{fresh.status}] {fresh.detail}")
    print(f"  seed {cfg.seed}   steps {cfg.total_timesteps}   "
          f"latent_k {getattr(cfg, 'latent_k', '?')}")
    print(f"  requires: z0 -> OP6 + pole_A overlay, z1 -> OP7, zero mismatches\n", flush=True)
    failures = [c.detail for c in (opp, fresh) if not c.passed]

    contract = {"record": "oracle-gated rollout assignment smoke", "utc": _now()}
    manifest = orchestrate_training_run(
        cfg,
        pre_rollout_env_setup=partial(
            configure_exp2b_live_environment,
            contract=contract,
            training_seed_range=(SMOKE_SEED, SMOKE_SEED + 320),
            manifest_key="oracle_rollout_protocol",
            context_label="oracle-gated rollout assignment smoke",
        ),
    )

    with Path(cfg.episode_csv_path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("REFUSING: no completed episodes; the smoke proved nothing")

    # Discover the env-identifying column rather than assuming its name. A previous
    # attempt assumed "env_index", found nothing, and silently reported None for
    # per-env coverage -- so persistence was never actually established. Absence is
    # now a hard failure, not a default.
    ENV_COL_CANDIDATES = ("env_index", "env", "env_id", "worker", "vec_index")
    env_col = next((c for c in ENV_COL_CANDIDATES if c in rows[0]), None)
    if env_col is None:
        failures.append(
            f"episode CSV has no env-identifying column (looked for "
            f"{list(ENV_COL_CANDIDATES)}); per-env reset coverage cannot be verified "
            f"and this smoke will not pass on an aggregate. Columns present: "
            f"{sorted(rows[0])[:12]}")

    counts = collections.Counter((r["latent_z"], r["opponent"]) for r in rows)
    per_env = collections.Counter(r[env_col] for r in rows) if env_col else {}
    by_latent, mismatches = {}, 0
    for z in ("0", "1"):
        total = sum(v for (zz, _), v in counts.items() if zz == z)
        if not total:
            failures.append(f"no completed episodes for z={z}")
            continue
        good = counts[(z, EXPECTED[z])]
        bad = total - good
        mismatches += bad
        by_latent[f"z{z}"] = {
            "n_episodes": total, "expected": EXPECTED[z],
            "pct_expected": good / total, "mismatches": bad,
            "observed": {op: c for (zz, op), c in counts.items() if zz == z}}
        print(f"  z={z}  n={total:4d}  {EXPECTED[z]}={good / total:.4f}  mismatches={bad}")

    # PER-ENV reset coverage, not an aggregate. An env that produced N episodes
    # crossed N-1 resets; every env must cross at least MIN_RESETS_PER_ENV.
    MIN_RESETS_PER_ENV = 2
    resets_per_env = {e: n - 1 for e, n in per_env.items()}
    envs_seen = len(per_env)
    envs_with_reset = sum(1 for v in resets_per_env.values() if v >= 1)
    min_resets = min(resets_per_env.values()) if resets_per_env else None

    if mismatches:
        failures.append(f"{mismatches} live (z, opponent) mismatches")
    if env_col:
        if envs_seen != 32:
            failures.append(f"only {envs_seen}/32 environments produced episodes")
        if envs_with_reset != 32:
            failures.append(f"only {envs_with_reset}/32 environments crossed a reset")
        if min_resets is not None and min_resets < MIN_RESETS_PER_ENV:
            failures.append(
                f"min resets per env is {min_resets}, below the required "
                f"{MIN_RESETS_PER_ENV}; persistence through the trainer lifecycle is "
                "not established")

    # Overlay evidence. assert_live_opponent_batch inside the env setup raises on a
    # mismatch, so reaching this point already implies it held -- but the record must
    # CARRY that evidence rather than rely on absence of a crash.
    resolved = (manifest or {}).get("oracle_rollout_protocol", {}) if isinstance(manifest, dict) else {}
    cells = resolved.get("resolved_live_cells")
    opp_rows = resolved.get("resolved_opponent_rows")
    overlay_mismatches = None
    if cells is None or opp_rows is None:
        failures.append(
            "env-setup manifest did not propagate resolved_live_cells / "
            "resolved_opponent_rows; the Pole-A overlay verification ran but was not "
            "captured as evidence, and this smoke will not pass on an uncaptured check")
    else:
        if cells != EXPECTED_CELLS:
            failures.append(f"live cells are not 16/0/0/16: {cells}")
        overlay_mismatches = sum(
            1 for r in opp_rows
            if (str(r.get("live_opponent_key")) == "OP6") != bool(r.get("profile_override_active", r.get("overlay_active")))
        )
        if overlay_mismatches:
            failures.append(
                f"{overlay_mismatches} environments have the Pole-A overlay attached to "
                "the wrong opponent; OP6 must carry it and OP7 must not")

    verdict = "PASS" if not failures else "FAIL"
    RECORD.write_text(json.dumps({
        "record": "Oracle-gated rollout assignment live smoke",
        "status": "FROZEN_RESULT", "classification": "DIAGNOSTIC", "utc": _now(),
        "implements": "ORACLE_GATED_REHEARSAL_SPEC.json AMENDMENT_1",
        "VERDICT": verdict,
        "requirement": ("z0 -> OP6 + SDS2_A_payoff_INIT_3 overlay, z1 -> OP7, "
                        "opponent_randomize False, persistent across every reset"),
        "verification_is_live_not_configurational": (
            "counts come from completed-episode rows, and assert_live_opponent_batch "
            "inside the env setup reads the resolved opponent AND behaviour-tree "
            "profile per environment, so the Pole-A overlay is confirmed ACTIVE"),
        "seed": int(cfg.seed), "steps": int(cfg.total_timesteps),
        "completed_episodes": len(rows),
        "env_column_used": env_col,
        "envs_seen": envs_seen,
        "envs_with_reset": f"{envs_with_reset}/32",
        "min_resets_per_env": min_resets,
        "min_resets_required": MIN_RESETS_PER_ENV,
        "resets_per_env": dict(sorted(resets_per_env.items())) if env_col else None,
        "by_latent": by_latent,
        "mapping_mismatches": mismatches,
        "overlay_mismatches": overlay_mismatches,
        "resolved_live_cells": cells,
        "resolved_opponent_rows": resolved.get("resolved_opponent_rows"),
        "rasr_arms_enabled": False,
        "failures": failures,
        "authorizes": "nothing; PPO launch remains a separate PI decision",
        "EVAL_touched": False,
    }, indent=2), encoding="utf-8")

    if not args.keep:
        shutil.rmtree(OUT_DIR, ignore_errors=True)
    print(f"\n  episodes {len(rows)}   envs_with_reset {envs_with_reset}/32   "
          f"min_resets_per_env {min_resets} (required >= {MIN_RESETS_PER_ENV})")
    print(f"  mapping_mismatches {mismatches}   overlay_mismatches {overlay_mismatches}")
    print(f"  VERDICT: {verdict}")
    for f in failures:
        print(f"    FAIL: {f}")
    print(f"  -> {RECORD}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
