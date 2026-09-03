"""Oracle-gated K=2 V2 production run. Implements ORACLE_GATED_K2_V2_RUN_SPEC.json.

Single-axis delta vs V1: larger diverse oracle rehearsal bank (legacy FIT + 320 new
collection seeds). Everything else frozen at V1.

EVAL 11200001..11200032 is SEALED during training.

Run:  python experiments/run_oracle_gated_k2_v2_production.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_oracle_gated_k2_production import (  # noqa: E402
    OracleRehearsalRunner,
    OPPONENT_ID_TO_POLE,
    _now,
)

SD = ROOT / "artifacts" / "strategic_demand"
SPEC = SD / "sppo" / "ORACLE_GATED_K2_V2_RUN_SPEC.json"
THRESHOLDS = SD / "sppo" / "ABSTENTION_THRESHOLDS.json"
BANK_ASSEMBLY = SD / "sppo" / "ORACLE_GATED_K2_V2_BANK_ASSEMBLY.json"
OUT_DIR = SD / "sppo" / "oracle_gated_k2_v2_production"
RECORD = SD / "sppo" / "ORACLE_GATED_K2_V2_PRODUCTION_RESULT.json"
EVAL_BLOCK = range(11_200_001, 11_200_033)


def _frozen() -> dict:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not spec["status"].startswith("FROZEN_RUN_SPEC"):
        raise SystemExit(f"REFUSING: run spec is not frozen: {spec['status']!r}")
    return spec


def build_production_config(spec: dict):
    from experiments.run_exp2_k2_latent_compression import build_exp2_config
    p = spec["PARAMETERS_RESOLVED"]
    cfg, _ = build_exp2_config()
    cfg.seed = int(p["training_seed"])
    cfg.total_timesteps = int(p["total_timesteps"])
    cfg.mode = "FIXED_OPPONENT"
    cfg.opponent_randomize = False
    cfg.latent_assignment_mode = "static_env"
    cfg.forced_latent_env_ids = tuple([0] * 16 + [1] * 16)
    cfg.load_path = None
    cfg.exp2_teacher_compression_enabled = False
    for flag in ("rasr_regime_qpsi", "rasr_private_critic_heads", "rasr_directed_identity"):
        if hasattr(cfg, flag):
            setattr(cfg, flag, False)
    cfg.run_tag = "oracle_gated_k2_v2_production"
    cfg.checkpoint_dir = str(OUT_DIR / "ckpts")
    cfg.metrics_csv_path = str(OUT_DIR / "metrics.csv")
    cfg.episode_csv_path = str(OUT_DIR / "episode_rows.csv")
    if int(cfg.seed) in EVAL_BLOCK:
        raise SystemExit("REFUSING: training seed lies inside the sealed V2 EVAL block")
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-steps", type=int, default=0)
    args = ap.parse_args()

    import torch  # noqa: F401
    from experiments.run_exp2b_specialization_preserving_compression import (
        configure_exp2b_live_environment,
    )
    from rl import launch_audit_hooks as hooks
    from rl.launch_gate import (
        LaunchGateError, check_fresh_training, check_opponent_mode,
        check_thresholds_frozen, format_checks,
    )
    from rl.oracle_rehearsal import load_bank_v2
    from rl.training.orchestrator import orchestrate_training_run

    spec = _frozen()
    if RECORD.is_file():
        raise SystemExit(f"REFUSING: {RECORD} exists; V2 production is one-shot")
    if not BANK_ASSEMBLY.is_file() and not args.dry_run:
        raise SystemExit(f"REFUSING: {BANK_ASSEMBLY} missing; run bank assembly audit first")

    cfg = build_production_config(spec)
    p = spec["PARAMETERS_RESOLVED"]
    verifying = args.verify_steps > 0
    if verifying:
        cfg.total_timesteps = int(args.verify_steps)
        cfg.run_tag = "oracle_gated_k2_v2_wiring_verification"
        cfg.checkpoint_dir = str(OUT_DIR / "verify_ckpts")
        cfg.metrics_csv_path = str(OUT_DIR / "verify_metrics.csv")
        cfg.episode_csv_path = str(OUT_DIR / "verify_episode_rows.csv")

    checks = [check_opponent_mode(cfg), check_fresh_training(cfg),
              check_thresholds_frozen(THRESHOLDS, "ORACLE_GATED_REHEARSAL")]
    failed = [c for c in checks if c.blocking and not c.passed]
    print("ORACLE-GATED K=2 V2 PRODUCTION RUN")
    print(format_checks(checks))
    if failed:
        raise LaunchGateError("LAUNCH REFUSED:\n" + format_checks(checks))

    if args.dry_run:
        if BANK_ASSEMBLY.is_file():
            bank = load_bank_v2(rng_seed=int(cfg.seed))
            comp = bank.composition()
            print(f"\n  bank eligible {comp['eligible']}  A-pref {comp['A_preferred']}  "
                  f"B-pref {comp['B_preferred']}")
        else:
            print("\n  bank assembly not yet PASS (collection in progress)")
        print("DRY RUN -- gates verified.")
        return 0

    bank = load_bank_v2(rng_seed=int(cfg.seed))
    comp = bank.composition()
    assembly = json.loads(BANK_ASSEMBLY.read_text(encoding="utf-8"))
    print(f"\n  seed {cfg.seed}   steps {cfg.total_timesteps:,}   fresh step 0")
    print(f"  V2 bank {comp['eligible']} eligible "
          f"({comp['A_preferred']} A-pref, {comp['B_preferred']} B-pref)")
    print(f"  assembly combined eligible {assembly['combined']['eligible']}")
    print(f"  lambda {p['rehearsal_lambda']}  cadence {p['rehearsal_cadence']}  "
          f"batch {p['rehearsal_batch_size']}")
    print(f"  EVAL {EVAL_BLOCK.start}..{EVAL_BLOCK.stop - 1} SEALED\n", flush=True)

    state = {}

    def _attach(trainer, _cfg):
        state["auditor"] = hooks.attach(
            trainer, z_to_pole={0: "A", 1: "B"},
            opponent_to_pole=OPPONENT_ID_TO_POLE,
            hard_fail=True, artifact_dir=OUT_DIR)
        state["runner"] = OracleRehearsalRunner(
            trainer, bank,
            lam=float(p["rehearsal_lambda"]),
            cadence=int(p["rehearsal_cadence"]),
            batch_size=int(p["rehearsal_batch_size"]))
        trainer.oracle_rehearsal_runner = state["runner"]

    try:
        manifest = orchestrate_training_run(
            cfg,
            pre_rollout_env_setup=partial(
                configure_exp2b_live_environment,
                contract={"record": "oracle-gated K=2 V2 production", "utc": _now()},
                training_seed_range=(int(cfg.seed), int(cfg.seed) + 320),
                manifest_key="oracle_gated_k2_v2_protocol",
                context_label="oracle-gated K=2 V2 production construction"),
            post_trainer_setup=_attach)
        verdict, reason = "COMPLETE", None
    except LaunchGateError as exc:
        manifest, verdict, reason = None, "INVALID_TREATMENT", str(exc)
        print(f"\n  RUN TERMINATED BY AUDITOR: {exc}")

    auditor = state.get("auditor")
    runner = state.get("runner")
    coverage = auditor.coverage(expected_envs=32, min_resets=2) if auditor else None
    record = {
        "record": "Oracle-gated K=2 V2 production run",
        "status": "FROZEN_RESULT", "utc": _now(),
        "implements": "ORACLE_GATED_K2_V2_RUN_SPEC.json",
        "parent": "ORACLE_GATED_K2_CROSSOVER_NOT_CONFIRMED",
        "diagnostic_fork": "MEMORIZATION_GENERALIZATION",
        "VERDICT": verdict,
        "termination_reason": reason,
        "seed": int(cfg.seed), "total_timesteps": int(cfg.total_timesteps),
        "fresh_step_0": True, "checkpoint_weights_loaded": False,
        "rehearsal": {
            "bank": comp,
            "bank_assembly": str(BANK_ASSEMBLY.relative_to(ROOT)),
            "lambda": p["rehearsal_lambda"], "cadence": p["rehearsal_cadence"],
            "batch_size": p["rehearsal_batch_size"],
            "telemetry": bank.telemetry(),
            "updates": getattr(runner, "n_updates", None)},
        "auditor": auditor.telemetry() if auditor else None,
        "coverage": coverage,
        "env_setup": (manifest or {}).get("oracle_gated_k2_v2_protocol") if isinstance(manifest, dict) else None,
        "EVAL_touched": False,
    }
    if verifying:
        print(f"\n  WIRING VERIFICATION ({cfg.total_timesteps} steps)")
        print(f"    rehearsal updates {getattr(runner, 'n_updates', 0)}")
        print(f"    tied exposures {bank.telemetry()['tied_exposures']}")
        ok = (getattr(runner, "n_updates", 0) > 0
              and bank.telemetry()["tied_exposures"] == 0 and verdict == "COMPLETE")
        print(f"    WIRING: {'OK' if ok else 'BROKEN'}")
        return 0 if ok else 1

    RECORD.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {RECORD}")
    return 0 if verdict == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
