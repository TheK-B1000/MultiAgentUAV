"""Oracle-gated K=2 production run. Implements ORACLE_GATED_K2_RUN_SPEC.json.

Fresh step-0 K=2 under the frozen treatment:

    z0 -> 16 envs -> OP6 + SDS2_A_payoff_INIT_3   (Pole A)
    z1 -> 16 envs -> OP7                           (Pole B)

    rehearsal gated by EXACT FIT labels:
        Delta < 0  A-preferred -> z0 toward pi_A, z1 untouched
        Delta > 0  B-preferred -> z1 toward pi_B, z0 untouched
        Delta == 0             -> ZERO strategic pressure

Rollout assignment uses POLE. Rehearsal gating uses MEASURED PREFERENCE. These are
deliberately different: 44 FIT pole-B states are A-preferred, and conflating them
would anchor z1 toward pi_A on exactly those states.

Both halves were verified before this existed -- rehearsal smoke 11e54ba7, rollout
smoke 26eafaef. Auditors are attached FAIL-CLOSED: if the treatment drifts, the run
dies on the offending episode and classifies itself INVALID_TREATMENT rather than
finishing and being discovered invalid.

EVAL 10700129..10700160 is SEALED. This trains; it does not look.

Run:  python experiments/run_oracle_gated_k2_production.py
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

SD = ROOT / "artifacts" / "strategic_demand"
SPEC = SD / "sppo" / "ORACLE_GATED_K2_RUN_SPEC.json"
THRESHOLDS = SD / "sppo" / "ABSTENTION_THRESHOLDS.json"
OUT_DIR = SD / "sppo" / "oracle_gated_k2_production"
RECORD = SD / "sppo" / "ORACLE_GATED_K2_PRODUCTION_RESULT.json"

OPPONENT_ID_TO_POLE = {5: "A", 6: "B"}      # OP6 -> Pole A, OP7 -> Pole B
EVAL_BLOCK = range(10_700_129, 10_700_161)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    cfg.load_path = None                       # fresh: step 0, no optimizer reuse

    # CRITICAL: build_exp2_config inherits EXP2C's UNGATED teacher compression --
    # distillation toward both teachers on EVERY state, including the 681 ties.
    # That is precisely what oracle gating replaces, and leaving it on would apply
    # full strategic pressure to states where the payoff evidence establishes no
    # preference, inverting the core principle of this entire arc.
    cfg.exp2_teacher_compression_enabled = False
    for flag in ("rasr_regime_qpsi", "rasr_private_critic_heads", "rasr_directed_identity"):
        if hasattr(cfg, flag):
            setattr(cfg, flag, False)
    cfg.run_tag = "oracle_gated_k2_production"
    cfg.checkpoint_dir = str(OUT_DIR / "ckpts")
    cfg.metrics_csv_path = str(OUT_DIR / "metrics.csv")
    cfg.episode_csv_path = str(OUT_DIR / "episode_rows.csv")

    if int(cfg.seed) in EVAL_BLOCK:
        raise SystemExit("REFUSING: training seed lies inside the sealed EVAL block")
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-steps", type=int, default=0,
                    help="short wiring verification; writes no production record")
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
    from rl.oracle_rehearsal import load_bank, rehearsal_anchor_loss
    from rl.training.orchestrator import orchestrate_training_run

    spec = _frozen()
    if RECORD.is_file():
        raise SystemExit(f"REFUSING: {RECORD} exists; this run is one-shot")
    cfg = build_production_config(spec)
    p = spec["PARAMETERS_RESOLVED"]
    verifying = args.verify_steps > 0
    if verifying:
        cfg.total_timesteps = int(args.verify_steps)
        cfg.run_tag = "oracle_gated_k2_wiring_verification"
        cfg.checkpoint_dir = str(OUT_DIR / "verify_ckpts")
        cfg.metrics_csv_path = str(OUT_DIR / "verify_metrics.csv")
        cfg.episode_csv_path = str(OUT_DIR / "verify_episode_rows.csv")

    checks = [check_opponent_mode(cfg), check_fresh_training(cfg),
              check_thresholds_frozen(THRESHOLDS, "ORACLE_GATED_REHEARSAL")]
    failed = [c for c in checks if c.blocking and not c.passed]
    print("ORACLE-GATED K=2 PRODUCTION RUN")
    print(format_checks(checks))
    if failed:
        raise LaunchGateError("LAUNCH REFUSED:\n" + format_checks(checks))

    bank = load_bank(rng_seed=int(cfg.seed))
    comp = bank.composition()
    print(f"\n  seed {cfg.seed}   steps {cfg.total_timesteps:,}   fresh step 0")
    print(f"  rehearsal bank {comp['eligible']} eligible "
          f"({comp['A_preferred']} A-pref, {comp['B_preferred']} B-pref), "
          f"{comp['tied_excluded_from_sampling']} tied EXCLUDED")
    print(f"  lambda {p['rehearsal_lambda']}  cadence {p['rehearsal_cadence']}  "
          f"batch {p['rehearsal_batch_size']}")
    print(f"  EVAL {EVAL_BLOCK.start}..{EVAL_BLOCK.stop - 1} SEALED\n", flush=True)

    if args.dry_run:
        print("DRY RUN -- gates verified, bank loaded, nothing trained.")
        return 0

    state = {}

    def _attach(trainer, _cfg):
        """Seam: auditors FAIL-CLOSED, plus the oracle-gated rehearsal runner."""
        state["auditor"] = hooks.attach(
            trainer, z_to_pole={0: "A", 1: "B"},
            opponent_to_pole=OPPONENT_ID_TO_POLE,
            hard_fail=True,                    # drift kills the run on the episode
            artifact_dir=OUT_DIR)
        state["runner"] = OracleRehearsalRunner(
            trainer, bank,
            lam=float(p["rehearsal_lambda"]),
            cadence=int(p["rehearsal_cadence"]),
            batch_size=int(p["rehearsal_batch_size"]))
        trainer.oracle_rehearsal_runner = state["runner"]   # the name the updater reads

    try:
        manifest = orchestrate_training_run(
            cfg,
            pre_rollout_env_setup=partial(
                configure_exp2b_live_environment,
                contract={"record": "oracle-gated K=2 production", "utc": _now()},
                training_seed_range=(int(cfg.seed), int(cfg.seed) + 320),
                manifest_key="oracle_gated_k2_protocol",
                context_label="oracle-gated K=2 production construction"),
            post_trainer_setup=_attach)
        verdict, reason = "COMPLETE", None
    except LaunchGateError as exc:
        manifest, verdict, reason = None, "INVALID_TREATMENT", str(exc)
        print(f"\n  RUN TERMINATED BY AUDITOR: {exc}")

    auditor = state.get("auditor")
    runner = state.get("runner")
    coverage = (auditor.coverage(expected_envs=32, min_resets=2)
                if auditor is not None else None)
    record = {
        "record": "Oracle-gated K=2 production run",
        "status": "FROZEN_RESULT", "utc": _now(),
        "implements": "ORACLE_GATED_K2_RUN_SPEC.json",
        "VERDICT": verdict,
        "termination_reason": reason,
        "seed": int(cfg.seed), "total_timesteps": int(cfg.total_timesteps),
        "fresh_step_0": True, "checkpoint_weights_loaded": False,
        "rehearsal": {
            "bank": comp,
            "lambda": p["rehearsal_lambda"], "cadence": p["rehearsal_cadence"],
            "batch_size": p["rehearsal_batch_size"],
            "telemetry": bank.telemetry(),
            "updates": getattr(runner, "n_updates", None)},
        "auditor": auditor.telemetry() if auditor is not None else None,
        "coverage": coverage,
        "env_setup": (manifest or {}).get("oracle_gated_k2_protocol") if isinstance(manifest, dict) else None,
        "budget_caveat": spec["BUDGET_DECISION_AND_ITS_CAVEAT"]["CAVEAT_TO_STATE_IN_ANY_COMPARISON"],
        "EVAL_touched": False,
        "authorizes": "nothing further; EVAL scoring is a separate frozen step",
    }
    if verifying:
        print(f"\n  WIRING VERIFICATION ({cfg.total_timesteps} steps, no record written)")
        print(f"    rehearsal updates      {getattr(runner, 'n_updates', 0)}  <- MUST be > 0")
        print(f"    ppo minibatches seen   {getattr(runner, 'n_ppo_minibatches', 0)}")
        print(f"    tied exposures         {bank.telemetry()['tied_exposures']}  <- MUST be 0")
        print(f"    envs observed          {coverage['envs_observed'] if coverage else None}/32")
        ok = (getattr(runner, 'n_updates', 0) > 0
              and bank.telemetry()['tied_exposures'] == 0
              and verdict == "COMPLETE")
        print(f"    WIRING: {'OK' if ok else 'BROKEN'}")
        return 0 if ok else 1

    RECORD.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    if coverage:
        print(f"  envs {coverage['envs_observed']}/32   min resets "
              f"{coverage['min_resets_observed']}   mismatches {coverage['total_mismatches']}")
    if runner is not None:
        print(f"  rehearsal updates {runner.n_updates}   "
              f"tied exposures {bank.telemetry()['tied_exposures']}")
    print(f"  -> {RECORD}")
    return 0 if verdict == "COMPLETE" else 1


class OracleRehearsalRunner:
    """Interleaved oracle-gated rehearsal, one step every ``cadence`` PPO minibatches.

    Mirrors the validated SAPPO AnchorRunner cadence semantics. The gating lives
    entirely in which (state, z, teacher-action) triples reach the loss -- a tied
    state never enters a batch, so the non-preferred latent receives no gradient
    rather than a push away.
    """

    def __init__(self, trainer, bank, *, lam: float, cadence: int, batch_size: int):
        if lam <= 0.0:
            raise ValueError("lambda must be > 0; disabled rehearsal means not attaching")
        if cadence < 1:
            raise ValueError("cadence must be >= 1")
        self.trainer = trainer
        self.bank = bank
        self.lam = float(lam)
        self.cadence = int(cadence)
        self.batch_size = int(batch_size)
        self.n_ppo_minibatches = 0
        self.n_updates = 0
        self.last_loss = float("nan")

    def note_ppo_minibatch(self) -> bool:
        self.n_ppo_minibatches += 1
        if self.n_ppo_minibatches % self.cadence:
            return False
        self.step()
        return True

    def step(self) -> None:
        from rl.oracle_rehearsal import rehearsal_anchor_loss
        batch = self.bank.sample(self.batch_size)
        device = str(getattr(self.trainer, "device", "cpu"))
        loss = self.lam * rehearsal_anchor_loss(self.trainer.model, batch, device=device)
        opt = self.trainer.optimizer
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        self.n_updates += 1
        self.last_loss = float(loss.detach())
        self.bank.assert_zero_tied_pressure()


if __name__ == "__main__":
    raise SystemExit(main())
