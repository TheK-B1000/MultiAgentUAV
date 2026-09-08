"""OG-PSP production run. Implements OG_PSP_PRODUCTION_RUN_SPEC.json.

Fresh step-0 K=2 under the corrected treatment:

    z0 -> 16 envs -> OP6 + SDS2_A_payoff_INIT_3   (Pole A)
    z1 -> 16 envs -> OP7                           (Pole B)

    every resolved state presents BOTH specialist targets:
        (s, z0) -> pi_A(s)   AND   (s, z1) -> pi_B(s)
    ties -> zero rehearsal pressure

V1 trained only the preferred latent per state, so a latent-independent state->action
map satisfied the whole objective and z carried almost nothing (z0-z1 JSD 0.0051 bits
against 0.3919 available). Paired targets on the same state make that shortcut
impossible wherever the teachers disagree -- 65.5% of the bank.

Deliberately NOT run_oracle_gated_k2_v2_production.py, which implements the superseded
bank-only V2 using V1's one-sided loss.

EVAL 11200001..11200032 is SEALED. This trains; it does not look.

Run:  python experiments/run_og_psp_production.py
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
SPEC = SD / "sppo" / "OG_PSP_PRODUCTION_RUN_SPEC.json"
THRESHOLDS = SD / "sppo" / "ABSTENTION_THRESHOLDS.json"
OUT_DIR = SD / "sppo" / "og_psp_production"
RECORD = SD / "sppo" / "OG_PSP_PRODUCTION_RESULT.json"

OPPONENT_ID_TO_POLE = {5: "A", 6: "B"}
EVAL_BLOCK = range(11_200_001, 11_200_033)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _frozen() -> dict:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not spec["status"].startswith("FROZEN_RUN_SPEC"):
        raise SystemExit(f"REFUSING: run spec is not frozen: {spec['status']!r}")
    return spec


def build_config(spec: dict):
    from experiments.run_exp2_k2_latent_compression import build_exp2_config
    t = spec["AUTHORIZED_TREATMENT"]

    cfg, _ = build_exp2_config()
    cfg.seed = int(t["training_seed"])
    cfg.total_timesteps = int(t["total_timesteps"])
    cfg.mode = "FIXED_OPPONENT"
    cfg.opponent_randomize = False
    cfg.latent_assignment_mode = "static_env"
    cfg.forced_latent_env_ids = tuple([0] * 16 + [1] * 16)
    cfg.load_path = None
    # EXP2C's UNGATED teacher compression would apply pressure to every state
    # including ties, inverting the treatment. Explicitly off, as in V1.
    cfg.exp2_teacher_compression_enabled = False
    for flag in ("rasr_regime_qpsi", "rasr_private_critic_heads", "rasr_directed_identity"):
        if hasattr(cfg, flag):
            setattr(cfg, flag, False)
    cfg.run_tag = "og_psp_production"
    cfg.checkpoint_dir = str(OUT_DIR / "ckpts")
    cfg.metrics_csv_path = str(OUT_DIR / "metrics.csv")
    cfg.episode_csv_path = str(OUT_DIR / "episode_rows.csv")
    if int(cfg.seed) in EVAL_BLOCK:
        raise SystemExit("REFUSING: training seed lies inside the sealed EVAL block")
    return cfg


class PairedRehearsalRunner:
    """Interleaved paired rehearsal, one step every ``cadence`` PPO minibatches.

    Both latents are supervised on the SAME state every step, so the invariants are
    checked after every update rather than only at the end -- a broken pairing is the
    V1 defect returning and must stop the run immediately.
    """

    def __init__(self, trainer, bank, *, lam: float, cadence: int, batch_states: int):
        if lam <= 0.0:
            raise ValueError("lambda must be > 0; disabled rehearsal means not attaching")
        if cadence < 1:
            raise ValueError("cadence must be >= 1")
        self.trainer = trainer
        self.bank = bank
        self.lam = float(lam)
        self.cadence = int(cadence)
        self.batch_states = int(batch_states)
        self.n_ppo_minibatches = 0
        self.n_updates = 0
        self.n_pairs_seen = 0
        self.n_disagreement_states = 0
        self.last_loss = float("nan")

    def note_ppo_minibatch(self) -> bool:
        self.n_ppo_minibatches += 1
        if self.n_ppo_minibatches % self.cadence:
            return False
        self.step()
        return True

    def step(self) -> None:
        from rl.paired_rehearsal import paired_rehearsal_loss
        batch = self.bank.sample(self.batch_states)
        device = str(getattr(self.trainer, "device", "cpu"))
        loss = self.lam * paired_rehearsal_loss(self.trainer.model, batch, device=device)
        opt = self.trainer.optimizer
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        self.n_updates += 1
        self.n_pairs_seen += int(batch["n_pairs"])
        self.n_disagreement_states += int(batch["teachers_disagree"].sum())
        self.last_loss = float(loss.detach())
        self.bank.assert_invariants()          # fail-closed, every update


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
    import rl.oracle_rehearsal as V1
    from rl.launch_gate import (
        LaunchGateError, check_fresh_training, check_opponent_mode,
        check_thresholds_frozen, format_checks,
    )
    from rl.paired_rehearsal import load_paired_bank
    from rl.training.orchestrator import orchestrate_training_run

    spec = _frozen()
    t = spec["AUTHORIZED_TREATMENT"]
    if RECORD.is_file():
        raise SystemExit(f"REFUSING: {RECORD} exists; this run is one-shot")
    cfg = build_config(spec)
    verifying = args.verify_steps > 0
    if verifying:
        cfg.total_timesteps = int(args.verify_steps)
        cfg.run_tag = "og_psp_wiring_verification"
        cfg.checkpoint_dir = str(OUT_DIR / "verify_ckpts")
        cfg.metrics_csv_path = str(OUT_DIR / "verify_metrics.csv")
        cfg.episode_csv_path = str(OUT_DIR / "verify_episode_rows.csv")

    checks = [check_opponent_mode(cfg), check_fresh_training(cfg),
              check_thresholds_frozen(THRESHOLDS, "ORACLE_GATED_REHEARSAL")]
    print("OG-PSP PRODUCTION RUN")
    print(format_checks(checks))
    if [c for c in checks if c.blocking and not c.passed]:
        raise LaunchGateError("LAUNCH REFUSED:\n" + format_checks(checks))

    bank = load_paired_bank(include_v2=True, rng_seed=int(cfg.seed))
    comp = bank.composition()
    print(f"\n  seed {cfg.seed}   steps {cfg.total_timesteps:,}   fresh step 0")
    print(f"  paired bank {comp['eligible']} eligible "
          f"({comp['A_preferred']} A-pref, {comp['B_preferred']} B-pref), "
          f"{comp['tied_excluded_from_sampling']} tied EXCLUDED")
    print(f"  masked teacher disagreement {comp['teacher_disagreement_frac']:.4f}")
    print(f"  lambda {t['lambda']}  cadence {t['cadence']}  "
          f"{t['batch_states']} states -> {t['batch_pairs']} pairs, MEAN")
    print(f"  EVAL {EVAL_BLOCK.start}..{EVAL_BLOCK.stop - 1} SEALED\n", flush=True)

    if args.dry_run:
        print("DRY RUN -- gates verified, paired bank loaded, nothing trained.")
        return 0

    # the superseded one-sided path must never fire during this run
    v1_calls = {"n": 0}
    _orig = V1.RehearsalBank.sample

    def _tripwire(self, *a, **k):
        v1_calls["n"] += 1
        return _orig(self, *a, **k)
    V1.RehearsalBank.sample = _tripwire

    state = {}

    def _attach(trainer, _cfg):
        state["auditor"] = hooks.attach(
            trainer, z_to_pole={0: "A", 1: "B"},
            opponent_to_pole=OPPONENT_ID_TO_POLE,
            hard_fail=True, artifact_dir=OUT_DIR)
        state["runner"] = PairedRehearsalRunner(
            trainer, bank, lam=float(t["lambda"]), cadence=int(t["cadence"]),
            batch_states=int(t["batch_states"]))
        trainer.oracle_rehearsal_runner = state["runner"]   # the name the updater reads

    try:
        manifest = orchestrate_training_run(
            cfg,
            pre_rollout_env_setup=partial(
                configure_exp2b_live_environment,
                contract={"record": "OG-PSP production", "utc": _now()},
                training_seed_range=(int(cfg.seed), int(cfg.seed) + 320),
                manifest_key="og_psp_protocol",
                context_label="OG-PSP production construction"),
            post_trainer_setup=_attach)
        verdict, reason = "COMPLETE", None
    except (LaunchGateError, Exception) as exc:                 # noqa: BLE001
        from rl.paired_rehearsal import PairedRehearsalError
        if not isinstance(exc, (LaunchGateError, PairedRehearsalError)):
            V1.RehearsalBank.sample = _orig
            raise
        manifest, verdict, reason = None, "INVALID_TREATMENT", str(exc)
        print(f"\n  RUN TERMINATED BY AN INVARIANT: {exc}")
    finally:
        V1.RehearsalBank.sample = _orig

    auditor, runner = state.get("auditor"), state.get("runner")
    tel = bank.telemetry()
    coverage = (auditor.coverage(expected_envs=32, min_resets=2)
                if auditor is not None else None)

    if verifying:
        ok = (getattr(runner, "n_updates", 0) > 0
              and tel["tied_exposures"] == 0
              and tel["latent_exposures"].get("z0") == tel["latent_exposures"].get("z1")
              and v1_calls["n"] == 0 and verdict == "COMPLETE")
        print(f"\n  WIRING VERIFICATION ({cfg.total_timesteps} steps, no record written)")
        print(f"    rehearsal updates       {getattr(runner, 'n_updates', 0)}  <- MUST be > 0")
        print(f"    pairs seen              {getattr(runner, 'n_pairs_seen', 0)}")
        print(f"    disagreement states     {getattr(runner, 'n_disagreement_states', 0)}")
        print(f"    z0 / z1 exposures       {tel['latent_exposures']}  <- MUST be equal")
        print(f"    tied exposures          {tel['tied_exposures']}  <- MUST be 0")
        print(f"    legacy one-sided calls  {v1_calls['n']}  <- MUST be 0")
        print(f"    WIRING: {'OK' if ok else 'BROKEN'}")
        return 0 if ok else 1

    record = {
        "record": "OG-PSP production run", "status": "FROZEN_RESULT", "utc": _now(),
        "implements": "OG_PSP_PRODUCTION_RUN_SPEC.json",
        "VERDICT": verdict, "termination_reason": reason,
        "seed": int(cfg.seed), "total_timesteps": int(cfg.total_timesteps),
        "fresh_step_0": True, "checkpoint_weights_loaded": False,
        "paired_rehearsal": {
            "bank": comp,
            "lambda": t["lambda"], "cadence": t["cadence"],
            "batch_states": t["batch_states"], "batch_pairs": t["batch_pairs"],
            "loss_aggregation": "MEAN over (state, latent) pairs",
            "updates": getattr(runner, "n_updates", None),
            "pairs_seen": getattr(runner, "n_pairs_seen", None),
            "disagreement_states_seen": getattr(runner, "n_disagreement_states", None),
            "telemetry": tel},
        "treatment_invariants": {
            "tied_exposures": tel["tied_exposures"],
            "latent_exposures_equal": tel["latent_exposures"].get("z0") == tel["latent_exposures"].get("z1"),
            "legacy_one_sided_path_calls": v1_calls["n"]},
        "auditor": auditor.telemetry() if auditor is not None else None,
        "coverage": coverage,
        "env_setup": (manifest or {}).get("og_psp_protocol") if isinstance(manifest, dict) else None,
        "interpretive_context": spec["INTERPRETIVE_CONTEXT_FROZEN_BEFORE_THE_RESULT"],
        "EVAL_touched": False,
        "authorizes": "nothing further; EVAL is a separate frozen decision",
    }
    RECORD.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    if coverage:
        print(f"  envs {coverage['envs_observed']}/32  min resets {coverage['min_resets_observed']}"
              f"  mismatches {coverage['total_mismatches']}")
    print(f"  rehearsal updates {getattr(runner, 'n_updates', 0)}   "
          f"tied {tel['tied_exposures']}   legacy calls {v1_calls['n']}   "
          f"z0/z1 {tel['latent_exposures']}")
    print(f"  -> {RECORD}")
    return 0 if verdict == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
