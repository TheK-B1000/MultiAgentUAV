"""H-OG-PSP V4 production run: V3 plus private latent critic heads. Nothing else.

Implements HOG_PSP_V4_SPEC.json.

V3 gave the ACTOR private capacity while both strategic modes still learned their task
advantage through ONE shared scalar value function:

    V_shared(s, z)   ->   V_0(s, z) if z=z0,   V_1(s, z) if z=z1

PPO does not optimise the actor from strategy identity; it optimises through advantages,
r_t -> V -> A_hat -> grad L_PPO. A single value estimator is the funnel through which
both strategies receive compromised task-learning signal.

EXACTLY ONE AXIS CHANGES: rasr_private_critic_heads = True. Every V3 component that was
proven live stays frozen -- private actor branches, paired rehearsal, trajectory-identity
PG, static persistence, 1M budget, poles, auditors, gate.

V3's runner is IMPORTED rather than copied, so the delta is auditable and the shared
machinery cannot drift between the two experiments.

Run:  python experiments/run_hog_psp_v4_production.py --verify-steps 12288
      python experiments/run_hog_psp_v4_production.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_hog_psp_v3_production as V3

SD = ROOT / "artifacts" / "strategic_demand"
SPEC = SD / "sppo" / "HOG_PSP_V4_SPEC.json"
THRESHOLDS = SD / "sppo" / "ABSTENTION_THRESHOLDS.json"
OUT_DIR = SD / "sppo" / "hog_psp_v4_production"
RECORD = SD / "sppo" / "HOG_PSP_V4_PRODUCTION_RESULT.json"

TRAINING_SEED = 11_400_001
EVAL_BLOCK = range(11_400_101, 11_400_133)
TOTAL_STEPS = 1_000_000


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _frozen() -> dict:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: V4 spec is not frozen: {spec['status']!r}")
    return spec


def build_config():
    """V3's config with ONE flag flipped. The delta is deliberately this small."""
    cfg = V3.build_config()

    # --- the single axis under test -------------------------------------------
    cfg.rasr_private_critic_heads = True

    # V3's build_config disables this flag in a loop copied from OG-PSP without
    # examining it -- the oversight this experiment exists to correct. Assert the
    # flip actually survived rather than trusting the assignment.
    if not getattr(cfg, "rasr_private_critic_heads", False):
        raise SystemExit("REFUSING: rasr_private_critic_heads did not stay True")

    cfg.seed = TRAINING_SEED
    cfg.total_timesteps = TOTAL_STEPS
    cfg.run_tag = "hog_psp_v4_production"
    cfg.checkpoint_dir = str(OUT_DIR / "ckpts")
    cfg.metrics_csv_path = str(OUT_DIR / "metrics.csv")
    cfg.episode_csv_path = str(OUT_DIR / "episode_rows.csv")
    if int(cfg.seed) in EVAL_BLOCK:
        raise SystemExit("REFUSING: training seed lies inside the sealed EVAL block")
    return cfg


class CriticHeadAuditor:
    """Counts which private value head served each sample, and watches for misroutes.

    Installed by wrapping CentralizedCritic.forward on the LIVE critic instance. Every
    PPO/GAE value query in this codebase goes through exactly one call site
    (policy.values -> critic(gs, extra=one_hot(z))), so this observes all of them.
    """

    def __init__(self):
        self.calls = 0
        self.rows_by_head = {0: 0, 1: 0}
        self.misroutes = 0
        self.missing_z = 0

    def observe(self, extra) -> None:
        self.calls += 1
        if extra is None:
            self.missing_z += 1
            return
        z = extra.argmax(dim=-1).detach().cpu().numpy()
        for head in (0, 1):
            self.rows_by_head[head] += int((z == head).sum())

    def telemetry(self) -> dict:
        return {"critic_calls": self.calls,
                "rows_served_by_head_V0": self.rows_by_head[0],
                "rows_served_by_head_V1": self.rows_by_head[1],
                "both_heads_live": self.rows_by_head[0] > 0 and self.rows_by_head[1] > 0,
                "value_queries_missing_z": self.missing_z}


def install_critic_auditor(trainer, auditor: CriticHeadAuditor):
    critic = trainer.model.critic
    if not getattr(critic, "private_z_heads", False):
        raise SystemExit("REFUSING: the live critic does not have private z heads; "
                         "V4's single axis is not actually active")
    original = critic.forward

    def wrapped(global_state, extra=None):
        auditor.observe(extra)
        return original(global_state, extra=extra)

    critic.forward = wrapped
    return critic, original


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-steps", type=int, default=0)
    args = ap.parse_args()

    import torch  # noqa: F401
    from experiments.run_exp2b_specialization_preserving_compression import (
        configure_exp2b_live_environment,
    )
    from experiments.run_og_psp_production import PairedRehearsalRunner
    from rl import launch_audit_hooks as hooks
    import rl.oracle_rehearsal as V1
    from rl.launch_gate import (
        LaunchGateError, check_fresh_training, check_opponent_mode,
        check_thresholds_frozen, format_checks,
    )
    from rl.paired_rehearsal import load_paired_bank
    from rl.trajectory_identity import FrozenDiscriminators
    from rl.training.orchestrator import orchestrate_training_run

    _frozen()
    if RECORD.is_file():
        raise SystemExit(f"REFUSING: {RECORD} exists; this run is one-shot")
    cfg = build_config()
    verifying = args.verify_steps > 0
    if verifying:
        cfg.total_timesteps = int(args.verify_steps)
        cfg.run_tag = "hog_psp_v4_wiring_verification"
        cfg.checkpoint_dir = str(OUT_DIR / "verify_ckpts")
        cfg.metrics_csv_path = str(OUT_DIR / "verify_metrics.csv")
        cfg.episode_csv_path = str(OUT_DIR / "verify_episode_rows.csv")

    checks = [check_opponent_mode(cfg), check_fresh_training(cfg),
              check_thresholds_frozen(THRESHOLDS, "ORACLE_GATED_REHEARSAL")]
    print("H-OG-PSP V4 PRODUCTION RUN  (V3 + private critic heads)")
    print(format_checks(checks))
    if [c for c in checks if c.blocking and not c.passed]:
        raise LaunchGateError("LAUNCH REFUSED:\n" + format_checks(checks))

    bank = load_paired_bank(include_v2=True, rng_seed=int(cfg.seed))
    D = FrozenDiscriminators(verify=True)
    comp = bank.composition()
    print(f"\n  seed {cfg.seed}   steps {cfg.total_timesteps:,}   fresh step 0")
    print(f"  THE ONE AXIS: rasr_private_critic_heads = {cfg.rasr_private_critic_heads}")
    print(f"  private actor branches: LRO deep, alpha {V3.LRO_FLAGS['latent_z_residual_alpha']}")
    print(f"  paired bank {comp['eligible']} eligible, {comp['tied_excluded_from_sampling']} tied EXCLUDED")
    print(f"  lambda paired {V3.LAM_PAIRED}  lambda trajectory {V3.LAM_TRAJECTORY}")
    print(f"  EVAL {EVAL_BLOCK.start}..{EVAL_BLOCK.stop - 1} SEALED\n", flush=True)

    if args.dry_run:
        print("DRY RUN -- gates verified, nothing trained.")
        return 0

    v1_calls = {"n": 0}
    _orig = V1.RehearsalBank.sample

    def _tripwire(self, *a, **k):
        v1_calls["n"] += 1
        return _orig(self, *a, **k)
    V1.RehearsalBank.sample = _tripwire

    state = {}
    critic_auditor = CriticHeadAuditor()

    def _attach(trainer, _cfg):
        state["auditor"] = hooks.attach(
            trainer, z_to_pole={0: "A", 1: "B"},
            opponent_to_pole=V3.OPPONENT_ID_TO_POLE, hard_fail=True, artifact_dir=OUT_DIR)
        state["paired"] = PairedRehearsalRunner(
            trainer, bank, lam=V3.LAM_PAIRED, cadence=V3.CADENCE,
            batch_states=V3.BATCH_STATES)
        trainer.oracle_rehearsal_runner = state["paired"]
        state["trajectory"] = V3.TrajectoryChannel(
            trainer, D, lam=V3.LAM_TRAJECTORY, n_envs=len(cfg.forced_latent_env_ids))
        V3.install(trainer, state["trajectory"])
        state["critic"], state["critic_orig"] = install_critic_auditor(trainer, critic_auditor)
        # head parameter snapshot, so cross-head motion is measurable at the end
        m = trainer.model
        state["head_before"] = {n: p.detach().cpu().numpy().copy()
                                for n, p in m.named_parameters() if "head_V" in n}

    try:
        manifest = orchestrate_training_run(
            cfg,
            pre_rollout_env_setup=partial(
                configure_exp2b_live_environment,
                contract={"record": "H-OG-PSP V4 production", "utc": _now()},
                training_seed_range=(int(cfg.seed), int(cfg.seed) + 320),
                manifest_key="hog_psp_v4_protocol",
                context_label="H-OG-PSP V4 production construction"),
            post_trainer_setup=_attach)
        verdict, reason = "COMPLETE", None
    except Exception as exc:                                   # noqa: BLE001
        from rl.paired_rehearsal import PairedRehearsalError
        from rl.trajectory_identity import TrajectoryIdentityError
        if not isinstance(exc, (LaunchGateError, PairedRehearsalError, TrajectoryIdentityError)):
            V1.RehearsalBank.sample = _orig
            raise
        manifest, verdict, reason = None, "INVALID_TREATMENT", str(exc)
        print(f"\n  RUN TERMINATED BY AN INVARIANT: {exc}")
    finally:
        V1.RehearsalBank.sample = _orig
        if "critic" in state:
            state["critic"].forward = state["critic_orig"]

    auditor = state.get("auditor")
    paired, traj = state.get("paired"), state.get("trajectory")
    tel = bank.telemetry()
    ttel = traj.telemetry() if traj else {}
    ctel = critic_auditor.telemetry()
    coverage = auditor.coverage(expected_envs=32, min_resets=2) if auditor else None

    head_moved = {}
    if "head_before" in state:
        for n, p in state["paired"].trainer.model.named_parameters():
            if "head_V" in n:
                head_moved[n] = not np.array_equal(state["head_before"][n],
                                                   p.detach().cpu().numpy())
    v0_moved = any(v for k, v in head_moved.items() if "head_V0" in k)
    v1_moved = any(v for k, v in head_moved.items() if "head_V1" in k)

    if verifying:
        ok = (getattr(paired, "n_updates", 0) > 0 and ttel.get("updates", 0) > 0
              and ctel["both_heads_live"] and ctel["value_queries_missing_z"] == 0
              and v0_moved and v1_moved
              and tel["tied_exposures"] == 0
              and tel["latent_exposures"].get("z0") == tel["latent_exposures"].get("z1")
              and v1_calls["n"] == 0 and verdict == "COMPLETE")
        print(f"\n  COMPOSITION VERIFICATION ({cfg.total_timesteps} steps, no record written)")
        print(f"    task PPO minibatches      {getattr(paired, 'n_ppo_minibatches', 0)}")
        print(f"    paired rehearsal updates  {getattr(paired, 'n_updates', 0)}")
        print(f"    trajectory PG updates     {ttel.get('updates', 0)}")
        print(f"    critic calls              {ctel['critic_calls']}")
        print(f"    rows served by head_V0    {ctel['rows_served_by_head_V0']}  <- MUST be > 0")
        print(f"    rows served by head_V1    {ctel['rows_served_by_head_V1']}  <- MUST be > 0")
        print(f"    value queries missing z   {ctel['value_queries_missing_z']}  <- MUST be 0")
        print(f"    head_V0 / head_V1 moved   {v0_moved} / {v1_moved}  <- BOTH must be True")
        print(f"    z0 / z1 exposures         {tel['latent_exposures']}")
        print(f"    tied / legacy             {tel['tied_exposures']} / {v1_calls['n']}")
        if coverage:
            print(f"    envs / mismatches         {coverage['envs_observed']}/32, "
                  f"{coverage['total_mismatches']}")
        print(f"    COMPOSITION: {'OK' if ok else 'BROKEN'}")
        return 0 if ok else 1

    RECORD.write_text(json.dumps({
        "record": "H-OG-PSP V4 production run", "status": "FROZEN_RESULT", "utc": _now(),
        "implements": "HOG_PSP_V4_SPEC.json", "VERDICT": verdict,
        "termination_reason": reason, "seed": int(cfg.seed),
        "total_timesteps": int(cfg.total_timesteps), "fresh_step_0": True,
        "THE_ONE_AXIS": {"rasr_private_critic_heads": True,
                         "everything_else": "identical to V3"},
        "private_capacity": V3.LRO_FLAGS,
        "critic_heads": {**ctel, "head_V0_moved": v0_moved, "head_V1_moved": v1_moved,
                         "per_param_moved": head_moved},
        "paired_rehearsal": {"bank": comp, "updates": getattr(paired, "n_updates", 0),
                             "telemetry": tel},
        "trajectory_identity": {"lambda": V3.LAM_TRAJECTORY,
                                "discriminator_sha256": D.sha, **ttel},
        "treatment_invariants": {
            "tied_exposures": tel["tied_exposures"],
            "latent_exposures_equal":
                tel["latent_exposures"].get("z0") == tel["latent_exposures"].get("z1"),
            "legacy_one_sided_path_calls": v1_calls["n"],
            "both_critic_heads_live": ctel["both_heads_live"],
            "value_queries_missing_z": ctel["value_queries_missing_z"]},
        "coverage": coverage,
        "EVAL_touched": False,
        "authorizes": "nothing further; EVAL is a separate frozen decision",
    }, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {RECORD}")
    return 0 if verdict == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
