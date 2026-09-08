"""H-OG-PSP V3 production run. Implements the frozen HOG_PSP_V3_SPEC.json.

Three optimisation channels under one trainer, three separate jobs:

    CTF reward            -> task PPO                -> win the game
    paired teacher bank   -> state-level rehearsal   -> specialist preservation
    frozen D_A / D_B      -> trajectory identity PG  -> long-horizon preservation

Component 1, private capacity: V6I26 LRO deep branches give each latent a private
2-layer trunk and action head. OG-PSP had exactly zero private parameters, so a
B-favouring and an A-favouring gradient necessarily competed for the same weights.

Component 2, trajectory identity: a frozen, pole-specific discriminator scores each
COMPLETED episode, and that score enters as a policy-gradient regulariser. It is NOT
added to reward. D's output never touches reward, returns, GAE, or critic targets.

Episodes are observed by wrapping the rollout collector at the INSTANCE level inside
post_trainer_setup. No shared rollout machinery is modified, so previously frozen
experiments are untouched.

Run:  python experiments/run_hog_psp_v3_production.py --verify-steps 12288
      python experiments/run_hog_psp_v3_production.py
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

SD = ROOT / "artifacts" / "strategic_demand"
SPEC = SD / "sppo" / "HOG_PSP_V3_SPEC.json"
THRESHOLDS = SD / "sppo" / "ABSTENTION_THRESHOLDS.json"   # the file carrying tau, as in OG-PSP
OUT_DIR = SD / "sppo" / "hog_psp_v3_production"
RECORD = SD / "sppo" / "HOG_PSP_V3_PRODUCTION_RESULT.json"

OPPONENT_ID_TO_POLE = {5: "A", 6: "B"}
OPPONENT_ID_TO_POLE_IDX = {5: 0, 6: 1}
EVAL_BLOCK = range(11_300_101, 11_300_133)

TRAINING_SEED = 11_300_001
TOTAL_STEPS = 1_000_000
LAM_PAIRED, CADENCE, BATCH_STATES = 0.1, 4, 64      # unchanged from OG-PSP
LAM_TRAJECTORY = 0.05                                # frozen; not tuned on smoke magnitudes

LRO_FLAGS = {
    "enable_latent_z_residual": True,
    "latent_population_birth_per_z_action_heads": True,
    "latent_lro_deep_branches": True,
    "latent_z_residual_alpha": 1.0,
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _frozen() -> dict:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: V3 spec is not frozen: {spec['status']!r}")
    if "AMENDMENT_1_COMPONENT_2_MECHANISM" not in spec:
        raise SystemExit("REFUSING: the PG-regulariser amendment is not in the spec")
    return spec


def build_config():
    from experiments.run_exp2_k2_latent_compression import build_exp2_config

    cfg, _ = build_exp2_config()
    cfg.seed = TRAINING_SEED
    cfg.total_timesteps = TOTAL_STEPS
    cfg.mode = "FIXED_OPPONENT"
    cfg.opponent_randomize = False
    cfg.latent_assignment_mode = "static_env"
    cfg.forced_latent_env_ids = tuple([0] * 16 + [1] * 16)
    cfg.load_path = None
    cfg.exp2_teacher_compression_enabled = False     # ungated compression inverts the gate
    for k, v in LRO_FLAGS.items():
        setattr(cfg, k, v)
    for flag in ("rasr_regime_qpsi", "rasr_private_critic_heads", "rasr_directed_identity"):
        if hasattr(cfg, flag):
            setattr(cfg, flag, False)
    cfg.run_tag = "hog_psp_v3_production"
    cfg.checkpoint_dir = str(OUT_DIR / "ckpts")
    cfg.metrics_csv_path = str(OUT_DIR / "metrics.csv")
    cfg.episode_csv_path = str(OUT_DIR / "episode_rows.csv")
    if int(cfg.seed) in EVAL_BLOCK:
        raise SystemExit("REFUSING: training seed lies inside the sealed EVAL block")
    return cfg


class TrajectoryChannel:
    """Observes rollouts, flushes completed episodes, applies identity PG.

    Installed by wrapping the collector's collect() on the INSTANCE, never on the
    class and never by editing shared rollout code.
    """

    def __init__(self, trainer, discriminators, *, lam: float, n_envs: int):
        from rl.trajectory_identity import (
            EpisodeFeatureAccumulator, OpenEpisodeTransitions, TrajectoryIdentityRunner,
        )
        self.trainer = trainer
        self.runner = TrajectoryIdentityRunner(discriminators, lam=lam)
        self.acc = None                       # built lazily; needs obs_vec shape
        self.transitions = OpenEpisodeTransitions(n_envs)
        self.n_envs = int(n_envs)
        self.updates = 0
        self.episodes_flushed = 0
        self.rollouts_seen = 0
        self.by_cell: dict[str, int] = {}
        self.last_loss = float("nan")
        self._EFA = EpisodeFeatureAccumulator

    def _ensure(self, n_agents: int, vec_dim: int) -> None:
        if self.acc is None:
            self.acc = self._EFA(self.n_envs, n_agents, vec_dim)

    def consume(self, buf) -> None:
        """Walk one rollout, feeding both accumulators and flushing finished episodes."""
        f = buf.fields
        obs_vec = f["obs_vec"].detach().cpu().numpy()
        actions = f["actions"].detach().cpu().numpy()
        grid = f["obs_grid"].detach().cpu().numpy()
        amask = f["obs_agent_mask"].detach().cpu().numpy()
        omask = f["obs_mask"].detach().cpu().numpy()
        z = f["z"].detach().cpu().numpy()
        opp = f["opponent_id"].detach().cpu().numpy()
        term = f["terminated"].detach().cpu().numpy()
        trunc = f["truncated"].detach().cpu().numpy()

        T = int(buf.pos if not buf.full else buf.buffer_size)
        self._ensure(obs_vec.shape[2], obs_vec.shape[3])
        self.rollouts_seen += 1
        ready = []

        for t in range(T):
            for b in range(self.n_envs):
                self.acc.observe(b, obs_vec[t, b], actions[t, b])
                self.transitions.observe(b, {"grid": grid[t, b], "vec": obs_vec[t, b],
                                             "agent_mask": amask[t, b], "mask": omask[t, b]},
                                         actions[t, b])
                if bool(term[t, b]) or bool(trunc[t, b]):
                    feats = self.acc.flush(b)
                    trans = self.transitions.flush(b)
                    if feats is None or trans is None:
                        continue
                    pole = OPPONENT_ID_TO_POLE_IDX.get(int(opp[t, b]))
                    if pole is None:
                        continue
                    ready.append({"z": int(z[t, b]), "pole": pole,
                                  "features": feats["features"],
                                  "obs": trans["obs"], "actions": trans["actions"]})
                    self.episodes_flushed += 1
                    cell = f"z{int(z[t, b])}|{'AB'[pole]}"
                    self.by_cell[cell] = self.by_cell.get(cell, 0) + 1
        if ready:
            self._update(ready)

    def _update(self, episodes: list[dict]) -> None:
        device = str(getattr(self.trainer, "device", "cpu"))
        loss = self.runner.loss(self.trainer.model, episodes, device=device)
        opt = self.trainer.optimizer
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        self.updates += 1
        self.last_loss = float(loss.detach())
        self.runner.D.assert_still_frozen()          # fail-closed, every update

    def telemetry(self) -> dict:
        return {"updates": self.updates, "episodes_flushed": self.episodes_flushed,
                "rollouts_seen": self.rollouts_seen, "by_cell": dict(sorted(self.by_cell.items())),
                "last_loss": self.last_loss,
                "episodes_dropped_too_long": self.transitions.dropped_too_long,
                **self.runner.telemetry()}


def install(trainer, channel: TrajectoryChannel) -> None:
    """Wrap collect() on the live collector instance. Shared code is untouched."""
    collector = getattr(trainer, "rollout_collector", None) or getattr(trainer, "collector", None)
    if collector is None or not hasattr(collector, "collect"):
        raise SystemExit("REFUSING: no rollout collector with collect() on the trainer; "
                         "the trajectory channel would be silently dead")
    original = collector.collect

    def wrapped(*a, **k):
        buf = original(*a, **k)
        channel.consume(buf)
        return buf

    collector.collect = wrapped
    trainer._hog_psp_trajectory_channel = channel


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
        cfg.run_tag = "hog_psp_v3_wiring_verification"
        cfg.checkpoint_dir = str(OUT_DIR / "verify_ckpts")
        cfg.metrics_csv_path = str(OUT_DIR / "verify_metrics.csv")
        cfg.episode_csv_path = str(OUT_DIR / "verify_episode_rows.csv")

    checks = [check_opponent_mode(cfg), check_fresh_training(cfg),
              check_thresholds_frozen(THRESHOLDS, "ORACLE_GATED_REHEARSAL")]
    print("H-OG-PSP V3 PRODUCTION RUN")
    print(format_checks(checks))
    if [c for c in checks if c.blocking and not c.passed]:
        raise LaunchGateError("LAUNCH REFUSED:\n" + format_checks(checks))

    bank = load_paired_bank(include_v2=True, rng_seed=int(cfg.seed))
    D = FrozenDiscriminators(verify=True)
    comp = bank.composition()
    print(f"\n  seed {cfg.seed}   steps {cfg.total_timesteps:,}   fresh step 0")
    print(f"  private capacity: LRO deep branches, fixed alpha {LRO_FLAGS['latent_z_residual_alpha']}")
    print(f"  paired bank {comp['eligible']} eligible, {comp['tied_excluded_from_sampling']} tied EXCLUDED")
    print(f"  D_A {D.sha['A'][:12]} ({D.record['per_pole']['A']['held_out_balanced_accuracy']:.4f})  "
          f"D_B {D.sha['B'][:12]} ({D.record['per_pole']['B']['held_out_balanced_accuracy']:.4f})")
    print(f"  lambda paired {LAM_PAIRED}  lambda trajectory {LAM_TRAJECTORY}")
    print(f"  EVAL {EVAL_BLOCK.start}..{EVAL_BLOCK.stop - 1} SEALED\n", flush=True)

    if args.dry_run:
        print("DRY RUN -- gates verified, bank and discriminators loaded, nothing trained.")
        return 0

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
            opponent_to_pole=OPPONENT_ID_TO_POLE, hard_fail=True, artifact_dir=OUT_DIR)
        state["paired"] = PairedRehearsalRunner(
            trainer, bank, lam=LAM_PAIRED, cadence=CADENCE, batch_states=BATCH_STATES)
        trainer.oracle_rehearsal_runner = state["paired"]     # the name the updater reads
        state["trajectory"] = TrajectoryChannel(
            trainer, D, lam=LAM_TRAJECTORY, n_envs=len(cfg.forced_latent_env_ids))
        install(trainer, state["trajectory"])

    try:
        manifest = orchestrate_training_run(
            cfg,
            pre_rollout_env_setup=partial(
                configure_exp2b_live_environment,
                contract={"record": "H-OG-PSP V3 production", "utc": _now()},
                training_seed_range=(int(cfg.seed), int(cfg.seed) + 320),
                manifest_key="hog_psp_v3_protocol",
                context_label="H-OG-PSP V3 production construction"),
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

    auditor = state.get("auditor")
    paired, traj = state.get("paired"), state.get("trajectory")
    tel = bank.telemetry()
    ttel = traj.telemetry() if traj else {}
    coverage = auditor.coverage(expected_envs=32, min_resets=2) if auditor else None

    if verifying:
        ok = (getattr(paired, "n_updates", 0) > 0
              and ttel.get("updates", 0) > 0
              and ttel.get("episodes_flushed", 0) > 0
              and tel["tied_exposures"] == 0
              and tel["latent_exposures"].get("z0") == tel["latent_exposures"].get("z1")
              and v1_calls["n"] == 0 and verdict == "COMPLETE")
        print(f"\n  COMPOSITION VERIFICATION ({cfg.total_timesteps} steps, no record written)")
        print(f"    task PPO minibatches      {getattr(paired, 'n_ppo_minibatches', 0)}  <- MUST be > 0")
        print(f"    paired rehearsal updates  {getattr(paired, 'n_updates', 0)}  <- MUST be > 0")
        print(f"    trajectory PG updates     {ttel.get('updates', 0)}  <- MUST be > 0")
        print(f"    episodes flushed          {ttel.get('episodes_flushed', 0)}  by cell {ttel.get('by_cell')}")
        print(f"    z0 / z1 exposures         {tel['latent_exposures']}  <- MUST be equal")
        print(f"    tied exposures            {tel['tied_exposures']}  <- MUST be 0")
        print(f"    legacy one-sided calls    {v1_calls['n']}  <- MUST be 0")
        print(f"    episodes dropped too long {ttel.get('episodes_dropped_too_long', 0)}")
        if coverage:
            print(f"    envs / min resets         {coverage['envs_observed']}/32, "
                  f"{coverage['min_resets_observed']}  mismatches {coverage['total_mismatches']}")
        print(f"    COMPOSITION: {'OK' if ok else 'BROKEN'}")
        return 0 if ok else 1

    RECORD.write_text(json.dumps({
        "record": "H-OG-PSP V3 production run", "status": "FROZEN_RESULT", "utc": _now(),
        "implements": "HOG_PSP_V3_SPEC.json", "VERDICT": verdict,
        "termination_reason": reason, "seed": int(cfg.seed),
        "total_timesteps": int(cfg.total_timesteps), "fresh_step_0": True,
        "private_capacity": LRO_FLAGS,
        "paired_rehearsal": {"bank": comp, "lambda": LAM_PAIRED, "cadence": CADENCE,
                             "batch_states": BATCH_STATES,
                             "updates": getattr(paired, "n_updates", 0), "telemetry": tel},
        "trajectory_identity": {"lambda": LAM_TRAJECTORY, "discriminator_sha256": D.sha,
                                "held_out_accuracy": {p: D.record["per_pole"][p]
                                                      ["held_out_balanced_accuracy"]
                                                      for p in ("A", "B")},
                                **ttel},
        "treatment_invariants": {"tied_exposures": tel["tied_exposures"],
                                 "latent_exposures_equal":
                                     tel["latent_exposures"].get("z0") == tel["latent_exposures"].get("z1"),
                                 "legacy_one_sided_path_calls": v1_calls["n"]},
        "coverage": coverage,
        "EVAL_touched": False,
        "authorizes": "nothing further; EVAL is a separate frozen decision",
    }, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {RECORD}")
    return 0 if verdict == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
