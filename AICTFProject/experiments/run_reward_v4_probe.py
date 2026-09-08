"""Reward V4 reliability probe -- objective-dominant budget, 300k steps, fresh seed.

Closes the entire tag-reward family in one experiment: with NO tag paying
anything (neither routine nor carrier), do
independent PPO seeds keep attacking instead of drifting toward camping?

A routine tag pays +100 -- exactly what a flag capture pays -- while being far
more frequent and far less risky, and DOUBLE what tagging the enemy carrier
pays (+50). The failed-action penalty is restored to its baseline -0.2 so that
this knob is the only factor that differs from the original G0-v2 run.

Only the tag-reward family differs from the frozen G0-v2 configuration.
Captures are still rewarded and OOB is still penalised.
Learning rate, entropy schedule, value coefficient, batch shape, every other
reward term, map, ruleset, opponents and horizon are all inherited unchanged
from ``run_g0_v2_seed`` so the comparison stays one-factor.

Every checkpoint records BOTH verdicts:

    SYSTEM_HEALTH -- is PPO numerically alive?
    TASK_HEALTH   -- is the policy still playing CTF?

The baseline collapse passed the first on all 33 checkpoints while failing the
second completely. Reporting them separately is the point.

Gate: artifacts/reward_v4/PROBE_GATE_FROZEN.json (frozen first).

Run:  python experiments/run_reward_v4_probe.py --seed 3000001
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from experiments.run_formal_ppo_smoke_2400001 import (  # noqa: E402
    HealthLedger,
    _check_event,
    _finalize_event_checks,
    _scan_stats,
    _scan_tensor_mapping,
)
from experiments.run_g0_v2_seed import (  # noqa: E402
    AGENTS,
    CANONICAL_MAP,
    EPISODE_HORIZON,
    N_ENVS,
    N_EPOCHS,
    N_STEPS,
    BATCH_SIZE,
    OPPONENTS,
    RULESET_ID,
)
from rl.training.task_health import (  # noqa: E402
    VALIDATION_OPPONENTS,
    VALIDATION_SEEDS,
    combined_verdict,
    evaluate_task_health,
)

V4_SEEDS = (3_000_001, 3_000_002, 3_000_003)
ABLATION_SEEDS = V4_SEEDS
TOTAL_TIMESTEPS = 300_000
CHECKPOINT_INTERVAL = 50_000
COLLAPSE_ZONE = 225_000
SPARSE_TAG_NO_FLAG_POINTS_ABLATED = 0.0
V4_ACTION_FAILED_PUNISHMENT = -0.004
V4_ENEMY_MAV_KILL_REWARD = 0.0
V4_OPPONENT_OOB_POINTS = 0.0
V4_OWN_OOB_POINTS = -70.0   # derived: 0.0181 x baseline, from 184 events/episode
BASELINE_ACTION_FAILED_PUNISHMENT = -0.2
SPARSE_TAG_WITH_FLAG_POINTS_ABLATED = 0.0
BASELINE_SPARSE_TAG_POINTS = 100.0
BASELINE_SPARSE_CARRIER_TAG_POINTS = 50.0
DEVICE = "cuda"


def run_tag_for(seed: int) -> str:
    return f"g0_v2_rewardv4_seed{seed}"


def artifact_dir_for(seed: int) -> Path:
    return PROJECT_ROOT / "artifacts" / "reward_v4" / run_tag_for(seed)


def build_config(seed: int):
    """Frozen G0-v2 config with exactly one field changed."""
    from rl.config.ppo_config import PPOConfig
    from rl.telemetry_mode import TrainingTelemetryMode

    art = artifact_dir_for(seed)
    cfg = PPOConfig()
    cfg.run_tag = run_tag_for(seed)
    cfg.seed = int(seed)
    cfg.total_timesteps = TOTAL_TIMESTEPS
    cfg.periodic_checkpoint_steps = CHECKPOINT_INTERVAL
    cfg.device = DEVICE
    cfg.map_layout = CANONICAL_MAP
    cfg.max_decision_steps = EPISODE_HORIZON
    cfg.max_blue_agents = AGENTS

    cfg.mode = "OPPONENT_POOL"
    cfg.opponent_randomize = True
    cfg.opponent_pool = OPPONENTS
    cfg.opponent_pool_weights = ()
    cfg.train_domain_randomization = False

    cfg.n_envs = N_ENVS
    cfg.n_steps = N_STEPS
    cfg.batch_size = BATCH_SIZE
    cfg.n_epochs = N_EPOCHS

    cfg.gpu_native_env = True
    cfg.use_latent_strategy = False
    cfg.use_stable_marl_ppo = False

    cfg.load_path = None
    cfg.additional_timesteps = 0
    cfg.load_weights_only = False

    cfg.tag_telemetry_enabled = True
    cfg.formal_run = True

    # ---- THE ONLY CHANGED FACTOR ----------------------------------------
    cfg.env_sparse_tag_no_flag_points = SPARSE_TAG_NO_FLAG_POINTS_ABLATED
    cfg.env_sparse_tag_with_flag_points = SPARSE_TAG_WITH_FLAG_POINTS_ABLATED
    # Explicitly restored to baseline so the previous probe's change cannot leak in.
    cfg.env_action_failed_punishment = V4_ACTION_FAILED_PUNISHMENT
    cfg.env_enemy_mav_kill_reward = V4_ENEMY_MAV_KILL_REWARD
    cfg.env_sparse_opponent_oob_points = V4_OPPONENT_OOB_POINTS
    cfg.env_sparse_own_oob_points = V4_OWN_OOB_POINTS

    cfg.checkpoint_dir = str(art / "ckpts")
    cfg.metrics_csv_path = str(art / "metrics.csv")
    cfg.episode_csv_path = str(art / "episode_rows.csv")
    cfg.training_telemetry_mode = TrainingTelemetryMode.OFF
    cfg.enable_progress_bar = False
    cfg.verbose_training = True
    return cfg


def run_validation_panel(trainer, *, global_step: int, device: str):
    """Tiny held-out panel: is this policy still playing CTF?"""
    from experiments.run_g0_v2_evaluation import run_eval_episode
    from rl.custom_ppo.inference_policy import CustomPPOInferencePolicy

    rows = []
    model = trainer.model
    was_training = model.training
    model.eval()
    try:
        policy = CustomPPOInferencePolicy(model=model, device=torch.device(device))
        for opp in VALIDATION_OPPONENTS:
            for vs in VALIDATION_SEEDS:
                rows.append(run_eval_episode(policy, opponent=opp, seed=vs, device=device))
    finally:
        model.train(was_training)
    return evaluate_task_health(rows, global_step=global_step)


def _reward_mass(art: Path, horizon: int = 240) -> dict:
    """Per-episode contribution and absolute mass share of every reward channel.

    If V3 fails, this shows immediately whether another channel inherited the
    crown -- which is the failure mode three prior probes all exhibited.
    """
    import csv as _csv
    import statistics as _stats

    p = art / "metrics.csv"
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        return {}

    def seg(col: str, lo: float, hi: float) -> float:
        vals = [float(r[col]) for r in rows
                if col in r and r[col] not in (None, "", "nan")
                and lo <= float(r["timesteps"]) < hi]
        return _stats.fmean(vals) if vals else float("nan")

    cols = {
        "terminal": "reward_terminal_mean",
        "sparse": "reward_sparse_mean",
        "failed_commit": "reward_failure_mean",
        "offense": "reward_offense_mean",
        "pbrs": "reward_pbrs_mean",
        "team": "reward_team_mean",
    }
    early = {k: seg(c, 0, 50_000) * horizon for k, c in cols.items()}
    late = {k: seg(c, 250_000, 310_000) * horizon for k, c in cols.items()}
    mass = {k: abs(v) for k, v in late.items() if not math.isnan(v)}
    total = sum(mass.values()) or 1e-12

    out = {}
    for k in cols:
        e, l = early.get(k, float("nan")), late.get(k, float("nan"))
        out[k] = {
            "per_episode_early": None if math.isnan(e) else round(e, 4),
            "per_episode_late": None if math.isnan(l) else round(l, 4),
            "abs_mass_share_late": round(mass.get(k, 0.0) / total, 4),
            "magnitude_rising": bool(
                not math.isnan(e) and not math.isnan(l) and abs(l) > abs(e)
            ),
        }
    dominant = max(out, key=lambda k: out[k]["abs_mass_share_late"])
    return {
        "channels": out,
        "dominant_channel": dominant,
        "terminal_share": out["terminal"]["abs_mass_share_late"],
        "objective_is_minority_of_mass": bool(out["terminal"]["abs_mass_share_late"] < 0.10),
    }


def _sparse_breakdown(c: dict) -> dict:
    """Sparse points attributable to tags vs captures, under both settings.

    Reported for BOTH the baseline and the ablated tag value so the size of the
    removed incentive is explicit rather than inferred.
    """
    from_tags_baseline = (
        BASELINE_SPARSE_TAG_POINTS * c["tag_blue_noncarrier"]
        + 50.0 * c["tag_blue_carrier"]
        - BASELINE_SPARSE_TAG_POINTS * c["tag_against_blue"]
    )
    from_tags_ablated = (
        SPARSE_TAG_NO_FLAG_POINTS_ABLATED * c["tag_blue_noncarrier"]
        + SPARSE_TAG_WITH_FLAG_POINTS_ABLATED * c["tag_blue_carrier"]
        - SPARSE_TAG_NO_FLAG_POINTS_ABLATED * c["tag_against_blue"]
    )
    from_captures = 100.0 * c["capture_blue"] - 100.0 * c["capture_red"]
    total_caps = c["capture_blue"] + c["capture_red"]
    return {
        **c,
        "tags_per_capture_event": (
            round((c["tag_blue_noncarrier"] + c["tag_blue_carrier"]) / total_caps, 3)
            if total_caps else None
        ),
        "sparse_points_from_tags_at_baseline_100": round(from_tags_baseline, 1),
        "sparse_points_from_tags_at_ablated_value": round(from_tags_ablated, 1),
        "sparse_points_from_captures": round(from_captures, 1),
        "tag_share_of_sparse_at_baseline": (
            round(abs(from_tags_baseline) / (abs(from_tags_baseline) + abs(from_captures)), 4)
            if (abs(from_tags_baseline) + abs(from_captures)) > 0 else None
        ),
    }


def install_probes(ledger: HealthLedger, records: list, art: Path, device: str,
                   sparse_counts: dict):
    from rl.custom_ppo.trainer import CustomPPOTrainer

    real_update = CustomPPOTrainer.update
    real_collect = CustomPPOTrainer.collect_rollout
    real_save = CustomPPOTrainer.save
    state = {"opt_wrapped": False, "tag_range": None}

    def _wrap_optimizers(trainer):
        if state["opt_wrapped"]:
            return
        state["opt_wrapped"] = True
        for name in ("primary", "actor", "critic", "router", "actor_cf"):
            opt = getattr(trainer.optimizers, name, None)
            if opt is None:
                continue
            real_step = opt.step

            def make(real_step=real_step, opt=opt):
                def step(*a, **k):
                    tot, n = 0.0, 0
                    for g in opt.param_groups:
                        for p in g["params"]:
                            if p.grad is None:
                                continue
                            if not torch.isfinite(p.grad).all():
                                ledger.fail("non-finite gradient at optimizer step")
                            tot += float(p.grad.detach().double().pow(2).sum().item())
                            n += 1
                    if n:
                        norm = math.sqrt(tot)
                        ledger.grad_norms.append(norm)
                        if norm <= 0.0:
                            ledger.zero_grad_steps += 1
                    ledger.optimizer_steps += 1
                    probe = None
                    for g in opt.param_groups:
                        for p in g["params"]:
                            if p.grad is not None and p.numel():
                                probe = (p, p.detach().clone())
                                break
                        if probe is not None:
                            break
                    out = real_step(*a, **k)
                    if probe is not None and not torch.equal(probe[0].detach(), probe[1]):
                        ledger.param_change_events += 1
                    return out
                return step
            opt.step = make()

    def collect_rollout(self, *a, **k):
        _wrap_optimizers(self)
        buf = real_collect(self, *a, **k)
        ledger.rollouts += 1
        fields = getattr(buf, "fields", None)
        if isinstance(fields, dict):
            _scan_tensor_mapping("rollout", fields, ledger.nonfinite_buffer, ledger)
        core = getattr(self.env, "core", None)
        if core is not None:
            if state["tag_range"] is None:
                state["tag_range"] = float(
                    getattr(core.cfg, "tag_radius", None) or getattr(core, "tag_radius", 0.0) or 0.0
                )
            try:
                for e in core.drain_tag_events():
                    _check_event(ledger, e, tag_range=state["tag_range"] or 1e9)
                    # Split the sparse ledger by SOURCE. Leaving tags and
                    # captures inside one reward_sparse bucket is what let the
                    # tag term hide during the previous diagnosis.
                    et = e.get("event_type")
                    if et == "tag_success":
                        if e.get("tagger_team") == "blue":
                            if e.get("target_was_carrying_flag"):
                                sparse_counts["tag_blue_carrier"] += 1
                            else:
                                sparse_counts["tag_blue_noncarrier"] += 1
                        else:
                            sparse_counts["tag_against_blue"] += 1
                    elif et == "capture_scored":
                        key = ("capture_blue" if e.get("scoring_team") == "blue"
                               else "capture_red")
                        sparse_counts[key] += 1
                    elif et == "mine_tag":
                        n = int(e.get("count", 1))
                        sparse_counts["mine_tag_blue" if e.get("team") == "blue"
                                      else "mine_tag_red"] += n
                    elif et == "out_of_bounds":
                        # Measurement only in V3: the -100 value is unchanged
                        # because its rate has never been observed.
                        sparse_counts["oob_blue" if e.get("team") == "blue"
                                      else "oob_red"] += 1
            except Exception as exc:
                ledger.fail(f"tag event drain failed: {exc}")
        return buf

    def update(self, buffer, *a, **k):
        stats = real_update(self, buffer, *a, **k)
        ledger.updates += 1
        _scan_stats(ledger, dict(stats))
        bad = [n for n, p in self.model.named_parameters()
               if p.is_floating_point() and not torch.isfinite(p).all()]
        if bad:
            ledger.nonfinite_params.extend(bad[:5])
            ledger.fail(f"non-finite model parameters: {bad[:5]}")
        return stats

    def save(self, path: str):
        real_save(self, path)
        step = int(getattr(self, "global_step", 0))
        system_ok = not (
            ledger.nonfinite_stats or ledger.nonfinite_buffer or ledger.nonfinite_params
            or ledger.nonfinite_optimizer or ledger.legality_violations
            or ledger.identity_violations or ledger.zero_grad_steps
        )
        try:
            panel = run_validation_panel(self, global_step=step, device=device)
            panel_d = panel.to_dict()
        except Exception as exc:
            panel_d = {"verdict": "ERROR", "error": f"{type(exc).__name__}: {exc}",
                       "global_step": step}
            ledger.fail(f"task-health panel failed at {step}: {exc}")

        verdicts = combined_verdict(system_ok, panel) if panel_d.get("verdict") != "ERROR" else {
            "SYSTEM_HEALTH": "PASS" if system_ok else "FAIL", "TASK_HEALTH": "ERROR"}

        rec = {"global_step": step, "checkpoint": str(path),
               **verdicts, "task_panel": panel_d,
               "grad_norm_mean_postclip": (
                   sum(ledger.grad_norms) / len(ledger.grad_norms)) if ledger.grad_norms else None,
               "zero_gradient_steps": ledger.zero_grad_steps}
        records.append(rec)
        (art / "health_timeline.json").write_text(
            json.dumps(records, indent=2, default=str), encoding="utf-8")
        print(f"[ABLATION] step={step} SYSTEM_HEALTH={rec['SYSTEM_HEALTH']} "
              f"TASK_HEALTH={rec['TASK_HEALTH']} "
              f"pickups={panel_d.get('pickups')} off={panel_d.get('offensive_commitment')} "
              f"def={panel_d.get('defensive_commitment')} wr={panel_d.get('win_rate')} "
              f"net={panel_d.get('net_captures')}")

    CustomPPOTrainer.collect_rollout = collect_rollout
    CustomPPOTrainer.update = update
    CustomPPOTrainer.save = save
    return lambda: (
        setattr(CustomPPOTrainer, "collect_rollout", real_collect),
        setattr(CustomPPOTrainer, "update", real_update),
        setattr(CustomPPOTrainer, "save", real_save),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=list(ABLATION_SEEDS))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(max(1, int(args.threads)))
    seed = int(args.seed)
    art = artifact_dir_for(seed)
    art.mkdir(parents=True, exist_ok=True)
    (art / "ckpts").mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("FATAL: CUDA required.")
        return 2

    print("=" * 78)
    print(f"REWARD V4 RELIABILITY PROBE  seed={seed}  steps={TOTAL_TIMESTEPS:,}")
    print(f"sparse_tag_no_flag_points:   {BASELINE_SPARSE_TAG_POINTS} -> {SPARSE_TAG_NO_FLAG_POINTS_ABLATED}")
    print(f"action_failed_punishment:    -0.2 -> {V4_ACTION_FAILED_PUNISHMENT}")
    print(f"enemy_mav_kill_reward:       0.5 -> {V4_ENEMY_MAV_KILL_REWARD}")
    print(f"sparse_opponent_oob_points:  100.0 -> {V4_OPPONENT_OOB_POINTS}  (OOB farm removed)")
    print(f"sparse_own_oob_points:       -100.0 -> {V4_OWN_OOB_POINTS}  (derived from measured rate)")
    print(f"sparse_mine_tag_points:      100.0 UNCHANGED (UNRESOLVED — measured this run)")
    print(f"sparse_tag_with_flag_points: {BASELINE_SPARSE_CARRIER_TAG_POINTS} -> {SPARSE_TAG_WITH_FLAG_POINTS_ABLATED}")
    print("[objective-dominant budget; OOB -100 UNCHANGED and measurement-only]")
    print(f"action_failed_punishment restored to baseline {BASELINE_ACTION_FAILED_PUNISHMENT}")
    print(f"collapse zone to cross: {COLLAPSE_ZONE:,}  panel every {CHECKPOINT_INTERVAL:,}")
    print(f"map={CANONICAL_MAP} ruleset={RULESET_ID} opponents={OPPONENTS}")
    print("=" * 78)

    from rl.training.orchestrator import orchestrate_training_run

    cfg = build_config(seed)
    ledger = HealthLedger()
    records: list = []
    sparse_counts = {"tag_blue_noncarrier": 0, "tag_blue_carrier": 0,
                     "tag_against_blue": 0, "capture_blue": 0, "capture_red": 0,
                     "oob_blue": 0, "oob_red": 0,
                     "mine_tag_blue": 0, "mine_tag_red": 0}
    restore = install_probes(ledger, records, art, DEVICE, sparse_counts)
    started = time.time()
    try:
        orchestrate_training_run(cfg)
        completed, error = True, ""
    except BaseException as exc:
        completed, error = False, f"{type(exc).__name__}: {exc}"
        ledger.fail(f"training raised: {error}")
        traceback.print_exc()
    finally:
        restore()

    _finalize_event_checks(ledger)

    final = records[-1]["task_panel"] if records else {}
    report = {
        "probe": "Reward V4 reliability probe",
        "seed": seed,
        "run_id": run_tag_for(seed),
        "sparse_tag_no_flag_points": SPARSE_TAG_NO_FLAG_POINTS_ABLATED,
        "baseline_sparse_tag_no_flag_points": BASELINE_SPARSE_TAG_POINTS,
        "action_failed_punishment": BASELINE_ACTION_FAILED_PUNISHMENT,
        "sparse_tag_with_flag_points": SPARSE_TAG_WITH_FLAG_POINTS_ABLATED,
        "changed": {"tag_no_flag": "100 -> 0", "tag_carrier": "50 -> 0", "failed_commit": "-0.2 -> -0.004",
                    "enemy_mav_kill_reward": "0.5 -> 0.0", "opponent_oob": "100 -> 0", "own_oob": "-100 -> -70"},
        "unresolved": {"sparse_mine_tag_points": 100.0},
        "unchanged": {"capture": 100.0, "terminal": "win 1.0 / lose -1.0 / draw -0.5", "oob": -100.0},
        "note": "budget redesign, NOT a single-factor ablation",
        "reward_mass": _reward_mass(art),
        "sparse_decomposition": _sparse_breakdown(sparse_counts),
        "total_timesteps": TOTAL_TIMESTEPS,
        "collapse_zone": COLLAPSE_ZONE,
        "training": {"completed": completed, "error": error,
                     "wall_seconds": round(time.time() - started, 2)},
        "health_timeline": records,
        "final_task_panel": final,
        "crossed_collapse_zone_alive": bool(
            final.get("pickups", 0) > 0
            and (final.get("offensive_commitment") or 0) > 0
            and (final.get("defensive_commitment") or 1.0) < 1.0
        ),
        "system_failures": list(ledger.failures),
    }
    (art / "ablation_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"seed {seed}: completed={completed} "
          f"crossed_collapse_zone_alive={report['crossed_collapse_zone_alive']}")
    print(f"report: {art / 'ablation_report.json'}")
    print("=" * 78)
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
