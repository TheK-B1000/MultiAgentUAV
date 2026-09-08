"""G0-v2 failed-action penalty ablation -- one factor, 300k steps, fresh seed.

Tests the collapse diagnosis causally: with ACTION_FAILED_PUNISHMENT removed,
do independent PPO seeds keep attacking instead of drifting to zero pickups
around 225,000 steps?

ONLY ``action_failed_punishment`` differs from the frozen G0-v2 configuration.
Learning rate, entropy schedule, value coefficient, batch shape, every other
reward term, map, ruleset, opponents and horizon are all inherited unchanged
from ``run_g0_v2_seed`` so the comparison stays one-factor.

Every checkpoint records BOTH verdicts:

    SYSTEM_HEALTH -- is PPO numerically alive?
    TASK_HEALTH   -- is the policy still playing CTF?

The baseline collapse passed the first on all 33 checkpoints while failing the
second completely. Reporting them separately is the point.

Gate: artifacts/g0_v2_penalty_ablation/PROBE_GATE_FROZEN.json (frozen first).

Run:  python experiments/run_g0_v2_penalty_ablation.py --seed 2600001
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

ABLATION_SEEDS = (2_600_001, 2_600_002, 2_600_003)
TOTAL_TIMESTEPS = 300_000
CHECKPOINT_INTERVAL = 50_000
COLLAPSE_ZONE = 225_000
ACTION_FAILED_PUNISHMENT_ABLATED = 0.0
DEVICE = "cuda"


def run_tag_for(seed: int) -> str:
    return f"g0_v2_nopen_seed{seed}"


def artifact_dir_for(seed: int) -> Path:
    return PROJECT_ROOT / "artifacts" / "g0_v2_penalty_ablation" / run_tag_for(seed)


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
    cfg.env_action_failed_punishment = ACTION_FAILED_PUNISHMENT_ABLATED

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


def install_probes(ledger: HealthLedger, records: list, art: Path, device: str):
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
    print(f"G0-v2 PENALTY ABLATION  seed={seed}  steps={TOTAL_TIMESTEPS:,}")
    print(f"action_failed_punishment: -0.2 (baseline) -> "
          f"{ACTION_FAILED_PUNISHMENT_ABLATED} (ablated)  [ONLY changed factor]")
    print(f"collapse zone to cross: {COLLAPSE_ZONE:,}  panel every {CHECKPOINT_INTERVAL:,}")
    print(f"map={CANONICAL_MAP} ruleset={RULESET_ID} opponents={OPPONENTS}")
    print("=" * 78)

    from rl.training.orchestrator import orchestrate_training_run

    cfg = build_config(seed)
    ledger = HealthLedger()
    records: list = []
    restore = install_probes(ledger, records, art, DEVICE)
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
        "probe": "G0-v2 failed-action penalty ablation",
        "seed": seed,
        "run_id": run_tag_for(seed),
        "action_failed_punishment": ACTION_FAILED_PUNISHMENT_ABLATED,
        "baseline_action_failed_punishment": -0.2,
        "only_changed_factor": "action_failed_punishment",
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
