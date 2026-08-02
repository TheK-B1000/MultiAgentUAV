"""G0-v2 formal training -- one fresh seed, 1,000,000 steps.

Runs the real production entrypoint with the locked G0-v2 configuration and
records a per-checkpoint health ledger alongside the formal artifact bundle.

Locked configuration (identical across seeds -- nothing here may vary):

    map                  map_a / map_a_open
    ruleset              RULESET_V2_AQUATICUS_10S
    opponents            admitted OP6-OP12 mixture (Gate 1)
    episode horizon      240
    domain randomization off
    initialization       fresh (V1 warm start forbidden)
    tag telemetry        true, through production PPOConfig
    checkpoint cadence   every 100,000 steps
    formal override      forbidden

The preregistered primary policy is the 1,000,000-step checkpoint. Intermediate
checkpoints exist for learning curves only and must not be promoted after the
fact.

Run:  python experiments/run_g0_v2_seed.py --seed 2500001
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from experiments.run_formal_ppo_smoke_2400001 import (  # noqa: E402
    NOT_COMPUTED_SENTINELS,
    HealthLedger,
    _check_event,
    _finalize_event_checks,
    _scan_stats,
    _scan_tensor_mapping,
)

# --- locked G0-v2 contract --------------------------------------------------

G0_SEEDS = (2_500_001, 2_500_002, 2_500_003)
TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_INTERVAL = 100_000
DEVICE = "cuda"
CANONICAL_MAP = "map_a"
RESOLVED_MAP = "map_a_open"
RULESET_ID = "RULESET_V2_AQUATICUS_10S"
EPISODE_HORIZON = 240
AGENTS = 2
OPPONENTS = ("OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12")

N_ENVS = 16
N_STEPS = 128
BATCH_SIZE = 512
N_EPOCHS = 4


def run_tag_for(seed: int) -> str:
    return f"g0_v2_seed{seed}"


def artifact_dir_for(seed: int) -> Path:
    return PROJECT_ROOT / "artifacts" / run_tag_for(seed)


def build_config(seed: int):
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

    # fresh: no warm start of any kind
    cfg.load_path = None
    cfg.additional_timesteps = 0
    cfg.load_weights_only = False

    # production telemetry wiring + formal gate
    cfg.tag_telemetry_enabled = True
    cfg.formal_run = True

    cfg.checkpoint_dir = str(art / "ckpts")
    cfg.metrics_csv_path = str(art / "metrics.csv")
    cfg.episode_csv_path = str(art / "episode_rows.csv")
    cfg.training_telemetry_mode = TrainingTelemetryMode.OFF
    cfg.enable_progress_bar = False
    cfg.verbose_training = True
    return cfg


def install_probes(ledger: HealthLedger, checkpoints: list, art: Path):
    """Observe the real loop; snapshot health at each checkpoint boundary."""
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
                    total_sq = 0.0
                    n = 0
                    for group in opt.param_groups:
                        for p in group["params"]:
                            if p.grad is None:
                                continue
                            if not torch.isfinite(p.grad).all():
                                ledger.fail("non-finite gradient at optimizer step")
                            total_sq += float(p.grad.detach().double().pow(2).sum().item())
                            n += 1
                    if n:
                        norm = math.sqrt(total_sq)
                        ledger.grad_norms.append(norm)
                        if norm <= 0.0:
                            ledger.zero_grad_steps += 1
                    ledger.optimizer_steps += 1
                    probe = None
                    for group in opt.param_groups:
                        for p in group["params"]:
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
        buffer = real_collect(self, *a, **k)
        ledger.rollouts += 1
        fields = getattr(buffer, "fields", None)
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
        return buffer

    def update(self, buffer, *a, **k):
        stats = real_update(self, buffer, *a, **k)
        ledger.updates += 1
        _scan_stats(ledger, dict(stats))
        bad = [n for n, p in self.model.named_parameters()
               if p.is_floating_point() and not torch.isfinite(p).all()]
        if bad:
            ledger.nonfinite_params.extend(bad[:5])
            ledger.fail(f"non-finite model parameters: {bad[:5]}")
        for name in ("primary", "actor", "critic", "router", "actor_cf"):
            opt = getattr(self.optimizers, name, None)
            if opt is None:
                continue
            for pstate in opt.state.values():
                for key, v in pstate.items():
                    if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():
                        ledger.nonfinite_optimizer.append(f"{name}.{key}")
                        ledger.fail(f"non-finite optimizer state: {name}.{key}")
        return stats

    def save(self, path: str):
        real_save(self, path)
        # Per-checkpoint health snapshot + identity/loadability proof.
        entry = {
            "path": str(path),
            "global_step": int(getattr(self, "global_step", 0)),
            "updates": ledger.updates,
            "optimizer_steps": ledger.optimizer_steps,
            "parameter_change_events": ledger.param_change_events,
            "zero_gradient_steps": ledger.zero_grad_steps,
            "grad_norm_mean": (sum(ledger.grad_norms) / len(ledger.grad_norms))
            if ledger.grad_norms else None,
            "nonfinite_any": bool(
                ledger.nonfinite_stats or ledger.nonfinite_buffer
                or ledger.nonfinite_params or ledger.nonfinite_optimizer
            ),
            "tag_success_events": ledger.tag_success,
            "cooldown_denial_events": ledger.tag_denied_cooldown,
            "capture_events": ledger.capture_events,
            "resets_observed": ledger.resets_observed,
            "hard_tag_legality_violations": dict(ledger.legality_violations),
            "event_identity_violations": dict(ledger.identity_violations),
        }
        try:
            from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
            from rl.ruleset_identity import (
                ARTIFACT_IDENTITY_KEY,
                verify_checkpoint_run_identity,
            )

            payload = read_checkpoint_payload(str(path), map_location="cpu")
            verify_checkpoint_run_identity(
                payload, self.run_identity, operation="load", context=str(path)
            )
            ai = payload.get(ARTIFACT_IDENTITY_KEY, {})
            entry["identity_valid"] = True
            entry["roundtrip_loadable"] = True
            entry["ruleset_id"] = ai.get("ruleset_id")
            entry["canonical_map"] = ai.get("canonical_map")
            entry["formal_result_eligible"] = ai.get("formal_result_eligible")
        except Exception as exc:
            entry["identity_valid"] = False
            entry["roundtrip_loadable"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
            ledger.fail(f"checkpoint identity/loadability failed at {path}: {exc}")
        checkpoints.append(entry)
        (art / "checkpoint_health.json").write_text(
            json.dumps(checkpoints, indent=2, default=str), encoding="utf-8"
        )
        print(f"[G0V2] checkpoint health recorded: step={entry['global_step']} "
              f"identity_valid={entry['identity_valid']} "
              f"tags={entry['tag_success_events']} denials={entry['cooldown_denial_events']}")

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
    ap.add_argument("--seed", type=int, required=True, choices=list(G0_SEEDS))
    ap.add_argument("--threads", type=int, default=4,
                    help="torch intra-op threads (keeps 3 parallel seeds from oversubscribing)")
    args = ap.parse_args()

    torch.set_num_threads(max(1, int(args.threads)))
    seed = int(args.seed)
    art = artifact_dir_for(seed)
    art.mkdir(parents=True, exist_ok=True)
    (art / "ckpts").mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("FATAL: CUDA required but unavailable.")
        return 2

    print("=" * 78)
    print(f"G0-v2  seed={seed}  run_id={run_tag_for(seed)}  steps={TOTAL_TIMESTEPS:,}")
    print(f"map={CANONICAL_MAP}->{RESOLVED_MAP}  ruleset={RULESET_ID}")
    print(f"opponents={OPPONENTS}  horizon={EPISODE_HORIZON}  ckpt_every={CHECKPOINT_INTERVAL:,}")
    print(f"artifact_dir={art}")
    print("=" * 78)

    from rl.training.orchestrator import orchestrate_training_run

    cfg = build_config(seed)
    ledger = HealthLedger()
    checkpoints: list = []
    restore = install_probes(ledger, checkpoints, art)
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

    if completed:
        if ledger.optimizer_steps == 0:
            ledger.fail("no optimizer steps were taken")
        if ledger.param_change_events == 0:
            ledger.fail("optimizer never changed a parameter")
        if not ledger.grad_norms or max(ledger.grad_norms) <= 0.0:
            ledger.fail("gradients were all zero")
        if ledger.tag_success == 0:
            ledger.fail("no tag_success events occurred")
        if ledger.tag_denied_cooldown == 0:
            ledger.fail("no cooldown-denial events occurred")

    grads = ledger.grad_norms
    report = {
        "verdict": "PASS" if (completed and not ledger.failures) else "FAIL",
        "seed": seed,
        "run_id": run_tag_for(seed),
        "artifact_dir": str(art),
        "primary_checkpoint": str(art / "ckpts" / f"final_{run_tag_for(seed)}.zip"),
        "preregistered_primary_step": TOTAL_TIMESTEPS,
        "configuration": {
            "total_timesteps_requested": TOTAL_TIMESTEPS,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "device": DEVICE,
            "canonical_map": CANONICAL_MAP,
            "resolved_map": RESOLVED_MAP,
            "ruleset_id": RULESET_ID,
            "opponents": list(OPPONENTS),
            "episode_horizon": EPISODE_HORIZON,
            "domain_randomization": False,
            "initialization": "fresh",
            "v1_warm_start": "forbidden",
            "formal_override": "forbidden",
            "tag_telemetry_enabled": True,
            "formal_run": True,
            "n_envs": N_ENVS, "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE, "n_epochs": N_EPOCHS,
        },
        "training": {"completed": completed, "error": error,
                     "wall_seconds": round(time.time() - started, 2)},
        "runtime_health": {
            "updates": ledger.updates,
            "rollouts": ledger.rollouts,
            "optimizer_steps": ledger.optimizer_steps,
            "parameter_change_events": ledger.param_change_events,
            "zero_gradient_steps": ledger.zero_grad_steps,
            "grad_norm_min": min(grads) if grads else None,
            "grad_norm_max": max(grads) if grads else None,
            "grad_norm_mean": (sum(grads) / len(grads)) if grads else None,
            "nonfinite_training_stats": ledger.nonfinite_stats[:10],
            "nonfinite_rollout_fields": ledger.nonfinite_buffer[:10],
            "nonfinite_parameters": ledger.nonfinite_params[:10],
            "nonfinite_optimizer_state": ledger.nonfinite_optimizer[:10],
            "inactive_diagnostics_not_computed": sorted(ledger.inactive_diagnostics),
            "events_observed": ledger.events_seen,
            "tag_success_events": ledger.tag_success,
            "cooldown_denial_events": ledger.tag_denied_cooldown,
            "capture_events": ledger.capture_events,
            "resets_observed": ledger.resets_observed,
            "hard_tag_legality_violations": dict(ledger.legality_violations),
            "event_identity_violations": dict(ledger.identity_violations),
            "parallel_envs": len(ledger._per_env_episode_keys),
        },
        "checkpoints": checkpoints,
        "failures": list(ledger.failures),
    }
    out = art / "g0_v2_training_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"G0-v2 seed {seed}: {report['verdict']}  ({report['training']['wall_seconds']}s)")
    print(f"report: {out}")
    for f in report["failures"]:
        print(f"  - {f}")
    print("=" * 78)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
