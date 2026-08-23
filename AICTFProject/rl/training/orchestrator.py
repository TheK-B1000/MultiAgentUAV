"""Training orchestration: coordinates the full PPO training run lifecycle.

:func:`orchestrate_training_run` replaces the body of
:func:`rl.train_ppo.train_ppo` — it owns the sequence of steps from config
validation through trainer teardown.  The original ``train_ppo`` function
delegates to this function for backward compatibility.

Responsibilities (in order):
1. Config validation gates (evaluation-only preset, gate_open ablation check)
2. Global seed + checkpoint directory creation
3. Resolved training config derivation
4. CSV path resolution + telemetry rotation
5. Training banner printing + run-lock acquisition + run-config JSON sidecar
6. Runtime clamping (team-size, CUDA fallback)
7. Environment construction
8. Trainer construction, checkpoint loading, timestep extension
9. ``trainer.learn`` loop (with ``KeyboardInterrupt`` emergency save)
10. Final checkpoint save + stats print
11. Teardown (telemetry, env, lock) in a ``finally`` block

Dependency direction: imports from ``rl.training.*`` sub-modules only,
never from ``rl.train_ppo`` or ``rl.training.cli``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

from rl.config.ppo_config import PPOConfig
from rl.training.banner import print_episode_stats_banner, print_training_banner
from rl.training.config_validation import normalize_and_validate_training_config
from rl.training.errors import EvaluationOnlyPresetError
from rl.training.factories import build_training_env
from rl.training.initialization import (
    build_trainer,
    maybe_extend_total_timesteps,
    maybe_configure_periodic_checkpoints,
    maybe_load_checkpoint,
)
from rl.training.lifecycle import (
    _clamp_runtime_config_for_team_size,
    _ensure_cuda_or_fallback,
    _resolve_metrics_csv_paths,
    _rotate_fresh_run_telemetry,
    set_global_seed,
    teardown_training,
)
from rl.training.resolved_config import resolve_training_config
from rl.training.run_artifacts import (
    _acquire_run_lock,
    write_evaluation_manifest_json,
    write_result_summary_json,
    write_startup_formal_artifacts,
)
from rl.training.run_context import RunContext


# ---------------------------------------------------------------------------
# Config validation gates
# ---------------------------------------------------------------------------

def _validate_config_gates(cfg: PPOConfig) -> None:
    """Raise typed errors for configs that must not start a training run.

    Three gates are checked in order:

    1. **Evaluation-only preset**: some presets (e.g. v6i2 promoted eval configs)
       set ``evaluation_only_preset=True`` to prevent accidental PPO training.
    2. **gate_open ablation**: the v6i5 CF-sweep gate-open ablation run must
       match the prior 8x sweep config exactly (modulo a small allowed-diff set)
       so the ablation comparison is apples-to-apples.
    """
    _validate_exp2_config_gates(cfg)
    # A formal run must carry the evidence needed to audit its own tagging.
    # Checked at start, not at report time: discovering after a million steps
    # that the tag ledger was never recorded costs the whole run.
    if bool(getattr(cfg, "formal_run", False)) and not bool(
        getattr(cfg, "tag_telemetry_enabled", False)
    ):
        raise ValueError(
            f"Run {getattr(cfg, 'run_tag', 'unknown')!r} sets formal_run=True but "
            "tag_telemetry_enabled=False. A formal run must record tag successes, "
            "cooldown denials and event identities. Set tag_telemetry_enabled=True."
        )

    if bool(getattr(cfg, "evaluation_only_preset", False)):
        runner = str(getattr(cfg, "evaluation_only_runner", "") or "the evaluation runner")
        raise EvaluationOnlyPresetError(
            f"Preset {getattr(cfg, 'cli_preset', getattr(cfg, 'run_tag', 'unknown'))!r} "
            "is evaluation-only and must not start PPO training. "
            f"Use {runner} with a promoted v6i2 checkpoint."
        )

    if cfg.run_tag and "gate_open" in cfg.run_tag:
        if cfg.latent_cf_require_competence:
            raise ValueError(
                f"Ablation run {cfg.run_tag} requires --no-latent-cf-require-competence "
                f"but resolved latent_cf_require_competence is True!"
            )
        prior_config_path = os.path.join("checkpoints", "4v4_diag", "v6i5_cf_sweep_8x_150k_4v4_run_config.json")
        if not os.path.exists(prior_config_path):
            raise FileNotFoundError(f"Prior run config not found: {prior_config_path}")
        with open(prior_config_path, "r", encoding="utf-8") as f:
            prior_data = json.load(f)
        prior_resolved = prior_data.get("resolved_ppo_config", {})
        allowed_diffs = {
            "run_tag", "metrics_csv_path", "episode_csv_path",
            "latent_cf_require_competence", "checkpoint_dir",
            "utc_timestamp"
        }
        current_dict = dataclasses.asdict(cfg)
        mismatches = []
        for key, prior_val in prior_resolved.items():
            if key in allowed_diffs:
                continue
            if key not in current_dict:
                continue
            curr_val = current_dict[key]
            if isinstance(prior_val, list):
                prior_val = tuple(prior_val)
            if isinstance(curr_val, list):
                curr_val = tuple(curr_val)
            if curr_val != prior_val:
                mismatches.append(f"{key}: prior={prior_val}, current={curr_val}")
        if mismatches:
            raise ValueError(
                "Configuration mismatch vs prior 8x sweep config:\n" + "\n".join(mismatches)
            )


def _validate_exp2_config_gates(cfg: PPOConfig) -> None:
    """Fail closed on any drift from the frozen EXP2 treatment."""
    if not bool(getattr(cfg, "exp2_teacher_compression_enabled", False)):
        return
    errors: list[str] = []
    if not bool(getattr(cfg, "use_latent_strategy", False)) or int(cfg.latent_k) != 2:
        errors.append("student must use latent strategy with K=2")
    if bool(getattr(cfg, "latent_strategy_encoder_enabled", True)):
        errors.append("q_phi/router must be structurally disabled")
    if str(getattr(cfg, "latent_assignment_mode", "")) != "static_env":
        errors.append("latent_assignment_mode must be static_env")
    ids = tuple(int(v) for v in getattr(cfg, "forced_latent_env_ids", ()))
    if len(ids) != int(cfg.n_envs) or ids.count(0) != int(cfg.n_envs) // 2 or ids.count(1) != int(cfg.n_envs) // 2:
        errors.append("forced_latent_env_ids must contain exactly half z0 and half z1")
    if abs(float(cfg.exp2_teacher_lambda) - 0.10) > 1e-12:
        errors.append("teacher lambda must equal 0.10")
    if int(cfg.exp2_teacher_cadence) != 4 or int(cfg.exp2_teacher_batch_size) != 64:
        errors.append("teacher cadence/batch must equal 4/64")
    if str(getattr(cfg, "sappo_anchor_dataset", "") or ""):
        errors.append("SAPPO offline anchor path must be absent")
    if len(tuple(cfg.exp2_teacher_checkpoints)) != 2 or len(tuple(cfg.exp2_teacher_sha256)) != 2:
        errors.append("exactly two teacher checkpoints and hashes are required")
    if any(float(getattr(cfg, name, 0.0) or 0.0) != 0.0 for name in (
        "latent_strategy_ppo_coef", "latent_lam_h", "latent_lam_p",
        "latent_kl_consecutive", "latent_episode_strategy_coef",
        "latent_actor_z_separation_coef", "latent_behavior_contrast_coef",
    )):
        errors.append("router/diversity/separation objectives must all be zero")
    protocol = Path(str(getattr(cfg, "exp2_protocol_path", "") or ""))
    if not protocol.is_file():
        errors.append(f"frozen protocol missing: {protocol}")
    else:
        try:
            payload = json.loads(protocol.read_text(encoding="utf-8"))
            if payload.get("protocol_id") not in {
                "EXP2_K2_LATENT_COMPRESSION_V1",
                "EXP2B_SPECIALIZATION_PRESERVING_LATENT_COMPRESSION_V1",
                "EXP2C_MODE_SPECIFIC_ACTOR_COMPRESSION_V1",
            }:
                errors.append("unexpected EXP2 protocol_id")
            if payload.get("status") != "FROZEN_BEFORE_IMPLEMENTATION_OR_TRAINING":
                errors.append("EXP2 protocol is not in the frozen pretraining state")
        except Exception as exc:
            errors.append(f"EXP2 protocol unreadable: {exc}")
    if errors:
        raise RuntimeError("EXP2 frozen-config gate failed: " + "; ".join(errors))

# ---------------------------------------------------------------------------
# Main orchestration entry point
# ---------------------------------------------------------------------------

def orchestrate_training_run(
    cfg: Optional[PPOConfig] = None,
    *,
    pre_rollout_env_setup: Optional[
        Callable[[Any, PPOConfig], Optional[dict[str, Any]]]
    ] = None,
) -> None:
    """Run the full local PPO/MAPPO training path.

    This is the canonical implementation extracted from
    :func:`rl.train_ppo.train_ppo`.  The original function delegates here so
    existing callers (scripts, presets, tests) keep working without changes.
    """
    cfg = normalize_and_validate_training_config(cfg or PPOConfig())
    _validate_config_gates(cfg)

    set_global_seed(cfg.seed, torch_seed=True, deterministic=cfg.use_deterministic)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    resolved = resolve_training_config(cfg)

    # Resolve CSV paths before banner so printed paths match trainer write targets.
    _resolve_metrics_csv_paths(cfg)

    print_training_banner(
        cfg,
        curriculum=resolved.curriculum,
        max_agents=resolved.max_agents,
        team_size=resolved.team_size,
    )

    run_lock = _acquire_run_lock(cfg)
    _rotate_fresh_run_telemetry(cfg)

    print_episode_stats_banner(
        cfg,
        curriculum=resolved.curriculum,
        initial_opponent_tag=resolved.initial_opponent_tag,
    )

    _clamp_runtime_config_for_team_size(cfg, resolved.max_agents)
    _ensure_cuda_or_fallback(cfg)

    # The environment is built BEFORE any artifact is written. Run identity must
    # be resolved from the LIVE environment, never from config defaults -- five
    # artifacts reconstructing "V2" independently is exactly how they end up
    # carrying five subtly different versions of it. Consequently
    # ``write_run_config_json`` now runs after this point, not before.
    env = build_training_env(
        cfg,
        initial_phase=resolved.initial_phase,
        initial_opponent_tag=resolved.initial_opponent_tag,
    )

    # Experiment-specific live-environment plumbing belongs at this one seam:
    # after construction, before identity/artifacts/trainer/rollout. The callback
    # may return fields that are stamped into training_manifest.json. Any
    # exception is fatal and therefore consumes zero training steps.
    training_manifest_extra = None
    if pre_rollout_env_setup is not None:
        try:
            training_manifest_extra = pre_rollout_env_setup(env, cfg)
        except BaseException:
            teardown_training(cfg, None, env, run_lock)
            raise

    from rl.ruleset_identity import RunIdentityError, build_formal_run_identity

    # Mandatory: a run that cannot resolve its identity fails here, before the
    # first rollout step, rather than producing unstampable artifacts.
    run_identity = build_formal_run_identity(env, run_id=str(cfg.run_tag))
    print(f"[PPO] Run identity: {run_identity.ruleset_id} "
          f"map={run_identity.canonical_map} "
          f"fingerprint={run_identity.ruleset_fingerprint[:12]} "
          f"formal_eligible={run_identity.formal_result_eligible}")

    # Both startup artifacts must be written from the same frozen object before
    # any rollout. Soft-failing here would recreate the unstamped-traveler bug.
    try:
        startup_paths = write_startup_formal_artifacts(
            cfg,
            run_identity=run_identity,
            training_manifest_extra=training_manifest_extra,
        )
    except Exception as exc:
        raise RunIdentityError(
            f"Failed to write startup formal artifacts before rollout: {exc}"
        ) from exc
    rc_path = startup_paths["run_config"]
    tm_path = startup_paths["training_manifest"]
    print(f"[PPO] Run config written: {rc_path}")
    print(f"[PPO] Training manifest written: {tm_path}")

    run_context = RunContext(
        run_lock=run_lock,
        run_identity=run_identity,
        rc_path=rc_path,
        training_manifest_path=tm_path,
    )

    trainer = None
    try:
        trainer = build_trainer(env, cfg, resolved, run_identity=run_identity)
        if getattr(trainer, "run_identity", None) is None:
            raise RunIdentityError(
                "Trainer was constructed without run_identity; refusing to "
                "start the first rollout step."
            )
        maybe_load_checkpoint(cfg, trainer)
        maybe_extend_total_timesteps(cfg, trainer)
        maybe_configure_periodic_checkpoints(cfg, trainer)
        _maybe_attach_sappo_anchor(cfg, trainer)
        _maybe_attach_exp2_teacher_compression(cfg, trainer)

        artifact_only = bool(getattr(cfg, "formal_artifact_bundle_only", False))
        if artifact_only:
            _write_formal_artifact_bundle_smoke(cfg, trainer, run_identity)
        else:
            try:
                stats = trainer.learn(total_timesteps=int(cfg.total_timesteps))
            except KeyboardInterrupt:
                interrupt_path = os.path.join(
                    cfg.checkpoint_dir,
                    f"interrupt_{cfg.run_tag}_{int(getattr(trainer, 'global_step', 0))}.zip",
                )
                trainer.save(interrupt_path)
                print(f"[PPO] KeyboardInterrupt: emergency checkpoint saved to: {interrupt_path}")
                raise

            final_path = os.path.join(cfg.checkpoint_dir, f"final_{cfg.run_tag}.zip")
            trainer.save(final_path)
            if stats:
                print(
                    "[PPO] Final stats: "
                    f"policy_loss={stats.get('policy_loss', 0.0):.4f}, "
                    f"value_loss={stats.get('value_loss', 0.0):.4f}, "
                    f"approx_kl={stats.get('approx_kl', 0.0):.5f}"
                )
            print(f"[PPO] Training complete. Final checkpoint saved to: {final_path}")
            _write_in_run_eval_and_summary(cfg, trainer, run_identity, checkpoint_path=final_path)
    finally:
        teardown_training(cfg, trainer, env, run_context.run_lock)


def _checkpoint_file_fingerprint(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_episode_rows(path: Optional[str]) -> list[dict]:
    import csv

    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_in_run_eval_and_summary(cfg, trainer, run_identity, *, checkpoint_path: str) -> None:
    """In-training evaluation/summary use the same frozen run_identity."""
    base = cfg.checkpoint_dir
    if getattr(cfg, "metrics_csv_path", None):
        d = os.path.dirname(str(cfg.metrics_csv_path))
        if d:
            base = d
    os.makedirs(base, exist_ok=True)

    ckpt_fp = _checkpoint_file_fingerprint(checkpoint_path) if os.path.isfile(checkpoint_path) else ""
    eval_path = os.path.join(base, "evaluation_manifest.json")
    # In-training evaluation: the checkpoint was produced by THIS run, so the
    # shared run id is a declared fact. Stated through the named constructor
    # rather than by omitting arguments and letting the writer infer it.
    from rl.ruleset_identity import VerifiedCheckpointLineage

    write_evaluation_manifest_json(
        eval_path,
        run_identity=run_identity,
        evaluation_run_id=run_identity.run_id,
        lineage=VerifiedCheckpointLineage.for_in_training_evaluation(
            run_identity, ckpt_fp),
        extra={"scope": "in_training"},
    )

    rows = _read_episode_rows(getattr(cfg, "episode_csv_path", None))
    if not rows:
        # Training may finish without completing an episode on tiny budgets.
        # Stamp a single completion marker so the formal bundle stays closed.
        from rl.ruleset_identity import stamp_csv_row

        rows = [stamp_csv_row({"episode_id": 0, "success": 0, "source": "completion_marker"}, run_identity)]
        ep_path = getattr(cfg, "episode_csv_path", None) or os.path.join(base, "episode_rows.csv")
        import csv
        from rl.ruleset_identity import CSV_IDENTITY_FIELDS

        fieldnames = list(dict.fromkeys(["episode_id", "success", "source", *CSV_IDENTITY_FIELDS]))
        with open(ep_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        cfg.episode_csv_path = ep_path

    summary_path = os.path.join(base, "result_summary.json")
    write_result_summary_json(
        summary_path,
        {
            "verdict": "TRAINING_COMPLETE",
            "total_timesteps": int(getattr(cfg, "total_timesteps", 0) or 0),
            "global_step": int(getattr(trainer, "global_step", 0) or 0),
            "checkpoint_path": checkpoint_path,
            "n_episode_rows": len(rows),
        },
        run_identity=run_identity,
        source_rows=rows,
    )
    print(f"[PPO] Evaluation manifest written: {eval_path}")
    print(f"[PPO] Result summary written: {summary_path}")


def _write_formal_artifact_bundle_smoke(cfg, trainer, run_identity) -> None:
    """Artifact-only production path for the formal-bundle integration test."""
    from rl.ruleset_identity import CSV_IDENTITY_FIELDS, stamp_csv_row

    base = cfg.checkpoint_dir
    if getattr(cfg, "metrics_csv_path", None):
        d = os.path.dirname(str(cfg.metrics_csv_path))
        if d:
            base = d
    os.makedirs(base, exist_ok=True)

    ep_path = getattr(cfg, "episode_csv_path", None) or os.path.join(base, "episode_rows.csv")
    row = stamp_csv_row(
        {
            "episode_id": 0,
            "success": 0,
            "blue_score": 0,
            "red_score": 0,
            "source": "formal_artifact_bundle_only",
        },
        run_identity,
    )
    import csv

    fieldnames = list(dict.fromkeys([*row.keys(), *CSV_IDENTITY_FIELDS]))
    with open(ep_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(row)
    cfg.episode_csv_path = ep_path

    ckpt_path = os.path.join(cfg.checkpoint_dir, f"final_{cfg.run_tag}.zip")
    trainer.save(ckpt_path)
    _write_in_run_eval_and_summary(cfg, trainer, run_identity, checkpoint_path=ckpt_path)
    print(f"[PPO] Formal artifact-bundle-only smoke complete: {base}")


def _maybe_attach_sappo_anchor(cfg, trainer) -> None:
    """SAPPO V1: attach interleaved teacher rehearsal, or do nothing at all.

    Attached AFTER checkpoint load so rehearsal targets the resumed weights, and
    BEFORE learn() so the very first minibatch group is counted.

    With no dataset configured this function returns without constructing
    anything, so the PPO path is untouched by construction rather than by a
    zero-scaled term. See SAPPO_V1_LOSS_SEMANTICS_AMENDMENT.json.
    """
    path = str(getattr(cfg, "sappo_anchor_dataset", "") or "")
    if not path:
        return
    from rl.custom_ppo.strategy_anchor import AnchorDataset, AnchorRunner

    ds = AnchorDataset(path, batch_size=int(getattr(cfg, "sappo_anchor_batch_size", 64)),
                       seed=int(getattr(cfg, "seed", 7) or 7))
    runner = AnchorRunner(
        trainer.model,
        trainer.optimizer,
        ds,
        lambda_anchor=float(cfg.sappo_anchor_lambda),
        cadence=int(cfg.sappo_anchor_cadence),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5)),
        device=str(getattr(cfg, "device", "cpu")),
    )
    trainer.sappo_anchor_runner = runner
    print(f"[SAPPO] anchor rehearsal ATTACHED: lambda={runner.lambda_anchor} "
          f"cadence=1:{runner.cadence} dataset={ds.describe()}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_attach_exp2_teacher_compression(cfg, trainer) -> None:
    """Attach frozen online teachers after checkpoint load and before learn()."""
    if not bool(getattr(cfg, "exp2_teacher_compression_enabled", False)):
        return
    if getattr(trainer, "sappo_anchor_runner", None) is not None:
        raise RuntimeError("EXP2 cannot coexist with the SAPPO offline anchor runner")
    model = trainer.model
    if int(getattr(model, "latent_k", 0)) != 2:
        raise RuntimeError("EXP2 student model did not resolve K=2")
    if getattr(model, "strategy_encoder", None) is not None:
        raise RuntimeError("EXP2 student unexpectedly constructed q_phi")

    checkpoints = tuple(Path(str(p)) for p in cfg.exp2_teacher_checkpoints)
    expected_hashes = tuple(str(v).lower() for v in cfg.exp2_teacher_sha256)
    for path, expected in zip(checkpoints, expected_hashes):
        if not path.is_file():
            raise RuntimeError(f"EXP2 teacher checkpoint missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"EXP2 teacher hash mismatch for {path}: {actual} != {expected}"
            )

    from rl.custom_ppo.exp2_teacher_compression import Exp2TeacherCompressionRunner
    from rl.custom_ppo.inference import load_custom_ppo_policy

    teachers = {}
    for z, path in enumerate(checkpoints):
        loaded = load_custom_ppo_policy(
            str(path), trainer.env.observation_space, trainer.env.action_space,
            device=str(trainer.device),
        )
        teacher = loaded.model
        if bool(getattr(teacher, "uses_latent_strategy", False)):
            raise RuntimeError(f"EXP2 teacher z={z} must be a non-latent SAPPO policy")
        if tuple(teacher.action_dims) != tuple(model.action_dims):
            raise RuntimeError(f"EXP2 teacher z={z} action space differs from student")
        teachers[z] = teacher

    protocol_payload = json.loads(Path(str(cfg.exp2_protocol_path)).read_text(encoding="utf-8"))
    protocol_id = protocol_payload.get("protocol_id")
    is_exp2b = protocol_id == "EXP2B_SPECIALIZATION_PRESERVING_LATENT_COMPRESSION_V1"
    is_exp2c = protocol_id == "EXP2C_MODE_SPECIFIC_ACTOR_COMPRESSION_V1"
    if is_exp2c:
        actor = model.latent_actor
        heads = getattr(actor, "latent_action_heads", None)
        if heads is None or len(heads) != 2:
            raise RuntimeError("EXP2C requires exactly two mode-specific final actor heads")
        if getattr(actor, "latent_adapters", None) is not None:
            raise RuntimeError("EXP2C forbids latent residual adapters")
        if getattr(actor, "latent_branch_trunks", None) is not None:
            raise RuntimeError("EXP2C forbids private deep actor trunks")
        if not bool(getattr(actor, "exp2c_mode_specific_action_heads", False)):
            raise RuntimeError("EXP2C private-head flag did not reach the live actor")
    runner = Exp2TeacherCompressionRunner(
        model,
        trainer.optimizer,
        teachers,
        lambda_teacher=float(cfg.exp2_teacher_lambda),
        cadence=int(cfg.exp2_teacher_cadence),
        batch_size=int(cfg.exp2_teacher_batch_size),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5)),
        seed=int(getattr(cfg, "seed", 0)) + 92_011,
        device=str(trainer.device),
        cell_counts=(16, 0, 0, 16) if (is_exp2b or is_exp2c) else (8, 8, 8, 8),
        gradient_cosine_enabled=is_exp2b or is_exp2c,
        clip_range=float(getattr(cfg, "clip_range", 0.2)),
    )
    pending = trainer.updater.consume_pending_exp2_teacher_state()
    if cfg.load_path and pending is None:
        raise RuntimeError(
            "EXP2 resume checkpoint has no teacher-runner state; refusing a "
            "cadence-reset resume"
        )
    if pending is not None:
        runner.load_state_dict(pending)
    trainer.exp2_teacher_compression_runner = runner
    print(
        f"[{'EXP2C' if is_exp2c else ('EXP2B' if is_exp2b else 'EXP2')}] online teacher KL ATTACHED: "
        f"lambda={runner.lambda_teacher} cadence=1:{runner.cadence} "
        f"batch={runner.batch_size} mapping=z0:pi_A,z1:pi_B q_phi=ABSENT "
        f"cells={runner.cell_counts} grad_cosine={runner.gradient_cosine_enabled}"
    )
