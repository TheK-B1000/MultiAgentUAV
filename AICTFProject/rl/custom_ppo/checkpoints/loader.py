from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import os
import time
import warnings

import torch

from rl.custom_ppo.policy import SharedActorCentralizedCritic
from rl.custom_ppo.inference_policy import CustomPPOInferencePolicy

from .archive import _torch_load_checkpoint, read_checkpoint_payload
from .errors import CheckpointModelConstructionError
from .hashing import describe_checkpoint
from .metadata import assert_compatible_global_state_dim, canonicalize_latent_strategy_cfg, parse_checkpoint_metadata
from .models import CheckpointLoadReport, CheckpointLoadRequest, LoadedCheckpoint, PolicyArchitecture
from .state_dict import _load_model_state_dict_compat

STRATEGY_GENERATOR_SEED_OFFSET = 0x1_0000_00D
ACTION_GENERATOR_SEED_OFFSET = 0x2_0000_02B

def _model_kwargs_from_cfg(cfg: Any) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    cfg = canonicalize_latent_strategy_cfg(cfg)
    experiment_id = str(cfg.get("experiment_id", "") or "").lower()
    router_context_mode = str(cfg.get("router_context_mode", "") or "")
    router_current_plus_delta = router_context_mode == "current_plus_delta"
    v6_staged = (
        bool(cfg.get("use_v6i1_curriculum", False))
        and str(cfg.get("training_mode", "default") or "default") == "staged_team_intent_curriculum"
        and str(cfg.get("experiment_family", "v6") or "v6") == "v6"
        and experiment_id in {"v6i1", "v6i2", "v6i3", "v6i5"}
    )
    kwargs: dict[str, Any] = {
        "actor_cnn_feature_dim": int(cfg.get("actor_cnn_feature_dim", 128)),
    }
    if bool(cfg.get("use_latent_strategy", False)):
        kwargs.update(
            {
                "latent_k": int(cfg.get("latent_k", 4)),
                "strategy_encoder_enabled": bool(
                    cfg.get("latent_strategy_encoder_enabled", True)
                ),
                "z_embed_dim": int(cfg.get("latent_z_embed_dim", 16)),
                "router_context_mode": router_context_mode,
                "router_context_dimension": int(cfg.get("router_context_dimension", 0) or 0),
                "strategy_hidden_dim": int(cfg.get("latent_strategy_hidden", 128)),
                "critic_hidden_dim": int(cfg.get("latent_vf_hidden", 128)),
                "use_strategy_aux_return_head": bool(
                    cfg.get("latent_strategy_aux_return_head", False)
                ),
                # Mirror the trainer-side gating in
                # ``rl/custom_ppo/trainer_config.py``: the
                # ``episode_strategy_value_head`` module is built when EITHER
                # episode-level strategy PPO is on, OR arc-credit is enabled
                # with the ``context_value`` baseline (which reuses this head
                # as V_phi(ctx, z)). v3i19+ runs (incl. v4i1) trip the arc-
                # credit branch; before this fix the loader was missing it
                # and rejected those checkpoints with ``unexpected=[
                # 'episode_strategy_value_head.*']``.
                "use_episode_strategy_value_head": bool(
                    cfg.get("latent_episode_strategy_ppo", False)
                    or v6_staged
                    or (
                        cfg.get("latent_arc_credit_enabled", False)
                        and str(
                            cfg.get("latent_arc_credit_baseline", "context_value")
                            or "context_value"
                        ).lower() == "context_value"
                    )
                ),
                "recurrent_selector_hidden_dim": int(
                    cfg.get("recurrent_selector_hidden_dim", None)
                    or cfg.get("v6i1_recurrent_selector_hidden", 32)
                    or 32
                ),
                "use_recurrent_selector": bool(
                    (v6_staged and not router_current_plus_delta)
                    or bool(cfg.get("use_recurrent_selector", False))
                    or (
                        router_context_mode == "current"
                        and int(cfg.get("recurrent_selector_hidden_dim", 0) or 0) > 0
                    )
                ),
                "strategy_tau": float(cfg.get("latent_strategy_tau", 1.0) or 1.0),
                "latent_actor_z_onehot_enabled": bool(
                    cfg.get("latent_actor_z_onehot_enabled", False)
                ),
                "latent_actor_z_onehot_scale": float(
                    1.0
                    if cfg.get("latent_actor_z_onehot_scale", 1.0) is None
                    else cfg.get("latent_actor_z_onehot_scale", 1.0)
                ),
                "latent_actor_z_embed_scale": float(
                    1.0
                    if cfg.get("latent_actor_z_embed_scale", 1.0) is None
                    else cfg.get("latent_actor_z_embed_scale", 1.0)
                ),
                "latent_actor_z_adapter_enabled": bool(
                    cfg.get("latent_actor_z_adapter_enabled", False)
                ),
                "latent_actor_z_adapter_scale": float(
                    cfg.get("latent_actor_z_adapter_scale", 0.0) or 0.0
                ),
                "latent_actor_z_adapter_init_std": float(
                    cfg.get("latent_actor_z_adapter_init_std", 0.02) or 0.02
                ),
                "latent_actor_z_film_layers": int(
                    cfg.get("latent_actor_z_film_layers", 1) or 1
                ),
                "enable_actor_z_film": bool(
                    cfg.get("enable_actor_z_film", False)
                ),
                "actor_z_film_init_scale": float(
                    cfg.get("actor_z_film_init_scale", 0.0) or 0.0
                ),
                "actor_z_film_layer": int(
                    cfg.get("actor_z_film_layer", 2) or 2
                ),
                # V6I7: per-latent residual adapters.
                "enable_latent_z_residual": bool(
                    cfg.get("enable_latent_z_residual", False)
                ),
                "latent_z_gate_init": float(
                    cfg.get("latent_z_gate_init", 0.01) or 0.01
                ),
                # V6I22E: fixed-alpha gate-free adapters.
                "latent_z_residual_alpha": float(
                    cfg.get("latent_z_residual_alpha", 0.0) or 0.0
                ),
                # V6I23: population-birth specialist heads / active-z residual.
                "latent_population_birth_active_z_only": bool(
                    cfg.get("latent_population_birth_active_z_only", False)
                ),
                "latent_population_birth_per_z_action_heads": bool(
                    cfg.get("latent_population_birth_per_z_action_heads", False)
                ),
                # V6I26 LRO: deep per-z trunks (last two MLP layers).
                "latent_lro_deep_branches": bool(
                    cfg.get("latent_lro_deep_branches", False)
                ),
            }
        )
    return kwargs
def apply_deterministic_sampling_generators(
    model: SharedActorCentralizedCritic,
    seed: int,
    *,
    device: torch.device | str,
) -> None:
    """Attach separate :class:`torch.Generator` copies for team-strategy vs per-head action sampling."""
    dev = torch.device(device)
    g_s = torch.Generator(device=dev)
    g_s.manual_seed((int(seed) + STRATEGY_GENERATOR_SEED_OFFSET) & 0xFFFF_FFFF)
    g_a = torch.Generator(device=dev)
    g_a.manual_seed((int(seed) + ACTION_GENERATOR_SEED_OFFSET) & 0xFFFF_FFFF)
    model.set_sampling_generators(strategy=g_s, action=g_a)


def _architecture_from_metadata(metadata, observation_space, action_space) -> PolicyArchitecture:
    dims = list(getattr(action_space, "nvec", []))
    grid_shape = observation_space.spaces["grid"].shape
    return PolicyArchitecture(
        observation_channels=int(grid_shape[1]),
        n_agents=int(grid_shape[0]),
        n_macros=int(dims[0]) if dims else 0,
        n_targets=int(dims[1]) if len(dims) > 1 else 0,
        latent_count=metadata.latent_count,
        model_kwargs=_model_kwargs_from_cfg(metadata.cfg),
    )


def _resolve_live_env_cfg(trainer) -> Any:
    from rl.ruleset_identity import RunIdentityError

    for attr in ("env", "vec_env", "_env"):
        env = getattr(trainer, attr, None)
        if env is None:
            continue
        core = getattr(env, "core", None) or getattr(getattr(env, "vec", None), "core", None)
        cfg = getattr(core, "cfg", None) if core is not None else None
        if cfg is not None:
            return cfg
        cfg = getattr(env, "cfg", None)
        if cfg is not None:
            return cfg
    raise RunIdentityError(
        "Cannot resolve live environment config for checkpoint identity check."
    )


def _env_ruleset_fingerprint(trainer) -> dict:
    """Tagging-ruleset fingerprint of the trainer's environment.

    FAILS CLOSED. Returning ``{}`` here would be safe at load time (the
    checkpoint classifies LEGACY_UNKNOWN and is rejected), but it would waste
    an entire training run producing checkpoints that later reject themselves.
    A formal run therefore refuses to write an unstamped checkpoint.

    Set ``trainer.allow_unstamped_checkpoint = True`` for explicitly labelled
    diagnostic runs; those checkpoints save as LEGACY_UNKNOWN and are not
    eligible as formal results.
    """
    from rl.ruleset_identity import (RULESET_FIELDS, RulesetFingerprintError,
                                     fingerprint)

    err: Exception | None = None
    for attr in ("env", "vec_env", "_env"):
        env = getattr(trainer, attr, None)
        if env is None:
            continue
        try:
            core = getattr(env, "core", None) or getattr(getattr(env, "vec", None), "core", None)
            cfg = getattr(core, "cfg", None) if core is not None else None
            if cfg is None:
                continue
            fp = dict(fingerprint(cfg))
            if all(k in fp for k in RULESET_FIELDS):
                return fp
        except Exception as exc:  # keep probing the remaining attributes
            err = exc

    if bool(getattr(trainer, "allow_unstamped_checkpoint", False)):
        warnings.warn(
            "Writing an UNSTAMPED checkpoint (ruleset LEGACY_UNKNOWN). This "
            "checkpoint is not eligible as a formal result.",
            RuntimeWarning, stacklevel=2,
        )
        return {}

    raise RulesetFingerprintError(
        "Cannot determine the environment's tagging ruleset, so this checkpoint "
        "would be written unstamped and would reject itself on load. Refusing to "
        "save. Expose the env as trainer.env / .vec_env / ._env, or set "
        "trainer.allow_unstamped_checkpoint = True for a labelled diagnostic run."
        + (f" Last probe error: {err!r}" if err is not None else "")
    )


def read_checkpoint_ruleset(path: str) -> dict:
    """Return the stored ruleset fingerprint, or {} for legacy checkpoints."""
    payload = read_checkpoint_payload(path, map_location="cpu")
    rs = payload.get("ruleset")
    return dict(rs) if isinstance(rs, dict) else {}


def verify_checkpoint_ruleset(path: str, env_cfg, *, allow_mismatch: bool = False) -> dict:
    """Enforce that a checkpoint's ruleset matches the environment's.

    Raises ``RulesetMismatchError`` unless ``allow_mismatch``; on override the
    returned dict carries ``formal_result_eligible=False``.
    """
    from rl.ruleset_identity import enforce, fingerprint

    return enforce(read_checkpoint_ruleset(path) or None, fingerprint(env_cfg),
                   allow_mismatch=allow_mismatch, context=str(path))


def load_custom_ppo_checkpoint(path: str, observation_space, action_space, *, device: str | torch.device = "cpu") -> LoadedCheckpoint:
    device_t = torch.device(device)
    payload = read_checkpoint_payload(path, map_location=device_t)
    assert_compatible_global_state_dim(payload, path)
    metadata = parse_checkpoint_metadata(payload, path, observation_space, action_space)
    arch = _architecture_from_metadata(metadata, observation_space, action_space)
    try:
        model = SharedActorCentralizedCritic(observation_space, action_space, **arch.model_kwargs).to(device_t)
    except Exception as exc:
        raise CheckpointModelConstructionError("Unable to construct checkpoint policy model", checkpoint_path=path, observed=exc) from exc
    _load_model_state_dict_compat(
        model,
        payload["model_state_dict"],
        checkpoint_cfg=payload.get("cfg"),
        observation_space=observation_space,
        action_space=action_space,
    )
    raw_ckpt_cfg = payload.get("cfg") or {}
    ckpt_cfg = canonicalize_latent_strategy_cfg(raw_ckpt_cfg) if isinstance(raw_ckpt_cfg, dict) else raw_ckpt_cfg
    if isinstance(ckpt_cfg, dict) and "seed" in ckpt_cfg:
        apply_deterministic_sampling_generators(model, int(ckpt_cfg["seed"]), device=device_t)
    policy = CustomPPOInferencePolicy(model, device=device_t, cfg=ckpt_cfg)
    descriptor = describe_checkpoint(path, metadata)
    report = CheckpointLoadReport(
        descriptor=descriptor,
        migrations=(),
        missing_keys=(),
        unexpected_keys=(),
        behavioral_equivalence=None,
        device=str(device_t),
        loaded_at=datetime.now(timezone.utc).isoformat(),
        torch_version=str(torch.__version__),
    )
    return LoadedCheckpoint(policy=policy, report=report)


def load_custom_ppo_policy(path: str, observation_space, action_space, *, device: str | torch.device = "cpu") -> CustomPPOInferencePolicy:
    return load_custom_ppo_checkpoint(path, observation_space, action_space, device=device).policy


def load_trainer_checkpoint(trainer: Any, path: str) -> CheckpointTimingReport:
    import time
    from .archive import _torch_load_checkpoint
    from .metadata import assert_compatible_global_state_dim
    from .state_dict import _load_model_state_dict_compat
    from .models import CheckpointTimingReport
    from rl.custom_ppo.inference import CUSTOM_PPO_FORMAT, CUSTOM_PPO_LATENT_FORMAT
    from rl.ruleset_identity import (
        ARTIFACT_IDENTITY_KEY,
        RunIdentity,
        RunIdentityError,
        assert_ruleset_matches_identity,
        build_formal_run_identity,
        fingerprint,
    )
    
    total_start = time.perf_counter()
    
    read_start = time.perf_counter()
    payload = _torch_load_checkpoint(path, map_location=trainer.device)
    archive_read_seconds = time.perf_counter() - read_start

    # Identity gate BEFORE model execution. Legacy / V1 / missing / mismatched
    # checkpoints fail closed here rather than silently entering the run.
    live_identity = getattr(trainer, "run_identity", None)
    if live_identity is None or not isinstance(live_identity, RunIdentity):
        live_identity = build_formal_run_identity(
            trainer.env, run_id=str(getattr(getattr(trainer, "cfg", None), "run_tag", "resume"))
        )
        trainer.run_identity = live_identity

    ckpt_ruleset = payload.get("ruleset") if isinstance(payload, dict) else None
    if not isinstance(ckpt_ruleset, dict) or not ckpt_ruleset:
        raise RunIdentityError(
            f"Checkpoint {path!r} has no ruleset identity (legacy/missing); "
            "refusing to load before model execution."
        )
    assert_ruleset_matches_identity(ckpt_ruleset, live_identity, context=path)

    ai = payload.get(ARTIFACT_IDENTITY_KEY) if isinstance(payload, dict) else None
    if isinstance(ai, dict):
        for key in ("canonical_map", "resolved_map", "ruleset_id", "ruleset_fingerprint"):
            if ai.get(key) is not None and str(ai.get(key)) != str(getattr(live_identity, key)):
                raise RunIdentityError(
                    f"Checkpoint {path!r} artifact_identity.{key} mismatch vs live "
                    f"run identity: {ai.get(key)!r} != {getattr(live_identity, key)!r}"
                )
    else:
        # Older stamped checkpoints may only carry ``ruleset``; that still must
        # match (checked above). Require the passport block for formal resumes.
        if bool(getattr(trainer, "require_checkpoint_artifact_identity", True)):
            raise RunIdentityError(
                f"Checkpoint {path!r} is missing artifact_identity; refusing formal load."
            )

    # Also reject when the stored ruleset disagrees with the live env fingerprint
    # even if the trainer's RunIdentity somehow drifted.
    env_fp = fingerprint(_resolve_live_env_cfg(trainer))
    from rl.ruleset_identity import enforce
    enforce(ckpt_ruleset, env_fp, context=path)
    
    model_start = time.perf_counter()
    assert_compatible_global_state_dim(payload, path)
    _load_model_state_dict_compat(
        trainer.model,
        payload["model_state_dict"],
        allow_active_actor_migration=bool(getattr(trainer.cfg, "allow_active_actor_module_migration", False)),
        checkpoint_cfg=payload.get("cfg"),
        target_cfg=trainer.cfg,
        observation_space=trainer.env.observation_space,
        action_space=trainer.env.action_space,
    )
    model_construction_seconds = time.perf_counter() - model_start
    
    state_start = time.perf_counter()
    load_weights_only = bool(getattr(trainer.cfg, "load_weights_only", False))
    if load_weights_only:
        print("[PPO] Skipping checkpoint optimizer state: --load-weights-only was set.")
        if bool(getattr(trainer.cfg, "router_reinitialize_on_load", False)):
            trainer._reinitialize_router_after_load()
    else:
        reinit_router = bool(getattr(trainer.cfg, "router_reinitialize_on_load", False))
        if reinit_router:
            print("[PPO] Skipping checkpoint optimizer state: router_reinitialize_on_load=True")
            trainer._reinitialize_router_after_load()
        else:
            allow_migration = bool(getattr(trainer.cfg, "allow_active_actor_module_migration", False))
            trainer.optimizers.load_checkpoint(payload, allow_architecture_migration=allow_migration)
            
    v6i1_latent_payload = dict(payload.get("latent_state_v6i1", {}) or {})
    if trainer.v6i1_curriculum is not None and "v6i1_curriculum_state" in payload:
        from rl.custom_ppo.v6i1_phase_runtime import load_v6i1_curriculum_state
        load_v6i1_curriculum_state(trainer.v6i1_curriculum, payload["v6i1_curriculum_state"])
        
    trainer.global_step = int(payload.get("global_step", 0))
    trainer._updates_completed = int(payload.get("updates_completed", 0))
    trainer.return_norm.load_state_dict(
        {
            "mean": payload.get("return_norm_mean", 0.0),
            "var": payload.get("return_norm_var", 1.0),
            "count": payload.get("return_norm_count", 1e-4),
        }
    )
    trainer.strategy_return_norm.load_state_dict(
        {
            "mean": payload.get("strategy_return_mean", 0.0),
            "var": payload.get("strategy_return_var", 1.0),
            "count": payload.get("strategy_return_count", 1e-4),
        }
    )
    trainer.last_stats = dict(payload.get("last_stats", {}))
    trainer._last_obs = None
    trainer._last_global_state = None
    trainer.latent_state.current_z = None
    if trainer.use_latent_strategy:
        trainer.latent_state.reset()
    if v6i1_latent_payload:
        from rl.custom_ppo.v6i1_phase_runtime import restore_latent_state_v6i1_checkpoint
        restore_latent_state_v6i1_checkpoint(trainer.latent_state, v6i1_latent_payload)
    if trainer.comm_runtime.enabled and "comm_runtime_state" in payload:
        trainer.comm_runtime.load_state_dict(dict(payload.get("comm_runtime_state", {}) or {}))
    trainer.updater.load_state_dict(dict(payload.get("ppo_updater_state", {}) or {}))
    state_load_seconds = time.perf_counter() - state_start
    
    import os
    hash_start = time.perf_counter()
    import hashlib
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        _ = h.hexdigest()
    except Exception:
        pass
    hash_seconds = time.perf_counter() - hash_start
    
    migration_start = time.perf_counter()
    sd = payload.get("model_state_dict", {})
    legacy_actor_keys = {"actor_body.", "actor_head.", "strategy_embedding."}
    migration_seconds = time.perf_counter() - migration_start
    
    behavioral_equivalence_seconds = 0.0
    total_seconds = time.perf_counter() - total_start
    
    return CheckpointTimingReport(
        archive_read_seconds=archive_read_seconds,
        model_construction_seconds=model_construction_seconds,
        migration_seconds=migration_seconds,
        state_load_seconds=state_load_seconds,
        behavioral_equivalence_seconds=behavioral_equivalence_seconds,
        hash_seconds=hash_seconds,
        total_seconds=total_seconds,
    )


def save_trainer_checkpoint(trainer: Any, path: str) -> CheckpointSaveTimingReport:
    import time
    import os
    from dataclasses import asdict
    import torch
    from .models import CheckpointSaveTimingReport
    from rl.custom_ppo.inference import CUSTOM_PPO_FORMAT, CUSTOM_PPO_LATENT_FORMAT, CUSTOM_PPO_ACTOR_ARCH, CUSTOM_PPO_VEC_SCHEMA_VERSION
    from rl.ruleset_identity import (
        ARTIFACT_IDENTITY_KEY,
        RunIdentity,
        assert_ruleset_matches_identity,
        build_formal_run_identity,
    )

    total_start = time.perf_counter()

    run_identity = getattr(trainer, "run_identity", None)
    if run_identity is None or not isinstance(run_identity, RunIdentity):
        # Unit-test / legacy call sites may construct the trainer without the
        # production wiring. Resolve once from the LIVE env so the checkpoint
        # still enters the same identity universe — never invent from cfg defaults.
        run_identity = build_formal_run_identity(
            trainer.env,
            run_id=str(getattr(getattr(trainer, "cfg", None), "run_tag", "") or "checkpoint"),
        )
        trainer.run_identity = run_identity

    ruleset_fp = _env_ruleset_fingerprint(trainer)
    assert_ruleset_matches_identity(ruleset_fp, run_identity, context=str(path))
    
    rn = trainer.return_norm.state_dict()
    srn = trainer.strategy_return_norm.state_dict()
    payload = {
        "model_state_dict": trainer.model.state_dict(),
        "global_step": trainer.global_step,
        "updates_completed": trainer._updates_completed,
        "return_norm_mean": rn["mean"],
        "return_norm_var": rn["var"],
        "return_norm_count": rn["count"],
        "strategy_return_mean": srn["mean"],
        "strategy_return_var": srn["var"],
        "strategy_return_count": srn["count"],
        "cfg": asdict(trainer.cfg),
        "last_stats": trainer.last_stats,
        "format": CUSTOM_PPO_LATENT_FORMAT if trainer.use_latent_strategy else CUSTOM_PPO_FORMAT,
        "actor_arch": CUSTOM_PPO_ACTOR_ARCH,
        "actor_cnn_feature_dim": int(trainer.model.actor_cnn_feature_dim),
        "global_state_dim": int(trainer.model.global_state_dim),
        "vec_schema_version": CUSTOM_PPO_VEC_SCHEMA_VERSION,
        # Tagging-ruleset fingerprint of the ENVIRONMENT this policy trained in.
        # Kept separate from "cfg" (the trainer config) because the rules live on
        # GPUFieldConfig. A RULESET_V1 policy learned a different game -- a lone
        # defender could not tag -- so loading one into a V2 run silently
        # corrupts the result. Absent => LEGACY_UNKNOWN, rejected for formal runs.
        "ruleset": ruleset_fp,
        # Same passport as run_config / episode CSV / manifests — two
        # representations of one identity, verified against ``ruleset`` above.
        ARTIFACT_IDENTITY_KEY: run_identity.artifact_identity(),
    }
    trainer.optimizers.write_checkpoint(payload)
    if trainer.v6i1_curriculum is not None:
        from rl.custom_ppo.v6i1_phase_runtime import (
            latent_state_v6i1_checkpoint,
            v6i1_curriculum_state_dict,
        )
        payload["v6i1_curriculum_state"] = v6i1_curriculum_state_dict(trainer.v6i1_curriculum)
        payload["latent_state_v6i1"] = latent_state_v6i1_checkpoint(trainer.latent_state)
    if trainer.comm_runtime.enabled:
        payload["comm_runtime_state"] = trainer.comm_runtime.state_dict()
    payload["ppo_updater_state"] = trainer.updater.state_dict()
    
    # Single shared boundary: checkpoint["ruleset"], checkpoint's
    # artifact_identity, and the live RunIdentity must be three representations
    # of one fact. A matching ruleset_id alone is never sufficient.
    from rl.ruleset_identity import verify_checkpoint_run_identity

    verify_checkpoint_run_identity(
        payload, run_identity, operation="save", context=str(path))

    # Atomic write: a failed identity check or a torn write must never leave an
    # apparently valid checkpoint behind for a later run to pick up.
    write_start = time.perf_counter()
    tmp_path = f"{path}.tmp"
    try:
        torch.save(payload, tmp_path)
        verify_checkpoint_run_identity(
            _torch_load_checkpoint(tmp_path, map_location="cpu"),
            run_identity, operation="save", context=f"{path} (readback)")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise
    write_seconds = time.perf_counter() - write_start
    
    hash_start = time.perf_counter()
    import hashlib
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        _ = h.hexdigest()
    except Exception:
        pass
    hash_seconds = time.perf_counter() - hash_start
    
    total_seconds = time.perf_counter() - total_start
    
    return CheckpointSaveTimingReport(
        write_seconds=write_seconds,
        hash_seconds=hash_seconds,
        total_seconds=total_seconds,
    )
