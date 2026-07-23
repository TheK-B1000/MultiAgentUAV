"""V6I24 shared-core extraction from a competent latent checkpoint.

Scientific framing
------------------
V6I23/V6I22E failed as *latent specialists* (adapters / per-z heads did not
separate π), but retained a competent shared trunk (Stage-C competence).
Path C recycles that shared competence into four independent full policies
and discards collapsed latent specialization machinery.

KEEP (shared core):
  actor_cnn, latent_actor.body, latent_actor.action_head, critic,
  return-norm statistics, and strategy_embedding (required so frozen z=0
  concat conditioning stays behaviorally competent).

DISCARD:
  adapters, per-z action heads, z_adapter / FiLM, strategy encoder / GRU /
  phase predictor, optimizer / router state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

# Substrings / prefixes that mark collapsed latent specialization machinery.
# Note: ``strategy_embedding`` is intentionally NOT listed — discarding it
# under concat conditioning would randomize the z=0 input and destroy the
# shared-core competence we are trying to recycle.
LATENT_ONLY_PREFIXES: tuple[str, ...] = (
    "latent_actor.latent_adapters",
    "latent_actor.latent_adapter_gates",
    "latent_actor.latent_action_heads",
    "latent_actor.latent_action_biases",
    "latent_actor.z_adapter",
    "latent_actor.actor_z_film",
    "strategy_encoder",
    "selector_gru",
    "phase_predictor",
    "episode_strategy_value_head",
    "strategy_aux_return_head",
    "strategy_q_head",
)


def is_shared_core_parameter(name: str) -> bool:
    """Return True if ``name`` is part of the recyclable shared policy core."""
    n = str(name)
    for prefix in LATENT_ONLY_PREFIXES:
        if n == prefix or n.startswith(prefix + ".") or n.startswith(prefix):
            return False
    return True


def filter_shared_core_state_dict(
    source_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Intersect source∩target by name+shape, dropping latent-only tensors.

    Returns ``(shared_state, report)``.
    """
    shared: dict[str, Any] = {}
    ignored_latent: list[str] = []
    shape_mismatch: list[str] = []
    missing_in_target: list[str] = []

    for name, tensor in source_state.items():
        if not is_shared_core_parameter(name):
            ignored_latent.append(name)
            continue
        if name not in target_state:
            missing_in_target.append(name)
            continue
        tgt = target_state[name]
        if hasattr(tensor, "shape") and hasattr(tgt, "shape") and tuple(tensor.shape) != tuple(tgt.shape):
            shape_mismatch.append(name)
            continue
        shared[name] = tensor

    report = {
        "n_source": len(source_state),
        "n_target": len(target_state),
        "n_shared_loaded": len(shared),
        "ignored_latent_keys": sorted(ignored_latent),
        "missing_in_target": sorted(missing_in_target),
        "shape_mismatch": sorted(shape_mismatch),
        "kept_strategy_embedding": any(k.endswith("strategy_embedding.weight") for k in shared),
    }
    return shared, report


@dataclass(frozen=True)
class SharedCoreMaterializeResult:
    output_path: Path
    report: dict[str, Any]


def _load_payload(path: Path | str, *, map_location: str = "cpu") -> dict[str, Any]:
    from rl.custom_ppo.checkpoints.archive import _torch_load_checkpoint

    return dict(_torch_load_checkpoint(str(path), map_location=map_location))


def extract_shared_core_into_model(
    source_checkpoint: Path | str,
    target_model: torch.nn.Module,
) -> dict[str, Any]:
    """Load shared-core tensors into ``target_model`` (in place). Returns report."""
    from rl.custom_ppo.policy import remap_legacy_actor_state_dict_keys
    from rl.custom_ppo.checkpoints.state_dict import (
        _expand_cnn_obs_channels,
        _remap_legacy_strategy_aux_head_state_dict,
    )

    payload = _load_payload(source_checkpoint)
    raw = dict(payload.get("model_state_dict") or {})
    remapped = remap_legacy_actor_state_dict_keys(
        _remap_legacy_strategy_aux_head_state_dict(raw)
    )
    # Match obstacle-channel expansion used by the normal loader.
    model_sd = dict(target_model.state_dict())
    cnn_key = None
    for candidate in ("latent_actor.actor_cnn.conv.0.weight", "actor_cnn.conv.0.weight"):
        if candidate in model_sd:
            cnn_key = candidate
            break
    if cnn_key is not None:
        remapped = _expand_cnn_obs_channels(remapped, int(model_sd[cnn_key].shape[1]))

    shared, report = filter_shared_core_state_dict(remapped, model_sd)
    incompatible = target_model.load_state_dict(shared, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    report["freshly_initialized_tensors"] = missing
    report["ignored_source_tensors_unexpected"] = unexpected
    report["source_checkpoint"] = str(source_checkpoint)
    report["return_norm"] = {
        "mean": float(payload.get("return_norm_mean", 0.0) or 0.0),
        "var": float(payload.get("return_norm_var", 1.0) or 1.0),
        "count": float(payload.get("return_norm_count", 1e-4) or 1e-4),
    }
    print(
        f"[v6i24 shared-core] Loaded {len(shared)} shared-core tensors from {source_checkpoint}"
    )
    print(f"[v6i24 shared-core] Freshly initialized tensors: {len(missing)}")
    print(
        f"[v6i24 shared-core] Ignored latent-only source tensors: {len(report['ignored_latent_keys'])}"
    )
    if report.get("kept_strategy_embedding"):
        print(
            "[v6i24 shared-core] Kept strategy_embedding so frozen z=0 concat "
            "warm-start preserves competence (adapters/per-z heads discarded)."
        )
    return report


def materialize_shared_core_member_checkpoint(
    *,
    source_checkpoint: Path | str | None,
    output_path: Path | str,
    seed: int,
    device: str = "cpu",
    mode: str = "shared-core",
) -> SharedCoreMaterializeResult:
    """Build a V6I24-shaped init zip with shared-core weights (or fresh identical init).

    Optimizer state is intentionally absent so ``--load-weights-only`` starts
    each member with a fresh AdamW.
    """
    from types import SimpleNamespace

    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from rl.config.ppo_config import PPOConfig
    from rl.custom_ppo.inference import (
        CUSTOM_PPO_ACTOR_ARCH,
        CUSTOM_PPO_LATENT_FORMAT,
        CUSTOM_PPO_VEC_SCHEMA_VERSION,
    )
    from rl.custom_ppo.policy import SharedActorCentralizedCritic
    from rl.custom_ppo.trainer_config import build_model_kwargs
    from rl.presets import PRESET_REGISTRY

    mode = str(mode).strip().lower().replace("_", "-")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg = PRESET_REGISTRY["v6i24"](PPOConfig())
    cfg.seed = int(seed)
    cfg.device = "cpu"
    cfg.load_weights_only = True

    env = GPUCTFVecEnv(
        GPUFieldConfig(
            n_envs=1,
            max_blue_agents=int(getattr(cfg, "max_blue_agents", 2) or 2),
            max_red_agents=int(getattr(cfg, "max_blue_agents", 2) or 2),
            map_layout=str(getattr(cfg, "map_layout", "map_b_split_lane") or "map_b_split_lane"),
            device="cpu",
            seed=int(seed),
            aquaticus_profile=True,
            rules_profile="OURS",
        )
    )
    try:
        torch.manual_seed(int(seed))
        hp = SimpleNamespace(
            use_latent_strategy=True,
            latent_k=int(getattr(cfg, "latent_k", 4) or 4),
        )
        kwargs = build_model_kwargs(cfg, hp)  # type: ignore[arg-type]
        model = SharedActorCentralizedCritic(
            env.observation_space, env.action_space, **kwargs
        ).to("cpu")
        report: dict[str, Any] = {
            "mode": mode,
            "seed": int(seed),
        }
        return_norm = {"mean": 0.0, "var": 1.0, "count": 1e-4}
        if mode == "shared-core":
            if source_checkpoint is None or not Path(source_checkpoint).is_file():
                raise FileNotFoundError(
                    f"shared-core mode requires an existing source checkpoint; got {source_checkpoint!r}"
                )
            report.update(extract_shared_core_into_model(source_checkpoint, model))
            return_norm = dict(report.get("return_norm") or return_norm)
        elif mode == "fresh":
            report["n_shared_loaded"] = 0
            report["note"] = "identical fresh init from seed; no donor weights"
        else:
            raise ValueError(f"Unknown checkpoint mode: {mode!r}")

        payload = {
            "model_state_dict": model.state_dict(),
            "global_step": 0,
            "updates_completed": 0,
            "return_norm_mean": float(return_norm["mean"]),
            "return_norm_var": float(return_norm["var"]),
            "return_norm_count": float(return_norm["count"]),
            "strategy_return_mean": 0.0,
            "strategy_return_var": 1.0,
            "strategy_return_count": 1e-4,
            "cfg": asdict(cfg),
            "last_stats": {},
            "format": CUSTOM_PPO_LATENT_FORMAT,
            "actor_arch": CUSTOM_PPO_ACTOR_ARCH,
            "actor_cnn_feature_dim": int(model.actor_cnn_feature_dim),
            "global_state_dim": int(model.global_state_dim),
            "vec_schema_version": CUSTOM_PPO_VEC_SCHEMA_VERSION,
            "v6i24_shared_core_report": report,
        }
        torch.save(payload, out)
    finally:
        env.close()

    return SharedCoreMaterializeResult(output_path=out, report=report)


def find_newest_competent_zip(roots: Iterable[Path | str]) -> Path | None:
    """Return newest ``*.zip`` under roots, preferring non-venv paths."""
    candidates: list[Path] = []
    for root in roots:
        r = Path(root)
        if not r.exists():
            continue
        for path in r.rglob("*.zip"):
            parts = {p.lower() for p in path.parts}
            if ".venv" in parts or "site-packages" in parts:
                continue
            if path.name.startswith("final_") or "seed" in path.name:
                candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


__all__ = [
    "LATENT_ONLY_PREFIXES",
    "SharedCoreMaterializeResult",
    "extract_shared_core_into_model",
    "filter_shared_core_state_dict",
    "find_newest_competent_zip",
    "is_shared_core_parameter",
    "materialize_shared_core_member_checkpoint",
]
