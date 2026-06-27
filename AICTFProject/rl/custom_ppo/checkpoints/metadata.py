from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_marl import CONTEXT_STATE_DIM

from .archive import read_checkpoint_payload
from .errors import CheckpointMetadataError, CheckpointSchemaError
from .models import CheckpointMetadata

CUSTOM_PPO_FORMAT = "custom_ppo_cnn_v1"
CUSTOM_PPO_LATENT_FORMAT = "custom_ppo_latent_cnn_v1"
CUSTOM_PPO_ACTOR_ARCH = "cnn_mlp"
CUSTOM_PPO_VEC_SCHEMA_VERSION = 1

_LATENT_STRATEGY_LEGACY_KEY_MAP: tuple[tuple[str, str], ...] = (
    ("latent_strategy_q_head", "latent_strategy_aux_return_head"),
    ("latent_strategy_q_coef", "latent_strategy_aux_return_coef"),
)


def canonicalize_latent_strategy_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(cfg)
    for legacy_key, canonical_key in _LATENT_STRATEGY_LEGACY_KEY_MAP:
        if legacy_key in out and canonical_key not in out:
            out[canonical_key] = out[legacy_key]
        out.pop(legacy_key, None)
    return out


def assert_compatible_global_state_dim(payload: dict[str, Any], path: str | Path) -> None:
    ckpt_dim = payload.get("global_state_dim")
    if ckpt_dim is None:
        return
    cfg = payload.get("cfg") or {}
    uses_latent = bool(cfg.get("use_latent_strategy", False)) if isinstance(cfg, dict) else False
    router_context_mode = str(cfg.get("router_context_mode", "") or "") if isinstance(cfg, dict) else ""
    if uses_latent and router_context_mode == "current":
        expected_dim = GLOBAL_STATE_DIM + 1
    else:
        expected_dim = CONTEXT_STATE_DIM if uses_latent else GLOBAL_STATE_DIM
    if int(ckpt_dim) != int(expected_dim):
        raise CheckpointSchemaError(
            "Checkpoint global_state_dim is incompatible with this code",
            checkpoint_path=str(path),
            expected=expected_dim,
            observed=int(ckpt_dim),
        )


def parse_checkpoint_metadata(payload: dict[str, Any], path: str | Path, observation_space: Any | None = None, action_space: Any | None = None) -> CheckpointMetadata:
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise CheckpointMetadataError("Not a custom PPO checkpoint", checkpoint_path=str(path))
    raw_cfg = payload.get("cfg") or {}
    cfg = canonicalize_latent_strategy_cfg(raw_cfg) if isinstance(raw_cfg, dict) else {}
    fmt = str(payload.get("format", "custom_ppo_v2"))
    n_agents = int(cfg.get("max_blue_agents", cfg.get("n_agents_per_team", 0)) or 0)
    if observation_space is not None:
        grid_shape = observation_space.spaces["grid"].shape
        n_agents = int(grid_shape[0])
        observation_channels = int(grid_shape[1])
    else:
        observation_channels = int(payload.get("observation_channels", cfg.get("observation_channels", 0)) or 0)
    if action_space is not None:
        dims = list(getattr(action_space, "nvec", []))
        n_macros = int(dims[0]) if dims else 0
        n_targets = int(dims[1]) if len(dims) > 1 else 0
    else:
        n_macros = int(payload.get("n_macros", cfg.get("n_macros", 0)) or 0)
        n_targets = int(payload.get("n_targets", cfg.get("n_targets", 0)) or 0)
    latent_count = int(cfg["latent_k"]) if "latent_k" in cfg else None
    return CheckpointMetadata(
        format=fmt,
        model_path=Path(path),
        cfg=cfg,
        actor_arch=str(payload.get("actor_arch", "flat_mlp" if fmt.endswith("_v2") else "unknown")),
        vec_schema_version=int(payload.get("vec_schema_version", 2 if fmt.endswith("_v2") else 0)),
        global_state_dim=int(payload.get("global_state_dim", GLOBAL_STATE_DIM)),
        observation_channels=observation_channels,
        n_agents=n_agents,
        n_macros=n_macros,
        n_targets=n_targets,
        latent_count=latent_count,
        schema_version=int(payload["schema_version"]) if "schema_version" in payload and payload["schema_version"] is not None else None,
        policy_version=str(payload["policy_version"]) if payload.get("policy_version") is not None else None,
        unknown_fields={k: v for k, v in payload.items() if k not in {"model_state_dict", "cfg"}},
    )


def read_custom_ppo_metadata(path: str) -> dict[str, Any]:
    payload = read_checkpoint_payload(path, map_location="cpu")
    metadata = parse_checkpoint_metadata(payload, path)
    return metadata.to_legacy_dict()
