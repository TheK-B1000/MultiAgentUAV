from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class CompatibilityMode(StrEnum):
    STRICT = "strict"
    ALLOW_SUPPORTED_MIGRATIONS = "allow_supported_migrations"


@dataclass(frozen=True)
class CheckpointDescriptor:
    path: Path
    sha256: str
    size_bytes: int
    schema_version: int | None
    policy_version: str | None
    observation_channels: int
    n_agents: int
    n_macros: int
    n_targets: int
    latent_count: int | None

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(frozen=True)
class CheckpointLoadRequest:
    path: Path
    device: str
    expected_observation_channels: int | None = None
    compatibility_mode: CompatibilityMode = CompatibilityMode.ALLOW_SUPPORTED_MIGRATIONS
    validate_behavior: bool = True
    strict_state_dict: bool = True


@dataclass(frozen=True)
class CheckpointMetadata:
    format: str
    model_path: Path
    cfg: dict[str, Any]
    actor_arch: str
    vec_schema_version: int
    global_state_dim: int
    observation_channels: int
    n_agents: int
    n_macros: int
    n_targets: int
    latent_count: int | None
    schema_version: int | None = None
    policy_version: str | None = None
    unknown_fields: dict[str, Any] | None = None

    def to_legacy_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "format": self.format,
            "model_path": str(self.model_path),
            "cfg": self.cfg,
            "actor_arch": self.actor_arch,
            "vec_schema_version": self.vec_schema_version,
            "global_state_dim": self.global_state_dim,
            "observation_channels": self.observation_channels,
            "n_blue": self.n_agents,
            "n_macros": self.n_macros,
            "n_targets": self.n_targets,
        }
        out["use_latent_strategy"] = bool(self.cfg.get("use_latent_strategy", False))
        out["fixed_latent_strategy"] = bool(self.cfg.get("fixed_latent_strategy", False))
        out["fixed_latent_strategy_id"] = int(self.cfg.get("fixed_latent_strategy_id", 0) or 0)
        out["actor_cnn_feature_dim"] = int(self.cfg.get("actor_cnn_feature_dim", 128))
        out["map_layout"] = str(self.cfg.get("map_layout", "map_a_open") or "map_a_open")
        if self.latent_count is not None:
            out["latent_k"] = self.latent_count
        return out


@dataclass(frozen=True)
class PolicyArchitecture:
    observation_channels: int
    n_agents: int
    n_macros: int
    n_targets: int
    latent_count: int | None
    model_kwargs: dict[str, Any]


@dataclass(frozen=True)
class MigrationRecord:
    migration_id: str
    source_version: str
    target_version: str
    changed_keys: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class StateDictLoadReport:
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    normalized_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class BehavioralEquivalenceReport:
    passed: bool
    mean_kl: float
    max_kl: float
    max_logit_difference: float
    argmax_difference_rate: float
    sample_count: int
    tolerance: float


@dataclass(frozen=True)
class CheckpointLoadReport:
    descriptor: CheckpointDescriptor
    migrations: tuple[MigrationRecord, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    behavioral_equivalence: BehavioralEquivalenceReport | None
    device: str
    loaded_at: str
    torch_version: str

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["descriptor"]["path"] = str(self.descriptor.path)
        return data


@dataclass(frozen=True)
class LoadedCheckpoint:
    policy: Any
    report: CheckpointLoadReport
