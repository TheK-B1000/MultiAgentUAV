from __future__ import annotations

from typing import Mapping, Protocol

import torch

from .models import CheckpointMetadata, MigrationRecord, PolicyArchitecture
from .state_dict import _expand_cnn_obs_channels


class CheckpointMigration(Protocol):
    migration_id: str

    def applies_to(self, metadata: CheckpointMetadata, state_dict: Mapping[str, torch.Tensor], target: PolicyArchitecture) -> bool:
        ...

    def apply(self, metadata: CheckpointMetadata, state_dict: Mapping[str, torch.Tensor], target: PolicyArchitecture) -> tuple[dict[str, torch.Tensor], MigrationRecord]:
        ...


class SevenToEightChannelCNNMigration:
    migration_id = "cnn_obs_7_to_8_zero_obstacle_channel_v1"

    def applies_to(self, metadata: CheckpointMetadata, state_dict: Mapping[str, torch.Tensor], target: PolicyArchitecture) -> bool:
        return metadata.observation_channels == 7 and target.observation_channels == 8

    def apply(self, metadata: CheckpointMetadata, state_dict: Mapping[str, torch.Tensor], target: PolicyArchitecture) -> tuple[dict[str, torch.Tensor], MigrationRecord]:
        before = dict(state_dict)
        after = _expand_cnn_obs_channels(before, target.observation_channels)
        changed = tuple(k for k in after if k in before and after[k] is not before[k])
        return after, MigrationRecord(
            migration_id=self.migration_id,
            source_version="obs_channels=7",
            target_version="obs_channels=8",
            changed_keys=changed,
            explanation="Expand the first actor CNN input layer and zero-initialize the obstacle channel.",
        )


REGISTERED_MIGRATIONS: tuple[CheckpointMigration, ...] = (SevenToEightChannelCNNMigration(),)
