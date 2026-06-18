"""Immutable episode-router training batch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from rl.custom_ppo.latent.records import stack_selector_hidden_records
from rl.custom_ppo.latent.types import EpisodeRouterBatch, RouterActionSource

if TYPE_CHECKING:
    from rl.custom_ppo.latent.state import LatentStrategyState


def _parse_action_source(raw: str) -> RouterActionSource:
    try:
        return RouterActionSource(str(raw))
    except ValueError:
        return RouterActionSource.ROUTER


class EpisodeBatchBuilder:
    def __init__(self, host: LatentStrategyState) -> None:
        self.host = host

    def build(self) -> EpisodeRouterBatch | None:
        trainer = self.host.trainer
        if (
            not trainer.latent_episode_strategy_ppo
            or trainer.fixed_latent_strategy
            or trainer.model.episode_strategy_value_head is None
        ):
            return None
        records = list(self.host.rollout_strategy_episode_records)
        if not records:
            return None
        device = trainer.device
        filtered: list[dict[str, Any]] = []
        for record in records:
            source = _parse_action_source(str(record.get("action_source", RouterActionSource.ROUTER.value)))
            if source is RouterActionSource.FORCED_REHEARSAL:
                continue
            filtered.append(record)
        if not filtered:
            return None

        states = torch.stack(
            [r["global_state_0"].detach().float() for r in filtered], dim=0
        ).to(device)
        executed_z = torch.as_tensor(
            [int(r.get("executed_z", r["z"])) for r in filtered],
            dtype=torch.long,
            device=device,
        )
        old_behavior_log_prob = torch.as_tensor(
            [float(r.get("behavior_log_prob", r["z_logprob_old"])) for r in filtered],
            dtype=torch.float32,
            device=device,
        )
        episode_returns = torch.as_tensor(
            [float(r["episode_return"]) for r in filtered],
            dtype=torch.float32,
            device=device,
        )
        opponent_ids = torch.as_tensor(
            [int(r.get("opponent_id", -1)) for r in filtered],
            dtype=torch.long,
            device=device,
        )
        bucket_ids = torch.as_tensor(
            [int(r.get("bucket_id", -1)) for r in filtered],
            dtype=torch.long,
            device=device,
        )
        selector_hidden = stack_selector_hidden_records(filtered, device=device)
        action_sources = tuple(
            _parse_action_source(str(r.get("action_source", RouterActionSource.ROUTER.value)))
            for r in filtered
        )
        return EpisodeRouterBatch(
            states=states,
            executed_z=executed_z,
            old_behavior_log_prob=old_behavior_log_prob,
            episode_returns=episode_returns,
            opponent_ids=opponent_ids,
            bucket_ids=bucket_ids,
            selector_hidden=selector_hidden,
            action_sources=action_sources,
        )

    def build_legacy_dict(self) -> Optional[dict[str, torch.Tensor]]:
        batch = self.build()
        if batch is None:
            return None
        out: dict[str, torch.Tensor] = {
            "states": batch.states,
            "z": batch.executed_z,
            "old_log_prob": batch.old_behavior_log_prob,
            "episode_returns": batch.episode_returns,
            "opponent_ids": batch.opponent_ids,
            "bucket_ids": batch.bucket_ids,
        }
        if batch.selector_hidden is not None:
            out["selector_hidden"] = batch.selector_hidden
        return out
