"""Typed credit records and episode recorder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch

from rl.custom_ppo.latent.types import RouterActionSource


@dataclass(frozen=True)
class EpisodeCreditRecord:
    episode_id: int
    state: torch.Tensor
    proposed_z: int
    executed_z: int
    behavior_log_prob: float
    router_log_prob: float
    action_source: RouterActionSource
    selector_hidden: torch.Tensor | None
    return_value: float
    win: int
    opponent_id: int
    bucket_id: int

    def validate(self, *, latent_k: int) -> None:
        if not math.isfinite(self.behavior_log_prob):
            raise ValueError("behavior_log_prob must be finite")
        if not (0 <= int(self.executed_z) < int(latent_k)):
            raise ValueError(f"executed_z out of range: {self.executed_z}")


@dataclass(frozen=True)
class MacroCreditRecord:
    global_state_0: torch.Tensor
    executed_z: int
    behavior_log_prob: float
    macro_return: float
    macro_length: int
    action_source: RouterActionSource
    selector_hidden_0: torch.Tensor | None = None
    reason: str = "boundary"


@dataclass(frozen=True)
class ArcCreditRecord:
    global_state_0: torch.Tensor
    executed_z: int
    behavior_log_prob: float
    arc_return: float
    arc_length: int
    action_source: RouterActionSource
    opponent_id: int = -1
    bucket_id: int = -1
    selector_hidden_0: torch.Tensor | None = None
    reason: str = "z_change"


@dataclass(frozen=True)
class RefreshCreditRecord:
    env_id: int
    episode_id: int
    decision_step: int
    prev_z: int
    executed_z: int
    reason_id: int
    flag_state_bucket: int
    return_at_refresh: float
    refresh_state: torch.Tensor
    future_return: float | None = None
    opponent_id: int = -1


def stack_selector_hidden_records(
    records: list[dict[str, Any]],
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if not records or "selector_hidden_0" not in records[0]:
        return None
    return torch.stack(
        [r["selector_hidden_0"].detach().float() for r in records], dim=0
    ).to(device)


class EpisodeStrategyRecorder:
    """Tracks sampled episode-level z actions for task-return PPO credit."""

    def __init__(self) -> None:
        self.pending: dict[int, dict[str, Any]] = {}
        self.completed: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.pending.clear()
        self.completed.clear()

    def clear_completed(self) -> None:
        self.completed.clear()

    def record_start(
        self,
        *,
        env_index: int,
        episode_id: int,
        global_state_0: torch.Tensor,
        proposed_z: int,
        executed_z: int,
        behavior_log_prob: float,
        router_log_prob: float,
        action_source: RouterActionSource,
        bucket_id: int,
        q_phi_probs: Iterable[float],
        selector_hidden_0: torch.Tensor | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "episode_id": int(episode_id),
            "global_state_0": global_state_0.detach().clone(),
            "proposed_z": int(proposed_z),
            "z": int(executed_z),
            "executed_z": int(executed_z),
            "behavior_log_prob": float(behavior_log_prob),
            "z_logprob_old": float(behavior_log_prob),
            "router_log_prob": float(router_log_prob),
            "action_source": str(action_source.value),
            "episode_return": None,
            "episode_win": None,
            "bucket_id": int(bucket_id),
            "opponent_id": -1,
            "q_phi_probs": [float(x) for x in q_phi_probs],
        }
        if selector_hidden_0 is not None:
            record["selector_hidden_0"] = selector_hidden_0.detach().clone().cpu()
        self.pending[int(env_index)] = record

    def record_outcome(
        self,
        *,
        env_index: int,
        episode_return: float,
        episode_win: int,
        opponent_id: int = -1,
    ) -> Optional[dict[str, Any]]:
        record = self.pending.pop(int(env_index), None)
        if record is None:
            return None
        record["episode_return"] = float(episode_return)
        record["episode_win"] = int(episode_win)
        record["opponent_id"] = int(opponent_id)
        self.completed.append(record)
        return record
