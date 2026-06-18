"""Explicit per-env episode lifecycle state."""

from __future__ import annotations

import torch

from rl.custom_ppo.latent.types import LifecycleError


class EpisodeLifecycleState:
    """Tracks episode id, warmup commit, forced-z flags, and active rows."""

    def __init__(self, *, n_envs: int, device: torch.device) -> None:
        self.episode_id = torch.zeros((n_envs,), dtype=torch.long, device=device)
        self.steps_since_start = torch.zeros((n_envs,), dtype=torch.long, device=device)
        self.committed = torch.zeros((n_envs,), dtype=torch.bool, device=device)
        self.active = torch.zeros((n_envs,), dtype=torch.bool, device=device)
        self.forced = torch.zeros((n_envs,), dtype=torch.bool, device=device)
        self.forced_z_id = torch.zeros((n_envs,), dtype=torch.long, device=device)
        self.rehearsal = torch.zeros((n_envs,), dtype=torch.bool, device=device)
        self.first_z_sample_step = torch.full((n_envs,), -1, dtype=torch.long, device=device)
        self.return_baseline_at_commit = torch.zeros((n_envs,), dtype=torch.float32, device=device)

    def begin(self, mask: torch.Tensor) -> None:
        if not bool(mask.any().item()):
            return
        if bool((self.committed[mask] & self.active[mask]).any().item()):
            raise LifecycleError("begin() called on committed active episode rows")
        self.active[mask] = True
        self.committed[mask] = False
        self.steps_since_start[mask] = 0
        self.forced[mask] = False
        self.forced_z_id[mask] = 0
        self.rehearsal[mask] = False
        self.first_z_sample_step[mask] = -1
        self.return_baseline_at_commit[mask] = 0.0

    def advance(self, mask: torch.Tensor) -> None:
        if bool(mask.any().item()):
            self.steps_since_start[mask] += 1

    def commit(self, mask: torch.Tensor) -> None:
        if bool(mask.any().item()):
            self.committed[mask] = True

    def complete(self, mask: torch.Tensor) -> None:
        if not bool(mask.any().item()):
            return
        self.active[mask] = False
        self.committed[mask] = False
        self.forced[mask] = False
        self.forced_z_id[mask] = 0
        self.rehearsal[mask] = False
        self.first_z_sample_step[mask] = -1
        self.return_baseline_at_commit[mask] = 0.0
        self.episode_id[mask] += 1

    def mark_forced(self, mask: torch.Tensor, z_id: torch.Tensor, *, rehearsal: bool = False) -> None:
        self.forced[mask] = True
        self.forced_z_id[mask] = z_id.long()
        self.rehearsal[mask] = rehearsal
