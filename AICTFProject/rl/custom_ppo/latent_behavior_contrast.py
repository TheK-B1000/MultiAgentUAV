"""Self-supervised latent behavior contrast for Summer-plan option separation.

The contrast signal is intentionally label-free. It never assigns z meanings
such as attacker/defender and never feeds labels into the policy. Forced-z
episodes produce completed-episode behavior embeddings; each forced episode is
rewarded for being different from other latent centroids in the same coarse
game-state bucket, up to a bounded margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rl.behavior_telemetry import N_TELEMETRY


@dataclass
class BehaviorContrastResult:
    bonus: torch.Tensor
    distance: float
    count: int
    active: int


@dataclass
class OutcomeDiversityResult:
    bonus: torch.Tensor
    distance: float
    count: int
    active: int


class BehaviorContrastMemory:
    """EMA centroids for behavior embeddings keyed by label-free context bucket."""

    def __init__(
        self,
        *,
        latent_k: int,
        telemetry_dim: int = N_TELEMETRY,
        ema: float = 0.9,
        margin: float = 0.25,
        device: torch.device | str = "cpu",
    ) -> None:
        self.latent_k = max(1, int(latent_k))
        self.telemetry_dim = max(1, int(telemetry_dim))
        self.ema = min(max(float(ema), 0.0), 0.999)
        self.margin = max(float(margin), 1e-6)
        self.device = torch.device(device)
        self._centroids: dict[int, torch.Tensor] = {}
        self._counts: dict[int, torch.Tensor] = {}

    def normalize(self, embedding: torch.Tensor, *, team_size: int) -> torch.Tensor:
        """Scale telemetry dimensions into a comparable range before distance."""
        team = max(1.0, float(team_size))
        scales = torch.tensor(
            [
                1.0,
                team,
                team,
                team,
                team,
                1.5,
                1.5,
                team,
                1.5,
                1.5,
                1.0,
                1.0,
                1.0,
            ],
            dtype=torch.float32,
            device=embedding.device,
        )
        if scales.numel() != embedding.shape[-1]:
            scales = torch.ones((embedding.shape[-1],), dtype=torch.float32, device=embedding.device)
        return embedding.float() / scales.clamp_min(1e-6)

    def _ensure_bucket(self, bucket_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = int(bucket_id)
        if key not in self._centroids:
            self._centroids[key] = torch.zeros(
                (self.latent_k, self.telemetry_dim), dtype=torch.float32, device=self.device
            )
            self._counts[key] = torch.zeros((self.latent_k,), dtype=torch.long, device=self.device)
        return self._centroids[key], self._counts[key]

    def score_and_update(
        self,
        *,
        bucket_id: int,
        z: int,
        embedding: torch.Tensor,
        coef: float,
    ) -> BehaviorContrastResult:
        """Return bounded separation bonus, then update the matching centroid."""
        z_idx = max(0, min(int(z), self.latent_k - 1))
        centroids, counts = self._ensure_bucket(int(bucket_id))
        emb = embedding.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
        if emb.numel() != self.telemetry_dim:
            raise ValueError(f"behavior embedding has {emb.numel()} dims, expected {self.telemetry_dim}")

        other_mask = counts > 0
        other_mask[z_idx] = False
        if bool(other_mask.any().item()):
            other = centroids[other_mask]
            distances = torch.linalg.vector_norm(other - emb.unsqueeze(0), ord=2, dim=1)
            distance = distances.mean()
            bounded = torch.clamp(distance, max=self.margin)
            bonus = float(max(coef, 0.0)) * bounded
            distance_f = float(distance.detach().cpu().item())
            active = 1
        else:
            bonus = torch.zeros((), dtype=torch.float32, device=self.device)
            distance_f = 0.0
            active = 0

        if int(counts[z_idx].detach().cpu().item()) <= 0:
            centroids[z_idx] = emb
        else:
            centroids[z_idx] = self.ema * centroids[z_idx] + (1.0 - self.ema) * emb
        counts[z_idx] += 1
        return BehaviorContrastResult(
            bonus=bonus,
            distance=distance_f,
            count=1,
            active=active,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "latent_k": self.latent_k,
            "telemetry_dim": self.telemetry_dim,
            "ema": self.ema,
            "margin": self.margin,
            "centroids": {int(k): v.detach().cpu() for k, v in self._centroids.items()},
            "counts": {int(k): v.detach().cpu() for k, v in self._counts.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._centroids.clear()
        self._counts.clear()
        for raw_key, value in dict(state.get("centroids", {})).items():
            key = int(raw_key)
            self._centroids[key] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        for raw_key, value in dict(state.get("counts", {})).items():
            key = int(raw_key)
            self._counts[key] = torch.as_tensor(value, dtype=torch.long, device=self.device)


class OutcomeDiversityMemory:
    """EMA outcome centroids keyed by label-free context bucket.

    This memory tracks only generic terminal outcome scalars such as score
    margin. It never consumes behavior-role metrics and never assigns semantic
    meanings to z indices.
    """

    def __init__(
        self,
        *,
        latent_k: int,
        ema: float = 0.9,
        margin: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.latent_k = max(1, int(latent_k))
        self.ema = min(max(float(ema), 0.0), 0.999)
        self.margin = max(float(margin), 1e-6)
        self.device = torch.device(device)
        self._means: dict[int, torch.Tensor] = {}
        self._counts: dict[int, torch.Tensor] = {}

    def _ensure_bucket(self, bucket_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = int(bucket_id)
        if key not in self._means:
            self._means[key] = torch.zeros((self.latent_k,), dtype=torch.float32, device=self.device)
            self._counts[key] = torch.zeros((self.latent_k,), dtype=torch.long, device=self.device)
        return self._means[key], self._counts[key]

    def score_and_update(
        self,
        *,
        bucket_id: int,
        z: int,
        outcome: torch.Tensor | float,
        coef: float,
    ) -> OutcomeDiversityResult:
        """Return bounded outcome-separation bonus, then update z's EMA."""
        z_idx = max(0, min(int(z), self.latent_k - 1))
        means, counts = self._ensure_bucket(int(bucket_id))
        outcome_t = torch.as_tensor(outcome, dtype=torch.float32, device=self.device).reshape(())

        other_mask = counts > 0
        other_mask[z_idx] = False
        if bool(other_mask.any().item()):
            other = means[other_mask]
            distances = torch.abs(other - outcome_t)
            distance = distances.mean()
            bounded = torch.clamp(distance, max=self.margin)
            bonus = float(max(coef, 0.0)) * bounded
            distance_f = float(distance.detach().cpu().item())
            active = 1
        else:
            bonus = torch.zeros((), dtype=torch.float32, device=self.device)
            distance_f = 0.0
            active = 0

        if int(counts[z_idx].detach().cpu().item()) <= 0:
            means[z_idx] = outcome_t
        else:
            means[z_idx] = self.ema * means[z_idx] + (1.0 - self.ema) * outcome_t
        counts[z_idx] += 1
        return OutcomeDiversityResult(
            bonus=bonus,
            distance=distance_f,
            count=1,
            active=active,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "latent_k": self.latent_k,
            "ema": self.ema,
            "margin": self.margin,
            "means": {int(k): v.detach().cpu() for k, v in self._means.items()},
            "counts": {int(k): v.detach().cpu() for k, v in self._counts.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._means.clear()
        self._counts.clear()
        for raw_key, value in dict(state.get("means", {})).items():
            key = int(raw_key)
            self._means[key] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        for raw_key, value in dict(state.get("counts", {})).items():
            key = int(raw_key)
            self._counts[key] = torch.as_tensor(value, dtype=torch.long, device=self.device)


__all__ = [
    "BehaviorContrastMemory",
    "BehaviorContrastResult",
    "OutcomeDiversityMemory",
    "OutcomeDiversityResult",
]
