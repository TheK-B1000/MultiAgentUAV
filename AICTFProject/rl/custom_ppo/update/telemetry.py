"""Telemetry accumulation with explicit aggregation modes."""

from __future__ import annotations

from enum import Enum
from typing import Mapping

import numpy as np


class AggregationMode(Enum):
    MEAN = "mean"
    SUM = "sum"
    LAST = "last"
    MIN = "min"
    MAX = "max"


DEFAULT_METRIC_SCHEMA: dict[str, AggregationMode] = {
    "policy_loss": AggregationMode.MEAN,
    "value_loss": AggregationMode.MEAN,
    "entropy": AggregationMode.MEAN,
    "approx_kl": AggregationMode.MEAN,
    "clip_fraction": AggregationMode.MEAN,
    "grad_norm": AggregationMode.MEAN,
    "strategy_entropy": AggregationMode.MEAN,
    "strategy_policy_loss": AggregationMode.MEAN,
    "strategy_approx_kl": AggregationMode.MEAN,
    "strategy_clip_fraction": AggregationMode.MEAN,
    "strategy_ratio_std": AggregationMode.MEAN,
    "strategy_aux_return_loss": AggregationMode.MEAN,
    "strategy_persist_loss": AggregationMode.MEAN,
    "strategy_grad_norm": AggregationMode.MEAN,
    "strategy_resample_fraction": AggregationMode.MEAN,
    "strategy_kl": AggregationMode.MEAN,
    "strategy_phase_loss": AggregationMode.MEAN,
    "strategy_marginal_entropy_loss": AggregationMode.MEAN,
    "strategy_marginal_entropy_nats": AggregationMode.MEAN,
    "strategy_marginal_entropy_kl": AggregationMode.MEAN,
    "router_rollout_soft_marginal_entropy_nats": AggregationMode.LAST,
    "router_rollout_soft_conditional_entropy_nats": AggregationMode.LAST,
    "router_rollout_soft_mi_proxy_nats": AggregationMode.LAST,
    "router_rollout_soft_argmax_occupancy_max": AggregationMode.LAST,
    "router_rollout_soft_argmax_occupancy_min": AggregationMode.LAST,
    "router_rollout_soft_argmax_occupancy_ratio": AggregationMode.LAST,
    "router_rollout_resample_count": AggregationMode.LAST,
    "latent_actor_z_separation_loss": AggregationMode.MEAN,
    "latent_actor_z_separation_jsd": AggregationMode.MEAN,
    "latent_actor_z_separation_jsd_min": AggregationMode.MEAN,
    "latent_actor_z_separation_jsd_max": AggregationMode.MEAN,
    "latent_actor_z_separation_active": AggregationMode.MEAN,
    "latent_actor_z_separation_train_active": AggregationMode.MEAN,
    "cf_actor_grad_norm": AggregationMode.MEAN,
    "ppo_actor_grad_norm": AggregationMode.MEAN,
    "cf_to_ppo_grad_ratio": AggregationMode.MEAN,
    "cf_batch_pairs_below_margin": AggregationMode.MEAN,
    "cf_hinge_active": AggregationMode.MEAN,
    "cf_hinge_effective": AggregationMode.MEAN,
    "cf_valid_team_groups": AggregationMode.MEAN,
    "cf_weight_sum": AggregationMode.MEAN,
    "cf_effective_pairs": AggregationMode.MEAN,
    "cf_loss_requires_grad": AggregationMode.MEAN,
    "actor_intervention_measurement_valid": AggregationMode.MEAN,
    "actor_intervention_valid_minibatches": AggregationMode.SUM,
}


def build_metric_schema(*, latent_k: int, pair_count: int) -> dict[str, AggregationMode]:
    schema = dict(DEFAULT_METRIC_SCHEMA)
    for idx in range(pair_count):
        schema[f"cf_batch_pair_jsd_{idx}"] = AggregationMode.MEAN
    for k_idx in range(latent_k):
        schema[f"router_rollout_soft_p_bar_z{k_idx}"] = AggregationMode.LAST
    return schema


def _aggregate(values: list[float], mode: AggregationMode) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if mode == AggregationMode.MEAN:
        return float(np.mean(arr))
    if mode == AggregationMode.SUM:
        return float(np.sum(arr))
    if mode == AggregationMode.LAST:
        return float(arr[-1])
    if mode == AggregationMode.MIN:
        return float(np.min(arr))
    if mode == AggregationMode.MAX:
        return float(np.max(arr))
    return float(np.mean(arr))


class UpdateStatsAccumulator:
    def __init__(self, schema: Mapping[str, AggregationMode]) -> None:
        self.schema = dict(schema)
        self._rows: dict[str, list[float]] = {}

    def record_minibatch(self, telemetry: Mapping[str, float]) -> None:
        for key, value in telemetry.items():
            self._rows.setdefault(key, []).append(float(value))

    def record_epoch(self, metrics: Mapping[str, float]) -> None:
        for key, value in metrics.items():
            self._rows.setdefault(key, []).append(float(value))

    def finalize(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, values in self._rows.items():
            mode = self.schema.get(key, AggregationMode.MEAN)
            out[key] = _aggregate(values, mode)
        return out

    @property
    def raw_rows(self) -> dict[str, list[float]]:
        return self._rows
