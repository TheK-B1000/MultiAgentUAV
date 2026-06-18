"""Pairwise intervention EMA state with duplicate-step rejection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PairwiseEMAState:
    values: np.ndarray
    valid_updates: int = 0
    last_update_step: int = -1
    consecutive_passes: int = 0

    @classmethod
    def zeros(cls, pair_count: int) -> PairwiseEMAState:
        return cls(values=np.zeros((int(pair_count),), dtype=np.float32))

    def update(
        self,
        pair_values: list[float] | tuple[float, ...] | np.ndarray,
        *,
        global_step: int,
        alpha: float,
        pass_predicate,
    ) -> bool:
        step = int(global_step)
        if self.valid_updates > 0 and step <= int(self.last_update_step):
            return False
        arr = np.asarray(pair_values, dtype=np.float32).reshape(-1)
        if arr.shape != self.values.shape or not np.all(np.isfinite(arr)):
            self.consecutive_passes = 0
            return False
        if self.valid_updates <= 0:
            self.values = arr.copy()
        else:
            self.values = (1.0 - float(alpha)) * self.values + float(alpha) * arr
        self.valid_updates += 1
        self.last_update_step = step
        if bool(pass_predicate(self.values)):
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0
        return True


@dataclass
class InterventionState:
    """Legacy macro JSD, actor CF JSD, and macro rollout EMA tracks."""

    pair_count: int
    legacy_macro: PairwiseEMAState = field(init=False)
    cf_actor: PairwiseEMAState = field(init=False)
    macro_rollout: PairwiseEMAState = field(init=False)
    jsd_gate_consecutive_updates: int = 0
    actor_intervention_consecutive_updates: int = 0

    def __post_init__(self) -> None:
        k = int(self.pair_count)
        self.legacy_macro = PairwiseEMAState.zeros(k)
        self.cf_actor = PairwiseEMAState.zeros(k)
        self.macro_rollout = PairwiseEMAState.zeros(k)

    @staticmethod
    def pair_count_for_latent_k(latent_k: int) -> int:
        k = int(latent_k)
        return k * (k - 1) // 2


class InterventionEMAController:
    """Migrates CF / macro pairwise JSD EMA updates off the monolith."""

    def __init__(self, host) -> None:
        self.host = host

    @staticmethod
    def _coerce_six_finite_pair_values(
        pair_values: list[float] | tuple[float, ...],
    ) -> np.ndarray | None:
        if len(pair_values) != 6:
            return None
        arr = np.asarray(pair_values, dtype=np.float32)
        if arr.shape != (6,) or not np.all(np.isfinite(arr)):
            return None
        return arr

    def update_cf_pair_jsd_ema(self, pair_values: list[float], timestep: int) -> bool:
        host = self.host
        step = int(timestep)
        if int(host.cf_pair_jsd_valid_updates) > 0 and step <= int(host.cf_pair_jsd_last_update_step):
            return False
        pair_arr = self._coerce_six_finite_pair_values(pair_values)
        if pair_arr is None:
            host.actor_intervention_consecutive_updates = 0
            return False
        cfg = host.trainer.cfg
        alpha = float(cfg.actor_jsd_ema_decay)
        if int(host.cf_pair_jsd_valid_updates) <= 0:
            host.cf_pair_jsd_ema = pair_arr.copy()
        else:
            host.cf_pair_jsd_ema = (1.0 - alpha) * host.cf_pair_jsd_ema + alpha * pair_arr
        host.cf_pair_jsd_valid_updates = int(host.cf_pair_jsd_valid_updates) + 1
        host.cf_pair_jsd_last_update_step = step
        margin = float(cfg.actor_jsd_margin)
        floor = float(cfg.actor_jsd_floor_fraction) * margin
        min_pairs = int(cfg.actor_jsd_min_passing_pairs)
        num_above = int(np.sum(host.cf_pair_jsd_ema >= margin))
        min_ema = float(np.min(host.cf_pair_jsd_ema))
        update_ok = num_above >= min_pairs and min_ema >= floor
        if update_ok:
            host.actor_intervention_consecutive_updates += 1
        else:
            host.actor_intervention_consecutive_updates = 0
        return True

    def update_macro_pair_jsd_ema(self, pair_values: list[float], timestep: int) -> bool:
        host = self.host
        step = int(timestep)
        if int(host.macro_pair_jsd_valid_updates) > 0 and step <= int(host.macro_pair_jsd_last_update_step):
            return False
        pair_arr = self._coerce_six_finite_pair_values(pair_values)
        if pair_arr is None:
            return False
        cfg = host.trainer.cfg
        alpha = float(cfg.macro_jsd_ema_decay)
        if int(host.macro_pair_jsd_valid_updates) <= 0:
            host.macro_pair_jsd_ema = pair_arr.copy()
        else:
            host.macro_pair_jsd_ema = (1.0 - alpha) * host.macro_pair_jsd_ema + alpha * pair_arr
        host.macro_pair_jsd_valid_updates = int(host.macro_pair_jsd_valid_updates) + 1
        host.macro_pair_jsd_last_update_step = step
        return True

    def update_intervention_gate_from_profile(self, profile_stats: dict[str, float]) -> bool:
        from rl.custom_ppo.gate_protocol import is_v6i2_gate_protocol
        from rl.custom_ppo.v6i1_cf_loss import extract_forced_z_pair_values

        pair_vals = extract_forced_z_pair_values(profile_stats)
        if pair_vals is None:
            if not is_v6i2_gate_protocol(self.host.trainer.cfg):
                self.host.jsd_gate_consecutive_updates = 0
            return False
        if is_v6i2_gate_protocol(self.host.trainer.cfg):
            timestep = int(getattr(self.host.trainer, "global_step", -1))
            return self.update_macro_pair_jsd_ema(pair_vals, timestep)
        return self._update_legacy_macro_intervention_ema(profile_stats)

    def update_actor_intervention_gate_from_cf_pairs(self, pair_vals: list[float]) -> bool:
        timestep = int(getattr(self.host.trainer, "global_step", -1))
        return self.update_cf_pair_jsd_ema(pair_vals, timestep)

    def update_macro_pair_jsd_ema_from_profile(self, profile_stats: dict[str, float]) -> bool:
        from rl.custom_ppo.v6i1_cf_loss import extract_forced_z_pair_values

        pair_vals = extract_forced_z_pair_values(profile_stats)
        if pair_vals is None:
            return False
        timestep = int(getattr(self.host.trainer, "global_step", -1))
        return self.update_macro_pair_jsd_ema(pair_vals, timestep)

    def _update_legacy_macro_intervention_ema(self, profile_stats: dict[str, float]) -> bool:
        from rl.custom_ppo.v6i1_cf_loss import extract_forced_z_pair_values

        host = self.host
        pair_vals = extract_forced_z_pair_values(profile_stats)
        if pair_vals is None:
            host.jsd_gate_consecutive_updates = 0
            return False
        trainer = host.trainer
        margin = float(getattr(trainer.cfg, "latent_cf_jsd_margin", 0.01))
        alpha = float(getattr(trainer.cfg, "latent_cf_jsd_ema_alpha", 0.10))
        pair_arr = np.asarray(pair_vals, dtype=np.float32)
        host.pair_jsd_ema = (1.0 - alpha) * host.pair_jsd_ema + alpha * pair_arr
        host.pairwise_ema_valid_updates = int(host.pairwise_ema_valid_updates) + 1
        host.pairwise_ema_last_update_step = int(getattr(trainer, "global_step", -1))
        num_valid = int(np.sum(host.pair_jsd_ema >= margin))
        min_jsd = float(np.min(host.pair_jsd_ema)) if host.pair_jsd_ema.size else 0.0
        update_ok = num_valid >= 5 and min_jsd >= 0.5 * margin
        if update_ok:
            host.jsd_gate_consecutive_updates += 1
        else:
            host.jsd_gate_consecutive_updates = 0
        return True
