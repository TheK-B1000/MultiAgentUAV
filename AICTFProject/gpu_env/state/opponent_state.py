"""Opponent and dynamics configuration state.

Manages the per-env opponent phase/kind/key tracking and exposes the public
setter/getter API (``set_phase``, ``set_next_opponent``, ``set_dynamics_config``,
etc.).  Also owns ``_apply_dynamics_bool`` which bridges ``set_dynamics_config``
to bool tensors.

Note: ``_apply_dynamics_tensor`` lives in ``_DynamicsMixin`` (``_dynamics.py``).
``set_dynamics_config`` calls it via ``self`` — this works because
``BatchedCTFCore`` inherits from both mixins.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch

from .._paths import _resolve_snapshot_path


class _OpponentStateMixin:
    """Manages opponent phase/kind tracking and dynamics config API."""

    def set_phase(
        self, phase: str, env_indices: Optional[Sequence[int]] = None
    ) -> None:
        phase_s = str(phase).upper()
        for env_i in self._normalize_env_indices(env_indices).detach().cpu().tolist():
            self._phase[env_i] = phase_s
        self._phase_tensor_cache.clear()
        self._red_control_mask_dirty = True

    def set_league_mode(
        self, league_mode: bool, env_indices: Optional[Sequence[int]] = None
    ) -> None:
        idx = self._normalize_env_indices(env_indices)
        if idx.numel() > 0:
            self._league_mode[idx] = bool(league_mode)

    def set_stress_schedule(
        self,
        schedule: Optional[dict],
        env_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self._stress_schedule = schedule

    def set_next_opponent(
        self,
        kind: str,
        key: str,
        env_indices: Optional[Sequence[int]] = None,
    ) -> None:
        kind_s = str(kind).upper()
        key_s = str(key) if kind_s == "SNAPSHOT" else str(key).upper()
        idx = self._normalize_env_indices(env_indices)
        for env_i in idx.detach().cpu().tolist():
            self._opponent_kind[env_i] = kind_s
            self._opponent_key[env_i] = key_s
        self._red_control_mask_dirty = True
        try:
            mask = torch.zeros((self.B,), dtype=torch.bool, device=self.device)
            if idx.numel() > 0:
                mask[idx] = True
            self._apply_opponent_params_for_mask(mask)
        except Exception as e:
            import warnings
            warnings.warn(
                f"BatchedCTFCore: set_next_opponent({key_s!r}) failed to apply params: {e}. "
                "Red team may still use previous opponent params; targeted opponent changes may lag."
            )

    def get_opponent_key(
        self, env_indices: Optional[Sequence[int]] = None
    ) -> str:
        """Return current red opponent key (OP1…OP7 scripted tags). For eval verification."""
        idx = self._normalize_env_indices(env_indices)
        if idx.numel() == 0:
            return "OP3"
        return str(self._opponent_key[int(idx[0].item())])

    def set_dynamics_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        if not isinstance(cfg, dict):
            return
        if "rules_profile" in cfg:
            self.rules_profile = str(cfg["rules_profile"]).upper().strip()
        if "aquaticus_profile" in cfg:
            self.cfg.aquaticus_profile = bool(cfg["aquaticus_profile"])
        for key in (
            "max_speed_cps",
            "max_accel_cps2",
            "max_yaw_rate_rps",
            "min_turn_radius_cells",
            "current_strength_cps",
            "drift_sigma_cells",
            "sensor_range_cells",
            "sensor_noise_sigma_cells",
            "sensor_dropout_prob",
        ):
            if key in cfg:
                setattr(self.cfg, key, float(cfg[key]))
        self._apply_dynamics_tensor(cfg, "deception_prob", "red_deception_prob", 0.0, 1.0)
        self._apply_dynamics_tensor(cfg, "speed_mult", "red_speed_mult", 0.25, 2.0)
        self._apply_dynamics_tensor(cfg, "attacker_style", "red_attacker_style", 0, 1, torch.int32)
        self._apply_dynamics_tensor(cfg, "defender_style", "red_defender_style", 0, 1, torch.int32)
        self._apply_dynamics_tensor(cfg, "role_switch_prob", "red_role_switch_prob", 0.0, 1.0)
        self._apply_dynamics_bool(cfg, "coordinated_attack", "red_coordinated_attack")
        self._apply_dynamics_tensor(
            cfg, "attack_sync_window", "red_attack_sync_window", 0, 32, torch.int32
        )

    def _apply_dynamics_bool(
        self, cfg: Dict[str, Any], key: str, attr: str
    ) -> None:
        if key not in cfg:
            return
        val = cfg[key]
        tensor = getattr(self, attr)
        if isinstance(val, torch.Tensor):
            t = val.to(device=self.device, dtype=torch.bool).reshape(-1)
            if t.numel() == self.B:
                tensor.copy_(t)
            else:
                tensor.fill_(bool(t.reshape(-1)[0].item()))
        else:
            tensor.fill_(bool(val))
