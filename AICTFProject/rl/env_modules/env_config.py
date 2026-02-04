"""
Config plumbing module for CTFGameFieldSB3Env.

Handles all configuration methods:
- Dynamics config (speed, acceleration, yaw rate)
- Disturbance config (current, drift)
- Robotics constraints (action delay, actuation noise)
- Sensor config (range, noise, dropout)
- Physics tag/enabled
- Phase and stress schedule
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class EnvConfigManager:
    """Manages all environment configuration state and forwarding to GameField."""

    def __init__(self) -> None:
        self._dynamics_config: Optional[dict] = None
        self._disturbance_config: Optional[dict] = None
        self._robotics_config: Optional[dict] = None
        self._sensor_config: Optional[dict] = None
        self._physics_tag: Optional[str] = None
        self._phase_name: str = "OP1"
        self._stress_schedule: Optional[dict] = None

    def set_dynamics_config(self, cfg: Optional[dict], game_field: Optional[Any] = None) -> None:
        """Set dynamics config (speed, acceleration, yaw rate)."""
        self._dynamics_config = None if cfg is None else dict(cfg)
        if game_field is None:
            return
        if hasattr(game_field, "set_dynamics_config"):
            try:
                game_field.set_dynamics_config(self._dynamics_config)
                return
            except Exception:
                pass
        gm = getattr(game_field, "manager", None)
        if gm is not None and hasattr(gm, "set_dynamics_config"):
            try:
                gm.set_dynamics_config(self._dynamics_config)
            except Exception:
                pass

    def get_dynamics_config(self) -> Optional[dict]:
        return None if self._dynamics_config is None else dict(self._dynamics_config)

    def set_disturbance_config(self, *args, game_field: Optional[Any] = None, **kwargs) -> None:
        """Set disturbance config (current_strength_cps, drift_sigma_cells)."""
        cfg = None
        if args and isinstance(args[0], dict):
            cfg = dict(args[0])
        elif "cfg" in kwargs and isinstance(kwargs["cfg"], dict):
            cfg = dict(kwargs["cfg"])
        if cfg is None:
            cfg = dict(kwargs)
        current_strength = cfg.pop("current_strength", None)
        drift_sigma = cfg.pop("drift_sigma", None)
        current_strength_cps = cfg.pop("current_strength_cps", None)
        drift_sigma_cells = cfg.pop("drift_sigma_cells", None)
        if current_strength_cps is None:
            current_strength_cps = current_strength
        if drift_sigma_cells is None:
            drift_sigma_cells = drift_sigma
        self._disturbance_config = {
            "current_strength_cps": 0.0 if current_strength_cps is None else float(current_strength_cps),
            "drift_sigma_cells": 0.0 if drift_sigma_cells is None else float(drift_sigma_cells),
        }
        if game_field is None:
            return
        if hasattr(game_field, "set_disturbance_config"):
            try:
                game_field.set_disturbance_config(
                    self._disturbance_config["current_strength_cps"],
                    self._disturbance_config["drift_sigma_cells"],
                )
                return
            except Exception:
                pass
        if hasattr(game_field, "set_disturbance_config_dict"):
            try:
                game_field.set_disturbance_config_dict(self._disturbance_config)
            except Exception:
                pass

    def get_disturbance_config(self) -> Optional[dict]:
        return None if self._disturbance_config is None else dict(self._disturbance_config)

    def set_robotics_constraints(self, *args, game_field: Optional[Any] = None, **kwargs) -> None:
        """Set robotics constraints (action_delay_steps, actuation_noise_sigma)."""
        cfg = None
        if args and isinstance(args[0], dict):
            cfg = dict(args[0])
        elif "cfg" in kwargs and isinstance(kwargs["cfg"], dict):
            cfg = dict(kwargs["cfg"])
        if cfg is None:
            cfg = dict(kwargs)
        action_delay_steps = cfg.get("action_delay_steps", 0)
        actuation_noise_sigma = cfg.get("actuation_noise_sigma", 0.0)
        self._robotics_config = {
            "action_delay_steps": int(action_delay_steps),
            "actuation_noise_sigma": float(actuation_noise_sigma),
        }
        if game_field is None:
            return
        if hasattr(game_field, "set_robotics_constraints"):
            try:
                game_field.set_robotics_constraints(
                    self._robotics_config["action_delay_steps"],
                    self._robotics_config["actuation_noise_sigma"],
                )
                return
            except Exception:
                pass
        if hasattr(game_field, "set_robotics_constraints_dict"):
            try:
                game_field.set_robotics_constraints_dict(self._robotics_config)
            except Exception:
                pass

    def get_robotics_constraints(self) -> Optional[dict]:
        return None if self._robotics_config is None else dict(self._robotics_config)

    def set_sensor_config(self, *args, game_field: Optional[Any] = None, **kwargs) -> None:
        """Set sensor config (sensor_range_cells, sensor_noise_sigma_cells, sensor_dropout_prob)."""
        cfg = None
        if args and isinstance(args[0], dict):
            cfg = dict(args[0])
        elif "cfg" in kwargs and isinstance(kwargs["cfg"], dict):
            cfg = dict(kwargs["cfg"])
        if cfg is None:
            cfg = dict(kwargs)
        sensor_range = cfg.get("sensor_range", cfg.get("sensor_range_cells", 9999.0))
        sensor_noise_sigma = cfg.get("sensor_noise_sigma", cfg.get("sensor_noise_sigma_cells", 0.0))
        sensor_dropout_prob = cfg.get("sensor_dropout_prob", 0.0)
        self._sensor_config = {
            "sensor_range_cells": float(sensor_range),
            "sensor_noise_sigma_cells": float(sensor_noise_sigma),
            "sensor_dropout_prob": float(sensor_dropout_prob),
        }
        if game_field is None:
            return
        if hasattr(game_field, "set_sensor_config"):
            try:
                game_field.set_sensor_config(
                    self._sensor_config["sensor_range_cells"],
                    self._sensor_config["sensor_noise_sigma_cells"],
                    self._sensor_config["sensor_dropout_prob"],
                )
                return
            except Exception:
                pass
        if hasattr(game_field, "set_sensor_config_dict"):
            try:
                game_field.set_sensor_config_dict(self._sensor_config)
            except Exception:
                pass

    def get_sensor_config(self) -> Optional[dict]:
        return None if self._sensor_config is None else dict(self._sensor_config)

    def set_physics_tag(self, tag: str, game_field: Optional[Any] = None) -> None:
        self._physics_tag = str(tag)
        if game_field is None:
            return
        if hasattr(game_field, "set_physics_tag"):
            try:
                game_field.set_physics_tag(self._physics_tag)
            except Exception:
                pass

    def set_physics_enabled(self, enabled: bool, game_field: Optional[Any] = None) -> None:
        """Turn ASV kinematics + maritime sensors on/off."""
        if game_field is None:
            return
        if hasattr(game_field, "set_physics_enabled"):
            try:
                game_field.set_physics_enabled(bool(enabled))
            except Exception:
                pass

    def set_phase(self, phase: str, game_field: Optional[Any] = None) -> None:
        """Set curriculum phase (OP1/OP2/OP3)."""
        from rl.curriculum import phase_from_tag, VALID_PHASES
        raw = str(phase).upper().strip()
        canonical = phase_from_tag(raw)
        assert canonical in VALID_PHASES, f"phase_from_tag({phase!r}) returned {canonical!r}"
        self._phase_name = canonical
        if game_field is not None and hasattr(game_field, "manager"):
            try:
                game_field.manager.set_phase(self._phase_name)
            except Exception:
                pass
        self._apply_stress_for_phase(self._phase_name, game_field)

    def get_phase(self) -> str:
        return self._phase_name

    def set_stress_schedule(self, schedule: Optional[dict]) -> None:
        """Set phase -> stress config mapping."""
        self._stress_schedule = None if schedule is None else dict(schedule)

    def _apply_stress_for_phase(self, phase: str, game_field: Optional[Any] = None) -> None:
        """Apply environment stress and naval realism for this phase if stress schedule is set."""
        if game_field is None or not self._stress_schedule:
            return
        cfg = self._stress_schedule.get(str(phase).upper())
        if not cfg or not isinstance(cfg, dict):
            return
        try:
            if "physics_enabled" in cfg:
                self.set_physics_enabled(bool(cfg.get("physics_enabled", False)), game_field)
            if cfg.get("relaxed_dynamics"):
                game_field.set_dynamics_config(
                    max_speed_cps=float(cfg.get("max_speed_cps", 2.8)),
                    max_accel_cps2=float(cfg.get("max_accel_cps2", 2.5)),
                    max_yaw_rate_rps=float(cfg.get("max_yaw_rate_rps", 5.0)),
                )
            elif "max_speed_cps" in cfg or "max_accel_cps2" in cfg or "max_yaw_rate_rps" in cfg:
                boat_cfg = getattr(game_field, "boat_cfg", None)
                game_field.set_dynamics_config(
                    max_speed_cps=float(cfg.get("max_speed_cps", getattr(boat_cfg, "max_speed_cps", 2.2) if boat_cfg else 2.2)),
                    max_accel_cps2=float(cfg.get("max_accel_cps2", getattr(boat_cfg, "max_accel_cps2", 2.0) if boat_cfg else 2.0)),
                    max_yaw_rate_rps=float(cfg.get("max_yaw_rate_rps", getattr(boat_cfg, "max_yaw_rate_rps", 4.0) if boat_cfg else 4.0)),
                )
            if "current_strength_cps" in cfg or "drift_sigma_cells" in cfg:
                self.set_disturbance_config(
                    current_strength_cps=float(cfg.get("current_strength_cps", 0.0)),
                    drift_sigma_cells=float(cfg.get("drift_sigma_cells", 0.0)),
                    game_field=game_field,
                )
            if "action_delay_steps" in cfg:
                self.set_robotics_constraints(
                    action_delay_steps=int(cfg.get("action_delay_steps", 0)),
                    actuation_noise_sigma=float(cfg.get("actuation_noise_sigma", 0.0)),
                    game_field=game_field,
                )
            if "sensor_noise_sigma_cells" in cfg or "sensor_dropout_prob" in cfg:
                boat_cfg = getattr(game_field, "boat_cfg", None)
                self.set_sensor_config(
                    sensor_range_cells=float(cfg.get("sensor_range_cells", getattr(boat_cfg, "sensor_range_cells", 9999.0) if boat_cfg else 9999.0)),
                    sensor_noise_sigma_cells=float(cfg.get("sensor_noise_sigma_cells", 0.0)),
                    sensor_dropout_prob=float(cfg.get("sensor_dropout_prob", 0.0)),
                    game_field=game_field,
                )
        except Exception:
            pass

    def apply_all_configs(self, game_field: Any) -> None:
        """Apply all stored configs to game_field (called after reset)."""
        if self._dynamics_config is not None:
            self.set_dynamics_config(self._dynamics_config, game_field)
        if self._disturbance_config is not None:
            self.set_disturbance_config(game_field=game_field, **self._disturbance_config)
        if self._robotics_config is not None:
            self.set_robotics_constraints(game_field=game_field, **self._robotics_config)
        if self._sensor_config is not None:
            self.set_sensor_config(game_field=game_field, **self._sensor_config)
        if self._physics_tag is not None:
            self.set_physics_tag(self._physics_tag, game_field)
        self._apply_stress_for_phase(self._phase_name, game_field)


__all__ = ["EnvConfigManager"]
