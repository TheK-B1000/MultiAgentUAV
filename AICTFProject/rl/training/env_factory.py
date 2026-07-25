"""Training environment construction for the local PPO trainer.

``build_training_env`` is the single entry point used by :mod:`rl.train_ppo`
to assemble a fully wired :class:`GPUCTFVecEnv`: it materializes the
``GPUFieldConfig`` (including domain-randomization knobs and optional reward
overrides), applies the stress schedule, sets the initial scripted opponent,
and sources opponent dynamics parameters from
:func:`opponent_params.sample_batched_opponent_params`.

Reproducibility contract: this module only reads ``PPOConfig`` attribute
values; field names and defaults live in :mod:`rl.config.ppo_config`. Reward
override keys here mirror ``GPUFieldConfig`` / ``RewardConfig`` 1-to-1.
"""

from __future__ import annotations

import os
import sys
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ``AICTFProject/`` -> two parents up from ``AICTFProject/rl/training/``;
# defensive in case env_factory is imported before ``rl.train_ppo`` has run
# the same sys.path injection.
_AICTF_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _AICTF_DIR not in sys.path:
    sys.path.insert(0, _AICTF_DIR)

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
from gpu_env._config import RouterRewardConfig
from opponent_params import sample_batched_opponent_params
from rl.config.ppo_config import PPOConfig
from rl.curriculum import phase_from_tag
from rl.stress_schedule import STRESS_BY_PHASE


def _gpu_env_reward_kwargs(cfg: PPOConfig) -> dict[str, Any]:
    """Map optional ``PPOConfig`` reward knobs onto ``GPUFieldConfig`` / ``RewardConfig`` field names."""
    pairs = (
        ("win_team_reward", getattr(cfg, "env_win_team_reward", None)),
        ("draw_team_penalty", getattr(cfg, "env_draw_team_penalty", None)),
        ("lose_team_punish", getattr(cfg, "env_lose_team_punish", None)),
        ("action_failed_punishment", getattr(cfg, "env_action_failed_punishment", None)),
        ("dense_weight", getattr(cfg, "env_dense_weight", None)),
        ("sparse_weight", getattr(cfg, "env_sparse_weight", None)),
        ("reward_scale", getattr(cfg, "env_reward_scale", None)),
        ("reward_clip", getattr(cfg, "env_reward_clip", None)),
        ("stalemate_penalty", getattr(cfg, "env_stalemate_penalty", None)),
        ("stalemate_max_steps", getattr(cfg, "env_stalemate_max_steps", None)),
        ("surface_score_margin_coef", getattr(cfg, "env_surface_score_margin_coef", None)),
        ("surface_blue_capture_tempo_bonus", getattr(cfg, "env_surface_blue_capture_tempo_bonus", None)),
        ("surface_red_flag_touch_penalty", getattr(cfg, "env_surface_red_flag_touch_penalty", None)),
        ("surface_red_carrier_progress_penalty", getattr(cfg, "env_surface_red_carrier_progress_penalty", None)),
        ("surface_blue_near_cap_bonus", getattr(cfg, "env_surface_blue_near_cap_bonus", None)),
    )
    out: dict[str, Any] = {}
    for name, raw in pairs:
        if raw is None:
            continue
        if name == "stalemate_max_steps":
            out[name] = max(1, int(raw))
        else:
            out[name] = float(raw)
    return out


def _apply_initial_opponent_params(
    env: GPUCTFVecEnv,
    cfg: PPOConfig,
    gpu_cfg: GPUFieldConfig,
    *,
    opponent_tag: str | None = None,
    phase: str | None = None,
) -> None:
    try:
        key = str(opponent_tag or cfg.fixed_opponent_tag).upper()
        phase_key = str(phase or phase_from_tag(key)).upper()
        opp_params = sample_batched_opponent_params(
            kind="SCRIPTED",
            key=key,
            phase=phase_key,
            n_agents=gpu_cfg.max_red_agents,
            batch_size=gpu_cfg.n_envs,
            device=gpu_cfg.device,
        )
        dyn_cfg: dict[str, object] = {
            key: value
            for key, value in opp_params.items()
            if key
            in {
                "deception_prob",
                "speed_mult",
                "attacker_style",
                "defender_style",
                "role_switch_prob",
                "coordinated_attack",
                "attack_sync_window",
            }
        }
        if dyn_cfg:
            env.env_method("set_dynamics_config", dyn_cfg)
    except Exception as exc:
        print(f"[PPO] opponent_params sampling failed; using defaults: {exc}")


def build_training_env(
    cfg: PPOConfig,
    *,
    initial_phase: str,
    initial_opponent_tag: str,
) -> GPUCTFVecEnv:
    """Construct and prime a ``GPUCTFVecEnv`` for the local PPO trainer.

    Mirrors the inline env construction that used to live at the top of
    :func:`rl.train_ppo.train_ppo`: builds the ``GPUFieldConfig`` (including
    domain randomization + reward overrides), installs the stress schedule
    and rules profile, sets the initial phase + scripted opponent tag, and
    seeds opponent dynamics params via
    :func:`_apply_initial_opponent_params`.

    Caller owns ``env.close()``. If any post-construction setup raises, the
    env is closed before the exception propagates so we don't leak the
    underlying CUDA buffers.
    """
    max_agents = max(1, int(getattr(cfg, "max_blue_agents", 2)))
    reward_kw = _gpu_env_reward_kwargs(cfg)
    if reward_kw:
        parts = [f"{k}={v}" for k, v in sorted(reward_kw.items())]
        print("[PPO] GPU env reward overrides: " + ", ".join(parts))
    map_pool = tuple(getattr(cfg, "map_pool", ()) or ())
    cells = tuple(getattr(cfg, "training_cell_distribution", ()) or ())
    if cells:
        print(
            "[PPO] map_pool: overridden by training_cell_distribution "
            "(joint opponent×map sampling via pre-reset hook)."
        )
    elif map_pool:
        print("[PPO] map_pool (per-episode uniform sample): " + ", ".join(map_pool))
    rrc: RouterRewardConfig | None = None
    if bool(getattr(cfg, "router_reward_enabled", False)):
        rrc = RouterRewardConfig(
            enabled=True,
            win_weight=float(getattr(cfg, "router_reward_win_weight", 1.0)),
            flag_cap_weight=float(getattr(cfg, "router_reward_flag_cap_weight", 0.5)),
            sparse_weight=float(getattr(cfg, "router_reward_sparse_weight", 0.2)),
            scale=float(getattr(cfg, "router_reward_scale", 1.0)),
            normalize=bool(getattr(cfg, "router_reward_normalize", True)),
        )
        print(
            f"[PPO] V6I7 router reward enabled: win_w={rrc.win_weight}, "
            f"flag_w={rrc.flag_cap_weight}, sparse_w={rrc.sparse_weight}, "
            f"scale={rrc.scale}, normalize={rrc.normalize}"
        )
    obstacle_raw = getattr(cfg, "obstacle_obs_channel", None)
    obstacle_kw: dict[str, Any] = {}
    if obstacle_raw is not None:
        obstacle_kw["obstacle_obs_channel"] = bool(obstacle_raw)
        print(
            "[PPO] obstacle_obs_channel override: "
            f"{bool(obstacle_raw)} "
            f"(map_layout={str(getattr(cfg, 'map_layout', 'map_a_open')).lower()})"
        )
    gpu_cfg = GPUFieldConfig(
        n_envs=max(1, int(cfg.n_envs)),
        n_agents_per_team=max_agents,
        map_set=str(getattr(cfg, "map_set", "train")).lower(),
        map_layout=str(getattr(cfg, "map_layout", "map_a_open")).lower(),
        map_pool=tuple(getattr(cfg, "map_pool", ()) or ()),
        max_decision_steps=max(1, int(cfg.max_decision_steps)),
        aquaticus_profile=True,
        rules_profile="OURS",
        device=str(cfg.device),
        seed=int(cfg.seed),
        train_domain_randomization=bool(getattr(cfg, "train_domain_randomization", False)),
        dr_sensor_noise_sigma_max=float(getattr(cfg, "dr_sensor_noise_sigma_max", 0.12)),
        dr_sensor_dropout_max=float(getattr(cfg, "dr_sensor_dropout_max", 0.08)),
        dr_blue_speed_jitter=float(getattr(cfg, "dr_blue_speed_jitter", 0.12)),
        router_reward_config=rrc,
        **obstacle_kw,
        **reward_kw,
    )
    print(
        "[PPO] Trainer reward target mirrors GPU RewardConfig: "
        f"dense_weight={float(gpu_cfg.dense_weight):.3f}, "
        f"reward_scale={float(gpu_cfg.reward_scale):.3f}, "
        f"reward_clip={float(gpu_cfg.reward_clip):.3f}, "
        f"stalemate_penalty={float(gpu_cfg.stalemate_penalty):.3f}."
    )
    env = GPUCTFVecEnv(gpu_cfg)
    try:
        env.env_method("set_stress_schedule", STRESS_BY_PHASE)
        env.env_method("set_dynamics_config", {"rules_profile": "OURS"})
        env.env_method("set_phase", initial_phase)
        env.env_method("set_next_opponent", "SCRIPTED", initial_opponent_tag)
        _apply_initial_opponent_params(
            env,
            cfg,
            gpu_cfg,
            opponent_tag=initial_opponent_tag,
            phase=initial_phase,
        )
    except BaseException:
        try:
            env.close()
        except Exception as close_exc:
            print(f"[PPO] WARNING: env.close() during setup failure raised: {close_exc}")
        raise
    return env


__all__ = [
    "_apply_initial_opponent_params",
    "_gpu_env_reward_kwargs",
    "build_training_env",
]
