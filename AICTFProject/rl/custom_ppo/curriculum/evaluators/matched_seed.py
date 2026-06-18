"""Matched-seed forced-z boundary evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum.context import GateContext, preserve_model_training_mode
from rl.custom_ppo.curriculum.types import (
    GATE_STATUS_ERROR,
    GATE_STATUS_NOT_RUN,
    GateResult,
    gate_family_result_from_bool,
)

_DEFAULT_OPPONENTS: tuple[str, ...] = ("OP5", "OP6", "OP7")
_DEFAULT_SEEDS: tuple[int, ...] = tuple(range(2000, 2020))


@dataclass
class MatchedSeedEvalConfig:
    opponents: list[str] = field(default_factory=lambda: list(_DEFAULT_OPPONENTS))
    seeds: list[int] = field(default_factory=lambda: list(_DEFAULT_SEEDS))
    max_episode_steps: int = 120
    route_ci_z: float = 1.96
    route_ci_threshold: float = 0.02
    wr_spread_threshold: float = 0.03

    @classmethod
    def from_cfg(cls, cfg: PPOConfig) -> MatchedSeedEvalConfig:
        opponents = list(
            getattr(cfg, "curriculum_gate_matched_seed_opponents", None) or _DEFAULT_OPPONENTS
        )
        seed_start = int(getattr(cfg, "curriculum_gate_matched_seed_start", 2000))
        seed_count = int(getattr(cfg, "curriculum_gate_matched_seed_count", 20))
        seeds = list(range(seed_start, seed_start + seed_count))
        return cls(
            opponents=opponents,
            seeds=seeds,
            max_episode_steps=int(getattr(cfg, "curriculum_gate_matched_seed_max_steps", 120)),
        )


def capture_reset_state(core: Any, info: dict[str, Any] | None = None) -> dict[str, Any]:
    _ = info
    return {
        "blue_score": core.blue_score.detach().cpu().tolist(),
        "red_score": core.red_score.detach().cpu().tolist(),
        "blue_flag_pos": core.blue_flag_pos.detach().cpu().tolist(),
        "red_flag_pos": core.red_flag_pos.detach().cpu().tolist(),
        "blue_x": core.blue_x.detach().cpu().tolist(),
        "blue_y": core.blue_y.detach().cpu().tolist(),
        "red_x": core.red_x.detach().cpu().tolist(),
        "red_y": core.red_y.detach().cpu().tolist(),
        "map_layout": str(getattr(core, "map_layout", "")),
    }


def _aggregate_seed_metrics(
    *,
    route_dists: list[float],
    behavior_dists: list[float],
    wr_by_z: dict[int, list[float]],
    latent_k: int,
    num_valid_seeds: int,
    route_ci_z: float,
    route_ci_threshold: float,
) -> dict[str, Any]:
    avg_route = float(np.mean(route_dists)) if route_dists else 0.0
    std_route = float(np.std(route_dists)) if route_dists else 0.0
    se = std_route / np.sqrt(len(route_dists)) if route_dists else 0.0
    ci_low = avg_route - route_ci_z * se
    ci_high = avg_route + route_ci_z * se
    wr_means = [
        float(np.mean(wr_by_z[z])) if wr_by_z[z] else 0.0 for z in range(latent_k)
    ]
    wr_spread = float(max(wr_means) - min(wr_means)) if wr_means else 0.0
    excludes_zero = ci_low > route_ci_threshold and avg_route > route_ci_threshold
    route_std_se = float(se)
    behav_mean = float(np.mean(behavior_dists)) if behavior_dists else 0.0
    behav_std = float(np.std(behavior_dists)) if behavior_dists else 0.0
    behav_se = behav_std / np.sqrt(len(behavior_dists)) if behavior_dists else 0.0
    return {
        "avg_route_distance": avg_route,
        "avg_behavior_distance": behav_mean,
        "route_std_error": route_std_se,
        "behavior_std_error": float(behav_se),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "paired_ci_excludes_zero": bool(excludes_zero),
        "forced_z_performance_spread": wr_spread,
        "effect_size": avg_route,
        "num_seeds": int(num_valid_seeds),
        "num_valid_seeds": int(num_valid_seeds),
    }


def collect_matched_seed_metrics(
    context: GateContext,
    config: MatchedSeedEvalConfig | None = None,
) -> tuple[dict[str, Any], bool]:
    """Run matched-seed forced-z rollouts; return per-opponent metrics and mismatch flag."""
    from rl.training.env_factory import build_training_env

    eval_config = MatchedSeedEvalConfig.from_cfg(context.cfg) if config is None else config
    latent_k = int(context.latent_k)
    op_reports: dict[str, Any] = {}
    any_reset_mismatch = False

    eval_cfg = PPOConfig()
    for key, value in context.cfg.__dict__.items():
        setattr(eval_cfg, key, value)
    eval_cfg.n_envs = 1

    with preserve_model_training_mode(context.eval_model):
        for opp in eval_config.opponents:
            env = build_training_env(eval_cfg, initial_phase="PHASE1", initial_opponent_tag=opp)
            route_dists: list[float] = []
            behavior_dists: list[float] = []
            wr_by_z: dict[int, list[float]] = {z: [] for z in range(latent_k)}
            valid_seed_count = 0
            try:
                for seed in eval_config.seeds:
                    reset_states: list[dict[str, Any]] = []
                    traj_pos: dict[int, np.ndarray] = {}
                    traj_beh: dict[int, np.ndarray] = {}
                    seed_mismatch = False

                    for z_val in range(latent_k):
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        if hasattr(env, "seed"):
                            env.seed(seed)
                        obs = env.reset()
                        core = env.core
                        reset_states.append(capture_reset_state(core))

                        context.configure_fixed_z(z_val)

                        history_pos: list[tuple[float, float]] = []
                        history_beh: list[float] = []
                        done = False
                        step_i = 0
                        blue_won = False
                        while not done and step_i < eval_config.max_episode_steps:
                            act = context.predict(obs)
                            env.step_async(act)
                            obs, _, done_arr, infos = env.step_wait()
                            done = bool(done_arr[0])
                            info0 = infos[0] if infos else {}
                            bx = float(core.blue_x[0].mean().item())
                            by = float(core.blue_y[0].mean().item())
                            history_pos.append((bx, by))
                            history_beh.append(float(info0.get("dense_reward", 0.0)))
                            step_i += 1
                        er = (infos[0].get("episode_result", infos[0]) if infos else {}) or {}
                        bs = int(er.get("blue_score", core.blue_score[0].item()))
                        rs = int(er.get("red_score", core.red_score[0].item()))
                        blue_won = bs > rs
                        traj_pos[z_val] = np.asarray(history_pos, dtype=np.float64)
                        traj_beh[z_val] = np.asarray(history_beh, dtype=np.float64)
                        wr_by_z[z_val].append(1.0 if blue_won else 0.0)

                    if reset_states and any(s != reset_states[0] for s in reset_states[1:]):
                        print(
                            f"[Curriculum Controller] WARNING: seed {seed} reset mismatch across z branches"
                        )
                        seed_mismatch = True
                        any_reset_mismatch = True

                    if seed_mismatch:
                        for z in range(latent_k):
                            if wr_by_z[z]:
                                wr_by_z[z].pop()
                        continue

                    valid_seed_count += 1
                    for z_a in range(latent_k):
                        for z_b in range(z_a + 1, latent_k):
                            t_len = min(len(traj_pos[z_a]), len(traj_pos[z_b]))
                            if t_len > 0:
                                diff = traj_pos[z_a][:t_len] - traj_pos[z_b][:t_len]
                                route_dists.append(float(np.mean(np.linalg.norm(diff, axis=-1))))
                            b_len = min(len(traj_beh[z_a]), len(traj_beh[z_b]))
                            if b_len > 0:
                                behavior_dists.append(
                                    float(
                                        np.mean(
                                            np.abs(traj_beh[z_a][:b_len] - traj_beh[z_b][:b_len])
                                        )
                                    )
                                )

                op_reports[opp] = _aggregate_seed_metrics(
                    route_dists=route_dists,
                    behavior_dists=behavior_dists,
                    wr_by_z=wr_by_z,
                    latent_k=latent_k,
                    num_valid_seeds=valid_seed_count,
                    route_ci_z=eval_config.route_ci_z,
                    route_ci_threshold=eval_config.route_ci_threshold,
                )
            finally:
                env.close()

    return op_reports, any_reset_mismatch


def evaluate_matched_seed_behavior(context: GateContext) -> GateResult:
    """V6I1 boundary gate: matched-seed forced-z behavioral separation."""
    if not bool(getattr(context.cfg, "curriculum_gate_run_boundary_eval", False)):
        return GateResult(
            status=GATE_STATUS_NOT_RUN,
            reason="curriculum_gate_run_boundary_eval=false",
        )

    print("[Curriculum Controller] Matched-seed boundary evaluation...")
    op_reports, any_mismatch = collect_matched_seed_metrics(context)
    if any_mismatch:
        return GateResult(
            status=GATE_STATUS_ERROR,
            reason="matched_seed_reset_mismatch",
            details={"opponents": op_reports},
        )
    if not op_reports:
        return GateResult(
            status=GATE_STATUS_ERROR,
            reason="matched_seed_eval_failed",
        )

    all_passed = True
    for rep in op_reports.values():
        excludes_zero = bool(rep.get("paired_ci_excludes_zero", False))
        wr_spread = float(rep.get("forced_z_performance_spread", 0.0))
        if not (excludes_zero and wr_spread >= 0.03):
            all_passed = False

    return gate_family_result_from_bool(
        all_passed,
        details={"opponents": op_reports, "matched_eval_passed": all_passed},
    )


__all__ = [
    "MatchedSeedEvalConfig",
    "capture_reset_state",
    "collect_matched_seed_metrics",
    "evaluate_matched_seed_behavior",
]
