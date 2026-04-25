"""
Shared custom PPO rollout for win-rate scripts and plot_eval_metrics.py.

Keeps episode stepping, opponent setup, and aggregates identical so
success_rate (eval table) matches W/L/D win rate (wins / n_episodes).
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from rl.global_state import coarse_game_phase_from_global_state

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _policy_entropy_first_step(model: Any, single_obs: dict) -> float:
    """Mean policy entropy at one observation (stochastic policy, eval mode)."""
    return float(model.entropy(single_obs))


def _strategy_phase_from_global_state(global_state: Any) -> str:
    """Compact phase label from the 14-float global state flag-status bits."""
    if global_state is None:
        return "unknown"
    return coarse_game_phase_from_global_state(global_state)


def run_eval_episodes(
    model_path: str,
    env: Any,
    n_episodes: int,
    device: str,
    opponent: str,
    *,
    deterministic: bool = True,
    record_entropy: bool = False,
    progress_every: int = 0,
) -> list[dict]:
    """Run n_episodes; each dict has success, steps, return, scores, etc. (same as plot_eval_metrics).

    If deterministic is False, uses stochastic policy actions (sampled); default True matches greedy argmax.
    If record_entropy is True, each episode dict includes policy_entropy (first-step mean entropy).
    If progress_every > 0, prints after episode 1, then every progress_every episodes, and on the last
    (flush=True) so long 8v8 runs do not look hung.
    """
    from rl.custom_ppo import load_custom_ppo_policy

    model = load_custom_ppo_policy(model_path, env.observation_space, env.action_space, device=device)
    if progress_every > 0:
        print(
            f"  checkpoint loaded; {n_episodes} episodes (first result prints after ep 1; 8v8 is slow)",
            flush=True,
        )

    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", "SCRIPTED", opponent)
        # Match training (train_ppo): phase-indexed current/drift via stress schedule.
        try:
            from rl.stress_schedule import STRESS_BY_PHASE

            env.env_method("set_stress_schedule", STRESS_BY_PHASE)
        except Exception:
            pass
        out = env.env_method("get_opponent_key")
        actual = (out[0] if out else "").strip().upper()
        requested = str(opponent).strip().upper()
        if actual != requested:
            import warnings

            warnings.warn(
                f"Opponent mismatch: requested {requested!r}, core has {actual!r}. "
                "Eval may not be against the intended opponent."
            )
    except Exception as e:
        import warnings

        warnings.warn(
            f"Failed to set opponent to {opponent!r}: {e}. "
            "Red team may still be using the previous opponent - OP3 vs OP4 results can look identical."
        )

    episodes: list[dict] = []
    obs = env.reset()
    if hasattr(model, "reset_strategy"):
        model.reset_strategy()

    for _ in range(n_episodes):
        ep_return = 0.0
        steps = 0
        ep_entropy_first = float("nan")
        strategy_counts: dict[int, int] = {}
        strategy_prev: int | None = None
        strategy_switches = 0
        strategy_resamples = 0
        strategy_steps = 0
        strategy_entropy_sum = 0.0
        strategy_k = 0
        strategy_phase_counts: dict[str, dict[int, int]] = {}
        while True:
            single = {
                k: v[0] if hasattr(v, "shape") and len(v.shape) > 1 and v.shape[0] == 1 else v
                for k, v in obs.items()
            }
            try:
                single["global_state"] = env.state()[0]
            except Exception:
                pass
            strategy_phase = _strategy_phase_from_global_state(single.get("global_state"))
            if record_entropy and steps == 0:
                try:
                    ep_entropy_first = _policy_entropy_first_step(model, single)
                except Exception:
                    ep_entropy_first = float("nan")
            act, _ = model.predict(single, deterministic=deterministic)
            strategy_info = model.strategy_info() if hasattr(model, "strategy_info") else {}
            if "strategy" in strategy_info:
                strategy = int(strategy_info["strategy"])
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                phase_counts = strategy_phase_counts.setdefault(strategy_phase, {})
                phase_counts[strategy] = phase_counts.get(strategy, 0) + 1
                if strategy_prev is not None and strategy != strategy_prev:
                    strategy_switches += 1
                strategy_prev = strategy
                if bool(strategy_info.get("strategy_resampled", False)):
                    strategy_resamples += 1
                strategy_steps += 1
                strategy_entropy_sum += float(strategy_info.get("strategy_entropy", 0.0))
                strategy_k = max(strategy_k, int(strategy_info.get("strategy_k", 0)))
            env.step_async(act)
            obs, rew, done, infos = env.step_wait()
            steps += 1
            ep_return += float(rew[0])
            if done.any():
                for i in range(len(done)):
                    if done[i]:
                        info = infos[i] if i < len(infos) else {}
                        ep_res = info.get("episode_result", info)
                        bs = int(ep_res.get("blue_score", 0))
                        rs = int(ep_res.get("red_score", 0))
                        success = 1 if bs > rs else 0
                        decision_steps = int(ep_res.get("decision_steps", info.get("decision_steps", 0)))
                        zone_cov = float(ep_res.get("zone_coverage", 0.0))
                        collision_free = int(ep_res.get("collision_free_episode", 1))
                        ttfs = ep_res.get("time_to_first_score")
                        try:
                            ttfs_f = float(ttfs) if ttfs is not None and ttfs != "" else np.nan
                        except (TypeError, ValueError):
                            ttfs_f = np.nan
                        mean_dist = ep_res.get("mean_inter_robot_dist")
                        try:
                            mean_dist_f = float(mean_dist) if mean_dist is not None and mean_dist != "" else np.nan
                        except (TypeError, ValueError):
                            mean_dist_f = np.nan
                        row = {
                            "success": success,
                            "blue_score": bs,
                            "red_score": rs,
                            "steps": decision_steps,
                            "return": ep_return,
                            "zone_coverage": zone_cov,
                            "collision_free": collision_free,
                            "win_margin": bs - rs,
                            "time_to_first_score": ttfs_f,
                            "mean_inter_robot_dist": mean_dist_f,
                        }
                        if record_entropy:
                            row["policy_entropy"] = ep_entropy_first
                        if strategy_steps > 0:
                            denom = float(max(1, strategy_steps))
                            row["strategy_switches"] = strategy_switches
                            row["strategy_switch_rate"] = float(strategy_switches) / float(max(1, strategy_steps - 1))
                            row["strategy_resamples"] = strategy_resamples
                            row["strategy_resample_rate"] = float(strategy_resamples) / denom
                            row["strategy_unique_count"] = len(strategy_counts)
                            row["strategy_entropy_mean"] = strategy_entropy_sum / denom
                            dominant = max(strategy_counts.items(), key=lambda kv: kv[1])[0]
                            row["strategy_dominant"] = dominant
                            for z_idx in range(strategy_k):
                                row[f"strategy_occupancy_{z_idx}"] = float(strategy_counts.get(z_idx, 0)) / denom
                            for phase, counts in sorted(strategy_phase_counts.items()):
                                phase_denom = float(max(1, sum(counts.values())))
                                for z_idx in range(strategy_k):
                                    row[f"strategy_phase_{phase}_occupancy_{z_idx}"] = (
                                        float(counts.get(z_idx, 0)) / phase_denom
                                    )
                        episodes.append(row)
                        if progress_every > 0:
                            le = len(episodes)
                            if le == 1 or le % progress_every == 0 or le == n_episodes:
                                print(f"  episode {le}/{n_episodes}", flush=True)
                        ep_return = 0.0
                if hasattr(model, "reset_strategy"):
                    model.reset_strategy()
                break

    return episodes


def count_wld(episodes: list[dict]) -> tuple[int, int, int]:
    """Wins / losses / draws consistent with success_rate = wins / len(episodes)."""
    w = sum(1 for e in episodes if int(e.get("blue_score", 0)) > int(e.get("red_score", 0)))
    l = sum(1 for e in episodes if int(e.get("blue_score", 0)) < int(e.get("red_score", 0)))
    d = sum(1 for e in episodes if int(e.get("blue_score", 0)) == int(e.get("red_score", 0)))
    return w, l, d


def binomial_se(wins: int, total: int) -> float:
    """Binomial standard error of a win-rate percentage: SE = sqrt(p*(1-p)/N) * 100.

    This is the canonical "std dev" reported on win-rate bar charts (also equals
    ``success_rate_std / sqrt(N)`` from ``compute_aggregates``). Returns 0.0 at
    the p=0 / p=1 extremes and at total<=0.
    """
    import math

    if total is None or int(total) <= 0:
        return 0.0
    p = float(wins) / float(total)
    p = max(0.0, min(1.0, p))
    return 100.0 * math.sqrt(p * (1.0 - p) / float(total))


def compute_aggregates(episodes: list[dict]) -> dict:
    """Mean and std (over episodes) for paper-ready tables."""
    base = {
        "success_rate": 0.0,
        "success_rate_std": 0.0,
        "mean_steps": 0.0,
        "mean_steps_std": 0.0,
        "mean_return": 0.0,
        "return_std": 0.0,
        "return_var": 0.0,
        "return_var_std": 0.0,
        "mean_captures": 0.0,
        "mean_captures_std": 0.0,
        "defense_shutout_rate": 0.0,
        "defense_shutout_std": 0.0,
        "coverage_efficiency": 0.0,
        "coverage_efficiency_std": 0.0,
        "collision_free_rate": 0.0,
        "collision_free_rate_std": 0.0,
        "win_margin_mean": 0.0,
        "win_margin_std": 0.0,
        "time_to_first_score_mean": float("nan"),
        "time_to_first_score_std": 0.0,
        "mean_inter_robot_dist_mean": float("nan"),
        "mean_inter_robot_dist_std": 0.0,
        "policy_entropy_mean": float("nan"),
        "policy_entropy_std": 0.0,
        "strategy_switch_rate_mean": float("nan"),
        "strategy_switch_rate_std": 0.0,
        "strategy_resample_rate_mean": float("nan"),
        "strategy_resample_rate_std": 0.0,
        "strategy_unique_count_mean": float("nan"),
        "strategy_unique_count_std": 0.0,
        "strategy_entropy_step_mean": float("nan"),
        "strategy_entropy_step_std": 0.0,
    }
    if not episodes:
        return base
    arr = np.array(
        [
            [
                e["success"],
                e["steps"],
                e["return"],
                e["zone_coverage"],
                e["collision_free"],
                e["win_margin"],
                e.get("time_to_first_score", np.nan),
                e.get("mean_inter_robot_dist", np.nan),
            ]
            for e in episodes
        ]
    )
    n = arr.shape[0]
    ddof = 1 if n > 1 else 0
    success_rate = float(np.mean(arr[:, 0])) * 100.0
    success_rate_std = float(np.std(arr[:, 0], ddof=ddof)) * 100.0
    mean_steps = float(np.mean(arr[:, 1]))
    mean_steps_std = float(np.std(arr[:, 1], ddof=ddof))
    mean_return = float(np.mean(arr[:, 2]))
    return_std = float(np.std(arr[:, 2], ddof=ddof))
    return_var = float(np.var(arr[:, 2], ddof=ddof))
    return_var_std = 0.0
    arr_bs = np.array([int(e["blue_score"]) for e in episodes], dtype=float)
    arr_rs = np.array([int(e["red_score"]) for e in episodes], dtype=float)
    mean_captures = float(np.mean(arr_bs))
    mean_captures_std = float(np.std(arr_bs, ddof=ddof))
    shutout = (arr_rs == 0).astype(float)
    defense_shutout_rate = float(np.mean(shutout)) * 100.0
    defense_shutout_std = float(np.std(shutout, ddof=ddof)) * 100.0
    ent_list = [
        float(e["policy_entropy"])
        for e in episodes
        if "policy_entropy" in e and np.isfinite(float(e["policy_entropy"]))
    ]
    if ent_list:
        ea = np.array(ent_list, dtype=float)
        policy_entropy_mean = float(np.mean(ea))
        policy_entropy_std = float(np.std(ea, ddof=1)) if len(ea) > 1 else 0.0
    else:
        policy_entropy_mean = float("nan")
        policy_entropy_std = 0.0

    def _optional_mean_std(key: str) -> tuple[float, float]:
        vals = [
            float(e[key])
            for e in episodes
            if key in e and np.isfinite(float(e[key]))
        ]
        if not vals:
            return float("nan"), 0.0
        arr_v = np.array(vals, dtype=float)
        return float(np.mean(arr_v)), float(np.std(arr_v, ddof=1)) if len(arr_v) > 1 else 0.0

    strategy_switch_rate_mean, strategy_switch_rate_std = _optional_mean_std("strategy_switch_rate")
    strategy_resample_rate_mean, strategy_resample_rate_std = _optional_mean_std("strategy_resample_rate")
    strategy_unique_count_mean, strategy_unique_count_std = _optional_mean_std("strategy_unique_count")
    strategy_entropy_step_mean, strategy_entropy_step_std = _optional_mean_std("strategy_entropy_mean")
    coverage_efficiency = float(np.mean(arr[:, 3])) * 100.0
    coverage_efficiency_std = float(np.std(arr[:, 3], ddof=ddof))
    collision_free_rate = float(np.mean(arr[:, 4])) * 100.0
    collision_free_rate_std = float(np.std(arr[:, 4], ddof=ddof))
    win_margin_mean = float(np.mean(arr[:, 5]))
    win_margin_std = float(np.std(arr[:, 5], ddof=ddof))
    ttfs = arr[:, 6]
    ttfs_valid = ttfs[np.isfinite(ttfs)]
    time_to_first_score_mean = float(np.mean(ttfs_valid)) if len(ttfs_valid) > 0 else np.nan
    time_to_first_score_std = float(np.std(ttfs_valid, ddof=1)) if len(ttfs_valid) > 1 else 0.0
    midist = arr[:, 7]
    midist_valid = midist[np.isfinite(midist)]
    mean_inter_robot_dist_mean = float(np.mean(midist_valid)) if len(midist_valid) > 0 else np.nan
    mean_inter_robot_dist_std = float(np.std(midist_valid, ddof=1)) if len(midist_valid) > 1 else 0.0
    result = {
        "success_rate": success_rate,
        "success_rate_std": success_rate_std,
        "mean_steps": mean_steps,
        "mean_steps_std": mean_steps_std,
        "mean_return": mean_return,
        "return_std": return_std,
        "return_var": return_var,
        "return_var_std": return_var_std,
        "mean_captures": mean_captures,
        "mean_captures_std": mean_captures_std,
        "defense_shutout_rate": defense_shutout_rate,
        "defense_shutout_std": defense_shutout_std,
        "coverage_efficiency": coverage_efficiency,
        "coverage_efficiency_std": coverage_efficiency_std,
        "collision_free_rate": collision_free_rate,
        "collision_free_rate_std": collision_free_rate_std,
        "win_margin_mean": win_margin_mean,
        "win_margin_std": win_margin_std,
        "time_to_first_score_mean": time_to_first_score_mean,
        "time_to_first_score_std": time_to_first_score_std,
        "mean_inter_robot_dist_mean": mean_inter_robot_dist_mean,
        "mean_inter_robot_dist_std": mean_inter_robot_dist_std,
        "policy_entropy_mean": policy_entropy_mean,
        "policy_entropy_std": policy_entropy_std,
        "strategy_switch_rate_mean": strategy_switch_rate_mean,
        "strategy_switch_rate_std": strategy_switch_rate_std,
        "strategy_resample_rate_mean": strategy_resample_rate_mean,
        "strategy_resample_rate_std": strategy_resample_rate_std,
        "strategy_unique_count_mean": strategy_unique_count_mean,
        "strategy_unique_count_std": strategy_unique_count_std,
        "strategy_entropy_step_mean": strategy_entropy_step_mean,
        "strategy_entropy_step_std": strategy_entropy_step_std,
    }
    occupancy_keys = sorted(
        {
            key
            for episode in episodes
            for key in episode.keys()
            if key.startswith("strategy_occupancy_") or key.startswith("strategy_phase_")
        }
    )
    for key in occupancy_keys:
        mean_v, std_v = _optional_mean_std(key)
        result[f"{key}_mean"] = mean_v
        result[f"{key}_std"] = std_v
    return result
