"""
Shared PPO rollout for win-rate scripts and plot_eval_metrics.py.

Keeps episode stepping, opponent setup, and aggregates identical so
success_rate (eval table) matches W/L/D win rate (wins / n_episodes).
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _numpy_compat_shim() -> None:
    if "numpy._core.numeric" not in sys.modules:
        try:
            import numpy.core as _core
            import numpy.core.numeric
            import numpy.core.multiarray
            import numpy.core.umath
            sys.modules["numpy._core"] = _core
            sys.modules["numpy._core.numeric"] = _core.numeric
            sys.modules["numpy._core.multiarray"] = _core.multiarray
            sys.modules["numpy._core.umath"] = _core.umath
        except Exception:
            pass
    try:
        import numpy.random._pickle as _np_pickle
        _orig_bg_ctor = _np_pickle.__bit_generator_ctor

        def _patched_bg_ctor(bit_generator_name: Any = "MT19937") -> Any:
            if isinstance(bit_generator_name, type):
                bit_generator_name = bit_generator_name.__name__
            return _orig_bg_ctor(bit_generator_name)

        _np_pickle.__bit_generator_ctor = _patched_bg_ctor
    except Exception:
        pass


def run_eval_episodes(
    model_path: str,
    env: Any,
    n_episodes: int,
    device: str,
    opponent: str,
) -> list[dict]:
    """Run n_episodes; each dict has success, steps, return, scores, etc. (same as plot_eval_metrics)."""
    from stable_baselines3 import PPO
    from rl.train_ppo import MaskedMultiInputPolicy

    _numpy_compat_shim()
    custom = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "policy_class": MaskedMultiInputPolicy,
    }
    model = PPO.load(model_path, device=device, custom_objects=custom)
    model.policy.set_training_mode(False)

    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", "SCRIPTED", opponent)
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
            "Red team may still be using the previous opponent — OP3 vs OP4 results can look identical."
        )

    episodes: list[dict] = []
    obs = env.reset()

    for _ in range(n_episodes):
        ep_return = 0.0
        steps = 0
        while True:
            single = {
                k: v[0] if hasattr(v, "shape") and len(v.shape) > 1 and v.shape[0] == 1 else v
                for k, v in obs.items()
            }
            act, _ = model.predict(single, deterministic=True)
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
                        episodes.append(
                            {
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
                        )
                        ep_return = 0.0
                break

    return episodes


def count_wld(episodes: list[dict]) -> tuple[int, int, int]:
    """Wins / losses / draws consistent with success_rate = wins / len(episodes)."""
    w = sum(1 for e in episodes if int(e.get("blue_score", 0)) > int(e.get("red_score", 0)))
    l = sum(1 for e in episodes if int(e.get("blue_score", 0)) < int(e.get("red_score", 0)))
    d = sum(1 for e in episodes if int(e.get("blue_score", 0)) == int(e.get("red_score", 0)))
    return w, l, d


def compute_aggregates(episodes: list[dict]) -> dict:
    """Mean and std (over episodes) for paper-ready tables."""
    base = {
        "success_rate": 0.0,
        "success_rate_std": 0.0,
        "mean_steps": 0.0,
        "mean_steps_std": 0.0,
        "mean_return": 0.0,
        "return_var": 0.0,
        "return_var_std": 0.0,
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
    return_var = float(np.var(arr[:, 2], ddof=ddof))
    return_var_std = 0.0
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
    return {
        "success_rate": success_rate,
        "success_rate_std": success_rate_std,
        "mean_steps": mean_steps,
        "mean_steps_std": mean_steps_std,
        "mean_return": mean_return,
        "return_var": return_var,
        "return_var_std": return_var_std,
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
    }
