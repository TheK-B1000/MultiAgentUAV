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


def ppo_load_custom_objects(env: Any) -> dict[str, Any]:
    """``custom_objects`` for ``PPO.load`` when unpickling checkpoints saved with another Python.

    On Python 3.12, lambdas/schedules in older pickles can raise
    ``code() argument 13 must be str, not int``. Inference only needs weights; replacing
    ``clip_range``, ``lr_schedule``, and ``cfg`` avoids that without affecting ``predict``.
    """
    from stable_baselines3.common.utils import FloatSchedule

    from rl.train_ppo import MaskedMultiInputPolicy, PPOConfig

    _cfg = PPOConfig()
    return {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "policy_class": MaskedMultiInputPolicy,
        "clip_range": float(_cfg.clip_range),
        "lr_schedule": FloatSchedule(_cfg.learning_rate),
        "cfg": _cfg,
    }


def _policy_entropy_first_step(model: Any, single_obs: dict) -> float:
    """Mean policy entropy at one observation (stochastic policy, eval mode)."""
    import torch

    with torch.no_grad():
        packed = model.policy.obs_to_tensor(single_obs)
        obs_tensor = packed[0] if isinstance(packed, tuple) else packed
        dist = model.policy.get_distribution(obs_tensor)
        ent = dist.entropy()
        return float(torch.mean(ent).item())


def _reseed_env(env: Any, seed: int) -> None:
    """Force identical episode starts across models by reseeding the GPU core RNG."""
    core = getattr(env, "core", None)
    if core is None:
        return
    rng = getattr(core, "_rng", None)
    if rng is not None and hasattr(rng, "manual_seed"):
        rng.manual_seed(int(seed))


def shared_episode_seeds(n_episodes: int, seed_base: int, opponent: str) -> list[int]:
    """Deterministic per-episode seeds shared by every compared checkpoint.

    OP4 uses a disjoint block so its scenarios never collide with OP3's list.
    """
    opp = str(opponent).strip().upper()
    block = 1_000_000 if opp == "OP4" else 0
    base = int(seed_base) + block
    return [base + i for i in range(int(n_episodes))]


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
    episode_seeds: list[int] | None = None,
    opponent_kind: str = "SCRIPTED",
    stress_schedule: dict | None = None,
) -> list[dict]:
    """Run n_episodes; each dict has success, steps, return, scores, etc. (same as plot_eval_metrics).

    If deterministic is False, uses stochastic policy actions (sampled); default True matches greedy argmax.
    If record_entropy is True, each episode dict includes policy_entropy (first-step mean entropy).
    If progress_every > 0, prints after episode 1, then every progress_every episodes, and on the last
    (flush=True) so long 8v8 runs do not look hung.
    If episode_seeds is provided (len == n_episodes), each episode is reseeded before reset so
    every model faces the same initial conditions for episode i.

    ``opponent_kind`` selects between SCRIPTED tags (OP1..OP4) and SPECIES tags
    (RUSHER/CAMPER/BALANCED). ``stress_schedule`` overrides the default
    per-phase current/drift schedule; pass one from
    ``rl.configuration_space.Configuration.stress_schedule`` to make the current
    profile an explicit evaluated factor instead of an inherited default.
    """
    from stable_baselines3 import PPO

    _numpy_compat_shim()
    model = PPO.load(model_path, device=device, custom_objects=ppo_load_custom_objects(env))
    model.policy.set_training_mode(False)
    if progress_every > 0:
        print(
            f"  checkpoint loaded; {n_episodes} episodes (first result prints after ep 1)",
            flush=True,
        )

    kind = str(opponent_kind).strip().upper()
    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", kind, opponent)
        # Match training (train_ppo): phase-indexed current/drift via stress schedule.
        try:
            if stress_schedule is not None:
                env.env_method("set_stress_schedule", stress_schedule)
            else:
                from rl.curriculum import STRESS_BY_PHASE

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
            f"Failed to set opponent to {kind}:{opponent!r}: {e}. "
            "Red team may still be using the previous opponent — OP3 vs OP4 results can look identical."
        )

    if episode_seeds is not None and len(episode_seeds) != int(n_episodes):
        raise ValueError(
            f"episode_seeds length {len(episode_seeds)} != n_episodes {n_episodes}"
        )

    episodes: list[dict] = []
    if episode_seeds is not None:
        _reseed_env(env, int(episode_seeds[0]))
    obs = env.reset()

    for ep_i in range(n_episodes):
        if episode_seeds is not None:
            if ep_i > 0:
                _reseed_env(env, int(episode_seeds[ep_i]))
                obs = env.reset()
        ep_return = 0.0
        steps = 0
        ep_entropy_first = float("nan")
        while True:
            single = {
                k: v[0] if hasattr(v, "shape") and len(v.shape) > 1 and v.shape[0] == 1 else v
                for k, v in obs.items()
            }
            if record_entropy and steps == 0:
                try:
                    ep_entropy_first = _policy_entropy_first_step(model, single)
                except Exception:
                    ep_entropy_first = float("nan")
            act, _ = model.predict(single, deterministic=deterministic)
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
                        episodes.append(row)
                        if progress_every > 0:
                            le = len(episodes)
                            if le == 1 or le % progress_every == 0 or le == n_episodes:
                                print(f"  episode {le}/{n_episodes}", flush=True)
                        ep_return = 0.0
                break

    return episodes


def count_wld(episodes: list[dict]) -> tuple[int, int, int]:
    """Wins / losses / draws consistent with success_rate = wins / len(episodes)."""
    w = sum(1 for e in episodes if int(e.get("blue_score", 0)) > int(e.get("red_score", 0)))
    l = sum(1 for e in episodes if int(e.get("blue_score", 0)) < int(e.get("red_score", 0)))
    d = sum(1 for e in episodes if int(e.get("blue_score", 0)) == int(e.get("red_score", 0)))
    return w, l, d


def match_score_from_wld(wins: int, losses: int, draws: int) -> float:
    """Match Score = (W + 0.5D) / (W+L+D), as a percentage in [0, 100]."""
    total = int(wins) + int(losses) + int(draws)
    if total <= 0:
        return float("nan")
    return 100.0 * (float(wins) + 0.5 * float(draws)) / float(total)


def episode_match_points(episodes: list[dict]) -> np.ndarray:
    """Per-episode match points: 1.0 win, 0.5 draw, 0.0 loss."""
    pts = np.empty(len(episodes), dtype=float)
    for i, e in enumerate(episodes):
        bs = int(e.get("blue_score", 0))
        rs = int(e.get("red_score", 0))
        if bs > rs:
            pts[i] = 1.0
        elif bs < rs:
            pts[i] = 0.0
        else:
            pts[i] = 0.5
    return pts


def bootstrap_ci_mean(
    values: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) for the mean of ``values`` via percentile bootstrap."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size == 1 or n_boot <= 0:
        return mean, mean, mean
    rng = rng or np.random.default_rng(0)
    n = arr.size
    boots = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        sample = arr[rng.integers(0, n, size=n)]
        boots[b] = float(np.mean(sample))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return mean, lo, hi


def paired_bootstrap_seed_mean(
    per_seed_points: list[np.ndarray],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Paired bootstrap over shared episode indices, then mean across seeds.

    Each seed must have the same episode count (identical eval seed list).
    Returns (mean_match_score_pct, ci_lo_pct, ci_hi_pct).
    """
    if not per_seed_points:
        return float("nan"), float("nan"), float("nan")
    mats = [np.asarray(p, dtype=float) for p in per_seed_points]
    n = int(mats[0].size)
    if n <= 0 or any(m.size != n for m in mats):
        raise ValueError("paired bootstrap requires equal-length per-seed episode vectors")
    # Point estimate: mean of seed-level means.
    seed_means = np.array([float(np.mean(m)) for m in mats], dtype=float)
    point = float(np.mean(seed_means)) * 100.0
    if n == 1 or n_boot <= 0:
        return point, point, point
    rng = rng or np.random.default_rng(0)
    boots = np.empty(int(n_boot), dtype=float)
    stacked = np.stack(mats, axis=0)  # (n_seeds, n_eps)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        resampled = stacked[:, idx]
        boots[b] = float(np.mean(np.mean(resampled, axis=1)))
    lo = float(np.quantile(boots, alpha / 2.0)) * 100.0
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0)) * 100.0
    return point, lo, hi


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
        "n_episodes": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "win_rate": 0.0,
        "loss_rate": 0.0,
        "draw_rate": 0.0,
        "match_score": 0.0,
        "match_score_ci_lo": 0.0,
        "match_score_ci_hi": 0.0,
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
    }
    if not episodes:
        return base
    w, l, d = count_wld(episodes)
    total = max(1, w + l + d)
    win_rate = 100.0 * float(w) / float(total)
    loss_rate = 100.0 * float(l) / float(total)
    draw_rate = 100.0 * float(d) / float(total)
    match_score = match_score_from_wld(w, l, d)
    pts = episode_match_points(episodes)
    _, ms_lo, ms_hi = bootstrap_ci_mean(pts * 100.0, n_boot=2000, alpha=0.05, rng=np.random.default_rng(0))

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
        "n_episodes": int(n),
        "wins": int(w),
        "losses": int(l),
        "draws": int(d),
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": draw_rate,
        "match_score": match_score,
        "match_score_ci_lo": ms_lo,
        "match_score_ci_hi": ms_hi,
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
    }


def run_two_policy_episodes(
    blue_model_path: str,
    red_model_path: str,
    env: Any,
    n_episodes: int,
    device: str,
    *,
    deterministic: bool = True,
    episode_seeds: list[int] | None = None,
    progress_every: int = 0,
) -> list[dict]:
    """Run blue_ckpt vs red_ckpt via BatchedCTFCore two-policy stepping.

    Mirrors LeagueCallback._run_side_swapped_mirror_eval's core.step(..., red_action_flat=...)
    path, but without side-swap: blue always uses ``blue_model_path``, red always uses
    ``red_model_path``. Each returned dict has blue_score / red_score from blue's perspective
    (compatible with count_wld / match_score_from_wld).
    """
    import torch
    from stable_baselines3 import PPO

    from rl.episode_result import parse_episode_result, scores_from_info

    _numpy_compat_shim()
    core = getattr(env, "core", None)
    if core is None:
        raise ValueError("run_two_policy_episodes requires an env with a .core (GPUCTFVecEnv)")

    custom = ppo_load_custom_objects(env)
    blue_model = PPO.load(blue_model_path, device=device, custom_objects=custom)
    red_model = PPO.load(red_model_path, device=device, custom_objects=custom)
    blue_model.policy.set_training_mode(False)
    red_model.policy.set_training_mode(False)

    try:
        env.env_method("set_league_mode", True)
        env.env_method("set_phase", "OP3")
    except Exception:
        pass

    if episode_seeds is not None and len(episode_seeds) != int(n_episodes):
        raise ValueError(
            f"episode_seeds length {len(episode_seeds)} != n_episodes {n_episodes}"
        )

    def _obs_numpy(side: str) -> dict:
        return {
            k: v.detach().cpu().numpy().astype(np.float32)
            for k, v in core.get_obs_tensors(side=side).items()
        }

    episodes: list[dict] = []
    for ep_i in range(int(n_episodes)):
        if episode_seeds is not None:
            _reseed_env(env, int(episode_seeds[ep_i]))
        core.reset_all()
        steps = 0
        while True:
            blue_obs = _obs_numpy("blue")
            red_obs = _obs_numpy("red")
            blue_np, _ = blue_model.predict(blue_obs, deterministic=deterministic)
            red_np, _ = red_model.predict(red_obs, deterministic=deterministic)
            blue_actions = torch.as_tensor(blue_np, dtype=torch.int64, device=core.device)
            red_actions = torch.as_tensor(red_np, dtype=torch.int64, device=core.device)
            _, _, terminated, truncated, infos = core.step(
                blue_actions,
                tensor_obs=True,
                red_action_flat=red_actions,
            )
            steps += 1
            done = torch.logical_or(terminated, truncated)
            if not bool(done[0].item()):
                continue
            # core.step() infos carry blue_score/red_score at the top level; the
            # nested "episode_result" dict only exists on GPUCTFVecEnv-wrapped
            # steps. Reading it via parse_episode_result alone yielded None for
            # every episode here, so this loop returned no episodes at all.
            scores = scores_from_info(infos[0])
            if scores is None:
                raise ValueError(
                    "two-policy stepping produced an info dict with no blue_score/red_score; "
                    f"keys={sorted(infos[0])}"
                )
            bs, rs = scores
            summary = parse_episode_result(infos[0])
            episodes.append(
                {
                    "success": 1 if bs > rs else 0,
                    "blue_score": bs,
                    "red_score": rs,
                    "steps": steps,
                    "return": 0.0,
                    "zone_coverage": float(getattr(summary, "zone_coverage", 0.0) or 0.0)
                    if summary is not None
                    else 0.0,
                    "collision_free": int(getattr(summary, "collision_free_episode", 1) or 1)
                    if summary is not None
                    else 1,
                    "win_margin": bs - rs,
                    "time_to_first_score": float("nan"),
                    "mean_inter_robot_dist": float("nan"),
                }
            )
            if progress_every > 0:
                le = len(episodes)
                if le == 1 or le % progress_every == 0 or le == n_episodes:
                    print(f"  episode {le}/{n_episodes}", flush=True)
            break

    return episodes
