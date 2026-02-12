from __future__ import annotations

"""
Analyze per-episode eval CSVs (from ctfviewer.evaluate_model) and print
research-style metrics:

- Performance: success rate, time to completion
- Coordination: role timing, coverage efficiency
- Robustness: (optional) OP3 vs OP4 comparison
- Stability: reward variance
- Specialization: flag_capture_variance
- Robotics metrics: safety (collisions), path quality (coverage)

Usage examples:

  # Single CSV (e.g. League vs OP3, 100 episodes)
  python analyze_eval_metrics.py --csv metrics/eval_ppo_OP3_100ep.csv --label League_OP3

  # Compare train-like vs held-out (e.g. OP3 vs OP4)
  python analyze_eval_metrics.py \
      --csv metrics/eval_ppo_OP3_100ep.csv --label League_OP3 \
      --compare-csv metrics/eval_ppo_OP4_100ep.csv --compare-label League_OP4
"""

import argparse
import csv
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeRow:
    """Typed view over a single CSV row (with safe defaults)."""

    success: int = 0
    time_to_first_score: float = math.nan
    time_to_game_over: float = math.nan
    collisions_per_episode: float = math.nan
    collision_events_per_episode: float = math.nan
    near_misses_per_episode: float = math.nan
    collision_free_episode: int = 0
    mean_inter_robot_dist: float = math.nan
    std_inter_robot_dist: float = math.nan
    zone_coverage: float = math.nan
    mean_pct_attacking: float = math.nan
    mean_pct_defending: float = math.nan
    flag_capture_variance: float = math.nan
    mean_flag_captures: float = math.nan
    blue_episode_reward: float = math.nan
    reward_per_timestep: float = math.nan
    collisions_per_100_steps: float = math.nan
    near_misses_per_100_steps: float = math.nan
    coverage_efficiency: float = math.nan
    phase_name: str = ""
    scripted_tag: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "EpisodeRow":
        def _f(key: str, default: float = math.nan) -> float:
            v = d.get(key, "")
            if v == "":
                return default
            try:
                return float(v)
            except Exception:
                return default

        def _i(key: str, default: int = 0) -> int:
            v = d.get(key, "")
            if v == "":
                return default
            try:
                return int(float(v))
            except Exception:
                return default

        return cls(
            success=_i("success", 0),
            time_to_first_score=_f("time_to_first_score"),
            time_to_game_over=_f("time_to_game_over"),
            collisions_per_episode=_f("collisions_per_episode"),
            collision_events_per_episode=_f("collision_events_per_episode"),
            near_misses_per_episode=_f("near_misses_per_episode"),
            collision_free_episode=_i("collision_free_episode", 0),
            mean_inter_robot_dist=_f("mean_inter_robot_dist"),
            std_inter_robot_dist=_f("std_inter_robot_dist"),
            zone_coverage=_f("zone_coverage"),
            mean_pct_attacking=_f("mean_pct_attacking"),
            mean_pct_defending=_f("mean_pct_defending"),
            flag_capture_variance=_f("flag_capture_variance"),
            mean_flag_captures=_f("mean_flag_captures"),
            blue_episode_reward=_f("blue_episode_reward"),
            reward_per_timestep=_f("reward_per_timestep"),
            collisions_per_100_steps=_f("collisions_per_100_steps"),
            near_misses_per_100_steps=_f("near_misses_per_100_steps"),
            coverage_efficiency=_f("coverage_efficiency"),
            phase_name=d.get("phase_name", ""),
            scripted_tag=d.get("scripted_tag", ""),
        )


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else math.nan


def _std(xs: List[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    if len(xs) <= 1:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def load_episodes(csv_path: str) -> List[EpisodeRow]:
    out: List[EpisodeRow] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(EpisodeRow.from_dict(row))
    return out


def summarize_episodes(episodes: List[EpisodeRow]) -> Dict[str, Any]:
    n = len(episodes)
    if n == 0:
        return {"num_episodes": 0}

    succ = [e.success for e in episodes]
    t_first = [e.time_to_first_score for e in episodes]
    t_end = [e.time_to_game_over for e in episodes]

    collisions = [e.collisions_per_episode for e in episodes]
    coll_events = [e.collision_events_per_episode for e in episodes]
    near_misses = [e.near_misses_per_episode for e in episodes]
    coll_100 = [e.collisions_per_100_steps for e in episodes]
    near_100 = [e.near_misses_per_100_steps for e in episodes]
    coll_free = [e.collision_free_episode for e in episodes]

    cover = [e.zone_coverage for e in episodes]
    cover_eff = [e.coverage_efficiency for e in episodes]
    att = [e.mean_pct_attacking for e in episodes]
    dfn = [e.mean_pct_defending for e in episodes]

    cap_var = [e.flag_capture_variance for e in episodes]
    cap_mean = [e.mean_flag_captures for e in episodes]

    rew = [e.blue_episode_reward for e in episodes]
    rew_step = [e.reward_per_timestep for e in episodes]

    return {
        "num_episodes": n,
        # Performance
        "success_rate": _mean(succ),
        "mean_time_to_first_score": _mean(t_first),
        "mean_time_to_game_over": _mean(t_end),
        # Stability
        "reward_mean": _mean(rew),
        "reward_std": _std(rew),
        "reward_per_timestep_mean": _mean(rew_step),
        "reward_per_timestep_std": _std(rew_step),
        # Coordination / coverage
        "mean_pct_attacking": _mean(att),
        "mean_pct_defending": _mean(dfn),
        "zone_coverage_mean": _mean(cover),
        "coverage_efficiency_mean": _mean(cover_eff),
        # Specialization
        "flag_capture_variance_mean": _mean(cap_var),
        "mean_flag_captures": _mean(cap_mean),
        # Safety / robotics
        "collisions_per_episode_mean": _mean(collisions),
        "collision_events_per_episode_mean": _mean(coll_events),
        "near_misses_per_episode_mean": _mean(near_misses),
        "collisions_per_100_steps_mean": _mean(coll_100),
        "near_misses_per_100_steps_mean": _mean(near_100),
        "collision_free_rate": _mean(coll_free),
    }


def print_summary(label: str, summary: Dict[str, Any]) -> None:
    n = summary.get("num_episodes", 0)
    if n == 0:
        print(f"{label}: no episodes")
        return

    print(f"\n=== {label} (N={n}) ===")

    # Performance
    print("Performance:")
    print(f"  Success rate:           {summary['success_rate']:.1%}")
    print(f"  Mean T_first_score:     {summary['mean_time_to_first_score']:.2f} s")
    print(f"  Mean T_game_over:       {summary['mean_time_to_game_over']:.2f} s")

    # Coordination
    print("Coordination:")
    print(f"  Mean % attacking:       {summary['mean_pct_attacking']:.1f}%")
    print(f"  Mean % defending:       {summary['mean_pct_defending']:.1f}%")
    print(f"  Zone coverage (mean):   {summary['zone_coverage_mean']:.3f}")
    print(f"  Coverage efficiency:    {summary['coverage_efficiency_mean']:.6f}")

    # Stability
    print("Stability:")
    print(f"  Episode reward mean:    {summary['reward_mean']:.3f}")
    print(f"  Episode reward std:     {summary['reward_std']:.3f}")
    print(f"  Reward/timestep mean:   {summary['reward_per_timestep_mean']:.6f}")
    print(f"  Reward/timestep std:    {summary['reward_per_timestep_std']:.6f}")

    # Specialization
    print("Specialization:")
    print(f"  Flag capture variance:  {summary['flag_capture_variance_mean']:.3f}")
    print(f"  Mean flag captures:     {summary['mean_flag_captures']:.3f}")

    # Robotics metrics
    print("Robotics (safety / path quality):")
    print(f"  Collisions/episode:     {summary['collisions_per_episode_mean']:.2f}")
    print(f"  Collision events/ep:    {summary['collision_events_per_episode_mean']:.2f}")
    print(f"  Near-misses/episode:    {summary['near_misses_per_episode_mean']:.2f}")
    print(f"  Collisions/100 steps:   {summary['collisions_per_100_steps_mean']:.2f}")
    print(f"  Near-misses/100 steps:  {summary['near_misses_per_100_steps_mean']:.2f}")
    print(f"  Collision-free rate:    {summary['collision_free_rate']:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze eval CSV from ctfviewer.evaluate_model")
    parser.add_argument("--csv", type=str, required=True, help="Path to eval CSV (per-episode metrics)")
    parser.add_argument("--label", type=str, default=None, help="Label for this run (e.g. League_OP3)")
    parser.add_argument(
        "--compare-csv",
        type=str,
        default=None,
        help="Optional second CSV for robustness comparison (e.g. OP4)",
    )
    parser.add_argument("--compare-label", type=str, default=None, help="Label for compare CSV (e.g. League_OP4)")
    args = parser.parse_args()

    episodes_a = load_episodes(args.csv)
    label_a = args.label or "Run A"
    summary_a = summarize_episodes(episodes_a)
    print_summary(label_a, summary_a)

    if args.compare_csv:
        episodes_b = load_episodes(args.compare_csv)
        label_b = args.compare_label or "Run B"
        summary_b = summarize_episodes(episodes_b)
        print_summary(label_b, summary_b)

        # Robustness: success drop and safety differences
        if summary_a.get("num_episodes", 0) > 0 and summary_b.get("num_episodes", 0) > 0:
            sa = summary_a["success_rate"]
            sb = summary_b["success_rate"]
            drop = sa - sb
            print("\nRobustness comparison:")
            print(f"  Success A={sa:.1%}, B={sb:.1%}, drop (A->B)={drop:.1%}")
            ca = summary_a["collisions_per_100_steps_mean"]
            cb = summary_b["collisions_per_100_steps_mean"]
            print(f"  Collisions/100 steps A={ca:.2f}, B={cb:.2f}")


if __name__ == "__main__":
    main()

