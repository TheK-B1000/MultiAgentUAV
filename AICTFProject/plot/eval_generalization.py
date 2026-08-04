#!/usr/bin/env python3
"""Seen-configuration performance vs held-out-configuration generalization.

Legs one and two of the VGC-Bench evaluation triad (the third,
approximate best-response exploitability: protocol in ``rl/eval_exploitability.py``,
CLI in ``plot/eval_exploitability.py``).

Two distinct claims are measured here, and the script keeps them separate on
purpose because conflating them is exactly what a reviewer will catch:

  PERFORMANCE      match score on C_seen -- configurations every compared method
                   encountered during training.
  GENERALIZATION   match score on C_heldout -- configurations no method
                   encountered. C_seen and C_heldout are disjoint by
                   construction (see rl/configuration_space.py).
  GENERALIZATION GAP  seen minus held-out, per method.

What this script does NOT measure is zero-shot team-size generalization.
Observation and action spaces are team-size dependent
(game_field_gpu._make_obs_action_spaces), so a 2v2 checkpoint cannot be loaded at
3v3 at all; the guard in configuration_space.assert_team_size_compatible refuses
to try. Independently trained 2v2/3v3/4v4 policies are a SCALABILITY result and
are reported under that name.

Statistics follow the same protocol as plot/eval_roastar_matrix.py: common random
numbers (every method sees the identical episode seed list for a configuration),
a default of 200 episodes per training seed, and uncertainty taken across
training seeds via paired bootstrap -- not a binomial interval that would treat
every episode as independent.

Usage (from AICTFProject):

  # List the protocol and the checkpoints that would be evaluated
  python plot/eval_generalization.py --settings 2v2,3v3,4v4 --list

  # Full sweep, paper protocol (200 episodes x 3 seeds = 600 games per config)
  python plot/eval_generalization.py --settings 2v2,3v3,4v4 --seeds 42,43,44 \\
      --episodes 200 --per-seed-out csv/generalization_per_seed.csv \\
      --out csv/generalization.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for _p in (PROJECT_ROOT, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_rollout import (  # noqa: E402
    compute_aggregates,
    episode_match_points,
    paired_bootstrap_seed_mean,
    run_eval_episodes,
)

from rl.configuration_space import (  # noqa: E402
    FIXED_MAP,
    FIXED_RULES_PROFILE,
    Configuration,
    assert_team_size_compatible,
    describe_split,
    episode_seeds,
    split,
)

# method label -> checkpoint basename templates, tried in order ({setting} and
# {seed} substituted). The unseeded variants are the legacy seed-42 filenames the
# ablation matrix wrote before the naming convention gained a seed suffix.
METHOD_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "selfplay": (
        "final_ppo_ablate_no_league_seed{seed}_{setting}.zip",
        "final_ppo_ablate_no_league_{setting}.zip",
    ),
    "sea-guard": (
        "final_ppo_ablate_ours_seed{seed}_{setting}.zip",
        "final_ppo_ablate_ours_{setting}.zip",
    ),
    "fp": ("final_ppo_roastar_fp_{setting}_seed{seed}.zip",),
    "do": ("final_ppo_roastar_do_{setting}_seed{seed}.zip",),
    "roastar-pfsp": ("final_ppo_roastar_pfsp_{setting}_seed{seed}.zip",),
    "roastar-exploiter": ("final_ppo_roastar_pfsp_exploiter_{setting}_seed{seed}.zip",),
}

# Legacy unseeded filenames belong to this training seed.
LEGACY_UNSEEDED_SEED = 42

PER_SEED_FIELDS = [
    "setting",
    "method",
    "seed",
    "split",
    "config",
    "opponent_kind",
    "opponent_key",
    "current_profile",
    "checkpoint",
    "n_episodes",
    "wins",
    "losses",
    "draws",
    "win_rate",
    "match_score",
    "mean_captures",
    "defense_shutout_rate",
    "mean_steps",
]

SUMMARY_FIELDS = [
    "setting",
    "method",
    "split",
    "n_seeds",
    "n_configs",
    "n_episodes_per_seed_per_config",
    "n_games_total",
    "match_score_mean",
    "match_score_seed_std",
    "match_score_ci_lo",
    "match_score_ci_hi",
    "win_rate_mean",
]

GAP_FIELDS = [
    "setting",
    "method",
    "seen_match_score",
    "heldout_match_score",
    "generalization_gap",
    "gap_ci_lo",
    "gap_ci_hi",
]


def agents_from_setting(setting: str) -> int:
    return int(str(setting).strip().lower().split("v")[0])


def team_size_from_checkpoint(path: str) -> Optional[int]:
    """Recover the team size a checkpoint was trained at from its filename."""
    import re

    m = re.search(r"_(\d+)v(\d+)", os.path.basename(path))
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return a if a == b else None


def discover_checkpoints(
    checkpoint_root: str,
    *,
    settings: Sequence[str],
    seeds: Sequence[int],
    methods: Sequence[str],
) -> Dict[Tuple[str, str, int], str]:
    """Return ``{(setting, method, seed): path}`` for checkpoints present on disk."""
    found: Dict[Tuple[str, str, int], str] = {}
    for setting in settings:
        for method in methods:
            templates = METHOD_PATTERNS[method]
            for seed in seeds:
                for i, template in enumerate(templates):
                    # Unseeded fallbacks only stand in for the legacy seed.
                    if "{seed}" not in template and int(seed) != LEGACY_UNSEEDED_SEED:
                        continue
                    filename = template.format(setting=setting, seed=int(seed))
                    path = os.path.join(checkpoint_root, setting, filename)
                    if os.path.isfile(path):
                        found[(setting, method, int(seed))] = os.path.abspath(path)
                        break
    return found


def _write_csv(rows: Sequence[Dict[str, Any]], path: str, fieldnames: Sequence[str]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _load_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate_configuration(
    checkpoint: str,
    config: Configuration,
    *,
    n_episodes: int,
    device: str,
    progress_every: int = 0,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Evaluate one checkpoint on one configuration with shared episode seeds."""
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    ckpt_size = team_size_from_checkpoint(checkpoint)
    if ckpt_size is not None:
        assert_team_size_compatible(ckpt_size, config.team_size)

    gpu_cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=int(config.team_size),
        max_red_agents=int(config.team_size),
        max_decision_steps=400,
        aquaticus_profile=True,
        rules_profile=FIXED_RULES_PROFILE,
        device=device,
        seed=0,
    )
    env = GPUCTFVecEnv(gpu_cfg)
    try:
        seeds = episode_seeds(config, int(n_episodes))
        episodes = run_eval_episodes(
            checkpoint,
            env,
            int(n_episodes),
            device,
            config.opponent_key,
            opponent_kind=config.opponent_kind,
            stress_schedule=config.stress_schedule(config.opponent_key),
            episode_seeds=seeds,
            progress_every=progress_every,
        )
    finally:
        env.close()
    return compute_aggregates(episodes), episode_match_points(episodes)


def aggregate_split(
    points_by_seed: Dict[int, List[np.ndarray]],
    *,
    setting: str,
    method: str,
    split_name: str,
    n_configs: int,
    n_episodes: int,
) -> Dict[str, Any]:
    """Aggregate one method's split across configurations and training seeds.

    Each training seed contributes one concatenated episode vector over all
    configurations in the split, so the paired bootstrap resamples episodes while
    the point estimate is the mean of per-seed means -- uncertainty therefore
    reflects both episode noise and training-seed spread.
    """
    seeds = sorted(points_by_seed)
    per_seed_vectors = [np.concatenate(points_by_seed[s]) for s in seeds if points_by_seed[s]]
    if not per_seed_vectors:
        return {}
    lengths = {v.size for v in per_seed_vectors}
    if len(lengths) == 1:
        mean, ci_lo, ci_hi = paired_bootstrap_seed_mean(per_seed_vectors)
    else:
        # Ragged (a seed is missing a configuration): fall back to a seed-mean
        # point estimate with no paired resampling rather than silently pairing
        # episodes that are not the same episodes.
        mean = 100.0 * float(np.mean([float(v.mean()) for v in per_seed_vectors]))
        ci_lo = ci_hi = float("nan")
    seed_means = [100.0 * float(v.mean()) for v in per_seed_vectors]
    return {
        "setting": setting,
        "method": method,
        "split": split_name,
        "n_seeds": len(per_seed_vectors),
        "n_configs": int(n_configs),
        "n_episodes_per_seed_per_config": int(n_episodes),
        "n_games_total": int(sum(v.size for v in per_seed_vectors)),
        "match_score_mean": mean,
        "match_score_seed_std": statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0,
        "match_score_ci_lo": ci_lo,
        "match_score_ci_hi": ci_hi,
        "win_rate_mean": 100.0
        * float(np.mean([float((v == 1.0).mean()) for v in per_seed_vectors])),
    }


def generalization_gap_row(
    seen_row: Dict[str, Any],
    heldout_row: Dict[str, Any],
    points: Dict[str, Dict[int, List[np.ndarray]]],
) -> Dict[str, Any]:
    """Seen-minus-held-out gap with a bootstrap CI over training seeds."""
    seen = float(seen_row.get("match_score_mean", float("nan")))
    heldout = float(heldout_row.get("match_score_mean", float("nan")))
    gap_lo = gap_hi = float("nan")

    seeds = sorted(set(points["seen"]) & set(points["heldout"]))
    if len(seeds) >= 2:
        gaps = []
        for s in seeds:
            seen_v = np.concatenate(points["seen"][s])
            held_v = np.concatenate(points["heldout"][s])
            gaps.append(100.0 * (float(seen_v.mean()) - float(held_v.mean())))
        rng = np.random.default_rng(0)
        arr = np.asarray(gaps, dtype=float)
        boots = np.array(
            [float(np.mean(arr[rng.integers(0, arr.size, size=arr.size)])) for _ in range(2000)]
        )
        gap_lo = float(np.quantile(boots, 0.025))
        gap_hi = float(np.quantile(boots, 0.975))

    return {
        "setting": seen_row.get("setting", ""),
        "method": seen_row.get("method", ""),
        "seen_match_score": seen,
        "heldout_match_score": heldout,
        "generalization_gap": seen - heldout,
        "gap_ci_lo": gap_lo,
        "gap_ci_hi": gap_hi,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--settings", default="2v2,3v3,4v4")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument(
        "--methods",
        default=",".join(METHOD_PATTERNS),
        help=f"Comma-separated subset of: {', '.join(METHOD_PATTERNS)}",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Episodes per training seed per configuration (200 x 3 seeds = 600 games)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-root", default=os.path.join(PROJECT_ROOT, "checkpoints_sb3"))
    parser.add_argument("--out", default="csv/generalization.csv")
    parser.add_argument("--per-seed-out", default="csv/generalization_per_seed.csv")
    parser.add_argument("--gap-out", default="csv/generalization_gap.csv")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--list", action="store_true", help="Print the protocol and plan, then exit")
    args = parser.parse_args(argv)

    settings = [s.strip().lower() for s in str(args.settings).split(",") if s.strip()]
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHOD_PATTERNS]
    if unknown:
        parser.error(f"unknown method(s) {unknown}; expected from {sorted(METHOD_PATTERNS)}")

    checkpoints = discover_checkpoints(
        str(args.checkpoint_root), settings=settings, seeds=seeds, methods=methods
    )

    print(f"[generalization] map={FIXED_MAP} rules={FIXED_RULES_PROFILE}")
    for setting in settings:
        parts = split(agents_from_setting(setting))
        print(
            f"[generalization] {setting}: {len(parts['seen'])} seen + "
            f"{len(parts['heldout'])} held-out configurations x {args.episodes} episodes x "
            f"{len(seeds)} seeds"
        )
    print(f"[generalization] {len(checkpoints)} checkpoint(s) found:")
    for (setting, method, seed), path in sorted(checkpoints.items()):
        print(f"    {setting} {method:18s} seed={seed} <- {os.path.basename(path)}")
    missing = [
        (setting, method, seed)
        for setting in settings
        for method in methods
        for seed in seeds
        if (setting, method, int(seed)) not in checkpoints
    ]
    for setting, method, seed in missing:
        print(f"    MISSING {setting} {method} seed={seed}")

    if args.list:
        print()
        print(describe_split(agents_from_setting(settings[0])))
        return 0
    if not checkpoints:
        print("[generalization] no checkpoints found; nothing to do.")
        return 1

    per_seed_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    gap_rows: List[Dict[str, Any]] = []

    for setting in settings:
        team_size = agents_from_setting(setting)
        parts = split(team_size)
        for method in methods:
            points: Dict[str, Dict[int, List[np.ndarray]]] = {"seen": {}, "heldout": {}}
            for seed in seeds:
                ckpt = checkpoints.get((setting, method, int(seed)))
                if ckpt is None:
                    continue
                for split_name in ("seen", "heldout"):
                    bucket = points[split_name].setdefault(int(seed), [])
                    for config in parts[split_name]:
                        print(
                            f"[generalization] {setting} {method} seed={seed} "
                            f"[{split_name}] {config.key}",
                            flush=True,
                        )
                        agg, pts = evaluate_configuration(
                            ckpt,
                            config,
                            n_episodes=int(args.episodes),
                            device=str(args.device),
                            progress_every=int(args.progress_every),
                        )
                        bucket.append(pts)
                        per_seed_rows.append(
                            {
                                "setting": setting,
                                "method": method,
                                "seed": int(seed),
                                "split": split_name,
                                "config": config.key,
                                "opponent_kind": config.opponent_kind,
                                "opponent_key": config.opponent_key,
                                "current_profile": config.current_profile,
                                "checkpoint": os.path.basename(ckpt),
                                "n_episodes": int(agg.get("n_episodes", 0)),
                                "wins": int(agg.get("wins", 0)),
                                "losses": int(agg.get("losses", 0)),
                                "draws": int(agg.get("draws", 0)),
                                "win_rate": float(agg.get("win_rate", float("nan"))),
                                "match_score": float(agg.get("match_score", float("nan"))),
                                "mean_captures": float(agg.get("mean_captures", float("nan"))),
                                "defense_shutout_rate": float(
                                    agg.get("defense_shutout_rate", float("nan"))
                                ),
                                "mean_steps": float(agg.get("mean_steps", float("nan"))),
                            }
                        )

            rows_by_split: Dict[str, Dict[str, Any]] = {}
            for split_name in ("seen", "heldout"):
                row = aggregate_split(
                    points[split_name],
                    setting=setting,
                    method=method,
                    split_name=split_name,
                    n_configs=len(parts[split_name]),
                    n_episodes=int(args.episodes),
                )
                if row:
                    summary_rows.append(row)
                    rows_by_split[split_name] = row
            if len(rows_by_split) == 2:
                gap_rows.append(
                    generalization_gap_row(
                        rows_by_split["seen"], rows_by_split["heldout"], points
                    )
                )

    _write_csv(per_seed_rows, str(args.per_seed_out), PER_SEED_FIELDS)
    _write_csv(summary_rows, str(args.out), SUMMARY_FIELDS)
    _write_csv(gap_rows, str(args.gap_out), GAP_FIELDS)
    print(f"[generalization] wrote {args.per_seed_out}, {args.out}, {args.gap_out}")

    for row in gap_rows:
        print(
            f"[generalization] {row['setting']} {row['method']:18s} "
            f"seen={row['seen_match_score']:.1f}% heldout={row['heldout_match_score']:.1f}% "
            f"gap={row['generalization_gap']:+.1f}pp"
        )
    print(
        "[generalization] NOTE: the per-team-size rows are a SCALABILITY result "
        "(independently trained policies), not zero-shot team-size generalization."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
