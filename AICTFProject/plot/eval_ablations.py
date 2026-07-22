#!/usr/bin/env python3
"""
Shared fixed-opponent evaluation for leave-one-out ablation checkpoints.

Evaluates every available final_ppo_ablate_*.zip against the same opponent suite
(OP3 / OP4 by default), seed convention, and episode count used by
plot/eval_roastar.py and plot/plot_eval_metrics.py. Aggregates across seeds into
method-level rows so training-time WRs (different opponent mixes) are not used
to compare arms.

Outputs (plot_eval_metrics.py CSV schema):
  --out            method-level mean +/- std across seeds (the paper table)
  --per-seed-out   optional per-(method, seed, opponent) detail rows

Usage (from AICTFProject):

  # Dry-run: list which finals will be evaluated
  python plot/eval_ablations.py --checkpoint-dir checkpoints_sb3/2v2 --list

  # Full shared eval (wait until all 12 finals exist for a complete table)
  python plot/eval_ablations.py --checkpoint-dir checkpoints_sb3/2v2 \\
      --episodes 100 --out csv/eval_ablation_2v2.csv \\
      --per-seed-out csv/eval_ablation_2v2_per_seed.csv

  # Plot
  python plot/plot_eval_metrics.py --metrics-csv csv/eval_ablation_2v2.csv --modes 2v2
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eval_roastar import (  # noqa: E402
    _row_from_aggregates,
    write_rows,
)


# Canonical ablation arms for the paper leave-one-out table.
@dataclass(frozen=True)
class AblationArm:
    key: str
    label: str
    # Glob fragments under checkpoint-dir (matched against basename).
    patterns: Tuple[str, ...]


ABLATION_ARMS: Tuple[AblationArm, ...] = (
    AblationArm(
        key="ours",
        label="Ours",
        patterns=(
            "final_ppo_ablate_ours_2v2.zip",
            "final_ppo_ablate_ours_seed*_2v2.zip",
        ),
    ),
    AblationArm(
        key="no_league",
        label="No league",
        patterns=(
            "final_ppo_ablate_no_league_2v2.zip",
            "final_ppo_ablate_no_league_seed*_2v2.zip",
        ),
    ),
    AblationArm(
        key="no_curriculum",
        label="No curriculum",
        patterns=(
            "final_ppo_ablate_no_curriculum_2v2.zip",
            "final_ppo_ablate_no_curriculum_seed*_2v2.zip",
        ),
    ),
    AblationArm(
        key="no_shaping",
        label="No shaping",
        patterns=(
            "final_ppo_ablate_no_shaping_2v2.zip",
            "final_ppo_ablate_no_shaping_seed*_2v2.zip",
            "final_ppo_ablate_no_shaping_seed*_rew_no_shaping_2v2.zip",
        ),
    ),
)

_SEED_RE = re.compile(r"_seed(\d+)_")


def parse_seed_from_filename(name: str, *, default_seed: int = 42) -> int:
    """Extract seed from ..._seed43_...; legacy seed-42 tags have no _seedN_."""
    m = _SEED_RE.search(name)
    if m:
        return int(m.group(1))
    return int(default_seed)


def discover_arm_checkpoints(
    checkpoint_dir: str,
    arm: AblationArm,
    *,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[int, str]:
    """
    Return {seed: absolute_path} for finals matching this arm.
    If multiple files match one seed, keep the newest by mtime.
    """
    found: Dict[int, Tuple[float, str]] = {}
    for pattern in arm.patterns:
        for path in glob.glob(os.path.join(checkpoint_dir, pattern)):
            if not os.path.isfile(path):
                continue
            base = os.path.basename(path)
            # Keep arm isolation if a broad glob ever overlaps.
            if arm.key == "ours" and any(
                x in base for x in ("no_league", "no_curriculum", "no_shaping")
            ):
                continue
            if arm.key != "ours" and f"ablate_{arm.key}" not in base:
                continue
            seed = parse_seed_from_filename(base)
            if seeds is not None and seed not in set(int(s) for s in seeds):
                continue
            mtime = os.path.getmtime(path)
            prev = found.get(seed)
            if prev is None or mtime >= prev[0]:
                found[seed] = (mtime, os.path.abspath(path))
    return {seed: path for seed, (_mt, path) in sorted(found.items())}


def discover_all_checkpoints(
    checkpoint_dir: str,
    *,
    arms: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Dict[int, str]]:
    """{arm_key: {seed: path}} for requested arms."""
    wanted = {a.lower() for a in arms} if arms else None
    out: Dict[str, Dict[int, str]] = {}
    for arm in ABLATION_ARMS:
        if wanted is not None and arm.key not in wanted:
            continue
        ckpts = discover_arm_checkpoints(checkpoint_dir, arm, seeds=seeds)
        if ckpts:
            out[arm.key] = ckpts
    return out


def arm_label(key: str) -> str:
    for arm in ABLATION_ARMS:
        if arm.key == key:
            return arm.label
    return key


_METRIC_KEYS = [
    "success_rate",
    "mean_steps",
    "collision_free_rate",
    "return_var",
    "coverage_efficiency",
    "win_margin_mean",
    "time_to_first_score_mean",
    "mean_inter_robot_dist_mean",
]


def _get_metric(agg: dict, key: str) -> float:
    # Map aggregate dict keys -> CSV mean fields.
    aliases = {
        "collision_free_rate": ("collision_free_rate", "collision_free"),
        "return_var": ("return_var", "return_variance"),
        "coverage_efficiency": ("coverage_efficiency",),
        "win_margin_mean": ("win_margin_mean", "win_margin"),
        "time_to_first_score_mean": ("time_to_first_score_mean", "time_to_first_score"),
        "mean_inter_robot_dist_mean": ("mean_inter_robot_dist_mean", "mean_inter_robot_dist"),
    }
    keys = aliases.get(key, (key,))
    for k in keys:
        if k in agg and agg[k] is not None:
            try:
                return float(agg[k])
            except (TypeError, ValueError):
                continue
    return float("nan")


def aggregate_across_seeds(
    setting: str,
    method_label: str,
    opponent: str,
    per_seed_aggs: Sequence[dict],
) -> dict:
    """
    Collapse per-seed episode aggregates into one CSV row.

    mean = mean of per-seed means; std = sample std across seeds (ddof=1),
    or 0.0 when only one seed is present.
    """
    def _mean_std(vals: List[float]) -> Tuple[float, float]:
        clean = [v for v in vals if v == v]  # drop NaN
        if not clean:
            return float("nan"), 0.0
        if len(clean) == 1:
            return clean[0], 0.0
        return statistics.mean(clean), statistics.stdev(clean)

    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for key in _METRIC_KEYS:
        vals = [_get_metric(agg, key) for agg in per_seed_aggs]
        mu, sd = _mean_std(vals)
        means[key] = mu
        stds[key] = sd

    # Build a fake aggregate dict compatible with _row_from_aggregates.
    fake = {
        "success_rate": means["success_rate"],
        "success_rate_std": stds["success_rate"],
        "mean_steps": means["mean_steps"],
        "mean_steps_std": stds["mean_steps"],
        "collision_free_rate": means["collision_free_rate"],
        "collision_free_rate_std": stds["collision_free_rate"],
        "return_var": means["return_var"],
        "return_var_std": stds["return_var"],
        "coverage_efficiency": means["coverage_efficiency"],
        "coverage_efficiency_std": stds["coverage_efficiency"],
        "win_margin_mean": means["win_margin_mean"],
        "win_margin_std": stds["win_margin_mean"],
        "time_to_first_score_mean": means["time_to_first_score_mean"],
        "time_to_first_score_std": stds["time_to_first_score_mean"],
        "mean_inter_robot_dist_mean": means["mean_inter_robot_dist_mean"],
        "mean_inter_robot_dist_std": stds["mean_inter_robot_dist_mean"],
    }
    return _row_from_aggregates(setting, method_label, opponent, fake)


def _print_discovery(discovered: Dict[str, Dict[int, str]]) -> None:
    if not discovered:
        print("[eval_ablations] no ablation finals found.")
        return
    for arm_key, by_seed in discovered.items():
        print(f"[eval_ablations] {arm_label(arm_key)} ({arm_key}): {len(by_seed)} seed(s)")
        for seed, path in by_seed.items():
            print(f"  seed={seed}  {os.path.basename(path)}")


def run_shared_eval(
    *,
    discovered: Dict[str, Dict[int, str]],
    setting: str,
    n_agents: int,
    opponents: Sequence[str],
    episodes: int,
    device: str,
    seed_base: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Returns (method_level_rows, per_seed_rows).
    """
    per_seed_rows: List[dict] = []
    method_rows: List[dict] = []

    # Collect raw aggs: arm -> opponent -> list[agg]
    from eval_rollout import compute_aggregates, run_eval_episodes
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    for arm_key, by_seed in discovered.items():
        label = arm_label(arm_key)
        aggs_by_opp: Dict[str, List[dict]] = {str(o).upper(): [] for o in opponents}

        for seed, ckpt in by_seed.items():
            for opp in opponents:
                opp_clean = str(opp).strip().upper()
                eval_seed = int(seed_base) + (1 if opp_clean == "OP4" else 0)
                cfg = GPUFieldConfig(
                    n_envs=1,
                    max_blue_agents=n_agents,
                    max_red_agents=n_agents,
                    max_decision_steps=400,
                    aquaticus_profile=True,
                    rules_profile="OURS",
                    device=device,
                    seed=eval_seed,
                )
                env = GPUCTFVecEnv(cfg)
                try:
                    print(
                        f"[eval_ablations] {label} seed={seed} vs {opp_clean} "
                        f"({episodes} ep, seed={eval_seed}) <- {os.path.basename(ckpt)}"
                    )
                    episode_dicts = run_eval_episodes(ckpt, env, episodes, device, opp_clean)
                    agg = compute_aggregates(episode_dicts)
                finally:
                    env.close()

                aggs_by_opp[opp_clean].append(agg)
                # Per-seed row uses episode-level std (same as eval_roastar single ckpt).
                seed_label = f"{label} (seed{seed})"
                per_seed_rows.append(_row_from_aggregates(setting, seed_label, opp_clean, agg))

        for opp_clean, aggs in aggs_by_opp.items():
            if not aggs:
                continue
            method_rows.append(
                aggregate_across_seeds(setting, label, opp_clean, aggs)
            )

    return method_rows, per_seed_rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.path.join("checkpoints_sb3", "2v2"),
        help="Directory containing final_ppo_ablate_*.zip",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default=None,
        help="Comma-separated subset: ours,no_league,no_curriculum,no_shaping",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed filter, e.g. 42,43,44 (default: all discovered)",
    )
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--setting", type=str, default=None, help="Default: {agents}v{agents}")
    parser.add_argument("--opponents", nargs="+", default=["OP3", "OP4"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument(
        "--out",
        default=os.path.join("csv", "eval_ablation_2v2.csv"),
        help="Method-level CSV (mean +/- std across seeds)",
    )
    parser.add_argument(
        "--per-seed-out",
        default=None,
        help="Optional per-seed detail CSV",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered finals and exit (no GPU eval)",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero unless every arm has seeds 42,43,44",
    )
    parser.add_argument(
        "--expected-seeds",
        type=str,
        default="42,43,44",
        help="Used with --require-complete (default: 42,43,44)",
    )
    args = parser.parse_args(argv)

    arms = [a.strip().lower() for a in args.arms.split(",")] if args.arms else None
    seed_filter = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    setting = args.setting or f"{args.agents}v{args.agents}"

    discovered = discover_all_checkpoints(
        args.checkpoint_dir, arms=arms, seeds=seed_filter
    )
    _print_discovery(discovered)

    if args.require_complete:
        expected = {int(s) for s in args.expected_seeds.split(",") if s.strip()}
        wanted_arms = arms or [a.key for a in ABLATION_ARMS]
        missing = []
        for arm_key in wanted_arms:
            have = set(discovered.get(arm_key, {}))
            for s in sorted(expected):
                if s not in have:
                    missing.append(f"{arm_key}/seed{s}")
        if missing:
            print("[eval_ablations] incomplete matrix, missing:")
            for m in missing:
                print(f"  - {m}")
            return 2

    if args.list:
        return 0

    if not discovered:
        print("[eval_ablations] nothing to evaluate.")
        return 1

    method_rows, per_seed_rows = run_shared_eval(
        discovered=discovered,
        setting=setting,
        n_agents=int(args.agents),
        opponents=args.opponents,
        episodes=int(args.episodes),
        device=str(args.device),
        seed_base=int(args.seed_base),
    )

    write_rows(method_rows, args.out, append=False)
    # Retag writer message
    print(f"[eval_ablations] method-level rows: {len(method_rows)} -> {args.out}")

    if args.per_seed_out:
        write_rows(per_seed_rows, args.per_seed_out, append=False)
        print(f"[eval_ablations] per-seed rows: {len(per_seed_rows)} -> {args.per_seed_out}")

    # Compact console summary (success rate)
    print("\n--- Shared eval success rate (mean +/- std across seeds) ---")
    by_method: Dict[str, Dict[str, dict]] = {}
    for row in method_rows:
        by_method.setdefault(row["method"], {})[row["opponent"]] = row
    for method, opp_map in by_method.items():
        parts = []
        for opp in args.opponents:
            opp_u = str(opp).upper()
            if opp_u not in opp_map:
                continue
            r = opp_map[opp_u]
            parts.append(
                f"{opp_u}={float(r['success_rate_mean']):.1f}%+/-{float(r['success_rate_std']):.1f}"
            )
        print(f"  {method:<16} " + "  ".join(parts))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
