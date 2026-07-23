#!/usr/bin/env python3
"""
Frozen shared OP3/OP4 evaluation for ROA-Star PFSP finals across team sizes.

This is the paper centerpiece: training win rates are NOT used. Every checkpoint
faces the same per-episode seed list against OP3 (in-distribution) and OP4
(held-out). Primary metric is Match Score = (W + 0.5D) / (W+L+D).

Evaluates:
  settings: 2v2, 3v3, 4v4
  seeds:    42, 43, 44
  opponents: OP3, OP4
  default:  1000 episodes per (checkpoint, opponent)

Usage (from AICTFProject):

  # Discover finals only
  python plot/eval_roastar_matrix.py --list

  # Full shared eval (paper protocol)
  python plot/eval_roastar_matrix.py --episodes 1000 --require-complete \\
      --out csv/eval_roastar_shared.csv \\
      --per-seed-out csv/eval_roastar_shared_per_seed.csv

  # Plot scaling + heatmap
  python plot/plot_roastar_shared_eval.py \\
      --metrics-csv csv/eval_roastar_shared.csv \\
      --per-seed-csv csv/eval_roastar_shared_per_seed.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eval_rollout import (  # noqa: E402
    compute_aggregates,
    episode_match_points,
    paired_bootstrap_seed_mean,
    run_eval_episodes,
    shared_episode_seeds,
)

_SEED_RE = re.compile(r"_seed(\d+)")
_SETTING_RE = re.compile(r"_(\d+)v(\d+)_")

METHOD_LABEL = "ROA-Star (PFSP)"

PER_SEED_FIELDS = [
    "setting",
    "method",
    "seed",
    "opponent",
    "checkpoint",
    "n_episodes",
    "wins",
    "losses",
    "draws",
    "win_rate",
    "loss_rate",
    "draw_rate",
    "match_score",
    "match_score_ci_lo",
    "match_score_ci_hi",
    "success_rate",
    "mean_steps",
    "mean_captures",
    "defense_shutout_rate",
    "collision_free_rate",
    "win_margin_mean",
    "mean_inter_robot_dist_mean",
]

METHOD_FIELDS = [
    "setting",
    "method",
    "opponent",
    "n_seeds",
    "n_episodes_per_seed",
    "win_rate_mean",
    "win_rate_std",
    "loss_rate_mean",
    "loss_rate_std",
    "draw_rate_mean",
    "draw_rate_std",
    "match_score_mean",
    "match_score_std",
    "match_score_ci_lo",
    "match_score_ci_hi",
    "success_rate_mean",
    "success_rate_std",
    "mean_steps_mean",
    "mean_steps_std",
    "mean_captures_mean",
    "mean_captures_std",
    "defense_shutout_rate_mean",
    "defense_shutout_rate_std",
    "collision_free_rate_mean",
    "collision_free_rate_std",
    "win_margin_mean",
    "win_margin_std",
]


def parse_seed(name: str) -> Optional[int]:
    m = _SEED_RE.search(name)
    return int(m.group(1)) if m else None


def parse_setting(name: str) -> Optional[str]:
    m = _SETTING_RE.search(name)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if a != b:
        return None
    return f"{a}v{a}"


def agents_from_setting(setting: str) -> int:
    return int(str(setting).split("v")[0])


def default_checkpoint_root() -> str:
    return os.path.join(PROJECT_ROOT, "checkpoints_sb3")


def discover_roastar_finals(
    checkpoint_root: str,
    *,
    settings: Sequence[str],
    seeds: Sequence[int],
    mode: str = "pfsp",
) -> Dict[str, Dict[int, str]]:
    """
    Return {setting: {seed: abs_path}} for final_ppo_roastar_<mode>_<NvN>_seedS.zip.
    """
    wanted_settings = {str(s).strip().lower() for s in settings}
    wanted_seeds = {int(s) for s in seeds}
    found: Dict[str, Dict[int, Tuple[float, str]]] = {}

    pattern = os.path.join(checkpoint_root, "**", f"final_ppo_roastar_{mode}_*.zip")
    for path in glob.glob(pattern, recursive=True):
        if not os.path.isfile(path):
            continue
        base = os.path.basename(path)
        setting = parse_setting(base)
        seed = parse_seed(base)
        if setting is None or seed is None:
            continue
        if setting not in wanted_settings or seed not in wanted_seeds:
            continue
        mtime = os.path.getmtime(path)
        bucket = found.setdefault(setting, {})
        prev = bucket.get(seed)
        if prev is None or mtime >= prev[0]:
            bucket[seed] = (mtime, os.path.abspath(path))

    return {
        setting: {seed: path for seed, (_mt, path) in sorted(by_seed.items())}
        for setting, by_seed in sorted(found.items())
    }


def _mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    clean = [float(v) for v in vals if v == v]
    if not clean:
        return float("nan"), 0.0
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def _write_csv(rows: List[dict], path: str, fieldnames: Sequence[str]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _append_csv_row(row: dict, path: str, fieldnames: Sequence[str]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    file_exists = os.path.isfile(path)
    write_header = (not file_exists) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _load_csv_rows(path: str) -> List[dict]:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _job_key(setting: str, seed: int, opponent: str) -> Tuple[str, int, str]:
    return (str(setting), int(seed), str(opponent).upper())


def _points_path(points_dir: str, setting: str, seed: int, opponent: str) -> str:
    return os.path.join(points_dir, f"{setting}_seed{seed}_{opponent.upper()}.npy")


def _completed_jobs_from_per_seed(
    rows: Sequence[dict],
    *,
    episodes: int,
) -> Dict[Tuple[str, int, str], dict]:
    """Map (setting, seed, opponent) -> row for jobs that already match episode budget."""
    out: Dict[Tuple[str, int, str], dict] = {}
    for row in rows:
        try:
            n_ep = int(float(row.get("n_episodes", 0)))
            seed = int(row["seed"])
        except (KeyError, TypeError, ValueError):
            continue
        if n_ep != int(episodes):
            continue
        key = _job_key(str(row["setting"]), seed, str(row["opponent"]))
        out[key] = dict(row)
    return out


def _agg_from_per_seed_row(row: dict) -> dict:
    """Rebuild a minimal aggregate dict from a saved per-seed CSV row."""
    return {
        "n_episodes": int(float(row["n_episodes"])),
        "wins": int(float(row.get("wins", 0))),
        "losses": int(float(row.get("losses", 0))),
        "draws": int(float(row.get("draws", 0))),
        "win_rate": float(row["win_rate"]),
        "loss_rate": float(row["loss_rate"]),
        "draw_rate": float(row["draw_rate"]),
        "match_score": float(row["match_score"]),
        "match_score_ci_lo": float(row.get("match_score_ci_lo", float("nan"))),
        "match_score_ci_hi": float(row.get("match_score_ci_hi", float("nan"))),
        "success_rate": float(row["success_rate"]),
        "mean_steps": float(row["mean_steps"]),
        "mean_captures": float(row["mean_captures"]),
        "defense_shutout_rate": float(row["defense_shutout_rate"]),
        "collision_free_rate": float(row["collision_free_rate"]),
        "win_margin_mean": float(row["win_margin_mean"]),
        "mean_inter_robot_dist_mean": float(
            row.get("mean_inter_robot_dist_mean", float("nan"))
        ),
    }


def _per_seed_row(
    *,
    setting: str,
    seed: int,
    opponent: str,
    checkpoint: str,
    agg: dict,
) -> dict:
    return {
        "setting": setting,
        "method": METHOD_LABEL,
        "seed": int(seed),
        "opponent": opponent,
        "checkpoint": os.path.basename(checkpoint),
        "n_episodes": int(agg.get("n_episodes", 0)),
        "wins": int(agg.get("wins", 0)),
        "losses": int(agg.get("losses", 0)),
        "draws": int(agg.get("draws", 0)),
        "win_rate": float(agg.get("win_rate", float("nan"))),
        "loss_rate": float(agg.get("loss_rate", float("nan"))),
        "draw_rate": float(agg.get("draw_rate", float("nan"))),
        "match_score": float(agg.get("match_score", float("nan"))),
        "match_score_ci_lo": float(agg.get("match_score_ci_lo", float("nan"))),
        "match_score_ci_hi": float(agg.get("match_score_ci_hi", float("nan"))),
        "success_rate": float(agg.get("success_rate", float("nan"))),
        "mean_steps": float(agg.get("mean_steps", float("nan"))),
        "mean_captures": float(agg.get("mean_captures", float("nan"))),
        "defense_shutout_rate": float(agg.get("defense_shutout_rate", float("nan"))),
        "collision_free_rate": float(agg.get("collision_free_rate", float("nan"))),
        "win_margin_mean": float(agg.get("win_margin_mean", float("nan"))),
        "mean_inter_robot_dist_mean": float(agg.get("mean_inter_robot_dist_mean", float("nan"))),
    }


def aggregate_setting_opponent(
    setting: str,
    opponent: str,
    per_seed_aggs: Sequence[dict],
    per_seed_points: Sequence[np.ndarray],
    *,
    n_boot: int,
) -> dict:
    wr_m, wr_s = _mean_std([float(a["win_rate"]) for a in per_seed_aggs])
    lr_m, lr_s = _mean_std([float(a["loss_rate"]) for a in per_seed_aggs])
    dr_m, dr_s = _mean_std([float(a["draw_rate"]) for a in per_seed_aggs])
    ms_m, ms_s = _mean_std([float(a["match_score"]) for a in per_seed_aggs])
    sr_m, sr_s = _mean_std([float(a["success_rate"]) for a in per_seed_aggs])
    st_m, st_s = _mean_std([float(a["mean_steps"]) for a in per_seed_aggs])
    cap_m, cap_s = _mean_std([float(a["mean_captures"]) for a in per_seed_aggs])
    shut_m, shut_s = _mean_std([float(a["defense_shutout_rate"]) for a in per_seed_aggs])
    cf_m, cf_s = _mean_std([float(a["collision_free_rate"]) for a in per_seed_aggs])
    wm_m, wm_s = _mean_std([float(a["win_margin_mean"]) for a in per_seed_aggs])

    # Paired bootstrap over shared episode indices (primary CI for the paper table).
    _point, ci_lo, ci_hi = paired_bootstrap_seed_mean(
        list(per_seed_points),
        n_boot=n_boot,
        alpha=0.05,
        rng=np.random.default_rng(0),
    )
    # Prefer seed-mean as the reported mean (matches "mean across three training seeds").
    match_mean = ms_m if ms_m == ms_m else _point

    n_eps = int(per_seed_aggs[0].get("n_episodes", 0)) if per_seed_aggs else 0
    return {
        "setting": setting,
        "method": METHOD_LABEL,
        "opponent": opponent,
        "n_seeds": len(per_seed_aggs),
        "n_episodes_per_seed": n_eps,
        "win_rate_mean": wr_m,
        "win_rate_std": wr_s,
        "loss_rate_mean": lr_m,
        "loss_rate_std": lr_s,
        "draw_rate_mean": dr_m,
        "draw_rate_std": dr_s,
        "match_score_mean": match_mean,
        "match_score_std": ms_s,
        "match_score_ci_lo": ci_lo,
        "match_score_ci_hi": ci_hi,
        "success_rate_mean": sr_m,
        "success_rate_std": sr_s,
        "mean_steps_mean": st_m,
        "mean_steps_std": st_s,
        "mean_captures_mean": cap_m,
        "mean_captures_std": cap_s,
        "defense_shutout_rate_mean": shut_m,
        "defense_shutout_rate_std": shut_s,
        "collision_free_rate_mean": cf_m,
        "collision_free_rate_std": cf_s,
        "win_margin_mean": wm_m,
        "win_margin_std": wm_s,
    }


def run_matrix_eval(
    *,
    discovered: Dict[str, Dict[int, str]],
    opponents: Sequence[str],
    episodes: int,
    device: str,
    seed_base: int,
    progress_every: int,
    n_boot: int,
    fixed_episode_seeds: bool,
    per_seed_out: Optional[str] = None,
    points_dir: Optional[str] = None,
    resume: bool = True,
) -> Tuple[List[dict], List[dict]]:
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    per_seed_rows: List[dict] = []
    method_rows: List[dict] = []

    existing_rows = _load_csv_rows(per_seed_out) if (resume and per_seed_out) else []
    completed = (
        _completed_jobs_from_per_seed(existing_rows, episodes=episodes) if resume else {}
    )
    if completed:
        print(f"[eval_roastar_matrix] resume: {len(completed)} completed job(s) will be skipped")

    if points_dir:
        os.makedirs(points_dir, exist_ok=True)

    # Keep in-memory rows ordered; start from any resumed rows that still apply.
    seen_keys = set()
    for row in existing_rows:
        try:
            key = _job_key(str(row["setting"]), int(row["seed"]), str(row["opponent"]))
            if key in completed and key not in seen_keys:
                per_seed_rows.append(dict(row))
                seen_keys.add(key)
        except (KeyError, TypeError, ValueError):
            continue

    for setting, by_seed in discovered.items():
        n_agents = agents_from_setting(setting)
        for opp in opponents:
            opp_clean = str(opp).strip().upper()
            ep_seeds = (
                shared_episode_seeds(episodes, seed_base, opp_clean)
                if fixed_episode_seeds
                else None
            )
            # Env construction seed: OP4 offset kept for compatibility with older scripts.
            env_seed = int(seed_base) + (1 if opp_clean == "OP4" else 0)

            aggs: List[dict] = []
            points: List[np.ndarray] = []
            for seed, ckpt in sorted(by_seed.items()):
                key = _job_key(setting, int(seed), opp_clean)
                pts_path = (
                    _points_path(points_dir, setting, int(seed), opp_clean)
                    if points_dir
                    else None
                )

                if key in completed and pts_path and os.path.isfile(pts_path):
                    row = completed[key]
                    agg = _agg_from_per_seed_row(row)
                    pts = np.load(pts_path)
                    print(
                        f"[eval_roastar_matrix] skip finished: {setting} seed={seed} vs {opp_clean} "
                        f"(MS={float(row['match_score']):.1f}%)",
                        flush=True,
                    )
                    aggs.append(agg)
                    points.append(np.asarray(pts, dtype=float))
                    continue

                if key in completed and not (pts_path and os.path.isfile(pts_path)):
                    print(
                        f"[eval_roastar_matrix] re-run {setting} seed={seed} vs {opp_clean}: "
                        "per-seed row exists but episode points missing",
                        flush=True,
                    )

                cfg = GPUFieldConfig(
                    n_envs=1,
                    max_blue_agents=n_agents,
                    max_red_agents=n_agents,
                    max_decision_steps=400,
                    aquaticus_profile=True,
                    rules_profile="OURS",
                    device=device,
                    seed=env_seed,
                )
                env = GPUCTFVecEnv(cfg)
                try:
                    print(
                        f"[eval_roastar_matrix] {setting} seed={seed} vs {opp_clean} "
                        f"({episodes} ep, fixed_seeds={fixed_episode_seeds}) "
                        f"<- {os.path.basename(ckpt)}",
                        flush=True,
                    )
                    episode_dicts = run_eval_episodes(
                        ckpt,
                        env,
                        episodes,
                        device,
                        opp_clean,
                        progress_every=progress_every,
                        episode_seeds=ep_seeds,
                    )
                    agg = compute_aggregates(episode_dicts)
                finally:
                    env.close()

                pts = episode_match_points(episode_dicts)
                row = _per_seed_row(
                    setting=setting,
                    seed=seed,
                    opponent=opp_clean,
                    checkpoint=ckpt,
                    agg=agg,
                )
                aggs.append(agg)
                points.append(pts)
                if key not in seen_keys:
                    per_seed_rows.append(row)
                    seen_keys.add(key)
                else:
                    # Replace stale resumed row with freshly computed one.
                    per_seed_rows = [
                        r
                        for r in per_seed_rows
                        if _job_key(str(r["setting"]), int(r["seed"]), str(r["opponent"])) != key
                    ]
                    per_seed_rows.append(row)

                if pts_path:
                    np.save(pts_path, pts)
                if per_seed_out:
                    # Rewrite full per-seed CSV so resume state stays consistent.
                    _write_csv(per_seed_rows, per_seed_out, PER_SEED_FIELDS)

                print(
                    f"  -> W/L/D={agg['wins']}/{agg['losses']}/{agg['draws']} "
                    f"WR={agg['win_rate']:.1f}% DR={agg['draw_rate']:.1f}% "
                    f"MS={agg['match_score']:.1f}% "
                    f"CI[{agg['match_score_ci_lo']:.1f},{agg['match_score_ci_hi']:.1f}]",
                    flush=True,
                )

            method_rows.append(
                aggregate_setting_opponent(
                    setting, opp_clean, aggs, points, n_boot=n_boot
                )
            )

    return method_rows, per_seed_rows


def _print_main_table(method_rows: Sequence[dict]) -> None:
    print("\n=== Main results (mean across training seeds; Match Score primary) ===")
    print(
        f"{'Team':<6} {'Opp':<4} {'Win%':>8} {'Draw%':>8} {'Match':>8} {'95% CI':>18} {'n_ep/seed':>10}"
    )
    order = [("2v2", "OP3"), ("2v2", "OP4"), ("3v3", "OP3"), ("3v3", "OP4"), ("4v4", "OP3"), ("4v4", "OP4")]
    by_key = {(r["setting"], r["opponent"]): r for r in method_rows}
    for setting, opp in order:
        r = by_key.get((setting, opp))
        if not r:
            continue
        ci = f"[{r['match_score_ci_lo']:.1f}, {r['match_score_ci_hi']:.1f}]"
        print(
            f"{setting:<6} {opp:<4} "
            f"{r['win_rate_mean']:8.1f} {r['draw_rate_mean']:8.1f} "
            f"{r['match_score_mean']:8.1f} {ci:>18} {int(r['n_episodes_per_seed']):10d}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="Root containing 2v2/3v3/4v4 subdirs (default: checkpoints_sb3)",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default="2v2,3v3,4v4",
        help="Comma-separated team sizes",
    )
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--mode", type=str, default="pfsp", help="ROA mode tag in filenames")
    parser.add_argument("--opponents", nargs="+", default=["OP3", "OP4"])
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Episodes per (checkpoint, opponent); paper default 1000",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument(
        "--no-fixed-episode-seeds",
        action="store_true",
        help="Disable per-episode reseeding (not recommended for paper comparisons)",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument(
        "--out",
        default=os.path.join("csv", "eval_roastar_shared.csv"),
        help="Method-level CSV (seed means + paired bootstrap CI)",
    )
    parser.add_argument(
        "--per-seed-out",
        default=os.path.join("csv", "eval_roastar_shared_per_seed.csv"),
        help="Per-(setting, seed, opponent) CSV (rewritten after each job for resume)",
    )
    parser.add_argument(
        "--points-dir",
        default=os.path.join("csv", "eval_roastar_shared_points"),
        help="Directory for per-job episode match-point .npy files (needed to resume bootstrap)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip jobs already present in --per-seed-out with matching episode count (default: true)",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit 2 unless every setting has every requested seed",
    )
    args = parser.parse_args(argv)

    ckpt_root = args.checkpoint_root or default_checkpoint_root()
    if not os.path.isabs(ckpt_root):
        ckpt_root = os.path.join(PROJECT_ROOT, ckpt_root)

    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    discovered = discover_roastar_finals(
        ckpt_root, settings=settings, seeds=seeds, mode=args.mode
    )

    print(f"[eval_roastar_matrix] checkpoint_root={ckpt_root}")
    if not discovered:
        print("[eval_roastar_matrix] no ROA-Star finals found.")
        return 1
    for setting, by_seed in discovered.items():
        print(f"[eval_roastar_matrix] {setting}: {len(by_seed)} seed(s)")
        for seed, path in by_seed.items():
            print(f"  seed={seed}  {path}")

    if args.require_complete:
        missing = []
        for setting in settings:
            have = set(discovered.get(setting, {}))
            for seed in seeds:
                if seed not in have:
                    missing.append(f"{setting}/seed{seed}")
        if missing:
            print("[eval_roastar_matrix] incomplete matrix, missing:")
            for m in missing:
                print(f"  - {m}")
            return 2

    if args.list:
        return 0

    method_rows, per_seed_rows = run_matrix_eval(
        discovered=discovered,
        opponents=args.opponents,
        episodes=int(args.episodes),
        device=str(args.device),
        seed_base=int(args.seed_base),
        progress_every=int(args.progress_every),
        n_boot=int(args.n_boot),
        fixed_episode_seeds=not bool(args.no_fixed_episode_seeds),
        per_seed_out=str(args.per_seed_out) if args.per_seed_out else None,
        points_dir=str(args.points_dir) if args.points_dir else None,
        resume=bool(args.resume),
    )

    _write_csv(method_rows, args.out, METHOD_FIELDS)
    print(f"[eval_roastar_matrix] method-level -> {args.out}")
    if args.per_seed_out:
        _write_csv(per_seed_rows, args.per_seed_out, PER_SEED_FIELDS)
        print(f"[eval_roastar_matrix] per-seed -> {args.per_seed_out}")

    _print_main_table(method_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
