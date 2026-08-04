#!/usr/bin/env python3
"""
Cross-play payoff matrix among named methods' final checkpoints (2v2 default).

For each ordered pair (row=blue, col=red), runs blue_policy vs red_policy via
BatchedCTFCore two-policy stepping (same seam as LeagueCallback mirror eval,
without side-swap). Records W/L/D and Match Score from blue's perspective.

Usage (from AICTFProject):

  python plot/eval_crossplay.py --checkpoint-dir checkpoints_sb3/2v2 --list
  python plot/eval_crossplay.py --checkpoint-dir checkpoints_sb3/2v2 \\
      --episodes 100 --seeds 42 --out csv/crossplay_2v2.csv --device cuda
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eval_rollout import (  # noqa: E402
    count_wld,
    match_score_from_wld,
    run_two_policy_episodes,
    shared_episode_seeds,
)

_SEED_RE = re.compile(r"_seed(\d+)")


@dataclass(frozen=True)
class CrossplayMethod:
    key: str
    label: str
    patterns: Tuple[str, ...]
    # When True, require _seedN_ in basename (or default seed 42 for legacy).
    multi_seed: bool = False


# Paper methods for the cross-play matrix.
CROSSPLAY_METHODS: Tuple[CrossplayMethod, ...] = (
    CrossplayMethod(
        key="ours",
        label="SEA-GUARD",
        patterns=(
            "final_ppo_ablate_ours_2v2.zip",
            "final_ppo_ablate_ours_seed*_2v2.zip",
            "final_ppo_ablate_ours_*v*.zip",
            "final_ppo_league_2v2.zip",
            "final_ppo_league_*v*.zip",
        ),
        multi_seed=True,
    ),
    CrossplayMethod(
        key="roastar_pfsp",
        label="ROA-Star (PFSP)",
        patterns=(
            "final_ppo_roastar_pfsp_*v*_seed*.zip",
            "final_ppo_roastar_pfsp_seed*_*.zip",
        ),
        multi_seed=True,
    ),
    CrossplayMethod(
        key="no_league",
        label="No league",
        patterns=(
            "final_ppo_ablate_no_league_2v2.zip",
            "final_ppo_ablate_no_league_seed*_2v2.zip",
            "final_ppo_ablate_no_league_*v*.zip",
        ),
        multi_seed=True,
    ),
    CrossplayMethod(
        key="no_curriculum",
        label="No curriculum",
        patterns=(
            "final_ppo_ablate_no_curriculum_2v2.zip",
            "final_ppo_ablate_no_curriculum_seed*_2v2.zip",
            "final_ppo_ablate_no_curriculum_*v*.zip",
        ),
        multi_seed=True,
    ),
    CrossplayMethod(
        key="no_shaping",
        label="No shaping",
        patterns=(
            "final_ppo_ablate_no_shaping_2v2.zip",
            "final_ppo_ablate_no_shaping_seed*_2v2.zip",
            "final_ppo_ablate_no_shaping_seed*_rew_no_shaping_2v2.zip",
            "final_ppo_ablate_no_shaping_*v*.zip",
        ),
        multi_seed=True,
    ),
    CrossplayMethod(
        key="self_play",
        label="Self-play",
        patterns=(
            "final_ppo_self_play_2v2.zip",
            "final_ppo_self_play_seed*_2v2.zip",
            "final_ppo_self_play_*v*.zip",
        ),
        multi_seed=True,
    ),
)


def parse_seed_from_filename(name: str, *, default_seed: int = 42) -> int:
    m = _SEED_RE.search(name)
    return int(m.group(1)) if m else int(default_seed)


def _method_by_key(key: str) -> Optional[CrossplayMethod]:
    for m in CROSSPLAY_METHODS:
        if m.key == key:
            return m
    return None


def _matches_method(base: str, method: CrossplayMethod) -> bool:
    if method.key == "ours":
        if "ablate_ours" in base:
            return True
        if "ablate_" in base:
            return False
        if re.search(r"final_ppo_league_\d+v\d+", base):
            return True
        return False
    if method.key == "roastar_pfsp":
        return "roastar_pfsp" in base and base.startswith("final_")
    if method.key == "self_play":
        return "self_play" in base and base.startswith("final_")
    # Ablation arms
    return f"ablate_{method.key}" in base and base.startswith("final_")


def _is_junk_checkpoint(basename: str) -> bool:
    upper = basename.upper()
    return any(tok in upper for tok in ("_OLD", "_BAK", "_BACKUP", ".TMP"))


def discover_method_checkpoints(
    checkpoint_dir: str,
    method: CrossplayMethod,
    *,
    seeds: Optional[Sequence[int]] = None,
    setting: str = "2v2",
) -> Dict[int, str]:
    """Return {seed: abs_path} for this method (newest mtime wins per seed)."""
    found: Dict[int, Tuple[float, str]] = {}
    setting_token = f"_{setting}"
    for pattern in method.patterns:
        for path in glob.glob(os.path.join(checkpoint_dir, pattern)):
            if not os.path.isfile(path):
                continue
            base = os.path.basename(path)
            if _is_junk_checkpoint(base):
                continue
            if setting_token not in base and not base.endswith(f"{setting}.zip"):
                # Allow patterns that already embed setting; skip wrong sizes.
                if re.search(r"_\d+v\d+", base) and setting_token not in base:
                    continue
            if not _matches_method(base, method):
                continue
            seed = parse_seed_from_filename(base)
            if seeds is not None and seed not in {int(s) for s in seeds}:
                continue
            mtime = os.path.getmtime(path)
            prev = found.get(seed)
            # Prefer ablate_ours over league when both exist for "ours".
            if method.key == "ours" and prev is not None:
                prev_base = os.path.basename(prev[1])
                if "ablate_ours" in prev_base and "ablate_ours" not in base:
                    continue
                if "ablate_ours" in base and "ablate_ours" not in prev_base:
                    found[seed] = (mtime, os.path.abspath(path))
                    continue
            if prev is None or mtime >= prev[0]:
                found[seed] = (mtime, os.path.abspath(path))
    return {seed: path for seed, (_mt, path) in sorted(found.items())}


def discover_crossplay_policies(
    checkpoint_dir: str,
    *,
    methods: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    setting: str = "2v2",
) -> List[Dict[str, Any]]:
    """Flatten discovered policies into list of {id, key, label, seed, path}."""
    wanted = {m.lower() for m in methods} if methods else None
    policies: List[Dict[str, Any]] = []
    for method in CROSSPLAY_METHODS:
        if wanted is not None and method.key not in wanted:
            continue
        by_seed = discover_method_checkpoints(
            checkpoint_dir, method, seeds=seeds, setting=setting
        )
        for seed, path in by_seed.items():
            pid = f"{method.key}_seed{seed}"
            policies.append(
                {
                    "id": pid,
                    "key": method.key,
                    "label": method.label,
                    "seed": int(seed),
                    "path": path,
                    "basename": os.path.basename(path),
                }
            )
    return policies


def wld_to_cell(wins: int, losses: int, draws: int) -> dict:
    """Pure aggregation helper for one matrix cell (blue perspective)."""
    ms = match_score_from_wld(wins, losses, draws)
    total = int(wins) + int(losses) + int(draws)
    return {
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "n_episodes": total,
        "win_rate": (100.0 * wins / total) if total else float("nan"),
        "match_score": ms,
    }


def build_payoff_matrix(
    cell_wlds: Dict[Tuple[str, str], Tuple[int, int, int]],
    row_ids: Sequence[str],
    col_ids: Sequence[str],
) -> List[List[float]]:
    """Return matrix of blue match scores; diagonal is 50.0 (self) if missing."""
    mat: List[List[float]] = []
    for r in row_ids:
        row: List[float] = []
        for c in col_ids:
            if r == c and (r, c) not in cell_wlds:
                row.append(50.0)
                continue
            wld = cell_wlds.get((r, c))
            if wld is None:
                row.append(float("nan"))
            else:
                row.append(match_score_from_wld(*wld))
        mat.append(row)
    return mat


def matrix_rows_for_csv(
    cell_wlds: Dict[Tuple[str, str], Tuple[int, int, int]],
    policies: Sequence[Dict[str, Any]],
) -> List[dict]:
    """Long-form CSV rows for every ordered pair."""
    rows: List[dict] = []
    by_id = {p["id"]: p for p in policies}
    for blue_id, red_id in sorted(cell_wlds.keys()):
        cell = wld_to_cell(*cell_wlds[(blue_id, red_id)])
        bp = by_id.get(blue_id, {})
        rp = by_id.get(red_id, {})
        rows.append(
            {
                "blue_id": blue_id,
                "red_id": red_id,
                "blue_method": bp.get("label", blue_id),
                "red_method": rp.get("label", red_id),
                "blue_seed": bp.get("seed", ""),
                "red_seed": rp.get("seed", ""),
                "blue_ckpt": bp.get("basename", ""),
                "red_ckpt": rp.get("basename", ""),
                **cell,
            }
        )
    return rows


def format_matrix_table(
    matrix: List[List[float]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
) -> str:
    """ASCII table of match scores for stdout."""
    col_w = max(10, max((len(c) for c in col_labels), default=8) + 1)
    row_w = max(12, max((len(r) for r in row_labels), default=8) + 1)
    header = " " * row_w + "".join(f"{c:>{col_w}}" for c in col_labels)
    lines = [header]
    for label, row in zip(row_labels, matrix):
        cells = []
        for v in row:
            cells.append(f"{v:>{col_w}.1f}" if v == v else f"{'nan':>{col_w}}")
        lines.append(f"{label:<{row_w}}" + "".join(cells))
    return "\n".join(lines)


def _write_csv(rows: List[dict], path: str, fieldnames: Sequence[str]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_crossplay(
    *,
    policies: Sequence[Dict[str, Any]],
    n_agents: int,
    episodes: int,
    device: str,
    seed_base: int,
    include_diagonal: bool,
) -> Dict[Tuple[str, str], Tuple[int, int, int]]:
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    cell_wlds: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
    ep_seeds = shared_episode_seeds(episodes, seed_base, "CROSS")
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=n_agents,
        max_red_agents=n_agents,
        max_decision_steps=400,
        aquaticus_profile=True,
        rules_profile="OURS",
        device=device,
        seed=int(seed_base),
    )
    env = GPUCTFVecEnv(cfg)
    try:
        for i, blue in enumerate(policies):
            for j, red in enumerate(policies):
                if (not include_diagonal) and i == j:
                    continue
                print(
                    f"[eval_crossplay] {blue['id']} (blue) vs {red['id']} (red) "
                    f"({episodes} eps)...",
                    flush=True,
                )
                eps = run_two_policy_episodes(
                    blue["path"],
                    red["path"],
                    env,
                    episodes,
                    device,
                    episode_seeds=ep_seeds,
                )
                cell_wlds[(blue["id"], red["id"])] = count_wld(eps)
    finally:
        env.close()
    return cell_wlds


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints_sb3/2v2")
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--setting", default="2v2")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Which training seeds to include when multiple finals exist",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of method keys (ours roastar_pfsp no_league no_curriculum no_shaping self_play)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--include-diagonal", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--out", default="csv/crossplay_2v2.csv")
    args = parser.parse_args()

    policies = discover_crossplay_policies(
        args.checkpoint_dir,
        methods=args.methods,
        seeds=args.seeds,
        setting=args.setting,
    )
    if args.list or not policies:
        print(f"[eval_crossplay] {len(policies)} polic(y/ies) under {args.checkpoint_dir}")
        for p in policies:
            print(f"  {p['id']:28s}  {p['basename']}")
        if args.list:
            return 0
        if not policies:
            return 1

    cell_wlds = run_crossplay(
        policies=policies,
        n_agents=args.agents,
        episodes=args.episodes,
        device=args.device,
        seed_base=args.seed_base,
        include_diagonal=args.include_diagonal,
    )

    ids = [p["id"] for p in policies]
    labels = [f"{p['label']} s{p['seed']}" for p in policies]
    matrix = build_payoff_matrix(cell_wlds, ids, ids)
    print("\n[eval_crossplay] Match Score matrix (row=blue, col=red):")
    print(format_matrix_table(matrix, labels, labels))

    rows = matrix_rows_for_csv(cell_wlds, policies)
    fields = [
        "blue_id",
        "red_id",
        "blue_method",
        "red_method",
        "blue_seed",
        "red_seed",
        "blue_ckpt",
        "red_ckpt",
        "wins",
        "losses",
        "draws",
        "n_episodes",
        "win_rate",
        "match_score",
    ]
    _write_csv(rows, args.out, fields)
    print(f"[eval_crossplay] wrote {len(rows)} cell(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
