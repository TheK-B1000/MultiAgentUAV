#!/usr/bin/env python3
"""
VGC-Bench style checkpoint selection among league snapshots for one run_tag.

Discovers ``{run_tag}_league_snapshot_ep*.zip`` (plus optional ``final_{run_tag}.zip``),
optionally annotates training metrics from ``{run_tag}_metrics.csv``, evaluates each
candidate vs OP3 (and optionally OP4) on a frozen val seed list, keeps the top
fraction by OP3 match score, cross-plays survivors, and writes a CSV with the winner.

Usage (from AICTFProject):

  python plot/checkpoint_tournament.py --checkpoint-dir checkpoints_sb3/2v2 \\
      --run-tag ppo_league_2v2 --agents 2 --list

  python plot/checkpoint_tournament.py --checkpoint-dir checkpoints_sb3/2v2 \\
      --run-tag ppo_league_2v2 --agents 2 --val-episodes 50 --cross-episodes 20 \\
      --out csv/tournament_ppo_league_2v2.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eval_rollout import (  # noqa: E402
    compute_aggregates,
    count_wld,
    match_score_from_wld,
    run_eval_episodes,
    run_two_policy_episodes,
    shared_episode_seeds,
)

_SNAPSHOT_EP_RE = re.compile(r"_league_snapshot_ep(\d+)\.zip$", re.IGNORECASE)
_SUCCESS_COLS = ("success", "win", "won", "is_win")
_WR_COLS = ("win_rate", "wr", "success_rate", "match_score")


def discover_tournament_candidates(
    checkpoint_dir: str,
    run_tag: str,
    *,
    include_final: bool = True,
) -> List[Dict[str, Any]]:
    """Return candidate dicts with path, basename, episode (or None for final)."""
    pattern = os.path.join(checkpoint_dir, f"{run_tag}_league_snapshot_ep*.zip")
    found: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(glob.glob(pattern)):
        if not os.path.isfile(path):
            continue
        abs_path = os.path.abspath(path)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        base = os.path.basename(path)
        m = _SNAPSHOT_EP_RE.search(base)
        ep = int(m.group(1)) if m else None
        found.append(
            {
                "path": abs_path,
                "basename": base,
                "episode": ep,
                "kind": "snapshot",
            }
        )
    found.sort(key=lambda c: (c["episode"] is None, c["episode"] or -1, c["basename"]))

    if include_final:
        final_path = os.path.join(checkpoint_dir, f"final_{run_tag}.zip")
        if os.path.isfile(final_path):
            abs_final = os.path.abspath(final_path)
            if abs_final not in seen:
                found.append(
                    {
                        "path": abs_final,
                        "basename": os.path.basename(final_path),
                        "episode": None,
                        "kind": "final",
                    }
                )
    return found


def _safe_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _row_success(row: dict) -> Optional[float]:
    for col in _SUCCESS_COLS:
        if col in row:
            v = _safe_float(row.get(col))
            if v is not None:
                return 1.0 if v >= 0.5 else 0.0
    for col in _WR_COLS:
        if col in row:
            v = _safe_float(row.get(col))
            if v is not None:
                # Accept either fraction or percentage.
                return v / 100.0 if v > 1.0 else v
    return None


def load_metrics_scores(
    metrics_csv: str,
    candidates: Sequence[Dict[str, Any]],
    *,
    window: int = 200,
) -> Dict[str, float]:
    """Map candidate path -> training success/WR score near snapshot episode.

    Returns empty dict if CSV missing or has no usable success/WR columns.
    """
    if not metrics_csv or not os.path.isfile(metrics_csv):
        return {}
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}

    # Require at least one usable success/WR column across the file.
    sample_ok = any(_row_success(r) is not None for r in rows[: min(50, len(rows))])
    if not sample_ok:
        return {}

    episode_ids: List[int] = []
    successes: List[float] = []
    for row in rows:
        sid = row.get("episode_id", row.get("episode", ""))
        try:
            eid = int(float(sid))
        except (TypeError, ValueError):
            continue
        s = _row_success(row)
        if s is None:
            continue
        episode_ids.append(eid)
        successes.append(float(s))
    if not episode_ids:
        return {}

    out: Dict[str, float] = {}
    for cand in candidates:
        ep = cand.get("episode")
        path = str(cand["path"])
        if ep is None:
            # Final: use last ``window`` successes.
            tail = successes[-max(1, int(window)) :]
            out[path] = 100.0 * (sum(tail) / len(tail))
            continue
        # Window ending at (or nearest below) snapshot episode.
        idxs = [i for i, eid in enumerate(episode_ids) if eid <= int(ep)]
        if not idxs:
            idxs = list(range(min(len(episode_ids), max(1, int(window)))))
        else:
            end = idxs[-1] + 1
            start = max(0, end - max(1, int(window)))
            idxs = list(range(start, end))
        vals = [successes[i] for i in idxs]
        out[path] = 100.0 * (sum(vals) / max(1, len(vals)))
    return out


def select_topk_indices(
    scores: Sequence[float],
    *,
    top_frac: float = 0.30,
    min_keep: int = 3,
) -> List[int]:
    """Return indices of the top fraction by score (descending), at least min_keep."""
    n = len(scores)
    if n == 0:
        return []
    keep = max(int(min_keep), int(math.ceil(float(top_frac) * n)))
    keep = min(keep, n)
    order = sorted(range(n), key=lambda i: (scores[i], -i), reverse=True)
    return order[:keep]


def mean_crossplay_score(pair_scores: Sequence[float]) -> float:
    """Mean of finite match scores (e.g. A-vs-B and optional B-vs-A)."""
    vals = [float(x) for x in pair_scores if x == x]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def aggregate_crossplay_matrix(
    cells: Dict[Tuple[str, str], Tuple[int, int, int]],
) -> Dict[str, float]:
    """Mean match score per row policy across opponents (diagonal skipped).

    ``cells`` maps (blue_id, red_id) -> (W, L, D) from blue's perspective.
    """
    ids = sorted({a for a, _ in cells} | {b for _, b in cells})
    means: Dict[str, float] = {}
    for blue in ids:
        scores: List[float] = []
        for red in ids:
            if blue == red:
                continue
            wld = cells.get((blue, red))
            if wld is None:
                continue
            scores.append(match_score_from_wld(*wld))
        means[blue] = mean_crossplay_score(scores)
    return means


def _write_csv(rows: List[dict], path: str, fieldnames: Sequence[str]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _eval_vs_scripted(
    checkpoint: str,
    *,
    n_agents: int,
    opponents: Sequence[str],
    episodes: int,
    device: str,
    seed_base: int,
) -> Dict[str, dict]:
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    out: Dict[str, dict] = {}
    for opp in opponents:
        opp_clean = str(opp).strip().upper()
        ep_seeds = shared_episode_seeds(episodes, seed_base, opp_clean)
        cfg = GPUFieldConfig(
            n_envs=1,
            max_blue_agents=n_agents,
            max_red_agents=n_agents,
            max_decision_steps=400,
            aquaticus_profile=True,
            rules_profile="OURS",
            device=device,
            seed=int(seed_base) + (1 if opp_clean == "OP4" else 0),
        )
        env = GPUCTFVecEnv(cfg)
        try:
            episodes_dicts = run_eval_episodes(
                checkpoint,
                env,
                episodes,
                device,
                opp_clean,
                episode_seeds=ep_seeds,
            )
            out[opp_clean] = compute_aggregates(episodes_dicts)
        finally:
            env.close()
    return out


def _crossplay_pair(
    blue_path: str,
    red_path: str,
    *,
    n_agents: int,
    episodes: int,
    device: str,
    seed_base: int,
) -> Tuple[int, int, int]:
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

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
        eps = run_two_policy_episodes(
            blue_path,
            red_path,
            env,
            episodes,
            device,
            episode_seeds=ep_seeds,
        )
    finally:
        env.close()
    return count_wld(eps)


def run_tournament(
    *,
    checkpoint_dir: str,
    run_tag: str,
    n_agents: int,
    device: str,
    val_episodes: int,
    cross_episodes: int,
    top_frac: float,
    min_keep: int,
    opponents: Sequence[str],
    seed_base: int,
    include_final: bool,
    swap_sides: bool,
    list_only: bool,
) -> Tuple[List[dict], Optional[str]]:
    candidates = discover_tournament_candidates(
        checkpoint_dir, run_tag, include_final=include_final
    )
    metrics_path = os.path.join(checkpoint_dir, f"{run_tag}_metrics.csv")
    metrics_scores = load_metrics_scores(metrics_path, candidates)

    if list_only:
        print(f"[checkpoint_tournament] {len(candidates)} candidate(s) for run_tag={run_tag}")
        for c in candidates:
            ms = metrics_scores.get(c["path"])
            ms_s = f"{ms:.1f}" if ms is not None else "n/a"
            print(
                f"  ep={c['episode']!s:>8}  train_score={ms_s:>6}  {c['basename']}"
            )
        return [], None

    if not candidates:
        print(f"[checkpoint_tournament] no snapshots found for run_tag={run_tag}")
        return [], None

    rows: List[dict] = []
    for c in candidates:
        print(f"[checkpoint_tournament] val-eval {c['basename']} ...", flush=True)
        aggs = _eval_vs_scripted(
            c["path"],
            n_agents=n_agents,
            opponents=opponents,
            episodes=val_episodes,
            device=device,
            seed_base=seed_base,
        )
        op3 = aggs.get("OP3", {})
        op4 = aggs.get("OP4", {})
        train_score = metrics_scores.get(c["path"], float("nan"))
        rows.append(
            {
                "run_tag": run_tag,
                "basename": c["basename"],
                "path": c["path"],
                "kind": c["kind"],
                "episode": "" if c["episode"] is None else int(c["episode"]),
                "train_score": train_score,
                "op3_wins": op3.get("wins", ""),
                "op3_losses": op3.get("losses", ""),
                "op3_draws": op3.get("draws", ""),
                "op3_match_score": op3.get("match_score", float("nan")),
                "op4_match_score": op4.get("match_score", "") if op4 else "",
                "survivor": 0,
                "cross_match_score": "",
                "selected": 0,
            }
        )

    # Rank by OP3 match score (primary); train_score is informational only.
    op3_scores = [
        float(r["op3_match_score"]) if r["op3_match_score"] == r["op3_match_score"] else float("-inf")
        for r in rows
    ]
    keep_idx = select_topk_indices(op3_scores, top_frac=top_frac, min_keep=min_keep)
    survivors = [rows[i] for i in keep_idx]
    for i in keep_idx:
        rows[i]["survivor"] = 1

    print(
        f"[checkpoint_tournament] keeping {len(survivors)}/{len(rows)} survivors for cross-play",
        flush=True,
    )

    # Cross-play: each survivor as blue vs every other as red (+ optional swap).
    cross_means: Dict[str, List[float]] = {s["path"]: [] for s in survivors}
    for i, a in enumerate(survivors):
        for j, b in enumerate(survivors):
            if i == j:
                continue
            print(
                f"[checkpoint_tournament] cross {a['basename']} (blue) vs {b['basename']} (red)",
                flush=True,
            )
            w, l, d = _crossplay_pair(
                a["path"],
                b["path"],
                n_agents=n_agents,
                episodes=cross_episodes,
                device=device,
                seed_base=seed_base + 17 * i + j,
            )
            ms = match_score_from_wld(w, l, d)
            cross_means[a["path"]].append(ms)
            if swap_sides:
                print(
                    f"[checkpoint_tournament] cross {b['basename']} (blue) vs {a['basename']} (red) [swap]",
                    flush=True,
                )
                w2, l2, d2 = _crossplay_pair(
                    b["path"],
                    a["path"],
                    n_agents=n_agents,
                    episodes=cross_episodes,
                    device=device,
                    seed_base=seed_base + 1700 + 17 * j + i,
                )
                # From A's perspective when A is red: A wins when blue (B) loses.
                # Record A's mirror score as red's complement of B-as-blue score.
                ms_a_as_red = match_score_from_wld(l2, w2, d2)
                cross_means[a["path"]].append(ms_a_as_red)

    best_path: Optional[str] = None
    best_score = float("-inf")
    for row in rows:
        if not row["survivor"]:
            continue
        mean_ms = mean_crossplay_score(cross_means.get(row["path"], []))
        row["cross_match_score"] = mean_ms
        if mean_ms == mean_ms and mean_ms > best_score:
            best_score = mean_ms
            best_path = row["path"]

    if best_path is not None:
        for row in rows:
            if row["path"] == best_path:
                row["selected"] = 1
                break
        print(
            f"[checkpoint_tournament] selected {os.path.basename(best_path)} "
            f"(cross MS={best_score:.1f}%)",
            flush=True,
        )
    return rows, best_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints_sb3/2v2")
    parser.add_argument("--run-tag", required=True, help="e.g. ppo_league_2v2 or ppo_roastar_pfsp_2v2_seed42")
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val-episodes", type=int, default=50)
    parser.add_argument("--cross-episodes", type=int, default=20)
    parser.add_argument("--top-frac", type=float, default=0.30)
    parser.add_argument("--min-keep", type=int, default=3)
    parser.add_argument("--opponents", nargs="+", default=["OP3"], help="Val opponents (OP3 required for ranking)")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--no-final", action="store_true", help="Do not include final_{run_tag}.zip")
    parser.add_argument("--no-swap", action="store_true", help="Skip side-swapped cross-play direction")
    parser.add_argument("--list", action="store_true", help="List candidates and exit")
    parser.add_argument("--out", default=None, help="CSV summary path")
    args = parser.parse_args()

    rows, selected = run_tournament(
        checkpoint_dir=args.checkpoint_dir,
        run_tag=args.run_tag,
        n_agents=args.agents,
        device=args.device,
        val_episodes=args.val_episodes,
        cross_episodes=args.cross_episodes,
        top_frac=args.top_frac,
        min_keep=args.min_keep,
        opponents=args.opponents,
        seed_base=args.seed_base,
        include_final=not args.no_final,
        swap_sides=not args.no_swap,
        list_only=args.list,
    )
    if args.list:
        return 0
    if not rows:
        return 1

    out_path = args.out or os.path.join(
        "csv", f"tournament_{args.run_tag}.csv"
    )
    fields = [
        "run_tag",
        "basename",
        "path",
        "kind",
        "episode",
        "train_score",
        "op3_wins",
        "op3_losses",
        "op3_draws",
        "op3_match_score",
        "op4_match_score",
        "survivor",
        "cross_match_score",
        "selected",
    ]
    _write_csv(rows, out_path, fields)
    print(f"[checkpoint_tournament] wrote {len(rows)} row(s) -> {out_path}")
    if selected:
        print(f"[checkpoint_tournament] winner: {selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
