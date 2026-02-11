"""
Compare League vs No-League: same seeds, same opponents (OP3 and sharpened).
Quiet run; single results table at the end.

  python compare_league_vs_noleague.py --episodes 50 --headless

Optional: --league-model, --no-league-model to override default paths.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

DEFAULT_LEAGUE = os.path.join(_SCRIPT_DIR, "checkpoints_sb3", "final_ppo_league.zip")
DEFAULT_NO_LEAGUE = os.path.join(_SCRIPT_DIR, "checkpoints_sb3", "final_ppo_noleague.zip")
SHARPENED = ("INTERCEPTOR", "MINELAYER", "CAMPER_ROTATE", "BAIT_SWITCH")


def _run_one(
    model_path: str,
    num_episodes: int,
    seeds: List[int],
    headless: bool,
    opponent: str,
) -> Dict[str, Any]:
    from ctfviewer import CTFViewer
    out = {"win_rate": 0.0, "wins": 0, "losses": 0, "draws": 0, "error": None}
    try:
        v = CTFViewer(ppo_model_path=model_path, viewer_use_obs_builder=True)
        if not v.blue_ppo_team.model_loaded:
            out["error"] = "Model failed to load"
            return out
        s = v.evaluate_model(
            num_episodes=num_episodes,
            headless=headless,
            opponent=opponent,
            eval_model="ppo",
            episode_seeds=seeds,
            quiet=True,
        )
        out["win_rate"] = s.get("win_rate", 0.0)
        out["wins"] = s.get("wins", 0)
        out["losses"] = s.get("losses", 0)
        out["draws"] = s.get("draws", 0)
    except Exception as e:
        out["error"] = str(e)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="League vs No-League: OP3 + sharpened. Results at end.")
    p.add_argument("--league-model", type=str, default=DEFAULT_LEAGUE, help="League checkpoint")
    p.add_argument("--no-league-model", type=str, default=DEFAULT_NO_LEAGUE, help="No-league checkpoint")
    p.add_argument("--episodes", type=int, default=50, help="Episodes per condition")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--headless", action="store_true", help="Headless")
    args = p.parse_args()

    if not os.path.exists(args.league_model):
        print(f"Error: league model not found: {args.league_model}")
        sys.exit(1)
    if not os.path.exists(args.no_league_model):
        print(f"Error: no-league model not found: {args.no_league_model}")
        sys.exit(1)

    n = max(1, args.episodes)
    seed = args.seed

    # OP3
    seeds_op3 = [seed + i for i in range(n)]
    league_op3 = _run_one(args.league_model, n, seeds_op3, args.headless, "OP3")
    no_league_op3 = _run_one(args.no_league_model, n, seeds_op3, args.headless, "OP3")

    # Sharpened (aggregate over all four)
    league_w, league_l, league_d = 0, 0, 0
    no_league_w, no_league_l, no_league_d = 0, 0, 0
    per = max(1, n // len(SHARPENED))
    for i, opp in enumerate(SHARPENED):
        count = per if i < len(SHARPENED) - 1 else (n - per * (len(SHARPENED) - 1))
        seeds = [seed + 2000 + i * 500 + j for j in range(count)]
        rl = _run_one(args.league_model, count, seeds, args.headless, opp)
        rn = _run_one(args.no_league_model, count, seeds, args.headless, opp)
        league_w += rl.get("wins", 0)
        league_l += rl.get("losses", 0)
        league_d += rl.get("draws", 0)
        no_league_w += rn.get("wins", 0)
        no_league_l += rn.get("losses", 0)
        no_league_d += rn.get("draws", 0)
    total_s = league_w + league_l + league_d
    league_sharp = {"win_rate": league_w / total_s if total_s else 0.0, "wins": league_w, "losses": league_l, "draws": league_d}
    total_n = no_league_w + no_league_l + no_league_d
    no_league_sharp = {"win_rate": no_league_w / total_n if total_n else 0.0, "wins": no_league_w, "losses": no_league_l, "draws": no_league_d}

    # Results table
    print("\n" + "=" * 60)
    print("LEAGUE vs NO-LEAGUE")
    print("=" * 60)
    print(f"Episodes per condition: {n}  |  Seed: {seed}")
    print()
    print(f"{'Condition':<16} {'League WR':>12} {'No-League WR':>14} {'Diff':>8}")
    print("-" * 60)
    for label, lr, nr in [
        ("vs OP3", league_op3, no_league_op3),
        ("vs Sharpened", league_sharp, no_league_sharp),
    ]:
        lw = lr.get("win_rate", 0.0) if not lr.get("error") else float("nan")
        nw = nr.get("win_rate", 0.0) if not nr.get("error") else float("nan")
        diff = (lw - nw) if (lw == lw and nw == nw) else float("nan")
        s1 = f"{lw:.1%}" if lw == lw else "ERROR"
        s2 = f"{nw:.1%}" if nw == nw else "ERROR"
        s3 = f"{diff:+.1%}" if diff == diff else "n/a"
        print(f"{label:<16} {s1:>12} {s2:>14} {s3:>8}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
