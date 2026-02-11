"""
Compare League vs No-League Training (same N test scenarios)

Runs the same N episodes (same seeds) for both models so the comparison is fair:
same opponent, same stress, same initial conditions. Both models see the exact
same scenarios.

Why can league show a lower win rate here than when you "test league alone"?
- Testing alone usually uses different random seeds each run, so you get a different
  set of scenarios; by chance those can be easier and league wins 100%.
- Here we fix seeds (e.g. 42, 43, ...) so both models face the same N scenarios.
  That set may include some harder ones (spawns, opponent RNG), so league might
  lose a few. Use --episodes 30 or 50 for a more stable WR, or --seed N to try
  another scenario set.
"""
import argparse
import os
import sys
from typing import Dict, List, Optional

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from ctfviewer import CTFViewer


def run_comparison_test(
    league_model_path: str,
    no_league_model_path: str,
    num_episodes: int = 10,
    opponent: str = "OP3",
    headless: bool = True,
    episode_seeds: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run comparison test: same N episodes (same seeds) for both models so the
    comparison is fair. Same environment: OP3 opponent, OP3 stress.
    
    Args:
        league_model_path: Path to league-trained model (.zip)
        no_league_model_path: Path to no-league-trained model (.zip)
        num_episodes: Number of episodes per model (default: 10)
        opponent: Opponent to test against (default: "OP3")
        headless: Run without display (faster)
        episode_seeds: If provided, use these seeds for both models (same scenarios).
                      If None, a fixed list [42, 43, ...] is used so both still see the same seeds.
    
    Returns:
        Dict with win rates for both models
    """
    if episode_seeds is None:
        episode_seeds = [42 + i for i in range(num_episodes)]
    elif len(episode_seeds) < num_episodes:
        episode_seeds = list(episode_seeds) + [42 + len(episode_seeds) + i for i in range(num_episodes - len(episode_seeds))]
    else:
        episode_seeds = list(episode_seeds)[:num_episodes]
    results = {}
    
    # Test League Model
    print("=" * 60)
    print("TESTING LEAGUE MODEL")
    print("=" * 60)
    print(f"Model: {league_model_path}")
    print(f"Episodes: {num_episodes} (fixed seeds for both models = same scenarios)")
    print(f"Opponent: {opponent}")
    print(f"Seeds: {episode_seeds[0]}..{episode_seeds[-1]} (use --seed to change scenario set)")
    print()
    
    try:
        viewer_league = CTFViewer(
            ppo_model_path=league_model_path,
            viewer_use_obs_builder=True,
        )
        
        if not viewer_league.blue_ppo_team.model_loaded:
            print(f"[ERROR] Failed to load league model: {league_model_path}")
            results["league"] = {"win_rate": 0.0, "error": "Model failed to load"}
        else:
            summary_league = viewer_league.evaluate_model(
                num_episodes=num_episodes,
                headless=headless,
                opponent=opponent,
                eval_model="ppo",
                episode_seeds=episode_seeds,
            )
            results["league"] = {
                "win_rate": summary_league.get("win_rate", 0.0),
                "wins": summary_league.get("wins", 0),
                "losses": summary_league.get("losses", 0),
                "draws": summary_league.get("draws", 0),
                "mean_time_to_first_score": summary_league.get("mean_time_to_first_score"),
                "collision_free_rate": summary_league.get("collision_free_rate"),
            }
            print(f"\n[League] Win Rate: {results['league']['win_rate']:.2%}")
    except Exception as e:
        print(f"[ERROR] League model test failed: {e}")
        import traceback
        traceback.print_exc()
        results["league"] = {"win_rate": 0.0, "error": str(e)}
    
    print("\n" + "=" * 60)
    print("TESTING NO-LEAGUE MODEL")
    print("=" * 60)
    print(f"Model: {no_league_model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Opponent: {opponent}")
    print()
    
    # Test No-League Model
    try:
        viewer_noleague = CTFViewer(
            ppo_model_path=no_league_model_path,
            viewer_use_obs_builder=True,
        )
        
        if not viewer_noleague.blue_ppo_team.model_loaded:
            print(f"[ERROR] Failed to load no-league model: {no_league_model_path}")
            results["no_league"] = {"win_rate": 0.0, "error": "Model failed to load"}
        else:
            summary_noleague = viewer_noleague.evaluate_model(
                num_episodes=num_episodes,
                headless=headless,
                opponent=opponent,
                eval_model="ppo",
                episode_seeds=episode_seeds,
            )
            results["no_league"] = {
                "win_rate": summary_noleague.get("win_rate", 0.0),
                "wins": summary_noleague.get("wins", 0),
                "losses": summary_noleague.get("losses", 0),
                "draws": summary_noleague.get("draws", 0),
                "mean_time_to_first_score": summary_noleague.get("mean_time_to_first_score"),
                "collision_free_rate": summary_noleague.get("collision_free_rate"),
            }
            print(f"\n[No-League] Win Rate: {results['no_league']['win_rate']:.2%}")
    except Exception as e:
        print(f"[ERROR] No-league model test failed: {e}")
        import traceback
        traceback.print_exc()
        results["no_league"] = {"win_rate": 0.0, "error": str(e)}
    
    # Print Comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    if "error" not in results.get("league", {}):
        league_wr = results["league"]["win_rate"]
        league_w = results["league"]["wins"]
        league_l = results["league"]["losses"]
        league_d = results["league"]["draws"]
        print(f"League Model:")
        print(f"  Win Rate: {league_wr:.2%} ({league_w}W/{league_l}L/{league_d}D)")
        tfs = results["league"].get("mean_time_to_first_score")
        if tfs is not None:
            print(f"  Mean Time to First Score: {tfs:.2f}s")
        cfr = results["league"].get("collision_free_rate")
        if cfr is not None:
            print(f"  Collision-Free Rate: {cfr:.0%}")
    else:
        print(f"League Model: ERROR - {results['league'].get('error', 'Unknown')}")
    
    if "error" not in results.get("no_league", {}):
        no_league_wr = results["no_league"]["win_rate"]
        no_league_w = results["no_league"]["wins"]
        no_league_l = results["no_league"]["losses"]
        no_league_d = results["no_league"]["draws"]
        print(f"No-League Model:")
        print(f"  Win Rate: {no_league_wr:.2%} ({no_league_w}W/{no_league_l}L/{no_league_d}D)")
        tfs = results["no_league"].get("mean_time_to_first_score")
        if tfs is not None:
            print(f"  Mean Time to First Score: {tfs:.2f}s")
        cfr = results["no_league"].get("collision_free_rate")
        if cfr is not None:
            print(f"  Collision-Free Rate: {cfr:.0%}")
    else:
        print(f"No-League Model: ERROR - {results['no_league'].get('error', 'Unknown')}")
    
    if "error" not in results.get("league", {}) and "error" not in results.get("no_league", {}):
        diff = results["league"]["win_rate"] - results["no_league"]["win_rate"]
        print(f"\nDifference: {diff:+.2%} (League - No-League)")
        if diff > 0:
            print(f"✅ League model performs {abs(diff):.2%} better")
        elif diff < 0:
            print(f"❌ No-League model performs {abs(diff):.2%} better")
        else:
            print(f"⚖️  Models perform equally")
    
    print("=" * 60)
    
    return results


def run_league_vs_noleague_red(
    league_model_path: str,
    no_league_model_path: str,
    num_episodes: int,
    headless: bool = True,
    episode_seeds: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Test League (blue) vs No-League (red). No-league is loaded as the red opponent.
    Same seeds so the comparison is fair. Returns league's win rate (blue wins).
    """
    if episode_seeds is None:
        episode_seeds = [42 + i for i in range(num_episodes)]
    else:
        episode_seeds = list(episode_seeds)[:num_episodes]
    out = {}
    try:
        viewer = CTFViewer(ppo_model_path=league_model_path, viewer_use_obs_builder=True)
        if not getattr(viewer.blue_ppo_team, "model_loaded", False):
            out["error"] = "League model failed to load"
            return out
        viewer._apply_blue_mode("PPO")
        summary = viewer.evaluate_model(
            num_episodes=num_episodes,
            headless=headless,
            red_model_path=no_league_model_path,
            episode_seeds=episode_seeds,
        )
        if not summary:
            out["win_rate"] = 0.0
            out["wins"] = out["losses"] = out["draws"] = 0
            return out
        out["win_rate"] = summary.get("win_rate", 0.0)
        out["wins"] = summary.get("wins", 0)
        out["losses"] = summary.get("losses", 0)
        out["draws"] = summary.get("draws", 0)
        out["mean_time_to_first_score"] = summary.get("mean_time_to_first_score")
        out["collision_free_rate"] = summary.get("collision_free_rate")
    except Exception as e:
        out["error"] = str(e)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Compare League vs No-League training: 10 random test cases"
    )
    parser.add_argument(
        "--league-model",
        type=str,
        required=True,
        help="Path to league-trained model (.zip)"
    )
    parser.add_argument(
        "--no-league-model",
        type=str,
        required=True,
        help="Path to no-league-trained model (.zip)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Number of episodes per model (default: 30; use 30–50 for more stable WR)"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="OP3",
        choices=["OP1", "OP2", "OP3", "INTERCEPTOR", "MINELAYER", "CAMPER_ROTATE", "BAIT_SWITCH"],
        help="Opponent to test against (default: OP3). Use sharpened names for held-out tough test."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display (faster)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for episode_seeds (default: 42); same N seeds used for both models"
    )
    parser.add_argument(
        "--sharpened",
        action="store_true",
        help="Run comparison vs all four sharpened opponents (INTERCEPTOR, MINELAYER, CAMPER_ROTATE, BAIT_SWITCH); same seeds per opponent"
    )
    parser.add_argument(
        "--league-vs-noleague-red",
        action="store_true",
        help="Also run League (blue) vs No-League (red): no-league as learned red opponent; report league WR"
    )
    
    args = parser.parse_args()
    
    # Validate model paths
    if not os.path.exists(args.league_model):
        print(f"[ERROR] League model not found: {args.league_model}")
        sys.exit(1)
    if not os.path.exists(args.no_league_model):
        print(f"[ERROR] No-league model not found: {args.no_league_model}")
        sys.exit(1)
    
    SHARPENED_OPPONENTS = ("INTERCEPTOR", "MINELAYER", "CAMPER_ROTATE", "BAIT_SWITCH")
    
    if getattr(args, "sharpened", False):
        # Test vs all four sharpened opponents (same seeds for both models per opponent)
        print("\n" + "=" * 60)
        print("COMPARISON VS SHARPENED OPPONENTS (held-out, physics on)")
        print("=" * 60)
        all_results = {}
        for opp in SHARPENED_OPPONENTS:
            offset = 1000 * (SHARPENED_OPPONENTS.index(opp) + 1)
            episode_seeds = [args.seed + offset + i for i in range(args.episodes)]
            all_results[opp] = run_comparison_test(
                league_model_path=args.league_model,
                no_league_model_path=args.no_league_model,
                num_episodes=args.episodes,
                opponent=opp,
                headless=args.headless,
                episode_seeds=episode_seeds,
            )
        # Summary table
        print("\n" + "=" * 60)
        print("LEAGUE vs NO-LEAGUE vs SHARPENED OPPONENTS (same setting)")
        print("=" * 60)
        print(f"{'Opponent':<20} {'League WR':>10} {'No-League WR':>12} {'Diff (L-NL)':>12}")
        print("-" * 60)
        for opp in SHARPENED_OPPONENTS:
            r = all_results.get(opp, {})
            lw = r.get("league", {}).get("win_rate", 0.0)
            nw = r.get("no_league", {}).get("win_rate", 0.0)
            diff = lw - nw if "error" not in r.get("league", {}) and "error" not in r.get("no_league", {}) else float("nan")
            if "error" in r.get("league", {}):
                lw_s = "ERROR"
            else:
                lw_s = f"{lw:.0%}"
            if "error" in r.get("no_league", {}):
                nw_s = "ERROR"
            else:
                nw_s = f"{nw:.0%}"
            diff_s = f"{diff:+.0%}" if not (diff != diff) else "n/a"
            print(f"{opp:<20} {lw_s:>10} {nw_s:>12} {diff_s:>12}")
        print("=" * 60)
        has_error = any(
            "error" in all_results.get(opp, {}).get("league", {}) or "error" in all_results.get(opp, {}).get("no_league", {})
            for opp in SHARPENED_OPPONENTS
        )
    else:
        # Single opponent (default or --opponent)
        print(f"[Compare] Using fixed seeds (base={args.seed}) so both models see the same {args.episodes} scenarios.")
        print("[Compare] League may show lower WR than 'test alone' because that run uses different random scenarios.\n")
        episode_seeds = [args.seed + i for i in range(args.episodes)]
        results = run_comparison_test(
            league_model_path=args.league_model,
            no_league_model_path=args.no_league_model,
            num_episodes=args.episodes,
            opponent=args.opponent,
            headless=args.headless,
            episode_seeds=episode_seeds,
        )
        has_error = "error" in results.get("league", {}) or "error" in results.get("no_league", {})

    # League (blue) vs No-League (red): does league beat no-league when no-league is the opponent?
    if getattr(args, "league_vs_noleague_red", False):
        print("\n" + "=" * 60)
        print("LEAGUE (blue) vs NO-LEAGUE (red) — no-league as learned red")
        print("=" * 60)
        seeds_cross = [args.seed + 5000 + i for i in range(args.episodes)]
        cross = run_league_vs_noleague_red(
            league_model_path=args.league_model,
            no_league_model_path=args.no_league_model,
            num_episodes=args.episodes,
            headless=args.headless,
            episode_seeds=seeds_cross,
        )
        if "error" in cross:
            print(f"[ERROR] {cross['error']}")
            has_error = True
        else:
            wr = cross.get("win_rate", 0.0)
            w, l, d = cross.get("wins", 0), cross.get("losses", 0), cross.get("draws", 0)
            print(f"League (blue) vs No-League (red): League WR = {wr:.0%}  (W/L/D: {w}/{l}/{d})")
            if cross.get("mean_time_to_first_score") is not None:
                print(f"  Mean time to first score (blue): {cross['mean_time_to_first_score']:.1f}s")
            if cross.get("collision_free_rate") is not None:
                print(f"  Collision-free rate: {cross['collision_free_rate']:.0%}")
        print("=" * 60)
    
    if has_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
