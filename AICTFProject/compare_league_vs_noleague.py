"""
Compare League vs No-League Training: 10 Random Test Cases

This script runs 10 random test episodes for both league and no-league models
to compare win rates against OP3 (standard test opponent).

Uses different random seeds for each episode to ensure diverse test cases
while maintaining reproducibility (same seeds for both models).
"""
import argparse
import os
import random
import sys
from typing import Dict, Optional

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
) -> Dict[str, Dict[str, float]]:
    """
    Run comparison test: 10 random episodes for each model.
    
    Args:
        league_model_path: Path to league-trained model (.zip)
        no_league_model_path: Path to no-league-trained model (.zip)
        num_episodes: Number of episodes per model (default: 10)
        opponent: Opponent to test against (default: "OP3")
        headless: Run without display (faster)
    
    Returns:
        Dict with win rates for both models
    """
    results = {}
    
    # Test League Model
    print("=" * 60)
    print("TESTING LEAGUE MODEL")
    print("=" * 60)
    print(f"Model: {league_model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Opponent: {opponent}")
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
            )
            results["league"] = {
                "win_rate": summary_league.get("win_rate", 0.0),
                "wins": summary_league.get("wins", 0),
                "losses": summary_league.get("losses", 0),
                "draws": summary_league.get("draws", 0),
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
            )
            results["no_league"] = {
                "win_rate": summary_noleague.get("win_rate", 0.0),
                "wins": summary_noleague.get("wins", 0),
                "losses": summary_noleague.get("losses", 0),
                "draws": summary_noleague.get("draws", 0),
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
    else:
        print(f"League Model: ERROR - {results['league'].get('error', 'Unknown')}")
    
    if "error" not in results.get("no_league", {}):
        no_league_wr = results["no_league"]["win_rate"]
        no_league_w = results["no_league"]["wins"]
        no_league_l = results["no_league"]["losses"]
        no_league_d = results["no_league"]["draws"]
        print(f"No-League Model:")
        print(f"  Win Rate: {no_league_wr:.2%} ({no_league_w}W/{no_league_l}L/{no_league_d}D)")
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
        default=10,
        help="Number of episodes per model (default: 10)"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="OP3",
        choices=["OP1", "OP2", "OP3", "OP3_EASY", "OP3_HARD"],
        help="Opponent to test against (default: OP3)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display (faster)"
    )
    
    args = parser.parse_args()
    
    # Validate model paths
    if not os.path.exists(args.league_model):
        print(f"[ERROR] League model not found: {args.league_model}")
        sys.exit(1)
    if not os.path.exists(args.no_league_model):
        print(f"[ERROR] No-league model not found: {args.no_league_model}")
        sys.exit(1)
    
    # Run comparison
    results = run_comparison_test(
        league_model_path=args.league_model,
        no_league_model_path=args.no_league_model,
        num_episodes=args.episodes,
        opponent=args.opponent,
        headless=args.headless,
    )
    
    # Exit with error if either model failed
    if "error" in results.get("league", {}) or "error" in results.get("no_league", {}):
        sys.exit(1)


if __name__ == "__main__":
    main()
