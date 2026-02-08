"""
Fast experiment: run 4 eval suites for the league model (and optionally no-league for comparison).

Suite 1: vs OP3 only (scripted OP3).
Suite 2: vs species only (BALANCED, RUSHER, CAMPER) — same seeds per species.
Suite 3: vs snapshot pool — sample N snapshots, run episodes vs each (round-robin seeds).
Suite 4: vs naval-realistic opponents (NAVAL_DEFENDER, NAVAL_RUSHER, NAVAL_BALANCED) — held-out, OP3 physics.

Interpretation: If league beats no-league on (2), (3), and (4) but loses on (1), league is doing its job
and you may only need OP3 anchoring if the OP3 baseline matters.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)


# League and no-league model paths (league = primary; no-league = optional comparison)
LEAGUE_MODEL_PATH = "rl/checkpoints_sb3/final_ppo_league_curriculum_v2.zip"
NO_LEAGUE_MODEL_PATH = "checkpoints_sb3/final_ppo_noleague.zip"

SPECIES_TAGS = ("BALANCED", "RUSHER", "CAMPER")
# Naval-realistic held-out opponents (OP3 stress; not used in training)
NAVAL_OPPONENTS = ("NAVAL_DEFENDER", "NAVAL_RUSHER", "NAVAL_BALANCED")


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_SCRIPT_DIR, path)


def _find_snapshot_zips(snapshot_dirs: List[str], glob_pattern: str = "*snapshot*.zip") -> List[str]:
    """Collect full paths of snapshot .zip files from given directories."""
    out: List[str] = []
    for d in snapshot_dirs:
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, glob_pattern)):
            if p.endswith(".zip") and os.path.isfile(p):
                out.append(os.path.abspath(p))
    return sorted(out)


def run_suite_op3(
    viewer: Any,
    num_episodes: int,
    episode_seeds: List[int],
    headless: bool,
) -> Dict[str, Any]:
    """Suite 1: vs OP3 only."""
    summary = viewer.evaluate_model(
        num_episodes=num_episodes,
        headless=headless,
        opponent="OP3",
        eval_model="ppo",
        episode_seeds=episode_seeds,
    )
    return {
        "win_rate": summary.get("win_rate", 0.0),
        "wins": summary.get("wins", 0),
        "losses": summary.get("losses", 0),
        "draws": summary.get("draws", 0),
        "mean_time_to_first_score": summary.get("mean_time_to_first_score"),
        "collision_free_rate": summary.get("collision_free_rate", 0.0),
    }


def run_suite_species(
    viewer: Any,
    num_episodes_per_species: int,
    seed_base: int,
    headless: bool,
) -> Dict[str, Dict[str, Any]]:
    """Suite 2: vs species only (BALANCED, RUSHER, CAMPER). Same total seeds used per species."""
    results = {}
    for tag in SPECIES_TAGS:
        seeds = [seed_base + i for i in range(num_episodes_per_species)]
        summary = viewer.evaluate_model(
            num_episodes=num_episodes_per_species,
            headless=headless,
            opponent="OP3",
            red_species_tag=tag,
            eval_model="ppo",
            episode_seeds=seeds,
        )
        results[tag] = {
            "win_rate": summary.get("win_rate", 0.0),
            "wins": summary.get("wins", 0),
            "losses": summary.get("losses", 0),
            "draws": summary.get("draws", 0),
            "mean_time_to_first_score": summary.get("mean_time_to_first_score"),
            "collision_free_rate": summary.get("collision_free_rate", 0.0),
        }
    return results


def run_suite_naval(
    viewer: Any,
    num_episodes_per_opponent: int,
    seed_base: int,
    headless: bool,
) -> Dict[str, Dict[str, Any]]:
    """Suite 4: vs naval-realistic opponents (NAVAL_DEFENDER, NAVAL_RUSHER, NAVAL_BALANCED). Same seeds per opponent."""
    results = {}
    for tag in NAVAL_OPPONENTS:
        seeds = [seed_base + i for i in range(num_episodes_per_opponent)]
        summary = viewer.evaluate_model(
            num_episodes=num_episodes_per_opponent,
            headless=headless,
            opponent=tag,
            eval_model="ppo",
            episode_seeds=seeds,
        )
        results[tag] = {
            "win_rate": summary.get("win_rate", 0.0),
            "wins": summary.get("wins", 0),
            "losses": summary.get("losses", 0),
            "draws": summary.get("draws", 0),
            "mean_time_to_first_score": summary.get("mean_time_to_first_score"),
            "collision_free_rate": summary.get("collision_free_rate", 0.0),
        }
    return results


def run_suite_snapshots(
    viewer: Any,
    snapshot_paths: List[str],
    num_episodes_total: int,
    seed_base: int,
    headless: bool,
) -> Dict[str, Any]:
    """Suite 3: vs snapshot pool. Distribute episodes across snapshots (round-robin)."""
    if not snapshot_paths:
        return {
            "win_rate": 0.0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "mean_time_to_first_score": None,
            "collision_free_rate": 0.0,
            "error": "No snapshot paths provided",
        }
    # Round-robin: episode i uses snapshot_paths[i % len(snapshot_paths)]
    all_wins, all_losses, all_draws = 0, 0, 0
    time_to_first_list: List[float] = []
    collision_free_count = 0
    seeds = [seed_base + i for i in range(num_episodes_total)]
    for ep_idx in range(num_episodes_total):
        snap_path = snapshot_paths[ep_idx % len(snapshot_paths)]
        try:
            summary = viewer.evaluate_model(
                num_episodes=1,
                headless=headless,
                opponent="OP3",
                red_model_path=snap_path,
                eval_model="ppo",
                episode_seeds=[seeds[ep_idx]],
            )
            all_wins += summary.get("wins", 0)
            all_losses += summary.get("losses", 0)
            all_draws += summary.get("draws", 0)
            t = summary.get("mean_time_to_first_score")
            if t is not None:
                time_to_first_list.append(float(t))
            if summary.get("collision_free_rate", 0) > 0.5:
                collision_free_count += 1
        except Exception as e:
            print(f"[WARN] Snapshot {snap_path} failed: {e}")
    n = num_episodes_total
    wr = (all_wins / n) if n else 0.0
    return {
        "win_rate": wr,
        "wins": all_wins,
        "losses": all_losses,
        "draws": all_draws,
        "mean_time_to_first_score": float(sum(time_to_first_list) / len(time_to_first_list)) if time_to_first_list else None,
        "collision_free_rate": collision_free_count / n if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 4 eval suites for league (and optionally no-league): OP3, species, snapshot pool, naval-realistic."
    )
    parser.add_argument("--league-model", type=str, default=LEAGUE_MODEL_PATH, help="League model .zip path")
    parser.add_argument("--no-league-model", type=str, default=NO_LEAGUE_MODEL_PATH, help="No-league model .zip (for comparison)")
    parser.add_argument("--compare-no-league", action="store_true", help="Also run all 3 suites for no-league and print comparison")
    parser.add_argument("--episodes-op3", type=int, default=20, help="Episodes for suite 1 (OP3 only)")
    parser.add_argument("--episodes-per-species", type=int, default=10, help="Episodes per species for suite 2 (total 30)")
    parser.add_argument("--episodes-per-naval", type=int, default=10, help="Episodes per naval opponent for suite 4 (total 30)")
    parser.add_argument("--episodes-snapshots", type=int, default=20, help="Total episodes for suite 3 (distributed across snapshots)")
    parser.add_argument("--snapshot-dirs", type=str, nargs="+", default=None,
                        help="Dirs to search for snapshot zips (default: rl/checkpoints_sb3, checkpoints_sb3)")
    parser.add_argument("--snapshot-glob", type=str, default="*snapshot*.zip",
                        help="Glob for snapshot files (default: *snapshot*.zip)")
    parser.add_argument("--max-snapshots", type=int, default=5, help="Max number of snapshots to use in suite 3 (sample if more found)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed for episode seeds (default: random per run for varied results)")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    args = parser.parse_args()

    # Different results each run: use random base seed unless --seed is set
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
        print(f"[Seed] Using random base seed: {args.seed} (use --seed {args.seed} to reproduce this run)")

    league_path = _resolve_path(args.league_model)
    no_league_path = _resolve_path(args.no_league_model)
    if not os.path.exists(league_path):
        print(f"[ERROR] League model not found: {league_path}")
        sys.exit(1)

    snapshot_dirs = args.snapshot_dirs
    if not snapshot_dirs:
        snapshot_dirs = [
            os.path.join(_SCRIPT_DIR, "rl", "checkpoints_sb3"),
            os.path.join(_SCRIPT_DIR, "checkpoints_sb3"),
        ]
    snapshot_dirs = [os.path.abspath(d) for d in snapshot_dirs]
    all_snapshots = _find_snapshot_zips(snapshot_dirs, args.snapshot_glob)
    # Prefer league snapshots over self-play for this experiment; then sample
    league_snaps = [p for p in all_snapshots if "league_snapshot" in os.path.basename(p)]
    other_snaps = [p for p in all_snapshots if p not in league_snaps]
    pool = (league_snaps + other_snaps)[: args.max_snapshots]
    if not pool and all_snapshots:
        random.Random(args.seed).shuffle(all_snapshots)
        pool = all_snapshots[: args.max_snapshots]
    print(f"[Snapshots] Using {len(pool)} snapshots for suite 3: {[os.path.basename(p) for p in pool]}")

    from ctfviewer import CTFViewer

    def run_all_suites(ppo_path: str, label: str) -> Dict[str, Any]:
        viewer = CTFViewer(ppo_model_path=ppo_path, viewer_use_obs_builder=True)
        if not viewer.blue_ppo_team.model_loaded:
            resolved = getattr(viewer.blue_ppo_team, "model_path", None) or ppo_path
            print(f"[ERROR] {label}: failed to load model {resolved}")
            print("        Check the traceback above (policy/obs mismatch or corrupted zip).")
            print("        Use --no-league-model <path> to try another checkpoint, or run without --compare-no-league.")
            return {"suite1_op3": None, "suite2_species": None, "suite3_snapshots": None, "suite4_naval": None}
        seed_base = args.seed
        seeds_op3 = [seed_base + i for i in range(args.episodes_op3)]
        # Suite 1
        print(f"\n[{label}] Suite 1: vs OP3 only ({args.episodes_op3} ep)")
        s1 = run_suite_op3(viewer, args.episodes_op3, seeds_op3, args.headless)
        # Suite 2
        print(f"\n[{label}] Suite 2: vs species only ({args.episodes_per_species} ep per species)")
        s2 = run_suite_species(viewer, args.episodes_per_species, seed_base + 1000, args.headless)
        # Suite 3
        print(f"\n[{label}] Suite 3: vs snapshot pool ({args.episodes_snapshots} ep over {len(pool)} snapshots)")
        s3 = run_suite_snapshots(viewer, pool, args.episodes_snapshots, seed_base + 2000, args.headless) if pool else {"win_rate": 0.0, "wins": 0, "losses": 0, "draws": 0, "mean_time_to_first_score": None, "collision_free_rate": 0.0, "error": "No snapshots"}
        # Suite 4: naval-realistic opponents
        print(f"\n[{label}] Suite 4: vs naval-realistic ({args.episodes_per_naval} ep per opponent: {', '.join(NAVAL_OPPONENTS)})")
        s4 = run_suite_naval(viewer, args.episodes_per_naval, seed_base + 3000, args.headless)
        return {"suite1_op3": s1, "suite2_species": s2, "suite3_snapshots": s3, "suite4_naval": s4}

    print("=" * 60)
    print("LEAGUE MODEL – 4 eval suites (OP3, species, snapshots, naval)")
    print("=" * 60)
    league_results = run_all_suites(league_path, "League")

    if args.compare_no_league:
        if os.path.exists(no_league_path):
            print("\n" + "=" * 60)
            print("NO-LEAGUE MODEL – 4 eval suites (comparison)")
            print("=" * 60)
            no_league_results = run_all_suites(no_league_path, "No-League")
            if no_league_results and all(no_league_results.get(k) is None for k in ("suite1_op3", "suite2_species", "suite3_snapshots", "suite4_naval")):
                no_league_results = None  # load failed; treat as no comparison
        else:
            print(f"[WARN] No-league model not found at: {no_league_path}")
            print("       Skipping no-league comparison. Use --no-league-model <path> to point to another checkpoint.")
            no_league_results = None
    else:
        no_league_results = None

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for suite_name, key in [
        ("Suite 1: vs OP3 only", "suite1_op3"),
        ("Suite 2: vs species (avg)", "suite2_species"),
        ("Suite 3: vs snapshot pool", "suite3_snapshots"),
        ("Suite 4: vs naval (avg)", "suite4_naval"),
    ]:
        lr = league_results.get(key)
        if key == "suite2_species" and lr and isinstance(lr, dict):
            avg_wr = sum(lr[t]["win_rate"] for t in SPECIES_TAGS if t in lr) / 3.0 if lr else 0.0
            print(f"\n{suite_name}")
            print(f"  League:    WR={avg_wr:.2%} (avg over BALANCED/RUSHER/CAMPER)")
            for t in SPECIES_TAGS:
                if t in lr:
                    print(f"    {t}: WR={lr[t]['win_rate']:.2%}")
            if no_league_results and key in no_league_results and no_league_results[key]:
                nl = no_league_results[key]
                nl_avg = sum(nl[t]["win_rate"] for t in SPECIES_TAGS if t in nl) / 3.0 if isinstance(nl, dict) else nl.get("win_rate", 0)
                print(f"  No-League: WR={nl_avg:.2%} (avg)")
                print(f"  League beats No-League on species: {'Yes' if avg_wr > nl_avg else 'No'}")
        elif key == "suite4_naval" and lr and isinstance(lr, dict):
            avg_wr = sum(lr[t]["win_rate"] for t in NAVAL_OPPONENTS if t in lr) / 3.0 if lr else 0.0
            print(f"\n{suite_name}")
            print(f"  League:    WR={avg_wr:.2%} (avg over NAVAL_DEFENDER/RUSHER/BALANCED)")
            for t in NAVAL_OPPONENTS:
                if t in lr:
                    print(f"    {t}: WR={lr[t]['win_rate']:.2%}")
            if no_league_results and key in no_league_results and no_league_results[key]:
                nl = no_league_results[key]
                nl_avg = sum(nl[t]["win_rate"] for t in NAVAL_OPPONENTS if t in nl) / 3.0 if isinstance(nl, dict) else nl.get("win_rate", 0)
                print(f"  No-League: WR={nl_avg:.2%} (avg)")
                print(f"  League beats No-League on naval: {'Yes' if avg_wr > nl_avg else 'No'}")
        elif lr and isinstance(lr, dict) and "win_rate" in lr:
            print(f"\n{suite_name}")
            print(f"  League:    WR={lr['win_rate']:.2%} (W/L/D={lr.get('wins',0)}/{lr.get('losses',0)}/{lr.get('draws',0)})")
            if no_league_results and key in no_league_results:
                nl = no_league_results[key]
                if nl and isinstance(nl, dict) and "win_rate" in nl:
                    print(f"  No-League: WR={nl['win_rate']:.2%} (W/L/D={nl.get('wins',0)}/{nl.get('losses',0)}/{nl.get('draws',0)})")
                    print(f"  League beats No-League: {'Yes' if lr['win_rate'] > nl['win_rate'] else 'No'}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    s1 = league_results.get("suite1_op3") or {}
    s2 = league_results.get("suite2_species") or {}
    s3 = league_results.get("suite3_snapshots") or {}
    s4 = league_results.get("suite4_naval") or {}
    l_wr_op3 = s1.get("win_rate", 0.0) if isinstance(s1, dict) else 0.0
    l_wr_species = (sum(s2[t]["win_rate"] for t in SPECIES_TAGS if t in s2) / 3.0) if isinstance(s2, dict) and s2 else 0.0
    l_wr_snap = s3.get("win_rate", 0.0) if isinstance(s3, dict) else 0.0
    l_wr_naval = (sum(s4[t]["win_rate"] for t in NAVAL_OPPONENTS if t in s4) / 3.0) if isinstance(s4, dict) and s4 else 0.0
    if no_league_results:
        n1 = no_league_results.get("suite1_op3") or {}
        n2 = no_league_results.get("suite2_species") or {}
        n3 = no_league_results.get("suite3_snapshots") or {}
        n4 = no_league_results.get("suite4_naval") or {}
        n_wr_op3 = n1.get("win_rate", 0.0) if isinstance(n1, dict) else 0.0
        n_wr_species = (sum(n2[t]["win_rate"] for t in SPECIES_TAGS if t in n2) / 3.0) if isinstance(n2, dict) and n2 else 0.0
        n_wr_snap = n3.get("win_rate", 0.0) if isinstance(n3, dict) else 0.0
        n_wr_naval = (sum(n4[t]["win_rate"] for t in NAVAL_OPPONENTS if t in n4) / 3.0) if isinstance(n4, dict) and n4 else 0.0
        beat_species = l_wr_species > n_wr_species
        beat_snap = l_wr_snap > n_wr_snap
        beat_naval = l_wr_naval > n_wr_naval
        lose_op3 = l_wr_op3 < n_wr_op3
        if beat_species and beat_snap and beat_naval and lose_op3:
            print("League beats no-league on (2) species, (3) snapshots, and (4) naval but loses on (1) OP3.")
            print("=> League is doing its job; consider OP3 anchoring if OP3 baseline matters.")
        else:
            print(f"League vs No-League: OP3 {l_wr_op3:.0%} vs {n_wr_op3:.0%} | Species {l_wr_species:.0%} vs {n_wr_species:.0%} | Snapshots {l_wr_snap:.0%} vs {n_wr_snap:.0%} | Naval {l_wr_naval:.0%} vs {n_wr_naval:.0%}")
    else:
        print(f"League only: OP3 WR={l_wr_op3:.2%} | Species(avg) WR={l_wr_species:.2%} | Snapshots WR={l_wr_snap:.2%} | Naval(avg) WR={l_wr_naval:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
