#!/usr/bin/env python3
"""Stage C gate evaluation for V6I8 checkpoints.

Answers two questions per checkpoint:
  1. Does oracle-z beat best-fixed-z?   (R_oracle > R_best_fixed)
  2. Does the best latent vary across map × opponent cells?

Oracle is computed per-episode across matched seeds:
  oracle_wr     = mean_i( max_z success[i, z] )
  oracle_margin = mean_i( max_z win_margin[i, z] )

All latents for the same (opponent, map) cell share the same env seed so the
initial-state distribution is matched.  See run_forced_z_cells for details.

Usage
-----
    python experiments/eval_v6i8_stage_c.py \\
        --checkpoints \\
            checkpoints\\2v2\\ckpt_..._250000.zip \\
            checkpoints\\2v2\\ckpt_..._500000.zip \\
            checkpoints\\2v2\\ckpt_..._750000.zip \\
        --episodes 100 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.calibrate_hard_pool import (  # noqa: E402
    LATENTS,
    OPPONENTS,
    MAPS,
    CellEpisodes,
    run_forced_z_cells,
    cells_to_mean_wr_matrix,
    _wr,
    _mean_margin,
)

# ── per-episode oracle ────────────────────────────────────────────────────────

def _oracle_per_episode(cells: CellEpisodes, opponents: List[str], maps: List[str]) -> Tuple[float, float]:
    """Compute oracle WR and oracle score margin using per-episode max across z.

    For each (opponent, map, episode_index), take the best outcome across latents.
    This is valid because all z cells share the same env seed → same episode sequence.
    """
    all_ep_win: List[float] = []
    all_ep_margin: List[float] = []

    for opponent in opponents:
        for map_name in maps:
            # Align episode lists: shortest length is the trusted common count.
            ep_lists = [cells.get((opponent, z, map_name), []) for z in LATENTS]
            n = min(len(eps) for eps in ep_lists)
            if n == 0:
                continue
            for i in range(n):
                best_win = max(int(ep_lists[z][i].get("success", 0)) for z in range(len(LATENTS)))
                best_margin = max(int(ep_lists[z][i].get("win_margin", 0)) for z in range(len(LATENTS)))
                all_ep_win.append(float(best_win))
                all_ep_margin.append(float(best_margin))

    if not all_ep_win:
        return float("nan"), float("nan")
    return sum(all_ep_win) / len(all_ep_win), sum(all_ep_margin) / len(all_ep_margin)


# ── best-fixed comparison ─────────────────────────────────────────────────────

def _best_fixed(cells: CellEpisodes, opponents: List[str], maps: List[str]) -> Tuple[int, float, float]:
    """Find the single fixed z with the best mean WR across all (opponent, map) cells."""
    best_z, best_wr_val, best_margin_val = -1, -1.0, -999.0
    for z in LATENTS:
        wrs = [_wr(cells.get((opp, z, m), [])) for opp in opponents for m in maps]
        margins = [_mean_margin(cells.get((opp, z, m), [])) for opp in opponents for m in maps]
        valid_wr = [v for v in wrs if v == v]
        valid_mg = [v for v in margins if v == v]
        mean_wr = sum(valid_wr) / len(valid_wr) if valid_wr else float("nan")
        if mean_wr > best_wr_val:
            best_z = z
            best_wr_val = mean_wr
            best_margin_val = sum(valid_mg) / len(valid_mg) if valid_mg else float("nan")
    return best_z, best_wr_val, best_margin_val


# ── gate ──────────────────────────────────────────────────────────────────────

def _best_z_per_cell(cells: CellEpisodes, opponents: List[str], maps: List[str]) -> Dict[Tuple[str, str], int]:
    return {
        (opp, m): max(LATENTS, key=lambda z: _wr(cells.get((opp, z, m), [])))
        for opp in opponents
        for m in maps
    }


def stage_c_gate(cells: CellEpisodes, opponents: List[str], maps: List[str]) -> bool:
    oracle_wr_val, oracle_mg = _oracle_per_episode(cells, opponents, maps)
    fixed_z, fixed_wr_val, fixed_mg = _best_fixed(cells, opponents, maps)
    best_per_cell = _best_z_per_cell(cells, opponents, maps)
    unique_best = set(best_per_cell.values())

    print(f"  Oracle-z   WR={oracle_wr_val:.1%}  margin={oracle_mg:+.2f}")
    print(f"  Best-fixed WR={fixed_wr_val:.1%}  margin={fixed_mg:+.2f}  (z={fixed_z})")
    print(f"  WR advantage   : {oracle_wr_val - fixed_wr_val:+.1%}")
    print(f"  Margin advantage: {oracle_mg - fixed_mg:+.2f}")
    print(f"  Best z per cell : {best_per_cell}")
    print(f"  Unique best-z   : {sorted(unique_best)} ({len(unique_best)} of {len(opponents) * len(maps)} cells)")

    gate_advantage = oracle_wr_val > fixed_wr_val
    gate_diversity = len(unique_best) >= 2
    print(f"\n  Gate 1 (oracle > best-fixed WR): {'PASS' if gate_advantage else 'FAIL'}")
    print(f"  Gate 2 (best z varies across map×opp cells): {'PASS' if gate_diversity else 'FAIL'}")
    return gate_advantage and gate_diversity


# ── display ───────────────────────────────────────────────────────────────────

def _print_matrix(cells: CellEpisodes, opponents: List[str], maps: List[str], label: str) -> None:
    print(f"\n--- {label} ---")
    for map_name in maps:
        print(f"  map={map_name}")
        print("  " + f"{'':12s}" + "".join(f"  z={z}" for z in LATENTS) + "  oracle")
        for opp in opponents:
            vals = [_wr(cells.get((opp, z, map_name), [])) for z in LATENTS]
            best = max((v for v in vals if v == v), default=float("nan"))
            row = f"  {opp:<12s}" + "".join(f"  {v:5.1%}" if v == v else "   nan" for v in vals)
            row += f"  {best:5.1%}" if best == best else "   nan"
            print(row)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I8 Stage C gate evaluation")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=100, help="Episodes per (opponent, latent, map) cell")
    p.add_argument("--device", default="cpu")
    p.add_argument("--opponents", nargs="+", default=list(OPPONENTS))
    p.add_argument("--maps", nargs="+", default=list(MAPS))
    p.add_argument("--stochastic", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        import plot.eval_rollout  # noqa: F401
    except ImportError as exc:
        print(f"ERROR: could not import eval infrastructure: {exc}")
        sys.exit(1)

    results: List[Tuple[str, bool]] = []

    for ckpt in args.checkpoints:
        label = os.path.basename(ckpt)
        print(f"\n{'='*60}")
        print(f"Checkpoint: {label}")
        print(f"{'='*60}")

        cells = run_forced_z_cells(
            checkpoint=ckpt,
            opponents=args.opponents,
            latents=tuple(LATENTS),
            maps=args.maps,
            n_episodes=args.episodes,
            device=args.device,
            deterministic=not args.stochastic,
        )
        _print_matrix(cells, args.opponents, args.maps, label)

        print("\n--- Stage C Gate ---")
        passed = stage_c_gate(cells, args.opponents, args.maps)
        results.append((label, passed))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for label, passed in results:
        status = "PASS — promote to Stage D" if passed else "FAIL — do not promote"
        print(f"  {label}: {status}")


if __name__ == "__main__":
    main()
