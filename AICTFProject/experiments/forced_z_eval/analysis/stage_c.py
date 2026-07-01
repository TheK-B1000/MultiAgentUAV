"""Stage C gate analysis over canonical forced-z episodes."""
from __future__ import annotations

from typing import Any

from experiments.forced_z_eval.io import CellEpisodes


def _wr(eps: list[dict[str, Any]]) -> float:
    if not eps:
        return float("nan")
    return sum(int(e.get("success", 0)) for e in eps) / len(eps)


def _mean_margin(eps: list[dict[str, Any]]) -> float:
    if not eps:
        return float("nan")
    return sum(int(e.get("win_margin", 0)) for e in eps) / len(eps)


def oracle_per_episode(
    cells: CellEpisodes,
    opponents: list[str],
    maps: list[str],
    latents: tuple[int, ...],
) -> tuple[float, float]:
    all_ep_win: list[float] = []
    all_ep_margin: list[float] = []
    for opponent in opponents:
        for map_name in maps:
            ep_lists = [cells.get((opponent, z, map_name), []) for z in latents]
            n = min(len(eps) for eps in ep_lists)
            if n == 0:
                continue
            for i in range(n):
                all_ep_win.append(float(max(int(ep_lists[z][i].get("success", 0)) for z in latents)))
                all_ep_margin.append(float(max(int(ep_lists[z][i].get("win_margin", 0)) for z in latents)))
    if not all_ep_win:
        return float("nan"), float("nan")
    return sum(all_ep_win) / len(all_ep_win), sum(all_ep_margin) / len(all_ep_margin)


def best_fixed(
    cells: CellEpisodes,
    opponents: list[str],
    maps: list[str],
    latents: tuple[int, ...],
) -> tuple[int, float, float]:
    best_z, best_wr_val, best_margin_val = -1, -1.0, -999.0
    for z in latents:
        wrs = [_wr(cells.get((opp, z, m), [])) for opp in opponents for m in maps]
        margins = [_mean_margin(cells.get((opp, z, m), [])) for opp in opponents for m in maps]
        valid_wr = [v for v in wrs if v == v]
        valid_mg = [v for v in margins if v == v]
        mean_wr = sum(valid_wr) / len(valid_wr) if valid_wr else float("nan")
        if mean_wr > best_wr_val:
            best_z = int(z)
            best_wr_val = mean_wr
            best_margin_val = sum(valid_mg) / len(valid_mg) if valid_mg else float("nan")
    return best_z, best_wr_val, best_margin_val


def best_z_per_cell(
    cells: CellEpisodes,
    opponents: list[str],
    maps: list[str],
    latents: tuple[int, ...],
) -> dict[tuple[str, str], int]:
    return {
        (opp, m): max(latents, key=lambda z: _wr(cells.get((opp, z, m), [])))
        for opp in opponents
        for m in maps
    }


def build_stage_c_report(
    cells: CellEpisodes,
    *,
    opponents: list[str],
    maps: list[str],
    latents: tuple[int, ...],
) -> dict[str, Any]:
    oracle_wr_val, oracle_mg = oracle_per_episode(cells, opponents, maps, latents)
    fixed_z, fixed_wr_val, fixed_mg = best_fixed(cells, opponents, maps, latents)
    best_per_cell = best_z_per_cell(cells, opponents, maps, latents)
    unique_best = set(best_per_cell.values())
    gate_advantage = bool(oracle_wr_val > fixed_wr_val)
    gate_diversity = len(unique_best) >= 2
    wr_matrix = {
        f"{opp}|{m}|z{z}": _wr(cells.get((opp, z, m), []))
        for opp in opponents
        for m in maps
        for z in latents
    }
    return {
        "oracle_wr": oracle_wr_val,
        "oracle_margin": oracle_mg,
        "best_fixed_z": int(fixed_z),
        "best_fixed_wr": fixed_wr_val,
        "best_fixed_margin": fixed_mg,
        "wr_advantage": float(oracle_wr_val - fixed_wr_val) if oracle_wr_val == oracle_wr_val else float("nan"),
        "margin_advantage": float(oracle_mg - fixed_mg) if oracle_mg == oracle_mg else float("nan"),
        "best_z_per_cell": {f"{opp}|{m}": z for (opp, m), z in best_per_cell.items()},
        "unique_best_z_count": len(unique_best),
        "unique_best_z_values": sorted(unique_best),
        "gate_oracle_beats_fixed_wr": gate_advantage,
        "gate_best_z_varies": gate_diversity,
        "passed": bool(gate_advantage and gate_diversity),
        "wr_matrix": wr_matrix,
    }


def print_stage_c_report(report: dict[str, Any]) -> None:
    print(f"  Oracle-z   WR={report['oracle_wr']:.1%}  margin={report['oracle_margin']:+.2f}")
    print(
        f"  Best-fixed WR={report['best_fixed_wr']:.1%}  margin={report['best_fixed_margin']:+.2f}  "
        f"(z={report['best_fixed_z']})"
    )
    print(f"  WR advantage   : {report['wr_advantage']:+.1%}")
    print(f"  Margin advantage: {report['margin_advantage']:+.2f}")
    print(f"  Best z per cell : {report['best_z_per_cell']}")
    print(
        f"  Unique best-z   : {report['unique_best_z_values']} "
        f"({report['unique_best_z_count']} cells)"
    )
    print(f"\n  Gate 1 (oracle > best-fixed WR): {'PASS' if report['gate_oracle_beats_fixed_wr'] else 'FAIL'}")
    print(f"  Gate 2 (best z varies across map×opp cells): {'PASS' if report['gate_best_z_varies'] else 'FAIL'}")


__all__ = [
    "best_fixed",
    "best_z_per_cell",
    "build_stage_c_report",
    "oracle_per_episode",
    "print_stage_c_report",
]
