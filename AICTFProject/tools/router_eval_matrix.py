"""Build the offline router-quality ledger from eval aggregate CSVs.

This is intentionally post-training only. It consumes aggregate CSVs produced
by ``plot/eval_checkpoint.py`` for these latent-selection modes:

* learned_qphi_switching
* uniform_episode_fixed
* uniform_random_at_router_opportunities
* qphi_initial_only_no_switch
* shuffled_qphi_outputs
* fixed, for every z in [0, K-1]

The resulting ledger answers whether q_phi harvested opponent-dependent
latent advantages, rather than merely proving that forced-z behaviors exist.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class EvalCell:
    map_set: str
    opponent: str
    latent_selection: str
    fixed_latent_id: int | None
    latent_resample_every: int | None
    episodes: int
    success_rate: float


def _norm_text(value: object) -> str:
    return str(value or "").strip()


def _norm_key(value: object) -> str:
    return _norm_text(value).upper()


def _parse_int(value: object, default: int = 0) -> int:
    text = _norm_text(value)
    if not text:
        return default
    return int(float(text))


def _parse_float(value: object) -> float:
    text = _norm_text(value)
    if not text:
        return float("nan")
    return float(text)


def _parse_fixed_latent_id(row: dict[str, str]) -> int | None:
    raw = _norm_text(row.get("fixed_latent_id"))
    if raw:
        return int(float(raw))
    selection = _norm_text(row.get("latent_selection")).lower()
    if selection.startswith("fixed_z"):
        return int(selection.removeprefix("fixed_z"))
    return None


def _parse_optional_int(value: object) -> int | None:
    text = _norm_text(value)
    if not text:
        return None
    return int(float(text))


def _normalize_selection(row: dict[str, str]) -> str:
    selection = _norm_text(row.get("latent_selection")).lower()
    resample_every = _parse_optional_int(row.get("latent_resample_every"))
    aliases = {
        "router": "learned_qphi_switching",
        "random-matched": "uniform_random_at_router_opportunities",
        "random-episode": "uniform_episode_fixed",
        "no-switch": "qphi_initial_only_no_switch",
        "shuffled": "shuffled_qphi_outputs",
    }
    selection = aliases.get(selection, selection)
    if selection == "learned_qphi_switching" and resample_every == 0:
        return "qphi_initial_only_no_switch"
    return selection


def load_eval_cells(paths: Iterable[Path]) -> list[EvalCell]:
    cells: list[EvalCell] = []
    for path in paths:
        with Path(path).open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                selection = _normalize_selection(row)
                if not selection:
                    continue
                cells.append(
                    EvalCell(
                        map_set=_norm_key(row.get("map_set")),
                        opponent=_norm_key(row.get("opponent")),
                        latent_selection=selection,
                        fixed_latent_id=_parse_fixed_latent_id(row),
                        latent_resample_every=_parse_optional_int(row.get("latent_resample_every")),
                        episodes=_parse_int(row.get("episodes")),
                        success_rate=_parse_float(row.get("success_rate")),
                    )
                )
    return cells


def _weighted_mean(values: Iterable[tuple[float, int]]) -> float:
    total_weight = 0
    total = 0.0
    for value, weight in values:
        if not math.isfinite(value) or int(weight) <= 0:
            continue
        total += float(value) * int(weight)
        total_weight += int(weight)
    if total_weight <= 0:
        return float("nan")
    return total / total_weight


def _mean_cells(cells: Iterable[EvalCell]) -> float:
    return _weighted_mean((c.success_rate, c.episodes) for c in cells)


def _split_name(opponent: str, holdout_opponents: set[str]) -> str:
    return "holdout" if _norm_key(opponent) in holdout_opponents else "train"


def _find_cell(
    by_key: dict[tuple[str, str, str, int | None], EvalCell],
    map_set: str,
    opponent: str,
    selection: str,
    fixed_z: int | None = None,
) -> EvalCell | None:
    return by_key.get((_norm_key(map_set), _norm_key(opponent), selection, fixed_z))


def _cell_success(cell: EvalCell | None) -> float:
    return float("nan") if cell is None else float(cell.success_rate)


def _cell_episodes(cell: EvalCell | None) -> int:
    return 0 if cell is None else int(cell.episodes)


def build_router_ledger(
    cells: list[EvalCell],
    *,
    latent_k: int,
    holdout_opponents: Iterable[str] = (),
    calibration_map_set: str | None = "calibration",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    holdout = {_norm_key(o) for o in holdout_opponents}
    calibration_key = _norm_key(calibration_map_set) if calibration_map_set else ""
    by_key: dict[tuple[str, str, str, int | None], EvalCell] = {}
    for cell in cells:
        key = (cell.map_set, cell.opponent, cell.latent_selection, cell.fixed_latent_id)
        by_key[key] = cell

    map_sets = sorted({c.map_set for c in cells if c.map_set != calibration_key})
    opponents = sorted({c.opponent for c in cells})
    per_pair: list[dict[str, object]] = []

    for map_set in map_sets:
        calibration_fixed: dict[int, list[EvalCell]] = {z: [] for z in range(latent_k)}
        if calibration_key:
            for opponent in opponents:
                if _split_name(opponent, holdout) == "holdout":
                    continue
                for z in range(latent_k):
                    fixed = _find_cell(by_key, calibration_key, opponent, "fixed", z)
                    if fixed is not None:
                        calibration_fixed[z].append(fixed)
        has_calibration = any(calibration_fixed[z] for z in range(latent_k))
        preselected_global_z = (
            max(range(latent_k), key=lambda z: _mean_cells(calibration_fixed.get(z, [])), default=0)
            if has_calibration
            else None
        )

        for opponent in opponents:
            fixed_cells = [
                _find_cell(by_key, map_set, opponent, "fixed", z)
                for z in range(latent_k)
            ]
            if any(c is None for c in fixed_cells):
                continue
            fixed_valid = [c for c in fixed_cells if c is not None]
            router = _find_cell(by_key, map_set, opponent, "learned_qphi_switching")
            uniform_episode = _find_cell(by_key, map_set, opponent, "uniform_episode_fixed")
            uniform_router = _find_cell(
                by_key, map_set, opponent, "uniform_random_at_router_opportunities"
            )
            no_switch = _find_cell(by_key, map_set, opponent, "qphi_initial_only_no_switch")
            shuffled = _find_cell(by_key, map_set, opponent, "shuffled_qphi_outputs")
            if router is None or uniform_episode is None or uniform_router is None:
                continue
            best_cell = max(fixed_valid, key=lambda c: c.success_rate)
            posthoc_global_z = max(range(latent_k), key=lambda z: fixed_valid[z].success_rate)
            posthoc_global_cell = fixed_valid[posthoc_global_z]
            preselected_global_cell = (
                fixed_valid[preselected_global_z] if preselected_global_z is not None else None
            )
            primary_baselines = [
                uniform_episode.success_rate,
                uniform_router.success_rate,
                _cell_success(preselected_global_cell),
                _cell_success(no_switch),
            ]
            primary_baseline = max(v for v in primary_baselines if math.isfinite(v))
            strict_baselines = [*primary_baselines, _cell_success(shuffled)]
            strict_baseline = max(v for v in strict_baselines if math.isfinite(v))
            per_pair.append(
                {
                    "split": _split_name(opponent, holdout),
                    "map_set": map_set,
                    "opponent": opponent,
                    "router_success_rate": router.success_rate,
                    "uniform_episode_fixed_success_rate": uniform_episode.success_rate,
                    "uniform_random_at_router_opportunities_success_rate": uniform_router.success_rate,
                    "no_switch_success_rate": _cell_success(no_switch),
                    "shuffled_success_rate": _cell_success(shuffled),
                    "preselected_global_fixed_z": (
                        "" if preselected_global_z is None else int(preselected_global_z)
                    ),
                    "preselected_global_fixed_success_rate": _cell_success(preselected_global_cell),
                    "preselected_global_fixed_available": bool(preselected_global_cell is not None),
                    "posthoc_global_fixed_oracle_z": int(posthoc_global_z),
                    "posthoc_global_fixed_oracle_success_rate": posthoc_global_cell.success_rate,
                    "fixed_best_per_opponent_z": int(best_cell.fixed_latent_id or 0),
                    "fixed_best_per_opponent_success_rate": best_cell.success_rate,
                    "posthoc_opponent_oracle_z": int(best_cell.fixed_latent_id or 0),
                    "posthoc_opponent_oracle_success_rate": best_cell.success_rate,
                    "primary_baseline_success_rate": primary_baseline,
                    "strict_baseline_success_rate": strict_baseline,
                    "delta_vs_uniform_episode_fixed": router.success_rate - uniform_episode.success_rate,
                    "delta_vs_uniform_random_at_router_opportunities": (
                        router.success_rate - uniform_router.success_rate
                    ),
                    "g_no_switch": router.success_rate - _cell_success(no_switch),
                    "g_shuffled": router.success_rate - _cell_success(shuffled),
                    "g_realized": router.success_rate - _cell_success(preselected_global_cell),
                    "g_available": (
                        best_cell.success_rate - _cell_success(preselected_global_cell)
                        if preselected_global_cell is not None
                        else float("nan")
                    ),
                    "g_oracle_gap": best_cell.success_rate - router.success_rate,
                    "delta_router_primary": router.success_rate - primary_baseline,
                    "delta_router_strict": router.success_rate - strict_baseline,
                    "beats_uniform_episode_fixed": router.success_rate > uniform_episode.success_rate,
                    "beats_uniform_random_at_router_opportunities": (
                        router.success_rate > uniform_router.success_rate
                    ),
                    "beats_no_switch": (
                        False if no_switch is None else router.success_rate > no_switch.success_rate
                    ),
                    "beats_shuffled": (
                        False if shuffled is None else router.success_rate > shuffled.success_rate
                    ),
                    "beats_preselected_global_fixed_z": (
                        False
                        if preselected_global_cell is None
                        else router.success_rate > preselected_global_cell.success_rate
                    ),
                    "no_material_harm_vs_oracle": router.success_rate >= best_cell.success_rate - 0.05,
                    "episodes_router": router.episodes,
                    "episodes_uniform_episode_fixed": uniform_episode.episodes,
                    "episodes_uniform_router_opportunities": uniform_router.episodes,
                    "episodes_no_switch": _cell_episodes(no_switch),
                    "episodes_shuffled": _cell_episodes(shuffled),
                    "episodes_fixed": min(c.episodes for c in fixed_valid),
                }
            )

    split_rows: list[dict[str, object]] = []
    for split in ("train", "holdout", "all"):
        rows = per_pair if split == "all" else [r for r in per_pair if r["split"] == split]
        if not rows:
            continue
        weights = [int(r["episodes_router"]) for r in rows]

        def wmean(field: str) -> float:
            return _weighted_mean((float(r[field]), w) for r, w in zip(rows, weights))

        split_rows.append(
            {
                "split": split,
                "n_cells": len(rows),
                "j_q_phi": wmean("router_success_rate"),
                "j_uniform_episode_fixed": wmean("uniform_episode_fixed_success_rate"),
                "j_uniform_random_at_router_opportunities": wmean(
                    "uniform_random_at_router_opportunities_success_rate"
                ),
                "j_no_switch": wmean("no_switch_success_rate"),
                "j_shuffled": wmean("shuffled_success_rate"),
                "j_preselected_global_fixed_z": wmean("preselected_global_fixed_success_rate"),
                "j_posthoc_global_fixed_oracle": wmean("posthoc_global_fixed_oracle_success_rate"),
                "j_fixed_best_per_opponent": wmean("fixed_best_per_opponent_success_rate"),
                "j_posthoc_opponent_oracle": wmean("posthoc_opponent_oracle_success_rate"),
                "j_primary_baseline": wmean("primary_baseline_success_rate"),
                "j_strict_baseline": wmean("strict_baseline_success_rate"),
                "delta_vs_uniform_episode_fixed": wmean("delta_vs_uniform_episode_fixed"),
                "delta_vs_uniform_random_at_router_opportunities": wmean(
                    "delta_vs_uniform_random_at_router_opportunities"
                ),
                "g_no_switch": wmean("g_no_switch"),
                "g_shuffled": wmean("g_shuffled"),
                "g_realized": wmean("g_realized"),
                "g_available": wmean("g_available"),
                "g_oracle_gap": wmean("g_oracle_gap"),
                "delta_router_primary": wmean("delta_router_primary"),
                "delta_router_strict": wmean("delta_router_strict"),
                "router_beats_uniform_episode_fixed": wmean("delta_vs_uniform_episode_fixed") > 0.0,
                "router_beats_uniform_random_at_router_opportunities": (
                    wmean("delta_vs_uniform_random_at_router_opportunities") > 0.0
                ),
                "router_beats_no_switch": wmean("g_no_switch") > 0.0,
                "router_beats_shuffled": wmean("g_shuffled") > 0.0,
                "router_beats_preselected_global_fixed_z": wmean("g_realized") > 0.0,
                "router_beats_primary_baseline": wmean("delta_router_primary") > 0.0,
                "router_beats_strict_baseline": wmean("delta_router_strict") > 0.0,
                "available_specialization": wmean("g_available") > 0.0,
            }
        )
    return per_pair, split_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, split_rows: list[dict[str, object]], pair_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Router Evaluation Matrix",
        "",
        "This report evaluates q_phi routing quality against locked v6i4 latent-selection controls. Posthoc oracle rows are upper bounds, not deployable baselines.",
        "",
        "## Split Ledger",
        "",
        "| Split | J(q_phi) | J(uniform episode) | J(uniform router times) | J(initial only) | J(shuffled) | J(preselected fixed) | J(posthoc opponent oracle) | Delta primary | G_available | G_oracle_gap | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in split_rows:
        verdict = "PASS" if row["available_specialization"] and row["router_beats_primary_baseline"] else "CHECK"
        lines.append(
            "| {split} | {j_q_phi:.4f} | {j_uniform_episode_fixed:.4f} | "
            "{j_uniform_random_at_router_opportunities:.4f} | {j_no_switch:.4f} | "
            "{j_shuffled:.4f} | {j_preselected_global_fixed_z:.4f} | "
            "{j_posthoc_opponent_oracle:.4f} | "
            "{delta_router_primary:+.4f} | {g_available:+.4f} | {g_oracle_gap:+.4f} | "
            "{verdict} |".format(**row, verdict=verdict)
        )
    lines.extend(
        [
            "",
            "## Per Opponent",
            "",
            "| Split | Map | Opponent | q_phi | Uniform episode | Uniform router times | Initial only | Shuffled | Preselected fixed | Posthoc global | Posthoc opponent | Delta primary | G_oracle_gap |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pair_rows:
        lines.append(
            "| {split} | {map_set} | {opponent} | {router_success_rate:.4f} | "
            "{uniform_episode_fixed_success_rate:.4f} | "
            "{uniform_random_at_router_opportunities_success_rate:.4f} | "
            "{no_switch_success_rate:.4f} | {shuffled_success_rate:.4f} | "
            "z{preselected_global_fixed_z} {preselected_global_fixed_success_rate:.4f} | "
            "z{posthoc_global_fixed_oracle_z} {posthoc_global_fixed_oracle_success_rate:.4f} | "
            "z{posthoc_opponent_oracle_z} {posthoc_opponent_oracle_success_rate:.4f} | "
            "{delta_router_primary:+.4f} | {g_oracle_gap:+.4f} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build q_phi router-quality ledger from eval aggregate CSVs.")
    parser.add_argument("--aggregate-csv", nargs="+", type=Path, required=True)
    parser.add_argument("--latent-k", type=int, default=4)
    parser.add_argument("--holdout-opponents", nargs="*", default=["OP4"])
    parser.add_argument(
        "--calibration-map-set",
        default="calibration",
        help=(
            "Map-set label used to preselect global fixed z before held-out evaluation. "
            "If absent, preselected_global_fixed_z is unavailable and posthoc oracle columns remain upper bounds."
        ),
    )
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()

    cells = load_eval_cells(args.aggregate_csv)
    pair_rows, split_rows = build_router_ledger(
        cells,
        latent_k=int(args.latent_k),
        holdout_opponents=args.holdout_opponents,
        calibration_map_set=args.calibration_map_set,
    )
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_pairs.csv"), pair_rows)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_summary.csv"), split_rows)
    write_report(args.out_prefix.with_name(args.out_prefix.name + "_report.md"), split_rows, pair_rows)
    print(f"[router_eval_matrix] cells={len(cells)} pairs={len(pair_rows)} splits={len(split_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
