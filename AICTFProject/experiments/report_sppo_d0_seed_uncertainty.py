"""Add prospectively frozen seed-level uncertainty to SPPPO D0.

This is a diagnostic-only postprocessor. It consumes the complete D0 decision
rows after the live replay finishes. It does not build an environment, load a
policy, query Q_psi, train anything, or alter the frozen SPPPO V1 verdict.

The reporting contract was frozen while D0 was at 24/192 seeds in
``D0_SEED_LEVEL_UNCERTAINTY_REPORTING_FREEZE.json``. Bootstrap draws resample
seeds, carry all decision rows for each sampled seed, and recompute the lower
quartile cutoff inside every replicate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPPO = ROOT / "artifacts" / "strategic_demand" / "sppo"
ROWS_PATH = SPPO / "D0_pole_b_decision_rows.csv"
BASE_REPORT = SPPO / "D0_pole_b_diagnostic.json"
FREEZE_PATH = SPPO / "D0_SEED_LEVEL_UNCERTAINTY_REPORTING_FREEZE.json"
OUT = SPPO / "D0_pole_b_diagnostic_with_seed_uncertainty.json"

EXPECTED_SEEDS = list(range(10_300_001, 10_300_193))
NUMERIC_FIELDS = (
    "seed", "step", "decision_idx", "jsd_bits", "margin_A_bits",
    "margin_B_bits", "argmax_disagree", "eligible_heads",
    "delta_B_hat_qpsi", "qpsi_ranks_z1_correct", "blue_carrying",
    "red_carrying", "own_flag_home", "blue_score", "red_score",
)
METRICS = (
    "mean_margin_B_bits",
    "frac_margin_B_negative",
    "mean_delta_B_hat",
    "qpsi_correct_rate",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"REFUSING: no D0 rows in {path}")
    for row in rows:
        for key in NUMERIC_FIELDS:
            row[key] = float(row[key])
        row["seed"] = int(row["seed"])
    return rows


def _weighted_cutoff(values_sorted: np.ndarray, weights_sorted: np.ndarray,
                     quantile: float) -> float:
    """Weighted inverted-CDF quantile frozen for replicate-local quartiles."""
    total = float(weights_sorted.sum())
    if total <= 0:
        return float("nan")
    idx = int(np.searchsorted(np.cumsum(weights_sorted), quantile * total,
                              side="left"))
    return float(values_sorted[min(idx, len(values_sorted) - 1)])


def _metric_values(margin: np.ndarray, delta: np.ndarray, correct: np.ndarray,
                   weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    w = weights * mask
    denom = float(w.sum())
    if denom <= 0:
        return np.full(4, np.nan, dtype=np.float64)
    return np.asarray([
        np.dot(w, margin) / denom,
        np.dot(w, margin < 0) / denom,
        np.dot(w, delta) / denom,
        np.dot(w, correct) / denom,
    ], dtype=np.float64)


def _summarize(samples: np.ndarray, point: np.ndarray, alpha: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for i, name in enumerate(METRICS):
        vals = samples[:, i]
        vals = vals[np.isfinite(vals)]
        result[name] = {
            "point": float(point[i]) if np.isfinite(point[i]) else None,
            "lcb95": float(np.quantile(vals, alpha / 2)) if len(vals) else None,
            "ucb95": float(np.quantile(vals, 1 - alpha / 2)) if len(vals) else None,
            "valid_bootstrap_samples": int(len(vals)),
        }
    return result


def seed_bootstrap(rows: list[dict[str, Any]], *, samples: int, alpha: float,
                   rng_seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    seeds = sorted({int(r["seed"]) for r in rows})
    seed_to_idx = {seed: i for i, seed in enumerate(seeds)}
    row_seed_idx = np.asarray([seed_to_idx[int(r["seed"])] for r in rows], dtype=np.int64)
    margin = np.asarray([r["margin_B_bits"] for r in rows], dtype=np.float64)
    delta = np.asarray([r["delta_B_hat_qpsi"] for r in rows], dtype=np.float64)
    correct = np.asarray([r["qpsi_ranks_z1_correct"] for r in rows], dtype=np.float64)

    point_q25 = float(np.percentile(margin, 25))
    fixed_masks = {
        "ALL": np.ones(len(rows), dtype=bool),
        "carrying": np.asarray([bool(r["blue_carrying"]) for r in rows]),
        "not_carrying": np.asarray([not bool(r["blue_carrying"]) for r in rows]),
        "own_flag_home": np.asarray([bool(r["own_flag_home"]) for r in rows]),
        "own_flag_stolen": np.asarray([not bool(r["own_flag_home"]) for r in rows]),
        "early": np.asarray([r["tertile"] == "early" for r in rows]),
        "mid": np.asarray([r["tertile"] == "mid" for r in rows]),
        "late": np.asarray([r["tertile"] == "late" for r in rows]),
    }
    point_masks = dict(fixed_masks)
    point_masks["worst_quartile_margin_B"] = margin <= point_q25

    names = list(point_masks)
    draws = {name: np.full((samples, len(METRICS)), np.nan) for name in names}
    cutoffs = np.full(samples, np.nan)
    rng = np.random.default_rng(rng_seed)
    order = np.argsort(margin, kind="stable")
    sorted_margin = margin[order]

    for bi in range(samples):
        seed_counts = rng.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)))
        weights = seed_counts[row_seed_idx].astype(np.float64)
        cutoff = _weighted_cutoff(sorted_margin, weights[order], 0.25)
        cutoffs[bi] = cutoff
        for name in names:
            mask = margin <= cutoff if name == "worst_quartile_margin_B" else fixed_masks[name]
            draws[name][bi] = _metric_values(margin, delta, correct, weights, mask)

    summaries = {
        name: _summarize(draws[name],
                         _metric_values(margin, delta, correct,
                                        np.ones(len(rows)), point_masks[name]), alpha)
        for name in names
    }
    quartile = {
        "point_cutoff": point_q25,
        "bootstrap_cutoff_lcb95": float(np.quantile(cutoffs, alpha / 2)),
        "bootstrap_cutoff_ucb95": float(np.quantile(cutoffs, 1 - alpha / 2)),
        "method": "replicate-local weighted inverted-CDF cutoff",
    }
    return summaries, quartile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=ROWS_PATH)
    ap.add_argument("--base-report", type=Path, default=BASE_REPORT)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    cfg = freeze["bootstrap"]
    rows = _read_rows(args.rows)
    seeds = sorted({int(r["seed"]) for r in rows})
    if seeds != EXPECTED_SEEDS:
        raise SystemExit(f"REFUSING: D0 rows cover {len(seeds)} seeds, expected exact 192-seed block")
    if not args.base_report.is_file():
        raise SystemExit("REFUSING: base D0 report is not complete")
    base = json.loads(args.base_report.read_text(encoding="utf-8"))
    if int(base.get("n_seeds", -1)) != 192:
        raise SystemExit("REFUSING: base D0 report is not the complete 192-seed result")

    summaries, quartile = seed_bootstrap(
        rows, samples=int(cfg["samples"]), alpha=float(cfg["alpha"]),
        rng_seed=int(cfg["rng_seed"]),
    )
    worst_q = summaries["worst_quartile_margin_B"]["qpsi_correct_rate"]
    if worst_q["lcb95"] > 0.5:
        interpretation = "EVIDENCE_FAVORS_OPTIMIZATION_INTERFERENCE"
        wording = ("The evidence favors optimization/interference over scorer "
                   "misranking as the dominant failure mode.")
    elif worst_q["ucb95"] < 0.5:
        interpretation = "EVIDENCE_FAVORS_SURROGATE_MISRANKING"
        wording = ("The evidence favors scorer misranking over optimization/interference "
                   "as the dominant failure mode.")
    else:
        interpretation = "MECHANISM_EVIDENCE_INCONCLUSIVE"
        wording = "Seed-level uncertainty does not distinguish the two mechanism hypotheses."

    record = {
        "record": "SPPPO V1 D0 with prospectively frozen seed-level uncertainty",
        "status": "DIAGNOSTIC_ONLY_NOT_A_GATE",
        "inputs": {
            "rows": str(args.rows.relative_to(ROOT)),
            "rows_sha256": _sha256(args.rows),
            "base_report": str(args.base_report.relative_to(ROOT)),
            "base_report_sha256": _sha256(args.base_report),
            "reporting_freeze": str(FREEZE_PATH.relative_to(ROOT)),
            "reporting_freeze_sha256": _sha256(FREEZE_PATH),
        },
        "bootstrap": cfg,
        "n_seeds": len(seeds),
        "n_decision_points": len(rows),
        "worst_quartile": quartile,
        "seed_level_uncertainty": summaries,
        "diagnostic_interpretation": interpretation,
        "paper_safe_wording": wording,
        "non_gating": ("This report does not alter SPPPO_V1_STRATEGIC_PAYOFF_"
                       "PRESERVING_PPO_NOT_CONFIRMED or authorize a new method."),
    }
    if args.out.exists():
        raise SystemExit(f"REFUSING: output already exists: {args.out}")
    args.out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"D0 seed-level uncertainty complete -> {args.out}")
    print(f"diagnostic interpretation: {interpretation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
