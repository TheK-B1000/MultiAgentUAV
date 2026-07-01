"""Summarize v6i5 CF-strength calibration metrics from diagnostic CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median


def _float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def _percentile(values: list[float], pct: float) -> float:
    vals = sorted(_finite(values))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * pct
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return vals[int(rank)]
    frac = rank - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _slope(values: list[float]) -> float:
    vals = _finite(values)
    if len(vals) < 2:
        return math.nan
    return vals[-1] - vals[0]


def summarize_csv(path: Path) -> dict[str, float | str | int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no metric rows")

    cf_rows = [r for r in rows if _float(r, "actor_cf_loss_evaluated", 0.0) >= 1.0]
    late_count = max(1, math.ceil(len(cf_rows) * 0.5)) if cf_rows else 0
    late_rows = cf_rows[-late_count:] if late_count else []

    ratios = [_float(r, "actor_cf_to_ppo_grad_ratio") for r in cf_rows]
    late_ratios = [_float(r, "actor_cf_to_ppo_grad_ratio") for r in late_rows]
    actor_z = [_float(r, "actor_z_jsd_mean") for r in cf_rows]
    cf_jsd = [_float(r, "cf_batch_pair_jsd_mean") for r in cf_rows]
    final = rows[-1]
    final_cf = cf_rows[-1] if cf_rows else final
    finite_late = _finite(late_ratios)

    return {
        "csv": str(path),
        "rows": len(rows),
        "cf_rows": len(cf_rows),
        "final_timesteps": int(_float(final, "timesteps", 0.0)),
        "cf_ratio_median": median(_finite(ratios)) if _finite(ratios) else math.nan,
        "cf_ratio_late_median": median(finite_late) if finite_late else math.nan,
        "cf_ratio_p95": _percentile(ratios, 0.95),
        "actor_z_jsd_start": actor_z[0] if actor_z else math.nan,
        "actor_z_jsd_final": actor_z[-1] if actor_z else math.nan,
        "actor_z_jsd_delta": _slope(actor_z),
        "cf_batch_jsd_start": cf_jsd[0] if cf_jsd else math.nan,
        "cf_batch_jsd_final": cf_jsd[-1] if cf_jsd else math.nan,
        "cf_batch_jsd_delta": _slope(cf_jsd),
        "pairs_above_margin_final": _float(final_cf, "actor_z_pairs_above_margin"),
        "actor_kl_final": _float(final, "approx_kl", _float(final, "actor_approx_kl")),
        "win_rate_final": _float(final, "win_rate"),
        "z_embedding_cf_grad_final": _float(final_cf, "actor_z_embedding_cf_grad_norm"),
        "film_gamma_cf_grad_final": _float(final_cf, "actor_film_gamma_cf_grad_norm"),
        "film_beta_cf_grad_final": _float(final_cf, "actor_film_beta_cf_grad_norm"),
        "cf_grad_final": _float(final_cf, "actor_cf_grad_norm_scaled"),
        "ppo_grad_final": _float(final_cf, "actor_ppo_grad_norm"),
        "nan_or_inf_required": int(
            any(
                not math.isfinite(v)
                for v in (
                    _float(final_cf, "actor_cf_to_ppo_grad_ratio"),
                    _float(final_cf, "actor_cf_grad_norm_scaled"),
                    _float(final_cf, "actor_ppo_grad_norm"),
                    _float(final_cf, "actor_z_embedding_cf_grad_norm"),
                )
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    args = parser.parse_args()

    summaries = [summarize_csv(path) for path in args.csv]
    fields = [
        "csv",
        "final_timesteps",
        "cf_rows",
        "cf_ratio_median",
        "cf_ratio_late_median",
        "cf_ratio_p95",
        "actor_z_jsd_delta",
        "cf_batch_jsd_delta",
        "pairs_above_margin_final",
        "actor_kl_final",
        "win_rate_final",
        "z_embedding_cf_grad_final",
        "nan_or_inf_required",
    ]
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
