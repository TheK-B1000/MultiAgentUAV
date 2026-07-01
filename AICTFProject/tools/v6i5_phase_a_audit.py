#!/usr/bin/env python3
"""Audit v6i5 Phase-A repertoire retention from a metrics CSV.

Pure analysis only. This script reads existing telemetry and computes the
cross-row retention quantities that are not safe to infer from a single row.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


KEY_METRICS: tuple[str, ...] = (
    "actor_z_jsd_mean",
    "actor_z_jsd_min",
    "actor_z_jsd_max",
    "actor_z_pairs_above_margin",
    "cf_batch_pair_jsd_mean",
    "actor_cf_to_ppo_grad_ratio",
    "actor_grad_ratio_cf_to_ppo",
    "actor_jsd_update_start",
    "actor_jsd_after_ppo",
    "actor_jsd_after_cf",
    "ppo_jsd_delta",
    "cf_jsd_delta",
    "cf_gain",
    "cf_retention_ratio",
    "actor_kl_after_ppo",
    "actor_kl_after_cf",
    "win_rate",
    "value_loss",
    "latent_marginal_entropy_nats",
    "latent_sampled_z_occupancy_ratio",
    "effective_num_latents",
    "strategy_entropy",
    "z_resampled_actual",
    "router_opportunity_count",
)


@dataclass(frozen=True)
class NumericSummary:
    count: int
    first: float | None
    last: float | None
    minimum: float | None
    maximum: float | None
    median_value: float | None
    delta: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "first": self.first,
            "last": self.last,
            "min": self.minimum,
            "max": self.maximum,
            "median": self.median_value,
            "delta": self.delta,
        }


def _to_float(value: Any) -> float:
    try:
        text = str(value).strip()
        if not text:
            return math.nan
        return float(text)
    except Exception:
        return math.nan


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def summarize_numeric(rows: list[dict[str, str]], key: str) -> NumericSummary:
    vals = _finite([_to_float(row.get(key, "")) for row in rows])
    if not vals:
        return NumericSummary(0, None, None, None, None, None, None)
    return NumericSummary(
        count=len(vals),
        first=vals[0],
        last=vals[-1],
        minimum=min(vals),
        maximum=max(vals),
        median_value=float(median(vals)),
        delta=vals[-1] - vals[0] if len(vals) >= 2 else 0.0,
    )


def compute_cross_row_retention(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx in range(len(rows) - 1):
        cur = rows[idx]
        nxt = rows[idx + 1]
        after_ppo = _to_float(cur.get("actor_jsd_after_ppo", ""))
        after_cf = _to_float(cur.get("actor_jsd_after_cf", ""))
        next_start = _to_float(nxt.get("actor_jsd_update_start", ""))
        cf_gain = after_cf - after_ppo
        retained_gain = next_start - after_ppo
        if not (math.isfinite(cf_gain) and math.isfinite(retained_gain)):
            ratio = math.nan
            reason = "missing_values"
        elif abs(cf_gain) <= 1e-12:
            ratio = math.nan
            reason = "no_measurable_cf_gain"
        else:
            ratio = retained_gain / max(abs(cf_gain), 1e-12)
            reason = ""
        out.append(
            {
                "row_index": idx,
                "update": cur.get("update", ""),
                "next_update": nxt.get("update", ""),
                "actor_jsd_after_ppo": after_ppo,
                "actor_jsd_after_cf": after_cf,
                "next_actor_jsd_update_start": next_start,
                "cf_gain": cf_gain,
                "retained_gain": retained_gain,
                "cross_row_retention_ratio": ratio,
                "retention_reason": reason,
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _window_rows(rows: list[dict[str, str]], max_rows: int) -> list[dict[str, str]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    return rows[-max_rows:]


def build_report(
    metrics_csv: Path,
    *,
    margin: float = 0.001,
    min_pairs: int = 5,
    target_ratio_low: float = 0.01,
    target_ratio_high: float = 0.05,
    window_rows: int = 10,
) -> dict[str, Any]:
    rows = _read_rows(metrics_csv)
    active_rows = [
        row
        for row in rows
        if _to_float(row.get("actor_cf_optimizer_step_count", "")) > 0.0
        or math.isfinite(_to_float(row.get("actor_jsd_after_cf", "")))
    ]
    recent = _window_rows(rows, window_rows)
    recent_active = _window_rows(active_rows, window_rows)
    retention_rows = compute_cross_row_retention(rows)
    finite_retention = [
        _to_float(row["cross_row_retention_ratio"])
        for row in retention_rows
        if math.isfinite(_to_float(row["cross_row_retention_ratio"]))
    ]
    latest = rows[-1] if rows else {}
    latest_ratio = _to_float(
        latest.get("actor_cf_to_ppo_grad_ratio")
        or latest.get("actor_grad_ratio_cf_to_ppo")
        or ""
    )
    latest_pairs = _to_float(latest.get("actor_z_pairs_above_margin", ""))
    latest_jsd = _to_float(latest.get("actor_z_jsd_mean", ""))
    latest_win = _to_float(latest.get("win_rate", ""))
    latest_kl = max(
        _to_float(latest.get("actor_kl_after_ppo", "")),
        _to_float(latest.get("actor_kl_after_cf", "")),
    )
    summaries = {key: summarize_numeric(rows, key).as_dict() for key in KEY_METRICS}
    recent_summaries = {key: summarize_numeric(recent, key).as_dict() for key in KEY_METRICS}
    active_summaries = {key: summarize_numeric(active_rows, key).as_dict() for key in KEY_METRICS}
    pass_checks = {
        "actor_z_jsd_above_margin": bool(math.isfinite(latest_jsd) and latest_jsd >= margin),
        "pairs_above_margin": bool(math.isfinite(latest_pairs) and latest_pairs >= min_pairs),
        "cf_ppo_ratio_in_band": bool(
            math.isfinite(latest_ratio) and target_ratio_low <= latest_ratio <= target_ratio_high
        ),
        "cf_delta_recent_positive": summarize_numeric(recent_active, "cf_jsd_delta").median_value is not None
        and float(summarize_numeric(recent_active, "cf_jsd_delta").median_value or 0.0) > 0.0,
        "ppo_delta_recent_smaller_or_negative": summarize_numeric(recent_active, "ppo_jsd_delta").median_value is not None
        and float(summarize_numeric(recent_active, "ppo_jsd_delta").median_value or 0.0)
        <= float(summarize_numeric(recent_active, "cf_jsd_delta").median_value or 0.0),
        "retention_not_consistently_zero": bool(finite_retention and float(median(finite_retention[-window_rows:])) > 0.0),
        "actor_kl_controlled": bool(not math.isfinite(latest_kl) or latest_kl < 0.05),
        "win_rate_not_collapsed": bool(not math.isfinite(latest_win) or latest_win >= 0.30),
    }
    recommendation = "continue_current_run"
    if rows and not all(pass_checks.values()):
        failed = [key for key, ok in pass_checks.items() if not ok]
        if "win_rate_not_collapsed" in failed:
            recommendation = "reduce_cf_lr_or_frequency"
        elif "retention_not_consistently_zero" in failed:
            recommendation = "inspect_retention_before_promotion"
        elif "cf_ppo_ratio_in_band" in failed:
            recommendation = "continue_until_cf_active_or_tune_cf_strength"
        else:
            recommendation = "continue_collecting_evidence"
    return {
        "protocol": "v6i5_phase_a_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_csv": str(metrics_csv),
        "row_count": len(rows),
        "cf_active_row_count": len(active_rows),
        "latest_update": latest.get("update", ""),
        "latest": {
            key: latest.get(key, "")
            for key in (
                "update",
                "actor_cf_update_mode",
                "z_resampled_actual",
                "actor_z_jsd_mean",
                "actor_z_pairs_above_margin",
                "actor_cf_to_ppo_grad_ratio",
                "cf_batch_pair_jsd_mean",
                "actor_jsd_update_start",
                "actor_jsd_after_ppo",
                "actor_jsd_after_cf",
                "ppo_jsd_delta",
                "cf_jsd_delta",
                "actor_kl_after_ppo",
                "actor_kl_after_cf",
                "win_rate",
                "value_loss",
            )
        },
        "summaries": summaries,
        "recent_summaries": recent_summaries,
        "cf_active_summaries": active_summaries,
        "cross_row_retention": retention_rows,
        "cross_row_retention_summary": summarize_numeric(
            [{ "ratio": str(v) } for v in finite_retention], "ratio"
        ).as_dict(),
        "pass_checks": pass_checks,
        "recommendation": recommendation,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    latest = report.get("latest", {})
    checks = report.get("pass_checks", {})
    lines = [
        "# v6i5 Phase A Audit",
        "",
        f"- metrics_csv: `{report.get('metrics_csv')}`",
        f"- rows: `{report.get('row_count')}`",
        f"- CF-active rows: `{report.get('cf_active_row_count')}`",
        f"- latest update: `{report.get('latest_update')}`",
        f"- recommendation: `{report.get('recommendation')}`",
        "",
        "## Latest",
        "",
    ]
    for key, value in latest.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Pass Checks", ""])
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Recent Metric Summary", ""])
    recent = report.get("recent_summaries", {})
    for key in (
        "actor_z_jsd_mean",
        "actor_z_pairs_above_margin",
        "actor_cf_to_ppo_grad_ratio",
        "cf_batch_pair_jsd_mean",
        "ppo_jsd_delta",
        "cf_jsd_delta",
        "actor_kl_after_ppo",
        "actor_kl_after_cf",
        "win_rate",
    ):
        summary = recent.get(key, {})
        lines.append(
            f"- `{key}`: first={summary.get('first')} last={summary.get('last')} "
            f"median={summary.get('median')} delta={summary.get('delta')}"
        )
    lines.extend(
        [
            "",
            "## Cross-Row Retention",
            "",
            "Retention is computed by shifting adjacent rows:",
            "",
            "`retained_gain = next.actor_jsd_update_start - current.actor_jsd_after_ppo`",
            "",
            "`cf_gain = current.actor_jsd_after_cf - current.actor_jsd_after_ppo`",
            "",
            "`retention_ratio = retained_gain / max(abs(cf_gain), 1e-12)`",
            "",
            f"Summary: `{report.get('cross_row_retention_summary')}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--margin", type=float, default=0.001)
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument("--window-rows", type=int, default=10)
    args = parser.parse_args(argv)
    metrics_csv = Path(args.metrics_csv).expanduser().resolve()
    if not metrics_csv.exists():
        raise FileNotFoundError(metrics_csv)
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else metrics_csv.parent / "v6i5_phase_a_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        metrics_csv,
        margin=float(args.margin),
        min_pairs=int(args.min_pairs),
        window_rows=int(args.window_rows),
    )
    stem = metrics_csv.stem
    json_path = out_dir / f"{stem}_phase_a_audit_report.json"
    retention_path = out_dir / f"{stem}_phase_a_cross_row_retention.csv"
    md_path = out_dir / f"{stem}_phase_a_audit_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(retention_path, report["cross_row_retention"])
    write_markdown(md_path, report)
    print(f"[v6i5_phase_a_audit] report: {json_path}")
    print(f"[v6i5_phase_a_audit] retention: {retention_path}")
    print(f"[v6i5_phase_a_audit] markdown: {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
