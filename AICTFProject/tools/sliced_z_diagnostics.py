"""Sliced per-z diagnostics report for v4i3+ latent runs.

Global MI / global z-WR-spread can hide z's that are useful only on
specific slices (against one opponent, when behind, while carrying the
flag, etc.). This tool surfaces those slices from existing telemetry
without changing the trainer:

    Section A (from {run_tag}_metrics.csv):
      A1. Per-z x per-opponent win-rate / count / score
      A2. Per-z behavior fingerprint (ranked by max-min spread across z)
      A3. Per-z bucket distributions (role / spread / pressure / a-d ratio)
      A4. Per-z marginal outcome stats

    Section B (from {run_tag}_e3_steps.csv, STREAMED over a tail window):
      B1. Per-z x score-state (ahead / tied / behind): count fraction
      B2. Per-z x score-state: mean behavior fingerprint
      B3. Per-z x flag-carry-active (carrier_escort_count > 0 / red
          carrier observed): count fraction + mean behavior
      B4. Per-z x team-phase: count fraction

The e3_steps file can be multiple GB. To stay tractable on a partial run
and to keep the report focused on RECENT behavior, the e3 streaming
respects ``--last-n-updates``; older updates are skipped row-by-row
(no full load into memory).

Usage:
    python tools/sliced_z_diagnostics.py \\
        --run-tag v4i3_summer_proof_OP5_OP6_OP7_4v4 \\
        --checkpoint-dir checkpoints/4v4 \\
        --last-n-updates 8 \\
        --out checkpoints/4v4/v4i3_sliced_z_report.md
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _to_int(value: Any, default: int = -1) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_metrics_tail(metrics_path: Path, last_n: int) -> list[dict[str, str]]:
    if not metrics_path.exists():
        return []
    with metrics_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-last_n:] if last_n > 0 else rows


def _column_floats(rows: list[dict[str, str]], col: str) -> list[float]:
    vals: list[float] = []
    for r in rows:
        if col not in r:
            continue
        v = r.get(col, "")
        if v == "" or v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv) or math.isinf(fv):
            continue
        vals.append(fv)
    return vals


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


# ---------------------------------------------------------------------------
# Section A: aggregates from metrics.csv tail
# ---------------------------------------------------------------------------


def _section_a1_z_by_opponent(
    rows: list[dict[str, str]], latent_k: int, n_opponents: int
) -> str:
    """Per-z x per-opponent table: WR, count, blue/red mean score.

    Columns ``episode_opp{o}_z{k}_count`` / ``..._win_rate`` are emitted
    by the latent diagnostics module. Counts are tail-mean (per-update
    counts averaged); win rates are tail-mean. Both are over the
    ``--last-n-updates`` tail window of the metrics CSV. Opponent columns
    with no episodes in the window are auto-pruned so the table only
    surfaces opponents the trainer actually saw.
    """
    matrix = [[None] * n_opponents for _ in range(latent_k)]
    for z in range(latent_k):
        for o in range(n_opponents):
            cnts = _column_floats(rows, f"episode_opp{o}_z{z}_count")
            wrs = _column_floats(rows, f"episode_opp{o}_z{z}_win_rate")
            if not cnts and not wrs:
                continue
            cnt_mean = _mean(cnts)
            if math.isnan(cnt_mean) or cnt_mean <= 0:
                continue
            matrix[z][o] = (_mean(wrs), cnt_mean)

    used_ops = [
        o
        for o in range(n_opponents)
        if any(matrix[z][o] is not None for z in range(latent_k))
    ]
    lines = ["### A1. Per-z x per-opponent (tail-mean over last N updates)", ""]
    if not used_ops:
        lines.append("_(no per-(opponent, z) episode counts in the tail window)_")
        return "\n".join(lines)

    # Canonical opponent_id -> OP-tag mapping (kept in sync with
    # ``rl/custom_ppo/csv_writers.py::_opponent_id_int_from_info``). We
    # don't import from the rl package here to keep this tool a
    # standalone CSV reader with no torch/rl dependency.
    op_tag_by_id = {0: "OP1", 1: "OP2", 2: "OP3", 3: "OP4", 4: "OP5", 5: "OP6", 6: "OP7"}

    def _label(o: int) -> str:
        return op_tag_by_id.get(o, f"opp{o}")

    lines.append(
        "Column headers are the public OP tags; the underlying CSV columns "
        "use the integer ``opponent_id`` (OP1->0 ... OP7->6 per `csv_writers.py`)."
    )
    lines.append("")
    lines.append(
        "| z | "
        + " | ".join(f"{_label(o)} WR / n" for o in used_ops)
        + " | row total n |"
    )
    lines.append("|---|" + "|".join(["---"] * (len(used_ops) + 1)) + "|")
    for z in range(latent_k):
        row_total = 0.0
        cells = []
        for o in used_ops:
            cell = matrix[z][o]
            if cell is None:
                cells.append("-")
            else:
                wr, cnt = cell
                cells.append(f"{wr:.3f} / {cnt:.1f}")
                row_total += cnt
        lines.append(f"| z{z} | " + " | ".join(cells) + f" | {row_total:.1f} |")
    return "\n".join(lines)


def _section_a2_behavior_fingerprint(
    rows: list[dict[str, str]], latent_k: int
) -> str:
    """Per-z behavior fingerprint -- highlights dims with biggest cross-z spread.

    Reads ``latent_z{k}_behavior_{dim}_mean`` columns, computes the
    tail-mean per z, then ranks dims by (max - min) across z. The top
    rows are the behavior dimensions where z is doing the most work.
    """
    sample = rows[-1] if rows else {}
    behavior_dims = sorted({
        c[len(f"latent_z0_behavior_") : -len("_mean")]
        for c in sample.keys()
        if c.startswith("latent_z0_behavior_") and c.endswith("_mean")
    })
    if not behavior_dims:
        return "### A2. Per-z behavior fingerprint\n\n(no `latent_z*_behavior_*_mean` columns found)\n"

    per_dim: dict[str, list[float]] = {}
    for dim in behavior_dims:
        per_z_vals: list[float] = []
        for z in range(latent_k):
            col = f"latent_z{z}_behavior_{dim}_mean"
            vals = _column_floats(rows, col)
            per_z_vals.append(_mean(vals))
        per_dim[dim] = per_z_vals

    def _spread(vs: list[float]) -> float:
        clean = [v for v in vs if not (math.isnan(v) or math.isinf(v))]
        if not clean:
            return 0.0
        return max(clean) - min(clean)

    def _scaled_spread(vs: list[float]) -> float:
        clean = [v for v in vs if not (math.isnan(v) or math.isinf(v))]
        if not clean:
            return 0.0
        rng = max(clean) - min(clean)
        mn = sum(abs(v) for v in clean) / len(clean)
        return rng / (mn + 1e-8)

    ranked = sorted(
        per_dim.items(),
        key=lambda kv: _scaled_spread(kv[1]),
        reverse=True,
    )

    lines = [
        "### A2. Per-z behavior fingerprint",
        "",
        "Top dims ranked by scaled spread `(max_z - min_z) / (mean|val| + eps)`.",
        "Large scaled spread = different z's produce noticeably different",
        "behavior on that dim. Tiny scaled spread = z is decorative for this dim.",
        "",
        "| dim | " + " | ".join(f"z{z}" for z in range(latent_k)) + " | spread | rel_spread |",
        "|---|" + "|".join(["---"] * (latent_k + 2)) + "|",
    ]
    for dim, vs in ranked[:20]:
        cells = []
        for v in vs:
            if math.isnan(v):
                cells.append("nan")
            else:
                cells.append(f"{v:.4f}")
        sp = _spread(vs)
        rs = _scaled_spread(vs)
        lines.append(
            f"| `{dim}` | " + " | ".join(cells) + f" | {sp:.4f} | {rs:.3f} |"
        )
    return "\n".join(lines)


def _section_a3_bucket_distributions(
    rows: list[dict[str, str]], latent_k: int
) -> str:
    """Per-z share of each behavior bucket (role / spread / pressure / a-d ratio).

    Columns ``latent_{bucket_kind}{b}_z{k}_frac`` give the fraction of
    decision steps that fell into bucket ``b`` with z=k. For each
    bucket-kind we print the per-z share so the reader can see whether
    one z is consistently picked when the team is in that bucket.
    """
    bucket_kinds = ["role", "spread", "pressure", "adr"]
    lines = ["### A3. Per-z share by team-behavior bucket", ""]
    for kind in bucket_kinds:
        # Discover how many buckets this kind has (latent_role0..N-1)
        sample = rows[-1] if rows else {}
        bucket_ids = sorted({
            int(c.split(kind)[-1].split("_")[0])
            for c in sample.keys()
            if c.startswith(f"latent_{kind}")
            and c.endswith("_z0_frac")
        })
        if not bucket_ids:
            continue
        lines.append(f"**{kind}** buckets:")
        lines.append("")
        lines.append(
            "| bucket | " + " | ".join(f"z{z}" for z in range(latent_k)) + " |"
        )
        lines.append("|---|" + "|".join(["---"] * latent_k) + "|")
        for b in bucket_ids:
            cells = []
            for z in range(latent_k):
                col = f"latent_{kind}{b}_z{z}_frac"
                vals = _column_floats(rows, col)
                m = _mean(vals)
                cells.append("-" if math.isnan(m) else f"{m:.3f}")
            lines.append(f"| {b} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _section_a4_z_marginal(
    rows: list[dict[str, str]], latent_k: int
) -> str:
    """Per-z marginal episode outcome stats: WR, blue/red score, margin."""
    lines = ["### A4. Per-z marginal outcome", ""]
    lines.append(
        "| z | episodes (mean per update) | WR | blue_score | red_score | win_margin |"
    )
    lines.append("|---|---|---|---|---|---|")
    for z in range(latent_k):
        cnt = _mean(_column_floats(rows, f"episode_z_{z}_count"))
        wr = _mean(_column_floats(rows, f"episode_z_{z}_win_rate"))
        b = _mean(_column_floats(rows, f"episode_z_{z}_blue_score_mean"))
        r = _mean(_column_floats(rows, f"episode_z_{z}_red_score_mean"))
        mg = _mean(_column_floats(rows, f"episode_z_{z}_win_margin_mean"))

        def _fmt(v: float) -> str:
            return "-" if math.isnan(v) else f"{v:.3f}"

        lines.append(
            f"| z{z} | {_fmt(cnt)} | {_fmt(wr)} | {_fmt(b)} | {_fmt(r)} | {_fmt(mg)} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section B: streaming aggregates from e3_steps.csv
# ---------------------------------------------------------------------------


@dataclass
class _SliceAccumulator:
    """Sum + count + per-dim sums for one (z, slice-key) bucket."""

    count: int = 0
    dim_sums: dict[str, float] = field(default_factory=dict)
    dim_counts: dict[str, int] = field(default_factory=dict)


def _stream_e3_slices(
    e3_path: Path,
    last_n_updates: int,
    behavior_dims: list[str],
) -> tuple[
    dict[tuple[int, str], _SliceAccumulator],  # (z, score_outcome)
    dict[tuple[int, str], _SliceAccumulator],  # (z, carry_state)
    dict[tuple[int, int], _SliceAccumulator],  # (z, team_phase int)
    int,
    int,
]:
    """Stream e3_steps and accumulate per-(z, slice) sums for behavior dims.

    Filters to the last ``last_n_updates`` `update` values present in the
    file. Returns four dicts plus (total rows scanned, rows accepted).
    """
    if not e3_path.exists():
        return {}, {}, {}, 0, 0

    # First pass: find max update value (cheap -- only reads 1 column).
    updates_seen: set[int] = set()
    with e3_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = _to_int(row.get("update"))
            if u >= 0:
                updates_seen.add(u)
    if not updates_seen:
        return {}, {}, {}, 0, 0
    keep_updates = set(sorted(updates_seen)[-last_n_updates:])
    print(
        f"  e3 updates kept (last {last_n_updates}): "
        f"{sorted(keep_updates)} of {len(updates_seen)} total",
        file=sys.stderr,
    )

    score_acc: dict[tuple[int, str], _SliceAccumulator] = defaultdict(
        _SliceAccumulator
    )
    carry_acc: dict[tuple[int, str], _SliceAccumulator] = defaultdict(
        _SliceAccumulator
    )
    phase_acc: dict[tuple[int, int], _SliceAccumulator] = defaultdict(
        _SliceAccumulator
    )

    total = 0
    accepted = 0
    with e3_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            u = _to_int(row.get("update"))
            if u not in keep_updates:
                continue
            accepted += 1
            z = _to_int(row.get("z_t"))
            if z < 0:
                continue

            score = str(row.get("score_outcome") or "").strip()
            # Flag-carry derivation: carrier_escort_count > 0 means at
            # least one blue carrier is currently active. We can't see
            # the red carrier directly, but ``n_intercept_near_enemy_carrier > 0``
            # implies at least one red carrier is on the field that
            # our intercepters are close to.
            blue_carrier = _to_float(row.get("carrier_escort_count")) > 0.5
            red_carrier_proxy = _to_float(row.get("n_intercept_near_enemy_carrier")) > 0.5
            if blue_carrier and red_carrier_proxy:
                carry = "both"
            elif blue_carrier:
                carry = "blue_carries"
            elif red_carrier_proxy:
                carry = "red_carries"
            else:
                carry = "neither"
            phase = _to_int(row.get("phase_id"))

            def _add(acc_dict: dict, key: tuple) -> None:
                acc = acc_dict[key]
                acc.count += 1
                for dim in behavior_dims:
                    v = row.get(dim)
                    if v is None or v == "":
                        continue
                    fv = _to_float(v)
                    acc.dim_sums[dim] = acc.dim_sums.get(dim, 0.0) + fv
                    acc.dim_counts[dim] = acc.dim_counts.get(dim, 0) + 1

            if score:
                _add(score_acc, (z, score))
            _add(carry_acc, (z, carry))
            if phase >= 0:
                _add(phase_acc, (z, phase))

    return score_acc, carry_acc, phase_acc, total, accepted


def _render_slice_table(
    title: str,
    description: str,
    accumulator: dict[tuple[int, Any], _SliceAccumulator],
    latent_k: int,
    slice_values: list[Any],
    primary_dim: str | None = None,
) -> str:
    """Render a per-z x per-slice count-fraction table.

    If ``primary_dim`` is provided, also includes the per-cell mean of
    that dim so the report can show both routing (count frac) and
    behavior (e.g. spread mean) on the same axis.
    """
    lines = [f"### {title}", "", description, ""]
    # Count table: fraction of z-rows in each slice.
    z_totals = {z: sum(accumulator[(z, s)].count for s in slice_values) for z in range(latent_k)}
    lines.append(
        "**Routing share** (fraction of z=k decision steps spent in this slice):"
    )
    lines.append("")
    lines.append("| z | " + " | ".join(str(s) for s in slice_values) + " | total n |")
    lines.append("|---|" + "|".join(["---"] * (len(slice_values) + 1)) + "|")
    for z in range(latent_k):
        cells = []
        total = z_totals[z] or 1
        for s in slice_values:
            n = accumulator[(z, s)].count
            cells.append(f"{n / total:.3f} (n={n})")
        lines.append(f"| z{z} | " + " | ".join(cells) + f" | {z_totals[z]} |")

    if primary_dim is not None:
        lines.append("")
        lines.append(
            f"**Mean `{primary_dim}` per (z, slice)** "
            "(z-row spread within a column = z drives different behavior in that slice):"
        )
        lines.append("")
        lines.append("| z | " + " | ".join(str(s) for s in slice_values) + " |")
        lines.append("|---|" + "|".join(["---"] * len(slice_values)) + "|")
        col_means: dict[Any, list[float]] = defaultdict(list)
        for z in range(latent_k):
            cells = []
            for s in slice_values:
                acc = accumulator[(z, s)]
                n = acc.dim_counts.get(primary_dim, 0)
                if n == 0:
                    cells.append("-")
                else:
                    m = acc.dim_sums.get(primary_dim, 0.0) / n
                    cells.append(f"{m:.3f}")
                    col_means[s].append(m)
            lines.append(f"| z{z} | " + " | ".join(cells) + " |")
        # Per-column max - min across z (the "z-effect size" in that slice).
        lines.append(
            "| **max-min** | "
            + " | ".join(
                f"{(max(col_means[s]) - min(col_means[s])):.3f}" if col_means.get(s) else "-"
                for s in slice_values
            )
            + " |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-z sliced diagnostics for v4i3+ latent runs."
    )
    p.add_argument("--run-tag", required=True)
    p.add_argument(
        "--checkpoint-dir",
        default="checkpoints/4v4",
        help="Directory containing {run_tag}_metrics.csv / _e3_steps.csv / _episodes.csv.",
    )
    p.add_argument(
        "--latent-k",
        type=int,
        default=4,
        help="Number of latent strategies K (default 4).",
    )
    p.add_argument(
        "--n-opponents",
        type=int,
        default=8,
        help="Max opponent_id index to look for in episode_opp{o}_z{k}_* columns.",
    )
    p.add_argument(
        "--last-n-updates",
        type=int,
        default=8,
        help=(
            "Restrict both metrics.csv tail and e3_steps streaming to "
            "the last N policy-update windows (default 8). Set 0 to use all."
        ),
    )
    p.add_argument(
        "--out",
        default=None,
        help="Markdown report path (default: {checkpoint_dir}/{run_tag}_sliced_z_report.md).",
    )
    p.add_argument(
        "--skip-e3",
        action="store_true",
        help="Skip the streaming e3 section (much faster; only Section A).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    ck = Path(args.checkpoint_dir)
    metrics = ck / f"{args.run_tag}_metrics.csv"
    e3 = ck / f"{args.run_tag}_e3_steps.csv"
    if not metrics.exists():
        print(f"[error] missing metrics CSV: {metrics}", file=sys.stderr)
        return 2
    out_path = (
        Path(args.out)
        if args.out
        else ck / f"{args.run_tag}_sliced_z_report.md"
    )

    metrics_rows = _read_metrics_tail(metrics, args.last_n_updates)
    if not metrics_rows:
        print(f"[error] metrics CSV has no rows: {metrics}", file=sys.stderr)
        return 2
    print(
        f"[sliced_z] read {len(metrics_rows)} tail metric rows; "
        f"latest_global_step={metrics_rows[-1].get('global_step', '?')}",
        file=sys.stderr,
    )

    sections: list[str] = [
        f"# Per-z sliced diagnostics: `{args.run_tag}`",
        "",
        f"- Generated from `{metrics.name}` (last {len(metrics_rows)} updates) "
        f"and `{e3.name}` (last {args.last_n_updates} updates).",
        f"- Latent K = {args.latent_k}",
        "",
        "Sections A1-A4 are tail-means over the metrics CSV; Section B is "
        "a streaming pass over the e3 per-step telemetry. Use the **max-min** "
        "rows in B to find slices where z is secretly useful (large max-min "
        "= different z's behave differently in that slice).",
        "",
        "---",
        "",
        "## Section A: metrics.csv tail aggregates",
        "",
        _section_a1_z_by_opponent(metrics_rows, args.latent_k, args.n_opponents),
        "",
        _section_a2_behavior_fingerprint(metrics_rows, args.latent_k),
        "",
        _section_a3_bucket_distributions(metrics_rows, args.latent_k),
        "",
        _section_a4_z_marginal(metrics_rows, args.latent_k),
        "",
    ]

    if not args.skip_e3 and e3.exists():
        size_mb = os.path.getsize(e3) / 1e6
        print(
            f"[sliced_z] streaming e3 ({size_mb:.1f} MB) -- this may take a moment...",
            file=sys.stderr,
        )
        behavior_dims = [
            "team_spread",
            "num_attackers",
            "num_defenders",
            "carrier_escort_count",
            "intercept_pressure",
            "defense_pressure",
            "attack_defense_ratio",
            "avg_blue_to_enemy_flag",
            "avg_blue_to_own_flag",
        ]
        score_acc, carry_acc, phase_acc, total_rows, accepted_rows = _stream_e3_slices(
            e3, args.last_n_updates, behavior_dims
        )
        print(
            f"[sliced_z] e3 scan: {total_rows} rows, {accepted_rows} in window",
            file=sys.stderr,
        )
        sections.extend([
            "---",
            "",
            "## Section B: e3 per-step slices (streaming)",
            "",
            f"- Rows scanned: {total_rows}; rows in window: {accepted_rows}",
            "",
            _render_slice_table(
                "B1. Per-z x score-state",
                "Score state at decision time. If `max-min` of mean attackers / "
                "spread / pressure stays tiny across z in EVERY column, z is not "
                "doing context-conditional work.",
                score_acc,
                args.latent_k,
                ["ahead", "tied", "behind"],
                primary_dim="num_attackers",
            ),
            "",
            _render_slice_table(
                "B2. Per-z x score-state (defense pressure)",
                "Same slice, different behavior column. Defense pressure is high "
                "when red is close to our flag -- a z that drives high "
                "defense_pressure here while another stays low is a coordinator.",
                score_acc,
                args.latent_k,
                ["ahead", "tied", "behind"],
                primary_dim="defense_pressure",
            ),
            "",
            _render_slice_table(
                "B3. Per-z x flag-carry state",
                "blue_carries = our team has the carrier (carrier_escort_count > 0). "
                "red_carries = red carrier observed near our intercepters. "
                "neither = neutral pre-pickup state.",
                carry_acc,
                args.latent_k,
                ["blue_carries", "red_carries", "both", "neither"],
                primary_dim="num_attackers",
            ),
            "",
            _render_slice_table(
                "B4. Per-z x team_phase (phase_id)",
                "Phase IDs come from `rl/latent_phase_labels.py::TEAM_PHASES`. "
                "Per-z occupancy fraction tells you which phases each z lives in.",
                phase_acc,
                args.latent_k,
                list(range(6)),
                primary_dim="num_attackers",
            ),
            "",
        ])
    elif args.skip_e3:
        sections.extend(["---", "", "_(Section B skipped: --skip-e3)_", ""])
    else:
        sections.extend([
            "---",
            "",
            f"_(Section B skipped: e3 CSV not found at `{e3}`)_",
            "",
        ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"[sliced_z] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
