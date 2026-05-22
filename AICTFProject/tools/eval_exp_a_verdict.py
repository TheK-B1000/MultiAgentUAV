"""Experiment A verdict: parse a sharp3-style training log and check the q_phi oracle pass criterion.

Reads ``[PPO|diag]`` lines from a training log and extracts the three signals locked at design time:
    * ``MI_z_o``         -- mutual information between latent z and opponent id
    * ``z_wr_spread``    -- max-min spread of per-z win rates
    * ``zH``             -- q_phi entropy (max is ln K, e.g. ln 4 = 1.386 for K=4)

Pass criterion (>= 2 of 3 by the *last completed update* in the log):
    * MI_z_o      >= 0.05
    * z_wr_spread >= 0.20 sustained over the last 3 diag prints (mean of last 3)
    * zH          <= 1.25  (sustained: mean of last 3)

If the criterion passes, the bottleneck is q_phi *observability* and the next step is Experiment B
(forced-z pretraining with better features). If it fails, the bottleneck is structural in the routing
objective itself, and the next step is Experiment C (router-only supervised training on
``best_z_per_opponent`` labels).

Usage:
    python tools/eval_exp_a_verdict.py logs/train_latent_sharp3_oracleA_*.log
    python tools/eval_exp_a_verdict.py --baseline-log logs/train_latent_sharp3_hardpool_20260521_204455.log \
        logs/train_latent_sharp3_oracleA_*.log
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def _read_log_text(path: Path) -> str:
    """Read a training log that may be UTF-8 OR UTF-16 (PowerShell Tee-Object default on Windows).

    Sniffs the BOM. Falls back to utf-8 with replacement only if nothing else parses cleanly.
    """
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


_DIAG_RE = re.compile(
    r"\[PPO\|diag\]\s*steps=(?P<steps>\d+)"
    r".*?ev=(?P<ev>-?[0-9.]+)"
    r".*?v_loss=(?P<v_loss>-?[0-9.]+)"
    r".*?qphi_grad=(?P<qphi_grad>-?[0-9.]+)"
    r".*?zH=(?P<zH>-?[0-9.]+)\([0-9.]+\)"
    r".*?z_wr_spread=(?P<z_wr_spread>-?[0-9.]+)"
    r".*?MI_z_o=(?P<MI_z_o>-?[0-9.]+)",
    re.DOTALL,
)


@dataclass(frozen=True)
class DiagRow:
    steps: int
    ev: float
    v_loss: float
    qphi_grad: float
    zH: float
    z_wr_spread: float
    MI_z_o: float


def _read_diag_rows(text: str) -> List[DiagRow]:
    """Extract every ``[PPO|diag]`` block (multi-line) and parse the keyed metrics we care about."""
    # Re-flow newlines so multi-line diag blocks become greppable: collapse any whitespace run
    # following a [PPO|diag] marker until the next blank-or-non-diag line. The regex below uses
    # DOTALL so a single re.findall over the original text already handles it -- but we still need
    # to chunk by [PPO|diag] occurrences so DOTALL doesn't cross blocks. The simplest robust way:
    # split on the marker.
    rows: List[DiagRow] = []
    parts = text.split("[PPO|diag]")
    for part in parts[1:]:  # drop preamble
        # Limit "part" to the first ~12 lines to avoid eating into later diag blocks.
        truncated = "\n".join(part.splitlines()[:12])
        full_block = "[PPO|diag] " + truncated
        m = _DIAG_RE.search(full_block)
        if not m:
            continue
        try:
            rows.append(
                DiagRow(
                    steps=int(m.group("steps")),
                    ev=float(m.group("ev")),
                    v_loss=float(m.group("v_loss")),
                    qphi_grad=float(m.group("qphi_grad")),
                    zH=float(m.group("zH")),
                    z_wr_spread=float(m.group("z_wr_spread")),
                    MI_z_o=float(m.group("MI_z_o")),
                )
            )
        except ValueError:
            continue
    return rows


def _final_wr_from_log(text: str) -> Optional[Tuple[int, int, int, float]]:
    """Pull the last 'W=.. L=.. D=.. WR=..%' line for a headline number."""
    last = None
    for m in re.finditer(r"W=(\d+)\s+L=(\d+)\s+D=(\d+)\s+WR=([\d.]+)%", text):
        last = m
    if last is None:
        return None
    return int(last.group(1)), int(last.group(2)), int(last.group(3)), float(last.group(4))


@dataclass(frozen=True)
class Verdict:
    label: str          # "PASS" | "FAIL" | "BORDERLINE"
    score: int          # 0..3 -- number of criteria met
    notes: List[str]


def _evaluate(rows: List[DiagRow]) -> Verdict:
    if not rows:
        return Verdict("FAIL", 0, ["no [PPO|diag] rows found in log"])
    last3 = rows[-3:] if len(rows) >= 3 else rows
    mi_last = rows[-1].MI_z_o
    spread_mean = sum(r.z_wr_spread for r in last3) / len(last3)
    zH_mean = sum(r.zH for r in last3) / len(last3)

    mi_pass = mi_last >= 0.05
    spread_pass = spread_mean >= 0.20
    zH_pass = zH_mean <= 1.25
    score = int(mi_pass) + int(spread_pass) + int(zH_pass)

    notes = [
        f"diag rows parsed: {len(rows)} (steps {rows[0].steps} .. {rows[-1].steps})",
        f"MI_z_o (last diag)         = {mi_last:.4f}   threshold >= 0.05   -> {'PASS' if mi_pass else 'FAIL'}",
        f"z_wr_spread (mean last 3)  = {spread_mean:.4f}   threshold >= 0.20   -> {'PASS' if spread_pass else 'FAIL'}",
        f"zH (mean last 3)           = {zH_mean:.4f}   threshold <= 1.25   -> {'PASS' if zH_pass else 'FAIL'}",
    ]
    if score >= 2:
        return Verdict("PASS", score, notes)
    if score == 1:
        return Verdict("BORDERLINE", score, notes)
    return Verdict("FAIL", score, notes)


def _next_step_guidance(label: str) -> str:
    if label == "PASS":
        return (
            "Bottleneck is observability/features. Next: Experiment B (forced-z pretraining) "
            "with better real features to recover routing without the oracle. Optionally first "
            "promote this 200k smoke to a 1M run with --qphi-oracle one_hot to see the WR ceiling."
        )
    if label == "BORDERLINE":
        return (
            "Mixed signal. Either extend the smoke to 500k for cleaner trajectory, or run the "
            "11-dim 'both' (one_hot + config) oracle to confirm whether identity alone was too thin."
        )
    return (
        "Routing objective is structurally broken (not observability). Next: Experiment C "
        "(freeze actor + critic; train q_phi supervised on best_z_per_opponent labels derived "
        "from a forced-z eval sweep). Do not invest more compute in larger sharp3-style runs."
    )


def _format_baseline_comparison(
    rows: List[DiagRow], baseline_rows: List[DiagRow]
) -> Optional[str]:
    if not rows or not baseline_rows:
        return None
    # Compare each oracle row to the closest-step baseline row.
    lines = ["", "Step-aligned comparison (oracle vs baseline same-step from sharp3 hardpool):", ""]
    header = f"{'step':>10s}  {'MI_z_o(o/b)':>16s}  {'z_wr_spread(o/b)':>20s}  {'zH(o/b)':>14s}"
    lines.append(header)
    lines.append("-" * len(header))
    base_by_step = {r.steps: r for r in baseline_rows}
    base_steps_sorted = sorted(base_by_step.keys())
    for r in rows:
        nearest = min(base_steps_sorted, key=lambda s: abs(s - r.steps))
        b = base_by_step[nearest]
        lines.append(
            f"{r.steps:>10d}  {r.MI_z_o:.4f}/{b.MI_z_o:.4f}      "
            f"{r.z_wr_spread:.4f}/{b.z_wr_spread:.4f}            "
            f"{r.zH:.4f}/{b.zH:.4f}"
        )
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="oracle-A training log to evaluate")
    parser.add_argument(
        "--baseline-log",
        type=Path,
        default=None,
        help="optional matched no-oracle log (e.g. sharp3 hardpool) for step-aligned comparison",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.log.exists():
        print(f"ERROR: log not found: {args.log}", file=sys.stderr)
        return 2

    text = _read_log_text(args.log)
    rows = _read_diag_rows(text)
    verdict = _evaluate(rows)

    print(f"\n========== Experiment A verdict: {args.log.name} ==========")
    for n in verdict.notes:
        print(f"  {n}")

    final = _final_wr_from_log(text)
    if final is not None:
        w, l, d, wr = final
        print(f"  final W/L/D = {w}/{l}/{d}  WR = {wr:.1f}%")

    if args.baseline_log is not None and args.baseline_log.exists():
        baseline_text = _read_log_text(args.baseline_log)
        baseline_rows = _read_diag_rows(baseline_text)
        cmp_block = _format_baseline_comparison(rows, baseline_rows)
        if cmp_block:
            print(cmp_block)

    print(f"\n  VERDICT: {verdict.label}  (criteria met: {verdict.score}/3)")
    print(f"  next step: {_next_step_guidance(verdict.label)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
