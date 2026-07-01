"""Tactical regime analysis: do the contexts actually exist, persist, and matter?

This tool answers the two prerequisite questions a Summer-faithful latent
strategy must satisfy BEFORE we ask whether ``q_phi`` is learning to
route between regimes:

  1. **Context duration.** For each tactical context category
     (team_phase, score_outcome, opponent, blue-carries-red-flag,
     red-carries-blue-flag), how long does the team typically stay in
     that context? Compared to ``latent_resample_every_n`` (= 64
     decision steps in v4i3), is the regime longer (router interrupts
     half-baked strategies), shorter (router refresh is too slow), or
     similar?

  2. **Context consequence.** Within each context, does the team's
     *behavior* actually vary, and does varying it correlate with
     measurable outcome differences? If the policy executes a single
     uniform behavior across every regime, no latent system can
     discover meaningful specialization there.

Data sources -- streamed/sampled from existing artifacts; no trainer
change required:

  * ``{run_tag}_e3_steps.csv`` -- per decision-step records including
    ``phase_id``, ``score_outcome``, ``opponent_id``, ``z_t``, the full
    behavior telemetry block (``num_attackers``, ``num_defenders``,
    ``carrier_escort_count``, ``intercept_pressure``, ``defense_pressure``,
    ``team_spread``, ``attack_defense_ratio``, ...). Rows are loaded
    only for the last ``--last-n-updates`` policy updates and sorted by
    (update, env_id, rollout_step) so consecutive rows for one env are
    a true time series.
  * ``{run_tag}_episodes.csv`` -- terminal per-episode records used for
    per-opponent / per-z win-rate breakdowns.

The output is a markdown report with the four sections described
above (plus a short interpretation footnote that ties each row of the
duration table to the ``latent_resample_every_n=64`` reference line).

Usage::

    python tools/tactical_regime_analysis.py \\
        --run-tag v4i3_summer_proof_OP5_OP6_OP7_4v4 \\
        --checkpoint-dir checkpoints/4v4 \\
        --last-n-updates 3 \\
        --resample-every-n 64 \\
        --out checkpoints/4v4/v4i3_tactical_regime_report.md

Run-time cost: load+sort over a 3-update window from a ~5 GB e3_steps
file is ~3 minutes (single-threaded CSV parse). Memory cost is ~30 MB
for 200k rows of the whitelisted columns.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


# ---------------------------------------------------------------------------
# Canonical labels
# ---------------------------------------------------------------------------


TEAM_PHASES: tuple[str, ...] = (
    "neutral",
    "attacking_enemy_flag",
    "carrying_flag_home",
    "defending_own_flag",
    "enemy_carrying_our_flag",
    "stalemate",
)

# Mirrors rl/custom_ppo/csv_writers.py::_OPPONENT_ID_TO_TAG; kept local
# so this tool stays a standalone CSV reader with no rl/torch deps.
_OPPONENT_TAG: dict[int, str] = {0: "OP1", 1: "OP2", 2: "OP3", 3: "OP4", 4: "OP5", 5: "OP6", 6: "OP7"}

BEHAVIOR_COLUMNS: tuple[str, ...] = (
    "num_attackers",
    "num_defenders",
    "carrier_escort_count",
    "n_intercept_near_enemy_carrier",
    "team_spread",
    "intercept_pressure",
    "defense_pressure",
    "attack_defense_ratio",
)

_E3_KEEP_COLUMNS: tuple[str, ...] = (
    "update",
    "env_id",
    "rollout_step",
    "global_step",
    "z_t",
    "opponent_id",
    "phase_id",
    "score_outcome",
    "blue_ahead",
) + BEHAVIOR_COLUMNS


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _to_float(v: Any, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        out = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _to_int(v: Any, default: int = -1) -> int:
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _opponent_tag(opp_id: int) -> str:
    return _OPPONENT_TAG.get(opp_id, f"opp{opp_id}")


def _phase_tag(p: int) -> str:
    if 0 <= p < len(TEAM_PHASES):
        return TEAM_PHASES[p]
    return f"phase{p}"


# ---------------------------------------------------------------------------
# Stream e3_steps -> sorted in-memory window
# ---------------------------------------------------------------------------


def _load_e3_window(
    e3_path: Path, last_n_updates: int
) -> tuple[list[dict[str, Any]], list[int]]:
    """Load only the last ``last_n_updates`` updates from e3_steps and sort.

    Returns a list of small per-row dicts (whitelisted columns only) and
    the list of update ids that were kept. Streaming guarantees that
    even multi-GB files stay tractable while we still get random
    access for run-length detection.
    """
    if not e3_path.exists():
        return [], []
    updates_seen: set[int] = set()
    with e3_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = _to_int(row.get("update"))
            if u >= 0:
                updates_seen.add(u)
    if not updates_seen:
        return [], []
    keep_updates = set(sorted(updates_seen)[-last_n_updates:])

    rows: list[dict[str, Any]] = []
    with e3_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = _to_int(row.get("update"))
            if u not in keep_updates:
                continue
            slim = {k: row.get(k, "") for k in _E3_KEEP_COLUMNS}
            rows.append(slim)
    rows.sort(
        key=lambda r: (
            _to_int(r.get("update")),
            _to_int(r.get("env_id")),
            _to_int(r.get("rollout_step")),
        )
    )
    return rows, sorted(keep_updates)


# ---------------------------------------------------------------------------
# Run-length detection over (update, env_id) sequences
# ---------------------------------------------------------------------------


def _runs_per_value(
    rows: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Any],
) -> tuple[dict[Any, list[int]], int]:
    """For each (update, env_id) time series, emit per-value run lengths.

    Returns ``(runs_by_value, total_steps)`` where ``runs_by_value`` maps
    a context value (e.g. an int phase_id) to a list of consecutive
    decision-step run lengths observed in that value, and
    ``total_steps`` is the total decision steps the contexts were
    sampled over (sum of runs across all values).
    """
    runs: dict[Any, list[int]] = defaultdict(list)
    total = 0
    cur_key: tuple[int, int] | None = None
    cur_val: Any = None
    cur_len = 0
    for row in rows:
        u = _to_int(row.get("update"))
        e = _to_int(row.get("env_id"))
        seq = (u, e)
        v = key_fn(row)
        if seq != cur_key:
            if cur_val is not None and cur_len > 0:
                runs[cur_val].append(cur_len)
            cur_key = seq
            cur_val = v
            cur_len = 1
            total += 1
            continue
        if v == cur_val:
            cur_len += 1
        else:
            if cur_val is not None and cur_len > 0:
                runs[cur_val].append(cur_len)
            cur_val = v
            cur_len = 1
        total += 1
    if cur_val is not None and cur_len > 0:
        runs[cur_val].append(cur_len)
    return runs, total


def _summarize_runs(
    runs: list[int],
) -> dict[str, float]:
    if not runs:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p25": float("nan"), "p75": float("nan"), "max": float("nan")}
    s = sorted(runs)
    n = len(s)

    def _q(p: float) -> float:
        if n == 1:
            return float(s[0])
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return float(s[lo])
        return float(s[lo] + (s[hi] - s[lo]) * (idx - lo))

    return {
        "n": n,
        "mean": sum(s) / n,
        "median": float(statistics.median(s)),
        "p25": _q(0.25),
        "p75": _q(0.75),
        "max": float(s[-1]),
    }


def _render_duration_table(
    title: str,
    description: str,
    runs_by_value: dict[Any, list[int]],
    total_steps: int,
    label_fn: Callable[[Any], str],
    resample_every_n: int,
) -> str:
    lines = [f"### {title}", "", description, ""]
    lines.append(
        f"Reference: latent_resample_every_n = **{resample_every_n}** decision steps. "
        "Read the ``vs_64`` column as:"
    )
    lines.append("")
    lines.append("  * ``=long`` (median > 2x resample) -- router interrupts mature strategies")
    lines.append("  * ``=short`` (median < 0.25x resample) -- resample is too slow")
    lines.append("  * ``=match`` -- resample roughly tracks regime length")
    lines.append("")
    lines.append(
        "| context | freq | n_runs | mean | median | p25 | p75 | max | vs_64 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    items = sorted(
        runs_by_value.items(), key=lambda kv: sum(kv[1]), reverse=True
    )
    for val, rs in items:
        stats = _summarize_runs(rs)
        n_steps_in_value = sum(rs)
        freq = n_steps_in_value / total_steps if total_steps else float("nan")
        med = stats["median"]
        if math.isnan(med):
            vs = "-"
        elif med > 2 * resample_every_n:
            vs = "=long"
        elif med < 0.25 * resample_every_n:
            vs = "=short"
        else:
            vs = "=match"

        def _f(x: float) -> str:
            return "-" if math.isnan(x) else f"{x:.1f}"

        lines.append(
            f"| `{label_fn(val)}` | {freq:.3f} | {int(stats['n'])} | "
            f"{_f(stats['mean'])} | {_f(med)} | {_f(stats['p25'])} | "
            f"{_f(stats['p75'])} | {_f(stats['max'])} | {vs} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 2: Behavior fingerprint within context
# ---------------------------------------------------------------------------


def _behavior_by_context(
    rows: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Any],
) -> dict[Any, dict[str, list[float]]]:
    by: dict[Any, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        v = key_fn(row)
        if v is None:
            continue
        for col in BEHAVIOR_COLUMNS:
            raw = row.get(col)
            if raw == "" or raw is None:
                continue
            try:
                by[v][col].append(float(raw))
            except (TypeError, ValueError):
                continue
    return by


def _render_behavior_table(
    title: str,
    description: str,
    behavior: dict[Any, dict[str, list[float]]],
    label_fn: Callable[[Any], str],
) -> str:
    lines = [f"### {title}", "", description, ""]
    lines.append(
        "| context | "
        + " | ".join(c for c in BEHAVIOR_COLUMNS)
        + " | n_steps |"
    )
    lines.append("|---|" + "|".join(["---"] * (len(BEHAVIOR_COLUMNS) + 1)) + "|")
    items = sorted(
        behavior.items(),
        key=lambda kv: -sum(len(v) for v in kv[1].values()),
    )
    for val, dim_map in items:
        cells = []
        n_steps = 0
        for col in BEHAVIOR_COLUMNS:
            vs = dim_map.get(col) or []
            if vs:
                cells.append(f"{sum(vs) / len(vs):.3f}")
                n_steps = max(n_steps, len(vs))
            else:
                cells.append("-")
        lines.append(
            f"| `{label_fn(val)}` | " + " | ".join(cells) + f" | {n_steps} |"
        )

    lines.append("")
    lines.append(
        "**Cross-context spread per dim** (max-min over context rows; "
        "large = the policy *does* execute different behaviors across "
        "contexts; near-zero = uniform behavior, latent specialization "
        "cannot help):"
    )
    lines.append("")
    spread: list[tuple[str, float, float]] = []
    for col in BEHAVIOR_COLUMNS:
        means = [
            (sum(vs) / len(vs))
            for vs in (dim_map.get(col) for _, dim_map in behavior.items())
            if vs
        ]
        if len(means) >= 2:
            rng = max(means) - min(means)
            avg_abs = sum(abs(m) for m in means) / len(means)
            rel = rng / (avg_abs + 1e-8)
            spread.append((col, rng, rel))
    spread.sort(key=lambda t: t[2], reverse=True)
    lines.append("| dim | absolute spread | relative spread |")
    lines.append("|---|---|---|")
    for col, a, r in spread:
        lines.append(f"| `{col}` | {a:.3f} | {r:.3f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 3: P(z | context) -- per-context router preference
# ---------------------------------------------------------------------------


def _z_by_context(
    rows: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Any],
    latent_k: int,
) -> dict[Any, list[int]]:
    counts: dict[Any, list[int]] = defaultdict(lambda: [0] * latent_k)
    for row in rows:
        v = key_fn(row)
        if v is None:
            continue
        z = _to_int(row.get("z_t"))
        if 0 <= z < latent_k:
            counts[v][z] += 1
    return counts


def _render_z_routing_table(
    title: str,
    description: str,
    z_counts: dict[Any, list[int]],
    label_fn: Callable[[Any], str],
    latent_k: int,
) -> str:
    lines = [f"### {title}", "", description, ""]
    lines.append(
        "| context | "
        + " | ".join(f"P(z{k})" for k in range(latent_k))
        + " | n_steps | uniform_kl |"
    )
    lines.append("|---|" + "|".join(["---"] * (latent_k + 2)) + "|")
    uniform = 1.0 / latent_k
    items = sorted(
        z_counts.items(), key=lambda kv: -sum(kv[1])
    )
    for val, cnts in items:
        total = sum(cnts)
        if total <= 0:
            continue
        probs = [c / total for c in cnts]
        kl = sum(
            (p * math.log(p / uniform)) for p in probs if p > 0
        )
        cells = [f"{p:.3f}" for p in probs]
        lines.append(
            f"| `{label_fn(val)}` | " + " | ".join(cells)
            + f" | {total} | {kl:.4f} |"
        )
    lines.append("")
    lines.append(
        "``uniform_kl`` = KL(P(z|context) || uniform). If this is "
        "consistently near zero across every context, the router is "
        "ignoring context (or all contexts look the same to ``q_phi``)."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 4: short-horizon phase transition matrix (consequence proxy)
# ---------------------------------------------------------------------------


def _phase_transition_lookahead(
    rows: list[dict[str, Any]],
    lookahead: int,
    bucket_fn: Callable[[dict[str, Any]], str | None],
    from_phase: int,
) -> dict[str, dict[str, int]]:
    """Within sequences in ``from_phase``, count terminal phase after ``lookahead`` steps.

    Bucketing is provided by ``bucket_fn(row)`` -- e.g. ``num_defenders``
    bucket. The returned dict is bucket -> phase_label -> count.
    """
    by_env: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        u = _to_int(row.get("update"))
        e = _to_int(row.get("env_id"))
        by_env[(u, e)].append(row)
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for seq in by_env.values():
        n = len(seq)
        for i, row in enumerate(seq):
            p = _to_int(row.get("phase_id"))
            if p != from_phase:
                continue
            bucket = bucket_fn(row)
            if bucket is None:
                continue
            j = min(i + lookahead, n - 1)
            future_p = _to_int(seq[j].get("phase_id"))
            label = _phase_tag(future_p) if future_p >= 0 else "unknown"
            out[bucket][label] += 1
    return out


def _render_transition_table(
    title: str,
    description: str,
    transitions: dict[str, dict[str, int]],
    lookahead: int,
) -> str:
    lines = [f"### {title}", "", description, ""]
    if not transitions:
        lines.append(
            "_(no steps matched the source phase in the window)_"
        )
        return "\n".join(lines)
    all_terminals = sorted({k for b in transitions.values() for k in b.keys()})
    lines.append(
        f"Distribution of phase at t+{lookahead} given the bucket of the "
        "behavior dim at t."
    )
    lines.append("")
    lines.append(
        "| bucket | n_t | "
        + " | ".join(f"P(t+{lookahead}={t})" for t in all_terminals)
        + " |"
    )
    lines.append("|---|" + "|".join(["---"] * (len(all_terminals) + 1)) + "|")
    for bucket in sorted(transitions.keys()):
        counts = transitions[bucket]
        n_t = sum(counts.values())
        cells = []
        for t in all_terminals:
            p = counts.get(t, 0) / n_t if n_t else 0.0
            cells.append(f"{p:.3f}")
        lines.append(f"| `{bucket}` | {n_t} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-opponent WR from episodes.csv (Section 4b)
# ---------------------------------------------------------------------------


def _per_opponent_wr(episodes_path: Path) -> dict[int, dict[str, float]]:
    if not episodes_path.exists():
        return {}
    by_opp: dict[int, list[tuple[int, float]]] = defaultdict(list)
    with episodes_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            opp = _to_int(row.get("opponent_id"))
            if opp < 0:
                continue
            success = _to_int(row.get("success"))
            margin = _to_float(row.get("win_margin"))
            by_opp[opp].append((success, margin))
    out: dict[int, dict[str, float]] = {}
    for opp, items in by_opp.items():
        wins = sum(1 for s, _ in items if s == 1)
        n = len(items)
        margins = [m for _, m in items]
        out[opp] = {
            "n": float(n),
            "wr": wins / n if n else float("nan"),
            "win_margin_mean": sum(margins) / n if n else float("nan"),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-tag", required=True)
    p.add_argument("--checkpoint-dir", default="checkpoints/4v4")
    p.add_argument("--latent-k", type=int, default=4)
    p.add_argument(
        "--last-n-updates",
        type=int,
        default=3,
        help="Restrict the e3 window to the last N policy updates (default 3).",
    )
    p.add_argument(
        "--resample-every-n",
        type=int,
        default=64,
        help="Reference latent_resample_every_n for the duration ``vs_N`` "
        "annotation column (default 64, the v4i3 value).",
    )
    p.add_argument(
        "--lookahead",
        type=int,
        default=20,
        help="Decision-step lookahead for the consequence transition matrix "
        "(default 20, a few latent arcs short of one resample interval).",
    )
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    ck = Path(args.checkpoint_dir)
    e3 = ck / f"{args.run_tag}_e3_steps.csv"
    eps = ck / f"{args.run_tag}_episodes.csv"
    if not e3.exists():
        print(f"[error] missing e3 CSV: {e3}", file=sys.stderr)
        return 2
    out_path = (
        Path(args.out)
        if args.out
        else ck / f"{args.run_tag}_tactical_regime_report.md"
    )

    size_mb = os.path.getsize(e3) / 1e6
    print(
        f"[tactical] streaming e3 ({size_mb:.1f} MB) -- "
        f"loading last {args.last_n_updates} updates into memory...",
        file=sys.stderr,
    )
    rows, kept_updates = _load_e3_window(e3, args.last_n_updates)
    if not rows:
        print(f"[error] no rows in window", file=sys.stderr)
        return 2
    print(
        f"[tactical] loaded {len(rows)} rows from updates {kept_updates}",
        file=sys.stderr,
    )

    n_attackers_bucket: Callable[[dict[str, Any]], str | None] = (
        lambda r: ("low" if _to_float(r.get("num_attackers")) < 1.5
                   else "mid" if _to_float(r.get("num_attackers")) < 2.5
                   else "high")
    )
    n_defenders_bucket: Callable[[dict[str, Any]], str | None] = (
        lambda r: ("low" if _to_float(r.get("num_defenders")) < 0.5
                   else "mid" if _to_float(r.get("num_defenders")) < 1.5
                   else "high")
    )
    carrier_escort_bucket: Callable[[dict[str, Any]], str | None] = (
        lambda r: ("none" if _to_float(r.get("carrier_escort_count")) < 0.5
                   else "low" if _to_float(r.get("carrier_escort_count")) < 1.5
                   else "high")
    )

    sections: list[str] = [
        f"# Tactical Regime Analysis: `{args.run_tag}`",
        "",
        f"- Source: `{e3.name}` (last {args.last_n_updates} updates, "
        f"{len(rows)} decision-step rows) and `{eps.name}`.",
        f"- Reference resample interval: `latent_resample_every_n = "
        f"{args.resample_every_n}` decision steps.",
        f"- Latent K = {args.latent_k}.",
        "",
        "This report answers two questions:",
        "",
        "  1. **Do tactical regimes exist with enough persistence that a "
        "K-step latent makes sense?** (Section 1)",
        "  2. **Within a regime, does the policy ACTUALLY execute different "
        "behaviors that correlate with different outcomes?** (Sections 2-4)",
        "",
        "If Section 1 shows the regimes the user cares about (own flag "
        "stolen, carrying enemy flag) routinely last more than "
        f"`{args.resample_every_n}` steps, the current resample interval "
        "interrupts strategy maturation. If those regimes last "
        f"< `{args.resample_every_n // 4}` steps, the resample is too slow "
        "to react. If the regime barely persists at all, episode-level "
        "latent strategy may not even be the right granularity.",
        "",
        "---",
        "",
        "## Section 1: Context duration",
        "",
    ]

    # Phase
    runs_phase, total_phase = _runs_per_value(
        rows, lambda r: _to_int(r.get("phase_id"))
    )
    sections.append(
        _render_duration_table(
            "1a. team_phase (TEAM_PHASES enum)",
            "Phase ids: 0 neutral | 1 attacking_enemy_flag | "
            "2 carrying_flag_home (team carries enemy flag) | "
            "3 defending_own_flag | 4 enemy_carrying_our_flag "
            "(own flag stolen) | 5 stalemate.",
            runs_phase,
            total_phase,
            lambda v: f"{v} {_phase_tag(v)}" if v >= 0 else "unknown",
            args.resample_every_n,
        )
    )
    sections.append("")

    # Score state
    runs_score, total_score = _runs_per_value(
        rows, lambda r: (r.get("score_outcome") or "").strip() or None
    )
    sections.append(
        _render_duration_table(
            "1b. score_outcome (running score)",
            "Ahead / tied / behind run-length distribution. Long "
            "``behind`` stretches with no specialized z is the classic "
            "case where latent strategy is most needed.",
            runs_score,
            total_score,
            lambda v: str(v),
            args.resample_every_n,
        )
    )
    sections.append("")

    # Opponent
    runs_opp, total_opp = _runs_per_value(
        rows, lambda r: _to_int(r.get("opponent_id"))
    )
    sections.append(
        _render_duration_table(
            "1c. opponent_id",
            "Within a single episode the opponent is fixed, so a 'run' "
            "here = one episode's worth of decision steps with that "
            "opponent. Mean run length doubles as 'mean episode length "
            "by opponent'.",
            runs_opp,
            total_opp,
            lambda v: _opponent_tag(v) if v >= 0 else "unknown",
            args.resample_every_n,
        )
    )
    sections.append("")

    # Derived: blue-carries-red-flag (proxy via phase 2)
    runs_blue, total_blue = _runs_per_value(
        rows, lambda r: "blue_carries" if _to_int(r.get("phase_id")) == 2 else "not_blue_carries"
    )
    sections.append(
        _render_duration_table(
            "1d. blue carries enemy flag (phase_id == 2)",
            "Direct proxy for 'team is carrying the enemy flag'. Long "
            "runs here mean escort strategies have time to develop; "
            "short runs mean the carrier is shot / drops the flag fast.",
            runs_blue,
            total_blue,
            lambda v: v,
            args.resample_every_n,
        )
    )
    sections.append("")

    # Derived: red-carries-blue-flag = own flag stolen (phase 4)
    runs_red, total_red = _runs_per_value(
        rows, lambda r: "own_flag_stolen" if _to_int(r.get("phase_id")) == 4 else "own_flag_safe"
    )
    sections.append(
        _render_duration_table(
            "1e. own flag stolen (phase_id == 4)",
            "Direct proxy for 'red is carrying our flag'. This is the "
            "decisive 'do we need defenders right now' regime.",
            runs_red,
            total_red,
            lambda v: v,
            args.resample_every_n,
        )
    )
    sections.append("")

    # ----- Section 2: per-context behavior -----
    sections.extend([
        "---",
        "",
        "## Section 2: Behavior fingerprint within context",
        "",
        "Mean behavior telemetry per context. If the policy already "
        "produces materially different behaviors across contexts "
        "(large cross-context spread on a dim), there is room for "
        "latent specialization within that dim. If a dim is flat "
        "across contexts (tiny relative spread), latent z cannot "
        "improve it.",
        "",
    ])
    sections.append(
        _render_behavior_table(
            "2a. Behavior by team_phase",
            "How does the team's behavior change with the tactical "
            "phase? Healthy signal: ``num_defenders`` rises in "
            "phase 3/4, ``carrier_escort_count`` rises in phase 2.",
            _behavior_by_context(
                rows, lambda r: _to_int(r.get("phase_id"))
            ),
            lambda v: _phase_tag(v) if v >= 0 else "unknown",
        )
    )
    sections.append("")
    sections.append(
        _render_behavior_table(
            "2b. Behavior by score_outcome",
            "Do we attack more when behind, defend more when ahead?",
            _behavior_by_context(
                rows, lambda r: (r.get("score_outcome") or "").strip() or None
            ),
            lambda v: str(v),
        )
    )
    sections.append("")
    sections.append(
        _render_behavior_table(
            "2c. Behavior by opponent",
            "Per-opponent behavioral fingerprint. Useful to inspect "
            "alongside `[Z Slices]` opp_wr_spread -- if behavior "
            "differs across opponents but z does not, the actor is "
            "doing the specialization that z was supposed to do.",
            _behavior_by_context(
                rows, lambda r: _to_int(r.get("opponent_id"))
            ),
            lambda v: _opponent_tag(v) if v >= 0 else "unknown",
        )
    )
    sections.append("")

    # ----- Section 3: router preference per context -----
    sections.extend([
        "---",
        "",
        "## Section 3: Router preference per context (P(z | context))",
        "",
        "Does `q_phi` execute different z distributions in different "
        "regimes? Uniform_kl close to zero everywhere = the router is "
        "context-blind. Strongly skewed distributions in some contexts "
        "= the router IS using context, but check Section 2 to see if "
        "that translates into actually-different behavior.",
        "",
    ])
    sections.append(
        _render_z_routing_table(
            "3a. P(z | team_phase)",
            "If `q_phi` is summer-faithful and the global state is "
            "informative, we expect P(z|phase) to vary across phases.",
            _z_by_context(
                rows,
                lambda r: _to_int(r.get("phase_id")) if _to_int(r.get("phase_id")) >= 0 else None,
                args.latent_k,
            ),
            lambda v: _phase_tag(v),
            args.latent_k,
        )
    )
    sections.append("")
    sections.append(
        _render_z_routing_table(
            "3b. P(z | opponent)",
            "Sanity check against MI(z; opponent). Strong skew here = "
            "router is doing opponent-conditional routing.",
            _z_by_context(
                rows,
                lambda r: _to_int(r.get("opponent_id")) if _to_int(r.get("opponent_id")) >= 0 else None,
                args.latent_k,
            ),
            lambda v: _opponent_tag(v),
            args.latent_k,
        )
    )
    sections.append("")
    sections.append(
        _render_z_routing_table(
            "3c. P(z | score_outcome)",
            "Does the router pick different z when behind vs ahead?",
            _z_by_context(
                rows,
                lambda r: (r.get("score_outcome") or "").strip() or None,
                args.latent_k,
            ),
            lambda v: str(v),
            args.latent_k,
        )
    )
    sections.append("")

    # ----- Section 4: consequence proxy -----
    sections.extend([
        "---",
        "",
        "## Section 4: Consequence proxies",
        "",
        f"For each diagnostic context+behavior split, distribution of "
        f"`phase_id` ``lookahead = {args.lookahead}`` decision steps "
        "later. Lower probability of bad phases (4: enemy carrying our "
        "flag) after the bucket = that behavior has measurable utility "
        "in that context. Uniform distributions across buckets = "
        "behavior choice does not consequentially matter (or the "
        "lookahead is wrong).",
        "",
    ])
    sections.append(
        _render_transition_table(
            "4a. From phase 3 (defending_own_flag): does more defenders help?",
            "If `num_defenders=high` shows lower P(t+L=4 enemy_carrying_our_flag) "
            "than `low`, defense behavior actually prevents flag steals.",
            _phase_transition_lookahead(
                rows, args.lookahead, n_defenders_bucket, from_phase=3
            ),
            args.lookahead,
        )
    )
    sections.append("")
    sections.append(
        _render_transition_table(
            "4b. From phase 2 (carrying_flag_home): does escort help capture?",
            "If `carrier_escort_count=high` shows higher P(t+L=0 neutral) "
            "than `none`, escorts actually convert carries into captures "
            "(neutral is the post-capture reset state).",
            _phase_transition_lookahead(
                rows, args.lookahead, carrier_escort_bucket, from_phase=2
            ),
            args.lookahead,
        )
    )
    sections.append("")
    sections.append(
        _render_transition_table(
            "4c. From phase 4 (enemy_carrying_our_flag): does more attackers help recover?",
            "From the worst regime: do we get back to safe phases when "
            "the team commits more attackers? If high-attackers shows "
            "higher P(t+L=0) than low-attackers, attacking out of a "
            "steal IS the right move.",
            _phase_transition_lookahead(
                rows, args.lookahead, n_attackers_bucket, from_phase=4
            ),
            args.lookahead,
        )
    )
    sections.append("")

    # Per-opponent WR from episodes.csv -- 4d
    per_opp = _per_opponent_wr(eps)
    sections.extend([
        "### 4d. Per-opponent terminal win-rate (from episodes.csv)",
        "",
        "From all completed episodes in the current run (NOT only the "
        f"e3 window). Pair this with Section 2c to see whether different "
        "behaviors against different opponents correlate with WR shifts.",
        "",
        "| opponent | n | WR | win_margin_mean |",
        "|---|---|---|---|",
    ])
    if per_opp:
        for opp_id in sorted(per_opp.keys()):
            d = per_opp[opp_id]
            sections.append(
                f"| `{_opponent_tag(opp_id)}` | {int(d['n'])} | "
                f"{d['wr']:.3f} | {d['win_margin_mean']:.3f} |"
            )
    else:
        sections.append("| - | - | - | - |")
    sections.append("")

    sections.extend([
        "---",
        "",
        "## How to read this report",
        "",
        f"- **Section 1**: any context whose median run-length crosses "
        f"the `vs_{args.resample_every_n}` boundary changes the design "
        "claim of v4i3. If `enemy_carrying_our_flag` lasts 5 decision "
        f"steps median, a 64-step resample is far too slow to switch "
        "to a defensive z when it matters; if `carrying_flag_home` "
        "lasts 150+ steps, the router will interrupt escort strategies "
        "before they finish.",
        "- **Section 2** vs **Section 3**: if behavior differs across "
        "contexts (Section 2 has large cross-context spread) but P(z|"
        "context) is uniform (Section 3 uniform_kl is small), then the "
        "actor is doing the specialization that z was supposed to do "
        "-- the latent is decorative because the local policy already "
        "absorbed the regime structure.",
        "- **Section 4**: if behavior buckets show no transition-rate "
        "difference, the policy can't get more reward by specializing "
        "more, and we should NOT punish v4i3 for failing to discover "
        "regimes that have no consequence.",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"[tactical] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
