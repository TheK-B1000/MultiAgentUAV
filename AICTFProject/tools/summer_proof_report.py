"""v4i3 Summer-Faithful Proof Suite -- combined report.

Assembles the five gates of the Summer Proof spec from the artifacts
produced by the v4i3 training run, the no-latent baseline, the fixed-z
q_probe, and the local counterfactual probe. Emits a single Markdown
report with pass/fail verdicts per gate and an overall verdict.

Gate 1: Wiring and non-collapse  (training metrics CSV of v4i3 latent run)
Gate 2: Forced-z behavior consequence  (q_probe summary CSV)
Gate 3: True local Q(s, z) consequence  (local-CF summary CSV)
Gate 4: Natural q_phi routing  (in-trainer MI metrics + local-CF best_z entropy)
Gate 5: Utility vs no-latent baseline  (latent metrics vs baseline metrics)

This tool is purely **read-only**: it does not run any rollouts, train,
or modify checkpoints. All it does is read the artifacts you already
have on disk, compute thresholds, and write a Markdown summary.

Usage (after both training runs and both probes have finished)::

    .\\.venv\\Scripts\\python.exe tools/summer_proof_report.py \\
        --latent-run-tag v4i3_summer_proof_OP5_OP6_OP7_4v4 \\
        --baseline-run-tag v4i3_no_latent_baseline_OP5_OP6_OP7_4v4 \\
        --checkpoint-dir checkpoints/4v4 \\
        --qprobe-dir checkpoints/4v4/v4i3_qprobe \\
        --local-cf-dir checkpoints/4v4/v4i3_local_cf_32 \\
        --out checkpoints/4v4/v4i3_summer_proof_report.md

You can omit any of the four artifact directories if that artifact has
not been produced yet -- the corresponding gate will be marked as
``UNKNOWN: <reason>`` and skipped in the overall verdict.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Gate plumbing
# ---------------------------------------------------------------------------


@dataclass
class _GateResult:
    name: str
    status: str  # "PASS", "FAIL", "UNKNOWN"
    summary_lines: list[str] = field(default_factory=list)
    sub_pass: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (label, status, detail)

    @property
    def is_known(self) -> bool:
        return self.status in ("PASS", "FAIL")


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


# ---------------------------------------------------------------------------
# CSV reading helpers
# ---------------------------------------------------------------------------


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    out: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out.append(dict(row))
    return out


def _to_float(s: Any) -> float | None:
    try:
        if s is None:
            return None
        s2 = str(s).strip()
        if not s2:
            return None
        v = float(s2)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _tail_mean(values: list[float], tail_frac: float = 0.10) -> float | None:
    """Mean of the last ``tail_frac`` fraction of finite values."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return None
    n_tail = max(1, int(round(tail_frac * len(finite))))
    return float(statistics.fmean(finite[-n_tail:]))


def _column(rows: list[dict[str, str]], key: str) -> list[float]:
    out: list[float] = []
    for r in rows:
        v = _to_float(r.get(key))
        if v is not None:
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------


def _find_metrics_csv(checkpoint_dir: Path, run_tag: str) -> Path | None:
    """Locate the trainer metrics CSV for a run tag.

    The trainer writes ``<run_tag>_metrics.csv`` directly into
    ``--checkpoint-dir``. Some legacy runs lived at the repo root, so we
    also look in ``checkpoint_dir.parent`` as a fallback.
    """
    for candidate in (
        checkpoint_dir / f"{run_tag}_metrics.csv",
        checkpoint_dir.parent / f"{run_tag}_metrics.csv",
    ):
        if candidate.exists():
            return candidate
    return None


def _find_episodes_csv(checkpoint_dir: Path, run_tag: str) -> Path | None:
    for candidate in (
        checkpoint_dir / f"{run_tag}_episodes.csv",
        checkpoint_dir.parent / f"{run_tag}_episodes.csv",
    ):
        if candidate.exists():
            return candidate
    return None


def _find_qprobe_summary(qprobe_dir: Path, latent_run_tag: str) -> Path | None:
    """Locate the q_probe summary CSV.

    ``q_probe.py`` writes ``<run_tag>_qprobe_summary.csv`` (note: the
    ``run_tag`` here is the q_probe run tag, which often equals the
    training run tag). Try a few naming conventions.
    """
    candidates = [
        qprobe_dir / f"{latent_run_tag}_qprobe_summary.csv",
        qprobe_dir / "qprobe_summary.csv",
    ]
    # Also pick up any *_qprobe_summary.csv if only one exists.
    if not any(c.exists() for c in candidates):
        any_match = sorted(qprobe_dir.glob("*_qprobe_summary.csv"))
        if any_match:
            return any_match[0]
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_local_cf_summary(
    local_cf_dir: Path, latent_run_tag: str
) -> Path | None:
    candidates = [
        local_cf_dir / f"{latent_run_tag}_qprobe_local_cf_summary.csv",
        local_cf_dir / "qprobe_local_cf_summary.csv",
    ]
    if not any(c.exists() for c in candidates):
        any_match = sorted(local_cf_dir.glob("*_qprobe_local_cf_summary.csv"))
        if any_match:
            return any_match[0]
    for c in candidates:
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Gate 1: Wiring + non-collapse (latent training metrics)
# ---------------------------------------------------------------------------


def _gate_1_wiring(latent_metrics_path: Path | None) -> _GateResult:
    g = _GateResult(name="Gate 1: Wiring + non-collapse", status="UNKNOWN")
    if latent_metrics_path is None:
        g.summary_lines.append(
            "Latent training metrics CSV not found. Pass --checkpoint-dir "
            "containing `<latent_run_tag>_metrics.csv`."
        )
        return g
    rows = _read_csv_rows(latent_metrics_path)
    if not rows:
        g.summary_lines.append(
            f"Latent metrics CSV is empty: {latent_metrics_path}"
        )
        return g
    g.summary_lines.append(
        f"Latent metrics CSV: `{latent_metrics_path.name}` ({len(rows)} updates)"
    )
    arc_len_tail = _tail_mean(_column(rows, "latent_arc_mean_length"))
    arc_count_tail = _tail_mean(_column(rows, "latent_arc_count"))
    arc_drop_tail = _tail_mean(_column(rows, "latent_arc_dropped_short_count"))
    qphi_grad_tail = _tail_mean(_column(rows, "q_phi_grad_norm"))
    z_ent_tail = _tail_mean(
        _column(rows, "latent_z_marginal_entropy_nats")
    )
    switch_tail = _tail_mean(_column(rows, "strategy_switch_fraction"))

    drop_frac = None
    if arc_drop_tail is not None and arc_count_tail not in (None, 0.0):
        drop_frac = arc_drop_tail / max(arc_count_tail, 1e-8)

    ln_k = math.log(4)  # v4i3 uses latent_k=4
    sub: list[tuple[bool, str, str]] = [
        (
            (arc_len_tail or 0.0) > 30.0,
            "arc length not collapsed to <30 steps",
            f"latent_arc_mean_length tail = {arc_len_tail!r}",
        ),
        (
            drop_frac is not None and drop_frac < 0.50,
            "dropped arcs not dominant (<50%)",
            f"drop_frac tail = {drop_frac!r}",
        ),
        (
            (qphi_grad_tail or 0.0) > 1e-6,
            "q_phi gradient signal is nonzero",
            f"q_phi_grad_norm tail = {qphi_grad_tail!r}",
        ),
        (
            (z_ent_tail or 0.0) > 0.30,
            "z marginal entropy not collapsed (>0.30 nats; ln(4)~1.39)",
            f"latent_z_marginal_entropy_nats tail = {z_ent_tail!r}",
        ),
        (
            switch_tail is not None and switch_tail < 0.85,
            "switch fraction not pathological (<0.85)",
            f"strategy_switch_fraction tail = {switch_tail!r}",
        ),
    ]
    overall = all(b for b, _, _ in sub)
    g.status = _verdict(overall)
    for passed, label, detail in sub:
        g.sub_pass.append((label, _verdict(passed), detail))
    return g


# ---------------------------------------------------------------------------
# Gate 2: Forced-z behavior consequence  (q_probe summary)
# ---------------------------------------------------------------------------


def _gate_2_forced_z(
    qprobe_summary_path: Path | None,
    *,
    threshold: float,
) -> _GateResult:
    g = _GateResult(name="Gate 2: Forced-z behavior consequence", status="UNKNOWN")
    if qprobe_summary_path is None:
        g.summary_lines.append(
            "q_probe summary CSV not found. Run `tools/q_probe.py` first."
        )
        return g
    rows = _read_csv_rows(qprobe_summary_path)
    if not rows:
        g.summary_lines.append(f"q_probe summary is empty: {qprobe_summary_path}")
        return g
    g.summary_lines.append(
        f"q_probe summary: `{qprobe_summary_path.name}` ({len(rows)} rows)"
    )
    # The q_probe summary writes "paired_return_contrast" per (checkpoint, opponent)
    # plus per-opponent / overall fields. We try common column names; if absent
    # we degrade to whichever variants exist.
    per_opp_contrast: dict[str, float] = {}
    per_opp_winrate_contrast: dict[str, float] = {}
    for r in rows:
        opp = str(r.get("opponent", "")).strip().upper()
        if not opp:
            continue
        candidates = (
            "paired_return_contrast",
            "return_contrast",
            "paired_return_contrast_overall",
        )
        v = None
        for k in candidates:
            v = _to_float(r.get(k))
            if v is not None:
                break
        if v is not None:
            # Keep the LAST row per opponent (assume CSV is checkpoint-ordered
            # and the final row is the latest checkpoint).
            per_opp_contrast[opp] = v
        wr = None
        for k in (
            "paired_winrate_contrast",
            "winrate_contrast",
            "winrate_contrast_overall",
            "blue_won_contrast",
        ):
            wr = _to_float(r.get(k))
            if wr is not None:
                break
        if wr is not None:
            per_opp_winrate_contrast[opp] = wr

    if not per_opp_contrast:
        g.summary_lines.append(
            "Could not locate per-opponent paired return contrast columns "
            "in the q_probe summary. Re-run q_probe with the v4i1 reporting "
            "schema, or update this tool to recognise the column names."
        )
        return g
    overall = statistics.fmean(per_opp_contrast.values())
    worst = min(per_opp_contrast.values())
    sub: list[tuple[bool, str, str]] = [
        (
            overall > float(threshold),
            f"overall paired return contrast > {threshold:.2f}",
            f"overall = {overall:+.4f}",
        ),
        (
            worst > float(threshold),
            f"worst-opponent paired return contrast > {threshold:.2f}",
            f"worst = {worst:+.4f}",
        ),
        (
            bool(per_opp_winrate_contrast)
            and any(v > 0.05 for v in per_opp_winrate_contrast.values()),
            "win-rate contrast > 5% on at least one opponent",
            f"per_opp_winrate_contrast = {per_opp_winrate_contrast!r}",
        ),
    ]
    g.summary_lines.append(
        "Per-opponent paired return contrast (last checkpoint): "
        + ", ".join(f"{k}={v:+.3f}" for k, v in sorted(per_opp_contrast.items()))
    )
    overall_pass = all(b for b, _, _ in sub)
    g.status = _verdict(overall_pass)
    for passed, label, detail in sub:
        g.sub_pass.append((label, _verdict(passed), detail))
    return g


# ---------------------------------------------------------------------------
# Gate 3: True local Q(s, z) consequence (local CF summary)
# ---------------------------------------------------------------------------


def _gate_3_local_cf(
    local_cf_summary_path: Path | None,
    *,
    threshold: float,
) -> _GateResult:
    g = _GateResult(name="Gate 3: True local Q(s, z) consequence", status="UNKNOWN")
    if local_cf_summary_path is None:
        g.summary_lines.append(
            "Local counterfactual summary CSV not found. Run "
            "`tools/q_probe_local_counterfactual.py` first."
        )
        return g
    rows = _read_csv_rows(local_cf_summary_path)
    if not rows:
        g.summary_lines.append(
            f"Local-CF summary is empty: {local_cf_summary_path}"
        )
        return g
    g.summary_lines.append(
        f"Local CF summary: `{local_cf_summary_path.name}` ({len(rows)} scenes)"
    )
    by_opp: dict[str, list[float]] = {}
    best_z_by_opp: dict[str, list[int]] = {}
    for r in rows:
        opp = str(r.get("opponent", "")).strip().upper()
        q = _to_float(r.get("Q_contrast"))
        if q is None:
            continue
        try:
            bz = int(float(r.get("best_z", 0) or 0))
        except (TypeError, ValueError):
            bz = 0
        by_opp.setdefault(opp, []).append(q)
        best_z_by_opp.setdefault(opp, []).append(bz)
    if not by_opp:
        g.summary_lines.append(
            "No usable Q_contrast values in the local-CF summary."
        )
        return g
    opp_means = {o: statistics.fmean(v) for o, v in by_opp.items()}
    overall = statistics.fmean(opp_means.values())
    worst = min(opp_means.values())
    ent_by_opp: dict[str, float] = {}
    max_share_by_opp: dict[str, float] = {}
    for opp, bzs in best_z_by_opp.items():
        counts: dict[int, int] = {}
        for bz in bzs:
            counts[bz] = counts.get(bz, 0) + 1
        n = sum(counts.values())
        ent = 0.0
        for c in counts.values():
            p = c / n
            if p > 0:
                ent -= p * math.log(p)
        ent_by_opp[opp] = ent
        max_share_by_opp[opp] = max(counts.values()) / n if n > 0 else 0.0
    g.summary_lines.append(
        "Per-opponent mean Q-contrast (local CF): "
        + ", ".join(f"{k}={v:+.3f}" for k, v in sorted(opp_means.items()))
    )
    g.summary_lines.append(
        "Per-opponent best_z entropy (nats): "
        + ", ".join(f"{k}={v:.3f}" for k, v in sorted(ent_by_opp.items()))
    )
    sub: list[tuple[bool, str, str]] = [
        (
            overall > float(threshold),
            f"overall mean Q-contrast > {threshold:.2f}",
            f"overall = {overall:+.4f}",
        ),
        (
            worst > float(threshold),
            f"worst-opponent mean Q-contrast > {threshold:.2f}",
            f"worst = {worst:+.4f}",
        ),
        (
            all(v > 0.0 for v in ent_by_opp.values()),
            "best_z entropy > 0 on every opponent",
            f"ent_by_opp = {ent_by_opp!r}",
        ),
        (
            all(v < 0.95 for v in max_share_by_opp.values()),
            "best_z varies (no single z dominates >=95% per opp)",
            f"max_share_by_opp = {max_share_by_opp!r}",
        ),
    ]
    overall_pass = all(b for b, _, _ in sub)
    g.status = _verdict(overall_pass)
    for passed, label, detail in sub:
        g.sub_pass.append((label, _verdict(passed), detail))
    return g


# ---------------------------------------------------------------------------
# Gate 4: Natural q_phi routing
# ---------------------------------------------------------------------------


def _gate_4_routing(
    latent_metrics_path: Path | None,
    local_cf_summary_path: Path | None,
) -> _GateResult:
    g = _GateResult(name="Gate 4: Natural q_phi routing", status="UNKNOWN")
    summary_bits: list[str] = []
    if latent_metrics_path is None:
        g.summary_lines.append(
            "Latent training metrics CSV not found. Gate 4 needs the "
            "in-trainer MI metrics to evaluate."
        )
        return g
    rows = _read_csv_rows(latent_metrics_path)
    if not rows:
        g.summary_lines.append(
            f"Latent metrics CSV is empty: {latent_metrics_path}"
        )
        return g
    g.summary_lines.append(
        f"Latent metrics CSV: `{latent_metrics_path.name}` ({len(rows)} updates)"
    )

    # MI floors per the v3i19 plan (and our explore findings):
    #   normalized_MI(z; opponent) > 0.02
    #   normalized_MI(z; phase)    > 0.01
    #   normalized_MI(z; flag)     > 0.02
    # The trainer emits these as ``latent_normalized_mi_z_{opponent,phase,
    # flag_state}`` (when ``latent_diagnostics`` is enabled). Missing columns
    # are reported in the summary lines but do not contribute a sub-check.
    def _tail(col: str) -> float | None:
        return _tail_mean(_column(rows, col))

    mi_opp = _tail("latent_normalized_mi_z_opponent")
    mi_phase = _tail("latent_normalized_mi_z_phase")
    mi_flag = _tail("latent_normalized_mi_z_flag_state")
    mi_outcome = _tail("latent_normalized_mi_z_outcome")
    strat_ent = _tail("strategy_entropy")
    strat_ent_frac = _tail("strategy_entropy_frac")

    sub: list[tuple[bool, str, str]] = []
    missing_cols: list[str] = []
    if mi_opp is not None:
        sub.append(
            (
                mi_opp > 0.02,
                "normalized MI(z; opponent) > 0.02 (v3i18 noise floor)",
                f"latent_normalized_mi_z_opponent tail = {mi_opp:.4f}",
            )
        )
    else:
        missing_cols.append("latent_normalized_mi_z_opponent")
    if mi_phase is not None:
        sub.append(
            (
                mi_phase > 0.01,
                "normalized MI(z; phase) > 0.01 (v3i18 noise floor)",
                f"latent_normalized_mi_z_phase tail = {mi_phase:.4f}",
            )
        )
    else:
        missing_cols.append("latent_normalized_mi_z_phase")
    if mi_flag is not None:
        sub.append(
            (
                mi_flag > 0.02,
                "normalized MI(z; flag) > 0.02 (v3i18 noise floor)",
                f"latent_normalized_mi_z_flag_state tail = {mi_flag:.4f}",
            )
        )
    else:
        missing_cols.append("latent_normalized_mi_z_flag_state")
    if mi_outcome is not None:
        # No hard threshold in the spec; record as informational only.
        g.summary_lines.append(
            f"latent_normalized_mi_z_outcome tail = {mi_outcome:.4f} (informational)"
        )
    if missing_cols:
        g.summary_lines.append(
            "Missing MI columns (informational, not a FAIL): "
            + ", ".join(missing_cols)
            + ". Enable latent_diagnostics in the trainer if these are needed."
        )
    if strat_ent_frac is not None:
        sub.append(
            (
                0.10 < strat_ent_frac < 0.95,
                "strategy_entropy_frac in (0.10, 0.95) (committed but not collapsed)",
                f"strategy_entropy_frac tail = {strat_ent_frac:.3f}",
            )
        )
    elif strat_ent is not None:
        ln_k = math.log(4)
        frac = strat_ent / ln_k if ln_k > 0 else 0.0
        sub.append(
            (
                0.10 < frac < 0.95,
                "strategy_entropy / ln(K) in (0.10, 0.95)",
                f"strategy_entropy tail = {strat_ent:.3f} (frac={frac:.3f})",
            )
        )

    # Local-CF best_z entropy (already computed in Gate 3, recomputed here as
    # a secondary signal for Gate 4 routing health).
    if local_cf_summary_path is not None:
        cf_rows = _read_csv_rows(local_cf_summary_path)
        best_z_all: list[int] = []
        for r in cf_rows:
            try:
                best_z_all.append(int(float(r.get("best_z", 0) or 0)))
            except (TypeError, ValueError):
                continue
        if best_z_all:
            counts: dict[int, int] = {}
            for bz in best_z_all:
                counts[bz] = counts.get(bz, 0) + 1
            n = sum(counts.values())
            ent_overall = 0.0
            for c in counts.values():
                p = c / n
                if p > 0:
                    ent_overall -= p * math.log(p)
            sub.append(
                (
                    ent_overall > 0.50,
                    "local-CF best_z entropy > 0.5 nats (z genuinely varies across states)",
                    f"local_cf_best_z_entropy = {ent_overall:.3f}",
                )
            )

    if not sub:
        g.summary_lines.append(
            "No usable Gate 4 metrics in the latent metrics CSV (no MI columns)."
        )
        return g
    overall_pass = all(b for b, _, _ in sub)
    g.status = _verdict(overall_pass)
    for passed, label, detail in sub:
        g.sub_pass.append((label, _verdict(passed), detail))
    return g


# ---------------------------------------------------------------------------
# Gate 5: Utility vs no-latent baseline
# ---------------------------------------------------------------------------


def _summarise_run_for_gate5(metrics_path: Path | None) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "win_rate_tail": None,
        "rolling_wr_200_tail": None,
        "rollout_return_tail": None,
        "rollout_win_margin_tail": None,
    }
    if metrics_path is None:
        return out
    rows = _read_csv_rows(metrics_path)
    if not rows:
        return out
    out["win_rate_tail"] = _tail_mean(_column(rows, "win_rate"))
    out["rolling_wr_200_tail"] = _tail_mean(
        _column(rows, "rolling_win_rate_200ep")
    )
    out["rollout_return_tail"] = _tail_mean(
        _column(rows, "rollout_return_mean")
    )
    out["rollout_win_margin_tail"] = _tail_mean(
        _column(rows, "rollout_win_margin_mean")
    )
    return out


def _gate_5_utility(
    latent_metrics: Path | None, baseline_metrics: Path | None
) -> _GateResult:
    g = _GateResult(
        name="Gate 5: Utility vs no-latent baseline", status="UNKNOWN"
    )
    if latent_metrics is None or baseline_metrics is None:
        g.summary_lines.append(
            "Gate 5 requires both the latent metrics CSV and the no-latent "
            "baseline metrics CSV. Run both training jobs to completion first."
        )
        return g
    lat = _summarise_run_for_gate5(latent_metrics)
    base = _summarise_run_for_gate5(baseline_metrics)
    g.summary_lines.append(
        f"Latent run:   `{latent_metrics.name}` win_rate_tail={lat['win_rate_tail']!r} "
        f"rolling_wr_200_tail={lat['rolling_wr_200_tail']!r} "
        f"rollout_return_tail={lat['rollout_return_tail']!r}"
    )
    g.summary_lines.append(
        f"No-latent run: `{baseline_metrics.name}` win_rate_tail={base['win_rate_tail']!r} "
        f"rolling_wr_200_tail={base['rolling_wr_200_tail']!r} "
        f"rollout_return_tail={base['rollout_return_tail']!r}"
    )
    # Beat on at least one of: win rate, mean return, win margin.
    # We require strictly greater than baseline on at least one signal
    # (tolerance 0 to keep the bar honest; a tie is not a win).
    sub: list[tuple[bool, str, str]] = []
    wr_lat = lat["rolling_wr_200_tail"] or lat["win_rate_tail"]
    wr_base = base["rolling_wr_200_tail"] or base["win_rate_tail"]
    if wr_lat is not None and wr_base is not None:
        sub.append(
            (
                wr_lat > wr_base,
                "win rate (latent > baseline)",
                f"{wr_lat:.4f} vs {wr_base:.4f} (delta {wr_lat - wr_base:+.4f})",
            )
        )
    if lat["rollout_return_tail"] is not None and base["rollout_return_tail"] is not None:
        sub.append(
            (
                lat["rollout_return_tail"] > base["rollout_return_tail"],
                "rollout return mean (latent > baseline)",
                f"{lat['rollout_return_tail']:+.4f} vs "
                f"{base['rollout_return_tail']:+.4f}",
            )
        )
    if lat["rollout_win_margin_tail"] is not None and base["rollout_win_margin_tail"] is not None:
        sub.append(
            (
                lat["rollout_win_margin_tail"] > base["rollout_win_margin_tail"],
                "rollout win-margin mean (latent > baseline)",
                f"{lat['rollout_win_margin_tail']:+.4f} vs "
                f"{base['rollout_win_margin_tail']:+.4f}",
            )
        )
    if not sub:
        g.summary_lines.append(
            "Neither metrics CSV had any of (win_rate, rolling_win_rate_200ep, "
            "rollout_return_mean, rollout_win_margin_mean). Gate 5 unknown."
        )
        return g
    any_wins = any(b for b, _, _ in sub)
    g.status = _verdict(any_wins)
    g.summary_lines.append(
        f"Pass requires latent to strictly beat baseline on at least ONE of "
        f"the {len(sub)} compared signals."
    )
    for passed, label, detail in sub:
        g.sub_pass.append((label, _verdict(passed), detail))
    return g


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _render_report(
    *,
    args: argparse.Namespace,
    artifact_paths: dict[str, Path | None],
    gates: list[_GateResult],
) -> str:
    lines: list[str] = []
    lines.append("# v4i3 Summer-Faithful Proof Suite Report")
    lines.append("")
    lines.append(
        "Generated by `tools/summer_proof_report.py`. Inputs are read-only; "
        "this report does not modify checkpoints, training data, or probe outputs."
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- latent run tag:   `{args.latent_run_tag}`")
    lines.append(f"- baseline run tag: `{args.baseline_run_tag or '(none provided)'}`")
    for k, v in artifact_paths.items():
        lines.append(f"- {k}: `{v if v is not None else '(missing)'}`")
    lines.append("")
    n_pass = sum(1 for g in gates if g.status == "PASS")
    n_fail = sum(1 for g in gates if g.status == "FAIL")
    n_unknown = sum(1 for g in gates if g.status == "UNKNOWN")
    if n_unknown == 0:
        overall = "PASS" if n_fail == 0 else "FAIL"
    else:
        overall = "PARTIAL" if n_fail == 0 else "FAIL (with UNKNOWN gates)"
    lines.append("## Overall verdict")
    lines.append("")
    lines.append(
        f"- gates evaluated: PASS={n_pass}, FAIL={n_fail}, UNKNOWN={n_unknown}"
    )
    lines.append(f"- **overall: {overall}**")
    lines.append("")
    if overall.startswith("FAIL"):
        lines.append(
            "If Gates 1-3 pass but Gate 4 fails, the actor expresses useful "
            "latent modes under forced z but pure end-to-end q_phi does NOT "
            "reliably learn to route to them. The honest follow-up is the "
            "post-Summer extension `latent_v4i4post_periodic_router_distill` "
            "(counterfactual router refinement). Frame it explicitly: "
            "\"we first evaluate the fully autonomous Summer method, then "
            "introduce a counterfactual router refinement that uses "
            "task-return-ranked latent interventions without human strategy "
            "labels.\""
        )
        lines.append("")
    for g in gates:
        lines.append(f"## {g.name}")
        lines.append("")
        lines.append(f"- **status: {g.status}**")
        for ln in g.summary_lines:
            lines.append(f"- {ln}")
        if g.sub_pass:
            lines.append("")
            lines.append("| sub-check | status | detail |")
            lines.append("|---|---|---|")
            for label, status, detail in g.sub_pass:
                # Escape pipes in detail for markdown table compatibility.
                safe_detail = detail.replace("|", "\\|")
                lines.append(f"| {label} | {status} | {safe_detail} |")
        lines.append("")
    lines.append("## Notes on the Summer Proof contract")
    lines.append("")
    lines.append(
        "- The Summer plan says z must be learned end-to-end from reward "
        "alone, with sparse refresh + persistence + entropy regularisation "
        "and no labels / aux heads. v4i3 is the experiment that tests "
        "whether this is sufficient for q_phi to route to useful modes."
    )
    lines.append(
        "- Gate 2 measures *consequence of forced z* (does the actor "
        "express different strategies?); Gate 3 measures *local Q(s, z) "
        "contrast at the exact decision points*; Gate 4 measures *whether "
        "q_phi routes to them naturally*. Passing 2+3 but failing 4 is the "
        "classic 'modes exist, router doesn't use them' diagnosis."
    )
    lines.append(
        "- Gate 5 is the only utility-vs-control gate. A win requires "
        "strictly beating the no-latent baseline on at least one signal."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="v4i3 Summer-Faithful Proof Suite report assembler."
    )
    p.add_argument(
        "--latent-run-tag",
        type=str,
        required=True,
        help="Run tag of the v4i3 latent training run (e.g. "
        "v4i3_summer_proof_OP5_OP6_OP7_4v4).",
    )
    p.add_argument(
        "--baseline-run-tag",
        type=str,
        default=None,
        help="Run tag of the no-latent baseline training run.",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/4v4"),
        help="Directory holding the trainer's metrics CSVs (default: checkpoints/4v4).",
    )
    p.add_argument(
        "--qprobe-dir",
        type=Path,
        default=None,
        help="Directory holding the fixed-z q_probe outputs.",
    )
    p.add_argument(
        "--local-cf-dir",
        type=Path,
        default=None,
        help="Directory holding the local counterfactual probe outputs.",
    )
    p.add_argument(
        "--gate2-threshold",
        type=float,
        default=0.10,
        help="Forced-z return contrast threshold for Gate 2 (default 0.10).",
    )
    p.add_argument(
        "--gate3-threshold",
        type=float,
        default=0.10,
        help="Local Q-contrast threshold for Gate 3 (default 0.10).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Where to write the assembled Markdown report.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    latent_metrics = _find_metrics_csv(
        Path(args.checkpoint_dir), str(args.latent_run_tag)
    )
    baseline_metrics = (
        _find_metrics_csv(Path(args.checkpoint_dir), str(args.baseline_run_tag))
        if args.baseline_run_tag
        else None
    )
    qprobe_summary = (
        _find_qprobe_summary(
            Path(args.qprobe_dir), str(args.latent_run_tag)
        )
        if args.qprobe_dir is not None
        else None
    )
    local_cf_summary = (
        _find_local_cf_summary(
            Path(args.local_cf_dir), str(args.latent_run_tag)
        )
        if args.local_cf_dir is not None
        else None
    )

    artifact_paths: dict[str, Path | None] = {
        "latent metrics CSV": latent_metrics,
        "baseline metrics CSV": baseline_metrics,
        "q_probe summary CSV": qprobe_summary,
        "local-CF summary CSV": local_cf_summary,
    }
    for k, v in artifact_paths.items():
        print(f"[summer_proof] {k}: {v if v is not None else '(missing)'}")

    gates = [
        _gate_1_wiring(latent_metrics),
        _gate_2_forced_z(qprobe_summary, threshold=float(args.gate2_threshold)),
        _gate_3_local_cf(local_cf_summary, threshold=float(args.gate3_threshold)),
        _gate_4_routing(latent_metrics, local_cf_summary),
        _gate_5_utility(latent_metrics, baseline_metrics),
    ]
    report_md = _render_report(
        args=args, artifact_paths=artifact_paths, gates=gates
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report_md, encoding="utf-8")

    n_pass = sum(1 for g in gates if g.status == "PASS")
    n_fail = sum(1 for g in gates if g.status == "FAIL")
    n_unknown = sum(1 for g in gates if g.status == "UNKNOWN")
    print(
        f"[summer_proof] gates: PASS={n_pass} FAIL={n_fail} UNKNOWN={n_unknown}"
    )
    print(f"[summer_proof] wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
