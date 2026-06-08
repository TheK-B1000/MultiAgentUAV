"""Post-hoc latent strategy diagnostics for v3i-style PPO runs.

This is the **analysis-only** companion to the in-trainer telemetry. It is
explicitly Summer-faithful in two ways:

1. It introduces *no* training-time supervision against phase / flag / outcome
   labels. Everything here is computed from already-saved CSVs.
2. It interprets ``z`` *post hoc* via per-z behavior fingerprints
   (spread, attacker count, distance to flags, intercept pressure, ...) and
   dwell-weighted segment-level MI, rather than per-step plug-in MI which is
   noisy when a single ``z`` spans dozens of micro-states (e.g. v3i16's 64-step
   persistence).

Usage
-----
::

    python tools/analyze_latent.py checkpoints/4v4/<run_tag>
    # or point at any of the run's CSVs:
    python tools/analyze_latent.py checkpoints/4v4/<run_tag>_e3_steps.csv

Outputs (all written next to the run, prefixed ``<run_tag>_analysis_``):

* ``behavior_by_z.csv``  — per-z behavior_telemetry means / counts / dwell share
* ``segments_summary.csv`` — global segment stats and dwell-by-z
* ``segment_mi.csv``  — dwell-weighted + uniform MI(z_seg ; phase/flag/outcome/opp)
                       plus normalized variants and segment-level H(z)
* ``z_wr_by_opponent.csv`` — episodes.csv groupby (latent_z, opponent_id)
* ``z_phase_segments.csv`` — (z, dominant_phase) segment count + dwell
* ``z_flag_segments.csv``  — (z, flag_state_at_start) segment count + dwell
* ``qphi_audit.csv``  — per-column stats of the 170-dim q_phi context, plus
                       which strategic global-state slots are actually populated
* ``report.md``  — human-readable executive summary
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

# Make ``rl.*`` importable when this script is run as ``python tools/analyze_latent.py``
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES
from rl.global_state import GLOBAL_STATE_DIM, GLOBAL_STATE_FIELD_NAMES
from rl.latent_phase_labels import OUTCOME_CLASSES, TEAM_PHASES

DEFAULT_CHUNK_ROWS = 50_000
TEMPORAL_FRAMES = 5  # q_phi_context is a 5-frame stack of GLOBAL_STATE_DIM=34 features.


# ---------------------------------------------------------------------------
# CLI / path resolution
# ---------------------------------------------------------------------------


def _resolve_run_paths(arg: str) -> dict[str, Path]:
    """Map a user-provided run identifier to the three expected CSV paths.

    Accepts a directory + tag prefix, a full ``*_metrics.csv`` path, or a full
    ``*_e3_steps.csv`` path.
    """
    path = Path(arg).expanduser()
    # Identify the "tag stem" by stripping any known suffix.
    suffixes = ["_e3_steps.csv", "_episodes.csv", "_metrics.csv"]
    if path.is_dir():
        e3_candidates = sorted(path.glob("*_e3_steps.csv"))
        if not e3_candidates:
            raise FileNotFoundError(f"No *_e3_steps.csv in directory: {path}")
        stem_path = e3_candidates[0]
        for sfx in suffixes:
            if str(stem_path).endswith(sfx):
                stem = str(stem_path)[: -len(sfx)]
                break
        else:
            raise RuntimeError("Unexpected file stem in directory.")
    else:
        s = str(path)
        for sfx in suffixes:
            if s.endswith(sfx):
                stem = s[: -len(sfx)]
                break
        else:
            stem = s
    stem_path = Path(stem)
    return {
        "stem": stem_path,
        "e3": Path(f"{stem}_e3_steps.csv"),
        "episodes": Path(f"{stem}_episodes.csv"),
        "metrics": Path(f"{stem}_metrics.csv"),
    }


# ---------------------------------------------------------------------------
# Streaming accumulators
# ---------------------------------------------------------------------------


class WelfordPerZ:
    """Online per-z mean (Welford-friendly) and count for a fixed metric set."""

    __slots__ = ("K", "names", "count", "mean", "m2")

    def __init__(self, K: int, names: tuple[str, ...]) -> None:
        self.K = int(K)
        self.names = tuple(names)
        n = len(self.names)
        self.count = np.zeros(self.K, dtype=np.int64)
        self.mean = np.zeros((self.K, n), dtype=np.float64)
        self.m2 = np.zeros((self.K, n), dtype=np.float64)

    def update_block(self, z_rows: np.ndarray, value_block: np.ndarray) -> None:
        """Update per-z accumulators from a chunk's (rows, n_names) values."""
        if value_block.shape[0] == 0:
            return
        for k in range(self.K):
            mask = z_rows == k
            n_new = int(mask.sum())
            if n_new == 0:
                continue
            x = value_block[mask].astype(np.float64)
            # Chan-Welford parallel update against the running (count, mean, m2).
            mean_new = x.mean(axis=0)
            m2_new = ((x - mean_new) ** 2).sum(axis=0)
            n_old = int(self.count[k])
            n_tot = n_old + n_new
            delta = mean_new - self.mean[k]
            self.mean[k] = self.mean[k] + delta * (n_new / n_tot)
            self.m2[k] = self.m2[k] + m2_new + (delta ** 2) * (n_old * n_new / n_tot)
            self.count[k] = n_tot


class ColumnStats:
    """Per-column streaming mean / m2 / min / max / zero-count for q_phi audit."""

    __slots__ = ("names", "count", "mean", "m2", "vmin", "vmax", "zero_count")

    def __init__(self, names: list[str]) -> None:
        self.names = list(names)
        n = len(self.names)
        self.count = 0
        self.mean = np.zeros(n, dtype=np.float64)
        self.m2 = np.zeros(n, dtype=np.float64)
        self.vmin = np.full(n, np.inf, dtype=np.float64)
        self.vmax = np.full(n, -np.inf, dtype=np.float64)
        self.zero_count = np.zeros(n, dtype=np.int64)

    def update_block(self, values: np.ndarray) -> None:
        if values.shape[0] == 0:
            return
        n_new = values.shape[0]
        x = values.astype(np.float64)
        mean_new = x.mean(axis=0)
        m2_new = ((x - mean_new) ** 2).sum(axis=0)
        if self.count == 0:
            self.mean = mean_new
            self.m2 = m2_new
        else:
            n_old = self.count
            n_tot = n_old + n_new
            delta = mean_new - self.mean
            self.mean = self.mean + delta * (n_new / n_tot)
            self.m2 = self.m2 + m2_new + (delta ** 2) * (n_old * n_new / n_tot)
        self.count += n_new
        self.vmin = np.minimum(self.vmin, x.min(axis=0))
        self.vmax = np.maximum(self.vmax, x.max(axis=0))
        self.zero_count += (x == 0).sum(axis=0).astype(np.int64)


# ---------------------------------------------------------------------------
# Segment builder
# ---------------------------------------------------------------------------


_OUTCOME_TO_ID = {"loss": 0, "draw": 1, "win": 2, "tied": 1, "blue_won": 2, "red_won": 0}


def _outcomes_to_ids(series: pd.Series) -> np.ndarray:
    """Map a pandas Series of outcome labels to int64 ids (unknown -> 1)."""
    return series.map(lambda s: _OUTCOME_TO_ID.get(str(s).strip().lower(), 1)).to_numpy(dtype=np.int64)


def _finalize_open(seg: dict[str, Any]) -> dict[str, Any]:
    """Convert per-env open segment (with bincount-style phase counts) to final dict."""
    phase_counts = seg.pop("phase_counts")
    seg["dominant_phase"] = int(np.argmax(phase_counts))
    seg["outcome_delta"] = int(seg["outcome_last"]) - int(seg["outcome_first"])
    seg["end_global_step"] = int(seg.pop("last_global_step"))
    return seg


def _open_from_slice(
    *, env_id: int, start_i: int, end_i: int, arrs: dict[str, np.ndarray], n_phases: int
) -> dict[str, Any]:
    """Build an "open segment" dict from arrs[start_i:end_i] (one contiguous z run)."""
    phase_counts = np.bincount(arrs["phase"][start_i:end_i], minlength=n_phases).astype(np.int64)
    return {
        "env_id": int(env_id),
        "z": int(arrs["z"][start_i]),
        "start_global_step": int(arrs["global_step"][start_i]),
        "start_update": int(arrs["update"][start_i]),
        "start_rollout_step": int(arrs["rollout_step"][start_i]),
        "dwell": int(end_i - start_i),
        "phase_counts": phase_counts,
        "flag_state_first": int(arrs["flag_state"][start_i]),
        "opponent_id_first": int(arrs["opponent_id"][start_i]),
        "outcome_first": int(arrs["outcome"][start_i]),
        "outcome_last": int(arrs["outcome"][end_i - 1]),
        "last_global_step": int(arrs["global_step"][end_i - 1]),
    }


def _extend_open(open_seg: dict[str, Any], start_i: int, end_i: int, arrs: dict[str, np.ndarray], n_phases: int) -> None:
    """Merge arrs[start_i:end_i] into an existing open segment for this env."""
    open_seg["dwell"] += int(end_i - start_i)
    open_seg["phase_counts"] = open_seg["phase_counts"] + np.bincount(
        arrs["phase"][start_i:end_i], minlength=n_phases
    ).astype(np.int64)
    open_seg["outcome_last"] = int(arrs["outcome"][end_i - 1])
    open_seg["last_global_step"] = int(arrs["global_step"][end_i - 1])


# ---------------------------------------------------------------------------
# Main streaming pass over _e3_steps.csv
# ---------------------------------------------------------------------------


def _process_env_slice(
    *,
    env_id: int,
    i0: int,
    i1: int,
    arrs: dict[str, np.ndarray],
    n_phases: int,
    open_segs: dict[int, dict[str, Any]],
    finalized: list[dict[str, Any]],
) -> None:
    """Vectorised per-env-slice segment builder.

    Detects within-slice z run-length boundaries via np.diff, decides whether
    the first run continues a previously-open segment, and pushes all complete
    segments to ``finalized``. The final run is left as the new open segment
    for this env (it may continue into the next chunk).
    """
    if i1 <= i0:
        return
    z_slice = arrs["z"][i0:i1]
    # Boundary positions (relative to slice) where z changes from previous row.
    diffs = np.flatnonzero(z_slice[1:] != z_slice[:-1]) + 1  # >=1 indices
    # Runs are [start, end) intervals within the slice.
    starts_local = np.concatenate(([0], diffs))
    ends_local = np.concatenate((diffs, [i1 - i0]))

    # Merge / close existing open segment using first run.
    first_z = int(z_slice[0])
    open_seg = open_segs.get(env_id)
    if open_seg is not None and open_seg["z"] == first_z:
        # Extend the open segment with the first run, then it may be closed
        # below if more runs follow.
        _extend_open(
            open_seg,
            i0 + starts_local[0],
            i0 + ends_local[0],
            arrs,
            n_phases,
        )
        first_run_consumed = True
    else:
        if open_seg is not None:
            finalized.append(_finalize_open(open_segs.pop(env_id)))
        first_run_consumed = False

    n_runs = len(starts_local)
    for r in range(n_runs):
        if r == 0 and first_run_consumed:
            # Already merged above.
            if n_runs > 1:
                # There's a later run in this slice, so close the merged open seg.
                finalized.append(_finalize_open(open_segs.pop(env_id)))
            else:
                # No further runs; merged seg stays open.
                pass
            continue
        seg = _open_from_slice(
            env_id=env_id,
            start_i=i0 + starts_local[r],
            end_i=i0 + ends_local[r],
            arrs=arrs,
            n_phases=n_phases,
        )
        if r < n_runs - 1:
            finalized.append(_finalize_open(seg))
        else:
            open_segs[env_id] = seg


def _stream_e3_steps(
    path: Path, K: int, chunk_rows: int
) -> tuple[
    list[dict[str, Any]],
    WelfordPerZ,
    ColumnStats,
    np.ndarray,
    np.ndarray,
    int,
]:
    behavior_acc = WelfordPerZ(K, BEHAVIOR_TELEMETRY_NAMES)
    qphi_cols = [f"q_phi_context_{i}" for i in range(TEMPORAL_FRAMES * GLOBAL_STATE_DIM)]
    qphi_acc = ColumnStats(qphi_cols)
    n_phases = len(TEAM_PHASES)
    switch_by_phase = np.zeros(n_phases, dtype=np.float64)
    rows_by_phase = np.zeros(n_phases, dtype=np.float64)

    open_segs: dict[int, dict[str, Any]] = {}
    finalized: list[dict[str, Any]] = []
    rows_seen = 0

    usecols = (
        ["update", "rollout_step", "env_id", "global_step", "z_t", "switched",
         "phase_id", "score_outcome", "opponent_id"]
        + list(BEHAVIOR_TELEMETRY_NAMES)
        + qphi_cols
    )

    reader: Iterator[pd.DataFrame] = pd.read_csv(
        path, chunksize=chunk_rows, usecols=usecols
    )
    for chunk in reader:
        # Behavior accumulators take all rows as-is (no need to sort).
        z_all = chunk["z_t"].to_numpy(dtype=np.int64)
        behavior_block = chunk[list(BEHAVIOR_TELEMETRY_NAMES)].to_numpy()
        behavior_acc.update_block(z_all, behavior_block)

        qphi_acc.update_block(chunk[qphi_cols].to_numpy())

        # Per-phase switch rates: count rows where switched==1 per phase_id.
        phase_all = chunk["phase_id"].to_numpy(dtype=np.int64)
        sw_all = chunk["switched"].to_numpy(dtype=np.float64)
        for p in range(n_phases):
            mask = phase_all == p
            rows_by_phase[p] += float(mask.sum())
            switch_by_phase[p] += float(sw_all[mask].sum())

        # Segment building requires per-env causal order; sort chunk.
        chunk_sorted = chunk.sort_values(["env_id", "update", "rollout_step"], kind="mergesort")
        env_arr = chunk_sorted["env_id"].to_numpy(dtype=np.int64)
        arrs = {
            "env_id": env_arr,
            "z": chunk_sorted["z_t"].to_numpy(dtype=np.int64),
            "switched": chunk_sorted["switched"].to_numpy(dtype=np.int64),
            "phase": chunk_sorted["phase_id"].to_numpy(dtype=np.int64),
            "opponent_id": chunk_sorted["opponent_id"].to_numpy(dtype=np.int64),
            "global_step": chunk_sorted["global_step"].to_numpy(dtype=np.int64),
            "update": chunk_sorted["update"].to_numpy(dtype=np.int64),
            "rollout_step": chunk_sorted["rollout_step"].to_numpy(dtype=np.int64),
            "outcome": _outcomes_to_ids(chunk_sorted["score_outcome"]),
            "flag_state": (
                (chunk_sorted["q_phi_context_10"].to_numpy(dtype=np.float32) > 0.5).astype(np.int64)
                + 2 * (chunk_sorted["q_phi_context_11"].to_numpy(dtype=np.float32) > 0.5).astype(np.int64)
            ),
        }

        # Per-env slice boundaries within the sorted chunk.
        env_change = np.flatnonzero(np.concatenate(([1], np.diff(env_arr) != 0)))
        env_change = np.append(env_change, len(env_arr))
        for g in range(len(env_change) - 1):
            i0, i1 = int(env_change[g]), int(env_change[g + 1])
            env_id = int(env_arr[i0])
            _process_env_slice(
                env_id=env_id, i0=i0, i1=i1, arrs=arrs, n_phases=n_phases,
                open_segs=open_segs, finalized=finalized,
            )

        rows_seen += len(chunk)
        if rows_seen % (chunk_rows * 10) == 0:
            print(f"  ... streamed {rows_seen:,} rows, finalized {len(finalized):,} segments")

    for seg in list(open_segs.values()):
        finalized.append(_finalize_open(seg))
    open_segs.clear()

    return finalized, behavior_acc, qphi_acc, switch_by_phase, rows_by_phase, rows_seen


# ---------------------------------------------------------------------------
# Post-pass: per-z, MI, cross-tabs
# ---------------------------------------------------------------------------


def _shannon_nats(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def _mi_nats(joint: np.ndarray) -> float:
    """Plug-in MI in nats from a joint (rows=z, cols=x) count matrix."""
    p = joint / max(joint.sum(), 1.0)
    pz = p.sum(axis=1, keepdims=True)
    px = p.sum(axis=0, keepdims=True)
    denom = pz @ px
    mask = (p > 0) & (denom > 0)
    return float((p[mask] * np.log(p[mask] / denom[mask])).sum())


def _segments_to_df(segments: list[dict[str, Any]]) -> pd.DataFrame:
    if not segments:
        return pd.DataFrame(columns=[
            "env_id", "z", "dwell", "dominant_phase", "flag_state_first",
            "opponent_id_first", "outcome_first", "outcome_last",
            "outcome_delta", "start_global_step", "end_global_step",
        ])
    df = pd.DataFrame(segments)
    return df


def _segment_mi_table(seg_df: pd.DataFrame, K: int) -> pd.DataFrame:
    """Compute uniform and dwell-weighted segment MI plus normalized variants."""
    if seg_df.empty:
        return pd.DataFrame()

    z = seg_df["z"].to_numpy(dtype=np.int64)
    dwell = seg_df["dwell"].to_numpy(dtype=np.float64)

    n_phases = len(TEAM_PHASES)
    n_flag = 4
    n_outcome = 3
    n_opp = int(seg_df["opponent_id_first"].max()) + 1 if not seg_df.empty else 1
    n_opp = max(n_opp, 1)

    contexts = {
        "phase_dominant": (seg_df["dominant_phase"].to_numpy(dtype=np.int64), n_phases),
        "flag_state_at_start": (seg_df["flag_state_first"].to_numpy(dtype=np.int64), n_flag),
        "outcome_at_end": (seg_df["outcome_last"].to_numpy(dtype=np.int64), n_outcome),
        "outcome_delta": (
            (seg_df["outcome_delta"].to_numpy(dtype=np.int64) + 2),  # shift -2..2 -> 0..4
            5,
        ),
        "opponent_id": (seg_df["opponent_id_first"].to_numpy(dtype=np.int64), n_opp),
    }

    def _build_joint(weights: np.ndarray, x: np.ndarray, n_x: int) -> np.ndarray:
        valid = (z >= 0) & (z < K) & (x >= 0) & (x < n_x)
        if not valid.any():
            return np.zeros((K, n_x), dtype=np.float64)
        idx = z[valid].astype(np.int64) * n_x + x[valid].astype(np.int64)
        # bincount weighted by dwell (or 1.0 for uniform).
        flat = np.bincount(idx, weights=weights[valid], minlength=K * n_x)
        return flat.reshape(K, n_x).astype(np.float64)

    rows: list[dict[str, Any]] = []
    for label, (x, n_x) in contexts.items():
        joint_unif = _build_joint(np.ones_like(dwell), x, n_x)
        joint_dwell = _build_joint(dwell, x, n_x)

        mi_unif = _mi_nats(joint_unif)
        mi_dwell = _mi_nats(joint_dwell)

        pz_unif = joint_unif.sum(axis=1) / max(joint_unif.sum(), 1.0)
        pz_dwell = joint_dwell.sum(axis=1) / max(joint_dwell.sum(), 1.0)
        h_z_unif = _shannon_nats(pz_unif)
        h_z_dwell = _shannon_nats(pz_dwell)

        rows.append({
            "context": label,
            "n_categories": int(n_x),
            "n_segments": int(joint_unif.sum()),
            "total_dwell": float(joint_dwell.sum()),
            "mi_uniform_nats": mi_unif,
            "mi_dwell_nats": mi_dwell,
            "H_z_uniform_nats": h_z_unif,
            "H_z_dwell_nats": h_z_dwell,
            "normalized_mi_uniform": mi_unif / h_z_unif if h_z_unif > 1e-12 else 0.0,
            "normalized_mi_dwell": mi_dwell / h_z_dwell if h_z_dwell > 1e-12 else 0.0,
        })

    return pd.DataFrame(rows)


def _segments_summary(seg_df: pd.DataFrame, K: int) -> pd.DataFrame:
    if seg_df.empty:
        return pd.DataFrame()
    rows = []
    total_dwell = float(seg_df["dwell"].sum())
    total_segments = float(len(seg_df))
    for k in range(K):
        sub = seg_df[seg_df["z"] == k]
        dwell_k = float(sub["dwell"].sum())
        rows.append({
            "z": k,
            "segment_count": int(len(sub)),
            "segment_share": float(len(sub) / max(total_segments, 1.0)),
            "total_dwell_steps": dwell_k,
            "dwell_share": float(dwell_k / max(total_dwell, 1.0)),
            "avg_dwell": float(sub["dwell"].mean()) if not sub.empty else 0.0,
            "median_dwell": float(sub["dwell"].median()) if not sub.empty else 0.0,
            "p90_dwell": float(sub["dwell"].quantile(0.9)) if not sub.empty else 0.0,
            "outcome_delta_mean": float(sub["outcome_delta"].mean()) if not sub.empty else 0.0,
        })
    return pd.DataFrame(rows)


def _behavior_by_z_df(acc: WelfordPerZ) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k in range(acc.K):
        row: dict[str, Any] = {"z": k, "step_count": int(acc.count[k])}
        for j, name in enumerate(acc.names):
            row[f"{name}_mean"] = float(acc.mean[k, j])
            if acc.count[k] > 1:
                var = float(acc.m2[k, j] / (acc.count[k] - 1))
                row[f"{name}_std"] = float(math.sqrt(max(var, 0.0)))
            else:
                row[f"{name}_std"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _z_wr_by_opponent(episodes_path: Path, K: int) -> pd.DataFrame:
    if not episodes_path.exists():
        return pd.DataFrame()
    ep = pd.read_csv(episodes_path)
    if ep.empty or "latent_z" not in ep.columns or "opponent_id" not in ep.columns:
        return pd.DataFrame()
    grouped = ep.groupby(["latent_z", "opponent_id"])
    out_rows: list[dict[str, Any]] = []
    for (z, opp), grp in grouped:
        out_rows.append({
            "z": int(z),
            "opponent_id": int(opp),
            "n_episodes": int(len(grp)),
            "win_rate": float(grp["success"].mean()) if "success" in grp.columns else float("nan"),
            "blue_score_mean": float(grp["blue_score"].mean()) if "blue_score" in grp.columns else float("nan"),
            "red_score_mean": float(grp["red_score"].mean()) if "red_score" in grp.columns else float("nan"),
            "win_margin_mean": float(grp["win_margin"].mean()) if "win_margin" in grp.columns else float("nan"),
        })
    return pd.DataFrame(out_rows).sort_values(["z", "opponent_id"]).reset_index(drop=True)


def _z_phase_segments(seg_df: pd.DataFrame, K: int) -> pd.DataFrame:
    if seg_df.empty:
        return pd.DataFrame()
    rows = []
    n_phases = len(TEAM_PHASES)
    grand_dwell = float(seg_df["dwell"].sum())
    for k in range(K):
        for p in range(n_phases):
            sub = seg_df[(seg_df["z"] == k) & (seg_df["dominant_phase"] == p)]
            dwell = float(sub["dwell"].sum())
            rows.append({
                "z": k,
                "dominant_phase": p,
                "phase_name": TEAM_PHASES[p],
                "segment_count": int(len(sub)),
                "total_dwell": dwell,
                "dwell_share": float(dwell / max(grand_dwell, 1.0)),
                "avg_dwell": float(sub["dwell"].mean()) if not sub.empty else 0.0,
                "outcome_delta_mean": float(sub["outcome_delta"].mean()) if not sub.empty else 0.0,
            })
    return pd.DataFrame(rows)


def _z_flag_segments(seg_df: pd.DataFrame, K: int) -> pd.DataFrame:
    if seg_df.empty:
        return pd.DataFrame()
    rows = []
    n_flag = 4
    grand_dwell = float(seg_df["dwell"].sum())
    flag_names = ("neutral", "blue_captured", "red_captured", "both_captured")
    for k in range(K):
        for f in range(n_flag):
            sub = seg_df[(seg_df["z"] == k) & (seg_df["flag_state_first"] == f)]
            dwell = float(sub["dwell"].sum())
            rows.append({
                "z": k,
                "flag_state": f,
                "flag_state_name": flag_names[f],
                "segment_count": int(len(sub)),
                "total_dwell": dwell,
                "dwell_share": float(dwell / max(grand_dwell, 1.0)),
                "outcome_delta_mean": float(sub["outcome_delta"].mean()) if not sub.empty else 0.0,
            })
    return pd.DataFrame(rows)


def _qphi_audit_df(acc: ColumnStats) -> pd.DataFrame:
    n = len(acc.names)
    if acc.count == 0:
        return pd.DataFrame({"context_index": np.arange(n), "global_state_field": [""] * n})
    var = acc.m2 / max(acc.count - 1, 1)
    std = np.sqrt(np.maximum(var, 0.0))
    frac_zero = acc.zero_count / max(acc.count, 1)
    # Map context_i to (frame_t, global_state_field): index = frame_idx * 34 + field_idx
    frames = np.arange(n) // GLOBAL_STATE_DIM
    field_idx = np.arange(n) % GLOBAL_STATE_DIM
    field_names = [GLOBAL_STATE_FIELD_NAMES[i] for i in field_idx]
    df = pd.DataFrame({
        "context_index": np.arange(n),
        "frame_offset": frames,
        "global_state_field": field_names,
        "mean": acc.mean,
        "std": std,
        "min": acc.vmin,
        "max": acc.vmax,
        "frac_zero": frac_zero,
        "is_constant": (std < 1e-9),
    })
    return df


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a GitHub-flavored markdown table without requiring `tabulate`.

    Mirrors ``DataFrame.to_markdown(index=False)`` for the subset of formatting
    this report needs. Floats are rendered using pandas' default formatting
    (callers ``.round()`` first to control precision).
    """
    if df.empty:
        return ""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body: list[str] = []
    for _, row in df.iterrows():
        cells: list[str] = []
        for v in row:
            if pd.isna(v):
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v}")
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


def _write_report(
    out_path: Path,
    *,
    behavior_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    seg_mi_df: pd.DataFrame,
    z_phase_df: pd.DataFrame,
    z_flag_df: pd.DataFrame,
    z_wr_opp_df: pd.DataFrame,
    qphi_df: pd.DataFrame,
    switch_by_phase: np.ndarray,
    rows_by_phase: np.ndarray,
    rows_seen: int,
    n_segments: int,
    K: int,
) -> None:
    lines: list[str] = []
    lines.append(f"# Latent strategy post-hoc analysis\n")
    lines.append(f"Source rows streamed: **{rows_seen:,}** | segments finalized: **{n_segments:,}** | latent K: **{K}**\n")

    lines.append("## Dwell-weighted segment MI (Summer-faithful)")
    lines.append("Computed at z-segment granularity, not per-step. Normalized = MI / H(z_seg).\n")
    if not seg_mi_df.empty:
        lines.append(_df_to_markdown(seg_mi_df.round(4)))
    lines.append("")

    lines.append("## Dwell-by-z")
    if not summary_df.empty:
        lines.append(_df_to_markdown(summary_df.round(3)))
    lines.append("")

    lines.append("## Switch rate by team phase")
    rows = []
    for p, name in enumerate(TEAM_PHASES):
        total = float(rows_by_phase[p])
        rate = float(switch_by_phase[p] / total) if total > 0 else 0.0
        rows.append({"phase_id": p, "phase_name": name, "rows": int(total), "switch_rate": rate})
    lines.append(_df_to_markdown(pd.DataFrame(rows).round(4)))
    lines.append("")

    lines.append("## Top distinguishing behaviors per z")
    if not behavior_df.empty and K > 1:
        means = behavior_df[[f"{n}_mean" for n in BEHAVIOR_TELEMETRY_NAMES]].to_numpy()
        # For each behavior, the per-z deviation from the across-z mean.
        across = means.mean(axis=0, keepdims=True)
        dev = means - across
        for k in range(K):
            order = np.argsort(-np.abs(dev[k]))
            rank = order[:3]
            picks = ", ".join(
                f"`{BEHAVIOR_TELEMETRY_NAMES[i]}` ({means[k, i]:+.3f} vs avg {across[0, i]:+.3f})"
                for i in rank
            )
            lines.append(f"- **z{k}**: {picks}")
    lines.append("")

    lines.append("## z_wr by opponent (from episodes.csv)")
    if not z_wr_opp_df.empty:
        lines.append(_df_to_markdown(z_wr_opp_df.round(3)))
    lines.append("")

    lines.append("## q_phi global-feature audit")
    lines.append(
        "Each `q_phi_context_i` is mapped to a frame offset + global-state field "
        "(34-dim base × 5 temporal frames). Columns with `std < 1e-9` or "
        "`frac_zero` near 1.0 are not contributing signal.\n"
    )
    if not qphi_df.empty:
        # Highlight strategic fields the user called out explicitly.
        strategic = {
            "blue_flag_captured", "red_flag_captured",
            "min_alive_blue_to_red_flag", "min_alive_red_to_blue_flag",
            "blue_std_x", "blue_std_y",
            "team_pairwise_distance_mean", "team_pairwise_distance_std",
            "flag_pressure_blue", "flag_pressure_red",
            "carrier_dist_home", "carrier_enemy_nearest_dist",
            "score_diff_norm", "blue_score_norm", "red_score_norm",
        }
        latest = qphi_df[qphi_df["frame_offset"] == 0].copy()
        latest["is_strategic"] = latest["global_state_field"].isin(strategic)
        cols = ["global_state_field", "mean", "std", "min", "max", "frac_zero", "is_constant", "is_strategic"]
        lines.append("### Current frame (frame_offset = 0)")
        lines.append(_df_to_markdown(latest[cols].round(4)))
        # Flag dead columns across all frames.
        dead = qphi_df[qphi_df["is_constant"]]
        if not dead.empty:
            lines.append(f"\n**{len(dead)} of {len(qphi_df)} q_phi context columns are constant (std < 1e-9).** "
                         "These are not contributing signal to q_phi.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(run_arg: str, *, K: int, out_dir: Path | None, chunk_rows: int) -> dict[str, Path]:
    paths = _resolve_run_paths(run_arg)
    e3 = paths["e3"]
    if not e3.exists():
        raise FileNotFoundError(f"Missing e3 step CSV: {e3}")
    out_dir = out_dir or e3.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = paths["stem"].name

    def _out(name: str) -> Path:
        return out_dir / f"{stem}_analysis_{name}"

    print(f"[analyze_latent] streaming {e3} (chunk_rows={chunk_rows}) ...")
    segments, behavior_acc, qphi_acc, switch_by_phase, rows_by_phase, rows_seen = _stream_e3_steps(
        e3, K, chunk_rows
    )
    print(f"[analyze_latent] streamed {rows_seen:,} rows -> {len(segments):,} segments")

    seg_df = _segments_to_df(segments)
    seg_mi_df = _segment_mi_table(seg_df, K)
    summary_df = _segments_summary(seg_df, K)
    behavior_df = _behavior_by_z_df(behavior_acc)
    z_wr_opp_df = _z_wr_by_opponent(paths["episodes"], K)
    z_phase_df = _z_phase_segments(seg_df, K)
    z_flag_df = _z_flag_segments(seg_df, K)
    qphi_df = _qphi_audit_df(qphi_acc)

    outputs: dict[str, Path] = {}
    if not behavior_df.empty:
        p = _out("behavior_by_z.csv"); behavior_df.to_csv(p, index=False); outputs["behavior_by_z"] = p
    if not summary_df.empty:
        p = _out("segments_summary.csv"); summary_df.to_csv(p, index=False); outputs["segments_summary"] = p
    if not seg_mi_df.empty:
        p = _out("segment_mi.csv"); seg_mi_df.to_csv(p, index=False); outputs["segment_mi"] = p
    if not z_wr_opp_df.empty:
        p = _out("z_wr_by_opponent.csv"); z_wr_opp_df.to_csv(p, index=False); outputs["z_wr_by_opponent"] = p
    if not z_phase_df.empty:
        p = _out("z_phase_segments.csv"); z_phase_df.to_csv(p, index=False); outputs["z_phase_segments"] = p
    if not z_flag_df.empty:
        p = _out("z_flag_segments.csv"); z_flag_df.to_csv(p, index=False); outputs["z_flag_segments"] = p
    if not qphi_df.empty:
        p = _out("qphi_audit.csv"); qphi_df.to_csv(p, index=False); outputs["qphi_audit"] = p

    report = _out("report.md")
    _write_report(
        report,
        behavior_df=behavior_df,
        summary_df=summary_df,
        seg_mi_df=seg_mi_df,
        z_phase_df=z_phase_df,
        z_flag_df=z_flag_df,
        z_wr_opp_df=z_wr_opp_df,
        qphi_df=qphi_df,
        switch_by_phase=switch_by_phase,
        rows_by_phase=rows_by_phase,
        rows_seen=rows_seen,
        n_segments=len(segments),
        K=K,
    )
    outputs["report"] = report

    print("[analyze_latent] wrote:")
    for name, p in outputs.items():
        print(f"  - {name}: {p}")
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="Path to a run (directory, *_metrics.csv, or *_e3_steps.csv).")
    parser.add_argument("--latent-k", type=int, default=4, help="Number of latent strategies K (default 4).")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: alongside the run CSVs).")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS,
                        help="Pandas chunk size for streaming e3_steps (default 50000).")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else None
    run(args.run, K=int(args.latent_k), out_dir=out_dir, chunk_rows=int(args.chunk_rows))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
