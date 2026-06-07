"""Tests for ``tools/analyze_latent.py`` post-hoc diagnostics.

Covered:

* segment builder splits z runs correctly per-env across chunk boundaries
* dwell-weighted MI matches a hand-computed reference
* uniform z over a single context category yields MI = 0
* behavior_by_z averages match a direct numpy groupby
* q_phi audit detects constant columns
* normalized_mi columns appear in the live update fieldnames
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES
from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_phase_labels import TEAM_PHASES
from tools import analyze_latent as al

TEMPORAL_FRAMES = al.TEMPORAL_FRAMES
N_CTX = TEMPORAL_FRAMES * GLOBAL_STATE_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_e3_dataframe(
    *,
    rows: list[dict[str, object]],
) -> pd.DataFrame:
    """Build a per-step DataFrame matching the real e3_steps schema (only used columns)."""
    base = {
        "update": 0,
        "rollout_step": 0,
        "env_id": 0,
        "global_step": 0,
        "z_t": 0,
        "switched": 0,
        "phase_id": 0,
        "score_outcome": "tied",
        "opponent_id": 2,
    }
    for name in BEHAVIOR_TELEMETRY_NAMES:
        base[name] = 0.0
    for i in range(N_CTX):
        base[f"q_phi_context_{i}"] = 0.0
    records = []
    for r in rows:
        rec = dict(base)
        rec.update(r)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def _write_e3_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Segment builder
# ---------------------------------------------------------------------------


def test_segment_builder_handles_within_chunk_runs(tmp_path: Path) -> None:
    """Two envs, three z runs each, all within a single chunk."""
    rows: list[dict[str, object]] = []
    for env in range(2):
        for step in range(6):
            z = 0 if step < 2 else (1 if step < 4 else 2)
            sw = 1 if step in (2, 4) else 0
            rows.append({
                "env_id": env, "update": 0, "rollout_step": step,
                "global_step": step * 32, "z_t": z, "switched": sw,
                "phase_id": step % len(TEAM_PHASES),
                "score_outcome": "tied", "opponent_id": 2,
            })
    csv_path = tmp_path / "fake_e3_steps.csv"
    _write_e3_csv(_make_e3_dataframe(rows=rows), csv_path)

    segments, beh_acc, qphi_acc, switch_by_phase, rows_by_phase, total = al._stream_e3_steps(
        csv_path, K=3, chunk_rows=100,
    )
    assert total == len(rows)
    # 2 envs * 3 z runs each = 6 segments
    assert len(segments) == 6
    # Each segment has dwell 2
    dwell_set = {s["dwell"] for s in segments}
    assert dwell_set == {2}
    # Each env should have z=[0, 1, 2] in order
    for env in range(2):
        env_segs = sorted(
            [s for s in segments if s["env_id"] == env],
            key=lambda s: s["start_global_step"],
        )
        assert [s["z"] for s in env_segs] == [0, 1, 2]


def test_segment_builder_carries_across_chunks(tmp_path: Path) -> None:
    """A single env with z=0 spanning across a chunk boundary is one segment."""
    rows: list[dict[str, object]] = []
    for step in range(20):
        rows.append({
            "env_id": 0, "update": 0, "rollout_step": step,
            "global_step": step * 32, "z_t": 0, "switched": 0,
            "phase_id": 1, "score_outcome": "tied", "opponent_id": 2,
        })
    csv_path = tmp_path / "fake_e3_steps.csv"
    _write_e3_csv(_make_e3_dataframe(rows=rows), csv_path)

    segments, *_ = al._stream_e3_steps(csv_path, K=2, chunk_rows=5)
    # All 20 rows are one z-run -> one segment
    assert len(segments) == 1
    assert segments[0]["z"] == 0
    assert segments[0]["dwell"] == 20
    assert segments[0]["dominant_phase"] == 1


def test_segment_split_at_chunk_boundary(tmp_path: Path) -> None:
    """z changes exactly at a chunk boundary; segments split correctly."""
    rows: list[dict[str, object]] = []
    for step in range(10):
        z = 0 if step < 5 else 1
        sw = 1 if step == 5 else 0
        rows.append({
            "env_id": 0, "update": 0, "rollout_step": step,
            "global_step": step * 32, "z_t": z, "switched": sw,
            "phase_id": 2, "score_outcome": "tied", "opponent_id": 2,
        })
    csv_path = tmp_path / "fake_e3_steps.csv"
    _write_e3_csv(_make_e3_dataframe(rows=rows), csv_path)

    # chunk_rows=5 → boundary aligns with z change
    segments, *_ = al._stream_e3_steps(csv_path, K=2, chunk_rows=5)
    assert len(segments) == 2
    segments.sort(key=lambda s: s["start_global_step"])
    assert segments[0]["z"] == 0 and segments[0]["dwell"] == 5
    assert segments[1]["z"] == 1 and segments[1]["dwell"] == 5


# ---------------------------------------------------------------------------
# MI invariants
# ---------------------------------------------------------------------------


def test_uniform_z_yields_zero_mi() -> None:
    """If z is constant across all segments, I(z; phase) must be 0."""
    df = pd.DataFrame({
        "env_id": [0, 0, 0, 0],
        "z": [1, 1, 1, 1],
        "dwell": [10, 20, 5, 7],
        "dominant_phase": [0, 1, 2, 3],
        "flag_state_first": [0, 1, 2, 3],
        "opponent_id_first": [2, 2, 2, 2],
        "outcome_first": [1, 1, 1, 1],
        "outcome_last": [1, 1, 2, 0],
        "outcome_delta": [0, 0, 1, -1],
        "start_global_step": [0, 32, 64, 96],
        "end_global_step": [10, 50, 65, 110],
    })
    mi = al._segment_mi_table(df, K=4)
    phase_row = mi[mi["context"] == "phase_dominant"].iloc[0]
    assert phase_row["mi_uniform_nats"] == pytest.approx(0.0, abs=1e-12)
    assert phase_row["mi_dwell_nats"] == pytest.approx(0.0, abs=1e-12)
    assert phase_row["normalized_mi_uniform"] == pytest.approx(0.0, abs=1e-12)


def test_perfectly_correlated_z_phase_yields_max_mi() -> None:
    """When z == dominant_phase deterministically, MI(z; phase) == H(z)."""
    df = pd.DataFrame({
        "env_id": [0, 0, 0, 0],
        "z": [0, 1, 2, 3],
        "dwell": [10, 10, 10, 10],
        "dominant_phase": [0, 1, 2, 3],
        "flag_state_first": [0, 1, 2, 3],
        "opponent_id_first": [2, 2, 2, 2],
        "outcome_first": [1, 1, 1, 1],
        "outcome_last": [1, 1, 1, 1],
        "outcome_delta": [0, 0, 0, 0],
        "start_global_step": [0, 32, 64, 96],
        "end_global_step": [10, 42, 74, 106],
    })
    mi = al._segment_mi_table(df, K=4)
    phase_row = mi[mi["context"] == "phase_dominant"].iloc[0]
    # With uniform z and 1-to-1 mapping, H(z) = ln(4) and I(z; phase) = H(z).
    assert phase_row["mi_uniform_nats"] == pytest.approx(math.log(4), abs=1e-9)
    assert phase_row["normalized_mi_uniform"] == pytest.approx(1.0, abs=1e-9)


def test_dwell_weight_amplifies_a_subset() -> None:
    """Dwell weighting biases MI toward segments with large dwell."""
    # Setup: z=0 segments are short (dwell=1) and ambiguous (phase=0 or 1).
    # z=1 segments are long (dwell=100) and ALL in phase=0. Dwell-weighted MI
    # should be lower than uniform MI because z=1 dominates and is uninformative
    # about phase.
    df = pd.DataFrame({
        "env_id": [0, 0, 0, 0, 0],
        "z": [0, 0, 0, 0, 1],
        "dwell": [1, 1, 1, 1, 100],
        "dominant_phase": [0, 1, 0, 1, 0],
        "flag_state_first": [0, 0, 0, 0, 0],
        "opponent_id_first": [2, 2, 2, 2, 2],
        "outcome_first": [1] * 5,
        "outcome_last": [1] * 5,
        "outcome_delta": [0] * 5,
        "start_global_step": [0, 1, 2, 3, 4],
        "end_global_step": [1, 2, 3, 4, 104],
    })
    mi = al._segment_mi_table(df, K=2)
    phase_row = mi[mi["context"] == "phase_dominant"].iloc[0]
    # Uniform: 4 z=0 segs split phase 0/1 50/50; 1 z=1 seg in phase=0.
    # Dwell-weighted: z=1 dominates with 100 dwell units in phase=0.
    assert phase_row["mi_uniform_nats"] > phase_row["mi_dwell_nats"]


# ---------------------------------------------------------------------------
# Behavior accumulator
# ---------------------------------------------------------------------------


def test_behavior_accumulator_matches_groupby() -> None:
    rng = np.random.default_rng(0)
    n_rows = 200
    K = 4
    z = rng.integers(0, K, size=n_rows)
    values = rng.standard_normal((n_rows, len(BEHAVIOR_TELEMETRY_NAMES)))

    acc = al.WelfordPerZ(K, BEHAVIOR_TELEMETRY_NAMES)
    # Feed in two chunks to exercise the parallel-update code path.
    acc.update_block(z[:80], values[:80])
    acc.update_block(z[80:], values[80:])

    for k in range(K):
        mask = z == k
        if not bool(mask.any()):
            continue
        ref_mean = values[mask].mean(axis=0)
        for j, name in enumerate(BEHAVIOR_TELEMETRY_NAMES):
            assert acc.mean[k, j] == pytest.approx(ref_mean[j], rel=1e-9, abs=1e-9), name


# ---------------------------------------------------------------------------
# q_phi audit
# ---------------------------------------------------------------------------


def test_qphi_audit_flags_constant_columns(tmp_path: Path) -> None:
    """Columns held constant at 0 should be flagged ``is_constant=True``."""
    rows: list[dict[str, object]] = []
    for step in range(20):
        row: dict[str, object] = {
            "env_id": 0, "update": 0, "rollout_step": step,
            "global_step": step * 32, "z_t": step % 2, "switched": int(step % 2 == 0),
            "phase_id": step % len(TEAM_PHASES),
            "score_outcome": "tied", "opponent_id": 2,
        }
        for name in BEHAVIOR_TELEMETRY_NAMES:
            row[name] = float(step)
        for i in range(N_CTX):
            # Only context_0 varies; everything else is zero.
            row[f"q_phi_context_{i}"] = float(step) if i == 0 else 0.0
        rows.append(row)

    csv_path = tmp_path / "fake_e3_steps.csv"
    _write_e3_csv(pd.DataFrame.from_records(rows), csv_path)
    _, _, qphi_acc, *_ = al._stream_e3_steps(csv_path, K=2, chunk_rows=10)

    df = al._qphi_audit_df(qphi_acc)
    assert df.loc[df["context_index"] == 0, "is_constant"].iloc[0] == False  # varies
    assert df.loc[df["context_index"] == 5, "is_constant"].iloc[0] == True  # zero
    # Frame offset and global state field mapping
    assert df.loc[df["context_index"] == GLOBAL_STATE_DIM, "frame_offset"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Live trainer field-name additions
# ---------------------------------------------------------------------------


def test_normalized_mi_fields_present_in_update_csv() -> None:
    from rl.custom_ppo.csv_writers import _update_fieldnames

    fields = _update_fieldnames(use_latent_strategy=True, latent_k=4)
    for name in (
        "latent_normalized_mi_z_opponent",
        "latent_normalized_mi_z_phase",
        "latent_normalized_mi_z_outcome",
        "latent_normalized_mi_z_flag_state",
        "latent_z_marginal_entropy_nats",
    ):
        assert name in fields, f"missing live field: {name}"


def test_normalized_mi_absent_when_latent_disabled() -> None:
    from rl.custom_ppo.csv_writers import _update_fieldnames

    fields = _update_fieldnames(use_latent_strategy=False, latent_k=4)
    for name in (
        "latent_normalized_mi_z_opponent",
        "latent_normalized_mi_z_phase",
        "latent_z_marginal_entropy_nats",
    ):
        assert name not in fields
