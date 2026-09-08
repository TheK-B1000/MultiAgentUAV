"""Unit tests for ``tools/qualitative_rollout.py``.

These tests cover the pure-Python aggregation, schema, and Summer-faithful
contract pieces of the qualitative rollout tool. They do NOT spin up the
GPU CTF env -- a separate smoke run against a real checkpoint validates the
env/model wiring end-to-end.

Contract assertions (locked in here so future refactors stay faithful):

* No supervised labels in any output row (no ``phase_label``, ``flag_label``,
  ``outcome_label`` columns).
* No prediction/loss columns -- only observed behavior.
* The per-z aggregator distinguishes natural vs fixed_z modes.
* OP5 maps to OP5_RUSHER (matches env's internal tag).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES
from rl.route_telemetry import ROUTE_CROSSING_KEYS, ROUTE_DERIVED_NAMES, ROUTE_TELEMETRY_NAMES
from tools import qualitative_rollout as qr


# ---------------------------------------------------------------------------
# Opponent tag mapping
# ---------------------------------------------------------------------------


def test_env_opponent_tag_maps_op5_to_op5_rusher():
    assert qr._env_opponent_tag("OP5") == "OP5_RUSHER"
    assert qr._env_opponent_tag("op5") == "OP5_RUSHER"
    assert qr._env_opponent_tag("OP6") == "OP6_IMMEDIATE_DUAL_RUSH"
    assert qr._env_opponent_tag("OP7") == "OP7_DEEP_FORTRESS"
    # Unknowns pass through (uppercased, trimmed) so the env's own validation
    # can flag mismatches.
    assert qr._env_opponent_tag("OP3") == "OP3"
    assert qr._env_opponent_tag("  OP4 ") == "OP4"


# ---------------------------------------------------------------------------
# Per-step CSV schema
# ---------------------------------------------------------------------------


def test_step_csv_fieldnames_includes_required_signals():
    fields = qr._step_csv_fieldnames(latent_k=4, n_blue=4, n_red=4)

    # User-requested per-step signals:
    required_metadata = {
        "map_layout", "opponent", "mode", "fixed_z_id", "episode_idx", "step",
        "z_active", "z_resampled", "q_phi_entropy",
        "blue_score", "red_score", "blue_score_delta", "red_score_delta",
        "blue_carrier_count", "red_carrier_count",
        "blue_picked_up_now", "red_picked_up_now",
        "blue_dropped_now", "red_dropped_now",
    }
    assert required_metadata.issubset(fields), (
        f"missing required metadata columns: {required_metadata - set(fields)}"
    )

    # All behavior telemetry signals must be present, including the four
    # the user explicitly called out.
    for name in BEHAVIOR_TELEMETRY_NAMES:
        assert name in fields
    for required in (
        "team_spread", "num_attackers", "intercept_pressure", "attack_defense_ratio",
    ):
        assert required in fields, f"required behavior signal {required!r} missing"

    for name in ROUTE_TELEMETRY_NAMES:
        assert name in fields, f"required route telemetry column {name!r} missing"
    for name in ROUTE_CROSSING_KEYS + ROUTE_DERIVED_NAMES:
        assert name in fields

    # q_phi probability columns must exist for K classes.
    for k in range(4):
        assert f"q_phi_prob_{k}" in fields

    # Position / alive / carrying per agent (both teams).
    for prefix in ("blue_x", "blue_y", "blue_alive", "blue_carrying"):
        for i in range(4):
            assert f"{prefix}_{i}" in fields
    for prefix in ("red_x", "red_y", "red_alive", "red_carrying"):
        for i in range(4):
            assert f"{prefix}_{i}" in fields


def test_step_csv_fieldnames_has_no_supervised_label_columns():
    """Summer-faithful: never emit phase/flag/outcome label columns."""
    fields = qr._step_csv_fieldnames(latent_k=4, n_blue=4, n_red=4)
    forbidden_substrings = ("phase_label", "flag_label", "outcome_label", "predicted_z")
    for col in fields:
        for sub in forbidden_substrings:
            assert sub not in col, f"forbidden supervised-label column emitted: {col!r}"


# ---------------------------------------------------------------------------
# Aggregation by (opponent, mode, z)
# ---------------------------------------------------------------------------


def _make_record(
    opponent: str,
    mode: str,
    fixed_z: int,
    episode_idx: int,
    z_timeline: list[int],
    blue_won: bool,
    *,
    blue_score: int = 1,
    red_score: int = 0,
) -> qr.EpisodeRecord:
    rec = qr.EpisodeRecord(
        opponent=opponent,
        mode=mode,
        fixed_z_id=fixed_z,
        episode_idx=episode_idx,
    )
    rec.outcome_blue_score = blue_score
    rec.outcome_red_score = red_score
    rec.outcome_blue_won = blue_won
    rec.outcome_decision_steps = len(z_timeline)
    rec.n_steps = len(z_timeline)
    rec.rows = []
    for step, z in enumerate(z_timeline):
        row: dict[str, float | int] = {
            "opponent": opponent,
            "map_layout": "map_b_split_lane_v2",
            "mode": mode,
            "fixed_z_id": fixed_z,
            "episode_idx": episode_idx,
            "step": step,
            "z_active": z,
            "z_resampled": 0,
            "q_phi_entropy": 0.0,
            "blue_score": blue_score if step == len(z_timeline) - 1 else 0,
            "red_score": red_score if step == len(z_timeline) - 1 else 0,
            "blue_score_delta": 1 if step == len(z_timeline) - 1 and blue_won else 0,
            "red_score_delta": 0,
            "blue_carrier_count": 0,
            "red_carrier_count": 0,
            "blue_picked_up_now": 1 if step == 0 else 0,
            "red_picked_up_now": 0,
            "blue_dropped_now": 0,
            "red_dropped_now": 0,
        }
        for name in BEHAVIOR_TELEMETRY_NAMES:
            # Deliberately z-dependent so the aggregator's mean computation
            # is easy to verify against hand-computed values.
            row[name] = float(z)
        rec.rows.append(row)
    return rec


def test_aggregate_by_z_fixed_mode_groups_per_forced_z():
    records = [
        _make_record("OP3", "fixed_z", 0, 0, [0, 0, 0], blue_won=True),
        _make_record("OP3", "fixed_z", 0, 1, [0, 0, 0], blue_won=False),
        _make_record("OP3", "fixed_z", 1, 0, [1, 1, 1, 1], blue_won=True),
    ]
    rows = qr._aggregate_by_z(records)
    # Two unique (opp, mode, z) groups: (OP3, fixed_z, 0) and (OP3, fixed_z, 1).
    assert len(rows) == 2
    by_z = {(r["opponent"], r["mode"], r["z"]): r for r in rows}
    assert (("OP3", "fixed_z", 0)) in by_z
    assert (("OP3", "fixed_z", 1)) in by_z

    z0 = by_z[("OP3", "fixed_z", 0)]
    assert z0["map_layout"] == "map_b_split_lane_v2"
    assert z0["n_episodes_touched"] == 2
    assert z0["n_steps"] == 6  # 2 episodes * 3 steps each
    # 1 of 2 episodes won.
    assert z0["blue_win_rate"] == pytest.approx(0.5)
    # behavior_mean == z, so for z=0 it must be 0.0 for every signal.
    for name in BEHAVIOR_TELEMETRY_NAMES:
        assert z0[f"{name}_mean"] == pytest.approx(0.0)

    z1 = by_z[("OP3", "fixed_z", 1)]
    assert z1["n_episodes_touched"] == 1
    assert z1["n_steps"] == 4
    assert z1["blue_win_rate"] == pytest.approx(1.0)
    for name in BEHAVIOR_TELEMETRY_NAMES:
        assert z1[f"{name}_mean"] == pytest.approx(1.0)


def test_aggregate_by_z_natural_mode_groups_by_step_level_z():
    """In natural mode each row is bucketed by the z that was active at that step."""
    # One episode spends 3 steps in z=0, 5 steps in z=1, then 2 steps in z=2.
    records = [
        _make_record("OP5", "natural", -1, 0, [0, 0, 0, 1, 1, 1, 1, 1, 2, 2], blue_won=True),
    ]
    rows = qr._aggregate_by_z(records)
    by_z = {r["z"]: r for r in rows if r["opponent"] == "OP5"}
    assert set(by_z.keys()) == {0, 1, 2}
    assert by_z[0]["n_steps"] == 3
    assert by_z[1]["n_steps"] == 5
    assert by_z[2]["n_steps"] == 2
    # Every z was touched in the same episode -> all WR=1.0.
    for z in (0, 1, 2):
        assert by_z[z]["n_episodes_touched"] == 1
        assert by_z[z]["blue_win_rate"] == pytest.approx(1.0)


def test_aggregate_by_z_natural_wr_per_episode_visited():
    """Natural-mode WR is computed per-episode-that-visited-z, not per-step."""
    # Episode 0 (loss) visits {0, 1}. Episode 1 (win) visits {0}.
    # So WR for z=0 should be 1/2 = 0.5, and WR for z=1 should be 0/1 = 0.0.
    records = [
        _make_record("OP6", "natural", -1, 0, [0, 1, 1], blue_won=False),
        _make_record("OP6", "natural", -1, 1, [0, 0, 0], blue_won=True),
    ]
    rows = qr._aggregate_by_z(records)
    by_z = {r["z"]: r for r in rows}
    assert by_z[0]["n_episodes_touched"] == 2
    assert by_z[0]["blue_win_rate"] == pytest.approx(0.5)
    assert by_z[1]["n_episodes_touched"] == 1
    assert by_z[1]["blue_win_rate"] == pytest.approx(0.0)


def test_strategy_evidence_rows_report_natural_best_worst_and_spread():
    records = [
        _make_record("OP5", "natural", -1, 0, [0, 1], blue_won=True),
        _make_record("OP5", "natural", -1, 1, [1, 2], blue_won=False),
        _make_record("OP5", "fixed_z", 0, 0, [0, 0], blue_won=False),
        _make_record("OP5", "fixed_z", 1, 0, [1, 1], blue_won=True),
        _make_record("OP5", "fixed_z", 1, 1, [1, 1], blue_won=True),
    ]
    agg = qr._aggregate_by_z(records)
    rows = qr._build_strategy_evidence_rows(records, agg)

    assert len(rows) == 1
    row = rows[0]
    assert row["opponent"] == "OP5"
    assert row["natural_win_rate"] == pytest.approx(0.5)
    assert row["best_z"] == 1
    assert row["best_forced_z_win_rate"] == pytest.approx(1.0)
    assert row["worst_z"] == 0
    assert row["worst_forced_z_win_rate"] == pytest.approx(0.0)
    assert row["forced_z_performance_spread"] == pytest.approx(1.0)
    assert row["forced_z_behavior_spread"] == pytest.approx(1.0)
    assert row["strategy_spread"] == "high"
    assert row["interpretation"] == "latent specialization"


def test_strategy_interpretation_requires_behavior_for_specialization():
    assert qr._strategy_interpretation(0.20, 0.02) == "performance differs; strategy meaning unclear"
    assert qr._strategy_interpretation(0.00, 0.20) == "behavior differs; performance unclear"
    assert qr._strategy_interpretation(0.00, 0.00) == "no causal strategy"


# ---------------------------------------------------------------------------
# Position summary
# ---------------------------------------------------------------------------


def test_summarise_positions_centroid_ignores_dead_agents():
    positions = {
        "blue_x": [0.1, 0.9, 0.5],
        "blue_y": [0.2, 0.8, 0.4],
        "blue_alive": [1, 0, 1],
        "blue_carrying": [0, 0, 0],
        "red_x": [0.0, 0.0],
        "red_y": [0.0, 0.0],
        "red_alive": [0, 0],
        "red_carrying": [0, 0],
    }
    out = qr._summarise_positions(positions)
    assert out["blue_alive_count"] == 2.0
    assert out["red_alive_count"] == 0.0
    # Only the alive agents at idx 0 and 2 contribute.
    assert out["blue_centroid_x"] == pytest.approx((0.1 + 0.5) / 2)
    assert out["blue_centroid_y"] == pytest.approx((0.2 + 0.4) / 2)
    # No red alive -> centroid falls back to 0.0 (documented behaviour).
    assert out["red_centroid_x"] == 0.0
    assert out["red_centroid_y"] == 0.0


# ---------------------------------------------------------------------------
# Markdown summary content sanity
# ---------------------------------------------------------------------------


def test_write_summary_md_baseline_no_latent_path(tmp_path: Path):
    """Non-latent checkpoint: per-opponent baseline summary, no per-z sections."""
    records = [
        _make_record("OP4", "natural", -1, 0, [-1, -1, -1, -1], blue_won=True),
        _make_record("OP4", "natural", -1, 1, [-1, -1, -1], blue_won=False),
    ]
    agg = qr._aggregate_by_z(records)
    # Baseline aggregation: one (opp, mode, z=-1) bucket.
    assert len(agg) == 1
    only = agg[0]
    assert only["z"] == -1
    assert only["opponent"] == "OP4"
    assert only["n_episodes_touched"] == 2
    assert only["blue_win_rate"] == pytest.approx(0.5)

    out_md = tmp_path / "rollout_summary.md"
    qr._write_summary_md(
        out_md,
        records=records,
        agg_rows=agg,
        checkpoint=Path("final_no_latent_baseline.zip"),
        latent_k=0,
        n_blue=4,
        n_red=4,
        opponents=["OP4"],
        deterministic=True,
        seed=42,
        is_latent=False,
        map_layout="map_b_split_lane_v2",
    )
    text = out_md.read_text(encoding="utf-8")

    # Non-latent flag in header.
    assert "(baseline -- no latent strategy)" in text
    assert "no_latent baseline" in text
    assert "map layout: **map_b_split_lane_v2**" in text

    # Baseline-only section names.
    assert "## Per-opponent WR -- baseline (no z)" in text
    assert "## Behavioral fingerprint per opponent" in text

    # Latent-only sections must be absent.
    assert "Win rate by (opponent, z) -- fixed-z mode" not in text
    assert "Natural q_phi routing" not in text
    assert "Behavioral fingerprint per z" not in text
    assert "Top 3 distinguishing behaviors per z" not in text

    # Summer-faithful footer still present.
    assert "## Summer-faithful audit" in text


def test_write_summary_md_includes_required_sections(tmp_path: Path):
    records = [
        _make_record("OP3", "fixed_z", 0, 0, [0, 0, 0], blue_won=True),
        _make_record("OP3", "fixed_z", 1, 0, [1, 1, 1], blue_won=False),
        _make_record("OP3", "natural", -1, 0, [0, 1, 1, 0], blue_won=True),
    ]
    agg = qr._aggregate_by_z(records)
    out_md = tmp_path / "rollout_summary.md"
    qr._write_summary_md(
        out_md,
        records=records,
        agg_rows=agg,
        checkpoint=Path("dummy.zip"),
        latent_k=2,
        n_blue=4,
        n_red=4,
        opponents=["OP3"],
        deterministic=True,
        seed=42,
        map_layout="map_b_split_lane_v2",
    )
    text = out_md.read_text(encoding="utf-8")
    # Section headers we care about for review.
    assert "# Qualitative rollout" in text
    assert "## Win rate by (opponent, z) -- fixed-z mode" in text
    assert "## Natural q_phi routing" in text
    assert "## Behavioral fingerprint per z" in text
    assert "## Route fingerprint per z" in text
    assert "## Strategy evidence table" in text
    assert "## Summer-faithful audit" in text
    assert "map layout: **map_b_split_lane_v2**" in text
    assert "Latent strategy evidence requires both forced-z behavior" in text
    # Explicitly states no supervised labels / no backward.
    assert "No backward pass" in text
    assert "No phase / flag / outcome prediction loss" in text


# ---------------------------------------------------------------------------
# Module-level contract: no training imports leak in
# ---------------------------------------------------------------------------


def test_module_does_not_import_trainer_or_optimizer():
    """Summer-faithful contract: pure eval module must not pull in the trainer
    or any optimizer state machinery."""
    import inspect
    src = inspect.getsource(qr)
    forbidden = (
        "CustomPPOTrainer",
        "torch.optim",
        ".backward(",
        "optimizer.step(",
        "compute_loss(",
    )
    for token in forbidden:
        assert token not in src, (
            f"qualitative_rollout.py contains forbidden token {token!r} -- "
            "this module must remain pure eval"
        )
