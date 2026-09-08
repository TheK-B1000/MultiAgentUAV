"""Gate 2B legal-capture progress metric.

The first implementation credited 2.0 whenever a carrier stood near its own
home, so every episode saturated and both primary contrasts came out exactly
zero -- Gate 2's floor problem inverted into a ceiling.

The corrected metric measures progress toward a LEGAL capture:

    2.0            reserved for an actual capture (authoritative score delta)
    1.0 - 2.0      returning WHILE our own flag is safely home
    1.0            enemy flag possessed but scoring may be blocked
    0.0 - 1.0      approaching the enemy flag

A carrier that reaches home while its own flag is missing cannot score, so it
must not be credited as if it had.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

from experiments.gate2b_affordance_scenarios_v2 import (  # noqa: E402
    CAPTURE_PROGRESS, NON_CAPTURE_CAP, TeamProgress,
)
from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402

V2 = dict(taggers_required=1, tag_min_interval_seconds=10.0, tag_nearest_only=True,
          tag_channel_seconds=0.0, suppression_attackers_required=2)


@pytest.fixture
def env():
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2, map_set="train",
        map_layout="map_a", max_decision_steps=240, aquaticus_profile=True,
        rules_profile="OURS", device="cpu", seed=1_900_001,
        obstacle_obs_channel=True, **V2)
    e = GPUCTFVecEnv(cfg)
    e.reset()
    yield e
    e.close()


def tracker(env):
    tp = TeamProgress(env.core, "blue")
    tp.anchor()
    return tp


def put_carrier_at_home(core, *, own_flag_home: bool):
    """Blue agent 0 carries the enemy flag and stands on its own home flag."""
    hx = float(core.blue_flag_home[0, 0])
    hy = float(core.blue_flag_home[0, 1])
    core.blue_x[0, 0] = hx
    core.blue_y[0, 0] = hy
    core.blue_alive[0, :] = True
    core.blue_carrying[0, 0] = True
    core.blue_score[0] = 0
    if own_flag_home:
        core.blue_flag_pos[0, 0] = hx
        core.blue_flag_pos[0, 1] = hy
    else:
        # Our own flag has been stolen and is out on the field.
        core.blue_flag_pos[0, 0] = hx + 8.0
        core.blue_flag_pos[0, 1] = hy + 3.0


def test_carrier_at_home_with_own_flag_away_is_below_capture(env):
    core = env.core
    tp = tracker(env)
    put_carrier_at_home(core, own_flag_home=False)
    p = tp.sample()
    assert p < CAPTURE_PROGRESS, f"illegal capture credited {p}"
    # Return credit is gated off entirely while our flag is missing.
    assert p <= 1.0 + 1e-9, f"return credited despite missing own flag: {p}"


def test_carrier_at_home_with_own_flag_home_is_still_below_capture(env):
    """Even a fully legal return must not equal 2.0 without an actual score."""
    core = env.core
    tp = tracker(env)
    put_carrier_at_home(core, own_flag_home=True)
    p = tp.sample()
    assert p < CAPTURE_PROGRESS, f"non-capture reached {p}"
    assert p <= NON_CAPTURE_CAP + 1e-12
    assert p > 1.0, "a legal return should still earn return credit"


def test_actual_score_increase_is_exactly_capture(env):
    core = env.core
    tp = tracker(env)
    put_carrier_at_home(core, own_flag_home=True)
    core.blue_score[0] = 1
    assert tp.sample() == pytest.approx(CAPTURE_PROGRESS)


def test_pickup_then_drop_preserves_earlier_maximum(env):
    core = env.core
    tp = tracker(env)
    put_carrier_at_home(core, own_flag_home=True)
    peak = tp.sample()
    assert peak > 1.0
    # Carrier is tagged and drops the flag well away from home.
    core.blue_carrying[0, 0] = False
    core.blue_x[0, 0] = float(core.red_flag_pos[0, 0])
    core.blue_y[0, 0] = float(core.red_flag_pos[0, 1])
    tp.sample()
    assert tp.max_progress == pytest.approx(peak), (
        "episode maximum must survive a later drop")


def test_no_carrier_cannot_enter_return_branch(env):
    """Standing on home without the enemy flag earns approach credit only."""
    core = env.core
    tp = tracker(env)
    core.blue_carrying[0, :] = False
    core.blue_alive[0, :] = True
    core.blue_score[0] = 0
    core.blue_x[0, :] = core.blue_flag_home[0, 0]
    core.blue_y[0, :] = core.blue_flag_home[0, 1]
    p = tp.sample()
    assert p <= 1.0, f"non-carrier entered the return branch: {p}"


def test_raw_return_fraction_records_illegal_return(env):
    """Descriptive evidence survives even when the gated metric refuses credit."""
    core = env.core
    tp = tracker(env)
    put_carrier_at_home(core, own_flag_home=False)
    tp.sample()
    assert tp.max_raw_return_fraction > 0.9, (
        "raw carrier return should record that the carrier physically got home")
    assert tp.max_progress <= 1.0 + 1e-9, "but gated progress must not credit it"


def test_progress_is_monotone_nondecreasing(env):
    core = env.core
    tp = tracker(env)
    seen = []
    for frac in (0.0, 0.5, 1.0):
        rf = core.red_flag_pos[0]
        sx = float(core.blue_x[0, 0])
        core.blue_x[0, 0] = sx + (float(rf[0]) - sx) * frac
        tp.sample()
        seen.append(tp.max_progress)
    assert seen == sorted(seen), f"max progress went backwards: {seen}"
