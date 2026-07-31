"""Gate 2D manipulation metric: mean_num_home_defenders.

Gate 2C failed manipulation at +0.2257 against a 0.25 floor. That was the ruler,
not the treatment: ``blue_home_defense_fraction`` is a binary per-step "is anyone
near home" checkbox, so ONE_DEFENDER sits near its ceiling while BOTH_ATTACK
still registers home presence during flag returns, tags and resets.

Gate 2D counts VEHICLES instead, symmetric with the attacker metric. A blue agent
counts as a home defender only when it is:

    alive
    NOT carrying the enemy flag
    in authoritative BLUE territory
    within the declared home-defense zone

These tests pin each clause independently, so a future refactor cannot quietly
drop one and inflate the manipulation contrast.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402

V2 = dict(taggers_required=1, tag_min_interval_seconds=10.0, tag_nearest_only=True,
          tag_channel_seconds=0.0, suppression_attackers_required=2)
HOME_DEFENSE_RADIUS = 8.0


@pytest.fixture
def core():
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2, map_set="train",
        map_layout="map_a", max_decision_steps=240, aquaticus_profile=True,
        rules_profile="OURS", device="cpu", seed=2_100_001,
        obstacle_obs_channel=True, **V2)
    env = GPUCTFVecEnv(cfg)
    env.reset()
    yield env.core
    env.close()


def count_home_defenders(core) -> int:
    """Exactly the Gate 2D definition, mirrored for test purposes."""
    bx = core.blue_x
    hf = core.blue_flag_home[0]
    d_home = np.hypot((bx[0] - hf[0]).detach().cpu().numpy(),
                      (core.blue_y[0] - hf[1]).detach().cpu().numpy())
    in_zone = d_home <= HOME_DEFENSE_RADIUS
    in_own = core._is_on_home_side("blue", bx)[0].detach().cpu().numpy().astype(bool)
    alive = core.blue_alive[0].detach().cpu().numpy().astype(bool)
    carrying = core.blue_carrying[0].detach().cpu().numpy().astype(bool)
    return int((in_zone & in_own & alive & (~carrying)).sum())


def place(core, i, x, y, *, alive=True, carrying=False):
    core.blue_x[0, i] = float(x)
    core.blue_y[0, i] = float(y)
    core.blue_alive[0, i] = bool(alive)
    core.blue_carrying[0, i] = bool(carrying)


def home(core):
    return float(core.blue_flag_home[0, 0]), float(core.blue_flag_home[0, 1])


def far_away(core):
    """A point comfortably outside the home zone, still in blue territory."""
    hx, hy = home(core)
    return hx + HOME_DEFENSE_RADIUS + 3.0, hy


def test_agent_on_home_counts(core):
    hx, hy = home(core)
    place(core, 0, hx, hy)
    place(core, 1, *far_away(core))
    assert count_home_defenders(core) == 1


def test_both_agents_home_count_two(core):
    hx, hy = home(core)
    place(core, 0, hx, hy)
    place(core, 1, hx + 1.0, hy)
    assert count_home_defenders(core) == 2, (
        "metric must be in AGENT units, not a 0/1 checkbox")


def test_metric_is_not_binary(core):
    """The whole point of Gate 2D: 2 defenders must outrank 1."""
    hx, hy = home(core)
    place(core, 0, hx, hy)
    place(core, 1, *far_away(core))
    one = count_home_defenders(core)
    place(core, 1, hx + 1.0, hy)
    two = count_home_defenders(core)
    assert two > one, "counting vehicles must distinguish one defender from two"


def test_dead_agent_does_not_count(core):
    hx, hy = home(core)
    place(core, 0, hx, hy, alive=False)
    place(core, 1, *far_away(core))
    assert count_home_defenders(core) == 0


def test_carrier_does_not_count(core):
    """A carrier standing on home is returning a flag, not defending."""
    hx, hy = home(core)
    place(core, 0, hx, hy, carrying=True)
    place(core, 1, *far_away(core))
    assert count_home_defenders(core) == 0


def test_outside_zone_does_not_count(core):
    place(core, 0, *far_away(core))
    place(core, 1, *far_away(core))
    assert count_home_defenders(core) == 0


def test_enemy_territory_does_not_count(core):
    """Uses the engine's own side predicate, not a recomputed midline."""
    mid_col = float(core.cols - 1) * 0.5
    place(core, 0, mid_col + 3.0, 10.0)
    place(core, 1, mid_col + 4.0, 10.0)
    assert not bool(core._is_on_home_side("blue", core.blue_x)[0].any())
    assert count_home_defenders(core) == 0


def test_zone_boundary_is_inclusive(core):
    """Exactly on the radius counts.

    Offset along Y, not X: blue home sits at x=2 and the midline is
    (cols-1)*0.5 = 9.5, so hx + 8.0 would land in RED territory and be rejected
    by the territory clause rather than the zone clause.
    """
    hx, hy = home(core)
    place(core, 0, hx, hy + HOME_DEFENSE_RADIUS)
    place(core, 1, *far_away(core))
    assert bool(core._is_on_home_side("blue", core.blue_x)[0][0]), (
        "fixture must keep the boundary agent in blue territory")
    assert count_home_defenders(core) == 1, "boundary must be inclusive"


def test_all_four_clauses_are_required(core):
    """Drop any one clause and the count must change -- no silent inflation."""
    hx, hy = home(core)
    place(core, 0, hx, hy)
    place(core, 1, *far_away(core))
    base = count_home_defenders(core)
    assert base == 1
    for mutate in (
        lambda: place(core, 0, hx, hy, alive=False),
        lambda: place(core, 0, hx, hy, carrying=True),
        lambda: place(core, 0, *far_away(core)),
    ):
        mutate()
        assert count_home_defenders(core) < base
        place(core, 0, hx, hy)   # restore
