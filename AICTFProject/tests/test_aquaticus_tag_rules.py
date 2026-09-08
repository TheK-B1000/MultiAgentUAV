"""Deterministic tag-rule tests for the Aquaticus-faithful ruleset (RULESET_V2).

RULESET_V1 required two simultaneous taggers and had no cooldown. In 2v2 that
made a lone defender strictly dominated -- it could neither tag nor suppress --
which removed the opportunity cost of committing both agents forward and
collapsed the strategy space onto a single non-dominated policy.

Official Aquaticus: one eligible defender tags by itself, the NEAREST eligible
opponent receives the tag (so a teammate can absorb one to protect a carrier),
and a successful tagger must wait a minimum interval before tagging again.

These tests drive the tag rule directly on placed positions -- no policy, no
opponent behaviour tree, no GPU training -- so each rule is pinned independently.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402


def make_env(**overrides):
    cfg_kwargs = dict(
        n_envs=1, max_blue_agents=2, max_red_agents=2,
        map_set="train", map_layout="map_a",
        max_decision_steps=240, aquaticus_profile=True, rules_profile="OURS",
        device="cpu", seed=12345,
    )
    cfg_kwargs.update(overrides)
    env = GPUCTFVecEnv(GPUFieldConfig(**cfg_kwargs))
    env.reset()
    return env


def place(core, *, blue, red, carrying_blue=None):
    """Position agents deterministically. `blue`/`red` are [(x, y), ...]."""
    for i, (x, y) in enumerate(blue):
        core.blue_x[0, i] = float(x)
        core.blue_y[0, i] = float(y)
    for i, (x, y) in enumerate(red):
        core.red_x[0, i] = float(x)
        core.red_y[0, i] = float(y)
    core.blue_tagged[0, :] = False
    core.red_tagged[0, :] = False
    core.blue_carrying[0, :] = False
    core.red_carrying[0, :] = False
    core.blue_tag_pressure_time[0, :] = 0.0
    core.red_tag_pressure_time[0, :] = 0.0
    core.blue_tag_cooldown[0, :] = 0.0
    core.red_tag_cooldown[0, :] = 0.0
    if carrying_blue is not None:
        core.blue_carrying[0, carrying_blue] = True


def fire(core):
    """Apply the tag rule once with no out-of-bounds agents."""
    z_b = torch.zeros_like(core.blue_tagged)
    z_r = torch.zeros_like(core.red_tagged)
    core._apply_aquaticus_tag_rules(z_b, z_r)


@pytest.fixture
def geom():
    """Coordinates: blue home is low x, red home is high x; midline = cols/2."""
    env = make_env()
    core = env.core
    mid = float(core.cols) * 0.5
    yield core, mid, env
    env.close()


# --- 1-2: a single defender suffices; two are not required ------------------

def test_one_defender_can_tag_one_intruder(geom):
    core, mid, _ = geom
    # One blue defender on blue's side; one red intruder next to it, also on
    # blue's side. Second blue is parked far away and cannot contribute.
    place(core, blue=[(mid - 2.0, 10.0), (1.0, 1.0)],
          red=[(mid - 1.0, 10.0), (mid + 20.0, 30.0)])
    fire(core)
    assert bool(core.red_tagged[0, 0]), "a lone eligible defender must be able to tag"


def test_two_defenders_are_not_required(geom):
    core, mid, _ = geom
    place(core, blue=[(mid - 2.0, 10.0), (1.0, 1.0)],
          red=[(mid - 1.0, 10.0), (mid + 20.0, 30.0)])
    fire(core)
    n_blue_in_range = int(((core.blue_x[0] - core.red_x[0, 0]).abs() <= 2.5).sum())
    assert n_blue_in_range == 1, "fixture must have exactly one defender in range"
    assert bool(core.red_tagged[0, 0])


# --- 3-4: eligibility ------------------------------------------------------

def test_defender_cannot_tag_from_enemy_side(geom):
    core, mid, _ = geom
    # Blue defender is on RED's side, so it is not an eligible tagger; the red
    # agent beside it is on its own side and not targetable either.
    place(core, blue=[(mid + 2.0, 10.0), (1.0, 1.0)],
          red=[(mid + 1.0, 10.0), (mid + 20.0, 30.0)])
    fire(core)
    assert not bool(core.red_tagged[0, 0]), "tagging from the enemy side must not work"


def test_tagged_defender_cannot_tag(geom):
    core, mid, _ = geom
    place(core, blue=[(mid - 2.0, 10.0), (1.0, 1.0)],
          red=[(mid - 1.0, 10.0), (mid + 20.0, 30.0)])
    core.blue_tagged[0, 0] = True
    fire(core)
    assert not bool(core.red_tagged[0, 0]), "a tagged vehicle must not be able to tag"


# --- 5: nearest eligible target --------------------------------------------

def test_nearest_eligible_intruder_is_tagged(geom):
    core, mid, _ = geom
    # Two red intruders in range of one blue defender; only the NEAREST is
    # tagged, which is how a teammate absorbs a tag for the carrier.
    place(core, blue=[(mid - 3.0, 10.0), (1.0, 1.0)],
          red=[(mid - 2.5, 10.0), (mid - 1.5, 10.0)])
    d0 = abs(float(core.red_x[0, 0]) - float(core.blue_x[0, 0]))
    d1 = abs(float(core.red_x[0, 1]) - float(core.blue_x[0, 0]))
    nearer, farther = (0, 1) if d0 < d1 else (1, 0)
    fire(core)
    assert bool(core.red_tagged[0, nearer]), "nearest eligible target must be tagged"
    assert not bool(core.red_tagged[0, farther]), "farther target must be spared"


# --- 6-8: cooldown ---------------------------------------------------------

def test_successful_tag_starts_cooldown(geom):
    core, mid, _ = geom
    place(core, blue=[(mid - 2.0, 10.0), (1.0, 1.0)],
          red=[(mid - 1.0, 10.0), (mid + 20.0, 30.0)])
    fire(core)
    assert bool(core.red_tagged[0, 0])
    assert float(core.blue_tag_cooldown[0, 0]) > 0.0, "tagger must go on cooldown"


def test_second_tag_denied_during_cooldown(geom):
    core, mid, _ = geom
    place(core, blue=[(mid - 2.0, 10.0), (1.0, 1.0)],
          red=[(mid - 1.0, 10.0), (mid - 3.0, 10.0)])
    fire(core)
    assert float(core.blue_tag_cooldown[0, 0]) > 0.0
    # Reset targets' tagged state; the defender is still on cooldown.
    core.red_tagged[0, :] = False
    core.red_tag_pressure_time[0, :] = 0.0
    fire(core)
    assert not bool(core.red_tagged[0].any()), "cooldown must deny a second tag"


def test_tag_works_again_after_cooldown(geom):
    core, mid, _ = geom
    place(core, blue=[(mid - 2.0, 10.0), (1.0, 1.0)],
          red=[(mid - 1.0, 10.0), (mid + 20.0, 30.0)])
    fire(core)
    assert bool(core.red_tagged[0, 0])
    core.red_tagged[0, :] = False
    core.red_tag_pressure_time[0, :] = 0.0
    core.blue_tag_cooldown[0, :] = 0.0  # cooldown elapsed
    fire(core)
    assert bool(core.red_tagged[0, 0]), "tagging must resume after the interval"


# --- 9-10: consequences of being tagged ------------------------------------

def test_tagged_carrier_drops_the_flag(geom):
    core, mid, _ = geom
    # Blue agent 0 carries the red flag but is caught on RED's side by a red
    # defender that is on its own side.
    place(core, blue=[(mid + 2.0, 10.0), (1.0, 1.0)],
          red=[(mid + 3.0, 10.0), (mid + 20.0, 30.0)], carrying_blue=0)
    assert bool(core.blue_carrying[0, 0])
    fire(core)
    assert bool(core.blue_tagged[0, 0]), "carrier on enemy side must be taggable"
    assert not bool(core.blue_carrying[0, 0]), "a tagged carrier must drop the flag"


def test_tagged_agent_untags_at_home(geom):
    core, mid, _ = geom
    place(core, blue=[(mid + 2.0, 10.0), (1.0, 1.0)],
          red=[(mid + 3.0, 10.0), (mid + 20.0, 30.0)])
    fire(core)
    assert bool(core.blue_tagged[0, 0])
    # Move the tagged agent onto its home flag; the untag rule must clear it.
    core.blue_x[0, 0] = float(core.blue_flag_home[0, 0])
    core.blue_y[0, 0] = float(core.blue_flag_home[0, 1])
    core._untag_if_home()
    assert not bool(core.blue_tagged[0, 0]), "returning home must clear the tag"


# --- 11: thresholds stay decoupled -----------------------------------------

def test_tag_threshold_does_not_alter_suppression_threshold():
    """Correcting tagging to 1 must not silently make suppression 1-agent."""
    env = make_env(taggers_required=1)
    try:
        assert int(env.core.cfg.taggers_required) == 1
        assert int(env.core.cfg.suppression_attackers_required) == 2, (
            "suppression is a separate project mechanic and must keep its own threshold"
        )
    finally:
        env.close()


# --- RULESET_V1 reproducibility --------------------------------------------

def test_ruleset_v1_is_reproducible():
    """The superseded two-tagger rule must still be expressible for old runs."""
    env = make_env(taggers_required=2, tag_nearest_only=False,
                   tag_min_interval_seconds=0.0, tag_channel_seconds=1.0)
    core = env.core
    try:
        mid = float(core.cols) * 0.5
        place(core, blue=[(mid - 2.0, 10.0), (1.0, 1.0)],
              red=[(mid - 1.0, 10.0), (mid + 20.0, 30.0)])
        fire(core)
        assert not bool(core.red_tagged[0, 0]), (
            "under RULESET_V1 a lone defender must NOT be able to tag"
        )
    finally:
        env.close()
