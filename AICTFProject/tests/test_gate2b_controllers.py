"""Gate 2B diagnostic controllers: isolation and behaviour tests.

Gate 2B prices the allocation decision by comparing two treatments that must
differ in EXACTLY one respect -- which blue controller is selected. Every
number the gate produces rests on that, so the isolation test comes first.

The V1-era BLUE_RUSH / BLUE_SPLIT styles are deliberately not reused: they
separated home-defense time by only ~0.57 vs ~0.43, far too weak a contrast to
price anything, and they were designed for a ruleset in which a lone defender
could not tag at all.

Measurement rule throughout: read authoritative environment state. Never
recompute geometry, sides, or flag state independently. The engine's midline is
``(cols - 1) * 0.5`` with inclusive bounds; reimplementing it as ``cols * 0.5``
is a half-cell error that already produced a full round of false rule
violations.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402
from gpu_env._core._scripted_blue_styles import (  # noqa: E402
    BLUE_STYLE_NAMES,
    gate2b_defender_hold_radius,
)
from rl.ruleset_identity import fingerprint  # noqa: E402

V2 = dict(taggers_required=1, tag_min_interval_seconds=10.0, tag_nearest_only=True,
          tag_channel_seconds=0.0, suppression_attackers_required=2)
BOTH = "BLUE_BOTH_ATTACK_V2"
ONE = "BLUE_ONE_DEFENDER_V2"
SEED = 1_800_001


def make(style: str, seed: int = SEED, opponent: str = "OP6"):
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2, map_set="train",
        map_layout="map_a", max_decision_steps=240, aquaticus_profile=True,
        rules_profile="OURS", device="cpu", seed=seed, obstacle_obs_channel=True,
        **V2,
    )
    env = GPUCTFVecEnv(cfg)
    env.env_method("set_phase", opponent)
    env.env_method("set_next_opponent", "SCRIPTED", opponent)
    env.core.blue_scripted = True
    env.core.set_blue_style(style)
    obs = env.reset()
    return env, obs


def env_snapshot(core):
    """Everything that must be identical between paired treatments."""
    return {
        "red_x": core.red_x.clone(), "red_y": core.red_y.clone(),
        "blue_x": core.blue_x.clone(), "blue_y": core.blue_y.clone(),
        "red_heading": core.red_heading.clone(), "blue_heading": core.blue_heading.clone(),
        "red_flag_pos": core.red_flag_pos.clone(),
        "blue_flag_pos": core.blue_flag_pos.clone(),
        "red_flag_home": core.red_flag_home.clone(),
        "blue_flag_home": core.blue_flag_home.clone(),
        "blue_score": core.blue_score.clone(), "red_score": core.red_score.clone(),
        "blue_tagged": core.blue_tagged.clone(), "red_tagged": core.red_tagged.clone(),
        "blue_carrying": core.blue_carrying.clone(),
        "red_carrying": core.red_carrying.clone(),
        "step_count": core.step_count.clone(),
        "sim_step_count": core.sim_step_count.clone(),
        "opponent": core.get_opponent_key(),
        "cols": core.cols, "rows": core.rows,
        "max_steps": core.max_steps,
        "ruleset": fingerprint(core.cfg),
    }


def assert_same(a: dict, b: dict, label: str):
    for k, v in a.items():
        if torch.is_tensor(v):
            assert torch.equal(v, b[k]), f"{label}: {k} differs"
        else:
            assert v == b[k], f"{label}: {k} differs ({v!r} vs {b[k]!r})"


# --- Phase 2: registration --------------------------------------------------

def test_both_styles_are_registered():
    assert BOTH in BLUE_STYLE_NAMES and ONE in BLUE_STYLE_NAMES


def test_existing_style_ids_are_unchanged():
    """Appending must never renumber the pre-existing probes."""
    assert BLUE_STYLE_NAMES[:4] == (
        "BLUE_RUSH", "BLUE_TURTLE", "BLUE_SPLIT", "BLUE_ESCORT")


def test_unknown_style_still_rejected():
    env, _ = make(BOTH)
    try:
        with pytest.raises(ValueError):
            env.core.set_blue_style("BLUE_NOT_A_STYLE")
    finally:
        env.close()


def test_hold_radius_is_derived_from_tag_range():
    env, _ = make(ONE)
    try:
        cfg = env.core.cfg
        assert gate2b_defender_hold_radius(cfg) == pytest.approx(
            float(cfg.tag_range_cells) + 0.5)
    finally:
        env.close()


# --- Phase 1: THE isolation test -------------------------------------------

def test_gate2b_style_switch_preserves_paired_environment():
    """Switching treatments must perturb nothing but the blue controller.

    This is the assertion every Gate 2B number depends on. If selecting a style
    also moved the opponent, the seed, or the reset state, the paired contrast
    would be confounded and nothing downstream would mean anything.
    """
    env, _ = make(BOTH)
    try:
        core = env.core
        before = env_snapshot(core)
        core.set_blue_style(ONE)
        after = env_snapshot(core)
        assert_same(before, after, "style switch")
        assert core._blue_style_id != 0
    finally:
        env.close()


def test_paired_arms_have_identical_initial_state_and_observations():
    """Same seed -> byte-identical start, including the first observation."""
    a, obs_a = make(BOTH)
    b, obs_b = make(ONE)
    try:
        assert_same(env_snapshot(a.core), env_snapshot(b.core), "paired reset")
        assert set(obs_a) == set(obs_b), "observation keys differ"
        for k in obs_a:
            assert np.array_equal(np.asarray(obs_a[k]), np.asarray(obs_b[k])), (
                f"initial observation {k!r} differs between arms")
    finally:
        a.close()
        b.close()


def test_ruleset_fingerprint_identical_across_arms():
    a, _ = make(BOTH)
    b, _ = make(ONE)
    try:
        fa, fb = fingerprint(a.core.cfg), fingerprint(b.core.cfg)
        assert fa == fb
        assert fa["ruleset_id"] == "RULESET_V2_AQUATICUS_10S"
    finally:
        a.close()
        b.close()


def test_opponent_and_map_unaffected_by_style_selection():
    env, _ = make(BOTH, opponent="OP9")
    try:
        core = env.core
        assert core.get_opponent_key().strip().upper() == "OP9"
        cols, rows = core.cols, core.rows
        core.set_blue_style(ONE)
        assert core.get_opponent_key().strip().upper() == "OP9"
        assert (core.cols, core.rows) == (cols, rows)
        assert fingerprint(core.cfg)["ruleset_id"] == "RULESET_V2_AQUATICUS_10S"
    finally:
        env.close()


# --- Phase 3: controller behaviour -----------------------------------------

def _roll(style: str, steps: int = 120, seed: int = SEED):
    env, _ = make(style, seed=seed)
    core = env.core
    try:
        hx = float(core.blue_flag_home[0, 0])
        hy = float(core.blue_flag_home[0, 1])
        d1, fwd, a1_on_red = [], [], []
        for _ in range(steps):
            env.step_async(env.action_space.sample() * 0)
            _o, _r, done, _i = env.step_wait()
            bx, by = core.blue_x, core.blue_y
            d1.append(float(torch.sqrt((bx[0, 1] - hx) ** 2 + (by[0, 1] - hy) ** 2)))
            on_enemy = core._is_on_home_side("red", bx)[0]
            fwd.append(float(on_enemy.float().sum()))
            a1_on_red.append(bool(on_enemy[1]))
            if bool(np.asarray(done).any()):
                break
        return {"mean_d1": float(np.mean(d1)), "max_d1": float(np.max(d1)),
                "mean_fwd": float(np.mean(fwd)),
                "a1_red_frac": float(np.mean(a1_on_red))}
    finally:
        env.close()


def test_both_attack_sends_both_agents_forward():
    r = _roll(BOTH)
    assert r["mean_fwd"] > 0.0, "BOTH_ATTACK never commits forward"
    assert r["a1_red_frac"] > 0.0, "BOTH_ATTACK agent1 never enters red territory"


def test_one_defender_keeps_agent1_near_home():
    r = _roll(ONE)
    env, _ = make(ONE)
    try:
        radius = gate2b_defender_hold_radius(env.core.cfg)
    finally:
        env.close()
    # Targets are clamped to the disc; allow slack for in-flight kinematics.
    assert r["mean_d1"] <= radius * 3.0, (
        f"defender strayed: mean distance from home {r['mean_d1']:.2f} "
        f"(hold radius {radius:.2f})")


def test_defender_does_not_chase_into_red_territory():
    r = _roll(ONE)
    assert r["a1_red_frac"] == 0.0, (
        f"defender entered red territory on {r['a1_red_frac']:.1%} of steps")


def test_defender_intercepts_intruder_then_returns_home():
    """Reacts to a legal intruder, and returns to the zone once it leaves."""
    env, _ = make(ONE)
    core = env.core
    try:
        hx = float(core.blue_flag_home[0, 0])
        hy = float(core.blue_flag_home[0, 1])
        # Plant a live, untagged red intruder inside blue territory near home.
        core.red_x[0, 0] = hx + 3.0
        core.red_y[0, 0] = hy
        core.red_alive[0, 0] = True
        core.red_tagged[0, 0] = False
        tx, _ty = core._assign_blue_style_targets()
        toward_intruder = float(tx[0, 1])
        # Now remove the intruder to red territory; defender should fall back.
        core.red_x[0, 0] = float(core.cols - 1)
        tx2, _ty2 = core._assign_blue_style_targets()
        back_home = float(tx2[0, 1])
        assert toward_intruder > back_home, (
            f"defender did not advance toward the intruder "
            f"(intruder target x={toward_intruder:.2f}, idle x={back_home:.2f})")
        assert abs(back_home - hx) <= gate2b_defender_hold_radius(core.cfg) + 1e-6, (
            "defender did not return to the defensive zone once the intruder left")
    finally:
        env.close()


def test_treatments_separate_home_defense_and_forward_commitment():
    """The manipulation must be strong, not nominal."""
    both, one = _roll(BOTH), _roll(ONE)
    assert one["mean_d1"] < both["mean_d1"], (
        f"defender not closer to home: ONE={one['mean_d1']:.2f} BOTH={both['mean_d1']:.2f}")
    assert both["mean_fwd"] > one["mean_fwd"], (
        f"forward commitment did not separate: BOTH={both['mean_fwd']:.3f} "
        f"ONE={one['mean_fwd']:.3f}")


def test_controllers_are_deterministic():
    """Same seed and style must reproduce exactly -- the paired design needs it."""
    assert _roll(ONE, steps=60) == _roll(ONE, steps=60)
    assert _roll(BOTH, steps=60) == _roll(BOTH, steps=60)


def test_styles_do_not_mutate_ruleset_map_or_seed():
    env, _ = make(BOTH)
    core = env.core
    try:
        before = (fingerprint(core.cfg), core.cols, core.rows,
                  core.max_steps, core.get_opponent_key())
        for _ in range(20):
            env.step_async(env.action_space.sample() * 0)
            env.step_wait()
        core.set_blue_style(ONE)
        for _ in range(20):
            env.step_async(env.action_space.sample() * 0)
            env.step_wait()
        after = (fingerprint(core.cfg), core.cols, core.rows,
                 core.max_steps, core.get_opponent_key())
        assert before == after
    finally:
        env.close()
