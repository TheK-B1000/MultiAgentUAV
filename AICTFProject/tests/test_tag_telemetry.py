"""Safety tests for tag-event telemetry.

Gate 1 originally reconstructed tags from post-step positions. That is
temporally invalid: by the time the step returns, the target has been
redirected home, the tagger has moved, cooldowns have been armed, and flags
have been dropped. The checker consequently reported violations that the
rule-level tests directly disprove.

These tests pin the replacement: events emitted INSIDE the tag rule at the
decision point, and -- above all -- telemetry that cannot change the game.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402

V2 = dict(taggers_required=1, tag_nearest_only=True,
          tag_min_interval_seconds=10.0, tag_channel_seconds=0.0,
          suppression_attackers_required=2)
V1 = dict(taggers_required=2, tag_nearest_only=False,
          tag_min_interval_seconds=0.0, tag_channel_seconds=1.0,
          suppression_attackers_required=2)


def make(telemetry: bool, n_envs: int = 1, rules=None, seed: int = 4242):
    cfg = GPUFieldConfig(
        n_envs=n_envs, max_blue_agents=2, max_red_agents=2, map_set="train",
        map_layout="map_a", max_decision_steps=240, aquaticus_profile=True,
        rules_profile="OURS", device="cpu", seed=seed,
        tag_telemetry_enabled=telemetry, **(rules or V2),
    )
    env = GPUCTFVecEnv(cfg)
    env.reset()
    return env


def stage_tag(core, env_i: int = 0):
    """One blue defender on its own side, one red intruder inside tag range."""
    mid = float(core.cols) * 0.5
    core.blue_x[env_i, 0] = mid - 2.0
    core.blue_y[env_i, 0] = 10.0
    core.blue_x[env_i, 1] = 1.0
    core.blue_y[env_i, 1] = 1.0
    core.red_x[env_i, 0] = mid - 1.0
    core.red_y[env_i, 0] = 10.0
    core.red_x[env_i, 1] = mid + 20.0
    core.red_y[env_i, 1] = 30.0
    core.blue_tagged[env_i, :] = False
    core.red_tagged[env_i, :] = False
    core.blue_tag_cooldown[env_i, :] = 0.0
    core.red_tag_cooldown[env_i, :] = 0.0
    core.blue_tag_pressure_time[env_i, :] = 0.0
    core.red_tag_pressure_time[env_i, :] = 0.0


def fire(core):
    core._apply_aquaticus_tag_rules(torch.zeros_like(core.blue_tagged),
                                    torch.zeros_like(core.red_tagged))


def successes(events):
    return [e for e in events if e["event_type"] == "tag_success"]


def denials(events):
    return [e for e in events if e["event_type"] == "tag_denied"]


# --- behaviour neutrality (the load-bearing requirement) --------------------

def test_telemetry_is_behaviour_neutral():
    """Same seed, telemetry on vs off -> identical states and outcomes."""
    outs = []
    for tel in (False, True):
        env = make(tel, seed=999)
        core = env.core
        try:
            rng = np.random.default_rng(0)
            trace = []
            for _ in range(60):
                act = rng.integers(0, 2, size=env.action_space.shape).astype(np.int64) * 0
                env.step_async(act)
                _o, rew, done, _i = env.step_wait()
                trace.append((
                    core.blue_x[0].clone(), core.blue_y[0].clone(),
                    core.red_x[0].clone(), core.red_y[0].clone(),
                    core.blue_tagged[0].clone(), core.red_tagged[0].clone(),
                    int(core.blue_score[0]), int(core.red_score[0]),
                    float(np.asarray(rew).sum()), bool(np.asarray(done).any()),
                ))
                if bool(np.asarray(done).any()):
                    break
            outs.append(trace)
        finally:
            env.close()

    off, on = outs
    assert len(off) == len(on), "telemetry changed episode length"
    for t, (a, b) in enumerate(zip(off, on)):
        for k in range(6):
            assert torch.equal(a[k], b[k]), f"tensor {k} diverged at step {t}"
        assert a[6:] == b[6:], f"scores/reward/done diverged at step {t}"


def test_disabled_telemetry_emits_nothing():
    env = make(False)
    core = env.core
    try:
        stage_tag(core)
        fire(core)
        assert core.drain_tag_events() == []
    finally:
        env.close()


# --- exactness of the success event ----------------------------------------

def test_exactly_one_success_event_per_tag():
    env = make(True)
    core = env.core
    try:
        core.drain_tag_events()
        stage_tag(core)
        fire(core)
        ev = successes(core.drain_tag_events())
        assert len(ev) == 1, f"expected exactly one success event, got {len(ev)}"
    finally:
        env.close()


def test_success_event_records_decision_point_facts():
    env = make(True)
    core = env.core
    try:
        core.drain_tag_events()
        stage_tag(core)
        gx = float(core.blue_x[0, 0])
        tx = float(core.red_x[0, 0])
        fire(core)
        e = successes(core.drain_tag_events())[0]
        assert e["tagger_team"] == "blue" and e["tagger_index"] == 0
        assert e["target_team"] == "red" and e["target_index"] == 0
        # Positions are the DECISION positions, not post-step ones.
        assert e["tagger_position_at_decision"][0] == pytest.approx(gx)
        assert e["target_position_at_decision"][0] == pytest.approx(tx)
        assert e["tagger_on_own_side"] is True
        assert e["target_on_tagger_side"] is True
        assert e["distance_at_decision"] == pytest.approx(abs(gx - tx), abs=1e-3)
        assert e["target_was_tagged"] is False
        assert e["selected_nearest_target"] == 0
        assert e["ruleset_id"] == "RULESET_V2_AQUATICUS_10S"
    finally:
        env.close()


def test_nearest_target_identity_matches_rule_decision():
    env = make(True)
    core = env.core
    try:
        core.drain_tag_events()
        mid = float(core.cols) * 0.5
        core.blue_x[0, 0] = mid - 3.0
        core.blue_y[0, 0] = 10.0
        core.blue_x[0, 1] = 1.0
        core.blue_y[0, 1] = 1.0
        core.red_x[0, 0] = mid - 2.5   # nearer
        core.red_y[0, 0] = 10.0
        core.red_x[0, 1] = mid - 1.5   # farther
        core.red_y[0, 1] = 10.0
        core.blue_tagged[0, :] = False
        core.red_tagged[0, :] = False
        core.blue_tag_cooldown[0, :] = 0.0
        core.blue_tag_pressure_time[0, :] = 0.0
        core.red_tag_pressure_time[0, :] = 0.0
        fire(core)
        ev = successes(core.drain_tag_events())
        assert len(ev) == 1
        assert ev[0]["selected_nearest_target"] == 0
        assert bool(core.red_tagged[0, 0]) and not bool(core.red_tagged[0, 1])
    finally:
        env.close()


# --- cooldown denial --------------------------------------------------------

def test_cooldown_denial_event_and_no_tag():
    env = make(True)
    core = env.core
    try:
        stage_tag(core)
        fire(core)
        core.drain_tag_events()
        # Same geometry, but the tagger is now on cooldown.
        core.red_tagged[0, :] = False
        core.red_tag_pressure_time[0, :] = 0.0
        assert float(core.blue_tag_cooldown[0, 0]) > 0.0
        fire(core)
        ev = core.drain_tag_events()
        assert not bool(core.red_tagged[0].any()), "cooldown must deny the tag"
        d = denials(ev)
        assert d, "a cooldown denial event must be emitted"
        assert d[0]["reason"] == "cooldown"
        assert d[0]["tagger_index"] == 0
        assert d[0]["cooldown_remaining"] > 0.0
        assert not successes(ev), "no success event may accompany a denial"
    finally:
        env.close()


# --- buffer hygiene ---------------------------------------------------------

def test_drain_clears_buffer_and_no_duplicates():
    env = make(True)
    core = env.core
    try:
        core.drain_tag_events()
        stage_tag(core)
        fire(core)
        first = core.drain_tag_events()
        assert len(successes(first)) == 1
        assert core.drain_tag_events() == [], "drain must clear the buffer"
    finally:
        env.close()


def test_env_index_correct_with_vectorized_envs():
    env = make(True, n_envs=3)
    core = env.core
    try:
        core.drain_tag_events()
        stage_tag(core, env_i=2)   # stage a tag ONLY in env 2
        fire(core)
        ev = successes(core.drain_tag_events())
        assert ev, "expected a tag in env 2"
        assert {e["env_index"] for e in ev} == {2}, \
            f"env_index wrong: {[e['env_index'] for e in ev]}"
    finally:
        env.close()


def test_reset_delimits_events():
    env = make(True)
    core = env.core
    try:
        core.drain_tag_events()
        stage_tag(core)
        fire(core)
        env.reset()
        ev = core.drain_tag_events()
        kinds = [e["event_type"] for e in ev]
        assert "tag_success" in kinds, "reset must not discard the finished episode"
        assert "episode_reset" in kinds, "reset must delimit the buffer"
        assert kinds.index("tag_success") < kinds.index("episode_reset"), \
            "event ordering must be stable"
    finally:
        env.close()


def test_v1_and_v2_share_the_telemetry_schema():
    keys = {}
    for label, rules in (("V2", V2), ("V1", V1)):
        env = make(True, rules=rules)
        core = env.core
        try:
            core.drain_tag_events()
            stage_tag(core)
            # V1 needs two taggers sustained; drive both in and step the channel.
            if label == "V1":
                mid = float(core.cols) * 0.5
                core.blue_x[0, 1] = mid - 2.2
                core.blue_y[0, 1] = 10.0
                for _ in range(200):
                    fire(core)
                    if bool(core.red_tagged[0, 0]):
                        break
            else:
                fire(core)
            ev = successes(core.drain_tag_events())
            assert ev, f"{label}: expected a tag to exercise the schema"
            keys[label] = set(ev[0].keys())
        finally:
            env.close()
    assert keys["V1"] == keys["V2"], (
        f"schema differs between rulesets: {keys['V1'] ^ keys['V2']}")


def test_ruleset_id_and_fields():
    env = make(False, rules=V2)
    try:
        assert env.core.cfg.ruleset_id == "RULESET_V2_AQUATICUS_10S"
        f = env.core.cfg.ruleset_fields()
        assert f["taggers_required"] == 1
        assert f["tag_min_interval_seconds"] == 10.0
        assert f["tag_nearest_only"] is True
        assert f["tag_channel_seconds"] == 0.0
        assert f["suppression_attackers_required"] == 2
    finally:
        env.close()
    env = make(False, rules=V1)
    try:
        assert env.core.cfg.ruleset_id == "RULESET_V1_TWO_TAGGER"
    finally:
        env.close()
