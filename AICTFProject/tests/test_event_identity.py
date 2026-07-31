"""Authoritative integer event identity.

Every event carries identity produced AT THE SOURCE:

    env_index  episode_id  reset_sequence
    simulation_step  decision_step  event_sequence  terminal_step

Consumers must never re-derive episode boundaries by counting reset markers.
That reconstruction is exactly what let identities collide across episodes and
produced phantom duplicate / "denied and succeeded at the same instant"
findings in Gate 1 -- five consecutive instrumentation bugs traced to consumers
reimplementing what the engine already knows.

Telemetry remains observational: identical simulation tensors with it on or off.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402

V2 = dict(taggers_required=1, tag_min_interval_seconds=10.0, tag_nearest_only=True,
          tag_channel_seconds=0.0, suppression_attackers_required=2)

IDENTITY_FIELDS = ("env_index", "episode_id", "reset_sequence", "simulation_step",
                   "decision_step", "event_sequence", "terminal_step")


def make(n_envs=1, telemetry=True, seed=2_300_001, opponent="OP6"):
    cfg = GPUFieldConfig(
        n_envs=n_envs, max_blue_agents=2, max_red_agents=2, map_set="train",
        map_layout="map_a", max_decision_steps=240, aquaticus_profile=True,
        rules_profile="OURS", device="cpu", seed=seed, obstacle_obs_channel=True,
        tag_telemetry_enabled=telemetry, **V2)
    env = GPUCTFVecEnv(cfg)
    env.env_method("set_phase", opponent)
    env.env_method("set_next_opponent", "SCRIPTED", opponent)
    env.core.blue_scripted = True
    env.core.set_blue_style("BLUE_BOTH_ATTACK_V2")
    env.reset()
    return env


def zero_action(env):
    """Zero action sized for the vec env (B * Nb * heads), not a single env."""
    core = env.core
    return np.zeros(int(core.B) * int(core.Nb) * 2, dtype=np.int64)


def drive(env, steps):
    """Run and collect every event, preserving emission order."""
    core = env.core
    core.drain_tag_events()
    out = []
    act = zero_action(env)
    for _ in range(steps):
        env.step_async(act)
        env.step_wait()
        out.extend(core.drain_tag_events())
    return out


def substantive(events):
    return [e for e in events if e["event_type"] != "episode_reset"]


# --- identity presence and shape -------------------------------------------

def test_every_event_carries_full_integer_identity():
    env = make()
    try:
        ev = substantive(drive(env, 160))
        assert ev, "expected some tag/capture events"
        for e in ev:
            for f in IDENTITY_FIELDS:
                assert f in e, f"{e['event_type']} missing {f}"
            for f in ("env_index", "episode_id", "reset_sequence",
                      "simulation_step", "decision_step", "event_sequence"):
                assert isinstance(e[f], int), f"{f} must be an int, got {type(e[f])}"
    finally:
        env.close()


def test_reset_marker_carries_ended_and_new_episode_ids():
    env = make()
    core = env.core
    try:
        core.drain_tag_events()
        env.reset()
        markers = [e for e in core.drain_tag_events()
                   if e["event_type"] == "episode_reset"]
        assert markers, "reset must emit a marker"
        m = markers[0]
        assert m["ended_episode_id"] + 1 == m["episode_id"], (
            "marker must name both the ending and beginning episode")
        assert m["reset_sequence"] >= 1
    finally:
        env.close()


# --- uniqueness and ordering ------------------------------------------------

def test_no_duplicate_event_identities():
    env = make()
    try:
        ev = drive(env, 200)
        seqs = [e["event_sequence"] for e in ev]
        assert len(seqs) == len(set(seqs)), "event_sequence must be unique"
    finally:
        env.close()


def test_event_sequence_is_strictly_increasing():
    env = make()
    try:
        ev = drive(env, 200)
        seqs = [e["event_sequence"] for e in ev]
        assert seqs == sorted(seqs), "events must be emitted in sequence order"
        assert all(b > a for a, b in zip(seqs, seqs[1:])), "sequence must be strict"
    finally:
        env.close()


def test_multiple_events_on_one_step_get_distinct_sequences():
    env = make()
    core = env.core
    try:
        core.drain_tag_events()
        found = None
        for _ in range(240):
            env.step_async(env.action_space.sample() * 0)
            env.step_wait()
            ev = substantive(core.drain_tag_events())
            if len(ev) > 1:
                found = ev
                break
        if found is None:
            pytest.skip("no step produced multiple events in this rollout")
        seqs = [e["event_sequence"] for e in found]
        assert len(seqs) == len(set(seqs)), "same-step events must differ in sequence"
        assert len({e["simulation_step"] for e in found}) == 1, "fixture assumption"
    finally:
        env.close()


# --- episode attribution ----------------------------------------------------

def test_episode_id_increments_exactly_once_per_reset():
    env = make()
    core = env.core
    try:
        before = int(core.episode_id[0].item())
        env.reset()
        assert int(core.episode_id[0].item()) == before + 1
        env.reset()
        assert int(core.episode_id[0].item()) == before + 2
    finally:
        env.close()


def test_events_stay_attached_to_the_ending_episode():
    """A terminal-step event must NOT be relabelled into the next episode."""
    env = make()
    core = env.core
    try:
        core.drain_tag_events()
        for _ in range(240):
            env.step_async(env.action_space.sample() * 0)
            _o, _r, done, _i = env.step_wait()
            ev = core.drain_tag_events()
            if bool(np.asarray(done).any()) and ev:
                subs = substantive(ev)
                markers = [e for e in ev if e["event_type"] == "episode_reset"]
                if subs and markers:
                    ended = markers[0]["ended_episode_id"]
                    for e in subs:
                        assert e["episode_id"] == ended, (
                            "terminal event was relabelled into the new episode")
                    # ordering: substantive events precede the reset marker
                    assert max(e["event_sequence"] for e in subs) < \
                        markers[0]["event_sequence"]
                    return
        pytest.skip("no terminal step produced substantive events")
    finally:
        env.close()


def test_capture_events_survive_auto_reset():
    """The ledger keeps captures that post-step score state would have erased."""
    env = make()
    core = env.core
    try:
        core.drain_tag_events()
        caps = []
        for _ in range(240):
            env.step_async(env.action_space.sample() * 0)
            _o, _r, done, _i = env.step_wait()
            caps += [e for e in core.drain_tag_events()
                     if e["event_type"] == "capture_scored"]
            if bool(np.asarray(done).any()):
                break
        assert caps, "expected captures in this rollout"
        for c in caps:
            assert c["score_after"] == c["score_before"] + 1
            for f in IDENTITY_FIELDS:
                assert f in c
    finally:
        env.close()


# --- parallel environments --------------------------------------------------

def test_parallel_envs_do_not_share_identity():
    env = make(n_envs=3)
    core = env.core
    try:
        ev = drive(env, 140)
        assert ev, "expected events across parallel envs"
        # Identity must be unique on (env_index, episode_id, event_sequence).
        keys = [(e["env_index"], e["episode_id"], e["event_sequence"]) for e in ev]
        assert len(keys) == len(set(keys))
        # env_index must actually vary, i.e. not hardcoded to 0.
        assert len({e["env_index"] for e in ev}) > 1, "env_index never varied"
        # Each env's episode_id is tracked independently.
        for b in {e["env_index"] for e in ev}:
            ids = [e["episode_id"] for e in ev if e["env_index"] == b]
            assert ids == sorted(ids), f"env {b} episode_id went backwards"
    finally:
        env.close()


def test_parallel_envs_reset_independently():
    env = make(n_envs=3)
    core = env.core
    try:
        # Assert RELATIVE advancement: construction/reset already advanced the
        # counters, so a zero baseline is not a valid assumption.
        before_ids = core.episode_id.detach().cpu().tolist()
        before_rs = core.reset_sequence.detach().cpu().tolist()
        core.reset_indices(torch.tensor([True, False, False], device=core.device))
        after_ids = core.episode_id.detach().cpu().tolist()
        after_rs = core.reset_sequence.detach().cpu().tolist()
        assert [a - b for a, b in zip(after_ids, before_ids)] == [1, 0, 0], (
            f"masked reset advanced the wrong envs: {before_ids} -> {after_ids}")
        assert [a - b for a, b in zip(after_rs, before_rs)] == [1, 0, 0]
    finally:
        env.close()


# --- observational guarantee ------------------------------------------------

def test_identity_telemetry_is_behaviour_neutral():
    """Same seed, telemetry on vs off -> identical simulation tensors."""
    traces = []
    for tel in (False, True):
        env = make(telemetry=tel, seed=2_300_777)
        core = env.core
        try:
            t = []
            for _ in range(80):
                env.step_async(env.action_space.sample() * 0)
                _o, rew, done, _i = env.step_wait()
                t.append((core.blue_x[0].clone(), core.red_x[0].clone(),
                          core.blue_tagged[0].clone(), core.red_tagged[0].clone(),
                          int(core.blue_score[0]), int(core.red_score[0]),
                          float(np.asarray(rew).sum()), bool(np.asarray(done).any())))
                if bool(np.asarray(done).any()):
                    break
            traces.append(t)
        finally:
            env.close()
    off, on = traces
    assert len(off) == len(on), "telemetry changed episode length"
    for i, (a, b) in enumerate(zip(off, on)):
        for k in range(4):
            assert torch.equal(a[k], b[k]), f"tensor {k} diverged at step {i}"
        assert a[4:] == b[4:], f"scores/reward/done diverged at step {i}"
