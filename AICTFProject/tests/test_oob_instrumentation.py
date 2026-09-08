"""Out-of-bounds events must be measurable, and measuring them must change nothing.

The OOB reward rate could never be recovered from the aggregate sparse residual,
so its -100 value has never been validated against an observed frequency. This
instrumentation exists so that term can be budgeted from data instead of guessed
at -- which is the mistake the whole reward-budget exercise is correcting.

Behaviour neutrality is the hard requirement: telemetry that perturbs the run it
measures is worse than no telemetry.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402

V2 = dict(
    taggers_required=1,
    tag_min_interval_seconds=10.0,
    tag_nearest_only=True,
    tag_channel_seconds=0.0,
    suppression_attackers_required=2,
)


def _env(telemetry: bool, seed: int = 4242, n_envs: int = 4):
    cfg = GPUFieldConfig(
        n_envs=n_envs, max_blue_agents=2, max_red_agents=2,
        map_set="train", map_layout="map_a", max_decision_steps=120,
        aquaticus_profile=True, rules_profile="OURS", device="cpu",
        seed=seed, obstacle_obs_channel=True,
        tag_telemetry_enabled=telemetry, **V2,
    )
    return GPUCTFVecEnv(cfg)


def _drive(env, steps: int = 120):
    n = env.core.B * env.core.Nb * 2
    rewards = []
    for i in range(steps):
        _, r, _, _ = env.step(np.full(n, i % 5, dtype=np.int64))
        rewards.append(float(np.asarray(r).sum()))
    return rewards


def test_oob_instrumentation_is_behaviour_neutral():
    off = _env(False)
    torch.manual_seed(0)
    off.reset()
    r_off = _drive(off)
    off.close()

    on = _env(True)
    torch.manual_seed(0)
    on.reset()
    r_on = _drive(on)
    on.close()

    assert r_off == r_on, "enabling OOB telemetry changed the trajectory"


def test_oob_events_are_emitted_when_an_agent_leaves_the_field():
    """Positive control: without this, a silent no-op would look like 'no OOB'."""
    env = _env(True)
    env.reset()
    core = env.core
    core.drain_tag_events()

    # Park every blue agent on the left boundary and aim them further left, so
    # the next integration step must land outside the field.
    core.blue_x[:, :] = 0.0
    core.blue_heading[:, :] = float(np.pi)
    core.blue_speed[:, :] = float(core.blue_speed.max().item() or 1.0)

    n = core.B * core.Nb * 2
    events = []
    for _ in range(5):
        env.step(np.zeros(n, dtype=np.int64))
        events.extend(e for e in core.drain_tag_events()
                      if e["event_type"] == "out_of_bounds")
        if events:
            break
    env.close()

    assert events, "agents driven off the field produced no out_of_bounds events"
    for e in events:
        assert e["team"] in ("blue", "red")
        assert "agent_index" in e
        assert e["sparse_points"] != 0.0
        # Integer event identity, same contract as tag events.
        for key in ("env_index", "episode_id", "reset_sequence",
                    "simulation_step", "decision_step", "event_sequence"):
            assert key in e, f"missing identity field {key}"


def test_blue_oob_is_signed_as_a_penalty_in_blues_ledger():
    env = _env(True)
    env.reset()
    core = env.core
    core.drain_tag_events()
    core.blue_x[:, :] = 0.0
    core.blue_heading[:, :] = float(np.pi)
    core.blue_speed[:, :] = float(core.blue_speed.max().item() or 1.0)

    n = core.B * core.Nb * 2
    blue_events = []
    for _ in range(5):
        env.step(np.zeros(n, dtype=np.int64))
        blue_events.extend(
            e for e in core.drain_tag_events()
            if e["event_type"] == "out_of_bounds" and e["team"] == "blue"
        )
        if blue_events:
            break
    env.close()

    assert blue_events
    assert all(e["sparse_points"] < 0 for e in blue_events), (
        "a BLUE agent leaving the field must cost BLUE points"
    )


def test_no_oob_events_when_telemetry_disabled():
    env = _env(False)
    env.reset()
    core = env.core
    core.blue_x[:, :] = 0.0
    core.blue_heading[:, :] = float(np.pi)
    n = core.B * core.Nb * 2
    for _ in range(5):
        env.step(np.zeros(n, dtype=np.int64))
    assert core.drain_tag_events() == []
    env.close()


def test_oob_points_value_is_unchanged_in_v3():
    """V3 measures OOB; it must not budget it before the rate is known."""
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2, map_set="train",
        map_layout="map_a", max_decision_steps=32, aquaticus_profile=True,
        rules_profile="OURS", device="cpu", seed=1, obstacle_obs_channel=True, **V2,
    )
    assert cfg.sparse_oob_points == -100.0
