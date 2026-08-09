"""Vectorization invariants for the O3 pre-action handoff.

These target the two bugs that would look completely normal in aggregate
training curves:

1. Ordering. If the detector runs after action selection, O3 starts at h+1 and
   the context under study shifts by one decision.
2. Per-env coupling. If any latch, carry-phase counter or reset is scalar rather
   than per environment, one env terminating silently changes another's control
   source.

Run against synthetic cores so the invariants are exercised directly rather than
inferred from a training run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

np = pytest.importorskip("numpy")

from rl.analysis.c_fork_detector import BatchedCForkDetector  # noqa: E402
from experiments.o3_handoff import HandoffState  # noqa: E402


class FakeCore:
    """Minimal core exposing only what the detector reads."""

    def __init__(self, n_envs: int, n_agents: int = 2, n_macros: int = 5, n_targets: int = 50):
        self.B = n_envs
        self.n_agents = n_agents
        self._carrying = np.zeros((n_envs, n_agents), dtype=bool)
        self._legal = np.ones((n_envs, n_agents), dtype=np.int64)
        self.cfg = type("cfg", (), {"n_macros": n_macros, "n_targets": n_targets})()
        self.blue_alive = np.ones((n_envs, n_agents), dtype=bool)
        self._n_macros = n_macros
        self._n_targets = n_targets

    @property
    def blue_carrying(self):
        return self._carrying

    def set_carrying(self, env_i: int, value: bool):
        self._carrying[env_i, 0] = bool(value)

    def set_legal_macros(self, env_i: int, agent_i: int, count: int):
        self._legal[env_i, agent_i] = int(count)

    def _build_action_mask(self, side: str = "blue"):
        mask = np.zeros((self.B, self.n_agents, self._n_macros + self._n_targets))
        for b in range(self.B):
            for a in range(self.n_agents):
                mask[b, a, : self._legal[b, a]] = 1.0
        return mask.reshape(self.B, -1)


def test_detector_fires_per_env_independently():
    core = FakeCore(4)
    det = BatchedCForkDetector(n_envs=4)

    # env1 carrying with 2 legal macros -> eligible; others not carrying.
    core.set_carrying(1, True)
    core.set_legal_macros(1, 0, 2)
    fired = det.step(core)
    assert fired.tolist() == [False, True, False, False]

    # Firing latches for that phase: it must not fire again next step.
    fired = det.step(core)
    assert not fired.any()


def test_carrying_with_single_legal_response_does_not_fire():
    core = FakeCore(2)
    core.set_carrying(0, True)          # 1 x 1 = 1 team response
    fired = BatchedCForkDetector(n_envs=2).step(core)
    assert not fired.any()


def test_new_carry_phase_re_arms_only_that_env():
    core = FakeCore(3)
    det = BatchedCForkDetector(n_envs=3)
    for e in (0, 2):
        core.set_carrying(e, True)
        core.set_legal_macros(e, 0, 2)
    assert det.step(core).tolist() == [True, False, True]

    core.set_carrying(0, False)         # env0 loses possession
    det.step(core)
    core.set_carrying(0, True)          # env0 picks up again -> re-arms
    fired = det.step(core)
    assert fired.tolist() == [True, False, False], "only env0 should re-fire"


def test_reset_envs_touches_only_terminated_environments():
    core = FakeCore(4)
    det = BatchedCForkDetector(n_envs=4)
    for e in range(4):
        core.set_carrying(e, True)
        core.set_legal_macros(e, 0, 2)
    det.step(core)
    assert det._fired_this_phase.all()

    det.reset_envs(np.array([True, False, False, True]))
    assert det._fired_this_phase.tolist() == [False, True, True, False]
    assert det._in_carry.tolist() == [False, True, True, False]


# --- HandoffState latch semantics -------------------------------------------


def test_latch_is_one_way_within_an_episode():
    st = HandoffState(n_envs=3)
    st.o3_active |= np.array([False, True, False])
    st.o3_active |= np.array([False, False, False])   # a later non-firing step
    assert st.o3_active.tolist() == [False, True, False], "latch must not clear mid-episode"


def test_episode_end_resets_only_terminated_envs():
    st = HandoffState(n_envs=3)
    st.o3_active |= np.array([True, True, False])
    st.trigger_step[:] = np.array([5, 7, -1])
    st.first_o3_action_step[:] = np.array([5, 7, -1])
    st._len_accum[:] = np.array([10, 20, 0])

    st.on_episode_end(np.array([True, False, False]))

    assert st.o3_active.tolist() == [False, True, False], "env1 must keep its latch"
    assert st.trigger_step.tolist() == [-1, 7, -1]
    assert st.episodes_seen == 1
    assert st.episodes_with_handoff == 1
    assert st.post_handoff_lengths == [10]


def test_mixed_batch_credit_vector_matches_latch_exactly():
    """env0 prefix, env1 already active, env2 fires now, env3 prefix."""
    st = HandoffState(n_envs=4)
    st.o3_active |= np.array([False, True, False, False])   # env1 already active
    fired = np.array([False, False, True, False])           # env2 fires this step
    newly = fired & ~st.o3_active
    st.trigger_step[newly] = np.array([3])
    st.o3_active |= fired
    credit = st.o3_active.copy()

    assert credit.tolist() == [False, True, True, False]
    assert st.trigger_step.tolist() == [-1, -1, 3, -1]


def test_throughput_fields_are_all_present():
    st = HandoffState(n_envs=2)
    st.environment_steps = 1000
    st.credited_o3_steps = 380
    st.episodes_seen = 10
    st.episodes_with_handoff = 7
    st.post_handoff_lengths = [40, 60]
    t = st.throughput()
    for k in ("episodes_seen", "episodes_with_handoff", "handoff_rate",
              "environment_steps", "credited_o3_steps", "credited_fraction",
              "mean_post_handoff_length", "effective_o3_steps_per_1k_env_steps"):
        assert k in t, f"missing throughput field {k}"
    assert t["credited_fraction"] == pytest.approx(0.38)
    assert t["effective_o3_steps_per_1k_env_steps"] == pytest.approx(380.0)
