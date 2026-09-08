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


# --- true row exclusion ------------------------------------------------------


class _RecordingPolicy:
    """Records exactly which rows it was asked to evaluate."""

    def __init__(self, tag: int):
        self.tag = tag
        self.calls = 0
        self.rows_seen = []

    def act(self, obs, global_state=None, z_idx=None, **kw):
        import torch

        ids = obs["id"]
        self.calls += 1
        self.rows_seen.append([int(x) for x in ids.tolist()])
        n = ids.shape[0]
        actions = torch.stack([ids, torch.full_like(ids, self.tag)], dim=1)
        values = ids.to(torch.float32) + self.tag * 1000.0
        log_probs = torch.full((n,), float(self.tag), dtype=torch.float32)
        return actions, values, log_probs, {}


def _make_handoff(n_envs, active_mask):
    """Install the handoff over recording policies with a forced latch."""
    import torch

    from experiments.o3_handoff import install_o3_handoff

    core = FakeCore(n_envs)
    o3 = _RecordingPolicy(tag=3)
    g0 = _RecordingPolicy(tag=0)
    class _Collector:
        def on_episode_done(self, info, **kw):
            return None

    trainer = type("T", (), {})()
    trainer.model = o3
    trainer.env = type("E", (), {"core": core, "num_envs": n_envs})()
    trainer.rollout_collector = _Collector()

    state, detector, uninstall = install_o3_handoff(trainer, g0, strict=False)
    state.o3_active |= np.asarray(active_mask, dtype=bool)
    obs = {"id": torch.arange(n_envs, dtype=torch.int64)}
    actions, values, log_probs, _ = trainer.model.act(obs, None, z_idx=None)
    uninstall()
    return o3, g0, actions, values, log_probs, state


def test_o3_never_sees_prefix_rows():
    """The RNG-isolation guarantee: O3's forward contains only active rows."""
    o3, g0, actions, _, _, _ = _make_handoff(6, [False, True, False, True, False, False])
    assert o3.calls == 1 and g0.calls == 1
    assert o3.rows_seen == [[1, 3]], f"O3 saw prefix rows: {o3.rows_seen}"
    assert g0.rows_seen == [[0, 2, 4, 5]]


def test_o3_not_called_at_all_when_no_env_is_active():
    o3, g0, actions, _, _, _ = _make_handoff(4, [False] * 4)
    assert o3.calls == 0, "O3 must not run when it controls nothing"
    assert g0.calls == 1 and g0.rows_seen == [[0, 1, 2, 3]]


def test_g0_not_called_at_all_when_every_env_is_active():
    o3, g0, actions, _, _, _ = _make_handoff(4, [True] * 4)
    assert g0.calls == 0, "G0 must not run when it controls nothing"
    assert o3.calls == 1 and o3.rows_seen == [[0, 1, 2, 3]]


def test_stitched_actions_come_from_the_right_policy_per_row():
    active = [False, True, False, True]
    _, _, actions, _, _, _ = _make_handoff(4, active)
    # column 1 carries the policy tag: 3 for O3, 0 for G0
    tags = actions[:, 1].tolist()
    assert tags == [0, 3, 0, 3]
    # column 0 carries the row id, proving no row got another row's action
    assert actions[:, 0].tolist() == [0, 1, 2, 3]


def test_prefix_rows_carry_no_o3_metadata():
    """G0 values/log-probs must never masquerade as O3 metadata."""
    active = [False, True, False, True]
    _, _, _, values, log_probs, _ = _make_handoff(4, active)
    # O3 rows carry O3's signature (id + 3000); prefix rows are left at zero.
    assert values[1].item() == pytest.approx(3001.0)
    assert values[3].item() == pytest.approx(3003.0)
    assert values[0].item() == 0.0 and values[2].item() == 0.0
    assert log_probs[1].item() == pytest.approx(3.0)
    assert log_probs[0].item() == 0.0


# --- credit flattening order -------------------------------------------------


def test_credit_flattening_selects_exactly_the_intended_cells():
    """[T,B] -> [T*B] must keep credit aligned with observations.

    A perfectly correct per-step mask becomes wrong if observations and credit
    are flattened in different orders. Self-identifying data makes an axis swap
    immediately visible.
    """
    T, B = 3, 4
    obs = np.array([[100 * t + e for e in range(B)] for t in range(T)])
    credit = np.zeros((T, B), dtype=bool)
    for t, e in [(0, 2), (1, 0), (2, 3)]:
        credit[t, e] = True

    flat_obs = obs.reshape(-1)
    flat_credit = credit.reshape(-1)
    selected = flat_obs[flat_credit].tolist()

    assert selected == [2, 100, 203], f"flattening misaligned: {selected}"
    assert flat_credit.sum() == 3


def test_transposed_flattening_is_detected_as_wrong():
    """Guard the guard: the wrong order must NOT produce the same answer."""
    T, B = 3, 4
    obs = np.array([[100 * t + e for e in range(B)] for t in range(T)])
    credit = np.zeros((T, B), dtype=bool)
    for t, e in [(0, 2), (1, 0), (2, 3)]:
        credit[t, e] = True

    right = obs.reshape(-1)[credit.reshape(-1)].tolist()
    wrong = obs.T.reshape(-1)[credit.T.reshape(-1)].tolist()
    assert sorted(right) == sorted(wrong)          # same cells, either way
    assert right != wrong, "test cannot distinguish orderings"
